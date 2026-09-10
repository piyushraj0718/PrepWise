import hashlib
import json
from dataclasses import replace
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from app.ai.assessment import AssessmentLLMProvider
from app.ai.assessment_evaluation import (
    AssessmentQuestionEvaluator,
    SemanticEvaluationError,
    TrustedEvaluationEvidence,
)
from app.ai.assessment_prompts import (
    ASSESSMENT_PROMPT_VERSION,
    ADAPTIVE_WEAK_AREAS_MAX,
    ASSESSMENT_EVALUATION_PROMPT_VERSION,
    build_assessment_prompt,
)
from app.ai.llm import LLMProviderError, LLMResponseError
from app.domain.assessment import BloomLevel, QuestionType
from app.domain.assessment_quality import (
    AssessmentQualityResult,
    AssessmentQualityValidator,
    QualityFinding,
    QualityDecision,
    QuestionQualityResult,
    RegenerationDecisionPolicy,
)
from app.domain.assessment_semantic import SemanticQualityPolicy
from app.models.document import Document
from app.repositories.assessments import AssessmentRepository
from app.schemas.assessment_generation import GeneratedAssessment, GeneratedMCQ
from app.schemas.assessments import QuestionGenerationRequest
from app.services.documents import (
    DocumentNotFoundError,
    DocumentProcessingStateError,
)
from app.services.reranking import (
    InvalidRerankingQuery,
    RerankedResult,
    RerankingService,
)
from app.services.retrieval import InvalidRetrievalQuery


class AssessmentGenerationError(RuntimeError):
    pass


class AssessmentValidationError(AssessmentGenerationError):
    def __init__(
        self, message: str, quality_result: AssessmentQualityResult | None = None
    ) -> None:
        super().__init__(message)
        self.quality_result = quality_result


class AssessmentGenerationExhaustedError(AssessmentValidationError):
    pass


class AssessmentSemanticEvaluationError(AssessmentGenerationError):
    pass


class AssessmentNoContextError(AssessmentGenerationError):
    pass


class AssessmentPersistenceError(AssessmentGenerationError):
    pass


class AssessmentGenerationService:
    def __init__(
        self,
        repository: AssessmentRepository,
        reranking: RerankingService,
        provider: AssessmentLLMProvider,
        embedding_model_name: str,
        reranker_model_name: str,
        prompt_version: str = ASSESSMENT_PROMPT_VERSION,
        quality_validator: AssessmentQualityValidator | None = None,
        regeneration_policy: RegenerationDecisionPolicy | None = None,
        evaluator: AssessmentQuestionEvaluator | None = None,
        semantic_quality_policy: SemanticQualityPolicy | None = None,
        semantic_evaluation_enabled: bool = False,
    ) -> None:
        self.repository = repository
        self.reranking = reranking
        self.provider = provider
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.prompt_version = prompt_version
        self.quality_validator = quality_validator or AssessmentQualityValidator()
        self.regeneration_policy = regeneration_policy or RegenerationDecisionPolicy()
        self.evaluator = evaluator
        self.semantic_quality_policy = semantic_quality_policy or SemanticQualityPolicy()
        self.semantic_evaluation_enabled = semantic_evaluation_enabled
        if self.semantic_evaluation_enabled and self.evaluator is None:
            raise ValueError("An evaluator is required when semantic evaluation is enabled")

    def generate(
        self,
        request: QuestionGenerationRequest,
        weak_areas_hint: list[str] | None = None,
    ) -> tuple[Any, list[Any]]:
        document_id = str(request.document_id) if request.document_id else None
        self._validate_document(document_id)
        try:
            candidates = self.reranking.search(
                request.query,
                top_k=request.top_k,
                candidate_k=request.candidate_k,
                document_id=document_id,
            )
        except (DocumentNotFoundError, InvalidRerankingQuery, InvalidRetrievalQuery):
            raise
        except Exception as error:
            raise AssessmentGenerationError(
                "Retrieval and reranking failed") from error
        if not candidates:
            raise AssessmentNoContextError(
                "No relevant processed document context was found")

        run_id = self._start_run(request, document_id)
        accepted_questions: dict[int, GeneratedMCQ] = {}
        accepted_attempt_ids: dict[int, str] = {}
        failed_stems: list[str] = []
        pending_slots = list(range(request.count))
        attempt_numbers = {slot: 1 for slot in pending_slots}
        last_quality_result: AssessmentQualityResult | None = None

        try:
            while pending_slots:
                attempt_request = request.model_copy(
                    update={"count": len(pending_slots)})
                generated = self._generate_questions(
                    attempt_request, candidates, weak_areas_hint=weak_areas_hint)
                accepted_in_batch = [
                    accepted_questions[slot]
                    for slot in sorted(accepted_questions)
                ]
                combined_questions = accepted_in_batch + generated.questions
                quality_result = self.quality_validator.validate(
                    combined_questions,
                    {candidate.chunk_id for candidate in candidates},
                )
                last_quality_result = quality_result
                candidate_offset = len(accepted_in_batch)
                next_pending: list[int] = []
                exhausted_slots: list[int] = []
                for index, slot in enumerate(pending_slots):
                    question = generated.questions[index]
                    result = quality_result.question_results[candidate_offset + index]
                    semantic_result = None
                    result = self._reject_previous_attempt_duplicate(
                        result, question, failed_stems, index)
                    if result.accepted and self.semantic_evaluation_enabled:
                        try:
                            semantic_result = self.evaluator.evaluate(  # type: ignore[union-attr]
                                question,
                                requested_difficulty=(
                                    request.difficulty.value if request.difficulty else None
                                ),
                                requested_bloom_level=(
                                    request.bloom_level.value if request.bloom_level else None
                                ),
                                requested_topic=request.query,
                                requested_skill=request.skill,
                                evidence=self._evaluation_evidence(question, candidates),
                            )
                        except Exception as error:
                            attempt_id = self.repository.record_question_attempt(
                                generation_run_id=run_id,
                                question_slot=slot,
                                attempt_number=attempt_numbers[slot],
                                status="evaluation_failed",
                                candidate_payload=question.model_dump(mode="json"),
                                failure_reasons=["semantic_evaluation_unavailable"],
                            )
                            self._record_quality_evaluations(
                                run_id, attempt_id, result, None, "evaluation_failed"
                            )
                            raise AssessmentSemanticEvaluationError(
                                "Semantic evaluation could not be completed"
                            ) from error
                        if not self.semantic_quality_policy.passes(semantic_result):
                            result = self._reject_semantic_quality(
                                result, index)
                    attempt_number = attempt_numbers[slot]
                    decision = self.regeneration_policy.decide(
                        result, attempt_number=attempt_number)
                    if decision is QualityDecision.ACCEPT:
                        attempt_id = self.repository.record_question_attempt(
                            generation_run_id=run_id,
                            question_slot=slot,
                            attempt_number=attempt_number,
                            status="accepted",
                            candidate_payload=question.model_dump(mode="json"),
                            failure_reasons=[],
                        )
                        accepted_questions[slot] = question
                        accepted_attempt_ids[slot] = attempt_id
                        self._record_quality_evaluations(
                            run_id, attempt_id, result, semantic_result, "accepted"
                        )
                    else:
                        failure_codes = self._failure_codes(result)
                        failed_stems.append(question.stem)
                        status = "exhausted" if decision is QualityDecision.REJECT else "rejected"
                        attempt_id = self.repository.record_question_attempt(
                            generation_run_id=run_id,
                            question_slot=slot,
                            attempt_number=attempt_number,
                            status=status,
                            candidate_payload=question.model_dump(mode="json"),
                            failure_reasons=failure_codes,
                        )
                        self._record_quality_evaluations(
                            run_id, attempt_id, result, semantic_result, status
                        )
                        if decision is QualityDecision.REGENERATE:
                            attempt_numbers[slot] = attempt_number + 1
                            next_pending.append(slot)
                        else:
                            exhausted_slots.append(slot)

                pending_slots = next_pending
                if exhausted_slots:
                    self._mark_run_exhausted(
                        run_id,
                        "One or more question slots exhausted regeneration attempts",
                    )
                    failure_codes = sorted({
                        finding.code for finding in quality_result.hard_failures
                    })
                    raise AssessmentGenerationExhaustedError(
                        "Assessment generation exhausted after bounded attempts: "
                        + ", ".join(failure_codes),
                        quality_result=quality_result,
                    )
                if not pending_slots:
                    break

            run_values = self._run_values(
                request, document_id, status="completed")
            question_values = [
                self._question_values(
                    accepted_questions[slot], request, candidates)
                for slot in sorted(accepted_questions)
            ]
            try:
                return self.repository.create_generation_batch(
                    run_values,
                    question_values,
                    existing_run_id=run_id,
                    accepted_attempt_ids=[
                        accepted_attempt_ids[slot]
                        for slot in sorted(accepted_attempt_ids)
                    ],
                )
            except Exception as error:
                raise AssessmentPersistenceError(
                    "The generated assessment could not be persisted") from error
        except (LLMProviderError, LLMResponseError):
            self._mark_run_failed(run_id, "LLM generation failed")
            raise
        except AssessmentGenerationExhaustedError:
            raise
        except AssessmentSemanticEvaluationError:
            self._mark_run_failed(run_id, "Semantic evaluation failed")
            raise
        except AssessmentValidationError:
            self._mark_run_failed(
                run_id, "Structured generation validation failed")
            raise
        except AssessmentPersistenceError:
            self._mark_run_failed(run_id, "Assessment persistence failed")
            raise
        except Exception as error:
            self._mark_run_failed(run_id, "Assessment generation exhausted")
            if last_quality_result is not None:
                raise AssessmentValidationError(
                    "Assessment generation exhausted after bounded attempts",
                    quality_result=last_quality_result,
                ) from error
            raise AssessmentPersistenceError(
                "The generated assessment could not be persisted") from error

    def _generate_questions(
        self,
        request: QuestionGenerationRequest,
        candidates: Sequence[RerankedResult],
        weak_areas_hint: list[str] | None = None,
    ) -> GeneratedAssessment:
        prompt = build_assessment_prompt(
            request.query,
            self._build_context(candidates),
            request.count,
            request.difficulty.value if request.difficulty else None,
            request.bloom_level.value if request.bloom_level else None,
            request.skill,
            weak_areas_hint=weak_areas_hint,
        )
        try:
            raw_output = self.provider.generate_structured(
                prompt, self._response_schema())
        except LLMProviderError:
            raise
        except Exception as error:
            raise LLMProviderError(
                "The assessment LLM provider failed") from error
        generated = self._parse_output(raw_output)
        self._validate_batch(generated, request, candidates)
        return generated

    def _start_run(self, request: QuestionGenerationRequest, document_id: str | None) -> str:
        try:
            run_values = self._run_values(
                request, document_id, status="generating")
            return self.repository.start_generation_run(**run_values)
        except IntegrityError:
            run_values["request_fingerprint"] = (
                f"{run_values['request_fingerprint'][:95]}-{uuid4().hex}"
            )
            try:
                return self.repository.start_generation_run(**run_values)
            except Exception as error:
                raise AssessmentPersistenceError(
                    "The generation run could not be started") from error
        except Exception as error:
            raise AssessmentPersistenceError(
                "The generation run could not be started") from error

    def _mark_run_failed(self, run_id: str, reason: str) -> None:
        try:
            self.repository.update_generation_run(
                run_id, status="failed", failure_reason=reason)
        except Exception:
            pass

    def _mark_run_exhausted(self, run_id: str, reason: str) -> None:
        try:
            self.repository.update_generation_run(
                run_id, status="exhausted", failure_reason=reason)
        except Exception:
            pass

    def _reject_previous_attempt_duplicate(
        self,
        result: QuestionQualityResult,
        question: GeneratedMCQ,
        failed_stems: Sequence[str],
        question_index: int,
    ) -> QuestionQualityResult:
        if not any(
            self.quality_validator.stems_conflict(question.stem, stem)
            for stem in failed_stems
        ):
            return result
        finding = QualityFinding(
            "duplicate_previous_attempt",
            "The question repeats a previously failed attempt",
            question_index,
        )
        return replace(
            result,
            accepted=False,
            hard_failures=[*result.hard_failures, finding],
            deterministic_score=max(0.0, result.deterministic_score - 0.2),
            decision=QualityDecision.REGENERATE,
        )

    @staticmethod
    def _reject_semantic_quality(
        result: QuestionQualityResult, question_index: int
    ) -> QuestionQualityResult:
        finding = QualityFinding(
            "semantic_quality_below_threshold",
            "The semantic evaluator did not recommend this question for acceptance",
            question_index,
        )
        return replace(
            result,
            accepted=False,
            hard_failures=[*result.hard_failures, finding],
            decision=QualityDecision.REGENERATE,
        )

    @staticmethod
    def _evaluation_evidence(
        question: GeneratedMCQ,
        candidates: Sequence[RerankedResult],
    ) -> list[TrustedEvaluationEvidence]:
        candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}
        evidence: list[TrustedEvaluationEvidence] = []
        remaining_characters = 8_000
        for chunk_id in question.cited_chunk_ids[:4]:
            candidate = candidates_by_id[chunk_id]
            if remaining_characters <= 0:
                break
            text = candidate.chunk_text[:remaining_characters]
            evidence.append(TrustedEvaluationEvidence(chunk_id=chunk_id, chunk_text=text))
            remaining_characters -= len(text)
        return evidence

    @staticmethod
    def _failure_codes(result: QuestionQualityResult) -> list[str]:
        codes = [finding.code for finding in result.hard_failures]
        if not codes and not result.accepted:
            codes.append("deterministic_score_below_threshold")
        return sorted(set(codes))

    def _record_quality_evaluations(
        self,
        run_id: str,
        attempt_id: str,
        deterministic_result: QuestionQualityResult,
        semantic_result: Any | None,
        attempt_status: str,
    ) -> None:
        self.repository.record_quality_evaluation(
            generation_run_id=run_id,
            question_attempt_id=attempt_id,
            evaluation_type="deterministic",
            evaluator_provider=None,
            evaluator_model=None,
            prompt_version=None,
            overall_score=deterministic_result.deterministic_score,
            dimension_scores={"hard_failure_codes": self._failure_codes(deterministic_result)},
            recommendation="pass" if deterministic_result.accepted else "fail",
            rationale="Deterministic structural and provenance validation",
            status=attempt_status,
        )
        if semantic_result is None:
            return
        self.repository.record_quality_evaluation(
            generation_run_id=run_id,
            question_attempt_id=attempt_id,
            evaluation_type="semantic",
            evaluator_provider="gemini" if self.evaluator else None,
            evaluator_model=getattr(self.evaluator, "model_name", None),
            prompt_version=ASSESSMENT_EVALUATION_PROMPT_VERSION,
            overall_score=semantic_result.overall_score,
            dimension_scores={
                name: getattr(semantic_result, name).score
                for name in ("groundedness", "correctness", "distractor_quality", "explanation_quality", "difficulty_alignment", "bloom_alignment")
            },
            recommendation=semantic_result.recommendation,
            rationale=semantic_result.rationale,
            status=attempt_status,
        )

    def _validate_document(self, document_id: str | None) -> None:
        if document_id is None:
            return
        document = self.repository.db.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError("Document was not found")
        if document.status != "processed":
            raise DocumentProcessingStateError(
                "Only processed documents can generate questions")

    @staticmethod
    def _build_context(candidates: Sequence[RerankedResult]) -> str:
        return "\n\n".join(
            f"[chunk_id={candidate.chunk_id}]\n{candidate.chunk_text}"
            for candidate in candidates
        )

    @staticmethod
    def _response_schema() -> dict[str, object]:
        return {
            "type": "OBJECT",
            "properties": {
                "questions": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "stem": {"type": "STRING"},
                            "options": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "option_key": {"type": "STRING"},
                                        "option_text": {"type": "STRING"},
                                    },
                                    "required": ["option_key", "option_text"],
                                },
                            },
                            "correct_option_key": {"type": "STRING"},
                            "explanation": {"type": "STRING"},
                            "difficulty": {"type": "STRING"},
                            "bloom_level": {
                                "type": "STRING",
                                "enum": [level.value for level in BloomLevel],
                            },
                            "topic": {"type": "STRING"},
                            "skill": {"type": "STRING"},
                            "cited_chunk_ids": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                            },
                        },
                        "required": [
                            "stem", "options", "correct_option_key", "explanation",
                            "difficulty", "bloom_level", "topic", "skill",
                            "cited_chunk_ids",
                        ],
                    },
                },
            },
            "required": ["questions"],
        }

    @staticmethod
    def _parse_output(raw_output: object) -> GeneratedAssessment:
        if not isinstance(raw_output, dict):
            raise LLMResponseError(
                "The LLM returned malformed structured questions")
        try:
            return GeneratedAssessment.model_validate(raw_output)
        except Exception as error:
            raise AssessmentValidationError(
                "The LLM returned malformed structured questions") from error

    def _validate_batch(
        self,
        assessment: GeneratedAssessment,
        request: QuestionGenerationRequest,
        candidates: Sequence[RerankedResult],
    ) -> None:
        if len(assessment.questions) != request.count:
            raise AssessmentValidationError(
                "The LLM returned an unexpected question count")
        candidate_ids = {candidate.chunk_id for candidate in candidates}
        for question in assessment.questions:
            if request.difficulty and question.difficulty != request.difficulty:
                raise AssessmentValidationError(
                    "Generated difficulty does not match the request")
            if request.bloom_level and question.bloom_level != request.bloom_level:
                raise AssessmentValidationError(
                    "Generated Bloom level does not match the request")
            if len(set(question.cited_chunk_ids)) != len(question.cited_chunk_ids):
                raise AssessmentValidationError(
                    "Generated citation IDs must be unique")
            if not set(question.cited_chunk_ids).issubset(candidate_ids):
                raise AssessmentValidationError(
                    "The LLM cited a chunk that was not retrieved")

    def _run_values(
        self,
        request: QuestionGenerationRequest,
        document_id: str | None,
        *,
        status: str = "completed",
    ) -> dict[str, object]:
        request_data = request.model_dump(mode="json")
        fingerprint = hashlib.sha256(
            json.dumps(request_data, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "request_fingerprint": fingerprint,
            "document_id": document_id,
            "query": request.query,
            "skill": request.skill,
            "question_type": QuestionType.MCQ,
            "difficulty": request.difficulty,
            "bloom_level": request.bloom_level,
            "requested_count": request.count,
            "retrieval_top_k": request.top_k,
            "retrieval_candidate_k": request.candidate_k,
            "embedding_model_name": self.embedding_model_name,
            "reranker_model_name": self.reranker_model_name,
            "llm_model_name": self.provider.model_name,
            "prompt_version": self.prompt_version,
            "seed": request.seed,
            "status": status,
        }

    @staticmethod
    def _question_values(
        question: GeneratedMCQ,
        request: QuestionGenerationRequest,
        candidates: Sequence[RerankedResult],
    ) -> dict[str, object]:
        candidates_by_id = {
            candidate.chunk_id: candidate for candidate in candidates}
        return {
            "document_id": str(request.document_id) if request.document_id else None,
            "question_type": QuestionType.MCQ,
            "stem": question.stem,
            "explanation": question.explanation,
            "difficulty": question.difficulty,
            "bloom_level": question.bloom_level,
            "topic": question.topic,
            "skill": question.skill,
            "correct_option_key": question.correct_option_key,
            "content_version": 1,
            "options": [
                {
                    "option_key": option.option_key,
                    "option_text": option.option_text,
                    "display_order": index,
                }
                for index, option in enumerate(question.options)
            ],
            "citations": [
                {
                    "chunk_id": chunk_id,
                    "citation_order": index,
                    "retrieval_rank": candidates_by_id[chunk_id].final_rank,
                    "reranker_score": candidates_by_id[chunk_id].reranker_score,
                    "hybrid_score": candidates_by_id[chunk_id].hybrid_score,
                    "chunk_text_snapshot": candidates_by_id[chunk_id].chunk_text,
                    "page_number_snapshot": candidates_by_id[chunk_id].page_number,
                }
                for index, chunk_id in enumerate(question.cited_chunk_ids)
            ],
        }
