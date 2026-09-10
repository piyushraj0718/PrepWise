import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from app.ai.assessment import AssessmentLLMProvider
from app.ai.assessment_prompts import (
    ASSESSMENT_PROMPT_VERSION,
    build_assessment_prompt,
)
from app.ai.llm import LLMProviderError, LLMResponseError
from app.domain.assessment import BloomLevel, QuestionType
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
    pass


class AssessmentNoContextError(AssessmentGenerationError):
    pass


class AssessmentPersistenceError(AssessmentGenerationError):
    pass


class AssessmentGenerationService:
    _MARKUP_PATTERN = re.compile(
        r"(?:\[\s*\d+\s*\]|\[\s*chunk[_ -]?id\s*=|\bchunk[_ -]?id\s*[:=]|\bcitation\s*[:=])",
        re.IGNORECASE,
    )
    _FORBIDDEN_OPTION_PATTERN = re.compile(
        r"^(?:all|none)\s+of\s+the\s+above$", re.IGNORECASE)

    def __init__(
        self,
        repository: AssessmentRepository,
        reranking: RerankingService,
        provider: AssessmentLLMProvider,
        embedding_model_name: str,
        reranker_model_name: str,
        prompt_version: str = ASSESSMENT_PROMPT_VERSION,
    ) -> None:
        self.repository = repository
        self.reranking = reranking
        self.provider = provider
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.prompt_version = prompt_version

    def generate(self, request: QuestionGenerationRequest) -> tuple[Any, list[Any]]:
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

        prompt = build_assessment_prompt(
            request.query,
            self._build_context(candidates),
            request.count,
            request.difficulty.value if request.difficulty else None,
            request.bloom_level.value if request.bloom_level else None,
            request.skill,
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
        run_values = self._run_values(request, document_id)
        question_values = [
            self._question_values(question, request, candidates)
            for question in generated.questions
        ]
        try:
            return self.repository.create_generation_batch(
                run_values, question_values)
        except Exception as error:
            raise AssessmentPersistenceError(
                "The generated assessment could not be persisted") from error

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
        stems: set[str] = set()
        for question in assessment.questions:
            normalized_stem = " ".join(question.stem.casefold().split())
            if normalized_stem in stems:
                raise AssessmentValidationError(
                    "Generated question stems must be unique")
            stems.add(normalized_stem)
            self._validate_text(question)
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

    def _validate_text(self, question: GeneratedMCQ) -> None:
        values = [question.stem, question.explanation,
                  question.topic, question.skill]
        values.extend(option.option_text for option in question.options)
        if any(self._MARKUP_PATTERN.search(value) for value in values):
            raise AssessmentValidationError(
                "Generated question content must not contain citation markup")
        if any(self._FORBIDDEN_OPTION_PATTERN.match(option.option_text.strip())
               for option in question.options):
            raise AssessmentValidationError(
                "Generated options must not use all/none of the above")

    def _run_values(
        self, request: QuestionGenerationRequest, document_id: str | None
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
            "status": "completed",
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
