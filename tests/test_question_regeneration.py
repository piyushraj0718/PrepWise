from uuid import uuid4

import pytest

from app.domain.assessment_quality import (
    AssessmentQualityConfig,
    AssessmentQualityValidator,
    RegenerationDecisionPolicy,
)
from app.ai.assessment_evaluation import SemanticDimensionResult, SemanticEvaluationResult
from app.domain.assessment_semantic import SemanticQualityPolicy
from app.models.assessment import (
    AssessmentGenerationRun,
    AssessmentQuestion,
    AssessmentQuestionAttempt,
    AssessmentQualityEvaluation,
)
from app.models.document import Document, DocumentChunk
from app.repositories.assessments import AssessmentRepository
from app.schemas.assessments import QuestionGenerationRequest
from app.services.assessment_generation import (
    AssessmentGenerationExhaustedError,
    AssessmentSemanticEvaluationError,
    AssessmentGenerationService,
)
from app.services.reranking import RerankedResult


def make_source(session_factory) -> tuple[str, str]:
    document_id = str(uuid4())
    with session_factory() as session:
        session.add(Document(
            id=document_id,
            original_filename="guide.pdf",
            storage_path="guide.pdf",
            content_type="application/pdf",
            size_bytes=1,
            status="processed",
        ))
        session.commit()
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=0,
            chunk_text="Database indexes improve lookup performance.",
            page_number=1,
        )
        session.add(chunk)
        session.commit()
        return document_id, chunk.id


def result(document_id: str, chunk_id: str) -> RerankedResult:
    return RerankedResult(
        chunk_id=chunk_id,
        document_id=document_id,
        similarity=0.9,
        bm25_score=0.8,
        hybrid_score=0.7,
        reranker_score=0.6,
        final_rank=1,
        chunk_text="Database indexes improve lookup performance.",
        chunk_index=0,
        page_number=1,
    )


def question(chunk_id: str, stem: str, *, forbidden_option: bool = False) -> dict:
    options = [
        {"option_key": "A", "option_text": "An index"},
        {"option_key": "B", "option_text": "A slower disk"},
        {"option_key": "C", "option_text": "A blank table"},
        {"option_key": "D", "option_text": "A deleted query"},
    ]
    if forbidden_option:
        options[1]["option_text"] = "All of the above"
    return {
        "stem": stem,
        "options": options,
        "correct_option_key": "A",
        "explanation": "The source says an index improves lookup performance.",
        "difficulty": "easy",
        "bloom_level": "understand",
        "topic": "database indexing",
        "skill": "identify indexing benefits",
        "cited_chunk_ids": [chunk_id],
    }


class FakeReranking:
    def __init__(self, document_id: str, chunk_id: str) -> None:
        self.result = result(document_id, chunk_id)

    def search(self, query: str, top_k: int, candidate_k: int | None,
               document_id: str | None) -> list[RerankedResult]:
        return [self.result]


class SequencedLLM:
    model_name = "fake-regeneration-llm"

    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = list(outputs)
        self.calls: list[int] = []

    def generate_structured(self, prompt: str, response_schema: dict[str, object]) -> object:
        self.calls.append(1)
        if not self.outputs:
            raise AssertionError(
                "The regeneration loop made an unexpected LLM call")
        return self.outputs.pop(0)


def make_service(
    session_factory,
    document_id: str,
    chunk_id: str,
    outputs: list[dict],
    max_attempts: int = 3,
    evaluator=None,
):
    return AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking(document_id, chunk_id),
        SequencedLLM(outputs),
        "fake-embedding",
        "fake-reranker",
        regeneration_policy=RegenerationDecisionPolicy(
            max_attempts=max_attempts),
        evaluator=evaluator,
        semantic_evaluation_enabled=evaluator is not None,
        semantic_quality_policy=SemanticQualityPolicy(0.75),
    )


def request(document_id: str, count: int = 1) -> QuestionGenerationRequest:
    return QuestionGenerationRequest(
        document_id=document_id,
        query="database indexing",
        count=count,
        difficulty="easy",
        bloom_level="understand",
    )


def test_first_attempt_succeeds_and_is_audited(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [{"questions": [
            question(chunk_id, "What improves lookup performance?")]}],
    )

    run, questions = service.generate(request(document_id))

    assert len(questions) == 1
    with session_factory() as session:
        attempts = session.query(AssessmentQuestionAttempt).all()
        stored_run = session.get(AssessmentGenerationRun, run.id)
    assert len(attempts) == 1
    assert attempts[0].attempt_number == 1
    assert attempts[0].status == "accepted"
    assert attempts[0].accepted_question_id == questions[0].id
    assert stored_run.status == "completed"


def test_only_failed_question_slot_is_regenerated(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [
            {"questions": [
                question(chunk_id, "What improves lookup performance?"),
                question(
                    chunk_id, "Which database feature reduces lookup cost?", forbidden_option=True),
            ]},
            {"questions": [
                question(chunk_id, "Which database feature organizes lookup access?")]},
        ],
    )

    run, questions = service.generate(request(document_id, count=2))

    assert [item.stem for item in questions] == [
        "What improves lookup performance?",
        "Which database feature organizes lookup access?",
    ]
    assert len(service.provider.calls) == 2
    with session_factory() as session:
        attempts = session.query(AssessmentQuestionAttempt).order_by(
            AssessmentQuestionAttempt.question_slot,
            AssessmentQuestionAttempt.attempt_number,
        ).all()
    assert [(item.question_slot, item.attempt_number, item.status) for item in attempts] == [
        (0, 1, "accepted"),
        (1, 1, "rejected"),
        (1, 2, "accepted"),
    ]
    assert run.status == "completed"


def test_regenerated_duplicate_is_rejected_and_bounded(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    stem = "Which database feature reduces lookup cost?"
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [
            {"questions": [question(chunk_id, stem, forbidden_option=True)]},
            {"questions": [question(chunk_id, stem)]},
        ],
        max_attempts=2,
    )
    with pytest.raises(AssessmentGenerationExhaustedError):
        service.generate(request(document_id))

    assert len(service.provider.calls) == 2
    with session_factory() as session:
        attempts = session.query(AssessmentQuestionAttempt).order_by(
            AssessmentQuestionAttempt.attempt_number).all()
        assert session.query(AssessmentQuestion).count() == 0
        run = session.query(AssessmentGenerationRun).one()
    assert [attempt.status for attempt in attempts] == [
        "rejected", "exhausted"]
    assert "duplicate_previous_attempt" in attempts[1].failure_reasons
    assert run.status == "exhausted"


def test_near_duplicate_regeneration_is_rejected(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [
            {"questions": [question(
                chunk_id, "What improves database lookup performance?", forbidden_option=True)]},
            {"questions": [question(
                chunk_id, "What improves database lookup speed?")]},
        ],
        max_attempts=2,
    )
    service.quality_validator = AssessmentQualityValidator(
        AssessmentQualityConfig(near_duplicate_threshold=0.8)
    )

    with pytest.raises(AssessmentGenerationExhaustedError):
        service.generate(request(document_id))

    with session_factory() as session:
        attempts = session.query(AssessmentQuestionAttempt).order_by(
            AssessmentQuestionAttempt.attempt_number).all()
    assert "duplicate_previous_attempt" in attempts[1].failure_reasons


def test_failed_again_reaches_maximum_without_infinite_loop(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    outputs = [
        {"questions": [
            question(chunk_id, f"Question attempt {index}", forbidden_option=True)]}
        for index in range(3)
    ]
    service = make_service(session_factory, document_id,
                           chunk_id, outputs, max_attempts=3)

    with pytest.raises(AssessmentGenerationExhaustedError):
        service.generate(request(document_id))

    assert len(service.provider.calls) == 3
    with session_factory() as session:
        attempts = session.query(AssessmentQuestionAttempt).all()
        assert session.query(AssessmentQuestion).count() == 0
    assert [attempt.attempt_number for attempt in attempts] == [1, 2, 3]


def test_all_questions_eventually_accepted_and_final_persistence_is_atomic(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [
            {"questions": [
                question(chunk_id, "What improves lookup performance?"),
                question(
                    chunk_id, "Which database feature reduces lookup cost?", forbidden_option=True),
            ]},
            {"questions": [
                question(chunk_id, "Which database feature organizes lookup access?")]},
        ],
    )

    _, questions = service.generate(request(document_id, count=2))

    with session_factory() as session:
        assert session.query(AssessmentQuestion).count() == 2
        assert session.query(
            AssessmentGenerationRun).one().status == "completed"
    assert len(questions) == 2


def test_multiple_regeneration_rounds_preserve_accepted_slots(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [
            {"questions": [
                question(chunk_id, "What improves lookup performance?"),
                question(chunk_id, "Which database feature reduces lookup cost?", forbidden_option=True),
            ]},
            {"questions": [
                question(chunk_id, "Which database feature speeds lookups?", forbidden_option=True),
            ]},
            {"questions": [
                question(chunk_id, "Which database feature organizes lookup access?"),
            ]},
        ],
        max_attempts=3,
    )

    _, questions = service.generate(request(document_id, count=2))

    assert [item.stem for item in questions] == [
        "What improves lookup performance?",
        "Which database feature organizes lookup access?",
    ]
    assert len(service.provider.calls) == 3
    with session_factory() as session:
        attempts = session.query(AssessmentQuestionAttempt).order_by(
            AssessmentQuestionAttempt.question_slot,
            AssessmentQuestionAttempt.attempt_number,
        ).all()
    assert [(item.question_slot, item.attempt_number, item.status) for item in attempts] == [
        (0, 1, "accepted"),
        (1, 1, "rejected"),
        (1, 2, "rejected"),
        (1, 3, "accepted"),
    ]


def test_exhaustion_leaves_no_final_questions_and_retains_all_attempts(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [
            {"questions": [
                question(chunk_id, "What improves lookup performance?"),
                question(chunk_id, "Which database feature reduces lookup cost?", forbidden_option=True),
            ]},
            {"questions": [
                question(chunk_id, "Which database feature speeds lookup?", forbidden_option=True),
            ]},
        ],
        max_attempts=2,
    )

    with pytest.raises(AssessmentGenerationExhaustedError):
        service.generate(request(document_id, count=2))

    with session_factory() as session:
        assert session.query(AssessmentQuestion).count() == 0
        attempts = session.query(AssessmentQuestionAttempt).order_by(
            AssessmentQuestionAttempt.question_slot,
            AssessmentQuestionAttempt.attempt_number,
        ).all()
        run = session.query(AssessmentGenerationRun).one()
    assert [(item.question_slot, item.attempt_number, item.status) for item in attempts] == [
        (0, 1, "accepted"),
        (1, 1, "rejected"),
        (1, 2, "exhausted"),
    ]
    assert all(item.failure_reasons for item in attempts if item.status != "accepted")
    assert run.status == "exhausted"


def semantic_result(*, passes: bool) -> SemanticEvaluationResult:
    score = 0.9 if passes else 0.2
    dimension = SemanticDimensionResult(score=score, reason="Fake evaluator result")
    return SemanticEvaluationResult(
        groundedness=dimension,
        correctness=dimension,
        distractor_quality=dimension,
        explanation_quality=dimension,
        difficulty_alignment=dimension,
        bloom_alignment=dimension,
        overall_score=score,
        recommendation="pass" if passes else "fail",
        rationale="Fake evaluator result",
    )


class FakeEvaluator:
    model_name = "fake-evaluator"

    def __init__(self, results: list[SemanticEvaluationResult]) -> None:
        self.results = list(results)
        self.calls = 0
        self.evidence = []

    def evaluate(self, question, **kwargs) -> SemanticEvaluationResult:
        self.calls += 1
        self.evidence.append(kwargs["evidence"])
        return self.results.pop(0)


def test_deterministic_failure_skips_semantic_evaluator(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    evaluator = FakeEvaluator([semantic_result(passes=True)])
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [{"questions": [question(chunk_id, "What improves lookup performance?", forbidden_option=True)]}],
        max_attempts=1,
        evaluator=evaluator,
    )

    with pytest.raises(AssessmentGenerationExhaustedError):
        service.generate(request(document_id))

    assert evaluator.calls == 0


def test_semantic_failure_regenerates_only_the_failed_slot(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    evaluator = FakeEvaluator([semantic_result(passes=True), semantic_result(passes=False), semantic_result(passes=True)])
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [
            {"questions": [
                question(chunk_id, "What improves lookup performance?"),
                question(chunk_id, "Which database feature reduces lookup cost?"),
            ]},
            {"questions": [question(chunk_id, "Which database feature organizes lookup access?")]},
        ],
        evaluator=evaluator,
    )

    _, questions = service.generate(request(document_id, count=2))

    assert [item.stem for item in questions] == [
        "What improves lookup performance?",
        "Which database feature organizes lookup access?",
    ]
    assert evaluator.calls == 3
    assert [item.chunk_id for item in evaluator.evidence[0]] == [chunk_id]
    with session_factory() as session:
        attempts = session.query(AssessmentQuestionAttempt).order_by(
            AssessmentQuestionAttempt.question_slot,
            AssessmentQuestionAttempt.attempt_number,
        ).all()
    assert [(item.question_slot, item.attempt_number, item.status) for item in attempts] == [
        (0, 1, "accepted"),
        (1, 1, "rejected"),
        (1, 2, "accepted"),
    ]
    assert "semantic_quality_below_threshold" in attempts[1].failure_reasons


def test_evaluator_failure_is_audited_and_never_accepts(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)

    class FailingEvaluator:
        model_name = "failing-evaluator"

        def evaluate(self, question, **kwargs):
            raise RuntimeError("provider unavailable")

    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        [{"questions": [question(chunk_id, "What improves lookup performance?")]}],
        evaluator=FailingEvaluator(),
    )

    with pytest.raises(AssessmentSemanticEvaluationError):
        service.generate(request(document_id))

    with session_factory() as session:
        attempt = session.query(AssessmentQuestionAttempt).one()
        run = session.query(AssessmentGenerationRun).one()
        evaluations = session.query(AssessmentQualityEvaluation).all()
        assert session.query(AssessmentQuestion).count() == 0
    assert attempt.status == "evaluation_failed"
    assert attempt.failure_reasons == ["semantic_evaluation_unavailable"]
    assert run.status == "failed"
    assert len(evaluations) == 1
    assert evaluations[0].evaluation_type == "deterministic"
    assert evaluations[0].status == "evaluation_failed"
