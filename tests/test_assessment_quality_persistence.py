from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.ai.assessment_evaluation import SemanticDimensionResult, SemanticEvaluationResult
from app.api.dependencies import (
    get_assessment_generation_service,
    get_assessment_repository,
)
from app.domain.assessment_semantic import SemanticQualityPolicy
from app.models.assessment import AssessmentQualityEvaluation, AssessmentQuestionAttempt
from app.repositories.assessments import AssessmentRepository
from app.schemas.assessments import QuestionGenerationRequest
from app.services.assessment_generation import (
    AssessmentGenerationService,
    AssessmentSemanticEvaluationError,
)
from app.services.reranking import RerankedResult
from tests.test_question_generation import (
    FakeAssessmentLLM,
    FakeReranking,
    configure_generation_api,
    make_result,
    make_source,
    request,
    valid_question,
)


def make_service(
    session_factory,
    document_id: str,
    chunk_id: str,
    output: object,
    *,
    evaluator=None,
) -> AssessmentGenerationService:
    return AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        FakeAssessmentLLM(output),
        "fake-embedding",
        "fake-reranker",
        evaluator=evaluator,
        semantic_evaluation_enabled=evaluator is not None,
        semantic_quality_policy=SemanticQualityPolicy(0.75),
    )


def semantic_result(*, passes: bool = True) -> SemanticEvaluationResult:
    score = 0.9 if passes else 0.2
    dimension = SemanticDimensionResult(
        score=score, reason="Supported by the supplied evidence."
    )
    return SemanticEvaluationResult(
        groundedness=dimension,
        correctness=dimension,
        distractor_quality=dimension,
        explanation_quality=dimension,
        difficulty_alignment=dimension,
        bloom_alignment=dimension,
        overall_score=score,
        recommendation="pass" if passes else "fail",
        rationale="The question is supported by the trusted source.",
    )


class FakeEvaluator:
    model_name = "fake-evaluator"

    def __init__(self, results: list[SemanticEvaluationResult]) -> None:
        self.results = list(results)

    def evaluate(self, question, **kwargs) -> SemanticEvaluationResult:
        return self.results.pop(0)


def test_successful_generation_persists_deterministic_quality(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        {"questions": [valid_question(chunk_id)]},
    )

    run, questions = service.generate(QuestionGenerationRequest(**request(document_id)))

    with session_factory() as session:
        evaluations = session.query(AssessmentQualityEvaluation).order_by(
            AssessmentQualityEvaluation.evaluation_type
        ).all()
        attempt = session.query(AssessmentQuestionAttempt).one()
    assert len(evaluations) == 1
    assert evaluations[0].evaluation_type == "deterministic"
    assert evaluations[0].generation_run_id == run.id
    assert evaluations[0].question_attempt_id == attempt.id
    assert evaluations[0].question_id == questions[0].id
    assert evaluations[0].recommendation == "pass"
    assert evaluations[0].status == "accepted"
    assert evaluations[0].dimension_scores == {"hard_failure_codes": []}


def test_rejected_attempt_persists_quality_without_question_id(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    from app.domain.assessment_quality import RegenerationDecisionPolicy

    bad_question = valid_question(chunk_id)
    bad_question["options"][1]["option_text"] = "all of the above"
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        FakeAssessmentLLM({"questions": [bad_question]}),
        "fake-embedding",
        "fake-reranker",
        regeneration_policy=RegenerationDecisionPolicy(max_attempts=2),
    )

    with pytest.raises(Exception):
        service.generate(QuestionGenerationRequest(**request(document_id)))

    with session_factory() as session:
        evaluations = session.query(AssessmentQualityEvaluation).order_by(
            AssessmentQualityEvaluation.created_at,
            AssessmentQualityEvaluation.id,
        ).all()
    assert len(evaluations) == 2
    assert [item.status for item in evaluations] == ["rejected", "exhausted"]
    assert all(item.question_id is None for item in evaluations)
    assert all(item.evaluation_type == "deterministic" for item in evaluations)


def test_semantic_evaluation_persists_both_records(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    evaluator = FakeEvaluator([semantic_result(passes=True)])
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        {"questions": [valid_question(chunk_id)]},
        evaluator=evaluator,
    )

    _, questions = service.generate(QuestionGenerationRequest(**request(document_id)))

    with session_factory() as session:
        evaluations = session.query(AssessmentQualityEvaluation).order_by(
            AssessmentQualityEvaluation.evaluation_type
        ).all()
    assert len(evaluations) == 2
    assert evaluations[0].evaluation_type == "deterministic"
    assert evaluations[1].evaluation_type == "semantic"
    assert evaluations[1].evaluator_provider == "gemini"
    assert evaluations[1].evaluator_model == "fake-evaluator"
    assert evaluations[1].overall_score == 0.9
    assert evaluations[1].dimension_scores == {
        "groundedness": 0.9,
        "correctness": 0.9,
        "distractor_quality": 0.9,
        "explanation_quality": 0.9,
        "difficulty_alignment": 0.9,
        "bloom_alignment": 0.9,
    }
    assert evaluations[1].question_id == questions[0].id


def test_evaluation_failed_persists_deterministic_quality(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)

    class FailingEvaluator:
        model_name = "failing-evaluator"

        def evaluate(self, question, **kwargs):
            raise RuntimeError("provider unavailable")

    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        {"questions": [valid_question(chunk_id)]},
        evaluator=FailingEvaluator(),
    )

    with pytest.raises(AssessmentSemanticEvaluationError):
        service.generate(QuestionGenerationRequest(**request(document_id)))

    with session_factory() as session:
        attempt = session.query(AssessmentQuestionAttempt).one()
        evaluations = session.query(AssessmentQualityEvaluation).all()
    assert attempt.status == "evaluation_failed"
    assert len(evaluations) == 1
    assert evaluations[0].evaluation_type == "deterministic"
    assert evaluations[0].status == "evaluation_failed"
    assert evaluations[0].recommendation == "pass"
    assert evaluations[0].question_id is None


def test_quality_read_apis(
    client: TestClient, session_factory
) -> None:
    document_id, chunk_id = make_source(session_factory)
    configure_generation_api(
        client,
        session_factory,
        document_id,
        chunk_id,
        {"questions": [valid_question(chunk_id)]},
    )
    response = client.post("/questions/generate", json=request(document_id))
    assert response.status_code == 201
    body = response.json()
    question_id = body["questions"][0]["id"]
    run_id = body["run"]["id"]

    question_quality = client.get(f"/questions/{question_id}/quality")
    run_quality = client.get(f"/question-generation-runs/{run_id}/quality")
    assert question_quality.status_code == run_quality.status_code == 200
    assert len(question_quality.json()["evaluations"]) == 1
    assert len(run_quality.json()["evaluations"]) == 1
    evaluation = question_quality.json()["evaluations"][0]
    assert evaluation["evaluation_type"] == "deterministic"
    assert evaluation["question_id"] == question_id
    assert evaluation["generation_run_id"] == run_id


def test_quality_read_apis_return_not_found(client: TestClient) -> None:
    missing_id = str(uuid4())
    assert client.get(f"/questions/{missing_id}/quality").status_code == 404
    assert client.get(
        f"/question-generation-runs/{missing_id}/quality"
    ).status_code == 404
