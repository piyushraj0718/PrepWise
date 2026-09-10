from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai.assessment_evaluation import (
    GeminiAssessmentQuestionEvaluator,
    SemanticEvaluationResult,
    TrustedEvaluationEvidence,
)
from app.ai.assessment_prompts import ASSESSMENT_EVALUATION_PROMPT_VERSION
from app.domain.assessment_semantic import SemanticQualityPolicy
from app.schemas.assessment_generation import GeneratedMCQ


def valid_output(*, score: float = 0.9, recommendation: str = "pass") -> dict:
    dimension = {"score": score, "reason": "Supported by the supplied evidence."}
    return {
        "groundedness": dimension,
        "correctness": dimension,
        "distractor_quality": dimension,
        "explanation_quality": dimension,
        "difficulty_alignment": dimension,
        "bloom_alignment": dimension,
        "overall_score": score,
        "recommendation": recommendation,
        "rationale": "The question is supported by the trusted source.",
    }


def generated_question() -> GeneratedMCQ:
    return GeneratedMCQ.model_validate({
        "stem": "What improves lookup performance?",
        "options": [
            {"option_key": "A", "option_text": "An index"},
            {"option_key": "B", "option_text": "A slower disk"},
            {"option_key": "C", "option_text": "A blank table"},
            {"option_key": "D", "option_text": "A deleted query"},
        ],
        "correct_option_key": "A",
        "explanation": "An index improves lookup performance.",
        "difficulty": "easy",
        "bloom_level": "understand",
        "topic": "indexes",
        "skill": "identify index benefits",
        "cited_chunk_ids": ["chunk-1"],
    })


def test_semantic_evaluator_schema_accepts_valid_output() -> None:
    result = SemanticEvaluationResult.model_validate(valid_output())

    assert result.overall_score == 0.9
    assert result.recommendation == "pass"
    assert SemanticQualityPolicy(0.75).passes(result)


@pytest.mark.parametrize(
    "output",
    [
        {},
        {key: value for key, value in valid_output().items() if key != "groundedness"},
        valid_output(score=1.1),
        valid_output(recommendation="approve"),
    ],
)
def test_semantic_evaluator_schema_rejects_malformed_output(output: dict) -> None:
    with pytest.raises(ValidationError):
        SemanticEvaluationResult.model_validate(output)


def test_semantic_policy_requires_recommendation_and_threshold() -> None:
    assert not SemanticQualityPolicy(0.75).passes(
        SemanticEvaluationResult.model_validate(valid_output(score=0.7))
    )
    assert not SemanticQualityPolicy(0.75).passes(
        SemanticEvaluationResult.model_validate(valid_output(recommendation="fail"))
    )
    output = valid_output()
    output["groundedness"] = {"score": 0.2, "reason": "The source does not support it."}
    assert not SemanticQualityPolicy(0.75).passes(
        SemanticEvaluationResult.model_validate(output)
    )


def test_gemini_evaluator_receives_only_trusted_bounded_evidence() -> None:
    class FakeProvider:
        model_name = "fake-evaluator"

        def __init__(self) -> None:
            self.prompt = ""

        def generate_structured(self, prompt: str, response_schema: dict[str, object]) -> object:
            self.prompt = prompt
            return valid_output()

    provider = FakeProvider()
    evaluator = GeminiAssessmentQuestionEvaluator(provider)
    result = evaluator.evaluate(
        generated_question(),
        requested_difficulty="easy",
        requested_bloom_level="understand",
        requested_topic="indexes",
        requested_skill="identify index benefits",
        evidence=[TrustedEvaluationEvidence("chunk-1", "Indexes improve lookup performance.")],
    )

    assert result.recommendation == "pass"
    assert ASSESSMENT_EVALUATION_PROMPT_VERSION == "mcq-semantic-evaluation-v1"
    assert "chunk-1" in provider.prompt
    assert "Indexes improve lookup performance." in provider.prompt
    assert "citation IDs" in provider.prompt
