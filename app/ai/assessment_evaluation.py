from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr, field_validator

from app.ai.assessment import AssessmentLLMProvider
from app.ai.assessment_prompts import build_assessment_evaluation_prompt
from app.ai.llm import LLMProviderError, LLMResponseError
from app.schemas.assessment_generation import GeneratedMCQ


class SemanticEvaluationError(RuntimeError):
    """Raised when semantic evaluation cannot produce a trusted result."""


class SemanticDimensionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    score: StrictFloat = Field(ge=0.0, le=1.0)
    reason: StrictStr

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Evaluation reasons must not be empty")
        return value


class SemanticEvaluationResult(BaseModel):
    """Strict, non-authoritative evaluator output on a 0.0 to 1.0 scale."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    groundedness: SemanticDimensionResult
    correctness: SemanticDimensionResult
    distractor_quality: SemanticDimensionResult
    explanation_quality: SemanticDimensionResult
    difficulty_alignment: SemanticDimensionResult
    bloom_alignment: SemanticDimensionResult
    overall_score: StrictFloat = Field(ge=0.0, le=1.0)
    recommendation: Literal["pass", "fail"]
    rationale: StrictStr

    @field_validator("rationale")
    @classmethod
    def rationale_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Evaluation rationale must not be empty")
        return value


@dataclass(frozen=True)
class TrustedEvaluationEvidence:
    chunk_id: str
    chunk_text: str


class AssessmentQuestionEvaluator(Protocol):
    model_name: str

    def evaluate(
        self,
        question: GeneratedMCQ,
        *,
        requested_difficulty: str | None,
        requested_bloom_level: str | None,
        requested_topic: str | None,
        requested_skill: str | None,
        evidence: Sequence[TrustedEvaluationEvidence],
    ) -> SemanticEvaluationResult:
        ...


class GeminiAssessmentQuestionEvaluator:
    """Gemini-backed evaluator behind the provider-neutral evaluator protocol."""

    def __init__(self, provider: AssessmentLLMProvider) -> None:
        self.provider = provider
        self.model_name = str(getattr(provider, "model_name", "unknown"))

    def evaluate(
        self,
        question: GeneratedMCQ,
        *,
        requested_difficulty: str | None,
        requested_bloom_level: str | None,
        requested_topic: str | None,
        requested_skill: str | None,
        evidence: Sequence[TrustedEvaluationEvidence],
    ) -> SemanticEvaluationResult:
        prompt = build_assessment_evaluation_prompt(
            question=question.model_dump(mode="json"),
            requested_difficulty=requested_difficulty,
            requested_bloom_level=requested_bloom_level,
            requested_topic=requested_topic,
            requested_skill=requested_skill,
            evidence=[
                {"chunk_id": item.chunk_id, "chunk_text": item.chunk_text}
                for item in evidence
            ],
        )
        try:
            generate_structured = getattr(self.provider, "generate_structured")
            raw_result = generate_structured(prompt, semantic_evaluation_response_schema())
            return SemanticEvaluationResult.model_validate(raw_result)
        except (LLMProviderError, LLMResponseError):
            raise
        except Exception as error:
            raise SemanticEvaluationError(
                "The semantic evaluator returned an invalid result"
            ) from error


def semantic_evaluation_response_schema() -> dict[str, object]:
    dimension = {
        "type": "OBJECT",
        "additionalProperties": False,
        "properties": {"score": {"type": "NUMBER"}, "reason": {"type": "STRING"}},
        "required": ["score", "reason"],
    }
    return {
        "type": "OBJECT",
        "additionalProperties": False,
        "properties": {
            "groundedness": dimension,
            "correctness": dimension,
            "distractor_quality": dimension,
            "explanation_quality": dimension,
            "difficulty_alignment": dimension,
            "bloom_alignment": dimension,
            "overall_score": {"type": "NUMBER"},
            "recommendation": {"type": "STRING", "enum": ["pass", "fail"]},
            "rationale": {"type": "STRING"},
        },
        "required": [
            "groundedness", "correctness", "distractor_quality", "explanation_quality",
            "difficulty_alignment", "bloom_alignment", "overall_score", "recommendation",
            "rationale",
        ],
    }
