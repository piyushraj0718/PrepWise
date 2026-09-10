from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from app.domain.assessment import BloomLevel, Difficulty
from app.schemas.assessments import (
    GenerationRunResponse,
    QuestionResponse,
)


BLOOM_LEVEL_ALIASES = {
    "remember": "remember",
    "knowledge": "remember",
    "understand": "understand",
    "comprehension": "understand",
    "understanding": "understand",
    "apply": "apply",
    "application": "apply",
    "analyze": "analyze",
    "analysis": "analyze",
    "evaluate": "evaluate",
    "evaluation": "evaluate",
    "create": "create",
    "creation": "create",
    "synthesis": "create",
}


def normalize_bloom_level(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.strip().casefold()
    return BLOOM_LEVEL_ALIASES.get(normalized, value)


class AssessmentOutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GeneratedOption(AssessmentOutputSchema):
    option_key: StrictStr
    option_text: StrictStr

    @field_validator("option_key", "option_text")
    @classmethod
    def value_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Generated option values must not be empty")
        return value


class GeneratedMCQ(AssessmentOutputSchema):
    stem: StrictStr
    options: list[GeneratedOption]
    correct_option_key: StrictStr
    explanation: StrictStr
    difficulty: Difficulty
    bloom_level: BloomLevel
    topic: StrictStr
    skill: StrictStr
    cited_chunk_ids: list[StrictStr] = Field(min_length=1)

    @field_validator("bloom_level", mode="before")
    @classmethod
    def normalize_bloom_alias(cls, value: object) -> object:
        return normalize_bloom_level(value)

    @field_validator("stem", "correct_option_key", "explanation", "topic", "skill")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Generated text must not be empty")
        return value

    @field_validator("cited_chunk_ids")
    @classmethod
    def citations_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if any(not citation_id for citation_id in value):
            raise ValueError("Citation IDs must not be empty")
        return value

    @model_validator(mode="after")
    def validate_mcq_shape(self) -> "GeneratedMCQ":
        if len(self.options) != 4:
            raise ValueError("Each generated MCQ must have exactly 4 options")
        keys = [option.option_key for option in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError("Generated option keys must be unique")
        if self.correct_option_key not in keys:
            raise ValueError("The correct option key must exist")
        return self


class GeneratedAssessment(AssessmentOutputSchema):
    questions: list[GeneratedMCQ]


class QuestionGenerationResponse(AssessmentOutputSchema):
    run: GenerationRunResponse
    questions: list[QuestionResponse]
