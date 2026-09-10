"""Learner-facing Pydantic schemas for M4A.

These schemas deliberately omit fields that must not be visible to a learner
before they have submitted an answer:
- correct_option_key is excluded from LearnerQuestionResponse
- citations are excluded (they expose internal source metadata)
"""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class LearnerSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


# ---------------------------------------------------------------------------
# Quiz session schemas
# ---------------------------------------------------------------------------

class QuizSessionCreateRequest(LearnerSchema):
    learner_id: str
    question_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("learner_id")
    @classmethod
    def learner_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("learner_id must not be blank")
        return value.strip()

    @field_validator("question_ids")
    @classmethod
    def question_ids_must_not_be_empty_strings(cls, value: list[str]) -> list[str]:
        if any(not qid.strip() for qid in value):
            raise ValueError("question_ids must not contain blank entries")
        return value


class QuizSessionResponse(LearnerSchema):
    id: str
    learner_id: str
    question_ids: list[str]
    status: str
    created_at: datetime
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Learner-safe question schemas (no correct_option_key, no citations)
# ---------------------------------------------------------------------------

class LearnerQuestionOptionResponse(LearnerSchema):
    id: str
    option_key: str
    option_text: str
    display_order: int


class LearnerQuestionResponse(LearnerSchema):
    """Question view exposed to a learner before answer submission.

    Intentionally omits correct_option_key and citations.
    """
    id: str
    generation_run_id: str
    document_id: str | None = None
    question_type: str
    stem: str
    explanation: str
    difficulty: str
    bloom_level: str
    topic: str
    skill: str
    content_version: int
    created_at: datetime
    options: list[LearnerQuestionOptionResponse] = Field(default_factory=list)


class QuizSessionQuestionsResponse(LearnerSchema):
    session_id: str
    questions: list[LearnerQuestionResponse]
