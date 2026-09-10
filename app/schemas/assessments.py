from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from app.domain.assessment import BloomLevel, Difficulty, QuestionType


class AssessmentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class QuestionGenerationRequest(AssessmentSchema):
    query: str
    skill: str | None = None
    document_id: UUID | None = None
    question_type: QuestionType = QuestionType.MCQ
    difficulty: Difficulty | None = None
    bloom_level: BloomLevel | None = None
    count: StrictInt = Field(default=1, ge=1, le=20)
    top_k: StrictInt = Field(default=5, ge=1, le=50)
    candidate_k: StrictInt | None = Field(default=None, ge=1, le=100)
    seed: StrictInt | None = None
    learner_id: str | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be empty")
        return value.strip()

    @field_validator("skill")
    @classmethod
    def skill_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Skill must not be blank")
        return value.strip() if value is not None else None

    @field_validator("learner_id")
    @classmethod
    def learner_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("learner_id must not be blank")
        return value.strip() if value is not None else None


class QuestionOptionResponse(AssessmentSchema):
    id: str
    option_key: str
    option_text: str
    display_order: int


class QuestionCitationResponse(AssessmentSchema):
    id: str
    chunk_id: str
    citation_order: int
    retrieval_rank: int
    reranker_score: float
    hybrid_score: float
    chunk_text_snapshot: str
    page_number_snapshot: int | None = None


class QuestionResponse(AssessmentSchema):
    id: str
    generation_run_id: str
    document_id: str | None = None
    question_type: QuestionType
    stem: str
    explanation: str
    difficulty: Difficulty
    bloom_level: BloomLevel
    topic: str
    skill: str
    correct_option_key: str | None = None
    content_version: int
    created_at: datetime
    options: list[QuestionOptionResponse] = Field(default_factory=list)
    citations: list[QuestionCitationResponse] = Field(default_factory=list)


class GenerationRunResponse(AssessmentSchema):
    id: str
    request_fingerprint: str
    document_id: str | None = None
    query: str
    skill: str | None = None
    question_type: QuestionType
    difficulty: Difficulty | None = None
    bloom_level: BloomLevel | None = None
    requested_count: int
    retrieval_top_k: int
    retrieval_candidate_k: int | None = None
    embedding_model_name: str
    reranker_model_name: str
    llm_model_name: str
    prompt_version: str
    seed: int | None = None
    status: str
    failure_reason: str | None = None
    created_at: datetime


class QuestionListFilter(AssessmentSchema):
    document_id: UUID | None = None
    question_type: QuestionType | None = None
    difficulty: Difficulty | None = None
    bloom_level: BloomLevel | None = None
    topic: str | None = None
    skill: str | None = None
    limit: StrictInt = Field(default=50, ge=1, le=100)
    offset: StrictInt = Field(default=0, ge=0)


class QuestionListResponse(AssessmentSchema):
    questions: list[QuestionResponse]
    limit: int
    offset: int


class QualityEvaluationResponse(AssessmentSchema):
    id: str
    generation_run_id: str
    question_attempt_id: str
    question_id: str | None = None
    evaluation_type: str
    evaluator_provider: str | None = None
    evaluator_model: str | None = None
    prompt_version: str | None = None
    overall_score: float | None = None
    dimension_scores: dict[str, object] | None = None
    recommendation: str
    rationale: str
    status: str
    created_at: datetime


class QualityEvaluationListResponse(AssessmentSchema):
    evaluations: list[QualityEvaluationResponse]
