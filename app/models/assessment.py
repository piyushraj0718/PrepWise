from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AssessmentGenerationRun(Base):
    __tablename__ = "assessment_generation_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()))
    request_fingerprint: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True)
    document_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("documents.id"), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    skill: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question_type: Mapped[str] = mapped_column(String(20), nullable=False)
    difficulty: Mapped[str | None] = mapped_column(String(20), nullable=True)
    bloom_level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    requested_count: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_candidate_k: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    embedding_model_name: Mapped[str] = mapped_column(
        String(255), nullable=False)
    reranker_model_name: Mapped[str] = mapped_column(
        String(255), nullable=False)
    llm_model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(
        String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    questions: Mapped[list["AssessmentQuestion"]] = relationship(
        back_populates="generation_run", cascade="all, delete-orphan"
    )
    attempts: Mapped[list["AssessmentQuestionAttempt"]] = relationship(
        back_populates="generation_run", cascade="all, delete-orphan"
    )
    quality_evaluations: Mapped[list["AssessmentQualityEvaluation"]] = relationship(
        back_populates="generation_run", cascade="all, delete-orphan"
    )


class AssessmentQuestion(Base):
    __tablename__ = "assessment_questions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()))
    generation_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessment_generation_runs.id"), nullable=False, index=True
    )
    document_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("documents.id"), nullable=True, index=True)
    question_type: Mapped[str] = mapped_column(String(20), nullable=False)
    stem: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False)
    bloom_level: Mapped[str] = mapped_column(String(20), nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    skill: Mapped[str] = mapped_column(String(255), nullable=False)
    correct_option_key: Mapped[str | None] = mapped_column(
        String(20), nullable=True)
    content_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    generation_run: Mapped[AssessmentGenerationRun] = relationship(
        back_populates="questions")
    options: Mapped[list["AssessmentQuestionOption"]] = relationship(
        back_populates="question", cascade="all, delete-orphan", order_by="AssessmentQuestionOption.display_order"
    )
    citations: Mapped[list["AssessmentQuestionCitation"]] = relationship(
        back_populates="question", cascade="all, delete-orphan", order_by="AssessmentQuestionCitation.citation_order"
    )
    quality_evaluations: Mapped[list["AssessmentQualityEvaluation"]] = relationship(
        back_populates="question"
    )


class AssessmentQuestionAttempt(Base):
    __tablename__ = "assessment_question_attempts"
    __table_args__ = (
        UniqueConstraint(
            "generation_run_id",
            "question_slot",
            "attempt_number",
            name="uq_assessment_question_attempt",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()))
    generation_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessment_generation_runs.id"),
        nullable=False, index=True
    )
    question_slot: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    candidate_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    failure_reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)
    accepted_question_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("assessment_questions.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    generation_run: Mapped[AssessmentGenerationRun] = relationship(
        back_populates="attempts")
    accepted_question: Mapped["AssessmentQuestion | None"] = relationship(
        foreign_keys=[accepted_question_id])
    quality_evaluations: Mapped[list["AssessmentQualityEvaluation"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )


class AssessmentQualityEvaluation(Base):
    __tablename__ = "assessment_quality_evaluations"
    __table_args__ = (
        UniqueConstraint("question_attempt_id", "evaluation_type", name="uq_attempt_quality_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    generation_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessment_generation_runs.id"), nullable=False, index=True
    )
    question_attempt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessment_question_attempts.id"), nullable=False, index=True
    )
    question_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("assessment_questions.id"), nullable=True, index=True
    )
    evaluation_type: Mapped[str] = mapped_column(String(30), nullable=False)
    evaluator_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    evaluator_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    overall_score: Mapped[float | None] = mapped_column(nullable=True)
    dimension_scores: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    recommendation: Mapped[str] = mapped_column(String(20), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    generation_run: Mapped[AssessmentGenerationRun] = relationship(back_populates="quality_evaluations")
    attempt: Mapped[AssessmentQuestionAttempt] = relationship(back_populates="quality_evaluations")
    question: Mapped["AssessmentQuestion | None"] = relationship(back_populates="quality_evaluations")


class AssessmentQuestionOption(Base):
    __tablename__ = "assessment_question_options"
    __table_args__ = (
        UniqueConstraint("question_id", "option_key",
                         name="uq_assessment_option_key"),
        UniqueConstraint("question_id", "display_order",
                         name="uq_assessment_option_order"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()))
    question_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessment_questions.id"), nullable=False, index=True
    )
    option_key: Mapped[str] = mapped_column(String(20), nullable=False)
    option_text: Mapped[str] = mapped_column(Text, nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    question: Mapped[AssessmentQuestion] = relationship(
        back_populates="options")


class AssessmentQuestionCitation(Base):
    __tablename__ = "assessment_question_citations"
    __table_args__ = (
        UniqueConstraint("question_id", "citation_order",
                         name="uq_assessment_citation_order"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()))
    question_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessment_questions.id"), nullable=False, index=True
    )
    chunk_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("document_chunks.id"), nullable=False, index=True
    )
    citation_order: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    reranker_score: Mapped[float] = mapped_column(nullable=False)
    hybrid_score: Mapped[float] = mapped_column(nullable=False)
    chunk_text_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    page_number_snapshot: Mapped[int | None] = mapped_column(
        Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    question: Mapped[AssessmentQuestion] = relationship(
        back_populates="citations")
