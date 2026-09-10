from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
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
