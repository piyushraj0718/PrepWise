from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class LearnerQuizSession(Base):
    __tablename__ = "learner_quiz_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()))
    learner_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True)
    question_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True)


class LearnerAnswerSubmission(Base):
    __tablename__ = "learner_answer_submissions"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "question_id",
            name="uq_learner_submission_session_question",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("learner_quiz_sessions.id"),
        nullable=False, index=True)
    question_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("assessment_questions.id"),
        nullable=False, index=True)
    learner_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True)
    submitted_option_key: Mapped[str] = mapped_column(
        String(20), nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
