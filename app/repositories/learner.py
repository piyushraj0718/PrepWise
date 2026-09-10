from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.assessment import AreaPerformance, compute_accuracy
from app.models.assessment import AssessmentQuestion
from app.models.learner import LearnerAnswerSubmission, LearnerQuizSession


class DuplicateSubmissionError(Exception):
    """Raised when an answer for (session_id, question_id) already exists."""


class LearnerRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Quiz sessions
    # ------------------------------------------------------------------

    def create_session(self, **values: Any) -> LearnerQuizSession:
        """Persist a new quiz session and return it."""
        try:
            session = LearnerQuizSession(**values)
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            return session
        except Exception:
            self.db.rollback()
            raise

    def get_session(self, session_id: str) -> LearnerQuizSession | None:
        """Return the quiz session with the given ID, or None."""
        result = self.db.scalar(
            select(LearnerQuizSession).where(LearnerQuizSession.id == session_id)
        )
        if result is not None:
            self.db.expunge(result)
        return result

    def complete_session(self, session_id: str) -> LearnerQuizSession:
        """Set session status to 'completed' and record completed_at timestamp.

        Returns the updated session. Caller must verify the session exists
        and is not already completed before calling.
        """
        try:
            session = self.db.scalar(
                select(LearnerQuizSession).where(LearnerQuizSession.id == session_id)
            )
            session.status = "completed"
            session.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(session)
            self.db.expunge(session)
            return session
        except Exception:
            self.db.rollback()
            raise

    # ------------------------------------------------------------------
    # Answer submissions
    # ------------------------------------------------------------------

    def submit_answer(
        self,
        session_id: str,
        question_id: str,
        learner_id: str,
        submitted_option_key: str,
        is_correct: bool,
    ) -> LearnerAnswerSubmission:
        """Persist a single answer submission.

        Raises DuplicateSubmissionError if the learner has already submitted
        an answer for this question in this session.
        """
        submission = LearnerAnswerSubmission(
            session_id=session_id,
            question_id=question_id,
            learner_id=learner_id,
            submitted_option_key=submitted_option_key,
            is_correct=is_correct,
        )
        try:
            self.db.add(submission)
            self.db.commit()
            self.db.refresh(submission)
            self.db.expunge(submission)
            return submission
        except IntegrityError:
            self.db.rollback()
            raise DuplicateSubmissionError(
                f"An answer for question {question_id} in session {session_id} "
                "has already been submitted."
            )
        except Exception:
            self.db.rollback()
            raise

    def get_submission(
        self, session_id: str, question_id: str
    ) -> LearnerAnswerSubmission | None:
        """Return a single submission for (session_id, question_id), or None."""
        result = self.db.scalar(
            select(LearnerAnswerSubmission).where(
                LearnerAnswerSubmission.session_id == session_id,
                LearnerAnswerSubmission.question_id == question_id,
            )
        )
        if result is not None:
            self.db.expunge(result)
        return result

    def get_submissions_for_session(
        self, session_id: str
    ) -> list[LearnerAnswerSubmission]:
        """Return all submissions for a session, ordered by submitted_at."""
        rows = self.db.scalars(
            select(LearnerAnswerSubmission)
            .where(LearnerAnswerSubmission.session_id == session_id)
            .order_by(LearnerAnswerSubmission.submitted_at)
        ).all()
        for row in rows:
            self.db.expunge(row)
        return list(rows)

    # ------------------------------------------------------------------
    # M4C: Performance aggregation
    # ------------------------------------------------------------------

    def get_all_submissions_for_learner(
        self, learner_id: str
    ) -> list[LearnerAnswerSubmission]:
        """Return every submission ever made by a learner, ordered by submitted_at."""
        rows = self.db.scalars(
            select(LearnerAnswerSubmission)
            .where(LearnerAnswerSubmission.learner_id == learner_id)
            .order_by(LearnerAnswerSubmission.submitted_at)
        ).all()
        for row in rows:
            self.db.expunge(row)
        return list(rows)

    def get_topic_performance_for_learner(
        self, learner_id: str
    ) -> list[AreaPerformance]:
        """Return per-topic performance aggregates for a learner.

        Joins submissions to assessment_questions to read the topic label.
        Only submissions whose question still exists are included.
        Results are sorted by topic label ascending.
        """
        rows = self.db.execute(
            select(
                AssessmentQuestion.topic,
                LearnerAnswerSubmission.is_correct,
            )
            .join(
                AssessmentQuestion,
                LearnerAnswerSubmission.question_id == AssessmentQuestion.id,
            )
            .where(LearnerAnswerSubmission.learner_id == learner_id)
        ).all()
        return _aggregate_area_performance(rows, label_index=0)

    def get_skill_performance_for_learner(
        self, learner_id: str
    ) -> list[AreaPerformance]:
        """Return per-skill performance aggregates for a learner.

        Joins submissions to assessment_questions to read the skill label.
        Only submissions whose question still exists are included.
        Results are sorted by skill label ascending.
        """
        rows = self.db.execute(
            select(
                AssessmentQuestion.skill,
                LearnerAnswerSubmission.is_correct,
            )
            .join(
                AssessmentQuestion,
                LearnerAnswerSubmission.question_id == AssessmentQuestion.id,
            )
            .where(LearnerAnswerSubmission.learner_id == learner_id)
        ).all()
        return _aggregate_area_performance(rows, label_index=0)


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _aggregate_area_performance(
    rows: list,
    label_index: int,
) -> list[AreaPerformance]:
    """Aggregate (label, is_correct) rows into AreaPerformance objects.

    rows: sequence of row-tuples where rows[i][label_index] is the label string
          and rows[i][1] is the is_correct bool.
    Returns a list sorted by label ascending.
    """
    totals: dict[str, list[int]] = {}  # label -> [attempts, correct]
    for row in rows:
        label: str = row[label_index]
        is_correct: bool = row[1]
        bucket = totals.setdefault(label, [0, 0])
        bucket[0] += 1
        if is_correct:
            bucket[1] += 1
    return sorted(
        [
            AreaPerformance(
                label=label,
                attempts=counts[0],
                correct=counts[1],
                accuracy=compute_accuracy(counts[0], counts[1]),
            )
            for label, counts in totals.items()
        ],
        key=lambda p: p.label,
    )
