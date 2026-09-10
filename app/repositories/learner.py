from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.learner import LearnerQuizSession


class LearnerRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

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
