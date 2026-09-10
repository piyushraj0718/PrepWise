from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.assessment import (
    AssessmentGenerationRun,
    AssessmentQuestion,
    AssessmentQuestionCitation,
    AssessmentQuestionOption,
)


class AssessmentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_generation_run(self, **values: Any) -> AssessmentGenerationRun:
        run = AssessmentGenerationRun(**values)
        self.db.add(run)
        self.db.flush()
        return run

    def create_questions(
        self, generation_run: AssessmentGenerationRun, question_values: Sequence[dict[str, Any]]
    ) -> list[AssessmentQuestion]:
        questions: list[AssessmentQuestion] = []
        for values in question_values:
            question = AssessmentQuestion(
                generation_run=generation_run, **values)
            self.db.add(question)
            questions.append(question)
        self.db.flush()
        return questions

    def create_options(
        self, question: AssessmentQuestion, option_values: Sequence[dict[str, Any]]
    ) -> list[AssessmentQuestionOption]:
        options = [
            AssessmentQuestionOption(question=question, **values)
            for values in option_values
        ]
        self.db.add_all(options)
        self.db.flush()
        return options

    def create_citations(
        self, question: AssessmentQuestion, citation_values: Sequence[dict[str, Any]]
    ) -> list[AssessmentQuestionCitation]:
        citations = [
            AssessmentQuestionCitation(question=question, **values)
            for values in citation_values
        ]
        self.db.add_all(citations)
        self.db.flush()
        return citations

    def create_generation_batch(
        self,
        run_values: dict[str, Any],
        question_values: Sequence[dict[str, Any]],
    ) -> tuple[AssessmentGenerationRun, list[AssessmentQuestion]]:
        """Persist a run and all of its child records in one transaction.

        Each question value may contain ``options`` and ``citations`` collections;
        those collections are consumed here rather than persisted as columns.
        """
        try:
            run = self.create_generation_run(**run_values)
            questions: list[AssessmentQuestion] = []
            for values in question_values:
                child_values = dict(values)
                options = child_values.pop("options", [])
                citations = child_values.pop("citations", [])
                question = self.create_questions(run, [child_values])[0]
                self.create_options(question, options)
                self.create_citations(question, citations)
                questions.append(question)
            run_id = run.id
            self.db.commit()
            persisted_run = self.get_generation_run(run_id)
            if persisted_run is None:
                raise RuntimeError(
                    "The persisted generation run could not be reloaded")
            return persisted_run, list(persisted_run.questions)
        except Exception:
            self.db.rollback()
            raise

    def get_question(self, question_id: str) -> AssessmentQuestion | None:
        statement = (
            select(AssessmentQuestion)
            .options(
                selectinload(AssessmentQuestion.options),
                selectinload(AssessmentQuestion.citations),
            )
            .where(AssessmentQuestion.id == question_id)
        )
        question = self.db.scalar(statement)
        if question is not None:
            self.db.expunge_all()
        return question

    def list_questions(
        self,
        *,
        document_id: str | None = None,
        question_type: str | None = None,
        difficulty: str | None = None,
        bloom_level: str | None = None,
        topic: str | None = None,
        skill: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AssessmentQuestion]:
        statement = (
            select(AssessmentQuestion)
            .options(
                selectinload(AssessmentQuestion.options),
                selectinload(AssessmentQuestion.citations),
            )
            .order_by(AssessmentQuestion.created_at, AssessmentQuestion.id)
            .limit(limit)
            .offset(offset)
        )
        filters = {
            AssessmentQuestion.document_id: document_id,
            AssessmentQuestion.question_type: question_type,
            AssessmentQuestion.difficulty: difficulty,
            AssessmentQuestion.bloom_level: bloom_level,
            AssessmentQuestion.topic: topic,
            AssessmentQuestion.skill: skill,
        }
        for column, value in filters.items():
            if value is not None:
                statement = statement.where(column == value)
        questions = list(self.db.scalars(statement).all())
        self.db.expunge_all()
        return questions

    def get_generation_run(self, run_id: str) -> AssessmentGenerationRun | None:
        statement = (
            select(AssessmentGenerationRun)
            .options(
                selectinload(AssessmentGenerationRun.questions)
                .selectinload(AssessmentQuestion.options),
                selectinload(AssessmentGenerationRun.questions)
                .selectinload(AssessmentQuestion.citations),
            )
            .where(AssessmentGenerationRun.id == run_id)
        )
        run = self.db.scalar(statement)
        if run is not None:
            self.db.expunge_all()
        return run
