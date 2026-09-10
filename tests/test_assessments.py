from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from app.domain.assessment import BloomLevel, Difficulty, QuestionType
from app.models.assessment import AssessmentQuestionOption
from app.models.document import Document, DocumentChunk
from app.repositories.assessments import AssessmentRepository
from app.schemas.assessments import QuestionGenerationRequest


def make_document(session_factory) -> tuple[str, str]:
    document_id = str(uuid4())
    with session_factory() as session:
        document = Document(
            id=document_id,
            original_filename="guide.pdf",
            storage_path="data/uploads/guide.pdf",
            content_type="application/pdf",
            size_bytes=10,
            status="processed",
        )
        session.add(document)
        session.commit()
        chunk = DocumentChunk(document_id=document_id,
                              chunk_index=0, chunk_text="source text")
        session.add(chunk)
        session.commit()
        return document_id, chunk.id


def run_values(document_id: str) -> dict[str, object]:
    return {
        "request_fingerprint": "fingerprint-1",
        "document_id": document_id,
        "query": "topic",
        "question_type": QuestionType.MCQ,
        "requested_count": 1,
        "retrieval_top_k": 5,
        "embedding_model_name": "embedding-v1",
        "reranker_model_name": "reranker-v1",
        "llm_model_name": "llm-v1",
        "prompt_version": "mcq-v1",
        "status": "completed",
    }


def question_values(document_id: str, chunk_id: str) -> list[dict[str, object]]:
    return [{
        "document_id": document_id,
        "question_type": QuestionType.MCQ,
        "stem": "What is true?",
        "explanation": "The source supports this answer.",
        "difficulty": Difficulty.EASY,
        "bloom_level": BloomLevel.UNDERSTAND,
        "topic": "topic",
        "skill": "skill",
        "correct_option_key": "A",
        "options": [
            {"option_key": "A", "option_text": "True", "display_order": 0},
            {"option_key": "B", "option_text": "False", "display_order": 1},
        ],
        "citations": [{
            "chunk_id": chunk_id,
            "citation_order": 0,
            "retrieval_rank": 1,
            "reranker_score": 0.9,
            "hybrid_score": 0.8,
            "chunk_text_snapshot": "source text",
            "page_number_snapshot": 1,
        }],
    }]


def test_generation_batch_persists_relationships(session_factory) -> None:
    document_id, chunk_id = make_document(session_factory)
    with session_factory() as session:
        repository = AssessmentRepository(session)
        run, questions = repository.create_generation_batch(
            run_values(document_id), question_values(document_id, chunk_id)
        )

        question = repository.get_question(questions[0].id)
        loaded_run = repository.get_generation_run(run.id)

    assert question is not None
    assert question.options[0].option_key == "A"
    assert question.citations[0].chunk_id == chunk_id
    assert loaded_run is not None
    assert loaded_run.questions[0].id == question.id


def test_generation_batch_rolls_back_all_children_on_constraint_failure(session_factory) -> None:
    document_id, chunk_id = make_document(session_factory)
    values = question_values(document_id, chunk_id)
    values[0]["options"] = [
        {"option_key": "A", "option_text": "One", "display_order": 0},
        {"option_key": "A", "option_text": "Duplicate", "display_order": 1},
    ]
    with session_factory() as session:
        repository = AssessmentRepository(session)
        with pytest.raises(IntegrityError):
            repository.create_generation_batch(run_values(document_id), values)
        assert session.query(AssessmentQuestionOption).count() == 0
        assert repository.get_generation_run("missing") is None


def test_question_option_database_constraints_are_scoped_to_question(session_factory) -> None:
    document_id, chunk_id = make_document(session_factory)
    with session_factory() as session:
        repository = AssessmentRepository(session)
        run, questions = repository.create_generation_batch(
            run_values(document_id), question_values(document_id, chunk_id)
        )
        with pytest.raises(IntegrityError):
            repository.create_options(
                questions[0], [
                    {"option_key": "A", "option_text": "Again", "display_order": 2}]
            )
            session.commit()


def test_generation_request_has_strict_enums_and_count_bounds() -> None:
    request = QuestionGenerationRequest(
        query="  database indexing  ",
        count=20,
        question_type="mcq",
        difficulty="hard",
        bloom_level="analyze",
    )
    assert request.query == "database indexing"
    assert request.question_type is QuestionType.MCQ
    assert request.difficulty is Difficulty.HARD
    assert request.bloom_level is BloomLevel.ANALYZE

    with pytest.raises(ValidationError):
        QuestionGenerationRequest(query="topic", count=21)
    with pytest.raises(ValidationError):
        QuestionGenerationRequest(query="topic", question_type="essay")
    with pytest.raises(ValidationError):
        QuestionGenerationRequest(query="topic", query_extra="unexpected")
