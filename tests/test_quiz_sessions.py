"""M4A tests: quiz session creation, retrieval, and learner-safe question delivery."""
from uuid import uuid4

import pytest

from app.domain.assessment import BloomLevel, Difficulty, QuestionType
from app.models.document import Document, DocumentChunk
from app.repositories.assessments import AssessmentRepository
from app.repositories.learner import LearnerRepository
from app.schemas.learner import QuizSessionCreateRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_question(session_factory) -> str:
    """Persist a minimal assessment question and return its ID."""
    document_id = str(uuid4())
    with session_factory() as session:
        document = Document(
            id=document_id,
            original_filename="test.pdf",
            storage_path="data/uploads/test.pdf",
            content_type="application/pdf",
            size_bytes=10,
            status="processed",
        )
        session.add(document)
        session.commit()
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=0,
            chunk_text="source text",
        )
        session.add(chunk)
        session.commit()
        chunk_id = chunk.id

    with session_factory() as session:
        repo = AssessmentRepository(session)
        _, questions = repo.create_generation_batch(
            {
                "request_fingerprint": f"fp-{uuid4()}",
                "document_id": document_id,
                "query": "topic",
                "question_type": QuestionType.MCQ,
                "requested_count": 1,
                "retrieval_top_k": 5,
                "embedding_model_name": "em-v1",
                "reranker_model_name": "rr-v1",
                "llm_model_name": "llm-v1",
                "prompt_version": "mcq-v1",
                "status": "completed",
            },
            [
                {
                    "document_id": document_id,
                    "question_type": QuestionType.MCQ,
                    "stem": "What is true?",
                    "explanation": "Because the source says so.",
                    "difficulty": Difficulty.EASY,
                    "bloom_level": BloomLevel.UNDERSTAND,
                    "topic": "topic-a",
                    "skill": "skill-a",
                    "correct_option_key": "A",
                    "options": [
                        {"option_key": "A", "option_text": "True", "display_order": 0},
                        {"option_key": "B", "option_text": "False", "display_order": 1},
                        {"option_key": "C", "option_text": "Maybe", "display_order": 2},
                        {"option_key": "D", "option_text": "Never", "display_order": 3},
                    ],
                    "citations": [
                        {
                            "chunk_id": chunk_id,
                            "citation_order": 0,
                            "retrieval_rank": 1,
                            "reranker_score": 0.9,
                            "hybrid_score": 0.8,
                            "chunk_text_snapshot": "source text",
                            "page_number_snapshot": 1,
                        }
                    ],
                }
            ],
        )
    return questions[0].id


# ---------------------------------------------------------------------------
# Schema-level (no I/O)
# ---------------------------------------------------------------------------

def test_quiz_session_create_request_validates_learner_id() -> None:
    with pytest.raises(Exception):
        QuizSessionCreateRequest(learner_id="   ", question_ids=["abc"])


def test_quiz_session_create_request_rejects_empty_list() -> None:
    with pytest.raises(Exception):
        QuizSessionCreateRequest(learner_id="learner-1", question_ids=[])


def test_quiz_session_create_request_rejects_over_50_ids() -> None:
    with pytest.raises(Exception):
        QuizSessionCreateRequest(
            learner_id="learner-1",
            question_ids=[str(uuid4()) for _ in range(51)],
        )


def test_quiz_session_create_request_rejects_blank_id_entries() -> None:
    with pytest.raises(Exception):
        QuizSessionCreateRequest(learner_id="learner-1", question_ids=["  "])


# ---------------------------------------------------------------------------
# Repository-level
# ---------------------------------------------------------------------------

def test_create_and_get_session(session_factory) -> None:
    question_id = make_question(session_factory)
    with session_factory() as session:
        repo = LearnerRepository(session)
        created = repo.create_session(
            learner_id="learner-1",
            question_ids=[question_id],
        )
        assert created.id is not None
        assert created.status == "active"
        assert created.question_ids == [question_id]

        fetched = repo.get_session(created.id)
        assert fetched is not None
        assert fetched.learner_id == "learner-1"


def test_get_session_returns_none_for_unknown_id(session_factory) -> None:
    with session_factory() as session:
        repo = LearnerRepository(session)
        assert repo.get_session("nonexistent") is None


# ---------------------------------------------------------------------------
# API-level
# ---------------------------------------------------------------------------

def test_create_quiz_session_returns_201(client, session_factory) -> None:
    question_id = make_question(session_factory)
    response = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-1", "question_ids": [question_id]},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["learner_id"] == "learner-1"
    assert data["question_ids"] == [question_id]
    assert data["status"] == "active"
    assert "id" in data
    assert "created_at" in data


def test_create_quiz_session_with_unknown_question_ids_returns_404(client) -> None:
    response = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-1", "question_ids": [str(uuid4())]},
    )
    assert response.status_code == 404


def test_create_quiz_session_with_empty_list_returns_422(client) -> None:
    response = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-1", "question_ids": []},
    )
    assert response.status_code == 422


def test_create_quiz_session_with_too_many_ids_returns_422(client) -> None:
    response = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-1", "question_ids": [str(uuid4()) for _ in range(51)]},
    )
    assert response.status_code == 422


def test_create_quiz_session_with_blank_learner_id_returns_422(client) -> None:
    response = client.post(
        "/quiz-sessions",
        json={"learner_id": "   ", "question_ids": [str(uuid4())]},
    )
    assert response.status_code == 422


def test_get_quiz_session_returns_session(client, session_factory) -> None:
    question_id = make_question(session_factory)
    create_response = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-2", "question_ids": [question_id]},
    )
    session_id = create_response.json()["id"]

    get_response = client.get(f"/quiz-sessions/{session_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == session_id
    assert get_response.json()["learner_id"] == "learner-2"


def test_get_quiz_session_unknown_id_returns_404(client) -> None:
    response = client.get("/quiz-sessions/does-not-exist")
    assert response.status_code == 404


def test_get_quiz_session_questions_excludes_correct_option_key(
    client, session_factory
) -> None:
    question_id = make_question(session_factory)
    create_resp = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-3", "question_ids": [question_id]},
    )
    session_id = create_resp.json()["id"]

    questions_resp = client.get(f"/quiz-sessions/{session_id}/questions")
    assert questions_resp.status_code == 200
    data = questions_resp.json()
    assert data["session_id"] == session_id
    assert len(data["questions"]) == 1
    q = data["questions"][0]
    assert q["id"] == question_id
    # Must NOT expose the correct answer before submission
    assert "correct_option_key" not in q


def test_get_quiz_session_questions_excludes_citations(
    client, session_factory
) -> None:
    question_id = make_question(session_factory)
    create_resp = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-4", "question_ids": [question_id]},
    )
    session_id = create_resp.json()["id"]

    questions_resp = client.get(f"/quiz-sessions/{session_id}/questions")
    assert questions_resp.status_code == 200
    q = questions_resp.json()["questions"][0]
    assert "citations" not in q


def test_get_quiz_session_questions_returns_options(
    client, session_factory
) -> None:
    question_id = make_question(session_factory)
    create_resp = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-5", "question_ids": [question_id]},
    )
    session_id = create_resp.json()["id"]

    questions_resp = client.get(f"/quiz-sessions/{session_id}/questions")
    q = questions_resp.json()["questions"][0]
    assert len(q["options"]) == 4
    option_keys = [o["option_key"] for o in q["options"]]
    assert "A" in option_keys


def test_get_quiz_session_questions_preserves_order(
    client, session_factory
) -> None:
    q1 = make_question(session_factory)
    q2 = make_question(session_factory)
    create_resp = client.post(
        "/quiz-sessions",
        json={"learner_id": "learner-6", "question_ids": [q1, q2]},
    )
    session_id = create_resp.json()["id"]

    questions_resp = client.get(f"/quiz-sessions/{session_id}/questions")
    ids_returned = [q["id"] for q in questions_resp.json()["questions"]]
    assert ids_returned == [q1, q2]


def test_get_quiz_session_questions_unknown_session_returns_404(client) -> None:
    response = client.get("/quiz-sessions/no-such-session/questions")
    assert response.status_code == 404
