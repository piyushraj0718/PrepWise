"""M4B tests: answer submission, deterministic MCQ scoring, and session completion."""
from uuid import uuid4

import pytest

from app.domain.assessment import BloomLevel, Difficulty, QuestionType, score_mcq_answer
from app.models.document import Document, DocumentChunk
from app.repositories.assessments import AssessmentRepository
from app.repositories.learner import DuplicateSubmissionError, LearnerRepository
from app.schemas.learner import AnswerSubmissionRequest


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_quiz_sessions.py make_question)
# ---------------------------------------------------------------------------

def make_question(session_factory, correct_option_key: str = "A") -> str:
    """Persist a minimal MCQ question and return its ID."""
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
                    "correct_option_key": correct_option_key,
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


def make_session(client, question_ids: list[str], learner_id: str = "learner-1") -> str:
    """Create a quiz session via the API and return its ID."""
    resp = client.post(
        "/quiz-sessions",
        json={"learner_id": learner_id, "question_ids": question_ids},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Domain-level: score_mcq_answer (no I/O)
# ---------------------------------------------------------------------------

def test_score_mcq_answer_correct() -> None:
    assert score_mcq_answer("A", "A") is True


def test_score_mcq_answer_incorrect() -> None:
    assert score_mcq_answer("B", "A") is False


def test_score_mcq_answer_case_sensitive() -> None:
    # Keys are stored uppercase; lowercase must not match.
    assert score_mcq_answer("a", "A") is False


def test_score_mcq_answer_all_keys() -> None:
    for key in ("A", "B", "C", "D"):
        assert score_mcq_answer(key, key) is True
        others = [k for k in ("A", "B", "C", "D") if k != key]
        for wrong in others:
            assert score_mcq_answer(wrong, key) is False


# ---------------------------------------------------------------------------
# Schema-level: AnswerSubmissionRequest (no I/O)
# ---------------------------------------------------------------------------

def test_answer_submission_request_rejects_blank_question_id() -> None:
    with pytest.raises(Exception):
        AnswerSubmissionRequest(question_id="  ", submitted_option_key="A")


def test_answer_submission_request_rejects_blank_option_key() -> None:
    with pytest.raises(Exception):
        AnswerSubmissionRequest(question_id="some-id", submitted_option_key="  ")


def test_answer_submission_request_rejects_option_key_too_long() -> None:
    with pytest.raises(Exception):
        AnswerSubmissionRequest(
            question_id="some-id", submitted_option_key="A" * 21
        )


# ---------------------------------------------------------------------------
# Repository-level
# ---------------------------------------------------------------------------

def test_submit_answer_persists_row(session_factory) -> None:
    question_id = make_question(session_factory)
    session_id = str(uuid4())
    # Create the quiz session directly so we have a valid session_id FK.
    with session_factory() as session:
        repo = LearnerRepository(session)
        quiz_session = repo.create_session(
            learner_id="learner-r1",
            question_ids=[question_id],
        )
        session_id = quiz_session.id

    with session_factory() as session:
        repo = LearnerRepository(session)
        submission = repo.submit_answer(
            session_id=session_id,
            question_id=question_id,
            learner_id="learner-r1",
            submitted_option_key="A",
            is_correct=True,
        )
        assert submission.id is not None
        assert submission.is_correct is True
        assert submission.submitted_option_key == "A"


def test_submit_answer_duplicate_raises(session_factory) -> None:
    question_id = make_question(session_factory)
    with session_factory() as session:
        repo = LearnerRepository(session)
        quiz_session = repo.create_session(
            learner_id="learner-r2",
            question_ids=[question_id],
        )
        session_id = quiz_session.id

    with session_factory() as session:
        repo = LearnerRepository(session)
        repo.submit_answer(
            session_id=session_id,
            question_id=question_id,
            learner_id="learner-r2",
            submitted_option_key="A",
            is_correct=True,
        )

    with session_factory() as session:
        repo = LearnerRepository(session)
        with pytest.raises(DuplicateSubmissionError):
            repo.submit_answer(
                session_id=session_id,
                question_id=question_id,
                learner_id="learner-r2",
                submitted_option_key="B",
                is_correct=False,
            )


def test_get_submissions_for_session_returns_all(session_factory) -> None:
    q1 = make_question(session_factory)
    q2 = make_question(session_factory)
    with session_factory() as session:
        repo = LearnerRepository(session)
        quiz_session = repo.create_session(
            learner_id="learner-r3",
            question_ids=[q1, q2],
        )
        session_id = quiz_session.id

    with session_factory() as session:
        repo = LearnerRepository(session)
        repo.submit_answer(
            session_id=session_id,
            question_id=q1,
            learner_id="learner-r3",
            submitted_option_key="A",
            is_correct=True,
        )
        repo.submit_answer(
            session_id=session_id,
            question_id=q2,
            learner_id="learner-r3",
            submitted_option_key="B",
            is_correct=False,
        )

    with session_factory() as session:
        repo = LearnerRepository(session)
        subs = repo.get_submissions_for_session(session_id)
        assert len(subs) == 2


def test_get_submission_returns_correct_row(session_factory) -> None:
    question_id = make_question(session_factory)
    with session_factory() as session:
        repo = LearnerRepository(session)
        quiz_session = repo.create_session(
            learner_id="learner-r4",
            question_ids=[question_id],
        )
        session_id = quiz_session.id

    with session_factory() as session:
        repo = LearnerRepository(session)
        repo.submit_answer(
            session_id=session_id,
            question_id=question_id,
            learner_id="learner-r4",
            submitted_option_key="C",
            is_correct=False,
        )

    with session_factory() as session:
        repo = LearnerRepository(session)
        sub = repo.get_submission(session_id, question_id)
        assert sub is not None
        assert sub.submitted_option_key == "C"
        assert sub.is_correct is False


def test_get_submission_returns_none_for_unknown(session_factory) -> None:
    with session_factory() as session:
        repo = LearnerRepository(session)
        assert repo.get_submission("no-session", "no-question") is None


def test_complete_session_sets_status_and_timestamp(session_factory) -> None:
    question_id = make_question(session_factory)
    with session_factory() as session:
        repo = LearnerRepository(session)
        quiz_session = repo.create_session(
            learner_id="learner-r5",
            question_ids=[question_id],
        )
        session_id = quiz_session.id

    with session_factory() as session:
        repo = LearnerRepository(session)
        completed = repo.complete_session(session_id)
        assert completed.status == "completed"
        assert completed.completed_at is not None


# ---------------------------------------------------------------------------
# API-level
# ---------------------------------------------------------------------------

def test_submit_correct_answer_returns_201_with_is_correct_true(
    client, session_factory
) -> None:
    question_id = make_question(session_factory, correct_option_key="A")
    session_id = make_session(client, [question_id])

    resp = client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": question_id, "submitted_option_key": "A"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["is_correct"] is True
    assert data["submitted_option_key"] == "A"
    assert data["session_id"] == session_id
    assert data["question_id"] == question_id
    assert "submitted_at" in data


def test_submit_incorrect_answer_returns_201_with_is_correct_false(
    client, session_factory
) -> None:
    question_id = make_question(session_factory, correct_option_key="A")
    session_id = make_session(client, [question_id])

    resp = client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": question_id, "submitted_option_key": "B"},
    )
    assert resp.status_code == 201
    assert resp.json()["is_correct"] is False


def test_submit_answer_unknown_session_returns_404(client) -> None:
    resp = client.post(
        "/quiz-sessions/no-such-session/answers",
        json={"question_id": str(uuid4()), "submitted_option_key": "A"},
    )
    assert resp.status_code == 404


def test_submit_answer_question_not_in_session_returns_404(
    client, session_factory
) -> None:
    question_id = make_question(session_factory)
    other_question_id = make_question(session_factory)
    session_id = make_session(client, [question_id])

    resp = client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": other_question_id, "submitted_option_key": "A"},
    )
    assert resp.status_code == 404


def test_submit_answer_duplicate_returns_409(client, session_factory) -> None:
    question_id = make_question(session_factory)
    session_id = make_session(client, [question_id])

    client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": question_id, "submitted_option_key": "A"},
    )
    resp = client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": question_id, "submitted_option_key": "B"},
    )
    assert resp.status_code == 409


def test_submit_answer_to_completed_session_returns_422(
    client, session_factory
) -> None:
    question_id = make_question(session_factory)
    session_id = make_session(client, [question_id])
    client.post(f"/quiz-sessions/{session_id}/complete")

    resp = client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": question_id, "submitted_option_key": "A"},
    )
    assert resp.status_code == 422


def test_submit_answer_blank_option_key_returns_422(client, session_factory) -> None:
    question_id = make_question(session_factory)
    session_id = make_session(client, [question_id])

    resp = client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": question_id, "submitted_option_key": "  "},
    )
    assert resp.status_code == 422


def test_get_answer_submission_returns_submission(client, session_factory) -> None:
    question_id = make_question(session_factory, correct_option_key="C")
    session_id = make_session(client, [question_id])
    client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": question_id, "submitted_option_key": "C"},
    )

    resp = client.get(f"/quiz-sessions/{session_id}/answers/{question_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["question_id"] == question_id
    assert data["submitted_option_key"] == "C"
    assert data["is_correct"] is True


def test_get_answer_submission_unknown_returns_404(client, session_factory) -> None:
    question_id = make_question(session_factory)
    session_id = make_session(client, [question_id])

    resp = client.get(f"/quiz-sessions/{session_id}/answers/{question_id}")
    assert resp.status_code == 404


def test_complete_session_returns_score(client, session_factory) -> None:
    q1 = make_question(session_factory, correct_option_key="A")
    q2 = make_question(session_factory, correct_option_key="B")
    session_id = make_session(client, [q1, q2])

    client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": q1, "submitted_option_key": "A"},  # correct
    )
    client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": q2, "submitted_option_key": "C"},  # wrong
    )

    resp = client.post(f"/quiz-sessions/{session_id}/complete")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["total_questions"] == 2
    assert data["correct_count"] == 1
    assert data["score_percent"] == 50.0
    assert len(data["submissions"]) == 2


def test_complete_session_with_no_submissions_scores_zero(
    client, session_factory
) -> None:
    question_id = make_question(session_factory)
    session_id = make_session(client, [question_id])

    resp = client.post(f"/quiz-sessions/{session_id}/complete")
    assert resp.status_code == 200
    data = resp.json()
    assert data["correct_count"] == 0
    assert data["score_percent"] == 0.0


def test_complete_session_updates_session_status(client, session_factory) -> None:
    question_id = make_question(session_factory)
    session_id = make_session(client, [question_id])

    client.post(f"/quiz-sessions/{session_id}/complete")

    get_resp = client.get(f"/quiz-sessions/{session_id}")
    assert get_resp.json()["status"] == "completed"
    assert get_resp.json()["completed_at"] is not None


def test_complete_session_twice_returns_409(client, session_factory) -> None:
    question_id = make_question(session_factory)
    session_id = make_session(client, [question_id])

    client.post(f"/quiz-sessions/{session_id}/complete")
    resp = client.post(f"/quiz-sessions/{session_id}/complete")
    assert resp.status_code == 409


def test_complete_session_unknown_returns_404(client) -> None:
    resp = client.post("/quiz-sessions/no-such-session/complete")
    assert resp.status_code == 404


def test_complete_session_all_correct_scores_100(client, session_factory) -> None:
    q1 = make_question(session_factory, correct_option_key="A")
    q2 = make_question(session_factory, correct_option_key="B")
    session_id = make_session(client, [q1, q2])

    client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": q1, "submitted_option_key": "A"},
    )
    client.post(
        f"/quiz-sessions/{session_id}/answers",
        json={"question_id": q2, "submitted_option_key": "B"},
    )

    resp = client.post(f"/quiz-sessions/{session_id}/complete")
    data = resp.json()
    assert data["correct_count"] == 2
    assert data["score_percent"] == 100.0
