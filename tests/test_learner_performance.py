"""M4C tests: performance accumulation and weak-topic detection."""
from uuid import uuid4

import pytest

from app.domain.assessment import (
    AreaPerformance,
    BloomLevel,
    Difficulty,
    QuestionType,
    WEAK_TOPIC_ACCURACY_THRESHOLD,
    WEAK_TOPIC_MIN_ATTEMPTS,
    compute_accuracy,
    detect_weak_areas,
)
from app.models.document import Document, DocumentChunk
from app.repositories.assessments import AssessmentRepository
from app.repositories.learner import LearnerRepository


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def make_question(
    session_factory,
    topic: str = "topic-a",
    skill: str = "skill-a",
    correct_option_key: str = "A",
) -> str:
    """Persist a minimal MCQ question with given topic/skill and return its ID."""
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
                    "stem": f"Q about {topic}?",
                    "explanation": "Because the source says so.",
                    "difficulty": Difficulty.EASY,
                    "bloom_level": BloomLevel.UNDERSTAND,
                    "topic": topic,
                    "skill": skill,
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


def submit(session_factory, learner_id: str, question_id: str, is_correct: bool) -> None:
    """Submit a single answer for a learner via the repository."""
    with session_factory() as session:
        session_repo = LearnerRepository(session)
        quiz_session = session_repo.create_session(
            learner_id=learner_id,
            question_ids=[question_id],
        )
        session_id = quiz_session.id
    with session_factory() as session:
        repo = LearnerRepository(session)
        repo.submit_answer(
            session_id=session_id,
            question_id=question_id,
            learner_id=learner_id,
            submitted_option_key="A" if is_correct else "B",
            is_correct=is_correct,
        )


# ---------------------------------------------------------------------------
# Domain-level: compute_accuracy (no I/O)
# ---------------------------------------------------------------------------

def test_compute_accuracy_perfect() -> None:
    assert compute_accuracy(4, 4) == 1.0


def test_compute_accuracy_zero_correct() -> None:
    assert compute_accuracy(4, 0) == 0.0


def test_compute_accuracy_zero_attempts() -> None:
    assert compute_accuracy(0, 0) == 0.0


def test_compute_accuracy_rounded() -> None:
    # 1/3 = 0.3333...
    assert compute_accuracy(3, 1) == 0.3333


# ---------------------------------------------------------------------------
# Domain-level: detect_weak_areas (no I/O)
# ---------------------------------------------------------------------------

def _area(label: str, attempts: int, correct: int) -> AreaPerformance:
    return AreaPerformance(
        label=label,
        attempts=attempts,
        correct=correct,
        accuracy=compute_accuracy(attempts, correct),
    )


def test_detect_weak_areas_threshold_constants() -> None:
    assert WEAK_TOPIC_MIN_ATTEMPTS == 3
    assert WEAK_TOPIC_ACCURACY_THRESHOLD == 0.60


def test_detect_weak_areas_below_min_attempts_not_weak() -> None:
    # 2 attempts < minimum of 3
    areas = [_area("math", 2, 0)]
    assert detect_weak_areas(areas) == []


def test_detect_weak_areas_exactly_min_attempts_and_at_threshold_is_weak() -> None:
    # accuracy = 3/5 = 0.6 exactly — boundary case: IS weak
    areas = [_area("math", 5, 3)]
    result = detect_weak_areas(areas)
    assert len(result) == 1
    assert result[0].label == "math"
    assert result[0].accuracy == 0.6


def test_detect_weak_areas_above_threshold_not_weak() -> None:
    # accuracy = 4/5 = 0.8 > 0.60
    areas = [_area("math", 5, 4)]
    assert detect_weak_areas(areas) == []


def test_detect_weak_areas_exactly_min_attempts_and_below_threshold() -> None:
    # 3 attempts, 1 correct = 0.3333 accuracy
    areas = [_area("science", 3, 1)]
    result = detect_weak_areas(areas)
    assert len(result) == 1
    assert result[0].label == "science"


def test_detect_weak_areas_sorted_by_accuracy_then_label() -> None:
    areas = [
        _area("zebra", 3, 0),   # accuracy 0.0
        _area("apple", 3, 0),   # accuracy 0.0 — same, label-sorted
        _area("mango", 3, 1),   # accuracy 0.3333
    ]
    result = detect_weak_areas(areas)
    assert [w.label for w in result] == ["apple", "zebra", "mango"]


def test_detect_weak_areas_mixed_weak_and_strong() -> None:
    areas = [
        _area("topic-a", 3, 2),   # 0.6667 — NOT weak (above threshold)
        _area("topic-b", 3, 1),   # 0.3333 — weak
        _area("topic-c", 2, 0),   # below min_attempts — not weak
    ]
    result = detect_weak_areas(areas)
    assert len(result) == 1
    assert result[0].label == "topic-b"


def test_detect_weak_areas_empty_input() -> None:
    assert detect_weak_areas([]) == []


# ---------------------------------------------------------------------------
# Repository-level
# ---------------------------------------------------------------------------

def test_get_all_submissions_for_learner_returns_correct_rows(session_factory) -> None:
    q1 = make_question(session_factory)
    q2 = make_question(session_factory)
    submit(session_factory, "learner-p1", q1, is_correct=True)
    submit(session_factory, "learner-p1", q2, is_correct=False)

    with session_factory() as session:
        repo = LearnerRepository(session)
        subs = repo.get_all_submissions_for_learner("learner-p1")
    assert len(subs) == 2
    assert all(s.learner_id == "learner-p1" for s in subs)


def test_get_all_submissions_learner_isolation(session_factory) -> None:
    q = make_question(session_factory)
    submit(session_factory, "learner-iso-a", q, is_correct=True)

    with session_factory() as session:
        repo = LearnerRepository(session)
        assert repo.get_all_submissions_for_learner("learner-iso-b") == []


def test_get_all_submissions_returns_empty_for_unknown_learner(session_factory) -> None:
    with session_factory() as session:
        repo = LearnerRepository(session)
        assert repo.get_all_submissions_for_learner("nobody") == []


def test_get_topic_performance_aggregates_correctly(session_factory) -> None:
    q1 = make_question(session_factory, topic="algebra", skill="sk-a")
    q2 = make_question(session_factory, topic="algebra", skill="sk-a")
    q3 = make_question(session_factory, topic="geometry", skill="sk-b")
    submit(session_factory, "learner-tp1", q1, is_correct=True)
    submit(session_factory, "learner-tp1", q2, is_correct=False)
    submit(session_factory, "learner-tp1", q3, is_correct=True)

    with session_factory() as session:
        repo = LearnerRepository(session)
        perfs = repo.get_topic_performance_for_learner("learner-tp1")

    by_label = {p.label: p for p in perfs}
    assert by_label["algebra"].attempts == 2
    assert by_label["algebra"].correct == 1
    assert by_label["geometry"].attempts == 1
    assert by_label["geometry"].correct == 1


def test_get_skill_performance_aggregates_correctly(session_factory) -> None:
    q1 = make_question(session_factory, topic="t-a", skill="reading")
    q2 = make_question(session_factory, topic="t-b", skill="reading")
    q3 = make_question(session_factory, topic="t-c", skill="writing")
    submit(session_factory, "learner-sp1", q1, is_correct=False)
    submit(session_factory, "learner-sp1", q2, is_correct=False)
    submit(session_factory, "learner-sp1", q3, is_correct=True)

    with session_factory() as session:
        repo = LearnerRepository(session)
        perfs = repo.get_skill_performance_for_learner("learner-sp1")

    by_label = {p.label: p for p in perfs}
    assert by_label["reading"].attempts == 2
    assert by_label["reading"].correct == 0
    assert by_label["writing"].attempts == 1
    assert by_label["writing"].correct == 1


def test_multiple_learners_are_isolated_in_topic_performance(session_factory) -> None:
    q = make_question(session_factory, topic="physics", skill="sk-x")
    submit(session_factory, "learner-la", q, is_correct=True)
    submit(session_factory, "learner-lb", q, is_correct=False)

    with session_factory() as session:
        repo = LearnerRepository(session)
        la_perfs = repo.get_topic_performance_for_learner("learner-la")
        lb_perfs = repo.get_topic_performance_for_learner("learner-lb")

    assert la_perfs[0].correct == 1
    assert lb_perfs[0].correct == 0


def test_unanswered_questions_do_not_appear_in_performance(session_factory) -> None:
    # A question that exists but has never been submitted should not appear.
    _unused_question = make_question(session_factory, topic="unused-topic", skill="unused-skill")

    with session_factory() as session:
        repo = LearnerRepository(session)
        perfs = repo.get_topic_performance_for_learner("learner-nosubmit")
    assert perfs == []


# ---------------------------------------------------------------------------
# API-level
# ---------------------------------------------------------------------------

def test_performance_endpoint_empty_learner(client) -> None:
    resp = client.get("/learners/unknown-learner/performance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["learner_id"] == "unknown-learner"
    assert data["total_attempts"] == 0
    assert data["total_correct"] == 0
    assert data["overall_accuracy"] == 0.0
    assert data["by_topic"] == []
    assert data["by_skill"] == []


def test_performance_endpoint_overall_accuracy(client, session_factory) -> None:
    q1 = make_question(session_factory, topic="t-x", skill="s-x")
    q2 = make_question(session_factory, topic="t-x", skill="s-x")
    q3 = make_question(session_factory, topic="t-x", skill="s-x")
    submit(session_factory, "learner-api-acc", q1, is_correct=True)
    submit(session_factory, "learner-api-acc", q2, is_correct=True)
    submit(session_factory, "learner-api-acc", q3, is_correct=False)

    resp = client.get("/learners/learner-api-acc/performance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_attempts"] == 3
    assert data["total_correct"] == 2
    assert data["overall_accuracy"] == round(2 / 3, 4)


def test_performance_endpoint_by_topic(client, session_factory) -> None:
    q1 = make_question(session_factory, topic="topic-api-1", skill="sk-api")
    q2 = make_question(session_factory, topic="topic-api-2", skill="sk-api")
    submit(session_factory, "learner-api-tp", q1, is_correct=True)
    submit(session_factory, "learner-api-tp", q2, is_correct=False)

    resp = client.get("/learners/learner-api-tp/performance")
    by_topic = {item["label"]: item for item in resp.json()["by_topic"]}
    assert by_topic["topic-api-1"]["correct"] == 1
    assert by_topic["topic-api-2"]["correct"] == 0


def test_performance_endpoint_by_skill(client, session_factory) -> None:
    q1 = make_question(session_factory, topic="t-api-sk", skill="skill-api-1")
    q2 = make_question(session_factory, topic="t-api-sk", skill="skill-api-2")
    submit(session_factory, "learner-api-sk", q1, is_correct=False)
    submit(session_factory, "learner-api-sk", q2, is_correct=True)

    resp = client.get("/learners/learner-api-sk/performance")
    by_skill = {item["label"]: item for item in resp.json()["by_skill"]}
    assert by_skill["skill-api-1"]["correct"] == 0
    assert by_skill["skill-api-2"]["correct"] == 1


def test_weak_topics_endpoint_empty_learner(client) -> None:
    resp = client.get("/learners/nobody/weak-topics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["learner_id"] == "nobody"
    assert data["weak_topics"] == []
    assert data["weak_skills"] == []
    assert data["min_attempts_threshold"] == WEAK_TOPIC_MIN_ATTEMPTS
    assert data["accuracy_threshold"] == WEAK_TOPIC_ACCURACY_THRESHOLD


def test_weak_topics_endpoint_below_min_attempts_not_flagged(
    client, session_factory
) -> None:
    # Only 2 submissions — below the minimum of 3
    q1 = make_question(session_factory, topic="rare-topic", skill="rare-skill")
    q2 = make_question(session_factory, topic="rare-topic", skill="rare-skill")
    submit(session_factory, "learner-rare", q1, is_correct=False)
    submit(session_factory, "learner-rare", q2, is_correct=False)

    resp = client.get("/learners/learner-rare/weak-topics")
    assert resp.json()["weak_topics"] == []
    assert resp.json()["weak_skills"] == []


def test_weak_topics_endpoint_accuracy_exactly_threshold_is_weak(
    client, session_factory
) -> None:
    # 3 correct out of 5 = 0.6 exactly — boundary, must be flagged
    learner = "learner-boundary"
    topic = "boundary-topic"
    skill = "boundary-skill"
    for i in range(5):
        q = make_question(session_factory, topic=topic, skill=skill)
        submit(session_factory, learner, q, is_correct=(i < 3))

    resp = client.get(f"/learners/{learner}/weak-topics")
    data = resp.json()
    assert len(data["weak_topics"]) == 1
    assert data["weak_topics"][0]["label"] == topic
    assert data["weak_topics"][0]["accuracy"] == 0.6
    assert len(data["weak_skills"]) == 1
    assert data["weak_skills"][0]["label"] == skill


def test_weak_topics_endpoint_above_threshold_not_flagged(
    client, session_factory
) -> None:
    # 4 correct out of 5 = 0.8 > 0.60
    learner = "learner-strong"
    for _ in range(4):
        q = make_question(session_factory, topic="strong-topic", skill="strong-skill")
        submit(session_factory, learner, q, is_correct=True)
    q = make_question(session_factory, topic="strong-topic", skill="strong-skill")
    submit(session_factory, learner, q, is_correct=False)

    resp = client.get(f"/learners/{learner}/weak-topics")
    assert resp.json()["weak_topics"] == []


def test_weak_topics_endpoint_multiple_learners_isolated(
    client, session_factory
) -> None:
    # learner-w1 is weak, learner-w2 is strong on same topic
    topic = "shared-topic"
    skill = "shared-skill"
    for _ in range(3):
        q = make_question(session_factory, topic=topic, skill=skill)
        submit(session_factory, "learner-w1", q, is_correct=False)  # 0% accuracy
    for _ in range(3):
        q = make_question(session_factory, topic=topic, skill=skill)
        submit(session_factory, "learner-w2", q, is_correct=True)   # 100% accuracy

    resp_w1 = client.get("/learners/learner-w1/weak-topics")
    resp_w2 = client.get("/learners/learner-w2/weak-topics")

    assert len(resp_w1.json()["weak_topics"]) == 1
    assert resp_w2.json()["weak_topics"] == []


def test_performance_learners_are_isolated_via_api(client, session_factory) -> None:
    q = make_question(session_factory, topic="iso-topic", skill="iso-skill")
    submit(session_factory, "learner-iso-x", q, is_correct=True)

    resp = client.get("/learners/learner-iso-y/performance")
    assert resp.json()["total_attempts"] == 0
