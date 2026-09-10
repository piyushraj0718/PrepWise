"""M4D tests: adaptive question generation using M4C weak-topic data.

All tests use fake providers and deterministic data — no LLM calls, no network.
The existing M3B/M3C quality pipeline is still exercised via the same service.
"""
from uuid import uuid4

import pytest

from app.ai.assessment_prompts import (
    ADAPTIVE_WEAK_AREAS_MAX,
    build_assessment_prompt,
)
from app.domain.assessment import (
    AreaPerformance,
    BloomLevel,
    Difficulty,
    QuestionType,
    compute_accuracy,
    detect_weak_areas,
)
from app.models.document import Document, DocumentChunk
from app.repositories.assessments import AssessmentRepository
from app.repositories.learner import LearnerRepository
from app.schemas.assessments import QuestionGenerationRequest
from app.services.assessment_generation import (
    AssessmentGenerationService,
    AssessmentValidationError,
)
from app.services.reranking import RerankedResult


# ---------------------------------------------------------------------------
# Shared fakes (mirrors test_question_generation.py helpers)
# ---------------------------------------------------------------------------

class FakeReranking:
    def __init__(self, results: list[RerankedResult]) -> None:
        self.results = results

    def search(self, query: str, top_k: int, candidate_k: int | None,
               document_id: str | None) -> list[RerankedResult]:
        return self.results[:top_k]


class FakeAssessmentLLM:
    model_name = "fake-adaptive-llm"

    def __init__(self, output: object) -> None:
        self.output = output
        self.prompts: list[str] = []

    def generate_structured(self, prompt: str, response_schema: dict) -> object:
        self.prompts.append(prompt)
        return self.output


def make_source(session_factory, status: str = "processed") -> tuple[str, str]:
    document_id = str(uuid4())
    with session_factory() as session:
        doc = Document(
            id=document_id,
            original_filename="guide.pdf",
            storage_path="data/uploads/guide.pdf",
            content_type="application/pdf",
            size_bytes=10,
            status=status,
        )
        session.add(doc)
        session.commit()
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=0,
            chunk_text="A database index improves lookup performance.",
            page_number=1,
        )
        session.add(chunk)
        session.commit()
        return document_id, chunk.id


def make_result(chunk_id: str, document_id: str) -> RerankedResult:
    return RerankedResult(
        chunk_id=chunk_id,
        document_id=document_id,
        similarity=0.9,
        bm25_score=0.8,
        hybrid_score=0.7,
        reranker_score=0.6,
        final_rank=1,
        chunk_text="A database index improves lookup performance.",
        chunk_index=0,
        page_number=1,
    )


def valid_question(chunk_id: str, stem: str = "What improves lookup performance?") -> dict:
    return {
        "stem": stem,
        "options": [
            {"option_key": "A", "option_text": "An index"},
            {"option_key": "B", "option_text": "A slower disk"},
            {"option_key": "C", "option_text": "A blank table"},
            {"option_key": "D", "option_text": "A deleted query"},
        ],
        "correct_option_key": "A",
        "explanation": "An index speeds up lookups according to the source.",
        "difficulty": "easy",
        "bloom_level": "understand",
        "topic": "database indexing",
        "skill": "identify indexing benefits",
        "cited_chunk_ids": [chunk_id],
    }


def make_service(session_factory, document_id: str, chunk_id: str, output: object):
    return AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        FakeAssessmentLLM(output),
        "fake-embedding",
        "fake-reranker",
    )


def submit_answer(session_factory, learner_id: str, question_id: str, is_correct: bool) -> None:
    with session_factory() as session:
        repo = LearnerRepository(session)
        quiz_session = repo.create_session(
            learner_id=learner_id, question_ids=[question_id])
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


def make_question_with_topic_skill(
    session_factory, topic: str, skill: str, correct_option_key: str = "A"
) -> str:
    document_id = str(uuid4())
    with session_factory() as session:
        doc = Document(
            id=document_id,
            original_filename="q.pdf",
            storage_path="data/uploads/q.pdf",
            content_type="application/pdf",
            size_bytes=10,
            status="processed",
        )
        session.add(doc)
        session.commit()
        chunk = DocumentChunk(
            document_id=document_id, chunk_index=0, chunk_text="source")
        session.add(chunk)
        session.commit()
        chunk_id = chunk.id
    with session_factory() as session:
        repo = AssessmentRepository(session)
        _, questions = repo.create_generation_batch(
            {
                "request_fingerprint": f"fp-{uuid4()}",
                "document_id": document_id,
                "query": "q",
                "question_type": QuestionType.MCQ,
                "requested_count": 1,
                "retrieval_top_k": 5,
                "embedding_model_name": "em",
                "reranker_model_name": "rr",
                "llm_model_name": "llm",
                "prompt_version": "v1",
                "status": "completed",
            },
            [
                {
                    "document_id": document_id,
                    "question_type": QuestionType.MCQ,
                    "stem": f"Q about {topic}?",
                    "explanation": "x",
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
                            "chunk_text_snapshot": "source",
                            "page_number_snapshot": 1,
                        }
                    ],
                }
            ],
        )
    return questions[0].id


# ---------------------------------------------------------------------------
# Prompt-level tests (no I/O)
# ---------------------------------------------------------------------------

def test_build_assessment_prompt_without_weak_areas_hint_unchanged() -> None:
    """No weak_areas_hint: prompt must be identical to original."""
    prompt = build_assessment_prompt(
        query="database indexing",
        context="[chunk_id=abc]\nsome text",
        count=1,
        difficulty="easy",
        bloom_level="understand",
        skill=None,
        weak_areas_hint=None,
    )
    assert "Learner preference" not in prompt
    assert "database indexing" in prompt


def test_build_assessment_prompt_empty_weak_areas_list_unchanged() -> None:
    """Empty list must not inject hint — same as None."""
    prompt = build_assessment_prompt(
        query="query",
        context="ctx",
        count=1,
        difficulty=None,
        bloom_level=None,
        skill=None,
        weak_areas_hint=[],
    )
    assert "Learner preference" not in prompt


def test_build_assessment_prompt_with_weak_areas_hint_contains_labels() -> None:
    prompt = build_assessment_prompt(
        query="database indexing",
        context="ctx",
        count=1,
        difficulty=None,
        bloom_level=None,
        skill=None,
        weak_areas_hint=["algebra", "geometry"],
    )
    assert "Learner preference" in prompt
    assert "algebra" in prompt
    assert "geometry" in prompt


def test_build_assessment_prompt_original_query_preserved_with_hint() -> None:
    """The original query/topic must still appear in the prompt with weak hints."""
    query = "database indexing"
    prompt = build_assessment_prompt(
        query=query,
        context="ctx",
        count=1,
        difficulty=None,
        bloom_level=None,
        skill=None,
        weak_areas_hint=["some-weak-topic"],
    )
    assert query in prompt
    assert "some-weak-topic" in prompt


def test_build_assessment_prompt_hint_is_labeled_soft() -> None:
    """The hint must be clearly soft/preferential, not a hard override."""
    prompt = build_assessment_prompt(
        query="q",
        context="c",
        count=1,
        difficulty=None,
        bloom_level=None,
        skill=None,
        weak_areas_hint=["weak-area-x"],
    )
    # Soft language must appear
    assert "prefer" in prompt.lower() or "soft" in prompt.lower()
    # Must not claim to override the query
    assert "replace" not in prompt.lower()


def test_adaptive_weak_areas_max_constant() -> None:
    assert ADAPTIVE_WEAK_AREAS_MAX == 5


# ---------------------------------------------------------------------------
# Schema-level (no I/O)
# ---------------------------------------------------------------------------

def test_question_generation_request_without_learner_id_is_valid() -> None:
    req = QuestionGenerationRequest(query="topic")
    assert req.learner_id is None


def test_question_generation_request_with_learner_id_is_valid() -> None:
    req = QuestionGenerationRequest(query="topic", learner_id="learner-x")
    assert req.learner_id == "learner-x"


def test_question_generation_request_blank_learner_id_is_rejected() -> None:
    import pytest as _pytest
    with _pytest.raises(Exception):
        QuestionGenerationRequest(query="topic", learner_id="   ")


# ---------------------------------------------------------------------------
# Service-level: weak_areas_hint flows into the prompt (no I/O)
# ---------------------------------------------------------------------------

def test_generate_without_learner_id_produces_no_hint(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    llm = FakeAssessmentLLM({"questions": [valid_question(chunk_id)]})
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        llm,
        "fake-embedding",
        "fake-reranker",
    )
    service.generate(QuestionGenerationRequest(
        query="database indexing", document_id=document_id))
    assert "Learner preference" not in llm.prompts[0]


def test_generate_with_weak_areas_hint_injects_into_prompt(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    llm = FakeAssessmentLLM({"questions": [valid_question(chunk_id)]})
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        llm,
        "fake-embedding",
        "fake-reranker",
    )
    service.generate(
        QuestionGenerationRequest(query="database indexing", document_id=document_id),
        weak_areas_hint=["algebra", "geometry"],
    )
    assert "Learner preference" in llm.prompts[0]
    assert "algebra" in llm.prompts[0]
    assert "geometry" in llm.prompts[0]


def test_generate_original_query_preserved_in_adaptive_prompt(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    llm = FakeAssessmentLLM({"questions": [valid_question(chunk_id)]})
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        llm,
        "fake-embedding",
        "fake-reranker",
    )
    service.generate(
        QuestionGenerationRequest(query="database indexing", document_id=document_id),
        weak_areas_hint=["some-weak-topic"],
    )
    assert "database indexing" in llm.prompts[0]


def test_generate_multiple_weak_areas_all_appear_in_prompt(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    llm = FakeAssessmentLLM({"questions": [valid_question(chunk_id)]})
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        llm,
        "fake-embedding",
        "fake-reranker",
    )
    weak = ["topic-a", "topic-b", "topic-c"]
    service.generate(
        QuestionGenerationRequest(query="database indexing", document_id=document_id),
        weak_areas_hint=weak,
    )
    for label in weak:
        assert label in llm.prompts[0]


def test_adaptive_generation_still_applies_m3c_quality_validation(
    session_factory,
) -> None:
    """Weak-areas hint must not bypass quality validation."""
    document_id, chunk_id = make_source(session_factory)
    bad_question = valid_question(chunk_id)
    bad_question["options"][1]["option_text"] = "An index"  # duplicate option text
    service = make_service(
        session_factory, document_id, chunk_id, {"questions": [bad_question]}
    )
    with pytest.raises(AssessmentValidationError):
        service.generate(
            QuestionGenerationRequest(query="database indexing", document_id=document_id),
            weak_areas_hint=["weak-topic"],
        )


def test_adaptive_generation_still_validates_citations(session_factory) -> None:
    """Citation validation must not be bypassed by the adaptive path."""
    document_id, chunk_id = make_source(session_factory)
    bad_question = valid_question(chunk_id)
    bad_question["cited_chunk_ids"] = ["not-a-real-chunk"]
    service = make_service(
        session_factory, document_id, chunk_id, {"questions": [bad_question]}
    )
    with pytest.raises((AssessmentValidationError, Exception)):
        service.generate(
            QuestionGenerationRequest(query="database indexing", document_id=document_id),
            weak_areas_hint=["weak-topic"],
        )


# ---------------------------------------------------------------------------
# Repository + domain: weak-area resolution for a known learner
# ---------------------------------------------------------------------------

def test_known_learner_weak_topic_detected_and_bounded(session_factory) -> None:
    """A learner with 3 all-wrong answers on a topic is detected as weak."""
    learner = "learner-adaptive-1"
    for _ in range(3):
        qid = make_question_with_topic_skill(
            session_factory, topic="weak-topic", skill="weak-skill")
        submit_answer(session_factory, learner, qid, is_correct=False)

    with session_factory() as session:
        repo = LearnerRepository(session)
        topic_perfs = repo.get_topic_performance_for_learner(learner)
        skill_perfs = repo.get_skill_performance_for_learner(learner)

    weak_topics = detect_weak_areas(topic_perfs)
    weak_skills = detect_weak_areas(skill_perfs)

    assert any(w.label == "weak-topic" for w in weak_topics)
    assert any(w.label == "weak-skill" for w in weak_skills)


def test_unknown_learner_has_no_weak_areas(session_factory) -> None:
    with session_factory() as session:
        repo = LearnerRepository(session)
        topic_perfs = repo.get_topic_performance_for_learner("nobody")
        skill_perfs = repo.get_skill_performance_for_learner("nobody")

    assert detect_weak_areas(topic_perfs) == []
    assert detect_weak_areas(skill_perfs) == []


def test_learner_with_few_attempts_not_weak(session_factory) -> None:
    """Below min_attempts, no weak areas even with 0% accuracy."""
    learner = "learner-few-attempts"
    for _ in range(2):  # below WEAK_TOPIC_MIN_ATTEMPTS=3
        qid = make_question_with_topic_skill(
            session_factory, topic="under-threshold-topic", skill="sk")
        submit_answer(session_factory, learner, qid, is_correct=False)

    with session_factory() as session:
        repo = LearnerRepository(session)
        topic_perfs = repo.get_topic_performance_for_learner(learner)

    assert detect_weak_areas(topic_perfs) == []


def test_weak_areas_bounded_to_adaptive_max(session_factory) -> None:
    """Even with many weak areas, the hint list is bounded by ADAPTIVE_WEAK_AREAS_MAX."""
    learner = "learner-many-weak"
    # Create ADAPTIVE_WEAK_AREAS_MAX + 2 distinct weak topics
    for i in range(ADAPTIVE_WEAK_AREAS_MAX + 2):
        for _ in range(3):
            qid = make_question_with_topic_skill(
                session_factory,
                topic=f"weak-topic-{i}",
                skill=f"weak-skill-{i}",
            )
            submit_answer(session_factory, learner, qid, is_correct=False)

    with session_factory() as session:
        repo = LearnerRepository(session)
        topic_perfs = repo.get_topic_performance_for_learner(learner)
        skill_perfs = repo.get_skill_performance_for_learner(learner)

    weak_topics = detect_weak_areas(topic_perfs)
    weak_skills = detect_weak_areas(skill_perfs)

    # Simulate the bounding logic in the route
    seen: set[str] = set()
    combined: list[str] = []
    for area in weak_topics + weak_skills:
        if area.label not in seen:
            seen.add(area.label)
            combined.append(area.label)
        if len(combined) >= ADAPTIVE_WEAK_AREAS_MAX:
            break

    assert len(combined) == ADAPTIVE_WEAK_AREAS_MAX


# ---------------------------------------------------------------------------
# API-level
# ---------------------------------------------------------------------------

def test_api_generation_without_learner_id_behaves_normally(
    client, session_factory
) -> None:
    """Omitting learner_id must produce the same response as before M4D."""
    from app.api.dependencies import get_assessment_generation_service, get_assessment_repository
    document_id, chunk_id = make_source(session_factory)
    llm = FakeAssessmentLLM({"questions": [valid_question(chunk_id)]})
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        llm,
        "fake-embedding",
        "fake-reranker",
    )
    client.app.dependency_overrides[get_assessment_generation_service] = lambda: service
    client.app.dependency_overrides[get_assessment_repository] = lambda: AssessmentRepository(
        session_factory()
    )
    try:
        resp = client.post(
            "/questions/generate",
            json={"document_id": document_id, "query": "database indexing"},
        )
        assert resp.status_code == 201
        assert "Learner preference" not in llm.prompts[0]
    finally:
        client.app.dependency_overrides.clear()


def test_api_generation_with_unknown_learner_id_behaves_normally(
    client, session_factory
) -> None:
    """An unknown learner_id (no submissions) produces no adaptive hint."""
    from app.api.dependencies import get_assessment_generation_service, get_assessment_repository
    document_id, chunk_id = make_source(session_factory)
    llm = FakeAssessmentLLM({"questions": [valid_question(chunk_id)]})
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        llm,
        "fake-embedding",
        "fake-reranker",
    )
    client.app.dependency_overrides[get_assessment_generation_service] = lambda: service
    client.app.dependency_overrides[get_assessment_repository] = lambda: AssessmentRepository(
        session_factory()
    )
    try:
        resp = client.post(
            "/questions/generate",
            json={
                "document_id": document_id,
                "query": "database indexing",
                "learner_id": "completely-unknown-learner",
            },
        )
        assert resp.status_code == 201
        # No weak areas → no hint appended
        assert "Learner preference" not in llm.prompts[0]
    finally:
        client.app.dependency_overrides.clear()


def test_api_generation_with_weak_learner_injects_hint(
    client, session_factory
) -> None:
    """A learner with enough wrong answers on a topic gets the hint injected."""
    from app.api.dependencies import get_assessment_generation_service, get_assessment_repository
    learner = "learner-api-adaptive"
    weak_topic = "fragile-topic"
    for _ in range(3):
        qid = make_question_with_topic_skill(
            session_factory, topic=weak_topic, skill="fragile-skill")
        submit_answer(session_factory, learner, qid, is_correct=False)

    document_id, chunk_id = make_source(session_factory)
    llm = FakeAssessmentLLM({"questions": [valid_question(chunk_id)]})
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        llm,
        "fake-embedding",
        "fake-reranker",
    )
    client.app.dependency_overrides[get_assessment_generation_service] = lambda: service
    client.app.dependency_overrides[get_assessment_repository] = lambda: AssessmentRepository(
        session_factory()
    )
    try:
        resp = client.post(
            "/questions/generate",
            json={
                "document_id": document_id,
                "query": "database indexing",
                "learner_id": learner,
            },
        )
        assert resp.status_code == 201
        assert "Learner preference" in llm.prompts[0]
        assert weak_topic in llm.prompts[0]
        # Original query must still be present
        assert "database indexing" in llm.prompts[0]
    finally:
        client.app.dependency_overrides.clear()


def test_api_generation_blank_learner_id_returns_422(client, session_factory) -> None:
    resp = client.post(
        "/questions/generate",
        json={"query": "topic", "learner_id": "  "},
    )
    assert resp.status_code == 422
