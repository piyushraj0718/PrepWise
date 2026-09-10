from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.ai.llm import LLMProviderError, LLMResponseError
from app.api.dependencies import (
    get_assessment_generation_service,
    get_assessment_repository,
)
from app.models.assessment import AssessmentQuestion
from app.models.document import Document, DocumentChunk
from app.repositories.assessments import AssessmentRepository
from app.schemas.assessment_generation import GeneratedMCQ
from app.schemas.assessments import QuestionGenerationRequest
from app.services.assessment_generation import (
    AssessmentGenerationService,
    AssessmentNoContextError,
    AssessmentPersistenceError,
    AssessmentValidationError,
)
from app.services.documents import DocumentNotFoundError, DocumentProcessingStateError
from app.services.reranking import RerankedResult


def make_source(session_factory, status: str = "processed") -> tuple[str, str]:
    document_id = str(uuid4())
    with session_factory() as session:
        document = Document(
            id=document_id,
            original_filename="guide.pdf",
            storage_path="data/uploads/guide.pdf",
            content_type="application/pdf",
            size_bytes=10,
            status=status,
        )
        session.add(document)
        session.commit()
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=0,
            chunk_text="A database index improves lookup performance.",
            page_number=3,
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
        page_number=3,
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
        "explanation": "The source states that an index improves lookup performance.",
        "difficulty": "easy",
        "bloom_level": "understand",
        "topic": "database indexing",
        "skill": "identify indexing benefits",
        "cited_chunk_ids": [chunk_id],
    }


class FakeReranking:
    def __init__(self, results: list[RerankedResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, int | None, str | None]] = []

    def search(self, query: str, top_k: int, candidate_k: int | None,
               document_id: str | None) -> list[RerankedResult]:
        self.calls.append((query, top_k, candidate_k, document_id))
        return self.results[:top_k]


class FakeAssessmentLLM:
    model_name = "fake-assessment-llm"

    def __init__(self, output: object) -> None:
        self.output = output
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []

    def generate_structured(self, prompt: str, response_schema: dict[str, object]) -> object:
        self.prompts.append(prompt)
        self.schemas.append(response_schema)
        return self.output


class FailingAssessmentLLM(FakeAssessmentLLM):
    def generate_structured(self, prompt: str, response_schema: dict[str, object]) -> object:
        raise LLMProviderError("provider unavailable")


def make_service(session_factory, document_id: str, chunk_id: str, output: object):
    return AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        FakeAssessmentLLM(output),
        "fake-embedding",
        "fake-reranker",
    )


def request(document_id: str, count: int = 1) -> dict:
    return {
        "document_id": document_id,
        "query": "database indexing",
        "count": count,
        "difficulty": "easy",
        "bloom_level": "understand",
    }


@pytest.mark.parametrize(
    ("raw_level", "canonical_level"),
    [
        ("comprehension", "understand"),
        ("knowledge", "remember"),
        ("application", "apply"),
        ("analysis", "analyze"),
        ("evaluation", "evaluate"),
        ("creation", "create"),
        ("remember", "remember"),
        ("understand", "understand"),
        ("apply", "apply"),
        ("analyze", "analyze"),
        ("evaluate", "evaluate"),
        ("create", "create"),
    ],
)
def test_bloom_aliases_normalize_before_strict_enum_validation(
    raw_level: str, canonical_level: str
) -> None:
    question = valid_question("chunk-1")
    question["bloom_level"] = raw_level

    parsed = GeneratedMCQ.model_validate(question)

    assert parsed.bloom_level.value == canonical_level


def test_unknown_bloom_value_is_rejected() -> None:
    question = valid_question("chunk-1")
    question["bloom_level"] = "invent"

    with pytest.raises(ValidationError):
        GeneratedMCQ.model_validate(question)


def test_successful_grounded_generation_persists_and_maps_citations(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    llm = FakeAssessmentLLM({"questions": [valid_question(chunk_id)]})
    service = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        llm,
        "fake-embedding",
        "fake-reranker",
    )

    run, questions = service.generate(
        QuestionGenerationRequest(**request(document_id)))

    assert run.status == "completed"
    assert questions[0].correct_option_key == "A"
    assert questions[0].options[0].display_order == 0
    assert questions[0].citations[0].chunk_id == chunk_id
    assert f"[chunk_id={chunk_id}]" in llm.prompts[0]
    assert "document_id" not in llm.prompts[0]


def test_generated_results_are_safe_after_persistence_session_closes(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        {"questions": [valid_question(chunk_id)]},
    )

    run, questions = service.generate(
        QuestionGenerationRequest(**request(document_id)))
    service.repository.db.close()

    assert run.query == "database indexing"
    assert questions[0].stem == "What improves lookup performance?"
    assert questions[0].options[0].option_text == "An index"
    assert questions[0].citations[0].chunk_text_snapshot.startswith(
        "A database")


def test_assessment_read_results_are_safe_after_repository_session_closes(
    session_factory,
) -> None:
    document_id, chunk_id = make_source(session_factory)
    service = make_service(
        session_factory,
        document_id,
        chunk_id,
        {"questions": [valid_question(chunk_id)]},
    )
    run, questions = service.generate(
        QuestionGenerationRequest(**request(document_id)))
    question_id = questions[0].id

    with session_factory() as session:
        repository = AssessmentRepository(session)
        loaded_question = repository.get_question(question_id)
        listed_questions = repository.list_questions(document_id=document_id)
        loaded_run = repository.get_generation_run(run.id)

    assert loaded_question is not None
    assert loaded_question.stem == questions[0].stem
    assert loaded_question.options[0].option_key == "A"
    assert loaded_question.citations[0].chunk_id == chunk_id
    assert listed_questions[0].options[0].option_text == "An index"
    assert listed_questions[0].citations[0].chunk_text_snapshot.startswith(
        "A database")
    assert loaded_run is not None
    assert loaded_run.questions[0].stem == questions[0].stem


def test_smoke_regression_comprehension_is_normalized_before_persistence(
    session_factory,
) -> None:
    document_id, chunk_id = make_source(session_factory)
    question = valid_question(chunk_id)
    question["bloom_level"] = "comprehension"
    service = make_service(
        session_factory, document_id, chunk_id, {"questions": [question]}
    )

    _, questions = service.generate(
        QuestionGenerationRequest(**request(document_id)))

    assert questions[0].bloom_level == "understand"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda question, chunk_id: question["options"].pop(),
        lambda question, chunk_id: question["options"].__setitem__(
            1, {"option_key": "A", "option_text": "Duplicate"}),
        lambda question, chunk_id: question.__setitem__(
            "correct_option_key", "Z"),
        lambda question, chunk_id: question.__setitem__("cited_chunk_ids", []),
        lambda question, chunk_id: question.__setitem__(
            "cited_chunk_ids", ["not-retrieved"]),
        lambda question, chunk_id: question["options"].__setitem__(
            2, {"option_key": "C", "option_text": "all of the above"}),
        lambda question, chunk_id: question.__setitem__(
            "explanation", "Supported by [chunk_id=x]"),
    ],
)
def test_invalid_generated_question_is_rejected(session_factory, mutation) -> None:
    document_id, chunk_id = make_source(session_factory)
    question = valid_question(chunk_id)
    mutation(question, chunk_id)
    service = make_service(session_factory, document_id,
                           chunk_id, {"questions": [question]})

    with pytest.raises(AssessmentValidationError):
        service.generate(QuestionGenerationRequest(**request(document_id)))


def test_quality_invalid_question_is_not_persisted(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    question = valid_question(chunk_id)
    question["options"][1]["option_text"] = "An index"
    service = make_service(
        session_factory, document_id, chunk_id, {"questions": [question]}
    )

    with pytest.raises(AssessmentValidationError) as error:
        service.generate(QuestionGenerationRequest(**request(document_id)))

    assert "duplicate_option_text" in str(error.value)
    with session_factory() as session:
        assert session.query(AssessmentQuestion).count() == 0


def test_duplicate_question_stems_are_rejected(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    first = valid_question(chunk_id)
    second = valid_question(chunk_id)
    second["options"][0]["option_text"] = "A different answer"
    service = make_service(
        session_factory, document_id, chunk_id, {"questions": [first, second]}
    )

    with pytest.raises(AssessmentValidationError):
        service.generate(QuestionGenerationRequest(**request(document_id, 2)))


def test_malformed_provider_output_and_provider_failure_are_explicit(session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    malformed = make_service(
        session_factory, document_id, chunk_id, ["not an object"])
    with pytest.raises(LLMResponseError):
        malformed.generate(QuestionGenerationRequest(**request(document_id)))

    failing = AssessmentGenerationService(
        AssessmentRepository(session_factory()),
        FakeReranking([make_result(chunk_id, document_id)]),
        FailingAssessmentLLM({}),
        "fake-embedding",
        "fake-reranker",
    )
    with pytest.raises(LLMProviderError):
        failing.generate(QuestionGenerationRequest(**request(document_id)))


def test_missing_unprocessed_and_empty_context_are_rejected(session_factory) -> None:
    missing_id = str(uuid4())
    service = make_service(session_factory, missing_id,
                           "chunk", {"questions": []})
    with pytest.raises(DocumentNotFoundError):
        service.generate(QuestionGenerationRequest(**request(missing_id)))

    unprocessed_id, unprocessed_chunk = make_source(
        session_factory, "uploaded")
    unprocessed = make_service(
        session_factory, unprocessed_id, unprocessed_chunk, {"questions": []})
    with pytest.raises(DocumentProcessingStateError):
        unprocessed.generate(
            QuestionGenerationRequest(**request(unprocessed_id)))

    processed_id, processed_chunk = make_source(session_factory)
    empty = make_service(session_factory, processed_id,
                         processed_chunk, {"questions": []})
    empty.reranking.results = []
    with pytest.raises(AssessmentNoContextError):
        empty.generate(QuestionGenerationRequest(**request(processed_id)))


def test_generation_rolls_back_when_persistence_fails(session_factory, monkeypatch) -> None:
    document_id, chunk_id = make_source(session_factory)
    repository = AssessmentRepository(session_factory())

    def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(repository, "create_generation_batch", fail)
    service = AssessmentGenerationService(
        repository,
        FakeReranking([make_result(chunk_id, document_id)]),
        FakeAssessmentLLM({"questions": [valid_question(chunk_id)]}),
        "fake-embedding",
        "fake-reranker",
    )
    with pytest.raises(AssessmentPersistenceError):
        service.generate(QuestionGenerationRequest(**request(document_id)))


def configure_generation_api(client: TestClient, session_factory, document_id: str, chunk_id: str, output: object) -> None:
    client.app.dependency_overrides[get_assessment_generation_service] = lambda: make_service(
        session_factory, document_id, chunk_id, output
    )
    client.app.dependency_overrides[get_assessment_repository] = lambda: AssessmentRepository(
        session_factory()
    )


def test_generation_and_question_read_api(client: TestClient, session_factory) -> None:
    document_id, chunk_id = make_source(session_factory)
    configure_generation_api(
        client, session_factory, document_id, chunk_id, {
            "questions": [valid_question(chunk_id)]}
    )
    response = client.post("/questions/generate", json=request(document_id))
    assert response.status_code == 201
    body = response.json()
    assert body["questions"][0]["correct_option_key"] == "A"
    question_id = body["questions"][0]["id"]
    run_id = body["run"]["id"]

    fetched = client.get(f"/questions/{question_id}")
    listed = client.get(
        "/questions", params={"difficulty": "easy", "topic": "database indexing"})
    run = client.get(f"/question-generation-runs/{run_id}")
    assert fetched.status_code == listed.status_code == run.status_code == 200
    assert fetched.json()["id"] == question_id
    assert len(listed.json()["questions"]) == 1
    assert run.json()["id"] == run_id


def test_generation_request_validation_and_missing_question_api(client: TestClient) -> None:
    assert client.post("/questions/generate",
                       json={"query": "topic", "count": 21}).status_code == 422
    assert client.get(f"/questions/{uuid4()}").status_code == 404
    assert client.get(
        f"/question-generation-runs/{uuid4()}").status_code == 404
