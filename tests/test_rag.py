import pytest
from fastapi.testclient import TestClient
from urllib.error import HTTPError

import app.ai.llm as llm_module
from app.ai.llm import (
    GeminiLLMProvider,
    LLMAnswer,
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
)
from app.api.dependencies import get_rag_service
from app.services.rag import GroundedAnswer, RAGService, SourceCitation
from app.services.rag import InvalidRAGQuery
from app.services.reranking import RerankedResult


def make_result(index: int, document_id: str = "document-1") -> RerankedResult:
    return RerankedResult(
        chunk_id=f"chunk-{index}",
        document_id=document_id,
        similarity=0.9 - index / 10,
        bm25_score=0.8 - index / 10,
        hybrid_score=0.7 - index / 10,
        reranker_score=0.6 - index / 10,
        final_rank=index + 1,
        chunk_text=f"source text {index}",
        chunk_index=index,
        page_number=index + 1,
    )


class FakeReranking:
    def __init__(self, results: list[RerankedResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, int | None, str | None]] = []

    def search(
        self,
        query: str,
        top_k: int,
        candidate_k: int | None,
        document_id: str | None,
    ) -> list[RerankedResult]:
        self.calls.append((query, top_k, candidate_k, document_id))
        return self.results[:top_k]


class FakeLLM:
    model_name = "fake-llm"

    def __init__(self, answer: LLMAnswer) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> LLMAnswer:
        self.prompts.append(prompt)
        return self.answer


class FailingLLM:
    model_name = "fake-llm"

    def generate(self, prompt: str) -> LLMAnswer:
        raise LLMProviderError("provider unavailable")


def test_grounded_answer_maps_only_retrieved_sources() -> None:
    reranking = FakeReranking([make_result(0), make_result(1)])
    llm = FakeLLM(LLMAnswer("Supported answer", ["chunk-1"]))
    service = RAGService(reranking, llm)

    result = service.ask("What is relevant?", top_k=2, candidate_k=4)

    assert result.answer == "Supported answer"
    assert result.retrieved_chunk_ids == ["chunk-0", "chunk-1"]
    assert [citation.chunk_id for citation in result.citations] == ["chunk-1"]
    assert result.citations[0].document_id == "document-1"
    assert result.citations[0].page_number == 2
    assert result.citations[0].reranker_score == pytest.approx(0.5)
    assert "answer questions using only the supplied retrieved context" in llm.prompts[0].lower(
    )
    assert "chunk_id=chunk-0" in llm.prompts[0]


def test_retrieval_reranking_generation_configuration_is_forwarded() -> None:
    reranking = FakeReranking([make_result(0)])
    llm = FakeLLM(LLMAnswer("Answer", ["chunk-0"]))
    service = RAGService(reranking, llm)

    service.ask("query", top_k=3, candidate_k=7, document_id="document-9")

    assert reranking.calls == [("query", 3, 7, "document-9")]


def test_no_context_returns_grounded_fallback_without_llm_call() -> None:
    reranking = FakeReranking([])
    llm = FakeLLM(LLMAnswer("should not be used", []))
    service = RAGService(reranking, llm)

    result = service.ask("query", top_k=1, candidate_k=1)

    assert result.answer == "I could not find relevant context in the processed documents."
    assert result.citations == []
    assert result.retrieved_chunk_ids == []
    assert llm.prompts == []


def test_unknown_citation_is_rejected() -> None:
    service = RAGService(
        FakeReranking([make_result(0)]),
        FakeLLM(LLMAnswer("Answer", ["not-retrieved"])),
    )

    with pytest.raises(LLMResponseError):
        service.ask("query", top_k=1, candidate_k=1)


def test_malformed_provider_response_is_rejected() -> None:
    service = RAGService(
        FakeReranking([make_result(0)]),
        FakeLLM(LLMAnswer("", [])),
    )

    with pytest.raises(LLMResponseError):
        service.ask("query", top_k=1, candidate_k=1)


def test_llm_failure_is_preserved() -> None:
    service = RAGService(FakeReranking([make_result(0)]), FailingLLM())

    with pytest.raises(LLMProviderError):
        service.ask("query", top_k=1, candidate_k=1)


def test_missing_llm_api_configuration_is_explicit() -> None:
    provider = GeminiLLMProvider(None, "gemini-test", 1.0)

    with pytest.raises(LLMConfigurationError):
        provider.generate("prompt")


def test_gemini_http_error_is_wrapped_without_real_network_call(monkeypatch) -> None:
    def fail_request(request, timeout):
        assert request.full_url.startswith(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent?key="
        )
        assert timeout == 3.0
        raise HTTPError(request.full_url, 404, "model not found", {}, None)

    monkeypatch.setattr(llm_module, "urlopen", fail_request)
    provider = GeminiLLMProvider("test-key", "gemini-2.5-flash", 3.0)

    with pytest.raises(LLMProviderError, match="LLM API request failed"):
        provider.generate("prompt")


def test_empty_query_is_rejected() -> None:
    service = RAGService(FakeReranking([]), FakeLLM(LLMAnswer("", [])))

    with pytest.raises(ValueError):
        service.ask(" ", top_k=1, candidate_k=1)


def test_ask_endpoint_returns_citations_and_scores(client: TestClient) -> None:
    class FakeAskService:
        def ask(self, query: str, top_k: int, candidate_k: int | None, document_id: str | None):
            result = make_result(0, document_id or "document-1")
            return GroundedAnswer("Grounded answer", [
                SourceCitation(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    chunk_index=result.chunk_index,
                    page_number=result.page_number,
                    chunk_text=result.chunk_text,
                    similarity=result.similarity,
                    bm25_score=result.bm25_score,
                    hybrid_score=result.hybrid_score,
                    reranker_score=result.reranker_score,
                ),
            ], [result.chunk_id])

    client.app.dependency_overrides[get_rag_service] = lambda: FakeAskService()

    response = client.post("/ask", json={"query": "query", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Grounded answer"
    assert body["retrieved_chunk_ids"] == ["chunk-0"]
    assert body["citations"][0]["chunk_id"] == "chunk-0"
    assert body["citations"][0]["reranker_score"] == pytest.approx(0.6)


def test_ask_endpoint_handles_invalid_rag_query(client: TestClient) -> None:
    class InvalidQueryService:
        def ask(self, query: str, top_k: int, candidate_k: int | None, document_id: str | None):
            raise InvalidRAGQuery("Query must not be empty")

    client.app.dependency_overrides[get_rag_service] = lambda: InvalidQueryService(
    )

    response = client.post("/ask", json={"query": "query"})

    assert response.status_code == 422
    assert response.json()["detail"] == "Query must not be empty"
