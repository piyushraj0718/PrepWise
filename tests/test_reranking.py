import math

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_search_service
from app.services.reranking import (
    InvalidRerankingQuery,
    RerankerProviderError,
    RerankingService,
)
from app.services.retrieval import RetrievalResult


def make_candidate(index: int, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{index}",
        document_id="document-1",
        similarity=float(index) / 10,
        bm25_score=float(index) / 20,
        hybrid_score=float(index) / 30,
        chunk_text=text,
        chunk_index=index,
        page_number=index + 1,
    )


class FakeRetrieval:
    def __init__(self, candidates: list[RetrievalResult]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int, str | None]] = []

    def search(self, query: str, top_k: int, document_id: str | None = None):
        self.calls.append((query, top_k, document_id))
        return self.candidates[:top_k]


class FakeReranker:
    model_name = "fake-reranker-v1"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    def score(self, query: str, candidates: list[tuple[str, str]]) -> list[float]:
        self.calls.append((query, candidates))
        return self.scores[:len(candidates)]


class FailingReranker:
    model_name = "fake-reranker-v1"

    def score(self, query: str, candidates: list[tuple[str, str]]) -> list[float]:
        raise RuntimeError("model unavailable")


def test_reranking_changes_order_and_preserves_scores_and_metadata() -> None:
    retrieval = FakeRetrieval([
        make_candidate(0, "general notes"),
        make_candidate(1, "database indexing"),
    ])
    provider = FakeReranker([0.1, 0.9])
    service = RerankingService(retrieval, provider)

    results = service.search("database", top_k=2, candidate_k=2)

    assert [result.chunk_id for result in results] == ["chunk-1", "chunk-0"]
    assert results[0].reranker_score == 0.9
    assert results[0].similarity == 0.1
    assert results[0].bm25_score == 0.05
    assert results[0].hybrid_score == pytest.approx(1 / 30)
    assert results[0].final_rank == 1
    assert results[0].page_number == 2
    assert provider.calls == [("database", [("chunk-0", "general notes"),
                                            ("chunk-1", "database indexing")])]


def test_candidate_k_limits_reranker_input_and_top_k_limits_output() -> None:
    retrieval = FakeRetrieval(
        [make_candidate(index, f"text {index}") for index in range(5)])
    provider = FakeReranker([0.1, 0.2, 0.9, 0.4, 0.5])
    service = RerankingService(retrieval, provider)

    results = service.search("query", top_k=2, candidate_k=3)

    assert retrieval.calls == [("query", 3, None)]
    assert len(provider.calls[0][1]) == 3
    assert len(results) == 2
    assert [result.chunk_id for result in results] == ["chunk-2", "chunk-1"]


def test_default_candidate_k_is_larger_than_top_k() -> None:
    retrieval = FakeRetrieval(
        [make_candidate(index, f"text {index}") for index in range(12)])
    service = RerankingService(retrieval, FakeReranker([1.0] * 12))

    service.search("query", top_k=2)

    assert retrieval.calls[0][1] == 10


def test_document_filter_is_forwarded() -> None:
    retrieval = FakeRetrieval([make_candidate(0, "filtered")])
    service = RerankingService(retrieval, FakeReranker([1.0]))

    service.search("query", top_k=1, candidate_k=1, document_id="document-1")

    assert retrieval.calls == [("query", 1, "document-1")]


def test_no_candidates_does_not_call_reranker() -> None:
    retrieval = FakeRetrieval([])
    provider = FakeReranker([])
    service = RerankingService(retrieval, provider)

    assert service.search("query", top_k=1, candidate_k=1) == []
    assert provider.calls == []


@pytest.mark.parametrize(
    ("top_k", "candidate_k"),
    [(0, None), (51, None), (2, 1), (1, 0), (1, 101), (1, True)],
)
def test_invalid_bounds_are_rejected(top_k: int, candidate_k: int | None) -> None:
    service = RerankingService(FakeRetrieval([]), FakeReranker([]))

    with pytest.raises(InvalidRerankingQuery):
        service.search("query", top_k=top_k, candidate_k=candidate_k)


def test_empty_query_is_rejected() -> None:
    service = RerankingService(FakeRetrieval([]), FakeReranker([]))

    with pytest.raises(InvalidRerankingQuery):
        service.search("   ", top_k=1, candidate_k=1)


def test_reranker_failure_is_wrapped() -> None:
    service = RerankingService(
        FakeRetrieval([make_candidate(0, "text")]), FailingReranker())

    with pytest.raises(RerankerProviderError):
        service.search("query", top_k=1, candidate_k=1)


def test_invalid_reranker_scores_are_rejected() -> None:
    service = RerankingService(
        FakeRetrieval([make_candidate(0, "text")]), FakeReranker([math.nan]))

    with pytest.raises(RerankerProviderError):
        service.search("query", top_k=1, candidate_k=1)


def test_malformed_candidate_data_is_rejected() -> None:
    service = RerankingService(FakeRetrieval([object()]), FakeReranker([1.0]))

    with pytest.raises(InvalidRerankingQuery):
        service.search("query", top_k=1, candidate_k=1)


def test_search_endpoint_uses_fake_reranker_without_loading_real_model(
    client: TestClient,
) -> None:
    class EmptySearch:
        def search(self, query: str, top_k: int, candidate_k: int | None, document_id: str | None):
            return []

    client.app.dependency_overrides[get_search_service] = lambda: EmptySearch()

    response = client.post(
        "/search", json={"query": "query", "top_k": 1, "candidate_k": 1}
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
