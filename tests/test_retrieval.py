from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.dependencies import get_search_service
from app.models.document import Document, DocumentChunk, DocumentEmbedding
from app.repositories.documents import DocumentRepository
from app.services.retrieval import RetrievalService
from app.services.reranking import RerankingService


class FakeRetrievalProvider:
    model_name = "fake-retrieval-v1"

    def __init__(self, query_vector: list[float] | None = None) -> None:
        self.query_vector = query_vector or [1.0, 0.0]

    def embed_text(self, text: str) -> list[float]:
        return self.query_vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.query_vector for _ in texts]


class FailingRetrievalProvider(FakeRetrievalProvider):
    def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("provider unavailable")


class PassthroughReranker:
    model_name = "fake-reranker-v1"

    def score(self, query: str, candidates: list[tuple[str, str]]) -> list[float]:
        return [float(len(candidates) - index)
                for index in range(len(candidates))]


def add_document(
    session_factory,
    vectors: list[list[float]],
    chunk_texts: list[str] | None = None,
) -> str:
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
        repository = DocumentRepository(session)
        texts = chunk_texts or [
            f"chunk {index}" for index in range(len(vectors))]
        chunks = repository.replace_chunks(
            document, [(text, index + 1) for index, text in enumerate(texts)]
        )
        repository.replace_embeddings(chunks, "fake-retrieval-v1", 2, vectors)
    return document_id


def configure_retrieval(client: TestClient, session_factory, provider) -> None:
    def service_override() -> RerankingService:
        retrieval = RetrievalService(
            DocumentRepository(session_factory()), provider)
        return RerankingService(retrieval, PassthroughReranker())

    client.app.dependency_overrides[get_search_service] = service_override


def test_empty_query_is_rejected(client: TestClient, session_factory) -> None:
    configure_retrieval(client, session_factory, FakeRetrievalProvider())

    response = client.post("/search", json={"query": "   "})

    assert response.status_code == 422


def test_top_k_validation(client: TestClient, session_factory) -> None:
    configure_retrieval(client, session_factory, FakeRetrievalProvider())

    assert client.post(
        "/search", json={"query": "topic", "top_k": 0}).status_code == 422
    assert client.post(
        "/search", json={"query": "topic", "top_k": 51}).status_code == 422


def test_search_ranks_by_cosine_similarity_and_preserves_metadata(
    client: TestClient, session_factory
) -> None:
    document_id = add_document(
        session_factory, [[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]])
    configure_retrieval(client, session_factory, FakeRetrievalProvider())

    response = client.post("/search", json={"query": "topic", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "topic"
    assert body["top_k"] == 2
    assert [result["chunk_index"] for result in body["results"]] == [1, 0]
    assert body["results"][0]["similarity"] == 1.0
    assert body["results"][0]["document_id"] == document_id
    assert body["results"][0]["chunk_text"] == "chunk 1"
    assert body["results"][0]["page_number"] == 2
    assert body["results"][0]["bm25_score"] == 0.0
    assert body["results"][0]["hybrid_score"] > 0.0


def test_bm25_ranks_matching_chunk_and_preserves_lexical_score(
    client: TestClient, session_factory
) -> None:
    document_id = add_document(
        session_factory,
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        ["python decorators", "database indexing", "python testing"],
    )
    configure_retrieval(client, session_factory, FakeRetrievalProvider())

    response = client.post("/search", json={"query": "database", "top_k": 1})

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["document_id"] == document_id
    assert result["chunk_text"] == "database indexing"
    assert result["bm25_score"] > 0


def test_hybrid_fusion_can_promote_lexical_match_over_dense_match(
    client: TestClient, session_factory
) -> None:
    add_document(
        session_factory,
        [[1.0, 0.0], [0.0, 1.0]],
        ["general preparation notes", "database indexing guide"],
    )
    configure_retrieval(client, session_factory, FakeRetrievalProvider())

    response = client.post("/search", json={"query": "database", "top_k": 2})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["chunk_index"] for result in results] == [1, 0]
    assert results[0]["similarity"] == 0.0
    assert results[0]["bm25_score"] > 0.0
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


def test_document_id_filtering_and_no_result_behavior(client: TestClient, session_factory) -> None:
    first_document_id = add_document(session_factory, [[1.0, 0.0]])
    second_document_id = add_document(session_factory, [[0.0, 1.0]])
    configure_retrieval(client, session_factory, FakeRetrievalProvider())

    filtered = client.post(
        "/search", json={"query": "topic", "document_id": second_document_id}
    )
    empty = client.post(
        "/search", json={"query": "topic", "document_id": str(uuid4())}
    )

    assert filtered.status_code == 200
    assert [result["document_id"] for result in filtered.json()["results"]] == [
        second_document_id]
    assert empty.status_code == 404
    assert first_document_id != second_document_id


def test_search_returns_empty_results_when_no_embeddings_exist(
    client: TestClient, session_factory
) -> None:
    add_document(session_factory, [])
    configure_retrieval(client, session_factory, FakeRetrievalProvider())

    response = client.post("/search", json={"query": "topic"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_provider_failure_returns_gateway_error(client: TestClient, session_factory) -> None:
    configure_retrieval(client, session_factory, FailingRetrievalProvider())

    response = client.post("/search", json={"query": "topic"})

    assert response.status_code == 502


def test_malformed_stored_embedding_is_skipped_without_breaking_bm25(
    client: TestClient, session_factory
) -> None:
    add_document(
        session_factory,
        [[1.0, 0.0]],
        ["database indexing guide"],
    )
    with session_factory() as session:
        embedding = session.query(DocumentEmbedding).one()
        embedding.vector = ["invalid"]
        session.commit()
    configure_retrieval(client, session_factory, FakeRetrievalProvider())

    response = client.post("/search", json={"query": "database"})

    assert response.status_code == 200
    assert response.json()["results"][0]["similarity"] == 0.0
    assert response.json()["results"][0]["bm25_score"] > 0.0
