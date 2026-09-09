from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.dependencies import get_embedding_service
from app.models.document import DocumentChunk, DocumentEmbedding
from app.repositories.documents import DocumentRepository
from app.services.embeddings import EmbeddingService


def make_pdf(content_stream: str) -> bytes:
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        ),
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(content_stream.encode('ascii'))} >>\nstream\n{content_stream}\nendstream\nendobj\n",
    ]
    body = "%PDF-1.4\n"
    offsets = []
    for object_text in objects:
        offsets.append(len(body.encode("ascii")))
        body += object_text
    xref_offset = len(body.encode("ascii"))
    xref = "xref\n0 6\n0000000000 65535 f \n" + "".join(
        f"{offset:010d} 00000 n \n" for offset in offsets
    )
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    return (body + xref + trailer).encode("ascii")


TEXT_PDF = make_pdf(
    "BT /F1 12 Tf 72 720 Td (First concept. Second concept. Third concept.) Tj ET"
)


class FakeEmbeddingProvider:
    model_name = "fake-embedding-v1"

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls = 0

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [
            [float(len(text)), float(index), 1.0][:self.dimension]
            for index, text in enumerate(texts)
        ]


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider unavailable")


def configure_embedding_provider(client: TestClient, session_factory, provider) -> None:
    def service_override() -> EmbeddingService:
        return EmbeddingService(DocumentRepository(session_factory()), provider)

    client.app.dependency_overrides[get_embedding_service] = service_override


def prepare_document(client: TestClient) -> str:
    uploaded = client.post(
        "/documents", files={"file": ("guide.pdf", TEXT_PDF, "application/pdf")}
    )
    document_id = uploaded.json()["document_id"]
    assert client.post(f"/documents/{document_id}/process").status_code == 200
    assert client.post(f"/documents/{document_id}/chunks").status_code == 200
    return document_id


def test_embedding_generation_persists_vectors_and_metadata(client: TestClient, session_factory) -> None:
    provider = FakeEmbeddingProvider()
    configure_embedding_provider(client, session_factory, provider)
    document_id = prepare_document(client)

    response = client.post(f"/documents/{document_id}/embeddings")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert body["model_name"] == provider.model_name
    assert body["dimension"] == 3
    assert body["embedded_chunk_count"] == 1
    with session_factory() as session:
        embeddings = session.scalars(select(DocumentEmbedding)).all()
        chunks = session.scalars(select(DocumentChunk)).all()
    assert len(embeddings) == 1
    assert embeddings[0].chunk_id == chunks[0].id
    assert embeddings[0].model_name == provider.model_name
    assert embeddings[0].dimension == 3
    assert len(embeddings[0].vector) == 3


def test_repeated_generation_is_idempotent_and_rebuild_is_explicit(
    client: TestClient, session_factory
) -> None:
    provider = FakeEmbeddingProvider()
    configure_embedding_provider(client, session_factory, provider)
    document_id = prepare_document(client)

    first = client.post(f"/documents/{document_id}/embeddings")
    second = client.post(f"/documents/{document_id}/embeddings")
    rebuilt = client.post(f"/documents/{document_id}/embeddings?rebuild=true")

    assert first.status_code == second.status_code == rebuilt.status_code == 200
    assert provider.calls == 2
    with session_factory() as session:
        assert len(session.scalars(select(DocumentEmbedding)).all()) == 1


def test_embedding_status_returns_summary_without_vectors(client: TestClient, session_factory) -> None:
    provider = FakeEmbeddingProvider()
    configure_embedding_provider(client, session_factory, provider)
    document_id = prepare_document(client)
    assert client.post(
        f"/documents/{document_id}/embeddings").status_code == 200

    response = client.get(f"/documents/{document_id}/embeddings")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "embeddings": [{
            "model_name": provider.model_name,
            "dimension": 3,
            "embedded_chunk_count": 1,
        }],
    }
    assert "vector" not in response.text


def test_embedding_generation_rejects_invalid_document_states(client: TestClient, session_factory) -> None:
    provider = FakeEmbeddingProvider()
    configure_embedding_provider(client, session_factory, provider)
    uploaded = client.post(
        "/documents", files={"file": ("guide.pdf", TEXT_PDF, "application/pdf")}
    )
    document_id = uploaded.json()["document_id"]

    assert client.post(
        f"/documents/{document_id}/embeddings").status_code == 409
    assert client.post("/documents/missing/embeddings").status_code == 404
    assert client.post(f"/documents/{document_id}/chunks").status_code == 409

    assert client.post(f"/documents/{document_id}/process").status_code == 200
    assert client.post(
        f"/documents/{document_id}/embeddings").status_code == 422


def test_provider_failure_returns_gateway_error(client: TestClient, session_factory) -> None:
    configure_embedding_provider(
        client, session_factory, FailingEmbeddingProvider())
    document_id = prepare_document(client)

    response = client.post(f"/documents/{document_id}/embeddings")

    assert response.status_code == 502
    with session_factory() as session:
        assert session.scalars(select(DocumentEmbedding)).all() == []


def test_invalid_embedding_dimension_is_rejected(client: TestClient, session_factory) -> None:
    provider = FakeEmbeddingProvider(dimension=0)
    configure_embedding_provider(client, session_factory, provider)
    document_id = prepare_document(client)

    response = client.post(f"/documents/{document_id}/embeddings")

    assert response.status_code == 422
    with session_factory() as session:
        assert session.scalars(select(DocumentEmbedding)).all() == []
