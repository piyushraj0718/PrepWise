from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.dependencies import get_document_service
from app.models.document import Document
from app.repositories.documents import DocumentRepository
from app.services.documents import DocumentService, LocalDocumentStorage


PDF_BYTES = b"%PDF-1.4\nminimal test document"


def test_valid_pdf_upload(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/documents",
        files={"file": ("study-guide.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "uploaded"
    assert body["document_id"]
    assert (tmp_path / "uploads" / f"{body['document_id']}.pdf").read_bytes() == PDF_BYTES


def test_invalid_file_type(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("notes.txt", b"notes", "text/plain")})

    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]


def test_missing_file(client: TestClient) -> None:
    response = client.post("/documents")

    assert response.status_code == 422


def test_database_document_creation(client: TestClient, session_factory) -> None:
    response = client.post(
        "/documents",
        files={"file": ("guide.pdf", PDF_BYTES, "application/pdf")},
    )

    document_id = response.json()["document_id"]
    with session_factory() as session:
        document = session.scalar(select(Document).where(Document.id == document_id))

    assert document is not None
    assert document.original_filename == "guide.pdf"
    assert document.status == "uploaded"
    assert document.size_bytes == len(PDF_BYTES)


def test_upload_failure_marks_document_failed(client: TestClient, session_factory, tmp_path: Path) -> None:
    class FailingStorage(LocalDocumentStorage):
        def save(self, document_id: str, content: bytes) -> Path:
            from app.services.documents import UploadStorageError

            raise UploadStorageError("disk is unavailable")

    def failing_service() -> DocumentService:
        return DocumentService(
            DocumentRepository(session_factory()), FailingStorage(tmp_path / "uploads")
        )

    app_overrides = client.app.dependency_overrides
    app_overrides[get_document_service] = failing_service
    response = client.post(
        "/documents",
        files={"file": ("guide.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 500
    with session_factory() as session:
        document = session.scalar(select(Document))
    assert document is not None
    assert document.status == "failed"
    assert document.failure_reason == "disk is unavailable"
