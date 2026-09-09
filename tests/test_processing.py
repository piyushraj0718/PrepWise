from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_document_service
from app.models.document import Document, DocumentText
from app.repositories.documents import DocumentRepository
from app.services.documents import DocumentService, LocalDocumentStorage


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


TEXT_PDF = make_pdf("BT /F1 12 Tf 72 720 Td (PrepWise study text) Tj ET")
EMPTY_PDF = make_pdf("")
CORRUPT_PDF = b"%PDF-1.4\nthis is not a complete PDF"


def upload_pdf(client: TestClient, content: bytes, filename: str = "guide.pdf") -> str:
    response = client.post(
        "/documents",
        files={"file": (filename, content, "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()["document_id"]


def test_successful_text_extraction_and_persistence(client: TestClient, session_factory) -> None:
    document_id = upload_pdf(client, TEXT_PDF)

    response = client.post(f"/documents/{document_id}/process")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id, "status": "processed"}
    with session_factory() as session:
        text = session.get(DocumentText, document_id)
    assert text is not None
    assert text.content == "PrepWise study text"


def test_processing_status_transitions_and_status_endpoint(
    client: TestClient, session_factory, tmp_path: Path
) -> None:
    document_id = upload_pdf(client, TEXT_PDF)
    observed_statuses: list[str] = []

    class ObservingExtractor:
        def extract(self, content: bytes) -> str:
            with session_factory() as session:
                document = session.get(Document, document_id)
            assert document is not None
            observed_statuses.append(document.status)
            return "PrepWise study text"

    def processing_service() -> DocumentService:
        return DocumentService(
            DocumentRepository(session_factory()),
            LocalDocumentStorage(tmp_path / "uploads"),
            ObservingExtractor(),
        )

    client.app.dependency_overrides[get_document_service] = processing_service

    before_processing = client.get(f"/documents/{document_id}")
    processed = client.post(f"/documents/{document_id}/process")
    after_processing = client.get(f"/documents/{document_id}")

    assert before_processing.json()["status"] == "uploaded"
    assert observed_statuses == ["processing"]
    assert processed.json()["status"] == "processed"
    assert after_processing.json()["status"] == "processed"


def test_corrupt_pdf_processing_marks_document_failed(client: TestClient, session_factory) -> None:
    document_id = upload_pdf(client, CORRUPT_PDF)

    response = client.post(f"/documents/{document_id}/process")

    assert response.status_code == 422
    with session_factory() as session:
        document = session.get(Document, document_id)
    assert document is not None
    assert document.status == "failed"
    assert document.failure_reason == "The uploaded PDF is corrupt or unreadable"


def test_empty_pdf_processing_marks_document_failed(client: TestClient, session_factory) -> None:
    document_id = upload_pdf(client, EMPTY_PDF)

    response = client.post(f"/documents/{document_id}/process")

    assert response.status_code == 422
    with session_factory() as session:
        document = session.get(Document, document_id)
    assert document is not None
    assert document.status == "failed"
    assert document.failure_reason == "No extractable text was found in the PDF"


def test_processing_rejects_an_already_processed_document(client: TestClient) -> None:
    document_id = upload_pdf(client, TEXT_PDF)
    assert client.post(f"/documents/{document_id}/process").status_code == 200

    response = client.post(f"/documents/{document_id}/process")

    assert response.status_code == 409


def test_chunk_generation_persists_ordered_chunks(client: TestClient, session_factory) -> None:
    long_text = (
        "BT /F1 12 Tf 72 720 Td (PrepWise study text. This is the first section. It covers the basic theory. "
        "It introduces the main concept. This section is longer than the default chunk size.) Tj ET "
        "BT /F1 12 Tf 72 700 Td (More study text here. This is the second section. It adds context. "
        "This paragraph also contains several sentences so chunking should split it meaningfully.) Tj ET"
    )
    document_id = upload_pdf(client, make_pdf(long_text))
    assert client.post(f"/documents/{document_id}/process").status_code == 200

    response = client.post(f"/documents/{document_id}/chunks")

    assert response.status_code == 200
    assert response.json()["document_id"] == document_id
    assert response.json()["chunk_count"] >= 2

    chunks_response = client.get(f"/documents/{document_id}/chunks")
    assert chunks_response.status_code == 200
    chunks = chunks_response.json()["chunks"]
    assert [chunk["chunk_index"]
            for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["document_id"] == document_id for chunk in chunks)
    assert all(chunk["chunk_text"] for chunk in chunks)
    assert all(chunk["page_number"] in {1, 2, None} for chunk in chunks)


def test_repeated_chunk_generation_rebuilds_deterministically(client: TestClient, session_factory) -> None:
    long_text = (
        "BT /F1 12 Tf 72 720 Td (First concept. Second concept. Third concept. Fourth concept. "
        "Fifth concept. Sixth concept. Seventh concept. Eighth concept.) Tj ET "
        "BT /F1 12 Tf 72 700 Td (Ninth concept. Tenth concept. Eleventh concept. Twelfth concept. "
        "Thirteenth concept. Fourteenth concept. Fifteenth concept.) Tj ET"
    )
    document_id = upload_pdf(client, make_pdf(long_text))
    assert client.post(f"/documents/{document_id}/process").status_code == 200

    first = client.post(f"/documents/{document_id}/chunks")
    second = client.post(f"/documents/{document_id}/chunks")

    assert first.status_code == 200
    assert second.status_code == 200
    first_chunks = client.get(
        f"/documents/{document_id}/chunks").json()["chunks"]
    second_chunks = client.get(
        f"/documents/{document_id}/chunks").json()["chunks"]

    assert [chunk["chunk_text"] for chunk in first_chunks] == [
        chunk["chunk_text"] for chunk in second_chunks]
    assert [chunk["chunk_index"] for chunk in first_chunks] == [
        chunk["chunk_index"] for chunk in second_chunks]


def test_chunk_generation_rejects_unprocessed_documents(client: TestClient) -> None:
    document_id = upload_pdf(client, TEXT_PDF)

    response = client.post(f"/documents/{document_id}/chunks")

    assert response.status_code == 409


def test_chunk_generation_rejects_missing_document(client: TestClient) -> None:
    response = client.post("/documents/does-not-exist/chunks")

    assert response.status_code == 404


def test_get_chunks_rejects_document_without_chunks(client: TestClient, session_factory) -> None:
    document_id = upload_pdf(client, TEXT_PDF)
    assert client.post(f"/documents/{document_id}/process").status_code == 200

    response = client.get(f"/documents/{document_id}/chunks")

    assert response.status_code == 422
