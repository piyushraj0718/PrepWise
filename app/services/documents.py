import logging
import re
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.models.document import Document, DocumentChunk
from app.repositories.documents import DocumentRepository

logger = logging.getLogger(__name__)

PDF_CONTENT_TYPE = "application/pdf"
PDF_SIGNATURE = b"%PDF-"


class UnsupportedDocumentError(ValueError):
    pass


class UploadStorageError(RuntimeError):
    pass


class DocumentNotFoundError(ValueError):
    pass


class DocumentProcessingStateError(ValueError):
    pass


class DocumentProcessingError(RuntimeError):
    pass


class LocalDocumentStorage:
    def __init__(self, uploads_dir: Path) -> None:
        self.uploads_dir = uploads_dir

    def save(self, document_id: str, content: bytes) -> Path:
        try:
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            path = self.uploads_dir / f"{document_id}.pdf"
            path.write_bytes(content)
        except OSError as error:
            raise UploadStorageError(
                "Could not store uploaded document") from error
        return path

    def read(self, storage_path: str) -> bytes:
        try:
            return Path(storage_path).read_bytes()
        except OSError as error:
            raise DocumentProcessingError(
                "Could not read the uploaded PDF") from error


class PdfTextExtractor:
    def extract(self, content: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text()
                             or "" for page in reader.pages).strip()
        except PdfReadError as error:
            raise DocumentProcessingError(
                "The uploaded PDF is corrupt or unreadable") from error

        if not text:
            raise DocumentProcessingError(
                "No extractable text was found in the PDF")
        return text


class DocumentChunker:
    def __init__(self, max_chars: int = 250, overlap_sentences: int = 1) -> None:
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def chunk_text(self, text: str) -> list[tuple[str, int | None]]:
        normalized = text.replace("\r\n", "\n")
        paragraphs = [paragraph.strip() for paragraph in re.split(
            r"\n\s*\n+", normalized) if paragraph.strip()]
        if not paragraphs:
            raise DocumentProcessingError(
                "Document has no extractable text to chunk")

        sentence_units: list[tuple[str, int | None]] = []
        for paragraph in paragraphs:
            page_number = self._extract_page_number(paragraph)
            sentence_parts = [part.strip() for part in re.split(
                r"(?<=[.!?])\s+", paragraph) if part.strip()]
            if not sentence_parts:
                sentence_parts = [paragraph]
            sentence_units.extend((part, page_number)
                                  for part in sentence_parts)

        if not sentence_units:
            raise DocumentProcessingError(
                "Document has no extractable text to chunk")

        chunks: list[tuple[str, int | None]] = []
        buffer: list[str] = []
        buffer_page: int | None = None

        for sentence, page_number in sentence_units:
            if len(sentence) > self.max_chars:
                if buffer:
                    chunks.append((" ".join(buffer), buffer_page))
                    buffer = []
                    buffer_page = None
                chunks.append((sentence, page_number))
                continue

            if not buffer:
                buffer = [sentence]
                buffer_page = page_number
                continue

            candidate = " ".join(buffer + [sentence])
            if len(candidate) <= self.max_chars:
                buffer.append(sentence)
                if buffer_page is None and page_number is not None:
                    buffer_page = page_number
                continue

            chunks.append((" ".join(buffer), buffer_page))
            overlap_count = min(self.overlap_sentences, len(
                buffer)) if self.overlap_sentences else 0
            buffer = buffer[-overlap_count:] if overlap_count else []
            if buffer and buffer_page is None:
                buffer_page = page_number
            buffer.append(sentence)
            if buffer_page is None and page_number is not None:
                buffer_page = page_number

        if buffer:
            chunks.append((" ".join(buffer), buffer_page))

        return chunks

    @staticmethod
    def _extract_page_number(text: str) -> int | None:
        match = re.search(r"(?im)(?:^|\s)Page\s+(\d+)(?:\s|$)", text)
        if match is None:
            return None
        return int(match.group(1))


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        storage: LocalDocumentStorage,
        extractor: PdfTextExtractor | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.extractor = extractor or PdfTextExtractor()
        self.chunker = chunker or DocumentChunker()

    async def upload(self, file: UploadFile) -> Document:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf") or file.content_type != PDF_CONTENT_TYPE:
            raise UnsupportedDocumentError(
                "Only PDF files with content type application/pdf are supported"
            )

        content = await file.read()
        if not content.startswith(PDF_SIGNATURE):
            raise UnsupportedDocumentError(
                "The uploaded file is not a valid PDF")

        storage_path = str(self.storage.uploads_dir / "pending.pdf")
        document = self.repository.create(
            original_filename=filename,
            storage_path=storage_path,
            content_type=file.content_type,
            size_bytes=len(content),
        )

        try:
            stored_path = self.storage.save(document.id, content)
        except UploadStorageError as error:
            self.repository.mark_failed(document, str(error))
            logger.exception(
                "Document upload failed after record creation: document_id=%s", document.id)
            raise

        document.storage_path = str(stored_path)
        self.repository.db.commit()
        self.repository.db.refresh(document)
        logger.info("Document uploaded: document_id=%s filename=%s",
                    document.id, filename)
        return document

    def get(self, document_id: str) -> Document:
        document = self.repository.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError("Document was not found")
        return document

    def process(self, document_id: str) -> Document:
        document = self.get(document_id)
        if document.status != "uploaded":
            raise DocumentProcessingStateError(
                "Only documents with uploaded status can be processed"
            )

        self.repository.mark_processing(document)
        try:
            content = self.storage.read(document.storage_path)
            extracted_text = self.extractor.extract(content)
            self.repository.save_extracted_text(document, extracted_text)
            self.repository.mark_processed(document)
        except DocumentProcessingError as error:
            self.repository.mark_failed(document, str(error))
            logger.warning(
                "Document processing failed: document_id=%s", document.id)
            raise

        logger.info("Document processed: document_id=%s", document.id)
        return document

    def generate_chunks(self, document_id: str) -> list[DocumentChunk]:
        document = self.get(document_id)
        if document.status != "processed":
            raise DocumentProcessingStateError(
                "Only documents with processed status can be chunked"
            )

        extracted_text = self.repository.get_extracted_text(document_id)
        if extracted_text is None or not extracted_text.content.strip():
            raise DocumentProcessingError(
                "Document has no extractable text to chunk")

        chunk_specs = self.chunker.chunk_text(extracted_text.content)
        saved_chunks = self.repository.replace_chunks(document, chunk_specs)
        logger.info("Generated %s chunks for document_id=%s",
                    len(saved_chunks), document_id)
        return saved_chunks

    def get_chunks(self, document_id: str) -> list[DocumentChunk]:
        self.get(document_id)
        chunks = self.repository.get_document_chunks(document_id)
        if not chunks:
            raise DocumentProcessingError("Document has no chunks yet")
        return chunks
