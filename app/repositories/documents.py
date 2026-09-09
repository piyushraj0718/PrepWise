from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentChunk, DocumentEmbedding, DocumentText


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        *,
        original_filename: str,
        storage_path: str,
        content_type: str,
        size_bytes: int,
    ) -> Document:
        document = Document(
            original_filename=original_filename,
            storage_path=storage_path,
            content_type=content_type,
            size_bytes=size_bytes,
            status="uploaded",
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def mark_failed(self, document: Document, reason: str) -> None:
        document.status = "failed"
        document.failure_reason = reason[:500]
        self.db.commit()
        self.db.refresh(document)

    def get_by_id(self, document_id: str) -> Document | None:
        return self.db.get(Document, document_id)

    def get_extracted_text(self, document_id: str) -> DocumentText | None:
        return self.db.get(DocumentText, document_id)

    def get_document_chunks(self, document_id: str) -> list[DocumentChunk]:
        rows = self.db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).scalars().all()
        return list(rows)

    def replace_chunks(self, document: Document, chunk_specs: list[tuple[str, int | None]]) -> list[DocumentChunk]:
        self.db.execute(delete(DocumentChunk).where(
            DocumentChunk.document_id == document.id))

        saved: list[DocumentChunk] = []
        for index, (chunk_text, page_number) in enumerate(chunk_specs):
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                chunk_text=chunk_text,
                page_number=page_number,
            )
            self.db.add(chunk)
            saved.append(chunk)

        self.db.commit()
        for chunk in saved:
            self.db.refresh(chunk)
        return saved

    def get_document_embeddings(self, document_id: str) -> list[DocumentEmbedding]:
        rows = self.db.execute(
            select(DocumentEmbedding)
            .join(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).scalars().all()
        return list(rows)

    def get_embeddings_for_chunks(
        self, chunk_ids: list[str], model_name: str
    ) -> list[DocumentEmbedding]:
        if not chunk_ids:
            return []
        rows = self.db.execute(
            select(DocumentEmbedding).where(
                DocumentEmbedding.chunk_id.in_(chunk_ids),
                DocumentEmbedding.model_name == model_name,
            )
        ).scalars().all()
        return list(rows)

    def replace_embeddings(
        self,
        chunks: list[DocumentChunk],
        model_name: str,
        dimension: int,
        vectors: list[list[float]],
    ) -> list[DocumentEmbedding]:
        try:
            self.db.execute(delete(DocumentEmbedding).where(
                DocumentEmbedding.chunk_id.in_([chunk.id for chunk in chunks]),
                DocumentEmbedding.model_name == model_name,
            ))
            saved: list[DocumentEmbedding] = []
            for chunk, vector in zip(chunks, vectors):
                embedding = DocumentEmbedding(
                    chunk_id=chunk.id,
                    model_name=model_name,
                    dimension=dimension,
                    vector=vector,
                )
                self.db.add(embedding)
                saved.append(embedding)
            self.db.commit()
            for embedding in saved:
                self.db.refresh(embedding)
            return saved
        except Exception:
            self.db.rollback()
            raise

    def mark_processing(self, document: Document) -> None:
        document.status = "processing"
        document.failure_reason = None
        self.db.commit()
        self.db.refresh(document)

    def save_extracted_text(self, document: Document, content: str) -> DocumentText:
        text = DocumentText(document_id=document.id, content=content)
        self.db.add(text)
        self.db.commit()
        self.db.refresh(text)
        return text

    def mark_processed(self, document: Document) -> None:
        document.status = "processed"
        document.failure_reason = None
        self.db.commit()
        self.db.refresh(document)
