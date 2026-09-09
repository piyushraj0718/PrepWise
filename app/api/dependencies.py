from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.ai.embeddings import SentenceTransformerEmbeddingProvider
from app.core.config import get_settings
from app.db.session import get_db
from app.repositories.documents import DocumentRepository
from app.services.documents import DocumentChunker, DocumentService, LocalDocumentStorage, PdfTextExtractor
from app.services.embeddings import EmbeddingService


def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    settings = get_settings()
    return DocumentService(
        repository=DocumentRepository(db),
        storage=LocalDocumentStorage(settings.uploads_dir),
        extractor=PdfTextExtractor(),
        chunker=DocumentChunker(),
    )


@lru_cache(maxsize=4)
def get_embedding_provider(model_name: str) -> SentenceTransformerEmbeddingProvider:
    return SentenceTransformerEmbeddingProvider(model_name)


def get_embedding_service(db: Session = Depends(get_db)) -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        repository=DocumentRepository(db),
        provider=get_embedding_provider(settings.embedding_model_name),
    )
