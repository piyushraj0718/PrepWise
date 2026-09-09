import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from app.ai.embeddings import EmbeddingProvider
from app.models.document import DocumentEmbedding
from app.repositories.documents import DocumentRepository
from app.services.documents import (
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentProcessingStateError,
)

logger = logging.getLogger(__name__)


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingPersistenceError(RuntimeError):
    pass


class InvalidEmbeddingError(ValueError):
    pass


@dataclass(frozen=True)
class EmbeddingSummary:
    document_id: str
    model_name: str
    dimension: int
    embedded_chunk_count: int


class EmbeddingService:
    def __init__(self, repository: DocumentRepository, provider: EmbeddingProvider) -> None:
        self.repository = repository
        self.provider = provider

    def generate(self, document_id: str, rebuild: bool = False) -> EmbeddingSummary:
        document = self.repository.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError("Document was not found")
        if document.status != "processed":
            raise DocumentProcessingStateError(
                "Only documents with processed status can be embedded"
            )

        chunks = self.repository.get_document_chunks(document_id)
        if not chunks:
            raise DocumentProcessingError("Document has no chunks yet")

        chunk_ids = [chunk.id for chunk in chunks]
        existing = self.repository.get_embeddings_for_chunks(
            chunk_ids, self.provider.model_name)
        if existing and len(existing) == len(chunks) and not rebuild:
            dimension = existing[0].dimension
            return EmbeddingSummary(
                document_id=document_id,
                model_name=self.provider.model_name,
                dimension=dimension,
                embedded_chunk_count=len(existing),
            )

        try:
            vectors = self.provider.embed_texts(
                [chunk.chunk_text for chunk in chunks])
            dimension = self._validate_vectors(vectors, len(chunks))
        except InvalidEmbeddingError:
            raise
        except Exception as error:
            logger.exception(
                "Embedding provider failed: document_id=%s", document_id)
            raise EmbeddingProviderError(
                "The embedding provider failed") from error

        try:
            saved = self.repository.replace_embeddings(
                chunks, self.provider.model_name, dimension, vectors)
        except Exception as error:
            logger.exception(
                "Embedding persistence failed: document_id=%s", document_id)
            raise EmbeddingPersistenceError(
                "The document embeddings could not be persisted"
            ) from error

        return EmbeddingSummary(
            document_id=document_id,
            model_name=self.provider.model_name,
            dimension=dimension,
            embedded_chunk_count=len(saved),
        )

    def get_status(self, document_id: str) -> list[DocumentEmbedding]:
        if self.repository.get_by_id(document_id) is None:
            raise DocumentNotFoundError("Document was not found")
        return self.repository.get_document_embeddings(document_id)

    @staticmethod
    def _validate_vectors(vectors: Sequence[Sequence[Real]], expected_count: int) -> int:
        if len(vectors) != expected_count or not vectors:
            raise InvalidEmbeddingError(
                "Provider returned an invalid vector count")
        dimension = len(vectors[0])
        if dimension <= 0:
            raise InvalidEmbeddingError(
                "Provider returned an invalid embedding dimension")
        for vector in vectors:
            if len(vector) != dimension or any(
                not isinstance(value, Real) or isinstance(
                    value, bool) or not math.isfinite(value)
                for value in vector
            ):
                raise InvalidEmbeddingError(
                    "Provider returned invalid embedding values")
        return dimension
