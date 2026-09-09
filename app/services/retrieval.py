import logging
import math
import re
from dataclasses import dataclass
from numbers import Real
from collections import Counter

from app.ai.embeddings import EmbeddingProvider
from app.models.document import DocumentChunk
from app.repositories.documents import DocumentRepository
from app.services.documents import DocumentNotFoundError

logger = logging.getLogger(__name__)


class RetrievalProviderError(RuntimeError):
    pass


class InvalidRetrievalQuery(ValueError):
    pass


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    similarity: float
    bm25_score: float
    hybrid_score: float
    chunk_text: str
    chunk_index: int
    page_number: int | None


class RetrievalService:
    _TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
    _RRF_K = 60

    def __init__(self, repository: DocumentRepository, provider: EmbeddingProvider) -> None:
        self.repository = repository
        self.provider = provider

    def search(
        self, query: str, top_k: int = 5, document_id: str | None = None
    ) -> list[RetrievalResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise InvalidRetrievalQuery("Query must not be empty")
        if not 1 <= top_k <= 50:
            raise InvalidRetrievalQuery("top_k must be between 1 and 50")
        if document_id is not None and self.repository.get_by_id(document_id) is None:
            raise DocumentNotFoundError("Document was not found")

        try:
            query_vector = self.provider.embed_text(normalized_query)
        except Exception as error:
            logger.exception("Retrieval embedding provider failed")
            raise RetrievalProviderError(
                "The embedding provider failed") from error
        self._validate_vector(query_vector)
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            raise InvalidRetrievalQuery(
                "The query embedding has zero magnitude")

        dense_results: list[RetrievalResult] = []
        for chunk, embedding in self.repository.get_retrieval_candidates(
            self.provider.model_name, document_id
        ):
            vector = embedding.vector
            if embedding.dimension != len(query_vector):
                continue
            try:
                self._validate_vector(vector)
            except InvalidRetrievalQuery:
                continue
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if vector_norm == 0:
                continue
            similarity = sum(
                query_value * chunk_value
                for query_value, chunk_value in zip(query_vector, vector)
            ) / (query_norm * vector_norm)
            dense_results.append(self._result(chunk, similarity, 0.0, 0.0))

        lexical_candidates = self.repository.get_lexical_candidates(
            document_id)
        bm25_scores = self._bm25_scores(normalized_query, lexical_candidates)
        dense_results.sort(key=self._dense_sort_key)
        dense_ranks = {result.chunk_id: rank for rank,
                       result in enumerate(dense_results, 1)}
        lexical_ranked = sorted(
            (
                (chunk, bm25_scores[chunk.id])
                for chunk in lexical_candidates
                if bm25_scores[chunk.id] > 0
            ),
            key=lambda item: (-item[1], item[0].document_id,
                              item[0].chunk_index),
        )
        lexical_ranks = {chunk.id: rank for rank,
                         (chunk, _) in enumerate(lexical_ranked, 1)}
        chunks_by_id = {chunk.id: chunk for chunk in lexical_candidates}
        dense_scores_by_id = {
            result.chunk_id: result.similarity for result in dense_results}

        results = []
        for chunk_id, chunk in chunks_by_id.items():
            dense_rank = dense_ranks.get(chunk_id)
            lexical_rank = lexical_ranks.get(chunk_id)
            hybrid_score = sum(
                1 / (self._RRF_K + rank)
                for rank in (dense_rank, lexical_rank)
                if rank is not None
            )
            results.append(self._result(
                chunk,
                dense_scores_by_id.get(
                    chunk_id, 0.0) if dense_rank is not None else 0.0,
                bm25_scores[chunk_id],
                hybrid_score,
            ))

        results.sort(key=self._hybrid_sort_key)
        return results[:top_k]

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        if not vector or any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in vector
        ):
            raise InvalidRetrievalQuery("The embedding vector is invalid")

    @staticmethod
    def _result(
        chunk: DocumentChunk,
        similarity: float,
        bm25_score: float,
        hybrid_score: float,
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            similarity=similarity,
            bm25_score=bm25_score,
            hybrid_score=hybrid_score,
            chunk_text=chunk.chunk_text,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
        )

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return [token.casefold() for token in cls._TOKEN_PATTERN.findall(text)]

    @classmethod
    def _bm25_scores(
        cls, query: str, chunks: list[DocumentChunk]
    ) -> dict[str, float]:
        query_terms = cls._tokens(query)
        tokenized = {chunk.id: cls._tokens(
            chunk.chunk_text) for chunk in chunks}
        document_frequency = Counter(
            term for tokens in tokenized.values() for term in set(tokens)
        )
        document_count = len(chunks)
        average_length = (
            sum(len(tokens) for tokens in tokenized.values()) / document_count
            if document_count else 0.0
        )
        scores: dict[str, float] = {}
        for chunk in chunks:
            tokens = tokenized[chunk.id]
            term_counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = term_counts[term]
                if not frequency:
                    continue
                document_frequency_value = document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency_value + 0.5)
                    / (document_frequency_value + 0.5)
                )
                length_factor = (
                    frequency + 1.5 * (1 - 0.75 + 0.75 *
                                       len(tokens) / average_length)
                    if average_length else frequency
                )
                score += inverse_document_frequency * (
                    frequency * 2.5 / length_factor
                )
            scores[chunk.id] = score
        return scores

    @staticmethod
    def _dense_sort_key(result: RetrievalResult) -> tuple[float, str, int]:
        return (-result.similarity, result.document_id, result.chunk_index)

    @staticmethod
    def _hybrid_sort_key(result: RetrievalResult) -> tuple[float, float, str, int]:
        return (-result.hybrid_score, -result.bm25_score, result.document_id, result.chunk_index)
