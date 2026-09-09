import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from app.ai.reranking import RerankerProvider
from app.services.documents import DocumentNotFoundError
from app.services.retrieval import RetrievalResult, RetrievalService

logger = logging.getLogger(__name__)


class RerankerProviderError(RuntimeError):
    pass


class InvalidRerankingQuery(ValueError):
    pass


@dataclass(frozen=True)
class RerankedResult:
    chunk_id: str
    document_id: str
    similarity: float
    bm25_score: float
    hybrid_score: float
    reranker_score: float
    final_rank: int
    chunk_text: str
    chunk_index: int
    page_number: int | None


class RerankingService:
    MAX_TOP_K = 50
    MAX_CANDIDATE_K = 100

    def __init__(
        self, retrieval: RetrievalService, provider: RerankerProvider
    ) -> None:
        self.retrieval = retrieval
        self.provider = provider

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RerankedResult]:
        normalized_query = self._validate_query(query)
        self._validate_top_k(top_k)
        resolved_candidate_k = self._resolve_candidate_k(top_k, candidate_k)

        candidates = self.retrieval.search(
            normalized_query,
            top_k=resolved_candidate_k,
            document_id=document_id,
        )
        self._validate_candidates(candidates)
        if not candidates:
            return []

        try:
            scores = self.provider.score(
                normalized_query,
                [(candidate.chunk_id, candidate.chunk_text)
                 for candidate in candidates],
            )
        except Exception as error:
            logger.exception("Retrieval reranker failed")
            raise RerankerProviderError(
                "The reranker provider failed") from error
        self._validate_scores(scores, len(candidates))

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: (
                -item[1],
                -item[0].hybrid_score,
                -item[0].bm25_score,
                item[0].document_id,
                item[0].chunk_index,
            ),
        )
        return [
            self._result(candidate, float(score), rank)
            for rank, (candidate, score) in enumerate(ranked[:top_k], 1)
        ]

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise InvalidRerankingQuery("Query must not be empty")
        return query.strip()

    @classmethod
    def _validate_top_k(cls, top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise InvalidRerankingQuery("top_k must be an integer")
        if not 1 <= top_k <= cls.MAX_TOP_K:
            raise InvalidRerankingQuery(
                f"top_k must be between 1 and {cls.MAX_TOP_K}")

    @classmethod
    def _resolve_candidate_k(cls, top_k: int, candidate_k: int | None) -> int:
        if candidate_k is None:
            return min(max(top_k * 3, 10), cls.MAX_CANDIDATE_K)
        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
            raise InvalidRerankingQuery("candidate_k must be an integer")
        if not 1 <= candidate_k <= cls.MAX_CANDIDATE_K:
            raise InvalidRerankingQuery(
                f"candidate_k must be between 1 and {cls.MAX_CANDIDATE_K}")
        if candidate_k < top_k:
            raise InvalidRerankingQuery("candidate_k must be at least top_k")
        return candidate_k

    @staticmethod
    def _validate_candidates(candidates: Sequence[RetrievalResult]) -> None:
        for candidate in candidates:
            try:
                candidate_values = (
                    candidate.chunk_id,
                    candidate.document_id,
                    candidate.chunk_text,
                    candidate.similarity,
                    candidate.bm25_score,
                    candidate.hybrid_score,
                )
            except AttributeError as error:
                raise InvalidRerankingQuery(
                    "A retrieval candidate is malformed") from error
            if not all(isinstance(value, str) and value for value in candidate_values[:3]):
                raise InvalidRerankingQuery(
                    "A retrieval candidate is malformed")
            for score in candidate_values[3:]:
                if (
                    not isinstance(score, Real)
                    or isinstance(score, bool)
                    or not math.isfinite(score)
                ):
                    raise InvalidRerankingQuery(
                        "A retrieval candidate has invalid scores")

    @staticmethod
    def _validate_scores(scores: Sequence[float], expected_count: int) -> None:
        if len(scores) != expected_count or any(
            not isinstance(score, Real)
            or isinstance(score, bool)
            or not math.isfinite(score)
            for score in scores
        ):
            raise RerankerProviderError("The reranker returned invalid scores")

    @staticmethod
    def _result(
        candidate: RetrievalResult, reranker_score: float, final_rank: int
    ) -> RerankedResult:
        return RerankedResult(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            similarity=candidate.similarity,
            bm25_score=candidate.bm25_score,
            hybrid_score=candidate.hybrid_score,
            reranker_score=reranker_score,
            final_rank=final_rank,
            chunk_text=candidate.chunk_text,
            chunk_index=candidate.chunk_index,
            page_number=candidate.page_number,
        )
