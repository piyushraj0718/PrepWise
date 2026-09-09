import logging
from dataclasses import dataclass
from numbers import Real

from app.ai.llm import (
    LLMAnswer,
    LLMProvider,
    LLMProviderError,
    LLMResponseError,
)
from app.ai.prompts import build_grounding_prompt
from app.services.reranking import (
    InvalidRerankingQuery,
    RerankedResult,
    RerankingService,
)
from app.services.retrieval import InvalidRetrievalQuery
from app.services.documents import DocumentNotFoundError

logger = logging.getLogger(__name__)


class RAGError(RuntimeError):
    pass


class InvalidRAGQuery(ValueError):
    pass


@dataclass(frozen=True)
class SourceCitation:
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int | None
    chunk_text: str
    similarity: float
    bm25_score: float
    hybrid_score: float
    reranker_score: float


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[SourceCitation]
    retrieved_chunk_ids: list[str]


class RAGService:
    def __init__(self, reranking: RerankingService, provider: LLMProvider) -> None:
        self.reranking = reranking
        self.provider = provider

    def ask(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        document_id: str | None = None,
    ) -> GroundedAnswer:
        normalized_query = self._validate_query(query)
        try:
            candidates = self.reranking.search(
                normalized_query,
                top_k=top_k,
                candidate_k=candidate_k,
                document_id=document_id,
            )
        except (
            DocumentNotFoundError,
            InvalidRAGQuery,
            InvalidRerankingQuery,
            InvalidRetrievalQuery,
        ):
            raise
        except Exception as error:
            raise RAGError("Retrieval and reranking failed") from error

        retrieved_chunk_ids = [candidate.chunk_id for candidate in candidates]
        if not candidates:
            return GroundedAnswer(
                answer="I could not find relevant context in the processed documents.",
                citations=[],
                retrieved_chunk_ids=[],
            )

        context = self._build_context(candidates)
        try:
            generated = self.provider.generate(
                build_grounding_prompt(normalized_query, context))
        except LLMProviderError:
            raise
        except Exception as error:
            logger.exception("Grounded answer provider failed")
            raise LLMProviderError("The LLM provider failed") from error
        self._validate_answer(generated)
        candidates_by_id = {
            candidate.chunk_id: candidate for candidate in candidates}
        if any(chunk_id not in candidates_by_id for chunk_id in generated.cited_chunk_ids):
            raise LLMResponseError(
                "The LLM cited a chunk that was not retrieved")
        citations = [
            self._citation(candidates_by_id[chunk_id])
            for chunk_id in self._unique_ids(generated.cited_chunk_ids)
        ]
        return GroundedAnswer(
            answer=generated.answer,
            citations=citations,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise InvalidRAGQuery("Query must not be empty")
        return query.strip()

    @staticmethod
    def _build_context(candidates: list[RerankedResult]) -> str:
        return "\n\n".join(
            "[chunk_id={chunk_id} document_id={document_id} chunk_index={chunk_index} "
            "page_number={page_number}]\n{text}".format(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                chunk_index=candidate.chunk_index,
                page_number=candidate.page_number,
                text=candidate.chunk_text,
            )
            for candidate in candidates
        )

    @staticmethod
    def _validate_answer(answer: LLMAnswer) -> None:
        if not isinstance(answer, LLMAnswer):
            raise LLMResponseError("The LLM returned an invalid response")
        if not isinstance(answer.answer, str) or not answer.answer.strip():
            raise LLMResponseError("The LLM returned an invalid response")
        if (
            not isinstance(answer.cited_chunk_ids, list)
            or not all(isinstance(chunk_id, str) and chunk_id
                       for chunk_id in answer.cited_chunk_ids)
        ):
            raise LLMResponseError("The LLM returned an invalid response")

    @staticmethod
    def _unique_ids(chunk_ids: list[str]) -> list[str]:
        return list(dict.fromkeys(chunk_ids))

    @staticmethod
    def _citation(candidate: RerankedResult) -> SourceCitation:
        scores = (
            candidate.similarity,
            candidate.bm25_score,
            candidate.hybrid_score,
            candidate.reranker_score,
        )
        if not all(isinstance(score, Real) for score in scores):
            raise LLMResponseError("Retrieved source scores are invalid")
        return SourceCitation(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            chunk_index=candidate.chunk_index,
            page_number=candidate.page_number,
            chunk_text=candidate.chunk_text,
            similarity=candidate.similarity,
            bm25_score=candidate.bm25_score,
            hybrid_score=candidate.hybrid_score,
            reranker_score=candidate.reranker_score,
        )
