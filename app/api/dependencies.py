from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.ai.embeddings import SentenceTransformerEmbeddingProvider
from app.ai.reranking import SentenceTransformerCrossEncoderReranker
from app.ai.llm import GeminiLLMProvider
from app.core.config import get_settings
from app.db.session import get_db
from app.repositories.documents import DocumentRepository
from app.repositories.assessments import AssessmentRepository
from app.services.documents import DocumentChunker, DocumentService, LocalDocumentStorage, PdfTextExtractor
from app.services.embeddings import EmbeddingService
from app.services.retrieval import RetrievalService
from app.services.reranking import RerankingService
from app.services.rag import RAGService
from app.services.assessment_generation import AssessmentGenerationService


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


def get_retrieval_service(db: Session = Depends(get_db)) -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        repository=DocumentRepository(db),
        provider=get_embedding_provider(settings.embedding_model_name),
    )


def get_search_service(db: Session = Depends(get_db)) -> RerankingService:
    settings = get_settings()
    retrieval = RetrievalService(
        repository=DocumentRepository(db),
        provider=get_embedding_provider(settings.embedding_model_name),
    )
    return RerankingService(
        retrieval=retrieval,
        provider=SentenceTransformerCrossEncoderReranker(
            settings.reranker_model_name),
    )


def get_rag_service(db: Session = Depends(get_db)) -> RAGService:
    settings = get_settings()
    retrieval = RetrievalService(
        repository=DocumentRepository(db),
        provider=get_embedding_provider(settings.embedding_model_name),
    )
    reranking = RerankingService(
        retrieval=retrieval,
        provider=SentenceTransformerCrossEncoderReranker(
            settings.reranker_model_name),
    )
    return RAGService(
        reranking=reranking,
        provider=GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model_name,
            timeout_seconds=settings.gemini_timeout_seconds,
            max_retries=settings.llm_max_retries,
            retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        ),
    )


def get_assessment_generation_service(
    db: Session = Depends(get_db),
) -> AssessmentGenerationService:
    settings = get_settings()
    retrieval = RetrievalService(
        repository=DocumentRepository(db),
        provider=get_embedding_provider(settings.embedding_model_name),
    )
    reranking = RerankingService(
        retrieval=retrieval,
        provider=SentenceTransformerCrossEncoderReranker(
            settings.reranker_model_name),
    )
    return AssessmentGenerationService(
        repository=AssessmentRepository(db),
        reranking=reranking,
        provider=GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model_name,
            timeout_seconds=settings.gemini_timeout_seconds,
            max_retries=settings.llm_max_retries,
            retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        ),
        embedding_model_name=settings.embedding_model_name,
        reranker_model_name=settings.reranker_model_name,
    )


def get_assessment_repository(db: Session = Depends(get_db)) -> AssessmentRepository:
    return AssessmentRepository(db)
