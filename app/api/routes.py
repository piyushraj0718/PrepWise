import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.api.dependencies import (
    get_assessment_generation_service,
    get_assessment_repository,
    get_document_service,
    get_embedding_service,
    get_rag_service,
    get_search_service,
)
from app.schemas.documents import (
    DocumentChunkGenerationResponse,
    DocumentChunksResponse,
    DocumentProcessingResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    EmbeddingGenerationResponse,
    EmbeddingStatusItem,
    EmbeddingStatusResponse,
    AskRequest,
    AskResponse,
    SourceCitationItem,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.schemas.assessment_generation import QuestionGenerationResponse
from app.schemas.assessments import (
    GenerationRunResponse,
    QualityEvaluationListResponse,
    QualityEvaluationResponse,
    QuestionGenerationRequest,
    QuestionListResponse,
    QuestionResponse,
    QuestionListFilter,
)
from app.services.documents import (
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentProcessingStateError,
    DocumentService,
    UnsupportedDocumentError,
    UploadStorageError,
)
from app.services.embeddings import (
    EmbeddingPersistenceError,
    EmbeddingProviderError,
    EmbeddingService,
    InvalidEmbeddingError,
)
from app.services.retrieval import InvalidRetrievalQuery, RetrievalProviderError, RetrievalService
from app.services.reranking import (
    InvalidRerankingQuery,
    RerankerProviderError,
    RerankingService,
)
from app.services.rag import InvalidRAGQuery, RAGError, RAGService
from app.services.assessment_generation import (
    AssessmentGenerationError,
    AssessmentGenerationService,
    AssessmentNoContextError,
    AssessmentPersistenceError,
    AssessmentValidationError,
)
from app.repositories.assessments import AssessmentRepository
from app.ai.llm import LLMConfigurationError, LLMProviderError, LLMResponseError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/documents", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    try:
        document = await service.upload(file)
    except UnsupportedDocumentError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except UploadStorageError as error:
        logger.error("Unable to store uploaded document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be stored",
        ) from error

    return DocumentUploadResponse(
        document_id=document.id,
        status=document.status,
        created_at=document.created_at,
    )


@router.post("/documents/{document_id}/process", response_model=DocumentProcessingResponse)
def process_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentProcessingResponse:
    try:
        document = service.process(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except DocumentProcessingError as error:
        logger.warning("Unable to process document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The document could not be processed",
        ) from error

    return DocumentProcessingResponse(document_id=document.id, status=document.status)


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentStatusResponse:
    try:
        document = service.get(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    return DocumentStatusResponse(
        document_id=document.id,
        status=document.status,
        created_at=document.created_at,
    )


@router.post("/documents/{document_id}/chunks", response_model=DocumentChunkGenerationResponse)
def generate_document_chunks(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentChunkGenerationResponse:
    try:
        chunks = service.generate_chunks(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except DocumentProcessingError as error:
        logger.warning(
            "Unable to generate chunks for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The document chunks could not be generated",
        ) from error

    return DocumentChunkGenerationResponse(document_id=document_id, chunk_count=len(chunks))


@router.get("/documents/{document_id}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentChunksResponse:
    try:
        chunks = service.get_chunks(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return DocumentChunksResponse(
        document_id=document_id,
        chunks=[
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "page_number": chunk.page_number,
                "created_at": chunk.created_at,
            }
            for chunk in chunks
        ],
    )


@router.post("/documents/{document_id}/embeddings", response_model=EmbeddingGenerationResponse)
def generate_document_embeddings(
    document_id: str,
    rebuild: bool = Query(False),
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingGenerationResponse:
    try:
        summary = service.generate(document_id, rebuild=rebuild)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (DocumentProcessingError, InvalidEmbeddingError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except EmbeddingProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except EmbeddingPersistenceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error

    return EmbeddingGenerationResponse(
        document_id=summary.document_id,
        model_name=summary.model_name,
        dimension=summary.dimension,
        embedded_chunk_count=summary.embedded_chunk_count,
    )


@router.get("/documents/{document_id}/embeddings", response_model=EmbeddingStatusResponse)
def get_document_embedding_status(
    document_id: str,
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingStatusResponse:
    try:
        embeddings = service.get_status(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    grouped: dict[str, list] = {}
    for embedding in embeddings:
        grouped.setdefault(embedding.model_name, []).append(embedding)
    return EmbeddingStatusResponse(
        document_id=document_id,
        embeddings=[
            EmbeddingStatusItem(
                model_name=model_name,
                dimension=rows[0].dimension,
                embedded_chunk_count=len(rows),
            )
            for model_name, rows in grouped.items()
        ],
    )


@router.post("/search", response_model=SearchResponse)
def search_documents(
    request: SearchRequest,
    service: RerankingService = Depends(get_search_service),
) -> SearchResponse:
    try:
        results = service.search(
            request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            document_id=str(
                request.document_id) if request.document_id else None,
        )
    except RetrievalProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except InvalidRetrievalQuery as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except RerankerProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except InvalidRerankingQuery as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        results=[SearchResultItem.model_validate(
            result) for result in results],
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    try:
        result = service.ask(
            request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
            document_id=str(
                request.document_id) if request.document_id else None,
        )
    except (InvalidRAGQuery, InvalidRerankingQuery, InvalidRetrievalQuery) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except (LLMConfigurationError, LLMProviderError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except LLMResponseError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except RAGError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    return AskResponse(
        query=request.query,
        answer=result.answer,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        citations=[
            SourceCitationItem(
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                chunk_index=citation.chunk_index,
                page_number=citation.page_number,
                chunk_text=citation.chunk_text,
                similarity=citation.similarity,
                bm25_score=citation.bm25_score,
                hybrid_score=citation.hybrid_score,
                reranker_score=citation.reranker_score,
            )
            for citation in result.citations
        ],
    )


@router.post(
    "/questions/generate",
    response_model=QuestionGenerationResponse,
    status_code=status.HTTP_201_CREATED,
)
def generate_questions(
    request: QuestionGenerationRequest,
    service: AssessmentGenerationService = Depends(
        get_assessment_generation_service),
) -> QuestionGenerationResponse:
    try:
        run, questions = service.generate(request)
    except (DocumentNotFoundError,) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except DocumentProcessingStateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (InvalidRerankingQuery, InvalidRetrievalQuery) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except (AssessmentNoContextError, AssessmentValidationError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except LLMResponseError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except AssessmentPersistenceError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
    except AssessmentGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    return QuestionGenerationResponse(
        run=GenerationRunResponse.model_validate(run),
        questions=[QuestionResponse.model_validate(
            question) for question in questions],
    )


@router.get("/questions/{question_id}", response_model=QuestionResponse)
def get_question(
    question_id: str,
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> QuestionResponse:
    question = repository.get_question(question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Question was not found")
    return QuestionResponse.model_validate(question)


@router.get("/questions", response_model=QuestionListResponse)
def list_questions(
    filters: QuestionListFilter = Depends(),
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> QuestionListResponse:
    questions = repository.list_questions(
        document_id=str(filters.document_id) if filters.document_id else None,
        question_type=filters.question_type.value if filters.question_type else None,
        difficulty=filters.difficulty.value if filters.difficulty else None,
        bloom_level=filters.bloom_level.value if filters.bloom_level else None,
        topic=filters.topic,
        skill=filters.skill,
        limit=filters.limit,
        offset=filters.offset,
    )
    return QuestionListResponse(
        questions=[QuestionResponse.model_validate(
            question) for question in questions],
        limit=filters.limit,
        offset=filters.offset,
    )


@router.get(
    "/question-generation-runs/{run_id}",
    response_model=GenerationRunResponse,
)
def get_generation_run(
    run_id: str,
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> GenerationRunResponse:
    run = repository.get_generation_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question generation run was not found",
        )
    return GenerationRunResponse.model_validate(run)


@router.get(
    "/questions/{question_id}/quality",
    response_model=QualityEvaluationListResponse,
)
def get_question_quality(
    question_id: str,
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> QualityEvaluationListResponse:
    question = repository.get_question(question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question was not found",
        )
    evaluations = repository.get_question_quality(question_id)
    return QualityEvaluationListResponse(
        evaluations=[
            QualityEvaluationResponse.model_validate(item) for item in evaluations
        ],
    )


@router.get(
    "/question-generation-runs/{run_id}/quality",
    response_model=QualityEvaluationListResponse,
)
def get_generation_run_quality(
    run_id: str,
    repository: AssessmentRepository = Depends(get_assessment_repository),
) -> QualityEvaluationListResponse:
    run = repository.get_generation_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question generation run was not found",
        )
    evaluations = repository.get_run_quality(run_id)
    return QualityEvaluationListResponse(
        evaluations=[
            QualityEvaluationResponse.model_validate(item) for item in evaluations
        ],
    )
