import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.api.dependencies import get_document_service, get_embedding_service
from app.schemas.documents import (
    DocumentChunkGenerationResponse,
    DocumentChunksResponse,
    DocumentProcessingResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    EmbeddingGenerationResponse,
    EmbeddingStatusItem,
    EmbeddingStatusResponse,
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
