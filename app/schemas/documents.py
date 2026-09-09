from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    status: str
    created_at: datetime


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    created_at: datetime


class DocumentProcessingResponse(BaseModel):
    document_id: str
    status: str


class DocumentChunkItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_text: str
    page_number: int | None = None
    created_at: datetime


class DocumentChunkGenerationResponse(BaseModel):
    document_id: str
    chunk_count: int


class DocumentChunksResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    chunks: list[DocumentChunkItem]


class EmbeddingGenerationResponse(BaseModel):
    document_id: str
    model_name: str
    dimension: int
    embedded_chunk_count: int


class EmbeddingStatusItem(BaseModel):
    model_name: str
    dimension: int
    embedded_chunk_count: int


class EmbeddingStatusResponse(BaseModel):
    document_id: str
    embeddings: list[EmbeddingStatusItem]
