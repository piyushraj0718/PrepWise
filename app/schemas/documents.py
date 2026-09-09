from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    document_id: UUID | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be empty")
        return value


class SearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    document_id: str
    similarity: float
    bm25_score: float
    hybrid_score: float
    reranker_score: float
    final_rank: int
    chunk_text: str
    chunk_index: int
    page_number: int | None = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    candidate_k: int | None = None
    results: list[SearchResultItem]


class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    document_id: UUID | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must not be empty")
        return value


class SourceCitationItem(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int | None = None
    chunk_text: str
    similarity: float
    bm25_score: float
    hybrid_score: float
    reranker_score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunk_ids: list[str]
    citations: list[SourceCitationItem]
