"""
Pydantic schemas for request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    TEXT = "text"
    DOCX = "docx"


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    document_id: str
    chunk_index: int
    source: str
    page: Optional[int] = None


class DocumentChunk(BaseModel):
    """A chunk of a document with embedding."""

    id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class DocumentBase(BaseModel):
    """Base document schema."""

    title: str = Field(..., min_length=1, max_length=500)
    document_type: DocumentType


class DocumentCreate(DocumentBase):
    """Schema for creating a document."""

    content: str = Field(..., min_length=1)


class DocumentResponse(DocumentBase):
    """Schema for document response."""

    id: str
    content: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Schema for listing documents."""

    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class Citation(BaseModel):
    """A citation from retrieved context."""

    chunk_id: str
    content: str
    source: str
    score: float
    page: Optional[int] = None


class AnswerRequest(BaseModel):
    """Schema for asking a question."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=10, ge=1, le=50)
    use_reranking: bool = Field(default=True)
    use_caching: bool = Field(default=True)


class AnswerResponse(BaseModel):
    """Schema for answer response."""

    answer: str
    question: str
    citations: List[Citation]
    sources: List[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_used: str
    cached: bool = False


class RetrievalResult(BaseModel):
    """Schema for retrieval result."""

    chunk_id: str
    content: str
    source: str
    score: float
    page: Optional[int] = None
    retrieval_method: str  # "bm25", "vector", "hybrid"


class RetrievalResponse(BaseModel):
    """Schema for retrieval response."""

    results: List[RetrievalResult]
    query: str
    total_results: int
    retrieval_time_ms: float


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    app_name: str
    version: str
    timestamp: float


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class EvaluationRequest(BaseModel):
    """Schema for evaluation request."""

    question: str
    ground_truth_answer: str
    retrieved_contexts: List[str]


class EvaluationResponse(BaseModel):
    """Schema for evaluation response."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float