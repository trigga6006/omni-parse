"""Pydantic schemas for request/response models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# Enums
class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SubscriptionTier(str, Enum):
    """Subscription tier levels."""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# Base Models
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime
    updated_at: Optional[datetime] = None


# User/Organization Schemas
class OrganizationBase(BaseModel):
    """Base organization schema."""

    name: str
    clerk_org_id: str


class OrganizationCreate(OrganizationBase):
    """Organization creation schema."""

    pass


class Organization(OrganizationBase, TimestampMixin):
    """Organization response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    document_count: int = 0
    query_count: int = 0
    storage_used_mb: float = 0.0


class OrganizationStats(BaseModel):
    """Organization statistics."""

    total_documents: int
    total_chunks: int
    total_queries: int
    storage_used_mb: float
    queries_this_month: int
    subscription_tier: SubscriptionTier


# Document Schemas
class DocumentBase(BaseModel):
    """Base document schema."""

    filename: str
    title: Optional[str] = None
    description: Optional[str] = None


class DocumentCreate(DocumentBase):
    """Document creation schema (from upload)."""

    pass


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""

    id: UUID
    filename: str
    status: DocumentStatus
    message: str


class Document(DocumentBase, TimestampMixin):
    """Document response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    file_path: str
    file_size: int
    mime_type: str
    status: DocumentStatus
    chunk_count: int = 0
    error_message: Optional[str] = None


class DocumentList(BaseModel):
    """Paginated document list."""

    documents: list[Document]
    total: int
    page: int
    page_size: int


# Chunk Schemas
class ChunkBase(BaseModel):
    """Base chunk schema."""

    content: str
    page_number: Optional[int] = None
    section_header: Optional[str] = None


class Chunk(ChunkBase):
    """Chunk response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    chunk_index: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkWithScore(Chunk):
    """Chunk with relevance score."""

    score: float
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None


# Query Schemas
class QueryRequest(BaseModel):
    """Query request schema."""

    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    document_ids: Optional[list[UUID]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True


class QuerySource(BaseModel):
    """Source information for query response."""

    document_id: UUID
    document_title: str
    chunk_id: UUID
    content: str
    page_number: Optional[int]
    section_header: Optional[str]
    relevance_score: float


class QueryResponse(BaseModel):
    """Query response schema."""

    answer: str
    sources: list[QuerySource] = Field(default_factory=list)
    session_id: str
    query_id: UUID
    cached: bool = False
    processing_time_ms: int


class QueryHistory(BaseModel):
    """Query history item."""

    id: UUID
    query: str
    answer: str
    created_at: datetime


# Session Schemas
class SessionCreate(BaseModel):
    """Session creation schema."""

    title: Optional[str] = None


class Session(BaseModel):
    """Session response schema."""

    id: str
    organization_id: UUID
    title: Optional[str]
    created_at: datetime
    last_activity: datetime
    message_count: int


class SessionMessage(BaseModel):
    """Message in a session."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[list[QuerySource]] = None


class SessionDetail(Session):
    """Session with messages."""

    messages: list[SessionMessage]


class SessionList(BaseModel):
    """List of sessions."""

    sessions: list[Session]
    total: int


# Billing Schemas
class CreateCheckoutRequest(BaseModel):
    """Checkout session request."""

    price_id: str
    success_url: str
    cancel_url: str


class CheckoutResponse(BaseModel):
    """Checkout session response."""

    checkout_url: str
    session_id: str


class SubscriptionStatus(BaseModel):
    """Subscription status response."""

    tier: SubscriptionTier
    status: str
    current_period_end: Optional[datetime]
    cancel_at_period_end: bool = False


class UsageStats(BaseModel):
    """Usage statistics."""

    queries_used: int
    queries_limit: int
    documents_used: int
    documents_limit: int
    storage_used_mb: float
    storage_limit_mb: float


# Webhook Schemas
class ClerkWebhookEvent(BaseModel):
    """Clerk webhook event."""

    type: str
    data: dict[str, Any]


class StripeWebhookEvent(BaseModel):
    """Stripe webhook event (handled via stripe library)."""

    pass


# Health Check
class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str
    database: str
    redis: str


# Error Response
class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
