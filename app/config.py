"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "TechDocs AI"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # API
    api_prefix: str = "/api/v1"

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    database_url: str

    # Redis (Upstash)
    redis_url: str
    redis_token: Optional[str] = None

    # Clerk Auth
    clerk_secret_key: str
    clerk_webhook_secret: str
    clerk_publishable_key: Optional[str] = None

    # OpenAI (optional - required for embeddings/queries)
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Anthropic (optional - required for LLM responses)
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096

    # Cohere (optional - required for reranking)
    cohere_api_key: Optional[str] = None
    rerank_model: str = "rerank-english-v3.0"
    rerank_top_n: int = 5

    # Stripe (optional - required for billing)
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_price_id_basic: Optional[str] = None
    stripe_price_id_pro: Optional[str] = None

    # Search Configuration
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    similarity_threshold: float = 0.3
    initial_results: int = 20

    # Cache Configuration
    cache_ttl_seconds: int = 3600
    embedding_cache_ttl: int = 86400
    query_cache_ttl: int = 1800

    # Session Configuration
    session_ttl_seconds: int = 1800
    max_conversation_turns: int = 20

    # Document Processing
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50

    # Storage
    storage_bucket: str = "documents"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
