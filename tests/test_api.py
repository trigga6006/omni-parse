"""Basic API tests."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock

# Mock settings before importing app
with patch.dict("os.environ", {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_ANON_KEY": "test-anon-key",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-key",
    "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
    "REDIS_URL": "redis://localhost:6379",
    "CLERK_SECRET_KEY": "sk_test_123",
    "CLERK_WEBHOOK_SECRET": "whsec_test",
    "OPENAI_API_KEY": "sk-test",
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "COHERE_API_KEY": "test-cohere",
    "STRIPE_SECRET_KEY": "sk_test_stripe",
    "STRIPE_WEBHOOK_SECRET": "whsec_stripe",
}):
    from app.main import app


@pytest.fixture
def mock_cache():
    """Mock cache service."""
    with patch("app.services.cache.cache_service") as mock:
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.health_check = AsyncMock(return_value=True)
        yield mock


@pytest.fixture
def mock_memory():
    """Mock memory service."""
    with patch("app.services.memory.memory_service") as mock:
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        yield mock


@pytest.fixture
def mock_db():
    """Mock database session."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_session.execute = AsyncMock(return_value=mock_result)
    return mock_session


@pytest.fixture
async def client(mock_cache, mock_memory):
    """Create test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint returns app info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health_live(client):
    """Test liveness probe."""
    response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_health_check(client, mock_db):
    """Test health check endpoint."""
    with patch("app.api.routes.health.get_db") as mock_get_db:
        mock_get_db.return_value.__anext__ = AsyncMock(return_value=mock_db)

        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "database" in data
        assert "redis" in data


@pytest.mark.asyncio
async def test_documents_requires_auth(client):
    """Test documents endpoint requires authentication."""
    response = await client.get("/api/v1/documents")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_requires_auth(client):
    """Test query endpoint requires authentication."""
    response = await client.post(
        "/api/v1/query",
        json={"query": "test question"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_sessions_requires_auth(client):
    """Test sessions endpoint requires authentication."""
    response = await client.get("/api/v1/sessions")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_organizations_requires_auth(client):
    """Test organizations endpoint requires authentication."""
    response = await client.get("/api/v1/organizations/current")
    assert response.status_code == 401


class TestSchemas:
    """Test Pydantic schemas."""

    def test_query_request_validation(self):
        """Test query request validation."""
        from app.models.schemas import QueryRequest

        # Valid request
        req = QueryRequest(query="What is the warranty period?")
        assert req.query == "What is the warranty period?"
        assert req.top_k == 5

        # Custom top_k
        req = QueryRequest(query="test", top_k=10)
        assert req.top_k == 10

    def test_query_request_empty_query(self):
        """Test query request rejects empty query."""
        from app.models.schemas import QueryRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_document_status_enum(self):
        """Test document status enum values."""
        from app.models.schemas import DocumentStatus

        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.PROCESSING == "processing"
        assert DocumentStatus.COMPLETED == "completed"
        assert DocumentStatus.FAILED == "failed"

    def test_subscription_tier_enum(self):
        """Test subscription tier enum values."""
        from app.models.schemas import SubscriptionTier

        assert SubscriptionTier.FREE == "free"
        assert SubscriptionTier.BASIC == "basic"
        assert SubscriptionTier.PRO == "pro"
        assert SubscriptionTier.ENTERPRISE == "enterprise"


class TestHelpers:
    """Test utility helper functions."""

    def test_generate_hash(self):
        """Test hash generation."""
        from app.utils.helpers import generate_hash

        hash1 = generate_hash("test")
        hash2 = generate_hash("test")
        hash3 = generate_hash("different")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16

    def test_truncate_text(self):
        """Test text truncation."""
        from app.utils.helpers import truncate_text

        text = "This is a long text that needs truncation"
        truncated = truncate_text(text, 20)

        assert len(truncated) == 20
        assert truncated.endswith("...")

        # Short text unchanged
        short = "Short"
        assert truncate_text(short, 20) == short

    def test_estimate_tokens(self):
        """Test token estimation."""
        from app.utils.helpers import estimate_tokens

        text = "This is a test sentence with several words"
        tokens = estimate_tokens(text)

        assert tokens > 0
        assert tokens > len(text.split())  # Should be more than word count

    def test_format_file_size(self):
        """Test file size formatting."""
        from app.utils.helpers import format_file_size

        assert "B" in format_file_size(100)
        assert "KB" in format_file_size(1024)
        assert "MB" in format_file_size(1024 * 1024)
        assert "GB" in format_file_size(1024 * 1024 * 1024)

    def test_is_valid_uuid(self):
        """Test UUID validation."""
        from app.utils.helpers import is_valid_uuid

        assert is_valid_uuid("550e8400-e29b-41d4-a716-446655440000")
        assert not is_valid_uuid("not-a-uuid")
        assert not is_valid_uuid("")

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        from app.utils.helpers import sanitize_filename

        assert sanitize_filename("test.pdf") == "test.pdf"
        assert "/" not in sanitize_filename("path/to/file.pdf")
        assert "\\" not in sanitize_filename("path\\to\\file.pdf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
