"""Pytest configuration and fixtures."""

import os
import pytest

# Set test environment variables before any imports
os.environ.update({
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
    "DEBUG": "true",
    "ENVIRONMENT": "test",
})


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
