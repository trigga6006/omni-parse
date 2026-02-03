"""FastAPI dependencies for dependency injection."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.clerk_auth import ClerkUser, require_org_membership
from app.models.database import get_async_session_factory

# Global session factory (initialized on startup)
_session_factory = None


def init_session_factory():
    """Initialize the session factory."""
    global _session_factory
    _session_factory = get_async_session_factory()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    if _session_factory is None:
        init_session_factory()

    async with _session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database session (for background tasks)."""
    if _session_factory is None:
        init_session_factory()

    async with _session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_org_uuid(
    user: ClerkUser = Depends(require_org_membership),
    db: AsyncSession = Depends(get_db),
) -> UUID:
    """Get the internal organization UUID from Clerk org ID.

    This maps the Clerk organization ID to our internal UUID.
    """
    if not user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization membership required",
        )

    # Look up internal org ID from Clerk org ID
    result = await db.execute(
        text("SELECT id FROM organizations WHERE clerk_org_id = :clerk_id"),
        {"clerk_id": user.org_id},
    )
    row = result.fetchone()

    if not row:
        # Organization doesn't exist yet - create it
        # This can happen if webhook didn't fire or was delayed
        result = await db.execute(
            text("""
                INSERT INTO organizations (clerk_org_id, name)
                VALUES (:clerk_id, :name)
                ON CONFLICT (clerk_org_id) DO UPDATE SET clerk_org_id = :clerk_id
                RETURNING id
            """),
            {"clerk_id": user.org_id, "name": "Organization"},
        )
        row = result.fetchone()
        await db.commit()

    return row.id


async def check_usage_limits(
    org_id: UUID = Depends(get_org_uuid),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Check if organization has exceeded usage limits.

    Raises HTTPException if limits are exceeded.
    """
    from app.api.routes.organizations import TIER_LIMITS
    from app.models.schemas import SubscriptionTier

    # Get org info
    result = await db.execute(
        text("""
            SELECT subscription_tier, document_count, storage_used_mb
            FROM organizations
            WHERE id = :org_id::uuid
        """),
        {"org_id": str(org_id)},
    )
    org = result.fetchone()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    # Get monthly query count
    monthly_result = await db.execute(
        text("""
            SELECT COUNT(*) FROM query_logs
            WHERE organization_id = :org_id::uuid
                AND created_at >= date_trunc('month', CURRENT_DATE)
        """),
        {"org_id": str(org_id)},
    )
    queries_this_month = monthly_result.scalar()

    # Get tier limits
    limits = TIER_LIMITS.get(org.subscription_tier, TIER_LIMITS[SubscriptionTier.FREE])

    # Check query limit
    if limits["queries_limit"] != -1 and queries_this_month >= limits["queries_limit"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Monthly query limit reached. Please upgrade your plan.",
        )


async def check_document_limit(
    org_id: UUID = Depends(get_org_uuid),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Check if organization can upload more documents."""
    from app.api.routes.organizations import TIER_LIMITS
    from app.models.schemas import SubscriptionTier

    result = await db.execute(
        text("""
            SELECT subscription_tier, document_count
            FROM organizations
            WHERE id = :org_id::uuid
        """),
        {"org_id": str(org_id)},
    )
    org = result.fetchone()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    limits = TIER_LIMITS.get(org.subscription_tier, TIER_LIMITS[SubscriptionTier.FREE])

    if limits["documents_limit"] != -1 and org.document_count >= limits["documents_limit"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document limit reached. Please upgrade your plan.",
        )


async def check_storage_limit(
    file_size_mb: float,
    org_id: UUID = Depends(get_org_uuid),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Check if organization has storage capacity."""
    from app.api.routes.organizations import TIER_LIMITS
    from app.models.schemas import SubscriptionTier

    result = await db.execute(
        text("""
            SELECT subscription_tier, storage_used_mb
            FROM organizations
            WHERE id = :org_id::uuid
        """),
        {"org_id": str(org_id)},
    )
    org = result.fetchone()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    limits = TIER_LIMITS.get(org.subscription_tier, TIER_LIMITS[SubscriptionTier.FREE])

    if limits["storage_limit_mb"] != -1:
        if org.storage_used_mb + file_size_mb > limits["storage_limit_mb"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Storage limit would be exceeded. Please upgrade your plan.",
            )
