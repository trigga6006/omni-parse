"""Organization stats and management endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.clerk_auth import ClerkUser, require_org_membership, require_org_admin
from app.dependencies import get_db, get_org_uuid
from app.models.schemas import Organization, OrganizationStats, UsageStats, SubscriptionTier

router = APIRouter(prefix="/organizations", tags=["organizations"])


# Tier limits
TIER_LIMITS = {
    SubscriptionTier.FREE: {
        "queries_limit": 100,
        "documents_limit": 10,
        "storage_limit_mb": 100,
    },
    SubscriptionTier.BASIC: {
        "queries_limit": 1000,
        "documents_limit": 100,
        "storage_limit_mb": 1024,
    },
    SubscriptionTier.PRO: {
        "queries_limit": 10000,
        "documents_limit": 500,
        "storage_limit_mb": 10240,
    },
    SubscriptionTier.ENTERPRISE: {
        "queries_limit": -1,  # Unlimited
        "documents_limit": -1,
        "storage_limit_mb": -1,
    },
}


@router.get("/current", response_model=Organization)
async def get_current_organization(
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Get the current organization details."""
    result = await db.execute(
        text("""
            SELECT
                id, clerk_org_id, name, subscription_tier,
                document_count, query_count, storage_used_mb,
                created_at, updated_at
            FROM organizations
            WHERE id = CAST(:org_id AS uuid)
        """),
        {"org_id": str(org_id)},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    return Organization(
        id=row.id,
        clerk_org_id=row.clerk_org_id,
        name=row.name,
        subscription_tier=row.subscription_tier,
        document_count=row.document_count,
        query_count=row.query_count,
        storage_used_mb=row.storage_used_mb,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.get("/current/stats", response_model=OrganizationStats)
async def get_organization_stats(
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Get detailed organization statistics."""
    # Get basic org info
    org_result = await db.execute(
        text("""
            SELECT subscription_tier, document_count, query_count, storage_used_mb
            FROM organizations
            WHERE id = CAST(:org_id AS uuid)
        """),
        {"org_id": str(org_id)},
    )
    org = org_result.fetchone()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    # Get total chunks
    chunks_result = await db.execute(
        text("""
            SELECT COUNT(*) FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.organization_id = CAST(:org_id AS uuid)
        """),
        {"org_id": str(org_id)},
    )
    total_chunks = chunks_result.scalar()

    # Get queries this month
    monthly_result = await db.execute(
        text("""
            SELECT COUNT(*) FROM query_logs
            WHERE organization_id = CAST(:org_id AS uuid)
                AND created_at >= date_trunc('month', CURRENT_DATE)
        """),
        {"org_id": str(org_id)},
    )
    queries_this_month = monthly_result.scalar()

    return OrganizationStats(
        total_documents=org.document_count,
        total_chunks=total_chunks,
        total_queries=org.query_count,
        storage_used_mb=org.storage_used_mb,
        queries_this_month=queries_this_month,
        subscription_tier=org.subscription_tier,
    )


@router.get("/current/usage", response_model=UsageStats)
async def get_usage_stats(
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Get current usage against tier limits."""
    # Get org info
    result = await db.execute(
        text("""
            SELECT subscription_tier, document_count, query_count, storage_used_mb
            FROM organizations
            WHERE id = CAST(:org_id AS uuid)
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
            WHERE organization_id = CAST(:org_id AS uuid)
                AND created_at >= date_trunc('month', CURRENT_DATE)
        """),
        {"org_id": str(org_id)},
    )
    queries_this_month = monthly_result.scalar()

    # Get tier limits
    limits = TIER_LIMITS.get(org.subscription_tier, TIER_LIMITS[SubscriptionTier.FREE])

    return UsageStats(
        queries_used=queries_this_month,
        queries_limit=limits["queries_limit"],
        documents_used=org.document_count,
        documents_limit=limits["documents_limit"],
        storage_used_mb=org.storage_used_mb,
        storage_limit_mb=limits["storage_limit_mb"],
    )


@router.patch("/current", response_model=Organization)
async def update_organization(
    name: str,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_admin),
):
    """Update organization name (admin only)."""
    await db.execute(
        text("""
            UPDATE organizations
            SET name = :name, updated_at = NOW()
            WHERE id = CAST(:org_id AS uuid)
        """),
        {"org_id": str(org_id), "name": name},
    )
    await db.commit()

    # Get updated org
    result = await db.execute(
        text("""
            SELECT
                id, clerk_org_id, name, subscription_tier,
                document_count, query_count, storage_used_mb,
                created_at, updated_at
            FROM organizations
            WHERE id = CAST(:org_id AS uuid)
        """),
        {"org_id": str(org_id)},
    )
    row = result.fetchone()

    return Organization(
        id=row.id,
        clerk_org_id=row.clerk_org_id,
        name=row.name,
        subscription_tier=row.subscription_tier,
        document_count=row.document_count,
        query_count=row.query_count,
        storage_used_mb=row.storage_used_mb,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
