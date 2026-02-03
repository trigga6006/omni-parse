"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db
from app.models.schemas import HealthCheck
from app.services.cache import cache_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheck)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthCheck:
    """Check application health status."""
    # Check database
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unhealthy"

    # Check Redis
    redis_status = "healthy"
    try:
        healthy = await cache_service.health_check()
        if not healthy:
            redis_status = "unhealthy"
    except Exception:
        redis_status = "unhealthy"

    # Determine overall status
    overall_status = "healthy"
    if db_status == "unhealthy" or redis_status == "unhealthy":
        overall_status = "degraded"

    return HealthCheck(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        database=db_status,
        redis=redis_status,
    )


@router.get("/health/live")
async def liveness():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """Kubernetes readiness probe endpoint."""
    # Check critical dependencies
    try:
        await db.execute(text("SELECT 1"))
        await cache_service.health_check()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not ready", "error": str(e)}
