"""Session management endpoints."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.middleware.clerk_auth import ClerkUser, require_org_membership
from app.dependencies import get_org_uuid
from app.models.schemas import (
    Session,
    SessionCreate,
    SessionDetail,
    SessionList,
    SessionMessage,
)
from app.services.memory import memory_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=Session)
async def create_session(
    request: SessionCreate,
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Create a new conversation session."""
    session_id = await memory_service.create_session(
        str(org_id), title=request.title
    )

    session = await memory_service.get_session(str(org_id), session_id)

    return Session(
        id=session["id"],
        organization_id=org_id,
        title=session["title"],
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        message_count=session["message_count"],
    )


@router.get("", response_model=SessionList)
async def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """List conversation sessions for the organization."""
    sessions, total = await memory_service.list_sessions(
        str(org_id), limit=limit, offset=offset
    )

    return SessionList(
        sessions=[
            Session(
                id=s["id"],
                organization_id=org_id,
                title=s["title"],
                created_at=s["created_at"],
                last_activity=s["last_activity"],
                message_count=s["message_count"],
            )
            for s in sessions
        ],
        total=total,
    )


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Get a specific session with its messages."""
    session = await memory_service.get_session(str(org_id), session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    messages = await memory_service.get_messages(str(org_id), session_id)

    return SessionDetail(
        id=session["id"],
        organization_id=org_id,
        title=session["title"],
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        message_count=session["message_count"],
        messages=messages,
    )


@router.patch("/{session_id}", response_model=Session)
async def update_session(
    session_id: str,
    title: str,
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Update session title."""
    success = await memory_service.update_session_title(
        str(org_id), session_id, title
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    session = await memory_service.get_session(str(org_id), session_id)

    return Session(
        id=session["id"],
        organization_id=org_id,
        title=session["title"],
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        message_count=session["message_count"],
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Delete a conversation session."""
    success = await memory_service.delete_session(str(org_id), session_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
