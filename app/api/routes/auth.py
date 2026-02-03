"""Clerk webhook handling for user/org sync."""

import hashlib
import hmac
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from sqlalchemy import text

from app.config import settings
from app.dependencies import get_db_context

router = APIRouter(prefix="/auth", tags=["auth"])


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Verify Clerk webhook signature.

    Args:
        payload: Raw request body
        signature: Svix-Signature header value
        secret: Webhook secret

    Returns:
        True if signature is valid
    """
    # Clerk uses Svix for webhooks
    # Signature format: v1,<timestamp>,<signature>
    try:
        parts = signature.split(",")
        if len(parts) < 2:
            return False

        timestamp = None
        signatures = []

        for part in parts:
            if part.startswith("v1="):
                signatures.append(part[3:])
            elif part.startswith("t="):
                timestamp = part[2:]

        if not timestamp or not signatures:
            return False

        # Create signed payload
        signed_payload = f"{timestamp}.{payload.decode()}"

        # Calculate expected signature
        expected = hmac.new(
            secret.encode(),
            signed_payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        # Check if any signature matches
        return any(
            hmac.compare_digest(expected, sig)
            for sig in signatures
        )
    except Exception:
        return False


@router.post("/webhook/clerk")
async def clerk_webhook(
    request: Request,
    svix_id: str = Header(None, alias="svix-id"),
    svix_timestamp: str = Header(None, alias="svix-timestamp"),
    svix_signature: str = Header(None, alias="svix-signature"),
):
    """Handle Clerk webhook events.

    Processes user and organization events from Clerk.
    """
    # Get raw body for signature verification
    body = await request.body()

    # Verify signature
    if svix_signature:
        # Construct the signature string that Svix uses
        signature_string = f"t={svix_timestamp},{svix_signature}"
        if not verify_webhook_signature(body, signature_string, settings.clerk_webhook_secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )

    # Parse event
    try:
        event = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    event_type = event.get("type")
    data = event.get("data", {})

    # Handle different event types
    if event_type == "organization.created":
        await handle_org_created(data)
    elif event_type == "organization.updated":
        await handle_org_updated(data)
    elif event_type == "organization.deleted":
        await handle_org_deleted(data)
    elif event_type == "user.created":
        await handle_user_created(data)
    elif event_type == "user.deleted":
        await handle_user_deleted(data)
    elif event_type == "organizationMembership.created":
        await handle_membership_created(data)
    elif event_type == "organizationMembership.deleted":
        await handle_membership_deleted(data)

    return {"status": "processed", "type": event_type}


async def handle_org_created(data: dict[str, Any]) -> None:
    """Handle organization creation event."""
    clerk_org_id = data.get("id")
    name = data.get("name", "Unnamed Organization")

    if not clerk_org_id:
        return

    async with get_db_context() as db:
        # Check if org already exists
        result = await db.execute(
            text("SELECT id FROM organizations WHERE clerk_org_id = :clerk_id"),
            {"clerk_id": clerk_org_id},
        )
        existing = result.fetchone()

        if not existing:
            await db.execute(
                text("""
                    INSERT INTO organizations (clerk_org_id, name)
                    VALUES (:clerk_id, :name)
                """),
                {"clerk_id": clerk_org_id, "name": name},
            )
            await db.commit()


async def handle_org_updated(data: dict[str, Any]) -> None:
    """Handle organization update event."""
    clerk_org_id = data.get("id")
    name = data.get("name")

    if not clerk_org_id:
        return

    async with get_db_context() as db:
        await db.execute(
            text("""
                UPDATE organizations
                SET name = :name, updated_at = NOW()
                WHERE clerk_org_id = :clerk_id
            """),
            {"clerk_id": clerk_org_id, "name": name},
        )
        await db.commit()


async def handle_org_deleted(data: dict[str, Any]) -> None:
    """Handle organization deletion event."""
    clerk_org_id = data.get("id")

    if not clerk_org_id:
        return

    async with get_db_context() as db:
        # Delete organization (cascades to documents, chunks, queries)
        await db.execute(
            text("DELETE FROM organizations WHERE clerk_org_id = :clerk_id"),
            {"clerk_id": clerk_org_id},
        )
        await db.commit()


async def handle_user_created(data: dict[str, Any]) -> None:
    """Handle user creation event.

    Currently we don't store users separately - they're identified
    through Clerk tokens. This is a placeholder for future use.
    """
    pass


async def handle_user_deleted(data: dict[str, Any]) -> None:
    """Handle user deletion event.

    Clean up any user-specific data if needed.
    """
    pass


async def handle_membership_created(data: dict[str, Any]) -> None:
    """Handle organization membership creation.

    Ensure organization exists when first member joins.
    """
    org_data = data.get("organization", {})
    clerk_org_id = org_data.get("id")
    name = org_data.get("name", "Unnamed Organization")

    if not clerk_org_id:
        return

    async with get_db_context() as db:
        # Ensure org exists
        result = await db.execute(
            text("SELECT id FROM organizations WHERE clerk_org_id = :clerk_id"),
            {"clerk_id": clerk_org_id},
        )
        existing = result.fetchone()

        if not existing:
            await db.execute(
                text("""
                    INSERT INTO organizations (clerk_org_id, name)
                    VALUES (:clerk_id, :name)
                """),
                {"clerk_id": clerk_org_id, "name": name},
            )
            await db.commit()


async def handle_membership_deleted(data: dict[str, Any]) -> None:
    """Handle organization membership deletion.

    We keep organizations even if all members leave,
    as they may have documents and data.
    """
    pass
