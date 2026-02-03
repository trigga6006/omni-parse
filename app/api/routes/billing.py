"""Stripe billing integration endpoints."""

from typing import Optional
from uuid import UUID

import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.clerk_auth import ClerkUser, require_org_admin, require_org_membership
from app.config import settings
from app.dependencies import get_db, get_org_uuid
from app.models.schemas import (
    CheckoutResponse,
    CreateCheckoutRequest,
    SubscriptionStatus,
    SubscriptionTier,
)

router = APIRouter(prefix="/billing", tags=["billing"])

# Initialize Stripe
stripe.api_key = settings.stripe_secret_key


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    request: CreateCheckoutRequest,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_admin),
):
    """Create a Stripe checkout session for subscription."""
    # Get or create Stripe customer
    result = await db.execute(
        text("""
            SELECT stripe_customer_id, clerk_org_id, name
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

    customer_id = org.stripe_customer_id

    if not customer_id:
        # Create new Stripe customer
        customer = stripe.Customer.create(
            name=org.name,
            metadata={
                "org_id": str(org_id),
                "clerk_org_id": org.clerk_org_id,
            },
        )
        customer_id = customer.id

        # Save customer ID
        await db.execute(
            text("""
                UPDATE organizations
                SET stripe_customer_id = :customer_id, updated_at = NOW()
                WHERE id = CAST(:org_id AS uuid)
            """),
            {"org_id": str(org_id), "customer_id": customer_id},
        )
        await db.commit()

    # Create checkout session
    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[
            {
                "price": request.price_id,
                "quantity": 1,
            }
        ],
        success_url=request.success_url,
        cancel_url=request.cancel_url,
        metadata={
            "org_id": str(org_id),
        },
    )

    return CheckoutResponse(
        checkout_url=session.url,
        session_id=session.id,
    )


@router.get("/subscription", response_model=SubscriptionStatus)
async def get_subscription_status(
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_membership),
):
    """Get current subscription status."""
    result = await db.execute(
        text("""
            SELECT subscription_tier, stripe_subscription_id
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

    # Default response for free tier
    response = SubscriptionStatus(
        tier=org.subscription_tier,
        status="active" if org.subscription_tier == SubscriptionTier.FREE else "unknown",
        current_period_end=None,
        cancel_at_period_end=False,
    )

    # Get Stripe subscription details if exists
    if org.stripe_subscription_id:
        try:
            subscription = stripe.Subscription.retrieve(org.stripe_subscription_id)
            response.status = subscription.status
            response.current_period_end = subscription.current_period_end
            response.cancel_at_period_end = subscription.cancel_at_period_end
        except stripe.error.StripeError:
            pass

    return response


@router.post("/portal")
async def create_portal_session(
    return_url: str,
    db: AsyncSession = Depends(get_db),
    org_id: UUID = Depends(get_org_uuid),
    user: ClerkUser = Depends(require_org_admin),
):
    """Create a Stripe customer portal session."""
    result = await db.execute(
        text("SELECT stripe_customer_id FROM organizations WHERE id = CAST(:org_id AS uuid)"),
        {"org_id": str(org_id)},
    )
    org = result.fetchone()

    if not org or not org.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No billing account found",
        )

    session = stripe.billing_portal.Session.create(
        customer=org.stripe_customer_id,
        return_url=return_url,
    )

    return {"url": session.url}


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="stripe-signature"),
):
    """Handle Stripe webhook events."""
    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, settings.stripe_webhook_secret
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid payload",
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature",
        )

    # Handle specific events
    if event["type"] == "checkout.session.completed":
        await handle_checkout_completed(event["data"]["object"])
    elif event["type"] == "customer.subscription.updated":
        await handle_subscription_updated(event["data"]["object"])
    elif event["type"] == "customer.subscription.deleted":
        await handle_subscription_deleted(event["data"]["object"])

    return {"status": "processed"}


async def handle_checkout_completed(session: dict) -> None:
    """Handle successful checkout."""
    from app.dependencies import get_db_context

    org_id = session.get("metadata", {}).get("org_id")
    subscription_id = session.get("subscription")

    if not org_id or not subscription_id:
        return

    # Get subscription details
    subscription = stripe.Subscription.retrieve(subscription_id)
    price_id = subscription["items"]["data"][0]["price"]["id"]

    # Determine tier from price ID
    tier = SubscriptionTier.FREE
    if price_id == settings.stripe_price_id_basic:
        tier = SubscriptionTier.BASIC
    elif price_id == settings.stripe_price_id_pro:
        tier = SubscriptionTier.PRO

    async with get_db_context() as db:
        await db.execute(
            text("""
                UPDATE organizations
                SET subscription_tier = :tier,
                    stripe_subscription_id = :sub_id,
                    updated_at = NOW()
                WHERE id = CAST(:org_id AS uuid)
            """),
            {"org_id": org_id, "tier": tier.value, "sub_id": subscription_id},
        )
        await db.commit()


async def handle_subscription_updated(subscription: dict) -> None:
    """Handle subscription update."""
    from app.dependencies import get_db_context

    subscription_id = subscription.get("id")
    status = subscription.get("status")

    if status == "canceled" or status == "unpaid":
        # Downgrade to free
        async with get_db_context() as db:
            await db.execute(
                text("""
                    UPDATE organizations
                    SET subscription_tier = 'free', updated_at = NOW()
                    WHERE stripe_subscription_id = :sub_id
                """),
                {"sub_id": subscription_id},
            )
            await db.commit()


async def handle_subscription_deleted(subscription: dict) -> None:
    """Handle subscription cancellation."""
    from app.dependencies import get_db_context

    subscription_id = subscription.get("id")

    async with get_db_context() as db:
        await db.execute(
            text("""
                UPDATE organizations
                SET subscription_tier = 'free',
                    stripe_subscription_id = NULL,
                    updated_at = NOW()
                WHERE stripe_subscription_id = :sub_id
            """),
            {"sub_id": subscription_id},
        )
        await db.commit()
