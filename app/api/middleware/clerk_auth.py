"""Clerk JWT authentication middleware."""

from typing import Optional
from uuid import UUID

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import settings


class ClerkUser(BaseModel):
    """Authenticated Clerk user."""

    user_id: str
    org_id: Optional[str] = None
    org_role: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None


class ClerkAuth:
    """Clerk JWT verification."""

    def __init__(self):
        self._jwks: Optional[dict] = None
        self._jwks_url = f"https://api.clerk.com/v1/jwks"

    async def _get_jwks(self) -> dict:
        """Fetch Clerk JWKS for token verification."""
        if self._jwks:
            return self._jwks

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self._jwks_url,
                headers={"Authorization": f"Bearer {settings.clerk_secret_key}"},
            )
            response.raise_for_status()
            self._jwks = response.json()

        return self._jwks

    async def verify_token(self, token: str) -> dict:
        """Verify a Clerk JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid
        """
        try:
            # Get JWKS
            jwks = await self._get_jwks()

            # Decode header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            # Find the matching key
            rsa_key = None
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    rsa_key = key
                    break

            if not rsa_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unable to find appropriate key",
                )

            # Verify and decode token
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                options={"verify_aud": False},
            )

            return payload

        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
            )

    async def get_current_user(self, token: str) -> ClerkUser:
        """Get current user from token.

        Args:
            token: JWT token string

        Returns:
            ClerkUser object
        """
        payload = await self.verify_token(token)

        # Extract user info from Clerk token
        user_id = payload.get("sub")
        org_id = payload.get("org_id")
        org_role = payload.get("org_role")

        # Get additional user info if available
        email = None
        name = None
        if "email" in payload:
            email = payload["email"]
        if "name" in payload:
            name = payload["name"]

        return ClerkUser(
            user_id=user_id,
            org_id=org_id,
            org_role=org_role,
            email=email,
            name=name,
        )


# Global auth instance
clerk_auth = ClerkAuth()

# Security scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> ClerkUser:
    """FastAPI dependency to get current authenticated user.

    Args:
        request: FastAPI request
        credentials: Bearer token credentials

    Returns:
        ClerkUser object

    Raises:
        HTTPException: If not authenticated
    """
    # Check for token in header
    token = None

    if credentials:
        token = credentials.credentials
    else:
        # Check for token in cookie (for browser requests)
        token = request.cookies.get("__session")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await clerk_auth.get_current_user(token)


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[ClerkUser]:
    """FastAPI dependency to optionally get current user.

    Returns None if not authenticated instead of raising an exception.
    """
    token = None

    if credentials:
        token = credentials.credentials
    else:
        token = request.cookies.get("__session")

    if not token:
        return None

    try:
        return await clerk_auth.get_current_user(token)
    except HTTPException:
        return None


async def require_org_membership(
    user: ClerkUser = Depends(get_current_user),
) -> ClerkUser:
    """Require user to be a member of an organization.

    Args:
        user: Current authenticated user

    Returns:
        ClerkUser object

    Raises:
        HTTPException: If user is not in an organization
    """
    if not user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization membership required",
        )
    return user


async def require_org_admin(
    user: ClerkUser = Depends(require_org_membership),
) -> ClerkUser:
    """Require user to be an organization admin.

    Args:
        user: Current authenticated user

    Returns:
        ClerkUser object

    Raises:
        HTTPException: If user is not an admin
    """
    if user.org_role not in ["admin", "org:admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def get_org_id(user: ClerkUser = Depends(require_org_membership)) -> UUID:
    """Get organization ID from authenticated user.

    This is a convenience dependency for routes that need the org ID.
    """
    from app.models.database import get_async_session_factory
    # Note: This returns the Clerk org_id, which needs to be mapped
    # to the internal UUID in the organizations table
    return user.org_id
