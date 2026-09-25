"""JWT bearer authentication for incoming requests.

The C# InvoiceGen backend issues its users JWTs signed with HMAC-SHA256 (HS256)
using a symmetric key. This service validates that same token on protected
endpoints: it checks the signature, expiry, issuer, and audience, then exposes
the authenticated user id (the ``sub`` claim) to the endpoint.

Usage::

    @app.post("/search")
    async def search(request: SearchRequest, user_id: str = Depends(require_user)):
        ...
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import Settings, get_settings

logger = logging.getLogger("invoice_search")

# auto_error=False so a missing/malformed Authorization header yields our own
# 401 (below) rather than HTTPBearer's default 403.
_bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing or invalid authentication token",
    headers={"WWW-Authenticate": "Bearer"},
)


async def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),  # noqa: B008 - FastAPI dependency-injection idiom
    settings: Settings = Depends(get_settings),  # noqa: B008 - FastAPI dependency-injection idiom
) -> str:
    """Validate the bearer JWT and return the authenticated user id.

    Raises ``401 Unauthorized`` if the token is absent, malformed, expired, or
    fails signature / issuer / audience validation. On success returns the value
    of the ``sub`` claim.
    """
    if credentials is None or not credentials.credentials:
        raise _UNAUTHORIZED

    if not settings.jwt_secret_key:
        # Fail closed: without a configured key we cannot verify signatures, so
        # refuse rather than accept unverified tokens.
        logger.error("JWT_SECRET_KEY is not configured; rejecting request")
        raise _UNAUTHORIZED

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=["HS256"],
            issuer=settings.jwt_issuer or None,
            audience=settings.jwt_audience or None,
        )
    except JWTError as exc:
        logger.info("JWT validation failed: %s", exc)
        raise _UNAUTHORIZED from exc

    user_id = payload.get("sub")
    if not user_id:
        logger.info("JWT is valid but has no 'sub' claim")
        raise _UNAUTHORIZED

    return str(user_id)
