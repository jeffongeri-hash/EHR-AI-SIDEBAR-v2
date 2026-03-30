"""
API Key Authentication
========================
HIPAA §164.312(d) — Person or Entity Authentication (Required)

All API endpoints that touch PHI are protected by an API key passed in
the  X-API-Key  request header.  Key comparison uses a constant-time
algorithm to prevent timing-based enumeration attacks.

When EHR_API_KEY is not set (development mode) all requests are allowed
through with a warning logged once at startup.
"""

from __future__ import annotations
from typing import Optional

import secrets

from fastapi import Header, HTTPException, Request, status
from loguru import logger

from app.core.audit import get_audit_logger

_warned_no_key = False


def _get_configured_key() -> Optional[str]:
    from app.config import settings  # deferred to avoid circular import
    return getattr(settings, "EHR_API_KEY", None) or None


async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    FastAPI dependency.  Inject into any route that handles PHI:

        @router.get("/documents/{id}", dependencies=[Depends(require_api_key)])
    """
    global _warned_no_key
    audit = get_audit_logger()
    client_ip = (request.client.host if request.client else "unknown")
    ua = request.headers.get("user-agent", "")

    configured = _get_configured_key()

    # Dev-mode: no key configured → allow all, warn once
    if configured is None:
        if not _warned_no_key:
            logger.warning(
                "EHR_API_KEY is not set — all API requests are unauthenticated. "
                "Set EHR_API_KEY in .env before handling real PHI."
            )
            _warned_no_key = True
        return

    # Production: key must be present and correct
    if x_api_key is None:
        audit.log(
            event="AUTH_FAIL",
            client_ip=client_ip,
            user_agent=ua,
            outcome="FAILURE",
            detail={"reason": "missing X-API-Key header"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not secrets.compare_digest(configured, x_api_key):
        audit.log(
            event="AUTH_FAIL",
            client_ip=client_ip,
            user_agent=ua,
            outcome="FAILURE",
            detail={"reason": "invalid key"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    audit.log(
        event="AUTH_OK",
        client_ip=client_ip,
        user_agent=ua,
        outcome="SUCCESS",
    )
