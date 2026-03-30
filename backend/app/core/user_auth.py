"""
User Authentication — JWT + bcrypt
====================================
HIPAA §164.312(d) — Person or Entity Authentication (Required)

Provides:
  - Password hashing / verification (bcrypt via passlib)
  - JWT access token creation and validation (python-jose)
  - FastAPI dependency get_current_user() that accepts:
      1. Authorization: Bearer <jwt>  (primary — human users)
      2. X-API-Key: <key>             (secondary — service accounts / CI)

User roles
----------
admin       Full access including user management and audit export
clinician   Read / write own documents + chat
researcher  Read-only documents; can create fine-tune jobs

Tokens
------
Access tokens expire after settings.ACCESS_TOKEN_EXPIRE_MINUTES (default 30).
No refresh tokens are implemented yet (add for production).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from sqlalchemy import select

from app.config import settings
from app.core.audit import get_audit_logger, set_audit_user
from app.db.base import get_session
from app.db.models import UserRow

# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(user_id: str, username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises JWTError on invalid / expired token."""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])


# ── User repository helpers (inline to keep auth self-contained) ──────────────

async def get_user_by_username(username: str) -> Optional[UserRow]:
    async with get_session() as session:
        result = await session.execute(
            select(UserRow).where(UserRow.username == username)
        )
        return result.scalar_one_or_none()


async def get_user_by_id(user_id: str) -> Optional[UserRow]:
    async with get_session() as session:
        return await session.get(UserRow, user_id)


async def create_user(
    username: str, email: str, password: str, role: str = "clinician"
) -> UserRow:
    async with get_session() as session:
        # Check for duplicates
        dup = await session.execute(
            select(UserRow).where(
                (UserRow.username == username) | (UserRow.email == email)
            )
        )
        if dup.scalar_one_or_none():
            raise ValueError("Username or email already registered")

        user = UserRow(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            hashed_password=hash_password(password),
            role=role,
        )
        session.add(user)
        return user


async def update_last_login(user_id: str) -> None:
    async with get_session() as session:
        user = await session.get(UserRow, user_id)
        if user:
            user.last_login = datetime.now(timezone.utc)


# ── FastAPI dependency — unified auth ─────────────────────────────────────────

class AuthenticatedUser:
    """Lightweight user context attached to every authenticated request."""

    def __init__(
        self,
        user_id: str,
        username: str,
        role: str,
        auth_method: str,  # "jwt" | "api_key"
    ):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.auth_method = auth_method

    def __repr__(self) -> str:
        return f"<AuthenticatedUser {self.username} role={self.role}>"


# Sentinel for requests that pass the old-style API key (no user identity)
_API_KEY_USER = AuthenticatedUser(
    user_id="service-account",
    username="service-account",
    role="admin",
    auth_method="api_key",
)

_warned_no_key = False


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> AuthenticatedUser:
    """
    FastAPI dependency — accepts JWT Bearer token OR legacy X-API-Key.

    Priority:
      1. Authorization: Bearer <jwt>  → fully identified user
      2. X-API-Key: <key>             → service account (backward compat)
      3. Neither set in dev mode      → unauthenticated (warning logged once)
    """
    global _warned_no_key
    audit = get_audit_logger()
    client_ip = request.client.host if request.client else "unknown"
    ua = request.headers.get("user-agent", "")

    # ── 1. JWT bearer ─────────────────────────────────────────────────────────
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
        try:
            payload = decode_access_token(token)
            user_id: str = payload["sub"]
            username: str = payload["username"]
            role: str = payload["role"]
            # Bind identity to audit context for this request
            set_audit_user(user_id, username)
            audit.log(
                event="AUTH_OK",
                client_ip=client_ip,
                user_agent=ua,
                outcome="SUCCESS",
                detail={"method": "jwt", "user_id": user_id, "username": username},
            )
            return AuthenticatedUser(
                user_id=user_id,
                username=username,
                role=role,
                auth_method="jwt",
            )
        except JWTError as exc:
            audit.log(
                event="AUTH_FAIL",
                client_ip=client_ip,
                user_agent=ua,
                outcome="FAILURE",
                detail={"method": "jwt", "reason": str(exc)},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ── 2. Legacy API key ─────────────────────────────────────────────────────
    configured_key = getattr(settings, "EHR_API_KEY", None) or None
    if configured_key is not None:
        import secrets as _secrets

        if x_api_key is None:
            audit.log(
                event="AUTH_FAIL",
                client_ip=client_ip,
                user_agent=ua,
                outcome="FAILURE",
                detail={"method": "api_key", "reason": "missing header"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization: Bearer <token> or X-API-Key header required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not _secrets.compare_digest(configured_key, x_api_key):
            audit.log(
                event="AUTH_FAIL",
                client_ip=client_ip,
                user_agent=ua,
                outcome="FAILURE",
                detail={"method": "api_key", "reason": "invalid key"},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        set_audit_user("service-account", "service-account")
        audit.log(
            event="AUTH_OK",
            client_ip=client_ip,
            user_agent=ua,
            outcome="SUCCESS",
            detail={"method": "api_key"},
        )
        return _API_KEY_USER

    # ── 3. Dev mode — no auth configured ─────────────────────────────────────
    if not _warned_no_key:
        logger.warning(
            "No auth configured (no EHR_API_KEY and no Bearer token). "
            "All requests pass unauthenticated — dev mode only."
        )
        _warned_no_key = True

    return AuthenticatedUser(
        user_id="anonymous",
        username="anonymous",
        role="admin",
        auth_method="none",
    )


def require_role(*roles: str):
    """
    Role-gating dependency factory.

    Usage:
        @router.delete("/admin/users/{id}", dependencies=[Depends(require_role("admin"))])
    """
    async def _check(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires one of these roles: {list(roles)}",
            )
        return user
    return _check
