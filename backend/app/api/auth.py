"""
Authentication API
==================
Endpoints for user registration, login, and profile.

POST /api/auth/register   — create account
POST /api/auth/login      — obtain JWT access token
GET  /api/auth/me         — return current user profile
"""

from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.core.audit import get_audit_logger
from app.core.user_auth import (
    AuthenticatedUser,
    create_access_token,
    create_user,
    get_current_user,
    get_user_by_username,
    update_last_login,
    verify_password,
)
from app.core.rate_limit import limiter
from fastapi import Request

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Request / response schemas ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: str = Field(default="clinician", pattern=r"^(admin|clinician|researcher)$")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user_id: str
    username: str
    role: str


class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
@limiter.limit("5/minute")
async def register(request: Request, body: RegisterRequest):
    """
    Register a new user account.

    Only an admin can create accounts with the **admin** role.
    In production, restrict this endpoint to admin-only or to an invite flow.
    """
    try:
        user = await create_user(
            username=body.username,
            email=body.email,
            password=body.password,
            role=body.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    get_audit_logger().log(
        event="USER_CREATED",
        resource="user",
        resource_id=user.user_id,
        detail={"username": user.username, "role": user.role},
    )
    return {"user_id": user.user_id, "username": user.username, "role": user.role}


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, body: LoginRequest):
    """
    Authenticate with username + password.
    Returns a short-lived JWT access token.
    """
    from app.config import settings

    user = await get_user_by_username(body.username)
    if user is None or not verify_password(body.password, user.hashed_password):
        get_audit_logger().log(
            event="AUTH_FAIL",
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", ""),
            outcome="FAILURE",
            detail={"reason": "invalid credentials", "username": body.username},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact your administrator.",
        )

    token = create_access_token(
        user_id=user.user_id,
        username=user.username,
        role=user.role,
    )
    await update_last_login(user.user_id)

    get_audit_logger().log(
        event="AUTH_OK",
        resource="user",
        resource_id=user.user_id,
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", ""),
        outcome="SUCCESS",
        detail={"method": "password", "username": user.username},
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=user.user_id,
        username=user.username,
        role=user.role,
    )


@router.get("/me", response_model=UserProfile)
async def get_me(current_user: AuthenticatedUser = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    from app.core.user_auth import get_user_by_id

    user = await get_user_by_id(current_user.user_id)
    if user is None:
        # Service accounts won't have a DB row
        return UserProfile(
            user_id=current_user.user_id,
            username=current_user.username,
            email="",
            role=current_user.role,
            is_active=True,
            created_at="",
            last_login=None,
        )

    return UserProfile(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )
