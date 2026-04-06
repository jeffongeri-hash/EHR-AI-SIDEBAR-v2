"""
EHR AI Sidebar – FastAPI Backend

Entry point for the application.
Run with:  uvicorn main:app --reload --port 8000
"""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api import chat
from app.api import documents
from app.api.audit_log import router as audit_router
from app.api.auth import router as auth_router
from app.api.routes.finetune import router as finetune_router
from app.config import settings
from app.core.rate_limit import limiter
from app.db.base import init_db
from app.middleware.security_headers import SecurityHeadersMiddleware

_INSECURE_SECRET = "change-this-secret-key-in-production"


def _validate_startup_security() -> None:
    """
    Enforce safe production defaults at startup.
    Logs warnings in DEBUG mode; hard-exits in production so misconfigured
    deployments never handle real PHI.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Secret key must be changed from the placeholder
    if settings.SECRET_KEY == _INSECURE_SECRET:
        msg = "SECRET_KEY is still the default placeholder — set a strong random value in .env"
        if settings.DEBUG:
            warnings.append(msg)
        else:
            errors.append(msg)

    # 2. Encryption enabled but no key supplied → cannot encrypt anything
    if settings.ENCRYPT_UPLOADS and not settings.EHR_ENCRYPTION_KEY:
        msg = (
            "ENCRYPT_UPLOADS=true but EHR_ENCRYPTION_KEY is not set. "
            "Generate one with: python -c \"import secrets,base64; "
            "print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())\""
        )
        if settings.DEBUG:
            warnings.append(msg)
        else:
            errors.append(msg)

    # 3. No API key means all endpoints are open — acceptable only in dev
    if not settings.EHR_API_KEY:
        warnings.append(
            "EHR_API_KEY is not set — all API endpoints are unauthenticated. "
            "This is acceptable for local development only."
        )

    for w in warnings:
        logger.warning(f"[SECURITY] {w}")

    if errors:
        for e in errors:
            logger.critical(f"[SECURITY] {e}")
        logger.critical(
            "Refusing to start: insecure configuration detected in production mode. "
            "Fix the above errors or set DEBUG=true for local development."
        )
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_startup_security()

    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Models directory: {settings.MODELS_DIR}")

    # Initialise database tables (no-op if already exist)
    await init_db()
    logger.info(f"Database ready: {settings.DATABASE_URL}")

    # Log HIPAA security posture at startup
    enc_on = settings.ENCRYPT_UPLOADS
    api_key_set = bool(settings.EHR_API_KEY)
    audit_path = settings.AUDIT_LOG_PATH

    logger.info(f"Encryption at rest:  {'ENABLED' if enc_on else 'DISABLED'}")
    logger.info(f"API key auth:        {'ENABLED' if api_key_set else 'DISABLED (dev only)'}")
    logger.info(f"HIPAA audit log:     {audit_path}")

    yield
    logger.info("Shutting down…")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-powered EHR sidebar with document OCR, layout extraction, "
        "multi-model chat, and LLM fine-tuning. "
        "HIPAA-aligned: AES-256-GCM encryption at rest, audit logging, API key auth."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── Rate limiting ─────────────────────────────────────────────────────────────

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Middleware (order matters: outermost runs last on response) ───────────────

app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SecurityHeadersMiddleware)        # HIPAA security headers
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)

# ── API Routers ───────────────────────────────────────────────────────────────

app.include_router(auth_router,     prefix="/api")
app.include_router(chat.router,     prefix="/api")
app.include_router(documents.router,prefix="/api")
app.include_router(finetune_router)               # prefix="/api/finetune" set in router
app.include_router(audit_router,    prefix="/api")


# ── Health & info ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/api/info")
async def info():
    enc_on = getattr(settings, "ENCRYPT_UPLOADS", False)
    api_key_set = bool(getattr(settings, "EHR_API_KEY", None))
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "claude_model": settings.CLAUDE_MODEL,
        "ollama_url": settings.OLLAMA_BASE_URL,
        "security": {
            "encryption_at_rest": enc_on,
            "api_key_auth": api_key_set,
            "audit_log": getattr(settings, "AUDIT_LOG_PATH", "./audit.jsonl"),
            "tls_headers": True,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
