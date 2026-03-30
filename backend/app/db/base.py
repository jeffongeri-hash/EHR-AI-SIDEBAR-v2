"""
Database Engine & Session Factory
===================================
Uses async SQLAlchemy so all DB calls are non-blocking.

The default DATABASE_URL is SQLite (aiosqlite) for local development.
In production swap to PostgreSQL:

    DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname

The same ORM models and queries work with both drivers.

Usage
-----
    from app.db.base import get_session

    async with get_session() as session:
        result = await session.execute(select(DocumentRow))
        docs = result.scalars().all()

Startup / shutdown
------------------
Call ``init_db()`` once at application startup (lifespan) to create all
tables that don't yet exist.  Alembic handles schema migrations in
production; ``init_db`` is the zero-config fallback for local dev.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# ── Engine ────────────────────────────────────────────────────────────────────

_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,          # log SQL in debug mode only
    future=True,
    # SQLite-specific: allow the same connection across async tasks
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    pool_pre_ping=True,           # detect stale connections
)

# ── Session factory ───────────────────────────────────────────────────────────

_SessionFactory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,       # keep objects usable after commit
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a session and commits / rolls back."""
    async with _SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Declarative base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Table creation (dev / test convenience) ───────────────────────────────────

async def init_db() -> None:
    """
    Create all tables that don't yet exist.

    Safe to call on every startup — it's a no-op if the tables already
    exist.  In production, use Alembic migrations instead.
    """
    # Import models so SQLAlchemy registers them on Base.metadata
    from app.db import models as _  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
