"""
SQLAlchemy ORM Models
=====================
Maps the three primary domain objects to database tables.

Complex nested structures (pages, messages, config, etc.) are stored as
JSON columns.  This avoids over-normalisation for the current scale while
keeping all data queryable via standard SQL for audit / compliance work.

Tables
------
documents           — uploaded / processed documents
conversations       — chat conversation histories
fine_tune_jobs      — model fine-tuning job records
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Users ────────────────────────────────────────────────────────────────────

class UserRow(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(256), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)

    # Role-based access: admin | clinician | researcher
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="clinician")

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentRow(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    ocr_engine: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Full extracted text (stored separately from pages for fast retrieval)
    full_text: Mapped[str] = mapped_column(Text, default="")

    # JSON blob: list[DocumentPage] (text_blocks, tables, raw_text per page)
    pages_json: Mapped[dict] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


# ── Conversations ─────────────────────────────────────────────────────────────

class ConversationRow(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(36), primary_key=True)

    # JSON blob: list[ChatMessage] (role, content, timestamp)
    messages_json: Mapped[list] = mapped_column(JSON, default=list)

    # JSON blob: list[str] document_ids referenced in this conversation
    document_ids_json: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# ── Fine-tune jobs ────────────────────────────────────────────────────────────

class FineTuneJobRow(Base):
    __tablename__ = "fine_tune_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)

    # JSON blob: TrainingConfig dict
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    current_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # JSON blob: list[str] log lines
    logs_json: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
