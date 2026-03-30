"""
Repository Layer
================
Thin async data-access objects (DAOs) for each domain entity.

All DB interaction goes through these classes so:
  - API handlers stay free of SQLAlchemy boilerplate.
  - Unit tests can mock a single boundary (the repository).
  - Switching from SQLite to PostgreSQL requires no changes here.

Each repository method handles its own session via get_session().
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, delete

from app.db.base import get_session
from app.db.models import ConversationRow, DocumentRow, FineTuneJobRow
from app.models.schemas import (
    ChatMessage,
    ConversationHistory,
    DocumentMetadata,
    DocumentPage,
    DocumentStatus,
    FineTuneJob,
    FineTuneStatus,
    MessageRole,
    ProcessedDocument,
    TrainingConfig,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(str(val))


# ── Document repository ───────────────────────────────────────────────────────

class DocumentRepository:
    """CRUD for ProcessedDocument records."""

    async def save(self, doc: ProcessedDocument) -> None:
        pages_data = [p.dict() for p in doc.pages]
        async with get_session() as session:
            existing = await session.get(DocumentRow, doc.document_id)
            if existing:
                existing.status = doc.status.value
                existing.full_text = doc.full_text
                existing.pages_json = pages_data
                existing.ocr_engine = doc.metadata.ocr_engine
                existing.processing_time_ms = doc.metadata.processing_time_ms
                existing.page_count = doc.metadata.page_count
                existing.error = doc.error
            else:
                row = DocumentRow(
                    document_id=doc.document_id,
                    filename=doc.metadata.filename,
                    file_size=doc.metadata.file_size,
                    file_type=doc.metadata.file_type,
                    page_count=doc.metadata.page_count,
                    ocr_engine=doc.metadata.ocr_engine,
                    processing_time_ms=doc.metadata.processing_time_ms,
                    status=doc.status.value,
                    error=doc.error,
                    full_text=doc.full_text,
                    pages_json=pages_data,
                )
                session.add(row)

    async def get(self, document_id: str) -> Optional[ProcessedDocument]:
        async with get_session() as session:
            row = await session.get(DocumentRow, document_id)
            return _row_to_doc(row) if row else None

    async def list_all(self) -> list[ProcessedDocument]:
        async with get_session() as session:
            result = await session.execute(select(DocumentRow))
            return [_row_to_doc(r) for r in result.scalars().all()]

    async def delete(self, document_id: str) -> bool:
        async with get_session() as session:
            row = await session.get(DocumentRow, document_id)
            if not row:
                return False
            await session.delete(row)
            return True


def _row_to_doc(row: DocumentRow) -> ProcessedDocument:
    pages = [DocumentPage(**p) for p in (row.pages_json or [])]
    return ProcessedDocument(
        document_id=row.document_id,
        metadata=DocumentMetadata(
            filename=row.filename,
            file_size=row.file_size,
            file_type=row.file_type,
            page_count=row.page_count,
            created_at=row.created_at,
            ocr_engine=row.ocr_engine,
            processing_time_ms=row.processing_time_ms,
        ),
        pages=pages,
        full_text=row.full_text,
        status=DocumentStatus(row.status),
        error=row.error,
    )


# ── Conversation repository ───────────────────────────────────────────────────

class ConversationRepository:
    """CRUD for ConversationHistory records."""

    async def save(self, history: ConversationHistory) -> None:
        msgs_data = []
        for m in history.messages:
            d = m.dict()
            # datetime → ISO string for JSON serialisation
            d["timestamp"] = d["timestamp"].isoformat() if d.get("timestamp") else None
            msgs_data.append(d)

        async with get_session() as session:
            existing = await session.get(ConversationRow, history.conversation_id)
            if existing:
                existing.messages_json = msgs_data
                existing.document_ids_json = history.document_ids
                existing.updated_at = datetime.now(timezone.utc)
            else:
                row = ConversationRow(
                    conversation_id=history.conversation_id,
                    messages_json=msgs_data,
                    document_ids_json=history.document_ids,
                    created_at=history.created_at,
                    updated_at=history.updated_at,
                )
                session.add(row)

    async def get(self, conversation_id: str) -> Optional[ConversationHistory]:
        async with get_session() as session:
            row = await session.get(ConversationRow, conversation_id)
            return _row_to_conv(row) if row else None

    async def delete(self, conversation_id: str) -> bool:
        async with get_session() as session:
            row = await session.get(ConversationRow, conversation_id)
            if not row:
                return False
            await session.delete(row)
            return True


def _row_to_conv(row: ConversationRow) -> ConversationHistory:
    messages = []
    for m in (row.messages_json or []):
        ts_raw = m.get("timestamp")
        ts = _parse_dt(ts_raw) or datetime.now(timezone.utc)
        messages.append(ChatMessage(
            role=MessageRole(m["role"]),
            content=m["content"],
            timestamp=ts,
            document_ids=m.get("document_ids"),
            metadata=m.get("metadata"),
        ))
    return ConversationHistory(
        conversation_id=row.conversation_id,
        messages=messages,
        document_ids=row.document_ids_json or [],
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ── Fine-tune job repository ──────────────────────────────────────────────────

class FineTuneJobRepository:
    """CRUD for FineTuneJob records."""

    async def save(self, job: FineTuneJob) -> None:
        async with get_session() as session:
            existing = await session.get(FineTuneJobRow, job.job_id)
            if existing:
                existing.status = job.status.value
                existing.progress = job.progress
                existing.current_epoch = job.current_epoch
                existing.current_loss = job.current_loss
                existing.error = job.error
                existing.output_path = job.output_path
                existing.logs_json = job.logs
                existing.started_at = job.started_at
                existing.completed_at = job.completed_at
            else:
                row = FineTuneJobRow(
                    job_id=job.job_id,
                    config_json=job.config.dict(),
                    status=job.status.value,
                    progress=job.progress,
                    current_epoch=job.current_epoch,
                    current_loss=job.current_loss,
                    error=job.error,
                    output_path=job.output_path,
                    logs_json=job.logs,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
                session.add(row)

    async def get(self, job_id: str) -> Optional[FineTuneJob]:
        async with get_session() as session:
            row = await session.get(FineTuneJobRow, job_id)
            return _row_to_job(row) if row else None

    async def list_all(self) -> list[FineTuneJob]:
        async with get_session() as session:
            result = await session.execute(select(FineTuneJobRow))
            return [_row_to_job(r) for r in result.scalars().all()]

    async def delete(self, job_id: str) -> bool:
        async with get_session() as session:
            row = await session.get(FineTuneJobRow, job_id)
            if not row:
                return False
            await session.delete(row)
            return True


def _row_to_job(row: FineTuneJobRow) -> FineTuneJob:
    return FineTuneJob(
        job_id=row.job_id,
        config=TrainingConfig(**row.config_json),
        status=FineTuneStatus(row.status),
        progress=row.progress,
        current_epoch=row.current_epoch,
        current_loss=row.current_loss,
        error=row.error,
        output_path=row.output_path,
        logs=row.logs_json or [],
        created_at=row.created_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
    )
