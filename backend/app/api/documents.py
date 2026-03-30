"""Document upload and processing API endpoints."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.core.audit import get_audit_logger
from app.core.encryption import get_encryption_service
from app.core.file_validation import validate_magic_bytes
from app.core.rate_limit import limiter
from app.core.user_auth import get_current_user
from app.db.repositories import DocumentRepository
from app.models.schemas import (
    DocumentStatus,
    DocumentUploadResponse,
    OCREngine,
    OCRRequest,
    ProcessedDocument,
)
from app.services.document_processor import DocumentProcessor

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(get_current_user)],
)

_repo = DocumentRepository()
_processor = DocumentProcessor()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# ── Upload & process ──────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentUploadResponse)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    ocr_engine: OCREngine = Form(default=OCREngine.AUTO),
    enhance_images: bool = Form(default=True),
    languages: str = Form(default="en"),
    extract_tables: bool = Form(default=True),
):
    """
    Upload a document (PDF, DOCX, or image) and process it.

    - **file**: PDF, DOCX, PNG, JPEG, TIFF, BMP, WEBP
    - **ocr_engine**: `auto` (default), `easyocr`, `tesseract`
    - **enhance_images**: run image pre-processing for low-quality scans
    - **languages**: comma-separated ISO language codes (e.g. `en,de`)
    - **extract_tables**: detect and extract table structures

    The file is encrypted at rest with AES-256-GCM after processing.
    """
    audit = get_audit_logger()
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {settings.SUPPORTED_FORMATS}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_UPLOAD_SIZE_MB} MB",
        )

    # Validate actual file content against declared extension (blocks renamed executables)
    magic_ok, magic_msg = validate_magic_bytes(content, suffix)
    if not magic_ok:
        audit.log(
            event="UPLOAD_REJECTED",
            resource="document",
            client_ip=ip,
            user_agent=ua,
            detail={"filename": file.filename, "reason": "magic_bytes_mismatch"},
        )
        raise HTTPException(status_code=415, detail=magic_msg)

    doc_id = str(uuid.uuid4())
    upload_path = Path(settings.UPLOAD_DIR) / f"{doc_id}{suffix}"
    upload_path.write_bytes(content)

    lang_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    logger.info(f"Processing document {doc_id} ({file.filename})")
    doc = _processor.process(
        file_path=upload_path,
        document_id=doc_id,
        ocr_engine=ocr_engine,
        enhance_images=enhance_images,
        languages=lang_list,
        extract_tables=extract_tables,
    )

    # Encrypt the file at rest after processing
    enc_svc = get_encryption_service()
    if enc_svc and upload_path.exists():
        try:
            enc_path = enc_svc.encrypt_file(upload_path)
            audit.log(
                event="ENCRYPT",
                resource="document",
                resource_id=doc_id,
                filename=file.filename or "",
                client_ip=ip,
                user_agent=ua,
                detail={"enc_path": enc_path.name},
            )
        except Exception as exc:
            logger.error(f"Encryption failed for {doc_id}: {exc}")

    await _repo.save(doc)

    audit.log(
        event="UPLOAD",
        resource="document",
        resource_id=doc_id,
        filename=file.filename or "",
        client_ip=ip,
        user_agent=ua,
        outcome="SUCCESS" if doc.status == DocumentStatus.COMPLETED else "FAILURE",
        detail={
            "pages": doc.metadata.page_count,
            "size_mb": round(size_mb, 2),
            "ocr_engine": ocr_engine.value,
            "encrypted": enc_svc is not None,
        },
    )

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename or "",
        status=doc.status,
        message=(
            f"Processed {doc.metadata.page_count} page(s) in "
            f"{doc.metadata.processing_time_ms}ms"
            if doc.status == DocumentStatus.COMPLETED
            else f"Processing failed: {doc.error}"
        ),
    )


# ── Retrieve ──────────────────────────────────────────────────────────────────

@router.get("/{document_id}", response_model=ProcessedDocument)
async def get_document(document_id: str, request: Request):
    """Return the full processed document including extracted text and layout."""
    doc = await _repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    get_audit_logger().log(
        event="ACCESS",
        resource="document",
        resource_id=document_id,
        filename=doc.metadata.filename,
        client_ip=_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
    )
    return doc


@router.get("/{document_id}/text")
async def get_document_text(document_id: str, request: Request, page: Optional[int] = None):
    """Return extracted plain text, optionally filtered to a single page."""
    doc = await _repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    get_audit_logger().log(
        event="OCR",
        resource="document",
        resource_id=document_id,
        filename=doc.metadata.filename,
        client_ip=_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
        detail={"page": page},
    )

    if page is not None:
        pages = [p for p in doc.pages if p.page_number == page]
        if not pages:
            raise HTTPException(status_code=404, detail=f"Page {page} not found")
        text = pages[0].raw_text
    else:
        text = doc.full_text

    return {"document_id": document_id, "page": page, "text": text}


@router.get("/{document_id}/tables")
async def get_document_tables(document_id: str, request: Request):
    """Return all tables extracted from the document."""
    doc = await _repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    get_audit_logger().log(
        event="ACCESS",
        resource="document_tables",
        resource_id=document_id,
        filename=doc.metadata.filename,
        client_ip=_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
    )

    tables = []
    for page in doc.pages:
        for tbl in page.tables:
            tables.append({
                "page": page.page_number,
                "rows": tbl.rows,
                "cols": tbl.cols,
                "cells": [c.dict() for c in tbl.cells],
            })
    return {"document_id": document_id, "tables": tables}


@router.get("/")
async def list_documents(request: Request):
    """List all uploaded documents."""
    get_audit_logger().log(
        event="ACCESS",
        resource="document_list",
        client_ip=_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
    )
    docs = await _repo.list_all()
    return [
        {
            "document_id": d.document_id,
            "filename": d.metadata.filename,
            "page_count": d.metadata.page_count,
            "status": d.status.value,
            "created_at": d.metadata.created_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/{document_id}")
async def delete_document(document_id: str, request: Request):
    """Delete a document and its uploaded file (plaintext and encrypted)."""
    doc = await _repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    suffix = doc.metadata.file_type
    # Remove plaintext copy (may not exist if encrypted)
    plain_path = Path(settings.UPLOAD_DIR) / f"{document_id}{suffix}"
    if plain_path.exists():
        plain_path.unlink()

    # Remove encrypted copy
    enc_path = Path(settings.UPLOAD_DIR) / f"{document_id}{suffix}.enc"
    if enc_path.exists():
        enc_path.unlink()

    await _repo.delete(document_id)

    get_audit_logger().log(
        event="DELETE",
        resource="document",
        resource_id=document_id,
        filename=doc.metadata.filename,
        client_ip=_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
    )
    return {"message": f"Document {document_id} deleted"}


# ── Re-process ────────────────────────────────────────────────────────────────

@router.post("/{document_id}/reprocess", response_model=DocumentUploadResponse)
@limiter.limit("10/minute")
async def reprocess_document(document_id: str, request: Request, req: OCRRequest):
    """Re-run OCR/layout extraction with different settings."""
    doc = await _repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    suffix = doc.metadata.file_type
    upload_path = Path(settings.UPLOAD_DIR) / f"{document_id}{suffix}"
    enc_path = Path(settings.UPLOAD_DIR) / f"{document_id}{suffix}.enc"

    # If only encrypted copy exists, decrypt to temp for reprocessing
    tmp_path: Path | None = None
    enc_svc = get_encryption_service()
    if not upload_path.exists() and enc_path.exists() and enc_svc:
        tmp_path = enc_svc.decrypt_to_temp(enc_path)
        upload_path = tmp_path
        get_audit_logger().log(
            event="DECRYPT",
            resource="document",
            resource_id=document_id,
            filename=doc.metadata.filename,
            client_ip=_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
            detail={"reason": "reprocess"},
        )

    if not upload_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found on disk")

    try:
        new_doc = _processor.process(
            file_path=upload_path,
            document_id=document_id,
            ocr_engine=req.engine,
            enhance_images=req.enhance_image,
            languages=req.languages,
            extract_tables=req.extract_tables,
        )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()  # always delete plaintext temp copy

    await _repo.save(new_doc)
    get_audit_logger().log(
        event="OCR",
        resource="document",
        resource_id=document_id,
        filename=doc.metadata.filename,
        client_ip=_client_ip(request),
        user_agent=request.headers.get("user-agent", ""),
        detail={"action": "reprocess", "engine": req.engine.value},
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=doc.metadata.filename,
        status=new_doc.status,
        message=(
            f"Re-processed {new_doc.metadata.page_count} page(s)"
            if new_doc.status == DocumentStatus.COMPLETED
            else f"Failed: {new_doc.error}"
        ),
    )


# ── Public helper used by chat API ────────────────────────────────────────────

async def get_document_context(document_ids: list[str], max_chars: int = 12000) -> str:
    """Build a context string from multiple documents for LLM prompts."""
    parts = []
    remaining = max_chars
    for did in document_ids:
        doc = await _repo.get(did)
        if not doc or doc.status != DocumentStatus.COMPLETED:
            continue
        header = f"--- Document: {doc.metadata.filename} ---\n"
        body = doc.full_text[:remaining]
        parts.append(header + body)
        remaining -= len(body)
        if remaining <= 0:
            break
    return "\n\n".join(parts)
