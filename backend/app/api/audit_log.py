"""
HIPAA Audit Log API
=====================
Exposes read-only access to the audit trail for compliance review,
security incident investigation, and administrative oversight.

HIPAA §164.308(a)(1)(ii)(D) — Information System Activity Review
"""

from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
import csv
import io
import json

from app.core.audit import get_audit_logger
from app.core.rate_limit import limiter
from app.core.user_auth import get_current_user, require_role

router = APIRouter(
    prefix="/audit",
    tags=["Audit Log"],
    # Audit log is admin-only — clinicians and researchers should not export
    dependencies=[Depends(require_role("admin"))],
)


@router.get("/logs")
@limiter.limit("20/minute")
async def list_audit_logs(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    event: Optional[str] = Query(default=None, description="Filter by event type, e.g. UPLOAD, ACCESS, AUTH_FAIL"),
):
    """
    Return recent audit log entries (most recent first).

    Supports filtering by event type and pagination.
    """
    entries = get_audit_logger().read_entries(
        limit=limit,
        offset=offset,
        event_filter=event,
    )
    return {
        "total_returned": len(entries),
        "offset": offset,
        "limit": limit,
        "entries": entries,
    }


@router.get("/logs/export")
@limiter.limit("5/minute")
async def export_audit_logs(
    request: Request,
    fmt: str = Query(default="json", pattern="^(json|csv)$"),
):
    """
    Export the full audit log as JSON or CSV for compliance reporting.
    Downloads as a file attachment.
    """
    audit = get_audit_logger()
    entries = audit.read_entries(limit=100_000)  # effectively unlimited
    audit.log(event="EXPORT", detail={"format": fmt, "entries_exported": len(entries)})

    if fmt == "csv":
        output = io.StringIO()
        if entries:
            writer = csv.DictWriter(output, fieldnames=entries[0].keys())
            writer.writeheader()
            for e in entries:
                # Flatten detail dict to a string
                e2 = {**e, "detail": json.dumps(e.get("detail", {}))}
                writer.writerow(e2)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=ehr_audit_log.csv"},
        )

    # JSON export
    return StreamingResponse(
        iter([json.dumps(entries, indent=2)]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=ehr_audit_log.json"},
    )


@router.get("/verify")
@limiter.limit("10/minute")
async def verify_audit_chain(request: Request):
    """
    Verify the tamper-evident hash chain of the audit log.

    A FAILURE result means at least one entry has been modified or
    deleted after it was written — indicating a potential security incident.
    """
    ok, message = get_audit_logger().verify_chain()
    return {
        "integrity": "PASS" if ok else "FAIL",
        "message": message,
    }
