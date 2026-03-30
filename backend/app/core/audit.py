"""
HIPAA Audit Logging Service
=============================
HIPAA §164.312(b)   — Audit Controls (Required)
HIPAA §164.308(a)(1)(ii)(D) — Information System Activity Review

Every access, creation, modification, or deletion of PHI must be logged
with sufficient detail to support security incident investigation and
compliance review.

Log format: JSONL (one JSON object per line, append-only)
Each entry includes a SHA-256 hash of the preceding entry, forming a
tamper-evident hash chain.  Any modification of an earlier entry will
cause all subsequent hash checks to fail.

Event types (HIPAA minimum required categories):
  UPLOAD     — PHI document ingested into the system
  ACCESS     — PHI document retrieved / read
  DELETE     — PHI document removed
  OCR        — OCR/text extraction performed on PHI document
  CHAT       — LLM query made with PHI document context
  FINETUNE   — Model training job started with PHI data
  AUTH_OK    — Successful API authentication
  AUTH_FAIL  — Failed authentication attempt
  ENCRYPT    — File encrypted at rest
  DECRYPT    — File decrypted for processing
  EXPORT     — Audit log exported
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

GENESIS_HASH = "0" * 64   # sentinel hash for the very first log entry

# Default maximum log file size before rotation (bytes).  Override via config.
_DEFAULT_MAX_BYTES = 50 * 1024 * 1024   # 50 MB

# ── Per-request user context ───────────────────────────────────────────────────
# Set this at authentication time; the audit logger reads it automatically so
# every log entry carries user identity without modifying each call site.

_audit_user_id: ContextVar[str] = ContextVar("audit_user_id", default="anonymous")
_audit_username: ContextVar[str] = ContextVar("audit_username", default="anonymous")


def set_audit_user(user_id: str, username: str) -> None:
    """Call once per request after authentication to bind user identity to the log."""
    _audit_user_id.set(user_id)
    _audit_username.set(username)


# ── Core logger ───────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Thread-safe, append-only HIPAA audit logger.

    Each log entry is a JSON object:
    {
        "ts":           "2026-03-24T12:00:00.000Z",   ISO-8601 UTC
        "event":        "UPLOAD",
        "resource":     "document",
        "resource_id":  "<uuid>",
        "filename":     "patient_record.pdf",          (if applicable)
        "client_ip":    "10.0.0.1",
        "user_agent":   "Mozilla/5.0 ...",
        "outcome":      "SUCCESS" | "FAILURE",
        "detail":       { ... extra structured data },
        "prev_hash":    "<sha256 of previous line>",
        "entry_hash":   "<sha256 of this line minus entry_hash field>"
    }
    """

    def __init__(
        self,
        log_path: str | Path,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._prev_hash = self._read_last_hash()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rotate_if_needed(self) -> None:
        """
        Rotate the log file if it exceeds max_bytes.

        The current file is renamed to  <stem>.<YYYY-MM-DDTHH-MM-SS>.<ext>
        and a fresh file is started.  The hash chain resets to GENESIS_HASH
        for the new segment — this is intentional and documented.

        HIPAA requires retention of rotated files for 6 years.
        Configure log shipping (e.g. to S3, CloudWatch, or a SIEM) to
        automatically archive rotated files off-system.
        """
        if not self._path.exists():
            return
        if self._path.stat().st_size < self._max_bytes:
            return

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        rotated = self._path.with_stem(f"{self._path.stem}.{ts}")
        self._path.rename(rotated)
        logger.info(f"[audit] Rotated log → {rotated}")
        self._prev_hash = GENESIS_HASH

    def _read_last_hash(self) -> str:
        """Read the entry_hash of the last line in the log file."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return GENESIS_HASH
        try:
            with open(self._path, "rb") as f:
                # Seek to last non-empty line
                f.seek(0, 2)
                size = f.tell()
                buf = b""
                pos = size - 1
                while pos >= 0:
                    f.seek(pos)
                    ch = f.read(1)
                    if ch == b"\n" and buf:
                        break
                    buf = ch + buf
                    pos -= 1
                if not buf.strip():
                    return GENESIS_HASH
                entry = json.loads(buf.decode("utf-8"))
                return entry.get("entry_hash", GENESIS_HASH)
        except Exception as exc:
            logger.warning(f"Could not read last audit hash: {exc}")
            return GENESIS_HASH

    @staticmethod
    def _hash(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        event: str,
        resource: str = "",
        resource_id: str = "",
        filename: str = "",
        client_ip: str = "unknown",
        user_agent: str = "",
        outcome: str = "SUCCESS",
        detail: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append one audit entry to the log file."""
        # Pull user identity from the per-request context variable
        user_id = _audit_user_id.get()
        username = _audit_username.get()

        with self._lock:
            # Rotate before writing if the file is too large
            self._rotate_if_needed()

            entry: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "event": event,
                # User identity — HIPAA §164.312(b) requires tracking who
                "user_id": user_id,
                "username": username,
                "resource": resource,
                "resource_id": resource_id,
                "filename": filename,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "outcome": outcome,
                "detail": detail or {},
                "prev_hash": self._prev_hash,
            }
            # Hash everything except entry_hash itself
            raw = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            entry["entry_hash"] = self._hash(raw)

            line = json.dumps(entry, separators=(",", ":")) + "\n"
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()  # ensure the write is flushed to OS buffers immediately

            self._prev_hash = entry["entry_hash"]

    def read_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        event_filter: Optional[str] = None,
    ) -> list[dict]:
        """Return parsed log entries (most recent first)."""
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception:
            return []

        entries = []
        for line in reversed(lines):
            try:
                e = json.loads(line)
                if event_filter and e.get("event") != event_filter:
                    continue
                entries.append(e)
            except Exception:
                continue

        return entries[offset: offset + limit]

    def verify_chain(self) -> tuple[bool, str]:
        """
        Walk the log and verify every hash link.
        Returns (True, "OK") or (False, "<reason>").
        """
        if not self._path.exists():
            return True, "OK (empty log)"
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        except Exception as exc:
            return False, str(exc)

        prev = GENESIS_HASH
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except Exception:
                return False, f"Line {i+1}: invalid JSON"

            if entry.get("prev_hash") != prev:
                return False, f"Line {i+1}: prev_hash mismatch (chain broken)"

            stored_hash = entry.pop("entry_hash", "")
            raw = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            expected = self._hash(raw)
            if stored_hash != expected:
                return False, f"Line {i+1}: entry_hash mismatch (tampered)"

            prev = stored_hash

        return True, f"OK ({len(lines)} entries verified)"


# ── Module singleton ──────────────────────────────────────────────────────────

_audit: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Return the module-level audit logger (lazy init from config)."""
    global _audit
    if _audit is None:
        from app.config import settings
        path = getattr(settings, "AUDIT_LOG_PATH", "./audit.jsonl")
        _audit = AuditLogger(path)
        logger.info(f"HIPAA audit log → {path}")
    return _audit
