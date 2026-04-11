"""
Rate Limiting — DoS / Resource-Exhaustion Protection
======================================================
Uses slowapi (Starlette/FastAPI adapter for limits).

Per-endpoint limits apply **only to Anthropic/Claude calls** for chat routes,
because local models (Ollama, HuggingFace, TinyLlama, medical models) run on
the user's own hardware and have no external API cost.

  upload         10 req/min   — OCR is CPU/GPU intensive
  chat (Claude)  30 req/min   — Anthropic token spend
  stream (Claude) 20 req/min  — Anthropic token spend (streaming)
  fine-tune POST  3 req/min   — spawns GPU job; primary DoS vector
  audit read     20 req/min   — PHI-adjacent; throttle bulk export
  default        60 req/min   — all other authenticated endpoints

Key function is the client IP extracted from:
  1. X-Forwarded-For (set by reverse proxy / load balancer)
  2. REMOTE_ADDR fallback for direct connections
"""

from __future__ import annotations

from fastapi import Request
from limits import parse as _parse
from limits import storage as _storage
from limits import strategies as _strategies
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Singleton limiter — imported by routers and registered on the app
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    # When Redis is available swap storage_uri to Redis for multi-instance:
    # storage_uri="redis://localhost:6379",
)

# ── Conditional Claude-only rate limiter ──────────────────────────────────────
# These use the same `limits` library internals as slowapi but are called
# programmatically so we can skip them when the user chose a local model.

_mem = _storage.MemoryStorage()
_window = _strategies.FixedWindowRateLimiter(_mem)
_CLAUDE_CHAT_LIMIT = _parse("30/minute")
_CLAUDE_STREAM_LIMIT = _parse("20/minute")

_CLAUDE_PROVIDERS = {"claude", "anthropic"}


def check_claude_chat_limit(request: Request, model_provider: str) -> None:
    """
    Raise HTTP 429 only when the caller is using a Claude/Anthropic model.
    Local Ollama, HuggingFace, TinyLlama, and medical models are unrestricted.
    """
    if model_provider.lower() not in _CLAUDE_PROVIDERS:
        return
    key = f"claude:chat:{get_remote_address(request)}"
    if not _window.hit(_CLAUDE_CHAT_LIMIT, key):
        raise RateLimitExceeded(_CLAUDE_CHAT_LIMIT)


def check_claude_stream_limit(request: Request, model_provider: str) -> None:
    """Same as check_claude_chat_limit but for the streaming endpoint."""
    if model_provider.lower() not in _CLAUDE_PROVIDERS:
        return
    key = f"claude:stream:{get_remote_address(request)}"
    if not _window.hit(_CLAUDE_STREAM_LIMIT, key):
        raise RateLimitExceeded(_CLAUDE_STREAM_LIMIT)
