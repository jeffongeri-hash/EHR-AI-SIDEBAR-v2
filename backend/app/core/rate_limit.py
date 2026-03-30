"""
Rate Limiting — DoS / Resource-Exhaustion Protection
======================================================
Uses slowapi (Starlette/FastAPI adapter for limits).

Per-endpoint limits are deliberately conservative for healthcare:

  upload         10 req/min   — OCR is CPU/GPU intensive
  chat           30 req/min   — LLM token spend
  stream         20 req/min   — LLM token spend (streaming)
  fine-tune POST  3 req/min   — spawns GPU job; primary DoS vector
  audit read     20 req/min   — PHI-adjacent; throttle bulk export
  default        60 req/min   — all other authenticated endpoints

Key function is the client IP extracted from:
  1. X-Forwarded-For (set by reverse proxy / load balancer)
  2. REMOTE_ADDR fallback for direct connections

In production, deploy behind nginx/ALB and ensure X-Forwarded-For
is set correctly.  If you trust a proxy, also set TRUSTED_PROXIES
in your nginx config so the real client IP is forwarded.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# Singleton limiter — imported by routers and registered on the app
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    # When Redis is available swap storage_uri to Redis for multi-instance:
    # storage_uri="redis://localhost:6379",
)
