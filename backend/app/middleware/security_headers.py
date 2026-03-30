"""
Security Headers Middleware
=============================
HIPAA §164.312(c)(1) — Integrity Controls
HIPAA §164.312(e)(1) — Transmission Security

Applies HTTP security headers to every response to defend against
common web attack vectors.  These headers are especially important
for applications that handle Protected Health Information (PHI).
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects HIPAA/security-hardening HTTP headers on every response.

    Header rationale
    ----------------
    Strict-Transport-Security
        Forces all future requests over HTTPS for 1 year (HIPAA transmission
        security).  Disabled on /api/docs* to avoid breaking dev Swagger UI
        over plain HTTP.

    X-Content-Type-Options: nosniff
        Prevents browsers from MIME-sniffing a response away from the
        declared Content-Type, blocking certain XSS vectors.

    X-Frame-Options: DENY
        Prohibits the app being loaded in an iframe — prevents clickjacking.

    X-XSS-Protection: 1; mode=block
        Activates older browsers' built-in XSS filter as a defence-in-depth
        measure (modern browsers rely on CSP instead).

    Referrer-Policy: strict-origin-when-cross-origin
        Sends the full URL only for same-origin requests; only the origin
        for cross-origin.  Stops PHI appearing in Referer headers sent to
        third-party services.

    Cache-Control: no-store
        Ensures PHI API responses are never stored in browser or proxy
        caches.  Applied to /api/* routes only to avoid breaking static
        asset caching.

    Permissions-Policy
        Disables browser features (camera, microphone, geolocation) that
        are not needed and could be exploited to access PHI.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        path = request.url.path

        # HSTS — only meaningful over HTTPS; skip Swagger UI to allow dev
        if not path.startswith("/api/docs") and not path.startswith("/api/redoc"):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )

        # No caching for API responses that may contain PHI
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"

        return response
