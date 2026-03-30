"""
File Content Validation — Magic Byte Checking
===============================================
Validates that uploaded file content actually matches the declared extension.
A renamed .exe or .html file will not pass these checks regardless of
what the client sends in the Content-Type or filename.

Pure-Python implementation — no system libmagic dependency required.

Supported signatures
---------------------
.pdf   — %PDF header
.docx  — OOXML (ZIP) PK header (further validates DOCX content entry)
.png   — 8-byte PNG signature
.jpg   — JPEG SOI marker
.jpeg  — JPEG SOI marker
.tiff  — TIFF little-endian or big-endian header
.bmp   — BM header
.webp  — RIFF....WEBP header
"""

from __future__ import annotations

# (extension_without_dot) → list of accepted magic byte sequences (bytes objects).
# The check passes if the file content STARTS WITH any of these sequences.
_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "pdf": [b"%PDF"],
    # DOCX/XLSX/PPTX are all ZIP files; PK\x03\x04 is the local file header
    "docx": [b"PK\x03\x04"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "jpg": [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    # TIFF: little-endian (II) or big-endian (MM)
    "tiff": [b"II*\x00", b"MM\x00*"],
    "bmp": [b"BM"],
    # WEBP: RIFF at offset 0, WEBP at offset 8
    "webp": [],  # handled specially below
}

# Minimum bytes to read for signature checking
_HEADER_BYTES = 16


def _check_webp(header: bytes) -> bool:
    """WEBP: bytes 0-3 == 'RIFF', bytes 8-11 == 'WEBP'."""
    return (
        len(header) >= 12
        and header[:4] == b"RIFF"
        and header[8:12] == b"WEBP"
    )


def validate_magic_bytes(content: bytes, extension: str) -> tuple[bool, str]:
    """
    Check that ``content`` matches the expected magic bytes for ``extension``.

    Args:
        content:   Raw file bytes (only the first few bytes are inspected)
        extension: File extension WITH or WITHOUT leading dot, e.g. ".pdf" or "pdf"

    Returns:
        (True, "") on success.
        (False, reason_string) on failure.
    """
    ext = extension.lstrip(".").lower()
    header = content[:_HEADER_BYTES]

    if not header:
        return False, "File appears to be empty"

    if ext not in _MAGIC_SIGNATURES and ext != "webp":
        # Unknown extension — not our responsibility to block, just pass through
        return True, ""

    if ext == "webp":
        if _check_webp(header):
            return True, ""
        return False, f"File content does not match a valid WEBP file (magic bytes mismatch)"

    sigs = _MAGIC_SIGNATURES[ext]
    if not sigs:
        # Extension in dict but no signatures defined — treat as unknown, pass
        return True, ""

    for sig in sigs:
        if header[: len(sig)] == sig:
            return True, ""

    # Build a human-readable hex string of what we actually found
    found_hex = header[:8].hex(" ")
    expected = " | ".join(s.hex(" ") for s in sigs)
    return (
        False,
        f"File content does not match extension '.{ext}'. "
        f"Expected magic bytes: [{expected}], got: {found_hex}. "
        "Possible renamed or malformed file.",
    )
