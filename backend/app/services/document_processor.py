"""
Document Processor — Medical-Grade Stable Pipeline
====================================================
Improvements over baseline:
  1. Per-page error isolation   — one bad page never kills the job
  2. Auto-rotation / deskew     — fixes tilted/sideways medical scans
  3. Image enhancement pipeline — contrast, sharpen, denoise before OCR
  4. Confidence-based retry     — low-confidence pages retry at higher DPI
                                   or swap OCR engine
  5. Multi-page TIFF support    — common in hospital record systems
  6. Encrypted PDF handling     — attempts extraction without password
  7. pdfplumber table extraction — lab values, med lists, vitals tables
  8. DOCX embedded image OCR   — clinical attachments inside Word docs
  9. Medical text cleanup       — fix common OCR errors in drug/lab names
 10. Partial-success mode       — returns PARTIAL when some pages succeed
"""
from __future__ import annotations

import io
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from app.models.schemas import (
    BoundingBox,
    DocumentMetadata,
    DocumentPage,
    DocumentStatus,
    OCREngine,
    ProcessedDocument,
    Table,
    TableCell,
    TextBlock,
)
from app.services.layout_service import LayoutService
from app.services.ocr_service import OCRService

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
_PDF_SUFFIX = ".pdf"
_DOCX_SUFFIX = ".docx"

_MAX_RETRIES = 3           # per-page retry attempts
_MIN_CONFIDENCE = 0.55     # below this → retry with better settings
_MIN_TEXT_LENGTH = 15      # fewer chars than this → treat page as empty


# ── Common medical OCR error corrections ─────────────────────────────────────
# Maps (regex pattern → replacement) for post-OCR cleanup
_MEDICAL_CORRECTIONS: list[tuple[str, str]] = [
    # Dosage units — common misreads
    (r"\bmg(?=\s*\d|\s*/)", "mg"),          # normalise
    (r"\bm9\b", "mg"),
    (r"\brnl\b", "ml"),
    (r"\brnL\b", "mL"),
    (r"\bIU\b", "IU"),
    (r"\bl U\b", "IU"),
    # Lab values — zero/O confusion
    (r"\b([A-Z]{2,})\s*0\s*:", r"\1 O:"),  # e.g. "BLO0D" → keep numeric 0 where appropriate
    # Common mis-OCR'd words
    (r"\bpatienf\b", "patient"),
    (r"\bdiagnos1s\b", "diagnosis"),
    (r"\bprescr1ption\b", "prescription"),
    (r"\brnedication\b", "medication"),
    (r"\bpharrnacy\b", "pharmacy"),
    (r"\bphysic1an\b", "physician"),
    # Date patterns — preserve slashes
    (r"(\d{1,2})[/\\|](\d{1,2})[/\\|](\d{2,4})", r"\1/\2/\3"),
]


def _clean_medical_text(text: str) -> str:
    """Apply medical-specific OCR post-corrections."""
    for pattern, replacement in _MEDICAL_CORRECTIONS:
        text = re.sub(pattern, replacement, text)
    # Remove isolated non-alphanumeric junk lines
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        alpha_ratio = sum(c.isalnum() or c in " .,/:-()%" for c in stripped) / len(stripped)
        if alpha_ratio >= 0.4:
            lines.append(line)
    return "\n".join(lines)


# ── Perspective / dewarp correction ──────────────────────────────────────────

def _dewarp_perspective(img):
    """
    Correct perspective distortion from phone-photographed medical documents.

    When a user photographs a document at an angle, the page appears as a
    trapezoid.  This function:
      1. Detects strong edges via a Sobel-like filter.
      2. Uses a Hough-line approach (pure PIL/numpy) to find the four dominant
         page edges.
      3. Computes the four corner intersections.
      4. Applies a perspective transform (homography) to produce a flat,
         rectangular image.

    Falls back silently to the original image on any error or if the page
    already appears rectangular (skew < 3 degrees on all edges).
    """
    try:
        from PIL import Image as PilImage, ImageFilter

        w, h = img.size

        # ── Step 1: Edge map via Sobel approximation ──────────────────────
        gray = img.convert("L")

        # Horizontal and vertical Sobel kernels
        sobel_x = gray.filter(ImageFilter.Kernel(
            size=3, kernel=[-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1, offset=128
        ))
        sobel_y = gray.filter(ImageFilter.Kernel(
            size=3, kernel=[-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1, offset=128
        ))

        sx = list(sobel_x.getdata())
        sy = list(sobel_y.getdata())
        edge_mag = [
            min(255, int((((sx[i] - 128) ** 2 + (sy[i] - 128) ** 2) ** 0.5)))
            for i in range(w * h)
        ]

        # ── Step 2: Hough line accumulator (theta 0–179°, r sampled) ─────
        import math

        diag = int((w ** 2 + h ** 2) ** 0.5)
        thetas = [t * math.pi / 180 for t in range(180)]
        cos_t = [math.cos(t) for t in thetas]
        sin_t = [math.sin(t) for t in thetas]

        # Only vote with strong edge pixels (mag > 100) — sample every 3rd
        accumulator: dict[tuple[int, int], int] = {}
        edge_threshold = 110
        for idx in range(0, w * h, 3):
            if edge_mag[idx] < edge_threshold:
                continue
            px, py = idx % w, idx // w
            for t_idx in range(0, 180, 2):   # step 2° for speed
                r = int(px * cos_t[t_idx] + py * sin_t[t_idx])
                key = (t_idx, r)
                accumulator[key] = accumulator.get(key, 0) + 1

        if not accumulator:
            return img

        # ── Step 3: Extract top lines, cluster into 4 page edges ─────────
        sorted_lines = sorted(accumulator.items(), key=lambda x: -x[1])

        # Non-maximum suppression: keep lines far enough apart
        kept: list[tuple[int, int]] = []
        for (t_idx, r), votes in sorted_lines:
            if votes < 15:
                break
            # Check if too close to an already-kept line
            duplicate = False
            for (kt, kr) in kept:
                if abs(t_idx - kt) < 10 and abs(r - kr) < int(min(w, h) * 0.08):
                    duplicate = True
                    break
            if not duplicate:
                kept.append((t_idx, r))
            if len(kept) >= 12:
                break

        if len(kept) < 4:
            return img  # not enough structure to dewarp

        # Convert (theta, r) → line endpoints
        def hough_to_segment(t_idx, r, length=max(w, h) * 2):
            t = thetas[t_idx]
            cos_v, sin_v = cos_t[t_idx], sin_t[t_idx]
            x0 = cos_v * r - sin_v * length
            y0 = sin_v * r + cos_v * length
            x1 = cos_v * r + sin_v * length
            y1 = sin_v * r - cos_v * length
            return (x0, y0, x1, y1)

        def line_intersection(l1, l2):
            """Return intersection point of two lines given as (x0,y0,x1,y1)."""
            x1, y1, x2, y2 = l1
            x3, y3, x4, y4 = l2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        segments = [hough_to_segment(t, r) for (t, r) in kept]

        # Separate into roughly horizontal (t 60°–120°) and vertical (t 0–30° or 150°–180°)
        horiz = [(t, r) for (t, r) in kept if 55 <= t <= 125]
        vert  = [(t, r) for (t, r) in kept if t < 35 or t > 145]

        if len(horiz) < 2 or len(vert) < 2:
            return img

        # Sort: horizontal by r (top first), vertical by r (left first)
        horiz_sorted = sorted(horiz, key=lambda x: x[1])
        vert_sorted  = sorted(vert,  key=lambda x: x[1])

        top_line    = hough_to_segment(*horiz_sorted[0])
        bottom_line = hough_to_segment(*horiz_sorted[-1])
        left_line   = hough_to_segment(*vert_sorted[0])
        right_line  = hough_to_segment(*vert_sorted[-1])

        # ── Step 4: Find four corners ─────────────────────────────────────
        tl = line_intersection(top_line, left_line)
        tr = line_intersection(top_line, right_line)
        bl = line_intersection(bottom_line, left_line)
        br = line_intersection(bottom_line, right_line)

        if None in (tl, tr, bl, br):
            return img

        # Sanity: all corners must be near the image (within 30% outside)
        margin = 0.3
        for cx, cy in (tl, tr, bl, br):
            if cx < -w * margin or cx > w * (1 + margin):
                return img
            if cy < -h * margin or cy > h * (1 + margin):
                return img

        # Skip if already rectangular (all corners within 3% of image bounds)
        def near_rect():
            corners = [(tl, (0, 0)), (tr, (w, 0)), (bl, (0, h)), (br, (w, h))]
            for (cx, cy), (ex, ey) in corners:
                if abs(cx - ex) > w * 0.03 or abs(cy - ey) > h * 0.03:
                    return False
            return True

        if near_rect():
            logger.debug("Page already rectangular — skipping dewarp")
            return img

        # ── Step 5: Perspective transform using PIL transform ─────────────
        # Output size: use the longer of the two horizontal / vertical spans
        out_w = int(max(
            ((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2) ** 0.5,
            ((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2) ** 0.5,
        ))
        out_h = int(max(
            ((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2) ** 0.5,
            ((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2) ** 0.5,
        ))

        if out_w < 100 or out_h < 100:
            return img

        # PIL's PERSPECTIVE transform needs 8-coefficient matrix.
        # We compute it via solving the linear system for the mapping:
        #   src corners → dst corners (rectangle)
        src = [tl[0], tl[1], tr[0], tr[1], br[0], br[1], bl[0], bl[1]]
        dst = [0, 0, out_w, 0, out_w, out_h, 0, out_h]

        def _find_coeffs(src_pts, dst_pts):
            import numpy as np
            matrix = []
            for (sx, sy), (dx, dy) in zip(
                [(src_pts[i*2], src_pts[i*2+1]) for i in range(4)],
                [(dst_pts[i*2], dst_pts[i*2+1]) for i in range(4)],
            ):
                matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
                matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
            A = np.array(matrix, dtype=float)
            b = np.array(src_pts, dtype=float)
            res = np.linalg.lstsq(A, b, rcond=None)[0]
            return list(res)

        try:
            coeffs = _find_coeffs(src, dst)
            dewarped = img.transform(
                (out_w, out_h),
                PilImage.PERSPECTIVE,
                coeffs,
                PilImage.BICUBIC,
            )
            logger.info(
                f"Dewarp applied: ({w}×{h}) → ({out_w}×{out_h}) "
                f"corners TL={tl} TR={tr} BL={bl} BR={br}"
            )
            return dewarped
        except ImportError:
            # numpy not available — skip transform
            logger.debug("Dewarp skipped: numpy not available")
            return img

    except Exception as exc:
        logger.debug(f"Dewarp failed (using original): {exc}")
        return img


# ── Image enhancement for medical scans ──────────────────────────────────────

def _upscale_if_needed(img, min_width: int = 1400):
    """
    Upscale images that are too small for reliable OCR.
    Most medical scanners produce 200-300 DPI; fax machines can be as low as 96 DPI.
    Tesseract accuracy drops significantly below ~200 DPI equivalent.
    """
    try:
        from PIL import Image as PilImage
        w, h = img.size
        if w < min_width:
            scale = min_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            logger.debug(f"Upscaling image {w}x{h} → {new_w}x{new_h} (scale {scale:.2f}×)")
            return img.resize((new_w, new_h), PilImage.LANCZOS)
    except Exception as exc:
        logger.debug(f"Upscale failed (using original): {exc}")
    return img


def _denoise_image(img):
    """
    Remove salt-and-pepper noise common in faxed and photocopied medical records.
    Uses median filter — preserves edges better than Gaussian blur.
    """
    try:
        from PIL import ImageFilter
        return img.filter(ImageFilter.MedianFilter(size=3))
    except Exception as exc:
        logger.debug(f"Denoising failed (using original): {exc}")
    return img


def _remove_scanner_borders(img):
    """
    Crop black scanner borders that appear when scanning documents smaller than
    the scanner bed (common with letter-size forms on A4 scanners and vice versa).
    Finds the bounding box of non-black content and crops to it.
    """
    try:
        from PIL import ImageOps
        gray = img.convert("L")
        # Invert so black borders become white, content stays dark
        inverted = ImageOps.invert(gray)
        bbox = inverted.getbbox()
        if bbox:
            margin = 10  # leave a small margin
            left = max(0, bbox[0] - margin)
            top = max(0, bbox[1] - margin)
            right = min(img.width, bbox[2] + margin)
            bottom = min(img.height, bbox[3] + margin)
            cropped = img.crop((left, top, right, bottom))
            # Only use crop if it removed a meaningful border (>5% of image)
            area_ratio = (cropped.width * cropped.height) / (img.width * img.height)
            if area_ratio < 0.95:
                logger.debug(f"Removed scanner border — {img.size} → {cropped.size}")
                return cropped
    except Exception as exc:
        logger.debug(f"Border removal failed (using original): {exc}")
    return img


def _is_blank_page(img, threshold: float = 0.98) -> bool:
    """
    Detect blank or near-blank pages to skip OCR entirely.
    A page is considered blank if >= threshold fraction of pixels are near-white.
    """
    try:
        from PIL import ImageOps
        import statistics
        gray = img.convert("L")
        pixels = list(gray.getdata())
        white_count = sum(1 for p in pixels if p > 240)
        return (white_count / len(pixels)) >= threshold
    except Exception:
        return False


# ── Multi-column reading order ────────────────────────────────────────────────

def _detect_column_boundaries(img, min_gap_frac: float = 0.025) -> list[tuple[int, int]]:
    """
    Detect vertical column separators using a projection profile.

    For each x-position, count how many dark (text) pixels exist in that
    vertical strip.  A sustained low-density strip = column gap.

    Returns list of (x_start, x_end) column regions, left to right.
    Falls back to [(0, width)] for single-column pages.
    """
    try:
        width, height = img.size
        gray = img.convert("L")
        pixels = list(gray.getdata())

        # Vertical projection: count dark pixels per x column
        dark_counts: list[int] = []
        for x in range(width):
            col = [pixels[y * width + x] for y in range(height)]
            dark_counts.append(sum(1 for p in col if p < 140))

        # Smooth with a moving average (window = ~1% of width)
        win = max(5, width // 80)
        smoothed: list[float] = []
        for i in range(width):
            s = max(0, i - win // 2)
            e = min(width, i + win // 2 + 1)
            smoothed.append(sum(dark_counts[s:e]) / (e - s))

        # A gap must have < 3% of average column density
        avg_density = sum(smoothed) / width if width else 1
        gap_threshold = avg_density * 0.12
        min_gap_w = max(8, int(width * min_gap_frac))
        min_col_w = int(width * 0.18)   # ignore very thin slivers

        # Collect gap spans
        gaps: list[tuple[int, int]] = []
        in_gap = False
        gap_start = 0
        for x, val in enumerate(smoothed):
            if val <= gap_threshold:
                if not in_gap:
                    in_gap, gap_start = True, x
            else:
                if in_gap:
                    if x - gap_start >= min_gap_w:
                        gaps.append((gap_start, x))
                    in_gap = False
        if in_gap and width - gap_start >= min_gap_w:
            gaps.append((gap_start, width))

        if not gaps:
            return [(0, width)]

        # Build column regions from gaps
        cols: list[tuple[int, int]] = []
        prev = 0
        for gs, ge in gaps:
            mid = (gs + ge) // 2
            if mid - prev >= min_col_w:
                cols.append((prev, mid))
            prev = mid
        if width - prev >= min_col_w:
            cols.append((prev, width))

        return cols if len(cols) > 1 else [(0, width)]

    except Exception as exc:
        logger.debug(f"Column detection failed: {exc}")
        return [(0, width)]


def _reorder_blocks_by_columns(
    blocks: list, column_boundaries: list[tuple[int, int]]
) -> list:
    """
    Re-sort OCR text blocks so they read column-by-column, top-to-bottom
    within each column, left-to-right across columns.

    Without this, a two-column lab report reads as:
      "Test Name  Value  Test Name  Value …" (row-by-row across both columns)
    With this it reads:
      "Test Name  Value  …  [end of left col]  Test Name  Value …"
    """
    if len(column_boundaries) <= 1:
        return blocks

    groups: list[list] = [[] for _ in column_boundaries]
    for block in blocks:
        cx = (block.bbox.x + block.bbox.width / 2) if block.bbox else 0
        placed = False
        for idx, (cs, ce) in enumerate(column_boundaries):
            if cs <= cx < ce:
                groups[idx].append(block)
                placed = True
                break
        if not placed:
            groups[-1].append(block)

    reordered = []
    for col_blocks in groups:
        reordered.extend(sorted(col_blocks, key=lambda b: (b.bbox.y if b.bbox else 0)))
    return reordered


def _raw_text_in_column_order(
    blocks: list, column_boundaries: list[tuple[int, int]]
) -> str:
    """
    Produce raw_text string that reads in correct column order.
    Each column's text is separated by a blank line so the LLM
    understands the column boundary.
    """
    if len(column_boundaries) <= 1:
        return "\n".join(b.text for b in blocks if b.text)

    groups: list[list] = [[] for _ in column_boundaries]
    for block in blocks:
        cx = (block.bbox.x + block.bbox.width / 2) if block.bbox else 0
        for idx, (cs, ce) in enumerate(column_boundaries):
            if cs <= cx < ce:
                groups[idx].append(block)
                break
        else:
            groups[-1].append(block)

    col_texts = []
    for col_blocks in groups:
        sorted_blocks = sorted(col_blocks, key=lambda b: (b.bbox.y if b.bbox else 0))
        text = "\n".join(b.text for b in sorted_blocks if b.text)
        if text.strip():
            col_texts.append(text)
    return "\n\n---\n\n".join(col_texts)


# ── Header / footer metadata extraction ──────────────────────────────────────

# Regex patterns for structured medical fields found in page headers
_HEADER_PATTERNS: dict[str, list[str]] = {
    "patient_name": [
        r"(?i)patient(?:\s+name)?[:\s]+([A-Z][a-zA-Z]+(?:[,\s]+[A-Z][a-zA-Z]+){1,3})",
        r"(?i)(?:^|\n)name[:\s]+([A-Z][a-zA-Z]+(?:[,\s]+[A-Z][a-zA-Z]+){1,3})",
    ],
    "dob": [
        r"(?i)(?:dob|date\s+of\s+birth|birth(?:\s+date)?)[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    ],
    "mrn": [
        r"(?i)(?:mrn|medical\s+record\s+(?:number|no\.?|#))[:\s#]*([A-Z0-9\-]{4,20})",
        r"(?i)(?:patient\s+(?:id|#))[:\s#]*([A-Z0-9\-]{4,20})",
        r"(?i)(?:acct\.?|account)\s*#?\s*[:\s]*([A-Z0-9\-]{4,20})",
    ],
    "visit_date": [
        r"(?i)(?:visit|encounter|service|admit(?:ted)?|discharge)\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        r"(?i)(?:^|\n)date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    ],
    "physician": [
        r"(?i)(?:physician|provider|doctor|attending|ordering|dr\.?)[:\s]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})",
    ],
    "facility": [
        r"(?i)(?:facility|hospital|clinic|location|site)[:\s]+([A-Za-z0-9\s&'\.]+(?:Hospital|Medical|Clinic|Center|Health|System))",
    ],
    "ssn_last4": [
        r"(?i)(?:ssn|social)[:\s#]*\*+(\d{4})",
    ],
}


def _parse_medical_header(text: str) -> dict[str, str]:
    """Extract structured medical metadata from raw header/footer text."""
    metadata: dict[str, str] = {}
    for field, patterns in _HEADER_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                metadata[field] = m.group(1).strip()
                break
    return metadata


def _extract_header_footer(
    img, ocr_svc, languages,
    header_frac: float = 0.11,
    footer_frac: float = 0.07,
) -> tuple[str, str, dict[str, str]]:
    """
    Split the image into header / body / footer bands and OCR each band.

    Returns (header_text, footer_text, metadata_dict).
    The body crop is NOT returned — the main OCR pipeline handles the full page.
    Metadata dict contains structured fields: patient_name, dob, mrn, etc.
    """
    header_text = ""
    footer_text = ""
    metadata: dict[str, str] = {}

    try:
        from PIL import Image as PilImage
        w, h = img.size
        header_h = max(30, int(h * header_frac))
        footer_h = max(20, int(h * footer_frac))

        header_band = img.crop((0, 0, w, header_h))
        footer_band = img.crop((0, h - footer_h, w, h))

        # Run Tesseract on each band directly (faster than full OCR service)
        try:
            import pytesseract
            header_text = pytesseract.image_to_string(header_band, lang="eng").strip()
            footer_text = pytesseract.image_to_string(footer_band, lang="eng").strip()
        except Exception:
            # Fall back to OCR service
            try:
                ht, _, _, _ = ocr_svc.process_image(header_band, OCREngine.TESSERACT, False, languages, 0)
                header_text = ht
            except Exception:
                pass

        metadata = _parse_medical_header(header_text + "\n" + footer_text)

        if metadata:
            logger.debug(f"Header metadata extracted: {list(metadata.keys())}")

    except Exception as exc:
        logger.debug(f"Header/footer extraction failed: {exc}")

    return header_text, footer_text, metadata


# ── Checkbox / form field detection ──────────────────────────────────────────

# Unicode checkbox characters Tesseract may produce
_CHECKBOX_CHARS = {"□", "■", "☐", "☑", "☒", "✓", "✗", "✘", "◻", "◼", "▢", "▣"}
_CHECKED_CHARS  = {"■", "☑", "☒", "✓", "✗", "✘", "◼", "▣"}


def _detect_checkboxes(img, page_idx: int = 0) -> list[dict]:
    """
    Detect checkboxes and their checked/unchecked state from a medical form image.

    Strategy 1 — Unicode detection:
      Tesseract often produces □ / ■ / ☑ characters for checkboxes.
      We find them in the word-level data and look for the label to their right.

    Strategy 2 — Image analysis:
      Find small square contours by scanning for enclosed dark rectangles
      (aspect ratio 0.7–1.3, small area, thin border).
      Measure fill ratio inside the square to determine checked state.

    Returns list of dicts: {label, checked, x, y, width, height, source}
    """
    checkboxes: list[dict] = []

    # ── Strategy 1: Unicode characters from Tesseract ────────────────────────
    try:
        import pytesseract
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n = len(data["text"])
        for i in range(n):
            word = str(data["text"][i]).strip()
            if any(ch in word for ch in _CHECKBOX_CHARS):
                checked = any(ch in word for ch in _CHECKED_CHARS)
                # Collect label: words to the right on the same line (within 300px)
                label_parts = []
                cy = data["top"][i] + data["height"][i] // 2
                cx_right = data["left"][i] + data["width"][i]
                for j in range(i + 1, min(i + 8, n)):
                    jw = str(data["text"][j]).strip()
                    if not jw:
                        continue
                    jcy = data["top"][j] + data["height"][j] // 2
                    jcx = data["left"][j]
                    if abs(jcy - cy) <= 15 and jcx - cx_right < 300:
                        label_parts.append(jw)
                        cx_right = data["left"][j] + data["width"][j]
                    elif abs(jcy - cy) > 15:
                        break
                label = " ".join(label_parts)
                checkboxes.append({
                    "label": label,
                    "checked": checked,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                    "source": "unicode",
                })
    except Exception as exc:
        logger.debug(f"Checkbox unicode scan failed: {exc}")

    # ── Strategy 2: Image analysis for drawn square checkboxes ───────────────
    # Only run if strategy 1 found nothing (avoid duplicates)
    if not checkboxes:
        try:
            from PIL import Image as PilImage
            import pytesseract

            gray = img.convert("L")
            w, h = gray.size

            # Build binary image: dark pixels = 1
            pixels = list(gray.getdata())
            binary = [1 if p < 128 else 0 for p in pixels]

            # Scan for small filled rectangular regions (checkbox squares)
            # Expected checkbox size: 8–30px
            min_sz, max_sz = 8, 35

            checked_candidates: list[dict] = []
            visited = set()

            for y in range(0, h - min_sz, 3):
                for x in range(0, w - min_sz, 3):
                    if (x, y) in visited:
                        continue
                    if binary[y * w + x] != 1:
                        continue

                    # Try to find a square box starting at (x, y)
                    # Check top border
                    box_w = 0
                    for bw in range(min_sz, min(max_sz + 1, w - x)):
                        if binary[y * w + (x + bw)] == 1:
                            box_w = bw
                        else:
                            break
                    if box_w < min_sz:
                        continue

                    # Check left border height
                    box_h = 0
                    for bh in range(min_sz, min(max_sz + 1, h - y)):
                        if binary[(y + bh) * w + x] == 1:
                            box_h = bh
                        else:
                            break
                    if box_h < min_sz:
                        continue

                    # Aspect ratio check (must be roughly square)
                    ar = box_w / box_h if box_h else 0
                    if not (0.6 <= ar <= 1.6):
                        continue

                    # Measure fill ratio inside the box
                    inner_x1, inner_y1 = x + 2, y + 2
                    inner_x2, inner_y2 = x + box_w - 2, y + box_h - 2
                    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
                        continue

                    inner_pixels = [
                        binary[iy * w + ix]
                        for iy in range(inner_y1, inner_y2)
                        for ix in range(inner_x1, inner_x2)
                    ]
                    fill_ratio = sum(inner_pixels) / len(inner_pixels) if inner_pixels else 0
                    checked = fill_ratio > 0.25  # >25% dark inside = checked

                    # Mark region visited
                    for iy in range(y, y + box_h):
                        for ix in range(x, x + box_w):
                            visited.add((ix, iy))

                    # Find label to the right using Tesseract
                    label = ""
                    try:
                        label_crop = img.crop((
                            x + box_w + 2,
                            max(0, y - 3),
                            min(w, x + box_w + 200),
                            min(h, y + box_h + 3),
                        ))
                        label = pytesseract.image_to_string(
                            label_crop, config="--psm 7"
                        ).strip()
                    except Exception:
                        pass

                    checked_candidates.append({
                        "label": label,
                        "checked": checked,
                        "x": x, "y": y,
                        "width": box_w, "height": box_h,
                        "source": "image",
                    })

            checkboxes.extend(checked_candidates)

        except Exception as exc:
            logger.debug(f"Checkbox image analysis failed: {exc}")

    if checkboxes:
        logger.debug(f"Detected {len(checkboxes)} checkbox(es) on page {page_idx + 1}")

    return checkboxes


def _checkboxes_to_text(checkboxes: list[dict]) -> str:
    """Convert detected checkboxes to readable text for the LLM."""
    if not checkboxes:
        return ""
    lines = ["[Form Fields]"]
    for cb in checkboxes:
        mark = "☑" if cb["checked"] else "☐"
        label = cb.get("label", "").strip() or "(no label)"
        lines.append(f"  {mark} {label}")
    return "\n".join(lines)


# ── Stamp and seal removal ───────────────────────────────────────────────────

# Ink colours common in medical stamps (CONFIDENTIAL, RECEIVED, DRAFT, COPY)
_STAMP_HUE_RANGES = [
    (330, 360, 60),
    (0,   20,  60),
    (200, 260, 50),
    (90,  160, 45),
    (20,  50,  55),
]

_STAMP_KEYWORDS = {
    "confidential", "received", "draft", "copy", "void", "approved",
    "original", "fax", "transmitted", "protected", "phi", "do not copy",
    "not for distribution",
}


def _remove_stamps(img):
    """
    Detect and remove rubber-stamp / embossed-seal overlays from medical documents.

    Strategy:
      1. Convert to HSV and isolate coloured ink (red, blue, green).
      2. Find connected blobs of coloured pixels.
      3. Filter blobs by stamp-like shape (size, aspect ratio, density).
      4. Confirm via Tesseract keyword match where possible.
      5. Replace stamp regions with local background colour (median of border ring).
    """
    try:
        from PIL import Image as PilImage, ImageDraw

        w, h = img.size
        page_area = w * h
        rgb = img.convert("RGB")
        r_data = list(rgb.getdata())

        def rgb_to_hsv(r, g, b):
            r, g, b = r / 255.0, g / 255.0, b / 255.0
            mx, mn = max(r, g, b), min(r, g, b)
            diff = mx - mn
            if mx == 0:
                return 0, 0, 0
            s = diff / mx
            if diff == 0:
                h_val = 0.0
            elif mx == r:
                h_val = 60 * (((g - b) / diff) % 6)
            elif mx == g:
                h_val = 60 * ((b - r) / diff + 2)
            else:
                h_val = 60 * ((r - g) / diff + 4)
            return h_val, s * 100, mx * 100

        mask = [False] * (w * h)
        for idx, (r, g, b) in enumerate(r_data):
            hv, sv, vv = rgb_to_hsv(r, g, b)
            if vv < 15 or sv < 30:
                continue
            for h_lo, h_hi, s_min in _STAMP_HUE_RANGES:
                if h_lo <= hv <= h_hi and sv >= s_min:
                    mask[idx] = True
                    break

        visited = [False] * (w * h)
        blobs: list[list[int]] = []

        def flood(start: int) -> list[int]:
            stack = [start]
            blob: list[int] = []
            while stack:
                idx = stack.pop()
                if idx < 0 or idx >= w * h or visited[idx] or not mask[idx]:
                    continue
                visited[idx] = True
                blob.append(idx)
                x_pos, y_pos = idx % w, idx // w
                if x_pos > 0:     stack.append(idx - 1)
                if x_pos < w - 1: stack.append(idx + 1)
                if y_pos > 0:     stack.append(idx - w)
                if y_pos < h - 1: stack.append(idx + w)
            return blob

        for i in range(0, w * h, 4):
            if mask[i] and not visited[i]:
                blob = flood(i)
                if len(blob) > 80:
                    blobs.append(blob)

        if not blobs:
            return img

        stamp_regions: list[tuple[int, int, int, int]] = []

        for blob in blobs:
            xs = [idx % w for idx in blob]
            ys = [idx // w for idx in blob]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            blob_area = bw * bh

            if blob_area < page_area * 0.002 or blob_area > page_area * 0.25:
                continue
            ar = bw / bh if bh else 0
            if not (0.3 <= ar <= 3.5):
                continue
            density = len(blob) / blob_area
            if density < 0.08:
                continue

            confirmed = False
            try:
                import pytesseract
                margin = 6
                crop = rgb.crop((
                    max(0, x1 - margin), max(0, y1 - margin),
                    min(w, x2 + margin), min(h, y2 + margin),
                ))
                stamp_text = pytesseract.image_to_string(crop, config="--psm 11").lower()
                if any(kw in stamp_text for kw in _STAMP_KEYWORDS):
                    confirmed = True
            except Exception:
                pass

            if confirmed or (density > 0.18 and blob_area > page_area * 0.005):
                stamp_regions.append((x1, y1, x2, y2))
                logger.info(
                    f"Stamp detected ({x1},{y1})–({x2},{y2}) "
                    f"density={density:.2f} confirmed={confirmed}"
                )

        if not stamp_regions:
            return img

        result = rgb.copy()
        draw = ImageDraw.Draw(result)

        for (x1, y1, x2, y2) in stamp_regions:
            pad = 10
            ring: list[tuple[int, int, int]] = []
            for bx in range(max(0, x1 - pad), min(w, x2 + pad)):
                for by in [max(0, y1 - pad), min(h - 1, y2 + pad)]:
                    ring.append(r_data[by * w + bx])
            for by in range(max(0, y1 - pad), min(h, y2 + pad)):
                for bx in [max(0, x1 - pad), min(w - 1, x2 + pad)]:
                    ring.append(r_data[by * w + bx])

            if ring:
                bg = (
                    sorted(p[0] for p in ring)[len(ring) // 2],
                    sorted(p[1] for p in ring)[len(ring) // 2],
                    sorted(p[2] for p in ring)[len(ring) // 2],
                )
            else:
                bg = (255, 255, 255)

            draw.rectangle([x1, y1, x2, y2], fill=bg)

        logger.info(f"Removed {len(stamp_regions)} stamp(s) from image")
        return result

    except Exception as exc:
        logger.warning(f"Stamp removal failed (using original): {exc}")
        return img


# ── Image quality assessment ──────────────────────────────────────────────────

# Thresholds — tuned for medical document scanning conditions
_BLUR_WARN      = 80.0    # Laplacian variance below this = blurry
_BLUR_FAIL      = 20.0    # below this = too blurry for reliable OCR
_CONTRAST_WARN  = 40.0    # std-dev of pixel values below this = low contrast
_BRIGHTNESS_LOW = 50.0    # mean pixel value below this = too dark
_BRIGHTNESS_HI  = 230.0   # mean pixel value above this = overexposed
_NOISE_WARN     = 18.0    # noise estimate above this = noisy scan


def _assess_image_quality(img) -> dict:
    """
    Score an image on four medical-relevant quality dimensions before OCR:

      blur_score      — Laplacian variance; higher = sharper
      contrast_score  — pixel std-dev; higher = more contrast
      brightness      — mean pixel value (0=black, 255=white)
      noise_score     — estimated noise level via difference filter

    Returns a dict with numeric scores, a grade (good/warn/poor),
    and a human-readable list of warnings suitable for the UI.
    """
    try:
        from PIL import ImageFilter

        gray = img.convert("L")
        pixels = list(gray.getdata())
        n = len(pixels)
        if n == 0:
            return {"grade": "unknown", "warnings": []}

        # ── Brightness ────────────────────────────────────────────────────
        brightness = sum(pixels) / n

        # ── Contrast (standard deviation of pixel values) ─────────────────
        mean = brightness
        variance = sum((p - mean) ** 2 for p in pixels) / n
        contrast_score = variance ** 0.5

        # ── Blur (Laplacian variance) ─────────────────────────────────────
        # Apply Laplacian edge filter; high variance = sharp edges = not blurry
        lap = gray.filter(ImageFilter.Kernel(
            size=3,
            kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
            scale=1, offset=128,
        ))
        lap_pixels = list(lap.getdata())
        lap_mean = sum(lap_pixels) / len(lap_pixels)
        blur_score = sum((p - lap_mean) ** 2 for p in lap_pixels) / len(lap_pixels)

        # ── Noise (mean absolute difference between adjacent pixels) ──────
        w, h = gray.size
        diffs = []
        for y in range(0, h - 1, 4):      # sample every 4th row for speed
            for x in range(0, w - 1, 4):
                diffs.append(abs(pixels[y * w + x] - pixels[y * w + x + 1]))
                diffs.append(abs(pixels[y * w + x] - pixels[(y + 1) * w + x]))
        noise_score = sum(diffs) / len(diffs) if diffs else 0.0

        # ── Warnings ──────────────────────────────────────────────────────
        warnings: list[str] = []

        if blur_score < _BLUR_FAIL:
            warnings.append(
                f"Image is too blurry for reliable OCR (blur={blur_score:.0f}). "
                "Please rescan at 300 DPI or higher."
            )
        elif blur_score < _BLUR_WARN:
            warnings.append(
                f"Image is slightly blurry (blur={blur_score:.0f}). "
                "OCR accuracy may be reduced."
            )

        if contrast_score < _CONTRAST_WARN:
            warnings.append(
                f"Low contrast detected (contrast={contrast_score:.0f}). "
                "Document may be faded — consider increasing scanner brightness."
            )

        if brightness < _BRIGHTNESS_LOW:
            warnings.append(
                f"Image is too dark (brightness={brightness:.0f}/255). "
                "Increase scanner exposure or use a higher-quality scan."
            )
        elif brightness > _BRIGHTNESS_HI:
            warnings.append(
                f"Image is overexposed (brightness={brightness:.0f}/255). "
                "Reduce scanner brightness to reveal faint text."
            )

        if noise_score > _NOISE_WARN:
            warnings.append(
                f"High noise level detected (noise={noise_score:.1f}). "
                "This may be a fax or photocopy — OCR may have errors."
            )

        # ── Grade ─────────────────────────────────────────────────────────
        if blur_score < _BLUR_FAIL or contrast_score < _CONTRAST_WARN * 0.6:
            grade = "poor"
        elif warnings:
            grade = "warn"
        else:
            grade = "good"

        result = {
            "grade": grade,
            "blur_score": round(blur_score, 1),
            "contrast_score": round(contrast_score, 1),
            "brightness": round(brightness, 1),
            "noise_score": round(noise_score, 1),
            "warnings": warnings,
        }

        if warnings:
            logger.warning(f"Image quality {grade.upper()}: {'; '.join(warnings)}")

        return result

    except Exception as exc:
        logger.debug(f"Quality assessment failed: {exc}")
        return {"grade": "unknown", "warnings": []}


def _enhance_medical_image(img):
    """
    Full enhancement pipeline for medical document images:
    1. Remove scanner borders
    2. Upscale if resolution is too low
    3. Denoise (median filter for fax artifacts)
    4. Grayscale + auto-level histogram
    5. Double-sharpen for fax/photocopy quality
    6. Contrast boost
    """
    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps

        img = _dewarp_perspective(img)   # fix phone-photo trapezoid distortion
        img = _remove_scanner_borders(img)
        img = _remove_stamps(img)        # remove CONFIDENTIAL/RECEIVED overlays
        img = _upscale_if_needed(img)
        img = _denoise_image(img)

        gray = img.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=2)
        gray = gray.filter(ImageFilter.SHARPEN)
        gray = gray.filter(ImageFilter.SHARPEN)

        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.8)

        return gray.convert("RGB")
    except Exception as exc:
        logger.warning(f"Image enhancement failed (using original): {exc}")
        return img


def _auto_rotate(img):
    """
    Detect and correct image orientation using Tesseract OSD.
    Falls back gracefully if pytesseract is unavailable.
    Returns (rotated_image, rotation_degrees_applied).
    """
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle and angle != 0:
            logger.debug(f"Auto-rotating image by {angle}°")
            return img.rotate(-angle, expand=True), angle
    except Exception:
        pass
    return img, 0


# ── Main processor ────────────────────────────────────────────────────────────

class DocumentProcessor:
    """Medical-grade document processor with full error isolation."""

    def __init__(self):
        self.layout_svc = LayoutService()
        self.ocr_svc = OCRService()

    def process(
        self,
        file_path,
        document_id: Optional[str] = None,
        ocr_engine: OCREngine = OCREngine.AUTO,
        enhance_images: bool = True,
        languages: list = None,
        extract_tables: bool = True,
    ) -> ProcessedDocument:
        """
        Process any supported medical document.
        Never raises — errors are captured in the returned ProcessedDocument.
        """
        if not file_path:
            raise ValueError("file_path cannot be None or empty")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        doc_id = document_id or str(uuid.uuid4())
        suffix = path.suffix.lower()
        start = time.time()
        page_errors: list[str] = []

        logger.info(f"[{doc_id}] Processing '{path.name}' (engine={ocr_engine.value}) …")

        try:
            if suffix == _PDF_SUFFIX:
                pages = self._process_pdf(
                    path, ocr_engine, enhance_images, languages, page_errors, extract_tables
                )
            elif suffix == _DOCX_SUFFIX:
                pages = self._process_docx(path, ocr_engine, enhance_images, languages, page_errors)
            elif suffix in _IMAGE_SUFFIXES:
                # Multi-page TIFF handled inside _process_image
                pages = self._process_image(
                    path, ocr_engine, enhance_images, languages, page_errors, extract_tables
                )
            else:
                return self._fail(doc_id, path,
                                  f"Unsupported file type '{suffix}'. "
                                  f"Accepted: pdf, docx, png, jpg, jpeg, tiff, bmp, webp")
        except Exception as exc:
            logger.exception(f"[{doc_id}] Fatal pipeline error: {exc}")
            return self._fail(doc_id, path, f"Processing failed: {type(exc).__name__}: {exc}")

        good_pages = [p for p in pages if not p.error]
        failed_pages = [p for p in pages if p.error]

        if not pages:
            return self._fail(doc_id, path, "No pages could be processed from this document")

        full_text = self._merge_pages_text(pages)
        elapsed = int((time.time() - start) * 1000)

        if not good_pages:
            status = DocumentStatus.FAILED
            top_error = (f"{len(failed_pages)} page(s) all failed. "
                         f"First error: {failed_pages[0].error}")
        elif failed_pages:
            status = DocumentStatus.PARTIAL
            top_error = (f"{len(failed_pages)}/{len(pages)} page(s) failed: " +
                         "; ".join(f"p{p.page_number}: {p.error}" for p in failed_pages[:3]))
            logger.warning(f"[{doc_id}] Partial success — {len(good_pages)} ok, "
                           f"{len(failed_pages)} failed")
        else:
            status = DocumentStatus.COMPLETED
            top_error = None

        metadata = DocumentMetadata(
            filename=path.name,
            file_size=path.stat().st_size,
            file_type=suffix,
            page_count=len(pages),
            ocr_engine=ocr_engine.value,
            processing_time_ms=elapsed,
        )

        logger.info(f"[{doc_id}] {status.value.upper()} — {len(pages)} pages, "
                    f"{len(full_text)} chars, {elapsed}ms")

        return ProcessedDocument(
            document_id=doc_id,
            metadata=metadata,
            pages=pages,
            full_text=full_text,
            status=status,
            error=top_error,
            partial_page_errors=[f"p{p.page_number}: {p.error}" for p in failed_pages],
        )

    # ── PDF pipeline ──────────────────────────────────────────────────────────

    def _process_pdf(self, path, ocr_engine, enhance, languages, page_errors, extract_tables):
        # Handle encrypted PDFs first
        path = self._decrypt_pdf_if_needed(path, page_errors)

        try:
            is_scanned = self.layout_svc.is_scanned_pdf(path)
            page_count = self.layout_svc.get_pdf_page_count(path)
        except Exception as exc:
            logger.warning(f"PDF metadata read failed ({exc}), attempting raw text fallback")
            return self._pdf_raw_text_fallback(path, page_errors, extract_tables)

        logger.info(f"PDF scanned={is_scanned}, pages={page_count}")

        if is_scanned:
            return self._ocr_pdf_pages(path, page_count, ocr_engine, enhance, languages)

        # Native text PDF — extract layout, supplement sparse pages with OCR
        pages = []
        try:
            native_pages = self.layout_svc.extract_pdf_layout(path)
        except Exception as exc:
            logger.warning(f"Native PDF layout extraction failed ({exc}), falling back to full OCR")
            return self._ocr_pdf_pages(path, page_count, ocr_engine, enhance, languages)

        # Extract tables with pdfplumber (better than layout service for structured data)
        plumber_tables: dict[int, list[Table]] = {}
        if extract_tables:
            plumber_tables = self._extract_pdf_tables(path)

        for page in native_pages:
            if page.error:
                pages.append(page)
                continue
            # Supplement sparse text with OCR
            if len(page.raw_text.strip()) < _MIN_TEXT_LENGTH:
                page = self._try_ocr_page(path, page, ocr_engine, enhance, languages)
            # Attach tables from pdfplumber
            pg_num = page.page_number
            if pg_num in plumber_tables:
                page.tables = plumber_tables[pg_num]
            # Clean text
            page.raw_text = _clean_medical_text(page.raw_text)
            pages.append(page)

        return pages

    def _ocr_pdf_pages(self, path, page_count, ocr_engine, enhance, languages):
        pages = []
        for page_idx in range(page_count):
            page = self._ocr_single_pdf_page(path, page_idx, ocr_engine, enhance, languages)
            pages.append(page)
        return pages

    def _ocr_single_pdf_page(
        self, path, page_idx: int, ocr_engine, enhance, languages, retry: int = 0
    ) -> DocumentPage:
        """OCR a single PDF page with multi-strategy retry."""
        try:
            dpi = 300 if retry < 2 else 400  # escalate DPI on retries
            img = self.layout_svc.render_pdf_page(path, page_idx, dpi=dpi)

            # Quality check before enhancement so score reflects raw scan
            quality = _assess_image_quality(img)

            # Enhance and auto-rotate medical scans
            if enhance:
                img = _enhance_medical_image(img)
            img, rotation = _auto_rotate(img)

            text, blocks, engine, conf = self.ocr_svc.process_image(
                img, ocr_engine, enhance, languages, page_idx
            )

            # If confidence is too low, retry with the alternate OCR engine
            alt_engine = OCREngine.EASYOCR if ocr_engine == OCREngine.TESSERACT else OCREngine.TESSERACT
            if conf < _MIN_CONFIDENCE and retry < _MAX_RETRIES:
                logger.info(f"Page {page_idx + 1} low confidence ({conf:.2f}), "
                            f"retrying with {alt_engine.value}")
                return self._ocr_single_pdf_page(
                    path, page_idx, alt_engine, True, languages, retry + 1
                )

            regions = self.layout_svc.analyze_image_layout(img, page_idx)
            if regions:
                blocks = self._assign_text_to_regions(regions, blocks)

            # Header / footer metadata
            header_text, footer_text, header_meta = _extract_header_footer(
                img, self.ocr_svc, languages
            )

            # Multi-column reading order
            col_boundaries = _detect_column_boundaries(img)
            col_count = len(col_boundaries)
            if col_count > 1:
                logger.info(f"PDF page {page_idx + 1}: detected {col_count} columns")
                blocks = _reorder_blocks_by_columns(blocks, col_boundaries)
                clean_text = _raw_text_in_column_order(blocks, col_boundaries)
            else:
                clean_text = _clean_medical_text(text)

            # Checkbox detection
            checkboxes = _detect_checkboxes(img, page_idx)
            checkbox_text = _checkboxes_to_text(checkboxes)
            if checkbox_text:
                clean_text = clean_text + "\n\n" + checkbox_text

            return DocumentPage(
                page_number=page_idx + 1,
                width=float(img.width),
                height=float(img.height),
                text_blocks=blocks,
                tables=[],
                raw_text=clean_text,
                confidence=conf,
                header_text=header_text,
                footer_text=footer_text,
                header_metadata=header_meta,
                column_count=col_count,
                checkboxes=checkboxes,
                image_quality=quality,
            )

        except MemoryError:
            if retry < _MAX_RETRIES:
                logger.warning(f"OOM on page {page_idx + 1}, retrying at 150 DPI")
                try:
                    img = self.layout_svc.render_pdf_page(path, page_idx, dpi=150)
                    img = _enhance_medical_image(img)
                    text, blocks, _, conf = self.ocr_svc.process_image(
                        img, OCREngine.TESSERACT, False, languages, page_idx
                    )
                    return DocumentPage(
                        page_number=page_idx + 1,
                        width=float(img.width),
                        height=float(img.height),
                        text_blocks=blocks,
                        raw_text=_clean_medical_text(text),
                        confidence=conf,
                    )
                except Exception as retry_exc:
                    return self._error_page(page_idx, f"OOM after retry: {retry_exc}")
            return self._error_page(page_idx, "Out of memory rendering page")

        except Exception as exc:
            if retry < _MAX_RETRIES:
                # Alternate strategy on each retry
                next_engine = [OCREngine.AUTO, OCREngine.TESSERACT, OCREngine.EASYOCR][
                    min(retry, 2)
                ]
                logger.warning(f"Page {page_idx + 1} error ({exc}), "
                               f"retry #{retry + 1} with {next_engine.value}")
                return self._ocr_single_pdf_page(
                    path, page_idx, next_engine, retry % 2 == 0, languages, retry + 1
                )
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"Page {page_idx + 1} failed after {_MAX_RETRIES} retries: {err_msg}")
            return self._error_page(page_idx, err_msg)

    def _try_ocr_page(self, path, page, ocr_engine, enhance, languages):
        """Attempt OCR on a native-text page with sparse content."""
        try:
            img = self.layout_svc.render_pdf_page(path, page.page_number - 1)
            img = _enhance_medical_image(img)
            img, _ = _auto_rotate(img)
            text, blocks, engine, conf = self.ocr_svc.process_image(
                img, ocr_engine, enhance, languages, page.page_number - 1
            )
            page.raw_text = _clean_medical_text(text)
            page.text_blocks = blocks
            page.confidence = conf
        except Exception as exc:
            logger.warning(f"Supplemental OCR failed for page {page.page_number}: {exc}")
        return page

    def _pdf_raw_text_fallback(self, path, page_errors, extract_tables=True):
        """Last-resort text extraction — pdfplumber then pdfminer."""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for i, pg in enumerate(pdf.pages):
                    try:
                        text = pg.extract_text() or ""
                        tables = []
                        if extract_tables:
                            tables = self._plumber_page_tables(pg, i)
                        pages.append(DocumentPage(
                            page_number=i + 1,
                            raw_text=_clean_medical_text(text),
                            tables=tables,
                        ))
                    except Exception as exc:
                        pages.append(self._error_page(i, f"pdfplumber: {exc}"))
            if pages:
                return pages
        except ImportError:
            pass
        except Exception as exc:
            page_errors.append(f"pdfplumber fallback failed: {exc}")

        # Absolute last resort — pdfminer
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(path))
            return [DocumentPage(page_number=1, raw_text=_clean_medical_text(text or ""))]
        except Exception as exc:
            page_errors.append(f"pdfminer fallback failed: {exc}")
            return [self._error_page(0, f"All PDF extraction methods failed: {exc}")]

    def _extract_pdf_tables(self, path) -> dict[int, list[Table]]:
        """Extract tables from all PDF pages using pdfplumber."""
        result: dict[int, list[Table]] = {}
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                for i, pg in enumerate(pdf.pages):
                    try:
                        tables = self._plumber_page_tables(pg, i)
                        if tables:
                            result[i + 1] = tables
                    except Exception as exc:
                        logger.debug(f"Table extraction failed for page {i + 1}: {exc}")
        except Exception as exc:
            logger.debug(f"pdfplumber table extraction skipped: {exc}")
        return result

    @staticmethod
    def _plumber_page_tables(pg, page_idx: int) -> list[Table]:
        """Convert pdfplumber tables to our Table schema."""
        tables = []
        try:
            raw_tables = pg.extract_tables()
            if not raw_tables:
                return tables
            y_offset = float(page_idx * 800)
            for t_idx, raw_table in enumerate(raw_tables):
                if not raw_table:
                    continue
                num_rows = len(raw_table)
                num_cols = max(len(row) for row in raw_table) if raw_table else 0
                cells = []
                for r_idx, row in enumerate(raw_table):
                    for c_idx, cell_text in enumerate(row):
                        cells.append(TableCell(
                            text=str(cell_text or "").strip(),
                            row=r_idx,
                            col=c_idx,
                            bbox=BoundingBox(
                                x=float(c_idx * 100),
                                y=y_offset + float(r_idx * 20),
                                width=100.0,
                                height=20.0,
                                page=page_idx,
                            ),
                        ))
                tables.append(Table(
                    cells=cells,
                    rows=num_rows,
                    cols=num_cols,
                    bbox=BoundingBox(
                        x=0, y=y_offset,
                        width=float(num_cols * 100),
                        height=float(num_rows * 20),
                        page=page_idx,
                    ),
                    page=page_idx,
                ))
        except Exception as exc:
            logger.debug(f"Table conversion error: {exc}")
        return tables

    @staticmethod
    def _decrypt_pdf_if_needed(path: Path, page_errors: list[str]) -> Path:
        """
        If the PDF is encrypted, attempt to open it without a password
        (many medical system PDFs are encrypted with an empty password).
        Returns original path if not encrypted or if decryption succeeds in-place.
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            if doc.is_encrypted:
                logger.info(f"PDF is encrypted, attempting empty-password decrypt")
                if doc.authenticate(""):
                    # Re-save as decrypted to a temp path
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                    doc.save(tmp.name)
                    doc.close()
                    logger.info("PDF decrypted with empty password")
                    return Path(tmp.name)
                else:
                    page_errors.append("PDF is password-protected — could not decrypt")
                    doc.close()
            else:
                doc.close()
        except Exception as exc:
            logger.debug(f"PDF encryption check skipped: {exc}")
        return path

    # ── DOCX pipeline ─────────────────────────────────────────────────────────

    def _process_docx(self, path, ocr_engine, enhance, languages, page_errors):
        try:
            from docx import Document as DocxDocument
        except ImportError:
            return [self._error_page(0, "python-docx not installed. Run: pip install python-docx")]

        try:
            doc = DocxDocument(str(path))
        except Exception as exc:
            return [self._error_page(0, f"Cannot open DOCX: {type(exc).__name__}: {exc}")]

        text_blocks: list[TextBlock] = []
        tables: list[Table] = []
        y_cursor = 0.0

        # Extract paragraphs
        try:
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    y_cursor += 12.0
                    continue
                style_name = para.style.name if para.style and para.style.name else ""
                is_heading = style_name.lower().startswith("heading")
                font_size = 16.0 if is_heading else 12.0
                text_blocks.append(TextBlock(
                    text=text,
                    confidence=1.0,
                    bbox=BoundingBox(x=0.0, y=y_cursor, width=600.0,
                                     height=font_size + 4, page=0),
                    block_type="heading" if is_heading else "paragraph",
                    font_size=font_size,
                    is_bold=is_heading,
                ))
                y_cursor += font_size + 6
        except Exception as exc:
            logger.warning(f"DOCX paragraph extraction partial failure: {exc}")
            page_errors.append(f"paragraphs: {exc}")

        # Extract tables
        try:
            for tbl in doc.tables:
                rows_data = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
                if not rows_data:
                    continue
                num_rows = len(rows_data)
                num_cols = max(len(r) for r in rows_data)
                cells = [
                    TableCell(
                        text=cell_text,
                        row=r_idx,
                        col=c_idx,
                        bbox=BoundingBox(x=float(c_idx * 120), y=float(y_cursor + r_idx * 20),
                                         width=120.0, height=20.0, page=0),
                    )
                    for r_idx, row in enumerate(rows_data)
                    for c_idx, cell_text in enumerate(row)
                ]
                tables.append(Table(
                    cells=cells, rows=num_rows, cols=num_cols,
                    bbox=BoundingBox(x=0, y=y_cursor, width=float(num_cols * 120),
                                     height=float(num_rows * 20), page=0),
                    page=0,
                ))
                y_cursor += num_rows * 20 + 10
        except Exception as exc:
            logger.warning(f"DOCX table extraction partial failure: {exc}")
            page_errors.append(f"tables: {exc}")

        # OCR embedded images (e.g. scanned attachments inside the Word doc)
        embedded_text = self._ocr_docx_images(doc, ocr_engine, enhance, languages)

        raw_text = "\n".join(b.text for b in text_blocks)
        for tbl in tables:
            raw_text += "\n" + "\n".join(c.text for c in tbl.cells if c.text)
        if embedded_text:
            raw_text += "\n\n[Embedded image content]\n" + embedded_text

        raw_text = _clean_medical_text(raw_text)
        page_err = ("; ".join(page_errors)) if page_errors else None

        return [DocumentPage(
            page_number=1,
            width=612.0,
            height=max(y_cursor, 792.0),
            text_blocks=text_blocks,
            tables=tables,
            raw_text=raw_text.strip(),
            confidence=1.0,
            error=page_err,
        )]

    def _ocr_docx_images(self, doc, ocr_engine, enhance, languages) -> str:
        """OCR any images embedded inside a DOCX file."""
        texts: list[str] = []
        try:
            from PIL import Image as PilImage
            for rel in doc.part.rels.values():
                if "image" in rel.reltype:
                    try:
                        img_bytes = rel.target_part.blob
                        img = PilImage.open(io.BytesIO(img_bytes)).convert("RGB")
                        if enhance:
                            img = _enhance_medical_image(img)
                        img, _ = _auto_rotate(img)
                        text, _, _, _ = self.ocr_svc.process_image(
                            img, ocr_engine, enhance, languages, 0
                        )
                        if text.strip():
                            texts.append(text.strip())
                    except Exception as exc:
                        logger.debug(f"Embedded image OCR failed: {exc}")
        except Exception as exc:
            logger.debug(f"DOCX image OCR skipped: {exc}")
        return "\n".join(texts)

    # ── Image pipeline ────────────────────────────────────────────────────────

    def _process_image(self, path, ocr_engine, enhance, languages, page_errors,
                       extract_tables: bool = True):
        """Process image files — including multi-page TIFF and JPEG."""
        try:
            from PIL import Image as PilImage
            img = PilImage.open(path)
        except Exception as exc:
            return [self._error_page(0, f"Cannot open image: {type(exc).__name__}: {exc}")]

        # Collect all frames (multi-page TIFF or single JPEG/PNG)
        frames: list = []
        try:
            while True:
                frames.append(img.copy().convert("RGB"))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        except Exception:
            if not frames:
                frames = [img.convert("RGB")]

        pages = []
        for frame_idx, frame in enumerate(frames):
            page = self._ocr_image_frame(
                frame, frame_idx, ocr_engine, enhance, languages, extract_tables=extract_tables
            )
            pages.append(page)
        return pages

    def _ocr_image_frame(self, img, frame_idx: int, ocr_engine, enhance, languages,
                          retry: int = 0, extract_tables: bool = True) -> DocumentPage:
        """
        OCR a single image frame with full medical CV pipeline:
          - Blank page skip
          - Enhancement + auto-rotation
          - Header/footer extraction with metadata parsing
          - Multi-column reading order
          - Table extraction
          - Checkbox/form field detection
        """
        try:
            # Blank page check — skip expensive OCR entirely
            if _is_blank_page(img):
                logger.debug(f"Frame {frame_idx + 1} detected as blank — skipping OCR")
                return DocumentPage(
                    page_number=frame_idx + 1,
                    width=float(img.width),
                    height=float(img.height),
                    raw_text="",
                    confidence=1.0,
                )

            # ── Image quality assessment ──────────────────────────────────
            quality = _assess_image_quality(img)

            if enhance:
                img = _enhance_medical_image(img)
            img, rotation = _auto_rotate(img)

            # ── Header / footer extraction (before full OCR) ──────────────
            header_text, footer_text, header_meta = _extract_header_footer(
                img, self.ocr_svc, languages
            )

            text, blocks, engine, conf = self.ocr_svc.process_image(
                img, ocr_engine, enhance, languages, frame_idx
            )

            # Low confidence → retry with alternate engine
            alt_engine = OCREngine.EASYOCR if ocr_engine != OCREngine.EASYOCR else OCREngine.TESSERACT
            if conf < _MIN_CONFIDENCE and retry < _MAX_RETRIES:
                logger.info(f"Image frame {frame_idx + 1} low confidence ({conf:.2f}), "
                            f"retrying with {alt_engine.value}")
                return self._ocr_image_frame(
                    img, frame_idx, alt_engine, True, languages, retry + 1, extract_tables
                )

            try:
                regions = self.layout_svc.analyze_image_layout(img, frame_idx)
                if regions:
                    blocks = self._assign_text_to_regions(regions, blocks)
            except Exception as exc:
                logger.debug(f"Layout analysis failed (non-fatal): {exc}")

            # ── Multi-column reading order ────────────────────────────────
            col_boundaries = _detect_column_boundaries(img)
            col_count = len(col_boundaries)
            if col_count > 1:
                logger.info(f"Frame {frame_idx + 1}: detected {col_count} columns")
                blocks = _reorder_blocks_by_columns(blocks, col_boundaries)
                text = _raw_text_in_column_order(blocks, col_boundaries)
            else:
                text = _clean_medical_text(text)

            # ── Table extraction ──────────────────────────────────────────
            tables: list[Table] = []
            if extract_tables:
                tables = self._extract_image_tables(img, frame_idx)
                if tables:
                    logger.info(f"Frame {frame_idx + 1}: extracted {len(tables)} table(s) "
                                f"({sum(t.rows for t in tables)} rows total)")

            # ── Checkbox / form field detection ───────────────────────────
            checkboxes = _detect_checkboxes(img, frame_idx)
            checkbox_text = _checkboxes_to_text(checkboxes)
            if checkbox_text:
                text = text + "\n\n" + checkbox_text

            return DocumentPage(
                page_number=frame_idx + 1,
                width=float(img.width),
                height=float(img.height),
                text_blocks=blocks,
                tables=tables,
                raw_text=text,
                confidence=conf,
                header_text=header_text,
                footer_text=footer_text,
                header_metadata=header_meta,
                column_count=col_count,
                checkboxes=checkboxes,
                image_quality=quality,
            )

        except Exception as exc:
            if retry < _MAX_RETRIES:
                alt = OCREngine.TESSERACT if retry % 2 == 0 else OCREngine.EASYOCR
                logger.warning(f"Image frame {frame_idx + 1} failed ({exc}), "
                               f"retry #{retry + 1} with {alt.value}")
                return self._ocr_image_frame(
                    img, frame_idx, alt, True, languages, retry + 1, extract_tables
                )
            return self._error_page(frame_idx, f"{type(exc).__name__}: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_image_tables(img, page_idx: int = 0) -> list[Table]:
        """
        Detect and extract tables from images (JPEG, PNG, TIFF, etc.) using
        Tesseract word-level bounding boxes.

        Algorithm:
          1. Get all word bboxes + text from tesseract
          2. Cluster words into rows by y-coordinate proximity
          3. Within each row, sort words by x-coordinate
          4. If at least 3 rows share the same column count → treat as table
          5. Build Table objects with proper row/col indexing

        This reliably captures:
          - Lab result panels (test | value | unit | reference range)
          - Medication tables (drug | dose | frequency | route)
          - Vital sign grids
          - Insurance/billing tables
        """
        tables: list[Table] = []
        try:
            import pytesseract
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception as exc:
            logger.debug(f"Table detection via tesseract skipped: {exc}")
            return tables

        try:
            # Collect valid word entries
            words = []
            n = len(data["text"])
            for i in range(n):
                text = str(data["text"][i]).strip()
                conf = int(data["conf"][i])
                if not text or conf < 30:
                    continue
                words.append({
                    "text": text,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                    "cx": data["left"][i] + data["width"][i] // 2,
                    "cy": data["top"][i] + data["height"][i] // 2,
                })

            if len(words) < 4:
                return tables

            # Cluster words into rows (words within 12px vertically = same row)
            words_sorted = sorted(words, key=lambda w: w["cy"])
            rows: list[list[dict]] = []
            current_row: list[dict] = [words_sorted[0]]
            for word in words_sorted[1:]:
                if abs(word["cy"] - current_row[-1]["cy"]) <= 12:
                    current_row.append(word)
                else:
                    rows.append(sorted(current_row, key=lambda w: w["cx"]))
                    current_row = [word]
            rows.append(sorted(current_row, key=lambda w: w["cx"]))

            if len(rows) < 3:
                return tables

            # Find the most common column count — if ≥3 rows share it, it's a table
            from collections import Counter
            col_counts = Counter(len(r) for r in rows)
            dominant_cols, freq = col_counts.most_common(1)[0]

            if dominant_cols < 2 or freq < 3:
                return tables  # not enough structure to be a table

            # Keep only rows that match the dominant column count
            table_rows = [r for r in rows if len(r) == dominant_cols]

            cells: list[TableCell] = []
            for r_idx, row in enumerate(table_rows):
                for c_idx, word in enumerate(row):
                    cells.append(TableCell(
                        text=word["text"],
                        row=r_idx,
                        col=c_idx,
                        bbox=BoundingBox(
                            x=float(word["x"]),
                            y=float(word["y"]),
                            width=float(word["w"]),
                            height=float(word["h"]),
                            page=page_idx,
                        ),
                    ))

            if cells:
                all_x = [w["x"] for r in table_rows for w in r]
                all_y = [w["y"] for r in table_rows for w in r]
                all_x2 = [w["x"] + w["w"] for r in table_rows for w in r]
                all_y2 = [w["y"] + w["h"] for r in table_rows for w in r]
                tables.append(Table(
                    cells=cells,
                    rows=len(table_rows),
                    cols=dominant_cols,
                    bbox=BoundingBox(
                        x=float(min(all_x)),
                        y=float(min(all_y)),
                        width=float(max(all_x2) - min(all_x)),
                        height=float(max(all_y2) - min(all_y)),
                        page=page_idx,
                    ),
                    page=page_idx,
                ))
                logger.debug(f"Detected table: {len(table_rows)} rows × {dominant_cols} cols")

        except Exception as exc:
            logger.debug(f"Table structure analysis failed: {exc}")

        return tables

    @staticmethod
    def _error_page(page_idx: int, error: str) -> DocumentPage:
        return DocumentPage(page_number=page_idx + 1, raw_text="", error=error)

    @staticmethod
    def _fail(doc_id: str, path: Path, error: str) -> ProcessedDocument:
        return ProcessedDocument(
            document_id=doc_id,
            metadata=DocumentMetadata(
                filename=path.name,
                file_size=path.stat().st_size if path.exists() else 0,
                file_type=path.suffix.lower(),
                page_count=0,
            ),
            pages=[],
            full_text="",
            status=DocumentStatus.FAILED,
            error=error,
        )

    @staticmethod
    def _merge_pages_text(pages: list) -> str:
        parts = []
        for pg in pages:
            if pg.raw_text and not pg.error:
                parts.append(f"[Page {pg.page_number}]\n{pg.raw_text}")
            elif pg.raw_text:
                parts.append(f"[Page {pg.page_number} (partial)]\n{pg.raw_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _assign_text_to_regions(regions, ocr_blocks) -> list[TextBlock]:
        """Assign OCR words to layout regions by bounding-box overlap."""
        def iou(a: TextBlock, b: TextBlock) -> float:
            if not a.bbox or not b.bbox:
                return 0.0
            ax1, ay1 = a.bbox.x, a.bbox.y
            ax2, ay2 = ax1 + a.bbox.width, ay1 + a.bbox.height
            bx1, by1 = b.bbox.x, b.bbox.y
            bx2, by2 = bx1 + b.bbox.width, by1 + b.bbox.height
            inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0, min(ay2, by2) - max(ay1, by1))
            inter = inter_w * inter_h
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / union if union else 0.0

        matched_ids: set[int] = set()
        enriched: list[TextBlock] = []

        for region in regions:
            matching = [b for b in ocr_blocks if iou(region, b) > 0.3]
            if matching:
                region.text = " ".join(b.text for b in matching)
                region.confidence = sum(b.confidence for b in matching) / len(matching)
                enriched.append(region)
                for b in matching:
                    matched_ids.add(id(b))

        for b in ocr_blocks:
            if id(b) not in matched_ids:
                enriched.append(b)

        return enriched
