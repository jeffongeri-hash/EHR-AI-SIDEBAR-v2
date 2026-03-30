"""
Layout Extraction Service

Extracts document structure:
- Title / headings / body text hierarchy
- Tables (cells, rows, columns)
- Figures / diagrams
- Columns / reading-order detection
- Page regions

Primary backend: PDFPlumber (native PDF) + LayoutParser / DocTR (images/scans)
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional, Union

import fitz  # PyMuPDF
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

from loguru import logger

from app.models.schemas import (
    BoundingBox,
    DocumentPage,
    Table,
    TableCell,
    TextBlock,
)


# ── PDF Layout Extraction ─────────────────────────────────────────────────────

class PDFLayoutExtractor:
    """
    Extract text blocks, tables, and layout from native PDF files.
    Falls back to image-based OCR for scanned PDFs via the caller.
    """

    _HEADING_FONT_SCALE = 1.3  # font size relative to page median → heading

    def extract(self, pdf_path: Union[str, Path]) -> list[DocumentPage]:
        path = str(pdf_path)
        pages: list[DocumentPage] = []

        try:
            with pdfplumber.open(path) as pdf, fitz.open(path) as mupdf:
                for page_idx, (plumber_page, mu_page) in enumerate(
                    zip(pdf.pages, mupdf)
                ):
                    doc_page = self._process_page(plumber_page, mu_page, page_idx)
                    pages.append(doc_page)
        except Exception as exc:
            logger.error(f"PDF layout extraction failed: {exc}")
            raise

        return pages

    def _process_page(self, plumber_page, mu_page, page_idx: int) -> DocumentPage:
        width = float(plumber_page.width)
        height = float(plumber_page.height)

        # ── Tables ────────────────────────────────────────────────────────────
        tables = self._extract_tables(plumber_page, page_idx)

        # ── Text blocks ───────────────────────────────────────────────────────
        text_blocks = self._extract_text_blocks(mu_page, page_idx)

        # ── Raw text (reading-order) ──────────────────────────────────────────
        raw_text = plumber_page.extract_text() or ""

        return DocumentPage(
            page_number=page_idx + 1,
            width=width,
            height=height,
            text_blocks=text_blocks,
            tables=tables,
            raw_text=raw_text,
            confidence=1.0,
        )

    def _extract_tables(self, plumber_page, page_idx: int) -> list[Table]:
        tables: list[Table] = []
        for tbl in plumber_page.extract_tables():
            if not tbl:
                continue
            cells: list[TableCell] = []
            for r_idx, row in enumerate(tbl):
                for c_idx, cell_text in enumerate(row):
                    cells.append(TableCell(
                        text=(cell_text or "").strip(),
                        row=r_idx,
                        col=c_idx,
                    ))
            bbox_raw = plumber_page.find_tables()[len(tables)].bbox if len(
                plumber_page.find_tables()
            ) > len(tables) else (0, 0, 0, 0)
            tables.append(Table(
                cells=cells,
                rows=len(tbl),
                cols=max(len(r) for r in tbl) if tbl else 0,
                bbox=BoundingBox(
                    x=bbox_raw[0], y=bbox_raw[1],
                    width=bbox_raw[2] - bbox_raw[0],
                    height=bbox_raw[3] - bbox_raw[1],
                    page=page_idx,
                ),
                page=page_idx,
            ))
        return tables

    def _extract_text_blocks(self, mu_page, page_idx: int) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        raw_blocks = mu_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)[
            "blocks"
        ]

        font_sizes: list[float] = []
        for blk in raw_blocks:
            if blk.get("type") != 0:  # 0 = text block
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span.get("size", 12.0))

        median_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 12.0

        for blk in raw_blocks:
            if blk.get("type") != 0:
                continue
            blk_text_parts: list[str] = []
            first_span_size = 12.0
            is_bold = False
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    blk_text_parts.append(span.get("text", ""))
                    first_span_size = span.get("size", 12.0)
                    flags = span.get("flags", 0)
                    is_bold = bool(flags & 2**4)  # bold flag

            full_text = " ".join(blk_text_parts).strip()
            if not full_text:
                continue

            r = blk["bbox"]  # (x0, y0, x1, y1)
            block_type = (
                "title" if first_span_size >= median_size * self._HEADING_FONT_SCALE else "text"
            )

            blocks.append(TextBlock(
                text=full_text,
                confidence=1.0,
                bbox=BoundingBox(
                    x=r[0], y=r[1],
                    width=r[2] - r[0],
                    height=r[3] - r[1],
                    page=page_idx,
                ),
                block_type=block_type,
                font_size=first_span_size,
                is_bold=is_bold,
            ))

        return blocks

    @staticmethod
    def is_scanned(pdf_path: Union[str, Path], sample_pages: int = 3) -> bool:
        """Return True if the PDF appears to be a scan (no native text layer)."""
        try:
            with fitz.open(str(pdf_path)) as doc:
                pages_to_check = min(sample_pages, len(doc))
                text_count = 0
                for i in range(pages_to_check):
                    text = doc[i].get_text().strip()
                    text_count += len(text)
                return text_count < 50 * pages_to_check  # < 50 chars/page → likely scanned
        except Exception:
            return True

    @staticmethod
    def render_page_as_image(pdf_path: Union[str, Path], page_number: int = 0, dpi: int = 300) -> object:
        """Render a PDF page to a PIL Image for OCR processing."""
        with fitz.open(str(pdf_path)) as doc:
            page = doc[page_number]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ── Image Layout Analysis ─────────────────────────────────────────────────────

class ImageLayoutAnalyzer:
    """
    Lightweight layout analysis for images / scanned pages.
    Uses contour-based region detection when DocTR is unavailable.
    """

    def analyze(self, image: object, page_idx: int = 0) -> list[TextBlock]:
        """
        Identify text regions via connected-component analysis.
        Returns approximate bounding boxes (text content filled by OCR).
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("cv2/numpy not available - returning empty layout")
            return []

        arr = np.array(image.convert("L"))
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate to merge nearby characters into words/lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: list[TextBlock] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 5:  # skip noise
                continue
            regions.append(TextBlock(
                text="",  # to be filled by OCR
                confidence=0.0,
                bbox=BoundingBox(x=x, y=y, width=w, height=h, page=page_idx),
                block_type="text",
            ))

        # Sort top-to-bottom, left-to-right (reading order)
        regions.sort(key=lambda b: (b.bbox.y // 20, b.bbox.x))
        return regions


# ── Public API ────────────────────────────────────────────────────────────────

class LayoutService:
    def __init__(self):
        self.pdf_extractor = PDFLayoutExtractor()
        self.image_analyzer = ImageLayoutAnalyzer()

    def extract_pdf_layout(self, pdf_path: Union[str, Path]) -> list[DocumentPage]:
        return self.pdf_extractor.extract(pdf_path)

    def is_scanned_pdf(self, pdf_path: Union[str, Path]) -> bool:
        return PDFLayoutExtractor.is_scanned(pdf_path)

    def render_pdf_page(self, pdf_path: Union[str, Path], page_number: int = 0) -> object:
        return PDFLayoutExtractor.render_page_as_image(pdf_path, page_number)

    def get_pdf_page_count(self, pdf_path: Union[str, Path]) -> int:
        with fitz.open(str(pdf_path)) as doc:
            return len(doc)

    def analyze_image_layout(self, image: object, page_idx: int = 0) -> list[TextBlock]:
        return self.image_analyzer.analyze(image, page_idx)
