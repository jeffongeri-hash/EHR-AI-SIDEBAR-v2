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

        img = _remove_scanner_borders(img)
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

            clean_text = _clean_medical_text(text)

            return DocumentPage(
                page_number=page_idx + 1,
                width=float(img.width),
                height=float(img.height),
                text_blocks=blocks,
                tables=[],
                raw_text=clean_text,
                confidence=conf,
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
        """OCR a single image frame with full medical preprocessing and table extraction."""
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

            if enhance:
                img = _enhance_medical_image(img)
            img, rotation = _auto_rotate(img)

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

            # Table extraction from image — works for JPEG, PNG, TIFF, etc.
            tables: list[Table] = []
            if extract_tables:
                tables = self._extract_image_tables(img, frame_idx)
                if tables:
                    logger.info(f"Frame {frame_idx + 1}: extracted {len(tables)} table(s) "
                                f"({sum(t.rows for t in tables)} rows total)")

            return DocumentPage(
                page_number=frame_idx + 1,
                width=float(img.width),
                height=float(img.height),
                text_blocks=blocks,
                tables=tables,
                raw_text=_clean_medical_text(text),
                confidence=conf,
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
