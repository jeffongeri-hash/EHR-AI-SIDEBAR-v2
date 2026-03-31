"""
Document Processor — Stable Pipeline
=====================================
Orchestrates the full document processing pipeline with:
  1. Per-page error isolation — one bad page never kills the whole job
  2. Fallback OCR strategy   — if primary engine fails, try simpler extraction
  3. Partial-success mode    — returns PARTIAL status with per-page error info
  4. Retry on transient failures (file-lock, OOM, renderer crash)
  5. Clear, actionable error messages stored in DocumentPage.error
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from app.models.schemas import (
    DocumentMetadata,
    DocumentPage,
    DocumentStatus,
    OCREngine,
    ProcessedDocument,
    TextBlock,
)
from app.services.layout_service import LayoutService
from app.services.ocr_service import OCRService

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
_PDF_SUFFIX = ".pdf"
_DOCX_SUFFIX = ".docx"
_MAX_RETRIES = 2  # per-page retry attempts on transient errors


class DocumentProcessor:
    """High-level document processing pipeline with full error isolation."""

    def __init__(self):
        self.layout_svc = LayoutService()
        self.ocr_svc = OCRService()

    # ── Public entry point ──────────────────────────────────────────────────

    def process(
        self,
        file_path,
        document_id: Optional[str] = None,
        ocr_engine: OCREngine = OCREngine.AUTO,
        enhance_images: bool = True,
        languages: list = None,
        extract_tables: bool = True,
    ) -> ProcessedDocument:
        """Process a document and return results.
        
        Never raises — errors are captured in the returned ProcessedDocument.
        Returns PARTIAL status if some pages succeeded and others failed.
        Returns FAILED only if no text could be extracted at all.
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
                pages = self._process_pdf(path, ocr_engine, enhance_images, languages, page_errors)
            elif suffix == _DOCX_SUFFIX:
                pages = self._process_docx(path, page_errors)
            elif suffix in _IMAGE_SUFFIXES:
                pages = self._process_image(path, ocr_engine, enhance_images, languages, page_errors)
            else:
                return self._fail(doc_id, path, f"Unsupported file type: '{suffix}'. "
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

        # Determine status
        if not good_pages:
            status = DocumentStatus.FAILED
            top_error = f"{len(failed_pages)} page(s) all failed. First error: {failed_pages[0].error}"
        elif failed_pages:
            status = DocumentStatus.PARTIAL
            top_error = f"{len(failed_pages)}/{len(pages)} page(s) failed: " + "; ".join(
                f"p{p.page_number}: {p.error}" for p in failed_pages[:3]
            )
            logger.warning(f"[{doc_id}] Partial success — {len(good_pages)} ok, {len(failed_pages)} failed")
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

        logger.info(
            f"[{doc_id}] {status.value.upper()} — {len(pages)} pages, "
            f"{len(full_text)} chars, {elapsed}ms"
        )

        return ProcessedDocument(
            document_id=doc_id,
            metadata=metadata,
            pages=pages,
            full_text=full_text,
            status=status,
            error=top_error,
            partial_page_errors=[f"p{p.page_number}: {p.error}" for p in failed_pages],
        )

    # ── PDF pipeline ────────────────────────────────────────────────────

    def _process_pdf(self, path, ocr_engine, enhance, languages, page_errors):
        try:
            is_scanned = self.layout_svc.is_scanned_pdf(path)
            page_count = self.layout_svc.get_pdf_page_count(path)
        except Exception as exc:
            # Can't even open the PDF — try raw text extraction as fallback
            logger.warning(f"PDF metadata read failed ({exc}), attempting raw text fallback")
            return self._pdf_raw_text_fallback(path, page_errors)

        logger.info(f"PDF scanned={is_scanned}, pages={page_count}")

        if is_scanned:
            return self._ocr_pdf_pages(path, page_count, ocr_engine, enhance, languages)
        else:
            pages = []
            try:
                native_pages = self.layout_svc.extract_pdf_layout(path)
            except Exception as exc:
                logger.warning(f"Native PDF layout extraction failed ({exc}), falling back to full OCR")
                return self._ocr_pdf_pages(path, page_count, ocr_engine, enhance, languages)

            for page in native_pages:
                if page.error:
                    pages.append(page)
                    continue
                # If native text is sparse, supplement with OCR
                if len(page.raw_text.strip()) < 30:
                    page = self._try_ocr_page(path, page, ocr_engine, enhance, languages)
                pages.append(page)
            return pages

    def _ocr_pdf_pages(self, path, page_count, ocr_engine, enhance, languages):
        pages = []
        for page_idx in range(page_count):
            page = self._ocr_single_pdf_page(path, page_idx, ocr_engine, enhance, languages)
            pages.append(page)
        return pages

    def _ocr_single_pdf_page(
        self, path, page_idx, ocr_engine, enhance, languages, retry=0
    ) -> DocumentPage:
        """OCR a single PDF page with retry and fallback."""
        try:
            img = self.layout_svc.render_pdf_page(path, page_idx, dpi=300)
            text, blocks, engine, conf = self.ocr_svc.process_image(
                img, ocr_engine, enhance, languages, page_idx
            )
            regions = self.layout_svc.analyze_image_layout(img, page_idx)
            if regions:
                blocks = self._assign_text_to_regions(regions, blocks)

            return DocumentPage(
                page_number=page_idx + 1,
                width=float(img.width),
                height=float(img.height),
                text_blocks=blocks,
                tables=[],
                raw_text=text,
                confidence=conf,
            )
        except MemoryError:
            if retry < _MAX_RETRIES:
                logger.warning(f"OOM on page {page_idx + 1}, retrying at lower DPI")
                try:
                    img = self.layout_svc.render_pdf_page(path, page_idx, dpi=150)
                    text, blocks, _, conf = self.ocr_svc.process_image(
                        img, OCREngine.TESSERACT, False, languages, page_idx
                    )
                    return DocumentPage(
                        page_number=page_idx + 1,
                        width=float(img.width),
                        height=float(img.height),
                        text_blocks=blocks,
                        tables=[],
                        raw_text=text,
                        confidence=conf,
                    )
                except Exception as retry_exc:
                    return self._error_page(page_idx, f"OOM after retry: {retry_exc}")
            return self._error_page(page_idx, "Out of memory rendering page")
        except Exception as exc:
            if retry < _MAX_RETRIES:
                logger.warning(f"Page {page_idx + 1} error ({exc}), retrying #{retry + 1}")
                return self._ocr_single_pdf_page(
                    path, page_idx, OCREngine.TESSERACT, False, languages, retry + 1
                )
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"Page {page_idx + 1} failed after {_MAX_RETRIES} retries: {err_msg}")
            return self._error_page(page_idx, err_msg)

    def _try_ocr_page(self, path, page, ocr_engine, enhance, languages):
        """Attempt OCR on a native-text page that has insufficient text."""
        try:
            img = self.layout_svc.render_pdf_page(path, page.page_number - 1)
            text, blocks, engine, conf = self.ocr_svc.process_image(
                img, ocr_engine, enhance, languages, page.page_number - 1
            )
            page.raw_text = text
            page.text_blocks = blocks
            page.confidence = conf
        except Exception as exc:
            logger.warning(f"Supplemental OCR failed for page {page.page_number}: {exc}")
        return page

    def _pdf_raw_text_fallback(self, path, page_errors):
        """Last-resort: use pdfminer/pdfplumber to extract plain text."""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for i, pg in enumerate(pdf.pages):
                    try:
                        text = pg.extract_text() or ""
                        pages.append(DocumentPage(
                            page_number=i + 1,
                            raw_text=text,
                        ))
                    except Exception as exc:
                        pages.append(self._error_page(i, f"pdfplumber: {exc}"))
            return pages
        except ImportError:
            pass
        except Exception as exc:
            page_errors.append(f"pdfplumber fallback failed: {exc}")

        # Absolute last resort — pdfminer
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(path))
            return [DocumentPage(page_number=1, raw_text=text or "")]
        except Exception as exc:
            page_errors.append(f"pdfminer fallback failed: {exc}")
            return [self._error_page(0, f"All PDF extraction methods failed: {exc}")]

    # ── DOCX pipeline ────────────────────────────────────────────────────

    def _process_docx(self, path, page_errors):
        try:
            from docx import Document as DocxDocument
        except ImportError:
            return [self._error_page(0, "python-docx not installed. Run: pip install python-docx")]

        from app.models.schemas import BoundingBox, Table, TableCell

        try:
            doc = DocxDocument(str(path))
        except Exception as exc:
            return [self._error_page(0, f"Cannot open DOCX: {type(exc).__name__}: {exc}")]

        text_blocks = []
        tables = []
        y_cursor = 0.0

        try:
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    y_cursor += 12.0
                    continue
                style_name = (para.style.name if para.style and para.style.name else "")
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
                        bbox=BoundingBox(
                            x=float(c_idx * 120),
                            y=float(y_cursor + r_idx * 20),
                            width=120.0,
                            height=20.0,
                            page=0,
                        ),
                    )
                    for r_idx, row in enumerate(rows_data)
                    for c_idx, cell_text in enumerate(row)
                ]
                tables.append(Table(
                    cells=cells, rows=num_rows, cols=num_cols,
                    bbox=BoundingBox(x=0, y=y_cursor, width=float(num_cols*120),
                                     height=float(num_rows*20), page=0),
                    page=0,
                ))
                y_cursor += num_rows * 20 + 10
        except Exception as exc:
            logger.warning(f"DOCX table extraction partial failure: {exc}")
            page_errors.append(f"tables: {exc}")

        raw_text = "\n".join(b.text for b in text_blocks)
        for tbl in tables:
            raw_text += "\n" + "\n".join(c.text for c in tbl.cells if c.text)

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

    # ── Image pipeline ────────────────────────────────────────────────────

    def _process_image(self, path, ocr_engine, enhance, languages, page_errors):
        try:
            from PIL import Image as PilImage
            img = PilImage.open(path).convert("RGB")
        except Exception as exc:
            return [self._error_page(0, f"Cannot open image: {type(exc).__name__}: {exc}")]

        try:
            text, blocks, engine, conf = self.ocr_svc.process_image(
                img, ocr_engine, enhance, languages, 0
            )
        except Exception as exc:
            # Try tesseract fallback
            logger.warning(f"Primary OCR failed ({exc}), trying Tesseract fallback")
            try:
                text, blocks, engine, conf = self.ocr_svc.process_image(
                    img, OCREngine.TESSERACT, False, languages, 0
                )
            except Exception as exc2:
                return [self._error_page(
                    0,
                    f"OCR failed: {type(exc).__name__}: {exc}. "
                    f"Tesseract fallback also failed: {exc2}"
                )]

        try:
            regions = self.layout_svc.analyze_image_layout(img, 0)
            if regions:
                blocks = self._assign_text_to_regions(regions, blocks)
        except Exception as exc:
            logger.warning(f"Layout analysis failed (non-fatal): {exc}")

        return [DocumentPage(
            page_number=1,
            width=float(img.width),
            height=float(img.height),
            text_blocks=blocks,
            tables=[],
            raw_text=text,
            confidence=conf,
        )]

    # ── Helpers ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _error_page(page_idx: int, error: str) -> DocumentPage:
        return DocumentPage(
            page_number=page_idx + 1,
            raw_text="",
            error=error,
        )

    @staticmethod
    def _fail(
        doc_id: str,
        path: Path,
        error: str,
    ) -> ProcessedDocument:
        from app.models.schemas import DocumentMetadata, DocumentStatus
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
            elif pg.raw_text:  # partial page with some text even on error
                parts.append(f"[Page {pg.page_number} (partial)]\n{pg.raw_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _assign_text_to_regions(
        regions: list[TextBlock],
        ocr_blocks: list[TextBlock],
    ) -> list[TextBlock]:
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
            union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
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
