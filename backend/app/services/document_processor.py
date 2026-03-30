"""
Document Processor

Orchestrates the full document processing pipeline:
  1. Detect file type (PDF vs image)
  2. For PDFs: check if scanned or native
  3. Extract layout (native PDF text OR OCR from rendered pages)
  4. Consolidate pages into a ProcessedDocument
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

# File type helpers
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
_PDF_SUFFIX = ".pdf"
_DOCX_SUFFIX = ".docx"


class DocumentProcessor:
    """High-level document processing pipeline."""

    def __init__(self):
        self.layout_svc = LayoutService()
        self.ocr_svc = OCRService()

    # ── Public entry point ────────────────────────────────────────────────────

    def process(
        self,
        file_path: str | Path,
        document_id: Optional[str] = None,
        ocr_engine: OCREngine = OCREngine.AUTO,
        enhance_images: bool = True,
        languages: list = None,
        extract_tables: bool = True,
    ) -> ProcessedDocument:
        if not file_path:
            raise ValueError("file_path cannot be None or empty")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        
        doc_id = document_id or str(uuid.uuid4())
        suffix = path.suffix.lower()
        start = time.time()

        logger.info(f"[{doc_id}] Processing '{path.name}' …")

        try:
            if suffix == _PDF_SUFFIX:
                pages = self._process_pdf(path, ocr_engine, enhance_images, languages)
            elif suffix == _DOCX_SUFFIX:
                pages = self._process_docx(path)
            elif suffix in _IMAGE_SUFFIXES:
                pages = self._process_image(path, ocr_engine, enhance_images, languages)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            full_text = self._merge_pages_text(pages)
            elapsed = int((time.time() - start) * 1000)

            metadata = DocumentMetadata(
                filename=path.name,
                file_size=path.stat().st_size,
                file_type=suffix,
                page_count=len(pages),
                ocr_engine=ocr_engine.value,
                processing_time_ms=elapsed,
            )

            logger.info(
                f"[{doc_id}] Done — {len(pages)} pages, "
                f"{len(full_text)} chars, {elapsed}ms"
            )

            return ProcessedDocument(
                document_id=doc_id,
                metadata=metadata,
                pages=pages,
                full_text=full_text,
                status=DocumentStatus.COMPLETED,
            )

        except Exception as exc:
            logger.exception(f"[{doc_id}] Processing failed: {exc}")
            return ProcessedDocument(
                document_id=doc_id,
                metadata=DocumentMetadata(
                    filename=path.name,
                    file_size=path.stat().st_size if path.exists() else 0,
                    file_type=suffix,
                    page_count=0,
                ),
                pages=[],
                full_text="",
                status=DocumentStatus.FAILED,
                error=str(exc),
            )

    # ── PDF pipeline ──────────────────────────────────────────────────────────

    def _process_pdf(
        self,
        path: Path,
        ocr_engine: OCREngine,
        enhance: bool,
        languages: list | None,
    ) -> list[DocumentPage]:
        is_scanned = self.layout_svc.is_scanned_pdf(path)
        page_count = self.layout_svc.get_pdf_page_count(path)
        logger.info(f"PDF scanned={is_scanned}, pages={page_count}")

        if is_scanned:
            # Render each page as an image and OCR it
            return self._ocr_pdf_pages(path, page_count, ocr_engine, enhance, languages)
        else:
            # Extract native text + fall back to OCR for pages with little text
            pages = self.layout_svc.extract_pdf_layout(path)
            enriched = []
            for page in pages:
                if len(page.raw_text.strip()) < 30:
                    # This page might be an embedded image
                    img = self.layout_svc.render_pdf_page(path, page.page_number - 1)
                    text, blocks, engine, conf = self.ocr_svc.process_image(
                        img, ocr_engine, enhance, languages, page.page_number - 1
                    )
                    page.raw_text = text
                    page.text_blocks = blocks
                    page.confidence = conf
                enriched.append(page)
            return enriched

    def _ocr_pdf_pages(
        self,
        path: Path,
        page_count: int,
        ocr_engine: OCREngine,
        enhance: bool,
        languages: list | None,
    ) -> list[DocumentPage]:
        pages: list[DocumentPage] = []
        for page_idx in range(page_count):
            img = self.layout_svc.render_pdf_page(path, page_idx, dpi=300)
            text, blocks, engine, conf = self.ocr_svc.process_image(
                img, ocr_engine, enhance, languages, page_idx
            )
            # Also attempt layout region detection
            regions = self.layout_svc.analyze_image_layout(img, page_idx)
            # Merge layout regions with OCR blocks
            if regions:
                blocks = self._assign_text_to_regions(regions, blocks)

            pages.append(DocumentPage(
                page_number=page_idx + 1,
                width=float(img.width),
                height=float(img.height),
                text_blocks=blocks,
                tables=[],
                raw_text=text,
                confidence=conf,
            ))
        return pages

    # ── DOCX pipeline ─────────────────────────────────────────────────────────

    def _process_docx(self, path: Path) -> list[DocumentPage]:
        """Extract text and tables from a .docx file using python-docx."""
        try:
            from docx import Document as DocxDocument
            from docx.oxml.ns import qn
        except ImportError:
            logger.warning("python-docx not available; cannot process .docx files")
            raise ValueError("DOCX processing not available: python-docx package not installed")
        
        from app.models.schemas import BoundingBox, Table, TableCell

        doc = DocxDocument(str(path))
        text_blocks: list[TextBlock] = []
        tables: list[Table] = []

        # --- Paragraphs ---
        y_cursor = 0.0
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                y_cursor += 12.0
                continue
            # Handle case where para.style might be None
            style_name = para.style.name if para.style and para.style.name else ""
            is_heading = style_name.lower().startswith("heading")
            font_size = 16.0 if is_heading else 12.0
            text_blocks.append(
                TextBlock(
                    text=text,
                    confidence=1.0,
                    bbox=BoundingBox(x=0.0, y=y_cursor, width=600.0, height=font_size + 4, page=0),
                    block_type="heading" if is_heading else "paragraph",
                    font_size=font_size,
                    is_bold=is_heading,
                )
            )
            y_cursor += font_size + 6

        # --- Tables ---
        for tbl_idx, tbl in enumerate(doc.tables):
            rows_data: list[list] = []
            for row in tbl.rows:
                rows_data.append([cell.text.strip() for cell in row.cells])

            if not rows_data:
                continue

            num_rows = len(rows_data)
            num_cols = max(len(r) for r in rows_data)
            cells: list[TableCell] = []
            for r_idx, row in enumerate(rows_data):
                for c_idx, cell_text in enumerate(row):
                    cells.append(
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
                    )
            tables.append(
                Table(
                    cells=cells,
                    rows=num_rows,
                    cols=num_cols,
                    bbox=BoundingBox(x=0.0, y=y_cursor, width=float(num_cols * 120), height=float(num_rows * 20), page=0),
                    page=0,
                )
            )
            y_cursor += num_rows * 20 + 10

        raw_text = "\n".join(b.text for b in text_blocks)
        # Append table text
        for tbl in tables:
            raw_text += "\n" + "\n".join(c.text for c in tbl.cells if c.text)

        return [DocumentPage(
            page_number=1,
            width=612.0,
            height=max(y_cursor, 792.0),
            text_blocks=text_blocks,
            tables=tables,
            raw_text=raw_text.strip(),
            confidence=1.0,
        )]

    # ── Image pipeline ────────────────────────────────────────────────────────

    def _process_image(
        self,
        path: Path,
        ocr_engine: OCREngine,
        enhance: bool,
        languages: list | None,
    ) -> list[DocumentPage]:
        from PIL import Image as PilImage
        img = PilImage.open(path).convert("RGB")

        text, blocks, engine, conf = self.ocr_svc.process_image(
            img, ocr_engine, enhance, languages, 0
        )
        regions = self.layout_svc.analyze_image_layout(img, 0)
        if regions:
            blocks = self._assign_text_to_regions(regions, blocks)

        return [DocumentPage(
            page_number=1,
            width=float(img.width),
            height=float(img.height),
            text_blocks=blocks,
            tables=[],
            raw_text=text,
            confidence=conf,
        )]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_pages_text(pages: list[DocumentPage]) -> str:
        parts = []
        for pg in pages:
            if pg.raw_text:
                parts.append(f"[Page {pg.page_number}]\n{pg.raw_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _assign_text_to_regions(
        regions: list[TextBlock],
        ocr_blocks: list[TextBlock],
    ) -> list[TextBlock]:
        """
        Assign OCR words to layout regions based on bounding box overlap.
        Returns enriched region blocks.
        """
        def iou(a: TextBlock, b: TextBlock) -> float:
            ax1, ay1 = a.bbox.x, a.bbox.y
            ax2, ay2 = ax1 + a.bbox.width, ay1 + a.bbox.height
            bx1, by1 = b.bbox.x, b.bbox.y
            bx2, by2 = bx1 + b.bbox.width, by1 + b.bbox.height
            inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
            inter_h = max(0, min(ay2, by2) - max(ay1, by1))
            inter = inter_w * inter_h
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / union if union else 0.0

        for region in regions:
            matching = [
                blk for blk in ocr_blocks
                if iou(region, blk) > 0.3
            ]
            if matching:
                region.text = " ".join(b.text for b in matching)
                region.confidence = sum(b.confidence for b in matching) / len(matching)

        # Return only enriched regions (with text) + any unmatched OCR blocks
        matched_ids = set()
        enriched: list[TextBlock] = []
        for region in regions:
            if region.text:
                enriched.append(region)
                # Mark ocr blocks used
                for blk in ocr_blocks:
                    if iou(region, blk) > 0.3:
                        matched_ids.add(id(blk))

        for blk in ocr_blocks:
            if id(blk) not in matched_ids:
                enriched.append(blk)

        return enriched
