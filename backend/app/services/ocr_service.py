"""
OCR Service - Multi-engine OCR with image enhancement for low-quality scans.

Supports:
- EasyOCR (primary, handles degraded images well)
- Tesseract (fallback, good for clean documents)
- Auto-selection based on image quality assessment

Image enhancement pipeline for low-quality scans:
1. Noise reduction (denoising)
2. Deskewing (rotation correction)
3. Binarization (Otsu / adaptive threshold)
4. Contrast enhancement (CLAHE)
5. Super-resolution upscaling (optional)
"""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageEnhance = None
    ImageFilter = None

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    pytesseract = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

from pathlib import Path
from typing import Optional, Tuple, Union
import time
import logging
from loguru import logger

from app.config import settings
from app.models.schemas import TextBlock, BoundingBox, OCREngine

# EasyOCR reader is expensive to init – cache it as a module-level singleton
_easyocr_reader = None


def _get_easyocr_reader(languages: list):
    global _easyocr_reader
    if not EASYOCR_AVAILABLE:
        raise ImportError("EasyOCR not available")
    if _easyocr_reader is None:
        logger.info("Initialising EasyOCR reader (first call)…")
        _easyocr_reader = easyocr.Reader(languages, gpu=False)
    return _easyocr_reader


# ── Image Enhancement ─────────────────────────────────────────────────────────

class ImageEnhancer:
    """Pre-processing pipeline that improves OCR accuracy on low-quality scans."""

    @staticmethod
    def to_numpy(image: object) -> object:
        return np.array(image.convert("RGB"))

    @staticmethod
    def to_pil(arr: object) -> object:
        return Image.fromarray(arr)

    # ── Quality assessment ────────────────────────────────────────────────────

    @staticmethod
    def estimate_quality(image: object) -> float:
        """Return a quality score in [0, 1].  Lower = worse quality."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalise: typical good doc ~ 500, blurry doc ~ 50
        score = min(1.0, laplacian_var / 500.0)
        return float(score)

    # ── Enhancement steps ─────────────────────────────────────────────────────

    @staticmethod
    def denoise(image: object) -> object:
        """Non-local means denoising."""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    @staticmethod
    def deskew(image: object) -> object:
        """Detect and correct document skew."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 5:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 0.5:  # skip tiny rotations
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    @staticmethod
    def enhance_contrast(image: object) -> object:
        """CLAHE contrast enhancement on the L channel (LAB colour space)."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        enhanced = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    @staticmethod
    def binarize(image: object) -> object:
        """Adaptive thresholding for uneven illumination."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        # Convert back to RGB for compatibility
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def upscale(image: object, scale: float = 2.0) -> object:
        """Simple bicubic upscaling to improve OCR on tiny text."""
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def remove_shadows(image: object) -> object:
        """Remove shadow effects from scanned documents."""
        rgb_planes = cv2.split(image)
        result_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(norm)
        return cv2.merge(result_planes)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def enhance(self, image: object, aggressive: bool = False) -> object:
        """Run the full enhancement pipeline."""
        arr = self.to_numpy(image)
        quality = self.estimate_quality(arr)
        logger.debug(f"Image quality score: {quality:.3f}")

        # Always apply
        arr = self.denoise(arr)
        arr = self.deskew(arr)
        arr = self.remove_shadows(arr)
        arr = self.enhance_contrast(arr)

        # For very low-quality images
        if quality < 0.3 or aggressive:
            arr = self.upscale(arr, scale=2.0)
            arr = self.binarize(arr)

        return self.to_pil(arr)


# ── OCR Engines ───────────────────────────────────────────────────────────────

class TesseractOCR:
    def __init__(self):
        if settings.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH

    def run(
        self, image: object, languages: list = None
    ) -> tuple:
        lang_str = "+".join(languages or settings.OCR_LANGUAGES)
        config = "--oem 3 --psm 6"

        # Full text
        text = pytesseract.image_to_string(image, lang=lang_str, config=config)

        # Bounding boxes
        data = pytesseract.image_to_data(
            image, lang=lang_str, config=config,
            output_type=pytesseract.Output.DICT
        )

        blocks: list
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            conf = float(data["conf"][i])
            if conf < 0:
                continue
            blocks.append(TextBlock(
                text=word,
                confidence=conf / 100.0,
                bbox=BoundingBox(
                    x=data["left"][i],
                    y=data["top"][i],
                    width=data["width"][i],
                    height=data["height"][i],
                    page=0,
                ),
            ))

        return text.strip(), blocks


class EasyOCREngine:
    def run(
        self, image: object, languages: list = None
    ) -> tuple:
        langs = languages or settings.OCR_LANGUAGES
        reader = _get_easyocr_reader(langs)

        arr = np.array(image)
        results = reader.readtext(arr, detail=1, paragraph=False)

        blocks: list
        lines: list

        for bbox_pts, text, conf in results:
            if conf < settings.OCR_CONFIDENCE_THRESHOLD:
                continue
            # bbox_pts: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            blocks.append(TextBlock(
                text=text.strip(),
                confidence=float(conf),
                bbox=BoundingBox(
                    x=min(xs), y=min(ys),
                    width=max(xs) - min(xs),
                    height=max(ys) - min(ys),
                    page=0,
                ),
            ))
            lines.append(text.strip())

        return "\n".join(lines), blocks


# ── Main OCR Service ──────────────────────────────────────────────────────────

class OCRService:
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.tesseract = TesseractOCR()
        self.easyocr_engine = EasyOCREngine()

    def _select_engine(self, image: object, requested: OCREngine) -> OCREngine:
        if requested != OCREngine.AUTO:
            return requested
        quality = ImageEnhancer.estimate_quality(image)
        # EasyOCR handles degraded images better; Tesseract is faster on clean docs
        return OCREngine.EASYOCR if quality < 0.6 else OCREngine.TESSERACT

    def process_image(
        self,
        image: object,
        engine: OCREngine = OCREngine.AUTO,
        enhance: bool = True,
        languages: list = None,
        page_number: int = 0,
    ) -> tuple:
        """
        Returns (full_text, blocks, engine_used, confidence).
        """
        start = time.time()

        arr = ImageEnhancer.to_numpy(image)
        chosen_engine = self._select_engine(arr, engine)

        if enhance:
            quality = ImageEnhancer.estimate_quality(arr)
            image = self.enhancer.enhance(image, aggressive=(quality < 0.2))

        if chosen_engine == OCREngine.EASYOCR:
            text, blocks = self.easyocr_engine.run(image, languages)
            engine_name = "easyocr"
        else:
            text, blocks = self.tesseract.run(image, languages)
            engine_name = "tesseract"

        # Assign page numbers
        for blk in blocks:
            blk.bbox.page = page_number

        avg_conf = (
            sum(b.confidence for b in blocks) / len(blocks) if blocks else 0.0
        )

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            f"OCR [{engine_name}] page={page_number} "
            f"words={len(blocks)} conf={avg_conf:.2f} time={elapsed_ms}ms"
        )

        return text, blocks, engine_name, avg_conf

    def process_image_path(
        self,
        path: Union[str, Path],
        engine: OCREngine = OCREngine.AUTO,
        enhance: bool = True,
        languages: list = None,
    ) -> tuple:
        image = Image.open(path).convert("RGB")
        return self.process_image(image, engine, enhance, languages)
