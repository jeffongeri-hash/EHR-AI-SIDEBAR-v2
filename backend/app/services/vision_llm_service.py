"""
Vision LLM Service — Medical Document Vision Pipeline
======================================================
Multi-tier vision pipeline.  Open-source models are tried in order of
quality; Claude Vision is the last resort and is always flagged.

OCR tier (text extraction from images)
  1. Surya        — SOTA open OCR, best on scanned/printed medical records
  2. DocTR        — transformer OCR, strong on forms and tables
  3. Nougat       — Meta model specialised for scientific/medical PDFs

VLM tier (image understanding + text extraction when OCR confidence is low)
  4. InternVL2-8B — best open-source for dense document layout
  5. Qwen2-VL-7B  — best at text-heavy images and structured reports
  6. LLaVA 1.6-7B — solid general VQA / visual document understanding

API tier (last resort — always flagged in the UI)
  7. Claude Vision — Anthropic claude-3-5-sonnet vision API

Each engine returns:
    VisionResult(text, engine_name, claude_used, confidence)

All models are lazy-loaded on first call and cached as singletons.
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class VisionResult:
    text: str
    engine_name: str
    claude_used: bool = False
    confidence: float = 0.0
    error: Optional[str] = None


# ── Lazy singleton registry ───────────────────────────────────────────────────

_surya_models: Optional[dict] = None
_doctr_model = None
_nougat_processor = None
_nougat_model = None
_internvl_model = None
_internvl_tokenizer = None
_qwen_model = None
_qwen_processor = None
_llava_model = None
_llava_processor = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OCR TIER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 1. Surya ──────────────────────────────────────────────────────────────────

def _load_surya():
    global _surya_models
    if _surya_models is not None:
        return _surya_models
    try:
        from surya.model.detection.model import load_model as load_det
        from surya.model.detection.processor import load_processor as load_det_proc
        from surya.model.recognition.model import load_model as load_rec
        from surya.model.recognition.processor import load_processor as load_rec_proc
        _surya_models = {
            "det_model":     load_det(),
            "det_processor": load_det_proc(),
            "rec_model":     load_rec(),
            "rec_processor": load_rec_proc(),
        }
        logger.info("Surya OCR models loaded")
    except Exception as exc:
        logger.warning(f"Surya not available: {exc}")
        _surya_models = {}
    return _surya_models


def run_surya(img) -> VisionResult:
    """Extract text using Surya SOTA OCR."""
    try:
        models = _load_surya()
        if not models:
            return VisionResult("", "surya", error="Surya not installed")

        from surya.ocr import run_ocr
        results = run_ocr(
            [img],
            [["en"]],
            models["det_model"],
            models["det_processor"],
            models["rec_model"],
            models["rec_processor"],
        )
        lines = []
        total_conf = 0.0
        count = 0
        for page in results:
            for line in page.text_lines:
                lines.append(line.text)
                total_conf += line.confidence
                count += 1
        text = "\n".join(lines)
        conf = (total_conf / count) if count else 0.0
        return VisionResult(text=text, engine_name="surya", confidence=conf)
    except Exception as exc:
        return VisionResult("", "surya", error=str(exc))


# ── 2. DocTR ──────────────────────────────────────────────────────────────────

def _load_doctr():
    global _doctr_model
    if _doctr_model is not None:
        return _doctr_model
    try:
        from doctr.models import ocr_predictor
        _doctr_model = ocr_predictor(pretrained=True)
        logger.info("DocTR model loaded")
    except Exception as exc:
        logger.warning(f"DocTR not available: {exc}")
        _doctr_model = False
    return _doctr_model


def run_doctr(img) -> VisionResult:
    """Extract text using DocTR transformer OCR."""
    try:
        model = _load_doctr()
        if not model:
            return VisionResult("", "doctr", error="DocTR not installed")

        import numpy as np
        from doctr.io import DocumentFile

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        doc = DocumentFile.from_images([buf.read()])
        result = model(doc)

        lines = []
        total_conf = 0.0
        count = 0
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words]
                    confs = [w.confidence for w in line.words]
                    if words:
                        lines.append(" ".join(words))
                        total_conf += sum(confs)
                        count += len(confs)

        text = "\n".join(lines)
        conf = (total_conf / count) if count else 0.0
        return VisionResult(text=text, engine_name="doctr", confidence=conf)
    except Exception as exc:
        return VisionResult("", "doctr", error=str(exc))


# ── 3. Nougat ─────────────────────────────────────────────────────────────────

def _load_nougat():
    global _nougat_processor, _nougat_model
    if _nougat_model is not None:
        return _nougat_processor, _nougat_model
    try:
        from transformers import NougatProcessor, VisionEncoderDecoderModel
        _nougat_processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        _nougat_model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        logger.info("Nougat model loaded")
    except Exception as exc:
        logger.warning(f"Nougat not available: {exc}")
        _nougat_processor = False
        _nougat_model = False
    return _nougat_processor, _nougat_model


def run_nougat(img) -> VisionResult:
    """Extract text using Nougat (specialised for scientific/medical PDFs)."""
    try:
        processor, model = _load_nougat()
        if not model:
            return VisionResult("", "nougat", error="Nougat not installed")

        import torch
        pixel_values = processor(img, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=1024,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
            )
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        text = processor.post_process_generation(text, fix_markdown=False)
        return VisionResult(text=text, engine_name="nougat", confidence=0.8)
    except Exception as exc:
        return VisionResult("", "nougat", error=str(exc))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VLM TIER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MEDICAL_PROMPT = (
    "You are a medical document OCR assistant. "
    "Extract all text from this medical document image exactly as it appears. "
    "Preserve headings, lab values, medications, dates, and table structure. "
    "Output only the extracted text with no commentary."
)


# ── 4. InternVL2-8B ──────────────────────────────────────────────────────────

def _load_internvl():
    global _internvl_model, _internvl_tokenizer
    if _internvl_model is not None:
        return _internvl_tokenizer, _internvl_model
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_id = "OpenGVLab/InternVL2-8B"
        _internvl_tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        _internvl_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            _internvl_model = _internvl_model.cuda()
        _internvl_model.eval()
        logger.info("InternVL2-8B loaded")
    except Exception as exc:
        logger.warning(f"InternVL2 not available: {exc}")
        _internvl_model = False
        _internvl_tokenizer = False
    return _internvl_tokenizer, _internvl_model


def run_internvl2(img) -> VisionResult:
    """Run InternVL2-8B on a document image."""
    try:
        tokenizer, model = _load_internvl()
        if not model:
            return VisionResult("", "internvl2-8b", error="InternVL2 not installed")

        import torch
        from torchvision import transforms

        # InternVL2 expects a 448×448 tensor input
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(img.convert("RGB")).unsqueeze(0)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda().half()

        generation_config = dict(max_new_tokens=1024, do_sample=False)
        response = model.chat(
            tokenizer, pixel_values, _MEDICAL_PROMPT, generation_config
        )
        return VisionResult(text=response, engine_name="internvl2-8b", confidence=0.85)
    except Exception as exc:
        return VisionResult("", "internvl2-8b", error=str(exc))


# ── 5. Qwen2-VL-7B ───────────────────────────────────────────────────────────

def _load_qwen():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return _qwen_processor, _qwen_model
    try:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        _qwen_processor = AutoProcessor.from_pretrained(model_id)
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            _qwen_model = _qwen_model.cuda()
        _qwen_model.eval()
        logger.info("Qwen2-VL-7B loaded")
    except Exception as exc:
        logger.warning(f"Qwen2-VL not available: {exc}")
        _qwen_model = False
        _qwen_processor = False
    return _qwen_processor, _qwen_model


def run_qwen2vl(img) -> VisionResult:
    """Run Qwen2-VL-7B on a document image."""
    try:
        processor, model = _load_qwen()
        if not model:
            return VisionResult("", "qwen2-vl-7b", error="Qwen2-VL not installed")

        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": _MEDICAL_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        result = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return VisionResult(text=result, engine_name="qwen2-vl-7b", confidence=0.85)
    except Exception as exc:
        return VisionResult("", "qwen2-vl-7b", error=str(exc))


# ── 6. LLaVA 1.6 ─────────────────────────────────────────────────────────────

def _load_llava():
    global _llava_model, _llava_processor
    if _llava_model is not None:
        return _llava_processor, _llava_model
    try:
        import torch
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        _llava_processor = LlavaNextProcessor.from_pretrained(model_id)
        _llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            _llava_model = _llava_model.cuda()
        _llava_model.eval()
        logger.info("LLaVA 1.6 loaded")
    except Exception as exc:
        logger.warning(f"LLaVA 1.6 not available: {exc}")
        _llava_model = False
        _llava_processor = False
    return _llava_processor, _llava_model


def run_llava(img) -> VisionResult:
    """Run LLaVA 1.6 on a document image."""
    try:
        processor, model = _load_llava()
        if not model:
            return VisionResult("", "llava-1.6-7b", error="LLaVA not installed")

        import torch
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _MEDICAL_PROMPT},
            ],
        }]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(img, prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1024)
        result = processor.decode(output[0], skip_special_tokens=True)
        # Strip the prompt prefix from the output
        if _MEDICAL_PROMPT in result:
            result = result.split(_MEDICAL_PROMPT)[-1].strip()
        return VisionResult(text=result, engine_name="llava-1.6-7b", confidence=0.80)
    except Exception as exc:
        return VisionResult("", "llava-1.6-7b", error=str(exc))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API TIER — Claude Vision (always flagged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_claude_vision(img) -> VisionResult:
    """
    Call Claude Vision API as absolute last resort.
    Always sets claude_used=True — the UI must show a banner.
    """
    try:
        from app.config import settings
        if not settings.ANTHROPIC_API_KEY:
            return VisionResult("", "claude-vision", claude_used=True,
                                error="ANTHROPIC_API_KEY not set")

        import anthropic

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _MEDICAL_PROMPT},
                ],
            }],
        )
        text = msg.content[0].text if msg.content else ""
        return VisionResult(
            text=text,
            engine_name="claude-vision",
            claude_used=True,
            confidence=0.95,
        )
    except Exception as exc:
        return VisionResult("", "claude-vision", claude_used=True, error=str(exc))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public cascade API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Ordered from best to worst (for medical document understanding).
# Each entry: (runner_fn, min_text_length_to_accept)
_OCR_CASCADE = [
    (run_surya,   20),
    (run_doctr,   20),
    (run_nougat,  20),
]

_VLM_CASCADE = [
    (run_internvl2, 30),
    (run_qwen2vl,   30),
    (run_llava,     30),
]


def vision_cascade(img, min_confidence: float = 0.50) -> VisionResult:
    """
    Run the full vision cascade and return the first successful result.

    Order:
      Surya → DocTR → Nougat → InternVL2 → Qwen2-VL → LLaVA → Claude Vision

    A result is accepted when:
      - No error
      - len(text) >= min_text_length
      - confidence >= min_confidence  (VLMs always report a fixed estimate)

    Claude Vision is called only if everything else fails.
    """
    all_errors: list[str] = []

    for runner, min_len in _OCR_CASCADE:
        result = runner(img)
        if result.error:
            all_errors.append(f"{result.engine_name}: {result.error}")
            continue
        if len(result.text.strip()) >= min_len and result.confidence >= min_confidence:
            logger.info(f"Vision cascade: accepted by {result.engine_name} "
                        f"(conf={result.confidence:.2f}, len={len(result.text)})")
            return result
        logger.debug(f"Vision cascade: {result.engine_name} rejected "
                     f"(conf={result.confidence:.2f}, len={len(result.text)})")

    for runner, min_len in _VLM_CASCADE:
        result = runner(img)
        if result.error:
            all_errors.append(f"{result.engine_name}: {result.error}")
            continue
        if len(result.text.strip()) >= min_len:
            logger.info(f"Vision cascade: accepted by {result.engine_name}")
            return result
        logger.debug(f"Vision cascade: {result.engine_name} returned too little text")

    # Last resort
    logger.warning("Vision cascade: all open-source engines failed, calling Claude Vision")
    for err in all_errors:
        logger.debug(f"  {err}")
    return run_claude_vision(img)


def run_single_engine(engine_name: str, img) -> VisionResult:
    """
    Run a specific engine by name — used for benchmarking / comparison.
    Valid names: surya, doctr, nougat, internvl2, qwen2vl, llava, claude-vision
    """
    mapping = {
        "surya":        run_surya,
        "doctr":        run_doctr,
        "nougat":       run_nougat,
        "internvl2":    run_internvl2,
        "internvl2-8b": run_internvl2,
        "qwen2vl":      run_qwen2vl,
        "qwen2-vl-7b":  run_qwen2vl,
        "llava":        run_llava,
        "llava-1.6-7b": run_llava,
        "claude-vision":run_claude_vision,
    }
    fn = mapping.get(engine_name.lower())
    if fn is None:
        return VisionResult("", engine_name, error=f"Unknown engine: {engine_name}")
    return fn(img)
