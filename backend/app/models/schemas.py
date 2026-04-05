"""
Pydantic Schemas
================
Central data models shared across the entire EHR AI Sidebar backend.
All imports:  from app.models.schemas import ...
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Enums ─────────────────────────────────────────────────────────────────────

class OCREngine(str, Enum):
    # Classic engines
    AUTO      = "auto"
    TESSERACT = "tesseract"
    EASYOCR   = "easyocr"
    NONE      = "none"
    # Transformer OCR engines
    SURYA     = "surya"       # SOTA open OCR
    DOCTR     = "doctr"       # DocTR transformer OCR
    NOUGAT    = "nougat"      # Meta Nougat (scientific/medical PDFs)
    # Vision-Language Models
    INTERNVL2  = "internvl2"  # InternVL2-8B
    QWEN2VL    = "qwen2vl"    # Qwen2-VL-7B
    LLAVA      = "llava"      # LLaVA 1.6-7B
    # API fallback (always flagged)
    CLAUDE_VISION = "claude_vision"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"   # some pages OK, some failed
    FAILED = "failed"


class ModelProvider(str, Enum):
    CLAUDE = "claude"
    OLLAMA = "ollama"
    TINYLLAMA = "tinyllama"   # backward-compat alias
    LOCAL_HF = "local_hf"     # any HuggingFace model loaded locally


class MedicalDocumentType(str, Enum):
    LAB_REPORT         = "lab_report"
    DISCHARGE_SUMMARY  = "discharge_summary"
    PRESCRIPTION       = "prescription"
    CLINICAL_NOTE      = "clinical_note"
    RADIOLOGY_REPORT   = "radiology_report"
    PATHOLOGY_REPORT   = "pathology_report"
    CONSENT_FORM       = "consent_form"
    REFERRAL_LETTER    = "referral_letter"
    INSURANCE_FORM     = "insurance_form"
    VACCINATION_RECORD = "vaccination_record"
    OPERATIVE_REPORT   = "operative_report"
    PROGRESS_NOTE      = "progress_note"
    INTAKE_FORM        = "intake_form"
    UNKNOWN            = "unknown"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class FineTuneStatus(str, Enum):
    QUEUED = "queued"
    PENDING = "pending"       # job accepted, not yet started
    RUNNING = "running"       # actively training
    TRAINING = "training"     # alias for RUNNING
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Document models ───────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    page: int = 0


class TextBlock(BaseModel):
    text: str = ""
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None
    block_type: str = "paragraph"
    font_size: float = 12.0
    is_bold: bool = False


class TableCell(BaseModel):
    text: str = ""
    row: int = 0
    col: int = 0
    bbox: Optional[BoundingBox] = None


class Table(BaseModel):
    cells: List[TableCell] = Field(default_factory=list)
    rows: int = 0
    cols: int = 0
    bbox: Optional[BoundingBox] = None
    page: int = 0


class DocumentMetadata(BaseModel):
    filename: str
    file_size: int = 0
    file_type: str = ""
    page_count: int = 0
    ocr_engine: str = "auto"
    processing_time_ms: int = 0
    created_at: datetime = Field(default_factory=_utcnow)


class DocumentPage(BaseModel):
    page_number: int
    width: float = 0.0
    height: float = 0.0
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    raw_text: str = ""
    confidence: float = 1.0
    error: Optional[str] = None          # per-page error when partial success
    # Computer-vision enrichment fields
    header_metadata: Dict[str, str] = Field(default_factory=dict)
    header_text: str = ""
    footer_text: str = ""
    column_count: int = 1
    checkboxes: List[Dict[str, Any]] = Field(default_factory=list)
    image_quality: Optional[Dict[str, Any]] = None       # blur, contrast, brightness, warnings
    handwriting_regions: List[Dict[str, Any]] = Field(default_factory=list)
    has_handwriting: bool = False
    chart_regions: List[Dict[str, Any]] = Field(default_factory=list)  # {x1,y1,x2,y2,chart_type,title}
    has_charts: bool = False
    # Vision engine tracking
    vision_engine_used: str = ""         # which engine produced text for this page
    claude_vision_used: bool = False     # True if Claude Vision API was called


class ProcessedDocument(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: DocumentMetadata
    pages: List[DocumentPage] = Field(default_factory=list)
    full_text: str = ""
    status: DocumentStatus = DocumentStatus.PENDING
    document_type: MedicalDocumentType = MedicalDocumentType.UNKNOWN
    document_type_confidence: float = 0.0   # 0.0–1.0
    document_type_signals: List[str] = Field(default_factory=list)  # why this type was chosen
    error: Optional[str] = None
    partial_page_errors: List[str] = Field(default_factory=list)
    # Vision engine tracking across all pages
    claude_vision_used: bool = False         # any page used Claude Vision API
    vision_engines_used: List[str] = Field(default_factory=list)  # all engines used


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str


class OCRRequest(BaseModel):
    engine: OCREngine = OCREngine.AUTO
    enhance_image: bool = True
    languages: List[str] = Field(default_factory=lambda: ["en"])
    extract_tables: bool = True


# ── Chat / LLM models ─────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    provider: ModelProvider
    display_name: str = ""
    context_length: int = 8192
    supports_vision: bool = False
    is_local: bool = False
    size_gb: float = 0.0
    description: str = ""
    fine_tunable: bool = False


class AvailableModels(BaseModel):
    models: List[ModelInfo] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=_utcnow)
    document_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    model_name: Optional[str] = None
    # Default to LOCAL_HF so open-source models are tried first.
    # Claude is only used as an automatic fallback if the local provider fails.
    model_provider: ModelProvider = ModelProvider.LOCAL_HF
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    conversation_id: Optional[str] = None
    document_ids: List[str] = Field(default_factory=list)
    stream: bool = False


class ChatResponse(BaseModel):
    conversation_id: str
    message: ChatMessage
    model_used: str = ""
    usage: Dict[str, Any] = Field(default_factory=dict)
    document_context: Optional[List[str]] = None
    # Transparency fields — always tell the frontend which provider answered
    provider_used: str = ""       # e.g. "local_hf", "ollama", "claude"
    claude_used: bool = False     # True when Claude was used as fallback


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    document_ids: List[str] = Field(default_factory=list)


# ── Fine-tuning models ────────────────────────────────────────────────────────

class TrainingExample(BaseModel):
    instruction: str
    input: str = ""
    output: str
    system: Optional[str] = None


class TrainingConfig(BaseModel):
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_name: str = "fine-tuned-medical"
    dataset_path: Optional[str] = None
    examples: List[TrainingExample] = Field(default_factory=list)
    num_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 20
    use_lora: bool = True
    load_in_4bit: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


class FineTuneJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: TrainingConfig
    status: FineTuneStatus = FineTuneStatus.QUEUED
    progress: float = 0.0
    current_epoch: int = 0
    current_loss: Optional[float] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class FineTuneJobResponse(BaseModel):
    job_id: str
    status: FineTuneStatus
    message: str = ""
    progress: float = 0.0


# ── Model comparison ──────────────────────────────────────────────────────────

class ModelCompareRequest(BaseModel):
    prompt: str
    models: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024


class ModelCompareResult(BaseModel):
    model_name: str
    provider: str
    response: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    error: Optional[str] = None


class ModelCompareResponse(BaseModel):
    prompt: str
    results: List[ModelCompareResult] = Field(default_factory=list)
    comparison_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
