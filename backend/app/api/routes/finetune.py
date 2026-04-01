"""
Fine-tuning API routes — LoRA/QLoRA training, SSE progress streaming,
auto Q&A pair generation from extracted document text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.models.schemas import FineTuneStatus, ModelProvider
from app.services.local_model_service import local_model_service
from app.services.tinyllama_service import tinyllama_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/finetune", tags=["fine-tuning"])

# ── In-memory job tracker ──────────────────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}
_progress_queues: Dict[str, asyncio.Queue] = {}

# ── Request / response schemas ─────────────────────────────────────────────────

class FineTuneRequest(BaseModel):
    training_data: List[Dict[str, str]]
    base_model: str = "tinyllama"
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    load_in_4bit: bool = True


class FineTuneJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    started_at: str


class GeneratePairsRequest(BaseModel):
    text: str = Field(..., description="Extracted document text to generate training pairs from")
    num_pairs: int = Field(10, ge=1, le=50)
    focus: str = Field("general", description="Focus area: general | diagnosis | medications | procedures | labs")


class TrainingPair(BaseModel):
    instruction: str
    input: str
    output: str
    approved: bool = False


class GeneratePairsResponse(BaseModel):
    success: bool
    pairs: List[TrainingPair]
    total: int
    source: str  # "claude" | "local_model" | "rule_based"


# ── Background fine-tune task ──────────────────────────────────────────────────

async def _run_fine_tune_job(job_id: str, request: FineTuneRequest) -> None:
    queue = _progress_queues[job_id]
    _jobs[job_id]["status"] = FineTuneStatus.RUNNING

    try:
        def _on_progress(event: Dict[str, Any]) -> None:
            asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, event)

        await local_model_service.fine_tune(
            model_id=request.base_model,
            training_data=request.training_data,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            max_seq_length=request.max_seq_length,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            load_in_4bit=request.load_in_4bit,
            progress_callback=_on_progress,
        )

        _jobs[job_id]["status"] = FineTuneStatus.COMPLETED
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        await queue.put({"type": "done", "message": "Fine-tuning complete"})

    except Exception as exc:
        logger.error("Fine-tune job %s failed: %s", job_id, exc)
        _jobs[job_id]["status"] = FineTuneStatus.FAILED
        _jobs[job_id]["error"] = str(exc)
        await queue.put({"type": "error", "message": str(exc)})


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/start", response_model=FineTuneJobResponse)
async def start_fine_tuning(request: FineTuneRequest) -> FineTuneJobResponse:
    """Start LoRA fine-tuning in the background and return a job_id immediately."""
    if not request.training_data:
        raise HTTPException(status_code=400, detail="training_data is required")

    for i, item in enumerate(request.training_data):
        has_qa = "question" in item and "answer" in item
        has_io = "input" in item and "output" in item
        has_inst = "instruction" in item and "output" in item
        if not (has_qa or has_io or has_inst):
            raise HTTPException(
                status_code=400,
                detail=f"Item {i}: must have question/answer, input/output, or instruction/output",
            )

    job_id = str(uuid.uuid4())
    _progress_queues[job_id] = asyncio.Queue()
    _jobs[job_id] = {
        "job_id": job_id,
        "status": FineTuneStatus.PENDING,
        "base_model": request.base_model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
    }

    asyncio.create_task(_run_fine_tune_job(job_id, request))

    return FineTuneJobResponse(
        job_id=job_id,
        status=FineTuneStatus.PENDING,
        message="Fine-tuning job queued. Stream progress at GET /api/finetune/progress/{job_id}",
        started_at=_jobs[job_id]["started_at"],
    )


@router.get("/progress/{job_id}")
async def stream_progress(job_id: str) -> StreamingResponse:
    """SSE stream of training progress events for a running job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def _event_stream() -> AsyncGenerator[str, None]:
        queue = _progress_queues.get(job_id)
        if queue is None:
            yield "data: {\"type\": \"error\", \"message\": \"Queue not found\"}\n\n"
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                # keep-alive ping
                yield "data: {\"type\": \"ping\"}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Poll-based status check for a fine-tune job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"success": True, **_jobs[job_id]}


@router.post("/load-model")
async def load_model(model_id: str) -> Dict[str, Any]:
    """Load a local HuggingFace model by alias or full HF model ID."""
    try:
        await local_model_service.load_model(model_id)
        return {"success": True, "message": f"Model '{model_id}' loaded successfully"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Auto Q&A pair generation ───────────────────────────────────────────────────

@router.post("/generate-pairs", response_model=GeneratePairsResponse)
async def generate_training_pairs(request: GeneratePairsRequest) -> GeneratePairsResponse:
    """
    Auto-generate Q&A training pairs from extracted document text.
    Uses Claude if ANTHROPIC_API_KEY is set, otherwise falls back to
    the loaded local model, then to rule-based extraction.
    Pairs are returned for human review — nothing is saved until approved.
    """
    text = request.text.strip()
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Text too short to generate meaningful pairs")

    # Truncate to avoid token limits
    text_snippet = text[:4000]

    focus_hints = {
        "diagnosis": "Focus on diagnostic findings, symptoms, and clinical impressions.",
        "medications": "Focus on medication names, dosages, indications, and side effects.",
        "procedures": "Focus on procedures performed, technique, and outcomes.",
        "labs": "Focus on lab values, reference ranges, and clinical significance.",
        "general": "Cover a broad range of medical facts and clinical details present in the text.",
    }
    focus_hint = focus_hints.get(request.focus, focus_hints["general"])

    generation_prompt = (
        f"You are a medical education expert. Given the following clinical document text, "
        f"generate exactly {request.num_pairs} high-quality question-and-answer training pairs "
        f"for fine-tuning a medical AI assistant. {focus_hint}\n\n"
        f"Rules:\n"
        f"- Each pair must be grounded in the provided text — no hallucination\n"
        f"- Questions should be clinically meaningful (not trivial)\n"
        f"- Answers should be complete and accurate\n"
        f"- Return ONLY a JSON array, no markdown, no explanation\n"
        f"- Format: [{{\"instruction\": \"...\", \"input\": \"<relevant excerpt>\", \"output\": \"...\"}}, ...]\n\n"
        f"Document text:\n{text_snippet}"
    )

    pairs: List[TrainingPair] = []
    source = "rule_based"

    # 1. Try Claude
    try:
        from app.config import settings
        if settings.ANTHROPIC_API_KEY:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": generation_prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            pairs = [TrainingPair(**p) for p in parsed]
            source = "claude"
    except Exception as exc:
        logger.warning("Claude pair generation failed: %s — trying local model", exc)

    # 2. Try local model
    if not pairs:
        try:
            if local_model_service.is_loaded():
                raw_response = await local_model_service.generate(
                    prompt=generation_prompt,
                    max_new_tokens=2048,
                )
                raw = raw_response.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw)
                pairs = [TrainingPair(**p) for p in parsed]
                source = "local_model"
        except Exception as exc:
            logger.warning("Local model pair generation failed: %s — using rule-based", exc)

    # 3. Rule-based fallback — extract sentences and form basic pairs
    if not pairs:
        pairs = _rule_based_pairs(text_snippet, request.num_pairs)
        source = "rule_based"

    return GeneratePairsResponse(
        success=True,
        pairs=pairs[: request.num_pairs],
        total=len(pairs),
        source=source,
    )


def _rule_based_pairs(text: str, num_pairs: int) -> List[TrainingPair]:
    """Fallback: create simple sentence-level Q&A pairs from document text."""
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
    pairs: List[TrainingPair] = []

    patterns = [
        (r"(?i)(diagnosis|assessment)[:\s]+(.+)", "What is the diagnosis or assessment?"),
        (r"(?i)(medications?|prescribed)[:\s]+(.+)", "What medications are prescribed?"),
        (r"(?i)(allergies)[:\s]+(.+)", "What allergies does the patient have?"),
        (r"(?i)(chief complaint)[:\s]+(.+)", "What is the chief complaint?"),
        (r"(?i)(treatment plan)[:\s]+(.+)", "What is the treatment plan?"),
        (r"(?i)(lab results?|laboratory)[:\s]+(.+)", "What do the lab results show?"),
        (r"(?i)(blood pressure)[:\s]+([\d/]+)", "What is the patient's blood pressure?"),
        (r"(?i)(heart rate)[:\s]+([\d]+)", "What is the patient's heart rate?"),
    ]

    for sentence in sentences:
        for pattern, question in patterns:
            m = re.search(pattern, sentence)
            if m:
                pairs.append(TrainingPair(
                    instruction=question,
                    input=sentence,
                    output=m.group(0),
                ))
                break
        else:
            # Generic pair
            words = sentence.split()
            if len(words) > 8:
                pairs.append(TrainingPair(
                    instruction="Summarize the following clinical information.",
                    input=sentence,
                    output=sentence,
                ))
        if len(pairs) >= num_pairs:
            break

    return pairs


# ── Existing supporting endpoints ──────────────────────────────────────────────

@router.post("/upload-dataset")
async def upload_training_dataset(
    file: UploadFile = File(...),
    description: str = Form(""),
) -> Dict[str, Any]:
    """Upload a .json or .jsonl training dataset file."""
    if not file.filename.endswith((".json", ".jsonl")):
        raise HTTPException(status_code=400, detail="File must be .json or .jsonl")

    content = await file.read()
    text_content = content.decode("utf-8")

    training_data: List[Dict] = []
    try:
        if file.filename.endswith(".jsonl"):
            for line in text_content.strip().split("\n"):
                if line.strip():
                    training_data.append(json.loads(line))
        else:
            data = json.loads(text_content)
            training_data = data if isinstance(data, list) else [data]
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    valid = sum(
        1 for item in training_data
        if isinstance(item, dict) and (
            ("question" in item and "answer" in item)
            or ("input" in item and "output" in item)
            or ("instruction" in item and "output" in item)
        )
    )
    if valid == 0:
        raise HTTPException(status_code=400, detail="No valid training examples found")

    dataset_dir = Path("./models/datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_path = dataset_dir / file.filename
    save_path.write_text(json.dumps(training_data, indent=2))

    return {
        "success": True,
        "filename": file.filename,
        "total_examples": len(training_data),
        "valid_examples": valid,
        "dataset_path": str(save_path),
        "description": description,
    }


@router.get("/datasets")
async def list_uploaded_datasets() -> Dict[str, Any]:
    """List all uploaded training datasets."""
    dataset_dir = Path("./models/datasets")
    if not dataset_dir.exists():
        return {"success": True, "datasets": []}

    datasets = []
    for fp in dataset_dir.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
            datasets.append({
                "filename": fp.name,
                "size_mb": round(fp.stat().st_size / 1024 / 1024, 2),
                "examples_count": len(data) if isinstance(data, list) else 1,
                "created_at": datetime.fromtimestamp(fp.stat().st_ctime, tz=timezone.utc).isoformat(),
            })
        except Exception:
            pass

    return {"success": True, "datasets": sorted(datasets, key=lambda x: x["created_at"], reverse=True)}


@router.get("/models")
async def list_fine_tunable_models() -> Dict[str, Any]:
    """List models available for fine-tuning."""
    from app.services.llm_service import llm_service
    models = await llm_service.list_all_models()
    local = [m for m in models if m.get("provider") in ("local_hf", "tinyllama")]
    return {"success": True, "models": local}


@router.get("/status")
async def get_fine_tuning_status() -> Dict[str, Any]:
    """Overall fine-tuning system status."""
    return {
        "success": True,
        "active_jobs": sum(1 for j in _jobs.values() if j["status"] == FineTuneStatus.RUNNING),
        "total_jobs": len(_jobs),
        "local_model_loaded": local_model_service.is_loaded(),
        "loaded_model_id": local_model_service.current_model_id,
    }


@router.post("/sample-data")
async def generate_sample_medical_data() -> Dict[str, Any]:
    """Return hard-coded sample medical Q&A pairs for testing."""
    sample = [
        {"instruction": "What are the symptoms of pneumonia?", "input": "",
         "output": "Fever, chills, productive cough, dyspnea, pleuritic chest pain, fatigue."},
        {"instruction": "How is Type 2 Diabetes diagnosed?", "input": "",
         "output": "Fasting glucose ≥126 mg/dL, random glucose ≥200 mg/dL with symptoms, HbA1c ≥6.5%, or 2-hour OGTT ≥200 mg/dL."},
        {"instruction": "What is the first-line treatment for hypertension?", "input": "",
         "output": "Lifestyle modification (diet, exercise, weight loss) plus thiazide diuretics, ACE inhibitors, ARBs, or CCBs depending on comorbidities."},
        {"instruction": "Interpret an elevated troponin level.", "input": "Troponin I: 2.4 ng/mL (normal <0.04)",
         "output": "Significantly elevated troponin suggests myocardial injury. In the appropriate clinical context this is consistent with acute myocardial infarction. Serial troponins and ECG correlation required."},
        {"instruction": "What are normal adult vital signs?", "input": "",
         "output": "HR 60-100 bpm, BP <120/80 mmHg, RR 12-20/min, Temp 36-37.2°C, SpO2 >95%."},
    ]
    return {"success": True, "data": sample, "count": len(sample)}
