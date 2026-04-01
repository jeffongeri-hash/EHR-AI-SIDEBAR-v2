"""
Fine-tuning API Routes
======================
Endpoints for in-app LoRA/QLoRA fine-tuning of local models.
No Google Notebook or Vertex AI — everything runs locally.

Models supported: TinyLlama, DeepSeek-R1 1.5B, Phi-3-mini, Mistral-7B (and any HF model).
"""
from __future__ import annotations

# ── stdlib (must be at top!) ────────────────────────────────────────────────────
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.local_model_service import (
    local_model_service,
    MODEL_CATALOGUE,
    TrainProgress,
)
from app.models.schemas import FineTuneStatus, ModelInfo, ModelProvider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/finetune", tags=["fine-tuning"])


# ── Request / response schemas ───────────────────────────────────────────────────

class FineTuneRequest(BaseModel):
    training_data: List[Dict[str, str]]
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    load_in_4bit: bool = True


class FineTuneStatusOut(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    base_model: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None


class FineTuneResponse(BaseModel):
    success: bool
    job_id: str
    message: str
    status: FineTuneStatusOut


# ── In-memory job tracker ────────────────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}      # job_id -> state
_progress_queues: Dict[str, asyncio.Queue] = {}   # job_id -> SSE queue


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/models")
async def list_fine_tunable_models() -> Dict[str, Any]:
    """List all models that can be fine-tuned locally."""
    models = []
    for alias, meta in MODEL_CATALOGUE.items():
        models.append({
            "alias": alias,
            "hf_id": meta["hf_id"],
            "display_name": meta["display_name"],
            "size_gb": meta["size_gb"],
            "min_ram_gb": meta["min_ram_gb"],
            "context_length": meta["context_length"],
            "description": meta["description"],
            "fine_tunable": meta["fine_tunable"],
        })
    return {
        "success": True,
        "models": models,
        "current_model": local_model_service.get_info(),
    }


@router.post("/start", response_model=FineTuneResponse)
async def start_fine_tuning(request: FineTuneRequest) -> FineTuneResponse:
    """Start a LoRA fine-tuning job asynchronously."""
    if not request.training_data:
        raise HTTPException(status_code=400, detail="training_data is required")

    # Validate examples
    valid = []
    for i, item in enumerate(request.training_data):
        has_qa = "question" in item and "answer" in item
        has_io = "input" in item and "output" in item
        has_ins = "instruction" in item and "output" in item
        if not (has_qa or has_io or has_ins):
            raise HTTPException(
                status_code=400,
                detail=f"Item {i}: must have question/answer, input/output, or instruction/output keys",
            )
        valid.append(item)

    job_id = f"ft-{int(datetime.now(timezone.utc).timestamp())}"
    created_at = _now_iso()

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "training",
        "progress": 0.0,
        "base_model": request.base_model,
        "created_at": created_at,
        "completed_at": None,
        "error_message": None,
        "output_path": None,
    }
    _progress_queues[job_id] = asyncio.Queue(maxsize=500)

    # Launch fine-tuning in background
    asyncio.create_task(_run_fine_tune_job(job_id, valid, request))

    status_out = FineTuneStatusOut(
        job_id=job_id, status="training", progress=0.0,
        base_model=request.base_model, created_at=created_at,
    )
    return FineTuneResponse(
        success=True, job_id=job_id,
        message=f"Fine-tuning started for {request.base_model}",
        status=status_out,
    )


async def _run_fine_tune_job(
    job_id: str,
    examples: List[Dict[str, str]],
    request: FineTuneRequest,
):
    """Background task that runs fine-tuning and posts progress updates."""
    q = _progress_queues.get(job_id)
    job = _jobs.get(job_id)
    if q is None or job is None:
        return

    def _cb(prog: TrainProgress):
        job["progress"] = prog.progress_pct
        try:
            q.put_nowait({
                "type": "progress",
                "progress": prog.progress_pct,
                "epoch": prog.epoch,
                "step": prog.step,
                "loss": prog.loss,
                "log": prog.log_line,
            })
        except asyncio.QueueFull:
            pass

    try:
        output_path = await local_model_service.fine_tune(
            examples=examples,
            model_id=request.base_model,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            max_seq_length=request.max_seq_length,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            load_in_4bit=request.load_in_4bit,
            progress_callback=_cb,
        )
        job["status"] = "completed"
        job["progress"] = 100.0
        job["output_path"] = output_path
        job["completed_at"] = _now_iso()
        q.put_nowait({"type": "done", "progress": 100.0, "output_path": output_path})
        logger.info(f"Job {job_id} completed. Model saved to {output_path}")
    except Exception as exc:
        err = str(exc)
        logger.error(f"Fine-tune job {job_id} failed: {err}")
        job["status"] = "failed"
        job["error_message"] = err
        job["completed_at"] = _now_iso()
        try:
            q.put_nowait({"type": "error", "message": err})
        except asyncio.QueueFull:
            pass


@router.get("/progress/{job_id}")
async def stream_progress(job_id: str):
    """
    SSE endpoint — streams training progress in real time.
    Connect with:  EventSource('/api/finetune/progress/{job_id}')
    Each event is a JSON object with keys: type, progress, epoch, step, loss, log.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]
    q = _progress_queues.get(job_id)

    async def event_generator():
        # If job already done, send final state immediately
        if job["status"] in ("completed", "failed"):
            payload = json.dumps({
                "type": "done" if job["status"] == "completed" else "error",
                "progress": job["progress"],
                "output_path": job.get("output_path"),
                "message": job.get("error_message", ""),
            })
            yield f"data: {payload}\n\n"
            return

        # Stream live events from queue
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Keep-alive ping
                yield "data: {\"type\": \"ping\"}\n\n"
                continue

            yield f"data: {json.dumps(event)}\n\n"

            if event.get("type") in ("done", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Poll training job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return {"success": True, **job}


@router.get("/jobs")
async def list_jobs() -> Dict[str, Any]:
    """List all fine-tuning jobs."""
    return {"success": True, "jobs": list(_jobs.values())}


@router.get("/status")
async def model_status() -> Dict[str, Any]:
    """Current model load status and fine-tuned adapter info."""
    info = local_model_service.get_info()
    ft_models = local_model_service.list_fine_tuned_models()
    return {
        "success": True,
        "model_info": info,
        "fine_tuned_models": ft_models,
        "fine_tuning_available": info["hf_available"] and info["torch_available"],
        "peft_available": info["peft_available"],
        "bnb_4bit_available": info["bnb_4bit_available"],
    }


@router.post("/load-model")
async def load_model(model_id: str) -> Dict[str, Any]:
    """
    Load a specific model into memory.
    model_id can be an alias ('tinyllama', 'deepseek-1.5b', 'phi3-mini', 'mistral-7b')
    or a full HuggingFace model ID.
    """
    success = await local_model_service.load_model(model_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model '{model_id}'. "
                   f"Check that transformers and torch are installed and you have enough RAM.",
        )
    return {
        "success": True,
        "message": f"Model loaded: {local_model_service._active_model_id}",
        "model_info": local_model_service.get_info(),
    }


@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    description: str = Form(""),
) -> Dict[str, Any]:
    """Upload a JSON or JSONL training dataset file."""
    if not (file.filename or "").endswith((".json", ".jsonl")):
        raise HTTPException(status_code=400, detail="File must be .json or .jsonl")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    training_data: List[Dict] = []
    try:
        if (file.filename or "").endswith(".jsonl"):
            for line in text.strip().splitlines():
                if line.strip():
                    training_data.append(json.loads(line))
        else:
            parsed = json.loads(text)
            training_data = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    valid_count = sum(
        1 for item in training_data
        if isinstance(item, dict) and (
            ("question" in item and "answer" in item)
            or ("input" in item and "output" in item)
            or ("instruction" in item and "output" in item)
        )
    )

    if valid_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid examples found. Each item must have "
                   "question/answer, input/output, or instruction/output.",
        )

    dataset_dir = Path("./models/datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_path = dataset_dir / f"uploaded_{file.filename}"
    save_path.write_text(json.dumps(training_data, indent=2, ensure_ascii=False))

    return {
        "success": True,
        "filename": file.filename,
        "total_examples": len(training_data),
        "valid_examples": valid_count,
        "dataset_path": str(save_path),
        "description": description,
    }


@router.get("/datasets")
async def list_datasets() -> Dict[str, Any]:
    """List uploaded training datasets."""
    dataset_dir = Path("./models/datasets")
    if not dataset_dir.exists():
        return {"success": True, "datasets": []}

    datasets = []
    for fp in sorted(dataset_dir.glob("*.json*"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            count = sum(1 for _ in fp.open())
            datasets.append({
                "filename": fp.name,
                "size_mb": round(fp.stat().st_size / 1e6, 2),
                "examples_count": count,
                "created_at": datetime.fromtimestamp(
                    fp.stat().st_ctime, tz=timezone.utc
                ).isoformat(),
                "path": str(fp),
            })
        except Exception:
            pass

    return {"success": True, "datasets": datasets}


@router.post("/sample-data")
async def sample_medical_data() -> Dict[str, Any]:
    """Return built-in sample medical Q&A data for testing."""
    data = [
        {
            "instruction": "What are the symptoms of pneumonia?",
            "output": (
                "Common symptoms: fever, chills, cough with phlegm, shortness of breath, "
                "chest pain when breathing, fatigue, and confusion in older adults. "
                "Seek medical attention if symptoms persist or worsen."
            ),
        },
        {
            "instruction": "How is type 2 diabetes diagnosed?",
            "output": (
                "Diagnosed by: Fasting glucose ≥126 mg/dL, Random glucose ≥200 mg/dL "
                "with symptoms, HbA1c ≥6.5%, or 2-hr glucose ≥200 mg/dL on OGTT. "
                "Confirm with repeat testing."
            ),
        },
        {
            "instruction": "What should I do for a patient presenting with chest pain?",
            "output": (
                "1) Check vitals and ECG immediately. 2) Assess pain (location, radiation, "
                "quality). 3) Order troponin. 4) CXR. 5) Oxygen if SpO2 <94%. "
                "6) Prepare for emergency intervention if acute MI suspected. 7) Continuous monitoring."
            ),
        },
        {
            "instruction": "What are normal adult vital signs?",
            "output": (
                "HR 60–100 bpm, BP <120/80 mmHg (normal), RR 12–20/min, "
                "Temp 97–99°F (36–37°C), SpO2 >95% on room air."
            ),
        },
        {
            "instruction": "How do you interpret a CBC with differential?",
            "output": (
                "WBC 4.5–11.0 K/µL (infection/inflammation marker), Hgb for anemia, "
                "Platelets 150–450 K/µL (bleeding risk), MCV classifies anemia type, "
                "differential for specific WBC abnormalities. Always correlate with clinical picture."
            ),
        },
        {
            "instruction": "Summarise this clinical note: Patient 65M, HTN/DM2, presents with "
                           "crushing chest pain radiating to left arm. ECG: ST elevation II/III/aVF. "
                           "Troponin 2.1 ng/mL.",
            "output": (
                "65M with HTN/DM2 presenting with acute inferior STEMI "
                "(ST elevation leads II/III/aVF, troponin 2.1). "
                "Urgent cath lab activation indicated."
            ),
        },
    ]
    return {
        "success": True,
        "data": data,
        "count": len(data),
        "description": "Built-in medical Q&A examples for fine-tuning demonstration",
    }
