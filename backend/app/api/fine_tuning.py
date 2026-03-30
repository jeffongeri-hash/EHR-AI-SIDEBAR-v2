"""Fine-tuning management API endpoints."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from loguru import logger

from app.config import settings
from app.core.rate_limit import limiter
from app.models.schemas import (
    FineTuneJob,
    FineTuneJobResponse,
    FineTuneStatus,
    ModelCompareRequest,
    ModelCompareResponse,
    TrainingConfig,
    TrainingExample,
)
from app.services.fine_tune_service import FineTuneService
from app.services.model_compare_service import compare_models

router = APIRouter(prefix="/fine-tune", tags=["Fine-Tuning"])
_svc = FineTuneService()


@router.post("/jobs", response_model=FineTuneJobResponse, status_code=201)
@limiter.limit("3/minute")
async def create_and_start_job(request: Request, config: TrainingConfig):
    """
    Create and immediately start a fine-tuning job.

    Supports:
    - **base_model**: HuggingFace model ID (e.g. `meta-llama/Llama-3.2-3B-Instruct`)
    - **training_examples**: inline training data (instruction/input/output triples)
    - **dataset_path**: path to a local JSONL file or HuggingFace dataset name
    - **use_lora**: use LoRA / QLoRA (recommended, greatly reduces VRAM)
    - **load_in_4bit**: 4-bit NF4 quantization (QLoRA)
    - **push_to_hub**: push trained adapter to HuggingFace Hub
    """
    job = await _svc.create_job(config)
    response = await _svc.start_job(job.job_id)
    return response


@router.get("/jobs", response_model=list[FineTuneJob])
async def list_jobs():
    """List all fine-tuning jobs."""
    return await _svc.list_jobs()


@router.get("/jobs/{job_id}", response_model=FineTuneJob)
async def get_job(job_id: str):
    """Get details and progress of a specific fine-tuning job."""
    job = await _svc.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running or queued fine-tuning job."""
    cancelled = await _svc.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=400,
            detail="Job not found or cannot be cancelled (already completed/failed)"
        )
    return {"message": f"Job {job_id} cancelled"}


@router.get("/models")
async def list_trained_models():
    """List locally available fine-tuned models."""
    return _svc.list_trained_models()


@router.get("/templates/ehr")
async def get_ehr_template():
    """
    Return a sample EHR training dataset template to help you get started.
    Modify these examples and POST to /fine-tune/jobs to start training.
    """
    examples = _svc.prepare_ehr_dataset_template()
    return {
        "description": "Sample EHR domain fine-tuning examples",
        "base_model_suggestion": "meta-llama/Llama-3.2-3B-Instruct",
        "examples": [e.dict() for e in examples],
        "tip": (
            "Add 50–500 domain-specific examples for best results. "
            "Use instruction/input/output format or plain 'text' field."
        ),
    }


@router.post("/upload-document")
async def upload_training_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    overlap: int = Form(64),
    format: str = Form("text"),  # "text" | "instruction"
):
    """
    Upload a PDF, DOCX, or image document and convert it into a JSONL training
    dataset ready for fine-tuning.

    - **chunk_size**: approximate token count per training record (default 512)
    - **overlap**: overlap in words between consecutive chunks (default 64)
    - **format**: `text` for raw-completion style, `instruction` for instruction-following style
    """
    suffix = Path(file.filename or "upload").suffix.lower()
    allowed = set(settings.SUPPORTED_FORMATS)
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    # Save upload to a temp path
    tmp_path = Path(settings.UPLOAD_DIR) / f"_ft_{file.filename}"
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        result = _svc.ingest_document_for_training(
            file_path=tmp_path,
            filename=file.filename or "upload",
            chunk_size=chunk_size,
            overlap=overlap,
            fmt=format,
        )
        return result
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/datasets")
async def list_training_datasets():
    """List all JSONL datasets available for fine-tuning."""
    return _svc.list_datasets()


@router.delete("/datasets/{dataset_id}")
async def delete_training_dataset(dataset_id: str):
    """Delete a saved training dataset."""
    deleted = _svc.delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return {"message": f"Dataset {dataset_id} deleted"}


@router.post("/compare", response_model=ModelCompareResponse)
@limiter.limit("5/minute")
async def compare_base_vs_finetuned(request: Request, body: ModelCompareRequest):
    """
    Run the same prompt against both the base Llama model (via Ollama) and a
    locally fine-tuned PEFT adapter, returning responses side-by-side with
    latency metrics so you can evaluate improvement.

    - **base_model**: Ollama model tag for the base model (e.g. `llama3.2`)
    - **fine_tuned_model**: Directory name under the models folder (e.g. `ehr-llama-3b`)
    - **prompt**: The test prompt to send to both models
    """
    try:
        return await compare_models(body)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/configs/recommended")
async def get_recommended_configs():
    """Return recommended training configurations for common model sizes."""
    return {
        "3B_model_consumer_gpu": {
            "base_model": "meta-llama/Llama-3.2-3B-Instruct",
            "use_lora": True,
            "load_in_4bit": True,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "lora_config": {"r": 16, "lora_alpha": 32},
            "notes": "Fits on 8GB VRAM (RTX 3060 etc.)",
        },
        "7B_model_high_end_gpu": {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "use_lora": True,
            "load_in_4bit": True,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "lora_config": {"r": 32, "lora_alpha": 64},
            "notes": "Requires 16GB+ VRAM (RTX 3090 / A100 etc.)",
        },
        "deepseek_r1": {
            "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "use_lora": True,
            "load_in_4bit": True,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-4,
            "num_epochs": 5,
            "lora_config": {"r": 16, "lora_alpha": 32},
            "notes": "1.5B DeepSeek-R1 distil – fast training on 6GB VRAM",
        },
    }
