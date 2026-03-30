"""
Fine-Tuning Service

Manages fine-tuning jobs for open-source LLMs (Llama, DeepSeek, Mistral, etc.)
using PEFT/LoRA via HuggingFace Transformers + TRL.

Features:
- LoRA / QLoRA (4-bit NF4 quantization)
- SFT (Supervised Fine-Tuning) via TRL SFTTrainer
- Dataset preparation from raw text, JSON, or JSONL
- Progress tracking via callbacks
- Export to GGUF / push to HuggingFace Hub
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import settings
from app.db.repositories import FineTuneJobRepository
from app.services.document_processor import DocumentProcessor
from app.models.schemas import (
    FineTuneJob,
    FineTuneJobResponse,
    FineTuneStatus,
    TrainingConfig,
    TrainingExample,
)

_job_repo = FineTuneJobRepository()

# Celery task IDs for active jobs (job_id → celery_task_id)
# Used to revoke tasks on cancellation.
_celery_task_ids: dict[str, str] = {}


# ── Dataset helpers ───────────────────────────────────────────────────────────

class DatasetPreparator:
    """Convert various input formats to HuggingFace Dataset."""

    @staticmethod
    def from_examples(examples: list[TrainingExample]) -> "datasets.Dataset":
        import datasets as hf_datasets

        records = []
        for ex in examples:
            if ex.system:
                prompt = f"<|system|>\n{ex.system}\n<|user|>\n"
            else:
                prompt = "<|user|>\n"
            if ex.input:
                prompt += f"{ex.instruction}\n\nInput: {ex.input}"
            else:
                prompt += ex.instruction
            prompt += f"\n<|assistant|>\n{ex.output}"
            records.append({"text": prompt})

        return hf_datasets.Dataset.from_list(records)

    @staticmethod
    def from_jsonl(path: str | Path) -> "datasets.Dataset":
        import datasets as hf_datasets
        return hf_datasets.load_dataset("json", data_files=str(path), split="train")

    @staticmethod
    def from_huggingface(dataset_name: str, split: str = "train") -> "datasets.Dataset":
        import datasets as hf_datasets
        return hf_datasets.load_dataset(dataset_name, split=split)


# ── Public service ────────────────────────────────────────────────────────────

class FineTuneService:
    async def create_job(self, config: TrainingConfig) -> FineTuneJob:
        job_id = str(uuid.uuid4())
        job = FineTuneJob(job_id=job_id, config=config)
        await _job_repo.save(job)
        return job

    async def start_job(self, job_id: str) -> FineTuneJobResponse:
        job = await _job_repo.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if job.status != FineTuneStatus.QUEUED:
            raise ValueError(f"Job {job_id} is already {job.status.value}")

        # Dispatch to Celery worker — clean lifecycle, concurrency limit, timeout
        from app.workers.fine_tune_task import run_fine_tune
        result = run_fine_tune.delay(job_id)
        _celery_task_ids[job_id] = result.id
        logger.info(f"Fine-tune job {job_id} enqueued as Celery task {result.id}")

        return FineTuneJobResponse(
            job_id=job_id,
            status=FineTuneStatus.TRAINING,
            message=f"Training enqueued (task_id={result.id})",
        )

    async def get_job(self, job_id: str) -> Optional[FineTuneJob]:
        return await _job_repo.get(job_id)

    async def list_jobs(self) -> list[FineTuneJob]:
        return await _job_repo.list_all()

    async def cancel_job(self, job_id: str) -> bool:
        job = await _job_repo.get(job_id)
        if job and job.status in (FineTuneStatus.QUEUED, FineTuneStatus.TRAINING):
            # Revoke the Celery task if we have its ID
            task_id = _celery_task_ids.get(job_id)
            if task_id:
                from app.workers.celery_app import celery_app
                celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
                _celery_task_ids.pop(job_id, None)
                logger.info(f"Revoked Celery task {task_id} for job {job_id}")

            job.status = FineTuneStatus.FAILED
            job.error = "Cancelled by user"
            await _job_repo.save(job)
            return True
        return False

    def list_trained_models(self) -> list[dict]:
        models_dir = Path(settings.MODELS_DIR)
        result = []
        for d in models_dir.iterdir():
            if d.is_dir() and (d / "config.json").exists():
                result.append({
                    "name": d.name,
                    "path": str(d),
                    "size_mb": round(
                        sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1
                    ),
                })
        return result

    # ── Document ingestion ────────────────────────────────────────────────────

    def ingest_document_for_training(
        self,
        file_path: str | Path,
        filename: str,
        chunk_size: int = 512,
        overlap: int = 64,
        fmt: str = "text",
    ) -> dict:
        """
        Process a document through the OCR/extraction pipeline, chunk the text,
        and save a JSONL dataset file in settings.DATASETS_DIR.

        Returns metadata dict with dataset_id, dataset_path, chunk_count, preview.
        """
        path = Path(file_path)
        processor = DocumentProcessor()
        doc = processor.process(path)

        if not doc.full_text.strip():
            raise ValueError("No text could be extracted from the document.")

        # Chunk the full text
        words = doc.full_text.split()
        chunks: list[str] = []
        step = max(1, chunk_size - overlap)
        i = 0
        while i < len(words):
            chunk_words = words[i: i + chunk_size]
            chunks.append(" ".join(chunk_words))
            i += step

        # Build JSONL records
        records: list[dict] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            if fmt == "instruction":
                records.append({
                    "instruction": "Summarise or answer questions about the following medical document excerpt.",
                    "input": chunk,
                    "output": "",  # user fills in or uses LLM to generate
                    "text": (
                        f"<|user|>\nSummarise or answer questions about the following medical document excerpt.\n\n"
                        f"Input: {chunk}\n<|assistant|>\n"
                    ),
                })
            else:
                records.append({"text": chunk})

        # Save to datasets dir
        dataset_id = str(uuid.uuid4())[:8]
        stem = Path(filename).stem[:40]
        dataset_filename = f"{stem}_{dataset_id}.jsonl"
        datasets_dir = Path(settings.DATASETS_DIR)
        datasets_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = datasets_dir / dataset_filename

        with open(dataset_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(f"Saved training dataset: {dataset_path} ({len(records)} chunks)")

        return {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "filename": filename,
            "chunk_count": len(records),
            "page_count": doc.metadata.page_count,
            "format": fmt,
            "preview": chunks[0][:300] if chunks else "",
        }

    def list_datasets(self) -> list[dict]:
        """List all JSONL files in the datasets directory."""
        datasets_dir = Path(settings.DATASETS_DIR)
        result = []
        if not datasets_dir.exists():
            return result
        for f in sorted(datasets_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                line_count = sum(1 for _ in open(f, encoding="utf-8"))
            except Exception:
                line_count = 0
            result.append({
                "dataset_id": f.stem.rsplit("_", 1)[-1] if "_" in f.stem else f.stem,
                "filename": f.name,
                "dataset_path": str(f),
                "chunk_count": line_count,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
        return result

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset file by its ID (last segment of stem)."""
        datasets_dir = Path(settings.DATASETS_DIR)
        for f in datasets_dir.glob("*.jsonl"):
            if f.stem.endswith(dataset_id):
                f.unlink()
                return True
        return False

    @staticmethod
    def prepare_ehr_dataset_template() -> list[TrainingExample]:
        """Return a minimal EHR domain template to help users get started."""
        return [
            TrainingExample(
                instruction="Summarise the following clinical note.",
                input="Patient is a 65-year-old male with a history of hypertension and type 2 diabetes. Presents with chest pain radiating to the left arm. ECG shows ST-elevation in leads II, III, aVF. Troponin elevated at 2.1 ng/mL.",
                output="65M with HTN/DM2 presenting with acute inferior STEMI (ST elevation II/III/aVF, troponin 2.1). Urgent cath lab activation indicated.",
                system="You are a clinical documentation assistant.",
            ),
            TrainingExample(
                instruction="Extract all medications from the following discharge summary.",
                input="The patient was discharged on metformin 1000mg BID, lisinopril 10mg QD, atorvastatin 40mg QHS, and aspirin 81mg QD.",
                output="Medications:\n1. Metformin 1000 mg twice daily\n2. Lisinopril 10 mg once daily\n3. Atorvastatin 40 mg at bedtime\n4. Aspirin 81 mg once daily",
                system="You are a clinical documentation assistant.",
            ),
            TrainingExample(
                instruction="What does this lab result indicate?",
                input="HbA1c: 9.2%",
                output="HbA1c of 9.2% indicates poor glycaemic control (target <7%). The patient's diabetes is inadequately managed and medication adjustment or lifestyle intervention should be considered.",
                system="You are a clinical documentation assistant.",
            ),
        ]
