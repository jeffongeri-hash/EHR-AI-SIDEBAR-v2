"""
Fine-Tuning Celery Task
========================
The actual LLM training logic extracted from FineTuneService and wrapped
as a proper Celery task so it:

  - Runs in an isolated worker process (not inside the FastAPI server)
  - Obeys the worker's concurrency limit (1 GPU job at a time)
  - Can be cleanly cancelled via celery_app.control.revoke(task_id)
  - Has an enforced time limit (4 hours soft, 4.5 hours hard)
  - Persists status to the DB at every lifecycle stage using synchronous
    SQLAlchemy (Celery workers are sync, not async)

DB access
---------
Celery workers cannot use FastAPI's async SQLAlchemy engine.
A separate SYNC engine is created in this module for the worker process.
The DATABASE_URL is adapted (aiosqlite → pysqlite, asyncpg → psycopg2).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import FineTuneJobRow
from app.workers.celery_app import celery_app


# ── Synchronous DB engine for the Celery worker process ──────────────────────

def _make_sync_url(async_url: str) -> str:
    """Convert async driver URL to sync equivalent."""
    return (
        async_url
        .replace("sqlite+aiosqlite", "sqlite")
        .replace("postgresql+asyncpg", "postgresql+psycopg2")
    )


_sync_engine = create_engine(
    _make_sync_url(settings.DATABASE_URL),
    echo=False,
    connect_args=(
        {"check_same_thread": False}
        if "sqlite" in settings.DATABASE_URL else {}
    ),
)


def _get_job(session: Session, job_id: str) -> Optional[FineTuneJobRow]:
    return session.get(FineTuneJobRow, job_id)


def _save_job(session: Session, row: FineTuneJobRow) -> None:
    session.add(row)
    session.commit()


# ── Celery task ───────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.workers.fine_tune_task.run_fine_tune",
    max_retries=0,  # training failures should not auto-retry
)
def run_fine_tune(self, job_id: str) -> dict:
    """
    Execute a fine-tuning job by job_id.

    The job record must already exist in the database (created by
    FineTuneService.create_job before the task is enqueued).

    Returns a dict with job_id, status, and output_path on success.
    """
    with Session(_sync_engine) as session:
        row = _get_job(session, job_id)
        if not row:
            raise ValueError(f"Job {job_id} not found in database")

        cfg_dict = row.config_json
        logs: list[str] = list(row.logs_json or [])

        def _update(status: str, progress: float = None, **kwargs):
            """Persist job state to DB."""
            row.status = status
            if progress is not None:
                row.progress = progress
            for k, v in kwargs.items():
                setattr(row, k, v)
            row.logs_json = logs
            _save_job(session, row)

        row.status = "preparing"
        row.started_at = datetime.now(timezone.utc)
        logs.append(f"[Celery task {self.request.id}] Starting fine-tune of '{cfg_dict['base_model']}'")
        _save_job(session, row)

    try:
        _do_training(job_id, cfg_dict, logs, _update_fn=lambda **kw: _persist(job_id, logs, **kw))
        return {"job_id": job_id, "status": "completed"}

    except SoftTimeLimitExceeded:
        logger.warning(f"Fine-tune job {job_id} hit soft time limit (4h)")
        _persist(job_id, logs, status="failed", error="Training exceeded 4-hour time limit")
        return {"job_id": job_id, "status": "failed", "reason": "timeout"}

    except Exception as exc:
        logger.exception(f"Fine-tune job {job_id} failed: {exc}")
        _persist(job_id, logs, status="failed", error=str(exc))
        return {"job_id": job_id, "status": "failed", "reason": str(exc)}


def _persist(
    job_id: str,
    logs: list[str],
    status: str = None,
    progress: float = None,
    error: str = None,
    output_path: str = None,
    completed_at: datetime = None,
) -> None:
    """Write job state to the DB from within the Celery task."""
    with Session(_sync_engine) as session:
        row = session.get(FineTuneJobRow, job_id)
        if not row:
            return
        if status:
            row.status = status
        if progress is not None:
            row.progress = progress
        if error is not None:
            row.error = error
        if output_path is not None:
            row.output_path = output_path
        if completed_at is not None:
            row.completed_at = completed_at
        row.logs_json = logs[-200:]  # cap at 200 entries
        session.commit()


# ── Core training logic (unchanged from fine_tune_service._run_training) ─────

def _do_training(job_id: str, cfg_dict: dict, logs: list[str], _update_fn) -> None:
    """
    Runs the actual HuggingFace training loop.
    Extracted here so it can be called from the Celery task.
    """
    from app.models.schemas import TrainingConfig
    cfg = TrainingConfig(**cfg_dict)

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from transformers import TrainerCallback
        import datasets as hf_datasets

        # 1. Tokenizer
        logs.append("Loading tokenizer…")
        _update_fn(status="preparing")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model, token=settings.HF_TOKEN, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 2. Quantization
        bnb_config = None
        if cfg.load_in_4bit:
            logs.append("Configuring QLoRA 4-bit…")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.float16 if cfg.bnb_4bit_compute_dtype == "float16"
                    else torch.bfloat16
                ),
            )

        # 3. Model
        logs.append(f"Loading '{cfg.base_model}'…")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            token=settings.HF_TOKEN,
            trust_remote_code=True,
        )
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        # 4. LoRA
        peft_config = None
        if cfg.use_lora:
            lora = cfg.lora_config
            logs.append(f"LoRA r={lora.r} alpha={lora.lora_alpha}…")
            peft_config = LoraConfig(
                r=lora.r,
                lora_alpha=lora.lora_alpha,
                lora_dropout=lora.lora_dropout,
                target_modules=lora.target_modules,
                bias=lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )

        # 5. Dataset
        logs.append("Preparing dataset…")
        _update_fn(status="training")
        if cfg.training_examples:
            records = []
            for ex in cfg.training_examples:
                prefix = f"<|system|>\n{ex.system}\n" if ex.system else ""
                body = f"<|user|>\n{ex.instruction}"
                if ex.input:
                    body += f"\n\nInput: {ex.input}"
                body += f"\n<|assistant|>\n{ex.output}"
                records.append({"text": prefix + body})
            dataset = hf_datasets.Dataset.from_list(records)
        elif cfg.dataset_path:
            p = Path(cfg.dataset_path)
            if p.suffix in (".jsonl", ".json"):
                dataset = hf_datasets.load_dataset("json", data_files=str(p), split="train")
            else:
                dataset = hf_datasets.load_dataset(cfg.dataset_path, split="train")
        else:
            raise ValueError("Either training_examples or dataset_path required")

        logs.append(f"Dataset ready — {len(dataset)} examples")

        # 6. Output dir
        output_dir = Path(settings.MODELS_DIR) / cfg.output_model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 7. Progress callback
        total_steps = max(
            1,
            (len(dataset) // (cfg.batch_size * cfg.gradient_accumulation_steps))
            * cfg.num_epochs,
        )

        class _Hook(TrainerCallback):
            def on_log(self_, args, state, control, logs_=None, **kwargs):
                if logs_ and "loss" in logs_:
                    step = state.global_step
                    loss = logs_["loss"]
                    progress = min(0.99, step / total_steps)
                    logs.append(f"Step {step}/{total_steps} loss={loss:.4f}")
                    _update_fn(progress=progress, status="training")
            def on_epoch_end(self_, args, state, control, **kwargs):
                logs.append(f"Epoch {int(state.epoch)} completed")

        # 8. Trainer
        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            fp16=cfg.fp16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            logging_steps=10,
            save_strategy="epoch",
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="text",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[_Hook()],
        )

        logs.append("Training started…")
        trainer.train()

        # 9. Save
        logs.append("Saving model…")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # 10. Push (optional)
        if cfg.push_to_hub and cfg.hf_repo_id:
            logs.append(f"Pushing to Hub: {cfg.hf_repo_id}…")
            trainer.model.push_to_hub(cfg.hf_repo_id, token=settings.HF_TOKEN)
            tokenizer.push_to_hub(cfg.hf_repo_id, token=settings.HF_TOKEN)

        logs.append(f"Training complete! Model saved to {output_dir}")
        _update_fn(
            status="completed",
            progress=1.0,
            output_path=str(output_dir),
            completed_at=datetime.now(timezone.utc),
        )

    except SoftTimeLimitExceeded:
        raise  # let the task handler deal with it
    except Exception:
        raise
