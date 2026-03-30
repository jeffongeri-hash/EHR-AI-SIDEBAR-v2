"""
Model Comparison Service

Runs the same prompt concurrently against:
  1. A base Llama model via the local Ollama server
  2. A fine-tuned PEFT adapter loaded from the local models directory

Both calls are made in parallel via asyncio.gather so the wall-clock wait is
roughly max(base_latency, ft_latency) instead of the sum.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from app.config import settings
from app.models.schemas import ModelCompareRequest, ModelCompareResult, ModelCompareResponse
from app.services.llm_service import EHR_SYSTEM_PROMPT

# Cache loaded HF pipelines so repeated comparisons don't reload the model
_hf_pipeline_cache: dict[str, object] = {}


# ── Ollama inference ──────────────────────────────────────────────────────────

async def _run_ollama(
    model: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> ModelCompareResult:
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        latency_ms = int((time.monotonic() - start) * 1000)
        response_text = data.get("response", "")
        tokens_generated = data.get("eval_count")

        return ModelCompareResult(
            model_name=model,
            response=response_text,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
        )

    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.warning(f"Ollama comparison failed for model {model}: {exc}")
        return ModelCompareResult(
            model_name=model,
            response="",
            latency_ms=latency_ms,
            error=str(exc),
        )


# ── HuggingFace PEFT inference (blocking → run in executor) ──────────────────

def _load_hf_pipeline(model_path: str):
    """Load (or retrieve from cache) a text-generation pipeline for a PEFT adapter."""
    if model_path in _hf_pipeline_cache:
        return _hf_pipeline_cache[model_path]

    # Import heavy deps lazily so the server starts even without GPU
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel, PeftConfig

    logger.info(f"Loading fine-tuned PEFT model from {model_path} …")
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_id = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    _hf_pipeline_cache[model_path] = pipe
    logger.info(f"Fine-tuned model loaded: {model_path}")
    return pipe


def _run_hf_inference_sync(
    model_path: str,
    model_name: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> ModelCompareResult:
    """Blocking HF inference — must be called inside run_in_executor."""
    start = time.monotonic()
    try:
        pipe = _load_hf_pipeline(model_path)
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"

        outputs = pipe(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,
        )
        generated = outputs[0]["generated_text"] if outputs else ""
        latency_ms = int((time.monotonic() - start) * 1000)

        # Approximate token count
        tokens_generated = len(pipe.tokenizer.encode(generated))

        return ModelCompareResult(
            model_name=model_name,
            response=generated,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
        )

    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.warning(f"HF inference failed for {model_path}: {exc}")
        return ModelCompareResult(
            model_name=model_name,
            response="",
            latency_ms=latency_ms,
            error=str(exc),
        )


async def _run_hf_async(
    model_path: str,
    model_name: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> ModelCompareResult:
    """Wrap blocking HF inference in asyncio executor so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _run_hf_inference_sync,
        model_path,
        model_name,
        prompt,
        system_prompt,
        max_tokens,
        temperature,
    )


# ── Public service function ────────────────────────────────────────────────────

async def compare_models(req: ModelCompareRequest) -> ModelCompareResponse:
    """
    Run the same prompt against the base Ollama model and a local fine-tuned
    model concurrently, returning both results with latency metrics.
    """
    models_dir = Path(settings.MODELS_DIR)
    model_path = str(models_dir / req.fine_tuned_model)

    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model '{req.fine_tuned_model}' not found in {settings.MODELS_DIR}"
        )

    system = req.system_prompt or EHR_SYSTEM_PROMPT

    base_task = _run_ollama(
        model=req.base_model,
        prompt=req.prompt,
        system_prompt=system,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    ft_task = _run_hf_async(
        model_path=model_path,
        model_name=req.fine_tuned_model,
        prompt=req.prompt,
        system_prompt=system,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    base_result, ft_result = await asyncio.gather(base_task, ft_task)

    logger.info(
        f"Comparison complete — base: {base_result.latency_ms}ms, "
        f"fine-tuned: {ft_result.latency_ms}ms"
    )

    return ModelCompareResponse(
        prompt=req.prompt,
        base=base_result,
        fine_tuned=ft_result,
    )
