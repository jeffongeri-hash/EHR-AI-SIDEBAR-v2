"""
Local Model Service
===================
Unified service for running ANY HuggingFace causal-LM model locally.

Supported out-of-the-box:
  • TinyLlama/TinyLlama-1.1B-Chat-v1.0          (~2 GB RAM)   — fast, lightweight
  • deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   (~4 GB RAM)   — strong medical reasoning
  • microsoft/Phi-3-mini-4k-instruct             (~8 GB RAM)   — excellent quality
  • mistralai/Mistral-7B-Instruct-v0.3           (~5 GB 4-bit) — best quality general local

Medical-specialised models:
  • BioMistral/BioMistral-7B                     (~5 GB 4-bit) — Mistral fine-tuned on PubMed Central
  • epfl-llm/meditron-7b                         (~5 GB 4-bit) — EPFL, clinical guidelines + PubMed  [GATED]
  • axiong/PMC_LLaMA_13B                         (~9 GB 4-bit) — LLaMA-13B trained on PubMed Central
  • wanglab/ClinicalCamel-70B                    (~40 GB 4-bit)— LLaMA-2-70B, strongest clinical model [GATED]
  • Any other HuggingFace causal-LM model

Features:
  • Lazy loading — model loaded on first use or explicit call
  • Memory management — unload old model before loading new one
  • 4-bit QLoRA quantization when bitsandbytes is available
  • LoRA fine-tuning with asyncio-based progress callbacks
  • Medical-domain prompt formatting
  • Save / reload fine-tuned adapters
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional heavy imports (never crash at module level) ────────────────────────

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("torch not installed — local inference disabled")

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.info("transformers not installed — local inference disabled")

try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.info("peft not installed — LoRA fine-tuning disabled")

try:
    import bitsandbytes  # noqa: F401
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    class Dataset:  # type: ignore[no-redef]
        @classmethod
        def from_list(cls, data):
            return None


# ── Supported model catalogue ───────────────────────────────────────────────────

MODEL_CATALOGUE: Dict[str, Dict[str, Any]] = {
    "tinyllama": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "display_name": "TinyLlama 1.1B Chat",
        "size_gb": 2.2,
        "min_ram_gb": 4,
        "context_length": 2048,
        "description": "Lightweight 1.1B model — fast inference on any hardware",
        "fine_tunable": True,
    },
    "deepseek-1.5b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "display_name": "DeepSeek-R1 1.5B (Distilled)",
        "size_gb": 3.5,
        "min_ram_gb": 6,
        "context_length": 32768,
        "description": "DeepSeek reasoning model — strong medical/clinical reasoning",
        "fine_tunable": True,
    },
    "phi3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "display_name": "Phi-3 Mini 4K",
        "size_gb": 7.6,
        "min_ram_gb": 8,
        "context_length": 4096,
        "description": "Microsoft Phi-3 Mini — high quality, efficient",
        "fine_tunable": True,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "display_name": "Mistral 7B Instruct",
        "size_gb": 4.5,   # 4-bit quantized
        "min_ram_gb": 6,
        "context_length": 32768,
        "description": "Mistral 7B — best quality among local models (4-bit)",
        "fine_tunable": True,
    },
    # ── Medical-specialised models ────────────────────────────────────────────
    "biomistral-7b": {
        "hf_id": "BioMistral/BioMistral-7B",
        "display_name": "BioMistral 7B",
        "size_gb": 4.5,
        "min_ram_gb": 6,
        "context_length": 32768,
        "description": "Mistral fine-tuned on PubMed Central — strong biomedical terminology",
        "fine_tunable": True,
        "medical": True,
        "gated": False,
    },
    "meditron-7b": {
        "hf_id": "epfl-llm/meditron-7b",
        "display_name": "Meditron 7B",
        "size_gb": 4.5,
        "min_ram_gb": 6,
        "context_length": 4096,
        "description": "EPFL — trained on PubMed + clinical guidelines; best for EHR Q&A [requires HF access]",
        "fine_tunable": True,
        "medical": True,
        "gated": True,
    },
    "pmc-llama-13b": {
        "hf_id": "axiong/PMC_LLaMA_13B",
        "display_name": "PMC-LLaMA 13B",
        "size_gb": 9.0,
        "min_ram_gb": 12,
        "context_length": 4096,
        "description": "LLaMA-13B trained on PubMed Central — strong clinical reasoning, open access",
        "fine_tunable": True,
        "medical": True,
        "gated": False,
    },
    "clinicalcamel-70b": {
        "hf_id": "wanglab/ClinicalCamel-70B",
        "display_name": "ClinicalCamel 70B",
        "size_gb": 40.0,
        "min_ram_gb": 48,
        "context_length": 4096,
        "description": "LLaMA-2-70B fine-tuned on clinical notes — highest clinical accuracy [requires HF access]",
        "fine_tunable": False,
        "medical": True,
        "gated": True,
    },
}


@dataclass
class LocalModelConfig:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cache_dir: str = "./models/local"
    max_new_tokens: int = 512
    temperature: float = 0.7
    device: str = "auto"
    load_in_4bit: bool = True     # use 4-bit quantization if bitsandbytes available
    trust_remote_code: bool = True


@dataclass
class TrainProgress:
    """Passed to progress callbacks during fine-tuning."""
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    progress_pct: float = 0.0
    log_line: str = ""
    done: bool = False
    error: Optional[str] = None


# ── Progress callback for Trainer ────────────────────────────────────────────────

class _ProgressCallback:  # type: ignore[misc]
    """HuggingFace TrainerCallback that emits progress to a queue."""

    def __init__(self, queue: asyncio.Queue, total_steps: int):
        self._q = queue
        self._total = total_steps

    def _put(self, p: TrainProgress):
        try:
            self._q.put_nowait(p)
        except asyncio.QueueFull:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss", logs.get("train_loss", 0.0))
        pct = (state.global_step / max(self._total, 1)) * 100
        self._put(TrainProgress(
            epoch=int(state.epoch or 0),
            step=state.global_step,
            total_steps=self._total,
            loss=round(float(loss), 4),
            progress_pct=round(pct, 1),
            log_line=f"Epoch {state.epoch:.1f}  step {state.global_step}/{self._total}  loss={loss:.4f}",
        ))

    def on_train_end(self, args, state, control, **kwargs):
        self._put(TrainProgress(progress_pct=100.0, done=True, log_line="Training complete"))


# ── Main service ─────────────────────────────────────────────────────────────────

class LocalModelService:
    """
    Manages a single loaded local model at a time.
    Call load_model(model_id) to switch models gracefully.
    """

    def __init__(self, config: Optional[LocalModelConfig] = None):
        self.config = config or LocalModelConfig()
        self._tokenizer = None
        self._model = None
        self._active_model_id: Optional[str] = None
        self._peft_model = None          # loaded fine-tuned adapter
        self._fine_tuned_path: Optional[str] = None
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def load_model(self, model_id: Optional[str] = None) -> bool:
        """Load a model by HF model ID or alias. Unloads the current model first."""
        target_id = self._resolve_model_id(model_id or self.config.model_id)
        if self._active_model_id == target_id and self._model is not None:
            logger.info(f"Model '{target_id}' already loaded")
            return True

        self._unload()
        return await asyncio.get_event_loop().run_in_executor(None, self._load_sync, target_id)

    def _load_sync(self, model_id: str) -> bool:
        if not HF_AVAILABLE or not TORCH_AVAILABLE:
            logger.error("transformers/torch not installed — cannot load local model")
            return False
        try:
            logger.info(f"Loading model '{model_id}' …")
            cache = self.config.cache_dir

            # Check if model is gated and warn if HF_TOKEN is missing
            entry = next(
                (v for v in MODEL_CATALOGUE.values() if v["hf_id"] == model_id or
                 model_id in (v.get("alias", ""),)),
                MODEL_CATALOGUE.get(model_id, {})
            )
            hf_token = settings.HF_TOKEN or None
            if entry.get("gated") and not hf_token:
                logger.warning(
                    f"Model '{model_id}' is gated on HuggingFace. "
                    "Set HF_TOKEN in your .env file and request access at "
                    f"https://huggingface.co/{model_id}"
                )

            token_kwargs = {"token": hf_token} if hf_token else {}

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache,
                trust_remote_code=self.config.trust_remote_code,
                **token_kwargs,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "right"

            load_kwargs: Dict[str, Any] = dict(
                cache_dir=cache,
                device_map=self.config.device,
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True,
            )

            if self.config.load_in_4bit and BNB_AVAILABLE:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["quantization_config"] = bnb
            else:
                load_kwargs["torch_dtype"] = torch.float16

            load_kwargs.update(token_kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            self._model.config.use_cache = False
            self._active_model_id = model_id

            # Auto-load fine-tuned adapter for this model if one exists
            self._try_load_peft_adapter(model_id)

            logger.info(f"Model '{model_id}' loaded successfully")
            return True
        except Exception as exc:
            logger.error(f"Failed to load model '{model_id}': {exc}")
            self._model = None
            self._tokenizer = None
            self._active_model_id = None
            return False

    def _unload(self):
        if self._model is not None:
            logger.info(f"Unloading model '{self._active_model_id}'")
            try:
                del self._peft_model
                del self._model
                del self._tokenizer
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._model = None
        self._tokenizer = None
        self._peft_model = None
        self._active_model_id = None

    def _try_load_peft_adapter(self, model_id: str):
        if not PEFT_AVAILABLE:
            return
        adapter_dir = Path(self.config.cache_dir) / "fine-tuned" / self._sanitize(model_id)
        if adapter_dir.exists():
            try:
                self._peft_model = PeftModel.from_pretrained(self._model, str(adapter_dir))
                self._fine_tuned_path = str(adapter_dir)
                logger.info(f"Loaded fine-tuned adapter from {adapter_dir}")
            except Exception as exc:
                logger.warning(f"Could not load adapter from {adapter_dir}: {exc}")

    # ── Inference ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return HF_AVAILABLE and TORCH_AVAILABLE and self._model is not None

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_fine_tuned: bool = True,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response. Auto-loads default model if not loaded."""
        if not self.is_available():
            if not await self.load_model():
                raise RuntimeError(
                    f"Cannot load local model. Install: pip install transformers torch\n"
                    f"Then pull model: {self.config.model_id}"
                )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
            max_new_tokens or self.config.max_new_tokens,
            temperature or self.config.temperature,
            use_fine_tuned,
            system_prompt,
        )

    def _generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        use_fine_tuned: bool,
        system_prompt: Optional[str],
    ) -> str:
        model = (self._peft_model if use_fine_tuned and self._peft_model else self._model)
        formatted = self._format_prompt(prompt, system_prompt)

        inputs = self._tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def stream_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate response and simulate streaming token-by-token."""
        text = await self.generate(prompt, max_new_tokens, temperature,
                                   system_prompt=system_prompt)
        for word in text.split():
            yield word + " "
            await asyncio.sleep(0.02)   # simulate streaming

    # ── Fine-tuning ──────────────────────────────────────────────────────────

    async def fine_tune(
        self,
        examples: List[Dict[str, str]],
        *,
        model_id: Optional[str] = None,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        max_seq_length: int = 512,
        gradient_accumulation_steps: int = 8,
        warmup_steps: int = 20,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        load_in_4bit: bool = True,
        progress_callback: Optional[Callable[[TrainProgress], None]] = None,
    ) -> str:
        """
        Fine-tune the current (or specified) model with LoRA/QLoRA.
        Returns the path to the saved adapter.

        examples: list of dicts with keys  'instruction', 'input' (opt), 'output'
                  OR 'question'/'answer'  OR 'input'/'output'
        """
        if not HF_AVAILABLE or not TORCH_AVAILABLE:
            raise RuntimeError("transformers + torch required. pip install transformers torch")
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft required for LoRA fine-tuning. pip install peft trl")
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets required. pip install datasets")
        if not examples:
            raise ValueError("No training examples provided")

        target_model_id = self._resolve_model_id(model_id or self.config.model_id)
        if not self.is_available() or self._active_model_id != target_model_id:
            if not await self.load_model(target_model_id):
                raise RuntimeError(f"Could not load model '{target_model_id}'")

        output_dir = str(
            Path(self.config.cache_dir) / "fine-tuned" / self._sanitize(target_model_id)
        )
        target_modules = lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

        # Build training text
        texts = [self._format_training_example(ex) for ex in examples]
        if not DATASETS_AVAILABLE:
            raise RuntimeError("pip install datasets")
        dataset = Dataset.from_list([{"text": t} for t in texts])

        # Total training steps
        total_steps = max(1, (len(dataset) * num_epochs) // (batch_size * gradient_accumulation_steps))

        # Progress queue (sync → async bridge)
        progress_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        loop = asyncio.get_event_loop()

        def _run_training():
            try:
                _train_lora(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    dataset=dataset,
                    output_dir=output_dir,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_seq_length=max_seq_length,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=warmup_steps,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    load_in_4bit=load_in_4bit and BNB_AVAILABLE,
                    progress_queue=progress_queue,
                    total_steps=total_steps,
                    loop=loop,
                )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    TrainProgress(done=True, error=str(exc))
                )

        # Run blocking training in thread-pool
        training_task = loop.run_in_executor(None, _run_training)

        # Stream progress to caller
        while True:
            try:
                prog = await asyncio.wait_for(progress_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                logger.warning("Training progress timeout — may still be running")
                continue

            if progress_callback:
                try:
                    progress_callback(prog)
                except Exception:
                    pass

            if prog.done:
                if prog.error:
                    await training_task
                    raise RuntimeError(f"Fine-tuning failed: {prog.error}")
                break

        await training_task

        # Reload the fine-tuned adapter
        self._try_load_peft_adapter(target_model_id)
        logger.info(f"Fine-tuning complete. Adapter saved to {output_dir}")
        return output_dir

    def list_fine_tuned_models(self) -> List[Dict[str, Any]]:
        ft_dir = Path(self.config.cache_dir) / "fine-tuned"
        if not ft_dir.exists():
            return []
        results = []
        for d in ft_dir.iterdir():
            if d.is_dir() and (d / "adapter_config.json").exists():
                try:
                    with open(d / "adapter_config.json") as f:
                        cfg = json.load(f)
                    size_mb = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e6
                    results.append({
                        "name": d.name,
                        "path": str(d),
                        "base_model": cfg.get("base_model_name_or_path", "unknown"),
                        "size_mb": round(size_mb, 1),
                        "lora_r": cfg.get("r", "?"),
                    })
                except Exception:
                    results.append({"name": d.name, "path": str(d)})
        return results

    # ── Info ─────────────────────────────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        return {
            "active_model": self._active_model_id,
            "loaded": self._model is not None,
            "has_fine_tuned_adapter": self._peft_model is not None,
            "fine_tuned_path": self._fine_tuned_path,
            "torch_available": TORCH_AVAILABLE,
            "hf_available": HF_AVAILABLE,
            "peft_available": PEFT_AVAILABLE,
            "bnb_4bit_available": BNB_AVAILABLE,
            "catalogue": list(MODEL_CATALOGUE.keys()),
        }

    # ── Helpers ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_model_id(model_id: str) -> str:
        """Resolve alias (e.g. 'tinyllama') to full HF model ID."""
        entry = MODEL_CATALOGUE.get(model_id.lower())
        if entry:
            return entry["hf_id"]
        return model_id   # assume full HF path

    @staticmethod
    def _sanitize(model_id: str) -> str:
        return model_id.replace("/", "--").replace(":", "_")

    @staticmethod
    def _format_prompt(user_prompt: str, system: Optional[str] = None) -> str:
        sys_text = system or (
            "You are a helpful medical AI assistant. Provide accurate professional "
            "medical information while recommending consultation with qualified "
            "healthcare professionals for specific patient advice."
        )
        return (
            f"<|system|>\n{sys_text}\n"
            f"<|user|>\n{user_prompt}\n"
            f"<|assistant|>\n"
        )

    @staticmethod
    def _format_training_example(ex: Dict[str, str]) -> str:
        instruction = (
            ex.get("instruction")
            or ex.get("question")
            or ex.get("input", "")
        )
        output = (
            ex.get("output")
            or ex.get("answer")
            or ex.get("response", "")
        )
        input_ctx = ex.get("input", "") if ex.get("instruction") else ""
        system = ex.get("system", "You are a helpful medical AI assistant.")

        prompt = f"<|system|>\n{system}\n<|user|>\n{instruction}"
        if input_ctx:
            prompt += f"\n\nContext: {input_ctx}"
        prompt += f"\n<|assistant|>\n{output}"
        return prompt


# ── LoRA training helper (runs in thread-pool) ──────────────────────────────────

def _train_lora(
    *,
    model,
    tokenizer,
    dataset,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    load_in_4bit: bool,
    progress_queue: asyncio.Queue,
    total_steps: int,
    loop: asyncio.AbstractEventLoop,
):
    """Blocking LoRA training — runs inside run_in_executor."""
    import torch

    # Prepare model for k-bit training if quantized
    if load_in_4bit and PEFT_AVAILABLE:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, peft_config)
    logger.info(f"Trainable params: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad):,}")

    # Tokenize
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        fp16=TORCH_AVAILABLE and torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=max(1, total_steps // 20),
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    cb = _ProgressCallback(progress_queue, total_steps)
    # _ProgressCallback is not a real TrainerCallback but quacks like one

    class _Callback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            cb.on_log(args, state, control, logs=logs)

        def on_train_end(self, args, state, control, **kw):
            cb.on_train_end(args, state, control)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[_Callback()],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved to {output_dir}")


# ── Global singleton ───────────────────────────────────────────────────────────

local_model_service = LocalModelService()
