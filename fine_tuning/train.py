#!/usr/bin/env python3
"""
Standalone fine-tuning script for Llama / DeepSeek / Mistral with LoRA/QLoRA.

Usage:
    python fine_tuning/train.py --config fine_tuning/configs/llama3_ehr.yaml
    python fine_tuning/train.py \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --dataset medalpaca/medical_meadow_medqa \
        --output_name ehr-llama-3b \
        --epochs 3 \
        --batch_size 4 \
        --use_lora \
        --load_in_4bit
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"


@dataclass
class TrainConfig:
    # Model
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_name: str = "fine-tuned-model"
    output_dir: str = "./models"
    hf_token: Optional[str] = None
    push_to_hub: bool = False
    hf_repo_id: Optional[str] = None

    # Dataset
    dataset: Optional[str] = None          # HF dataset name or local JSONL path
    dataset_split: str = "train"
    text_field: str = "text"               # column name in dataset containing training text

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    save_strategy: str = "epoch"

    # LoRA / QLoRA
    use_lora: bool = True
    load_in_4bit: bool = True
    bnb_compute_dtype: str = "float16"
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Hardware
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        lora_data = data.pop("lora", {})
        cfg = cls(**data)
        cfg.lora = LoRAConfig(**lora_data)
        return cfg

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        cfg = cls()
        for k, v in vars(args).items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        return cfg


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(cfg: TrainConfig):
    import datasets as hf_datasets

    path = cfg.dataset
    if not path:
        raise ValueError("Dataset path / name must be provided via --dataset")

    p = Path(path)
    if p.exists() and p.suffix in (".jsonl", ".json"):
        logger.info(f"Loading local JSONL dataset from {p}")
        ds = hf_datasets.load_dataset("json", data_files=str(p), split="train")
    else:
        logger.info(f"Loading HuggingFace dataset '{path}' (split={cfg.dataset_split})")
        ds = hf_datasets.load_dataset(path, split=cfg.dataset_split,
                                       token=cfg.hf_token, trust_remote_code=True)

    logger.info(f"Dataset loaded: {len(ds)} examples, columns: {ds.column_names}")
    return ds


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> str:
    """Run training and return the output directory path."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    logger.info(f"Starting fine-tuning — base model: {cfg.base_model}")
    logger.info(f"Use LoRA: {cfg.use_lora}  |  4-bit: {cfg.load_in_4bit}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, token=cfg.hf_token, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Quantization ──────────────────────────────────────────────────────────
    bnb_config = None
    if cfg.load_in_4bit:
        dtype = torch.float16 if cfg.bnb_compute_dtype == "float16" else torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading model '{cfg.base_model}'…")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        token=cfg.hf_token,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_config = None
    if cfg.use_lora:
        lora = cfg.lora
        logger.info(f"Configuring LoRA  r={lora.r}  alpha={lora.lora_alpha}  modules={lora.target_modules}")
        peft_config = LoraConfig(
            r=lora.r,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            target_modules=lora.target_modules,
            bias=lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(cfg)

    # ── Output dir ────────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir) / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16 and not cfg.load_in_4bit,
        bf16=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field=cfg.text_field,
        report_to="none",
        dataloader_num_workers=cfg.dataloader_num_workers,
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config if not cfg.use_lora else None,  # already applied above
    )

    logger.info("Training started…")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save config
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    # ── Push to Hub ───────────────────────────────────────────────────────────
    if cfg.push_to_hub and cfg.hf_repo_id:
        logger.info(f"Pushing to HuggingFace Hub: {cfg.hf_repo_id}")
        trainer.model.push_to_hub(cfg.hf_repo_id, token=cfg.hf_token)
        tokenizer.push_to_hub(cfg.hf_repo_id, token=cfg.hf_token)

    logger.success(f"Training complete! Model saved to {output_dir}")
    return str(output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Llama/DeepSeek with LoRA/QLoRA")
    p.add_argument("--config", type=str, help="Path to YAML config file")
    p.add_argument("--base_model", type=str)
    p.add_argument("--dataset", type=str, help="HF dataset name or local JSONL path")
    p.add_argument("--output_name", type=str)
    p.add_argument("--output_dir", type=str, default="./models")
    p.add_argument("--epochs", type=int, dest="num_epochs")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float, dest="learning_rate")
    p.add_argument("--max_seq_length", type=int)
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", action="store_false", dest="use_lora")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", action="store_false", dest="load_in_4bit")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_repo_id", type=str)
    p.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        cfg = TrainConfig.from_yaml(args.config)
    else:
        cfg = TrainConfig.from_args(args)

    if not cfg.base_model:
        logger.error("--base_model is required")
        sys.exit(1)
    if not cfg.output_name:
        logger.error("--output_name is required")
        sys.exit(1)

    train(cfg)
