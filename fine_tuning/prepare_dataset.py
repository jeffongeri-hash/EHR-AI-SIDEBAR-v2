#!/usr/bin/env python3
"""
Dataset Preparation Utilities

Converts various formats to the instruction-following JSONL format
used by the fine-tuning pipeline.

Supported input formats:
  - CSV: columns [instruction, input, output] or [question, answer]
  - JSONL: various schemas
  - PDF/text documents → chunked training examples
  - HuggingFace datasets → reformatting

Output format (each line):
  {"text": "<|system|>\\n{system}\\n<|user|>\\n{instruction}\\n<|assistant|>\\n{output}"}

Usage:
    python fine_tuning/prepare_dataset.py \
        --input data/ehr_notes.csv \
        --output datasets/ehr_train.jsonl \
        --format csv \
        --system "You are an EHR clinical assistant."
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger


ALPACA_SYSTEM = "You are a helpful EHR clinical documentation assistant."

PROMPT_TEMPLATE = "{system}\n<|user|>\n{user}\n<|assistant|>\n{output}"


# ── Converters ────────────────────────────────────────────────────────────────

def _make_record(instruction: str, output: str, input_: str = "", system: str = ALPACA_SYSTEM) -> dict:
    user = instruction if not input_ else f"{instruction}\n\nContext: {input_}"
    return {"text": PROMPT_TEMPLATE.format(system=system, user=user, output=output)}


def from_csv(path: Path, system: str = ALPACA_SYSTEM) -> Iterator[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            # Alpaca-style
            if "instruction" in cols and "output" in cols:
                yield _make_record(
                    row.get("instruction", ""),
                    row.get("output", ""),
                    row.get("input", ""),
                    system,
                )
            # Q&A style
            elif "question" in cols and "answer" in cols:
                yield _make_record(row["question"], row["answer"], "", system)
            else:
                # Try first two columns as (user, assistant)
                values = list(row.values())
                if len(values) >= 2:
                    yield _make_record(values[0], values[1], "", system)


def from_jsonl(path: Path, system: str = ALPACA_SYSTEM) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Already in text format
            if "text" in obj:
                yield obj
            # Alpaca / instruction format
            elif "instruction" in obj and "output" in obj:
                yield _make_record(
                    obj["instruction"], obj["output"],
                    obj.get("input", ""), system
                )
            # Conversation format [{"role": ..., "content": ...}]
            elif "conversations" in obj or "messages" in obj:
                convs = obj.get("conversations") or obj.get("messages", [])
                pairs = []
                for i in range(0, len(convs) - 1, 2):
                    if convs[i]["role"] in ("human", "user") and convs[i+1]["role"] in ("gpt", "assistant"):
                        pairs.append(_make_record(convs[i]["content"], convs[i+1]["content"], "", system))
                yield from pairs
            # Sharepoint / QA
            elif "question" in obj and "answer" in obj:
                yield _make_record(obj["question"], obj["answer"], "", system)


def from_text_chunks(
    path: Path,
    chunk_size: int = 512,
    system: str = ALPACA_SYSTEM,
    instruction: str = "Continue the following clinical text:",
) -> Iterator[dict]:
    """Split a raw text file into overlapping chunks for text completion training."""
    text = path.read_text(encoding="utf-8")
    words = text.split()
    step = int(chunk_size * 0.8)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) < 50:
            continue
        # Split chunk at ~60% for context/continuation
        split = int(len(chunk) * 0.6)
        context = chunk[:split]
        continuation = chunk[split:]
        yield _make_record(f"{instruction}\n\n{context}", continuation, "", system)


def from_hf_dataset(
    dataset_name: str,
    split: str = "train",
    system: str = ALPACA_SYSTEM,
    token: Optional[str] = None,
) -> Iterator[dict]:
    import datasets as hf
    ds = hf.load_dataset(dataset_name, split=split, token=token, trust_remote_code=True)
    for row in ds:
        # Try various known field names
        instruction = row.get("instruction") or row.get("question") or row.get("input") or ""
        output = row.get("output") or row.get("answer") or row.get("response") or ""
        context = row.get("context") or row.get("input") or ""
        if instruction and output:
            yield _make_record(instruction, output, context if context != instruction else "", system)


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(records: list[dict], min_len: int = 30) -> list[dict]:
    seen: set[str] = set()
    result = []
    for r in records:
        text = r.get("text", "")
        if len(text) < min_len:
            continue
        # Normalise whitespace for dedup key
        key = re.sub(r"\s+", " ", text).strip()[:200]
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument("--input", required=True, help="Input file path or HF dataset name")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--format", choices=["csv", "jsonl", "text", "hf"], default="jsonl")
    parser.add_argument("--system", default=ALPACA_SYSTEM, help="System prompt")
    parser.add_argument("--split", default="train", help="HF dataset split")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--chunk_size", type=int, default=512, help="Words per chunk (text format)")
    parser.add_argument("--no_dedup", action="store_true", help="Skip deduplication")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {args.input} → {args.output} (format={args.format})")

    records: list[dict] = []
    if args.format == "csv":
        records = list(from_csv(input_path, args.system))
    elif args.format == "jsonl":
        records = list(from_jsonl(input_path, args.system))
    elif args.format == "text":
        records = list(from_text_chunks(input_path, args.chunk_size, args.system))
    elif args.format == "hf":
        records = list(from_hf_dataset(args.input, args.split, args.system, args.hf_token))
    else:
        logger.error(f"Unknown format: {args.format}")
        sys.exit(1)

    logger.info(f"Loaded {len(records)} examples")

    if not args.no_dedup:
        records = deduplicate(records)
        logger.info(f"After deduplication: {len(records)} examples")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.success(f"Saved {len(records)} examples to {output_path}")


if __name__ == "__main__":
    main()
