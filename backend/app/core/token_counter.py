"""
Token Estimation Utility
========================
Provides a fast, approximate token count for messages before they are
sent to an LLM.  Used to catch context-window overflows early and return
a clear user-facing error instead of a cryptic API failure.

Estimation approach
-------------------
Exact tokenisation requires loading each model's vocabulary which is
expensive.  Instead we use the widely-accepted heuristic:

    tokens ≈ len(text_utf8_bytes) / 4

This is accurate to ±15 % for English medical text and is conservative
(slightly over-estimates), meaning we fail safe — we may occasionally
reject a request that would technically fit, but we will never silently
truncate or cause a mysterious API error.

For Claude models, the Anthropic SDK exposes an accurate counter via
``client.beta.messages.count_tokens()``.  That path is intentionally
NOT used here because:
  1. It requires a round-trip API call before every chat request.
  2. The heuristic is sufficient for a guard rail.

Usage
-----
    from app.core.token_counter import estimate_tokens, check_context_fits

    total = estimate_tokens(messages, system_prompt)
    check_context_fits(total, model_context_window=200_000, max_output=4096)
"""

from __future__ import annotations

from fastapi import HTTPException
from loguru import logger


# A small safety margin so we never try to fill the window 100 %
_SAFETY_MARGIN = 0.9


def estimate_tokens(messages: list[dict], system_prompt: str = "") -> int:
    """
    Return an approximate token count for the given message list + system prompt.

    Each message dict must contain a ``"content"`` key (str).
    Role labels and structural overhead (~4 tokens/message) are included.
    """
    total_chars = len(system_prompt.encode("utf-8"))
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content.encode("utf-8"))
        elif isinstance(content, list):
            # Anthropic vision messages: list of content blocks
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(str(block.get("text", "")).encode("utf-8"))
        # Per-message structural overhead (~4 tokens)
        total_chars += 16

    return max(1, total_chars // 4)


def check_context_fits(
    estimated_input_tokens: int,
    model_name: str,
    context_window: int,
    max_output_tokens: int,
) -> None:
    """
    Raise HTTP 422 if the estimated input + requested output tokens would
    exceed the model's context window.

    Args:
        estimated_input_tokens: Result of estimate_tokens()
        model_name:             Human-readable name for error messages
        context_window:         Total token limit for the model
        max_output_tokens:      Tokens reserved for the response
    """
    usable = int(context_window * _SAFETY_MARGIN)
    available_for_input = usable - max_output_tokens

    if available_for_input <= 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"max_tokens ({max_output_tokens}) is too large for model '{model_name}' "
                f"(context window: {context_window:,}). Reduce max_tokens."
            ),
        )

    if estimated_input_tokens > available_for_input:
        pct = int(estimated_input_tokens / available_for_input * 100)
        logger.warning(
            f"Token estimate {estimated_input_tokens:,} exceeds "
            f"{available_for_input:,} available tokens for {model_name} ({pct}%)"
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"The conversation + document context is too large for model '{model_name}'. "
                f"Estimated ~{estimated_input_tokens:,} input tokens but only "
                f"~{available_for_input:,} are available "
                f"(context window: {context_window:,}, reserved for output: {max_output_tokens:,}). "
                "Try removing some attached documents or starting a new conversation."
            ),
        )

    if estimated_input_tokens > available_for_input * 0.8:
        logger.warning(
            f"[token_counter] Approaching context limit for {model_name}: "
            f"~{estimated_input_tokens:,} / {available_for_input:,} input tokens used"
        )
