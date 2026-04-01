"""
LLM Service
===========
Supports multiple providers:
  - Anthropic Claude (claude-sonnet-4-6, claude-opus-4-6, etc.)
  - Ollama  (local: deepseek-r1, mistral, llama3, phi3, etc.)
  - LocalHF (HuggingFace models loaded directly: TinyLlama, DeepSeek, Mistral, Phi-3)

All providers expose a unified async interface.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import anthropic
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.token_counter import check_context_fits, estimate_tokens
from app.db.repositories import ConversationRepository
from app.services.flight_service import flight_service
from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    MessageRole,
    ModelInfo,
    ModelProvider,
)

_conv_repo = ConversationRepository()

# ── Context window limits (tokens) ───────────────────────────────────────────
_CONTEXT_WINDOWS: dict[str, int] = {
    # Claude
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # Ollama / local Llama
    "llama3.2": 128_000,
    "llama3.1": 128_000,
    "llama3": 8_192,
    "llama2": 4_096,
    # Mistral
    "mistral": 32_768,
    "mistral-7b": 32_768,
    "mistral-nemo": 128_000,
    # DeepSeek
    "deepseek-r1": 64_000,
    "deepseek-r1:1.5b": 32_768,
    "deepseek-r1:7b": 64_000,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 32_768,
    # Phi
    "phi3": 128_000,
    "phi3:mini": 4_096,
    "microsoft/Phi-3-mini-4k-instruct": 4_096,
    # TinyLlama / local HF
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2_048,
    "tinyllama": 2_048,
    "mistralai/Mistral-7B-Instruct-v0.3": 32_768,
    # Other
    "gemma2": 8_192,
    "qwen2": 32_768,
}
_DEFAULT_CONTEXT = 8_192

EHR_SYSTEM_PROMPT = """You are an intelligent EHR (Electronic Health Record) assistant with
expertise in medical documentation, clinical notes, lab results, and patient data analysis.

You can:
- Analyse uploaded medical documents (PDFs, images, scans)
- Answer questions about patient records, diagnoses, and treatments
- Extract structured information from unstructured clinical text
- Summarise lengthy documents
- Identify key medical findings, dates, medications, and lab values

Always be accurate, cite the document when quoting, and flag any ambiguities.
Do NOT provide diagnoses or treatment recommendations — refer clinical decisions to qualified practitioners."""


# ── Base provider ─────────────────────────────────────────────────────────────

class BaseLLMProvider:
    async def chat(self, request: ChatRequest, context: str = "") -> ChatResponse:
        raise NotImplementedError

    async def stream_chat(self, request: ChatRequest, context: str = "") -> AsyncGenerator[str, None]:
        raise NotImplementedError

    async def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError


# ── Anthropic Claude ──────────────────────────────────────────────────────────

class ClaudeProvider(BaseLLMProvider):
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def chat(self, request: ChatRequest, context: str = "") -> ChatResponse:
        model = request.model_name or settings.CLAUDE_MODEL
        messages = await self._build_messages(request, context)
        system = request.system_prompt or EHR_SYSTEM_PROMPT

        last_msg = messages[-1].get("content", "") if messages else ""
        tools = self._flight_tools() if self._is_travel_query(last_msg) else None

        ctx = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT)
        check_context_fits(
            estimated_input_tokens=estimate_tokens(messages, system),
            model_name=model, context_window=ctx, max_output_tokens=request.max_tokens,
        )

        params = dict(
            model=model, max_tokens=request.max_tokens,
            temperature=request.temperature, system=system, messages=messages,
        )
        if tools:
            params["tools"] = tools

        response = await self.client.messages.create(**params)

        # Handle tool use
        if tools and response.content and len(response.content) > 1:
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_result = await self._execute_tool(block)
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": block.id, "content": tool_result}
                    ]})
                    response = await self.client.messages.create(
                        model=model, max_tokens=request.max_tokens,
                        temperature=request.temperature, system=system,
                        messages=messages, tools=tools,
                    )
                    content = response.content[0].text if response.content else ""
                    return self._build_response(request, content, model, {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    })

        content = response.content[0].text if response.content else ""
        return self._build_response(request, content, model, {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

    async def stream_chat(self, request: ChatRequest, context: str = "") -> AsyncGenerator[str, None]:
        model = request.model_name or settings.CLAUDE_MODEL
        messages = await self._build_messages(request, context)
        system = request.system_prompt or EHR_SYSTEM_PROMPT
        ctx = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT)
        check_context_fits(
            estimated_input_tokens=estimate_tokens(messages, system),
            model_name=model, context_window=ctx, max_output_tokens=request.max_tokens,
        )
        async with self.client.messages.stream(
            model=model, max_tokens=request.max_tokens,
            temperature=request.temperature, system=system, messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name="claude-sonnet-4-6", provider=ModelProvider.CLAUDE,
                      display_name="Claude Sonnet 4.6", context_length=200000,
                      supports_vision=True,
                      description="Latest Claude Sonnet — fast & powerful"),
            ModelInfo(name="claude-opus-4-6", provider=ModelProvider.CLAUDE,
                      display_name="Claude Opus 4.6", context_length=200000,
                      supports_vision=True, description="Most capable Claude model"),
            ModelInfo(name="claude-haiku-4-5-20251001", provider=ModelProvider.CLAUDE,
                      display_name="Claude Haiku 4.5", context_length=200000,
                      supports_vision=True, description="Fastest Claude model"),
        ]

    # helpers
    @staticmethod
    def _is_travel_query(msg: str) -> bool:
        kws = ["flight", "travel", "airport", "chase rewards", "chase points",
               "first class", "business class", "book a flight", "find flights"]
        ml = msg.lower()
        return any(k in ml for k in kws)

    @staticmethod
    def _flight_tools():
        return [{
            "name": "search_flights",
            "description": "Search for flights with Chase Ultimate Rewards pricing",
            "input_schema": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "departure_date": {"type": "string"},
                    "class_type": {"type": "string", "enum": ["economy", "business", "first"]},
                    "chase_points": {"type": "integer"},
                },
                "required": ["origin", "destination", "departure_date", "chase_points"],
            },
        }]

    async def _execute_tool(self, block) -> str:
        try:
            if block.name == "search_flights":
                p = block.input
                recs = await flight_service.get_flight_recommendations(
                    origin=p.get("origin", ""),
                    destination=p.get("destination", ""),
                    budget_points=p.get("chase_points", 0),
                    preferred_class=p.get("class_type", "economy"),
                )
                return self._fmt_flights(recs, p.get("chase_points", 0))
        except Exception as exc:
            return f"Flight search error: {exc}"
        return "Tool not found"

    @staticmethod
    def _fmt_flights(recs: dict, pts: int) -> str:
        if not recs.get("recommendations"):
            return "No flights found."
        lines = [f"Found options for {pts:,} Chase points:\n"]
        for i, r in enumerate(recs["recommendations"][:5], 1):
            f = r["flight"]
            rw = r["rewards_info"]
            lines.append(
                f"{i}. {f.airline} {f.flight_number} — {f.origin}→{f.destination} "
                f"${f.price_usd:,.0f} / {rw.points_needed:,} pts"
            )
        return "\n".join(lines)

    @staticmethod
    async def _build_messages(request: ChatRequest, context: str) -> list[dict]:
        history = None
        if request.conversation_id:
            history = await _conv_repo.get(request.conversation_id)
        if history is None:
            history = ConversationHistory(
                conversation_id=request.conversation_id or "",
                messages=[], created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        msgs: list[dict] = []
        for msg in history.messages[-20:]:
            msgs.append({"role": msg.role.value, "content": msg.content})
        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"
        msgs.append({"role": "user", "content": user_content})
        return msgs

    @staticmethod
    def _build_response(request: ChatRequest, content: str, model: str, usage: dict) -> ChatResponse:
        conv_id = request.conversation_id or str(uuid.uuid4())
        return ChatResponse(
            conversation_id=conv_id,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            model_used=model, usage=usage,
        )


# ── Ollama (local server) ─────────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    BASE = settings.OLLAMA_BASE_URL

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def chat(self, request: ChatRequest, context: str = "") -> ChatResponse:
        model = request.model_name or settings.OLLAMA_DEFAULT_MODEL
        messages = await self._build_messages(request, context)
        ctx = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT)
        check_context_fits(
            estimated_input_tokens=estimate_tokens(messages),
            model_name=model, context_window=ctx, max_output_tokens=request.max_tokens,
        )
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.BASE}/api/chat",
                json={"model": model, "messages": messages, "stream": False,
                      "options": {"temperature": request.temperature,
                                  "num_predict": request.max_tokens}},
            )
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        conv_id = request.conversation_id or str(uuid.uuid4())
        return ChatResponse(
            conversation_id=conv_id,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            model_used=model,
            usage={"input_tokens": data.get("prompt_eval_count", 0),
                   "output_tokens": data.get("eval_count", 0)},
        )

    async def stream_chat(self, request: ChatRequest, context: str = "") -> AsyncGenerator[str, None]:
        model = request.model_name or settings.OLLAMA_DEFAULT_MODEL
        messages = await self._build_messages(request, context)
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{self.BASE}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            text = chunk.get("message", {}).get("content", "")
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            pass

    async def list_models(self) -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.BASE}/api/tags")
                resp.raise_for_status()
                data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                size_gb = round(m.get("size", 0) / 1e9, 1)
                models.append(ModelInfo(
                    name=name, provider=ModelProvider.OLLAMA,
                    display_name=name,
                    context_length=_CONTEXT_WINDOWS.get(name.split(":")[0], 8192),
                    supports_vision="llava" in name or "vision" in name,
                    is_local=True, size_gb=size_gb,
                    description=f"Local Ollama model — {size_gb} GB",
                    fine_tunable=False,
                ))
            return models
        except Exception as exc:
            logger.warning(f"Ollama not reachable: {exc}")
            return []

    @staticmethod
    async def _build_messages(request: ChatRequest, context: str) -> list[dict]:
        msgs: list[dict] = [{"role": "system",
                             "content": request.system_prompt or EHR_SYSTEM_PROMPT}]
        if request.conversation_id:
            history = await _conv_repo.get(request.conversation_id)
            if history:
                for msg in history.messages[-18:]:
                    msgs.append({"role": msg.role.value, "content": msg.content})
        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"
        msgs.append({"role": "user", "content": user_content})
        return msgs


# ── Local HuggingFace provider ────────────────────────────────────────────────

class LocalHFProvider(BaseLLMProvider):
    """
    Runs any HuggingFace causal-LM model locally via local_model_service.
    Supports TinyLlama, DeepSeek-R1, Phi-3-mini, Mistral-7B, and more.
    """

    def __init__(self):
        from app.services.local_model_service import local_model_service, MODEL_CATALOGUE
        self._svc = local_model_service
        self._catalogue = MODEL_CATALOGUE

    async def chat(self, request: ChatRequest, context: str = "") -> ChatResponse:
        model_id = request.model_name or self._svc.config.model_id

        # Load model if not already loaded or if different model requested
        if not self._svc.is_available() or self._svc._active_model_id != self._svc._resolve_model_id(model_id):
            logger.info(f"Loading local model: {model_id}")
            if not await self._svc.load_model(model_id):
                raise RuntimeError(
                    f"Could not load local model '{model_id}'. "
                    f"Install: pip install transformers torch accelerate"
                )

        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"

        response_text = await self._svc.generate(
            prompt=user_content,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            system_prompt=request.system_prompt or EHR_SYSTEM_PROMPT,
        )

        conv_id = request.conversation_id or str(uuid.uuid4())
        display = self._svc._active_model_id or model_id
        if self._svc._peft_model:
            display += " (fine-tuned)"

        return ChatResponse(
            conversation_id=conv_id,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response_text),
            model_used=display,
            usage={
                "input_tokens": len(user_content.split()),
                "output_tokens": len(response_text.split()),
            },
        )

    async def stream_chat(self, request: ChatRequest, context: str = "") -> AsyncGenerator[str, None]:
        model_id = request.model_name or self._svc.config.model_id
        if not self._svc.is_available():
            await self._svc.load_model(model_id)

        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"

        async for chunk in self._svc.stream_generate(
            prompt=user_content,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            system_prompt=request.system_prompt or EHR_SYSTEM_PROMPT,
        ):
            yield chunk

    async def list_models(self) -> list[ModelInfo]:
        from app.services.local_model_service import MODEL_CATALOGUE, HF_AVAILABLE, TORCH_AVAILABLE
        models = []
        for alias, meta in MODEL_CATALOGUE.items():
            available = HF_AVAILABLE and TORCH_AVAILABLE
            loaded = (self._svc._active_model_id == meta["hf_id"])
            desc = meta["description"]
            if loaded:
                desc += " \u2014 LOADED"
            elif not available:
                desc += " (install transformers + torch to enable)"
            models.append(ModelInfo(
                name=meta["hf_id"],
                provider=ModelProvider.LOCAL_HF,
                display_name=meta["display_name"],
                context_length=meta["context_length"],
                supports_vision=False,
                is_local=True,
                size_gb=meta["size_gb"],
                description=desc,
                fine_tunable=meta["fine_tunable"],
            ))
        # Also show any fine-tuned adapters as separate entries
        for ft in self._svc.list_fine_tuned_models():
            models.append(ModelInfo(
                name=ft["path"],
                provider=ModelProvider.LOCAL_HF,
                display_name=f"{ft['name']} (fine-tuned)",
                context_length=2048,
                is_local=True,
                size_gb=round(ft.get("size_mb", 0) / 1024, 2),
                description=f"Fine-tuned adapter on {ft.get('base_model', 'unknown')}",
                fine_tunable=True,
            ))
        return models


# ── TINYLLAMA alias (backward-compat) ─────────────────────────────────────────

class TinyLlamaProvider(LocalHFProvider):
    """Kept for backward-compat — now delegates to LocalHFProvider."""

    async def list_models(self) -> list[ModelInfo]:
        from app.services.local_model_service import HF_AVAILABLE, TORCH_AVAILABLE
        models = [
            ModelInfo(
                name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                provider=ModelProvider.TINYLLAMA,
                display_name="TinyLlama 1.1B Chat",
                context_length=2048, supports_vision=False, is_local=True,
                size_gb=2.2, fine_tunable=True,
                description="Lightweight 1.1B model — fast on any hardware",
            )
        ]
        if self._svc._peft_model:
            models.append(ModelInfo(
                name="TinyLlama-Medical",
                provider=ModelProvider.TINYLLAMA,
                display_name="TinyLlama Medical (fine-tuned)",
                context_length=2048, is_local=True, size_gb=2.3, fine_tunable=True,
                description="TinyLlama fine-tuned on EHR data",
            ))
        return models


# ── LLM Service router ────────────────────────────────────────────────────────

class LLMService:
    def __init__(self):
        self._providers: dict[ModelProvider, BaseLLMProvider] = {}

    def _get_provider(self, provider: ModelProvider) -> BaseLLMProvider:
        if provider not in self._providers:
            if provider == ModelProvider.CLAUDE:
                self._providers[provider] = ClaudeProvider()
            elif provider == ModelProvider.OLLAMA:
                self._providers[provider] = OllamaProvider()
            elif provider in (ModelProvider.LOCAL_HF, ModelProvider.TINYLLAMA):
                self._providers[provider] = LocalHFProvider()
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._providers[provider]

    async def chat(self, request: ChatRequest, document_context: str = "") -> ChatResponse:
        provider = self._get_provider(request.model_provider)
        response = await provider.chat(request, document_context)
        await self._update_history(request, response)
        return response

    async def stream_chat(
        self, request: ChatRequest, document_context: str = ""
    ) -> AsyncGenerator[str, None]:
        provider = self._get_provider(request.model_provider)
        collected = []
        async for chunk in provider.stream_chat(request, document_context):
            collected.append(chunk)
            yield chunk
        full = "".join(collected)
        conv_id = request.conversation_id or str(uuid.uuid4())
        synthetic = ChatResponse(
            conversation_id=conv_id,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=full),
            model_used=request.model_name or "unknown",
        )
        await self._update_history(request, synthetic)

    async def list_all_models(self) -> list[ModelInfo]:
        claude = await self._get_provider(ModelProvider.CLAUDE).list_models()
        ollama = await self._get_provider(ModelProvider.OLLAMA).list_models()
        local_hf = await self._get_provider(ModelProvider.LOCAL_HF).list_models()
        return claude + ollama + local_hf

    async def get_conversation(self, conv_id: str) -> Optional[ConversationHistory]:
        return await _conv_repo.get(conv_id)

    async def clear_conversation(self, conv_id: str) -> bool:
        return await _conv_repo.delete(conv_id)

    @staticmethod
    async def _update_history(request: ChatRequest, response: ChatResponse) -> None:
        conv_id = response.conversation_id
        history = await _conv_repo.get(conv_id)
        if history is None:
            history = ConversationHistory(
                conversation_id=conv_id, messages=[],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        history.messages.append(ChatMessage(role=MessageRole.USER, content=request.message))
        history.messages.append(response.message)
        history.updated_at = datetime.now(timezone.utc)
        for did in request.document_ids:
            if did not in history.document_ids:
                history.document_ids.append(did)
        await _conv_repo.save(history)
