"""
LLM Service
===========
Provider priority (open-source first):
  1. Local HuggingFace model  (LOCAL_HF / TINYLLAMA)
  2. Ollama                   (OLLAMA)
  3. Anthropic Claude         (CLAUDE) — fallback only, used only when
                               the requested open-source provider fails.

Every ChatResponse includes:
  - provider_used  : which provider actually answered
  - claude_used    : True when Claude stepped in as fallback

The UI should display a visible notice whenever claude_used is True.
"""

from __future__ import annotations

import json
import uuid
import re
from datetime import datetime
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
from app.services.tinyllama_service import tinyllama_service

# ── Conversation repository ───────────────────────────────────────────────────
_conv_repo = ConversationRepository()

# Providers that are open-source and should be preferred
_OPEN_SOURCE_PROVIDERS = {ModelProvider.LOCAL_HF, ModelProvider.TINYLLAMA, ModelProvider.OLLAMA}

# ── Context window limits per model (tokens) ──────────────────────────────────
_CONTEXT_WINDOWS: dict[str, int] = {
    # Claude models
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # Ollama models
    "llama3.2": 128_000,
    "llama3.1": 128_000,
    "llama3": 8_192,
    "llama2": 4_096,
    "mistral": 32_768,
    "deepseek-r1": 64_000,
    "phi3": 128_000,
    "gemma2": 8_192,
    # Local HF models
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2_048,
    "tinyllama": 2_048,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 32_768,
    "deepseek-1.5b": 32_768,
    "microsoft/Phi-3-mini-4k-instruct": 4_096,
    "phi3-mini": 4_096,
    "mistralai/Mistral-7B-Instruct-v0.3": 32_768,
    "mistral-7b": 32_768,
    # Medical-specialised
    "BioMistral/BioMistral-7B": 32_768,
    "biomistral-7b": 32_768,
    "epfl-llm/meditron-7b": 4_096,
    "meditron-7b": 4_096,
    "wanglab/ClinicalCamel-13B": 4_096,
    "clinicalcamel-13b": 4_096,
    "wanglab/ClinicalCamel-70B": 4_096,
    "clinicalcamel-70b": 4_096,
}
_DEFAULT_CONTEXT_WINDOW = 8_192

# ── System prompt ─────────────────────────────────────────────────────────────
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
    provider_name: str = "unknown"

    async def chat(self, request: ChatRequest, context: str = "") -> ChatResponse:
        raise NotImplementedError

    async def stream_chat(self, request: ChatRequest, context: str = "") -> AsyncGenerator[str, None]:
        raise NotImplementedError
        yield  # make it a generator

    async def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError


# ── Local HuggingFace provider ────────────────────────────────────────────────

class LocalHFProvider(BaseLLMProvider):
    """Routes to local_model_service — supports any HF causal-LM."""
    provider_name = "local_hf"

    def __init__(self, model_id: Optional[str] = None):
        self._model_id = model_id  # None = use whatever is currently loaded

    async def chat(self, request: ChatRequest, context: str = "") -> ChatResponse:
        from app.services.local_model_service import local_model_service

        model_id = self._model_id or request.model_name or settings.DEFAULT_LOCAL_MODEL

        if not local_model_service.is_loaded():
            await local_model_service.load_model(model_id)

        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"

        prompt = f"{request.system_prompt or EHR_SYSTEM_PROMPT}\n\nUser: {user_content}\nAssistant:"
        response_text = await local_model_service.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
        )

        conv_id = request.conversation_id or str(uuid.uuid4())
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
        return ChatResponse(
            conversation_id=conv_id,
            message=msg,
            model_used=local_model_service.current_model_id or model_id,
            usage={"input_tokens": len(prompt.split()), "output_tokens": len(response_text.split())},
            provider_used=self.provider_name,
            claude_used=False,
        )

    async def stream_chat(self, request: ChatRequest, context: str = "") -> AsyncGenerator[str, None]:
        from app.services.local_model_service import local_model_service

        model_id = self._model_id or request.model_name or settings.DEFAULT_LOCAL_MODEL
        if not local_model_service.is_loaded():
            await local_model_service.load_model(model_id)

        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"

        prompt = f"{request.system_prompt or EHR_SYSTEM_PROMPT}\n\nUser: {user_content}\nAssistant:"
        async for chunk in local_model_service.stream(prompt=prompt, max_new_tokens=request.max_tokens):
            yield chunk

    async def list_models(self) -> list[ModelInfo]:
        from app.services.local_model_service import local_model_service, MODEL_CATALOGUE
        models = []
        for alias, meta in MODEL_CATALOGUE.items():
            models.append(ModelInfo(
                name=alias,
                provider=ModelProvider.LOCAL_HF,
                display_name=meta.get("display_name", alias),
                context_length=_CONTEXT_WINDOWS.get(alias, 4096),
                supports_vision=False,
                is_local=True,
                size_gb=meta.get("size_gb", 0.0),
                description=meta.get("description", ""),
                fine_tunable=True,
            ))
        # Also list any saved fine-tuned adapters
        ft_dir = __import__("pathlib").Path(settings.LOCAL_MODEL_DIR) / "fine-tuned"
        if ft_dir.exists():
            for d in ft_dir.iterdir():
                if d.is_dir():
                    models.append(ModelInfo(
                        name=f"fine-tuned/{d.name}",
                        provider=ModelProvider.LOCAL_HF,
                        display_name=f"{d.name} (fine-tuned)",
                        context_length=2048,
                        is_local=True,
                        fine_tunable=False,
                        description="Locally fine-tuned adapter",
                    ))
        return models


class TinyLlamaProvider(LocalHFProvider):
    """Backward-compat alias — delegates to LocalHFProvider with tinyllama model."""
    provider_name = "local_hf"

    def __init__(self):
        super().__init__(model_id="tinyllama")


# ── Anthropic Claude (fallback only) ──────────────────────────────────────────

class ClaudeProvider(BaseLLMProvider):
    provider_name = "claude"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def chat(self, request: ChatRequest, context: str = "", claude_used: bool = True) -> ChatResponse:
        model = request.model_name or settings.CLAUDE_MODEL
        messages = await self._build_messages(request, context)
        system = request.system_prompt or EHR_SYSTEM_PROMPT

        # Travel / flight tool-use
        last_message = messages[-1].get("content", "") if messages else ""
        tools = self._travel_tools() if self._is_travel_query(last_message) else None
        if tools:
            system += "\n\nYou are also a travel assistant with access to flight search capabilities."

        ctx_window = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
        check_context_fits(
            estimated_input_tokens=estimate_tokens(messages, system),
            model_name=model,
            context_window=ctx_window,
            max_output_tokens=request.max_tokens,
        )

        call_params = dict(
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=system,
            messages=messages,
        )
        if tools:
            call_params["tools"] = tools

        response = await self.client.messages.create(**call_params)

        # Handle tool-use
        if tools and response.content and len(response.content) > 1:
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_result = await self._execute_tool_call(block)
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": block.id, "content": tool_result}
                    ]})
                    response = await self.client.messages.create(
                        model=model, max_tokens=request.max_tokens,
                        temperature=request.temperature, system=system,
                        messages=messages, tools=tools,
                    )
                    break

        content = response.content[0].text if response.content else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return self._build_response(request, content, model, usage, claude_used=claude_used)

    async def stream_chat(self, request: ChatRequest, context: str = "") -> AsyncGenerator[str, None]:
        model = request.model_name or settings.CLAUDE_MODEL
        messages = await self._build_messages(request, context)
        system = request.system_prompt or EHR_SYSTEM_PROMPT

        ctx_window = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
        check_context_fits(
            estimated_input_tokens=estimate_tokens(messages, system),
            model_name=model,
            context_window=ctx_window,
            max_output_tokens=request.max_tokens,
        )

        async with self.client.messages.stream(
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name="claude-sonnet-4-6", provider=ModelProvider.CLAUDE,
                      display_name="Claude Sonnet 4.6 (fallback)", context_length=200_000,
                      supports_vision=True,
                      description="Fallback only — used when open-source providers fail"),
            ModelInfo(name="claude-opus-4-6", provider=ModelProvider.CLAUDE,
                      display_name="Claude Opus 4.6 (fallback)", context_length=200_000,
                      supports_vision=True,
                      description="Fallback only — used when open-source providers fail"),
            ModelInfo(name="claude-haiku-4-5-20251001", provider=ModelProvider.CLAUDE,
                      display_name="Claude Haiku 4.5 (fallback)", context_length=200_000,
                      supports_vision=True,
                      description="Fallback only — used when open-source providers fail"),
        ]

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_travel_query(message: str) -> bool:
        keywords = ["flight", "travel", "airplane", "airport", "chase rewards",
                    "chase points", "ultimate rewards", "first class", "business class",
                    "book a flight", "find flights", "points", "rewards"]
        lower = message.lower()
        return any(k in lower for k in keywords)

    @staticmethod
    def _travel_tools() -> list[dict]:
        return [{
            "name": "search_flights",
            "description": "Search for flights with pricing and reward point calculations",
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

    async def _execute_tool_call(self, tool_call) -> str:
        try:
            if tool_call.name == "search_flights":
                p = tool_call.input
                recs = await flight_service.get_flight_recommendations(
                    origin=p.get("origin", ""),
                    destination=p.get("destination", ""),
                    budget_points=p.get("chase_points", 0),
                    preferred_class=p.get("class_type", "economy"),
                )
                return self._format_flight_results(recs, p.get("chase_points", 0))
        except Exception as exc:
            logger.error(f"Tool execution failed: {exc}")
            return f"Error searching flights: {exc}"
        return "Tool not found"

    @staticmethod
    def _format_flight_results(recommendations: dict, chase_points: int) -> str:
        if not recommendations.get("recommendations"):
            return "No flights found."
        result = f"Found options for {chase_points:,} Chase points:\n\n"
        for i, rec in enumerate(recommendations["budget_analysis"].get("best_options", [])[:3], 1):
            f = rec["flight"]
            r = rec["rewards_info"]
            result += (f"{i}. {f.airline} {f.flight_number} — {f.origin}→{f.destination}\n"
                       f"   ${f.price_usd:,.2f} | {r.points_needed:,} pts | {f.duration}\n\n")
        return result

    @staticmethod
    async def _build_messages(request: ChatRequest, context: str) -> list[dict]:
        history = await _conv_repo.get(request.conversation_id) if request.conversation_id else None
        if history is None:
            history = ConversationHistory(
                conversation_id=request.conversation_id or "",
                messages=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        msgs = [{"role": m.role.value, "content": m.content} for m in history.messages[-20:]]
        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"
        msgs.append({"role": "user", "content": user_content})
        return msgs

    @staticmethod
    def _build_response(
        request: ChatRequest, content: str, model: str, usage: dict,
        claude_used: bool = True,
    ) -> ChatResponse:
        conv_id = request.conversation_id or str(uuid.uuid4())
        return ChatResponse(
            conversation_id=conv_id,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            model_used=model,
            usage=usage,
            provider_used="claude",
            claude_used=claude_used,
        )


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    provider_name = "ollama"
    BASE = settings.OLLAMA_BASE_URL

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def chat(self, request: ChatRequest, context: str = "") -> ChatResponse:
        model = request.model_name or settings.OLLAMA_DEFAULT_MODEL
        messages = await self._build_messages(request, context)

        ctx_window = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
        check_context_fits(
            estimated_input_tokens=estimate_tokens(messages),
            model_name=model,
            context_window=ctx_window,
            max_output_tokens=request.max_tokens,
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.BASE}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": request.temperature, "num_predict": request.max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        usage = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }
        conv_id = request.conversation_id or str(uuid.uuid4())
        return ChatResponse(
            conversation_id=conv_id,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=content),
            model_used=model,
            usage=usage,
            provider_used="ollama",
            claude_used=False,
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
            return [
                ModelInfo(
                    name=m.get("name", ""),
                    provider=ModelProvider.OLLAMA,
                    display_name=m.get("name", ""),
                    context_length=8192,
                    supports_vision="llava" in m.get("name", "") or "vision" in m.get("name", ""),
                    is_local=True,
                    size_gb=round(m.get("size", 0) / 1e9, 1),
                    fine_tunable=False,
                )
                for m in data.get("models", [])
            ]
        except Exception as exc:
            logger.warning(f"Could not reach Ollama: {exc}")
            return []

    @staticmethod
    async def _build_messages(request: ChatRequest, context: str) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": request.system_prompt or EHR_SYSTEM_PROMPT}]
        if request.conversation_id:
            history = await _conv_repo.get(request.conversation_id)
            if history:
                for m in history.messages[-18:]:
                    msgs.append({"role": m.role.value, "content": m.content})
        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"
        msgs.append({"role": "user", "content": user_content})
        return msgs


# ── LLM Service (router + fallback logic) ─────────────────────────────────────

class LLMService:
    """
    Routes requests to the appropriate provider.

    Fallback chain:
      open-source provider (LOCAL_HF / OLLAMA) fails
        → Claude steps in automatically
        → response.claude_used = True  (UI shows warning banner)
    """

    def __init__(self):
        self._providers: dict[ModelProvider, BaseLLMProvider] = {}

    def _get_provider(self, provider: ModelProvider) -> BaseLLMProvider:
        if provider not in self._providers:
            if provider in (ModelProvider.LOCAL_HF, ModelProvider.TINYLLAMA):
                self._providers[provider] = LocalHFProvider()
            elif provider == ModelProvider.OLLAMA:
                self._providers[provider] = OllamaProvider()
            elif provider == ModelProvider.CLAUDE:
                self._providers[provider] = ClaudeProvider()
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._providers[provider]

    def _claude(self) -> ClaudeProvider:
        if ModelProvider.CLAUDE not in self._providers:
            self._providers[ModelProvider.CLAUDE] = ClaudeProvider()
        return self._providers[ModelProvider.CLAUDE]  # type: ignore[return-value]

    async def chat(self, request: ChatRequest, document_context: str = "") -> ChatResponse:
        provider = self._get_provider(request.model_provider)

        # If the user explicitly requested Claude, use it directly (no fallback needed)
        if request.model_provider == ModelProvider.CLAUDE:
            response = await provider.chat(request, document_context)
            response.claude_used = True
            response.provider_used = "claude"
            await self._update_history(request, response)
            return response

        # Try open-source provider first
        try:
            response = await provider.chat(request, document_context)
            await self._update_history(request, response)
            return response

        except Exception as primary_exc:
            logger.warning(
                "Primary provider %s failed (%s) — falling back to Claude",
                request.model_provider.value, primary_exc,
            )

        # Fallback to Claude
        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError(
                f"Primary provider '{request.model_provider.value}' failed and no "
                "ANTHROPIC_API_KEY is set for Claude fallback."
            )

        response = await self._claude().chat(request, document_context, claude_used=True)
        response.claude_used = True
        response.provider_used = "claude"
        await self._update_history(request, response)
        return response

    async def stream_chat(
        self, request: ChatRequest, document_context: str = ""
    ) -> AsyncGenerator[str, None]:
        provider = self._get_provider(request.model_provider)
        collected: list[str] = []

        is_claude = request.model_provider == ModelProvider.CLAUDE
        used_fallback = False

        try:
            async for chunk in provider.stream_chat(request, document_context):
                collected.append(chunk)
                yield chunk
        except Exception as exc:
            if is_claude or not settings.ANTHROPIC_API_KEY:
                raise
            logger.warning("Streaming provider %s failed (%s) — falling back to Claude", request.model_provider.value, exc)
            used_fallback = True
            # Stream from Claude instead
            async for chunk in self._claude().stream_chat(request, document_context):
                collected.append(chunk)
                yield chunk

        # Persist after stream
        full_content = "".join(collected)
        conv_id = request.conversation_id or str(uuid.uuid4())
        synthetic_response = ChatResponse(
            conversation_id=conv_id,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=full_content),
            model_used=request.model_name or ("claude" if (is_claude or used_fallback) else "unknown"),
            provider_used="claude" if (is_claude or used_fallback) else request.model_provider.value,
            claude_used=is_claude or used_fallback,
        )
        await self._update_history(request, synthetic_response)

    async def list_all_models(self) -> list[ModelInfo]:
        """Returns open-source models first, Claude last."""
        local = await self._get_provider(ModelProvider.LOCAL_HF).list_models()
        ollama = await self._get_provider(ModelProvider.OLLAMA).list_models()
        claude = await self._claude().list_models()
        # Open-source first, Claude always last
        return local + ollama + claude

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
                conversation_id=conv_id,
                messages=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        history.messages.append(ChatMessage(role=MessageRole.USER, content=request.message))
        history.messages.append(response.message)
        history.updated_at = datetime.utcnow()
        if request.document_ids:
            for did in request.document_ids:
                if did not in history.document_ids:
                    history.document_ids.append(did)
        await _conv_repo.save(history)


# ── Singleton ─────────────────────────────────────────────────────────────────
llm_service = LLMService()
