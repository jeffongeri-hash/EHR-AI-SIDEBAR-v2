"""
LLM Service

Supports multiple providers:
  - Anthropic Claude (claude-sonnet-4-6, claude-opus-4-6, etc.)
  - Ollama (local: llama3, deepseek-r1, mistral, phi3, etc.)
  - HuggingFace Transformers (custom / fine-tuned models)

All providers expose a unified async interface.
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

# ── Conversation repository (persistent DB store) ─────────────────────────────
_conv_repo = ConversationRepository()

# ── Context window limits per model (tokens) ─────────────────────────────────
_CONTEXT_WINDOWS: dict[str, int] = {
    # Claude models — 200 k context
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # Common Ollama models — conservative defaults
    "llama3.2": 128_000,
    "llama3.1": 128_000,
    "llama3": 8_192,
    "llama2": 4_096,
    "mistral": 32_768,
    "deepseek-r1": 64_000,
    "phi3": 128_000,
    "gemma2": 8_192,
    # TinyLlama models
    "TinyLlama-1.1B-Chat": 2_048,
    "tinyllama": 2_048,
}
_DEFAULT_CONTEXT_WINDOW = 8_192  # conservative fallback for unknown models

# ── Default system prompt for EHR context ────────────────────────────────────
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

    async def stream_chat(
        self, request: ChatRequest, context: str = ""
    ) -> AsyncGenerator[str, None]:
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
        
        # Check if this is a travel/flight-related query
        last_message = messages[-1].get("content", "") if messages else ""
        if self._is_travel_query(last_message):
            # Enhanced system prompt for travel queries
            system = f"{system}\n\nYou are also a travel assistant with access to flight search and Chase Ultimate Rewards capabilities. When users ask about flights, rewards points, or travel booking, you can search for flights and calculate reward point values."
            
            # Add function calling capabilities for flight search
            tools = [
                {
                    "name": "search_flights",
                    "description": "Search for flights between two locations with pricing and reward point calculations",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string", "description": "Origin airport code or city"},
                            "destination": {"type": "string", "description": "Destination airport code or city"},
                            "departure_date": {"type": "string", "description": "Departure date (YYYY-MM-DD format)"},
                            "class_type": {"type": "string", "enum": ["economy", "business", "first"], "description": "Travel class"},
                            "chase_points": {"type": "integer", "description": "Available Chase Ultimate Rewards points"}
                        },
                        "required": ["origin", "destination", "departure_date", "chase_points"]
                    }
                }
            ]
        else:
            tools = None

        ctx_window = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
        check_context_fits(
            estimated_input_tokens=estimate_tokens(messages, system),
            model_name=model,
            context_window=ctx_window,
            max_output_tokens=request.max_tokens,
        )

        # Make API call with or without tools
        call_params = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": system,
            "messages": messages,
        }
        
        if tools:
            call_params["tools"] = tools
            
        response = await self.client.messages.create(**call_params)

        # Handle function calling response
        if response.content and len(response.content) > 1:
            for content_block in response.content:
                if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                    # Execute the tool call
                    tool_result = await self._execute_tool_call(content_block)
                    
                    # Continue conversation with tool result
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    messages.append({
                        "role": "user", 
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result
                            }
                        ]
                    })
                    
                    # Get final response with tool results
                    final_response = await self.client.messages.create(
                        model=model,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        system=system,
                        messages=messages,
                        tools=tools
                    )
                    
                    content = final_response.content[0].text if final_response.content else ""
                    usage = {
                        "input_tokens": response.usage.input_tokens + final_response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens + final_response.usage.output_tokens,
                    }
                    
                    return self._build_response(request, content, model, usage)

        content = response.content[0].text if response.content else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return self._build_response(request, content, model, usage)
    
    def _is_travel_query(self, message: str) -> bool:
        """Detect if a message is asking about travel, flights, or rewards"""
        travel_keywords = [
            "flight", "flights", "travel", "airplane", "airport",
            "chase rewards", "chase points", "ultimate rewards",
            "first class", "business class", "economy class",
            "international flight", "domestic flight",
            "book a flight", "find flights", "flight search",
            "points", "rewards", "redemption"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in travel_keywords)
    
    async def _execute_tool_call(self, tool_call) -> str:
        """Execute a tool call and return the result"""
        try:
            if tool_call.name == "search_flights":
                params = tool_call.input
                
                # Extract parameters
                origin = params.get("origin", "")
                destination = params.get("destination", "")
                departure_date = params.get("departure_date", "")
                class_type = params.get("class_type", "economy")
                chase_points = params.get("chase_points", 0)
                
                # Search flights
                recommendations = await flight_service.get_flight_recommendations(
                    origin=origin,
                    destination=destination,
                    budget_points=chase_points,
                    preferred_class=class_type
                )
                
                # Format results for Claude
                result = self._format_flight_results(recommendations, chase_points)
                return result
                
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"I encountered an error while searching for flights: {str(e)}"
        
        return "Tool not found"
    
    def _format_flight_results(self, recommendations: dict, chase_points: int) -> str:
        """Format flight search results for Claude"""
        if not recommendations["recommendations"]:
            return "No flights found for the specified criteria."
        
        result = f"Found flight options for your {chase_points:,} Chase Ultimate Rewards points:\n\n"
        
        affordable_flights = recommendations["budget_analysis"]["best_options"]
        if affordable_flights:
            result += "✈️ **Affordable Options:**\n"
            for i, rec in enumerate(affordable_flights[:3], 1):
                flight = rec["flight"]
                rewards = rec["rewards_info"]
                
                result += f"{i}. **{flight.airline} {flight.flight_number}**\n"
                result += f"   • Route: {flight.origin} → {flight.destination}\n"
                result += f"   • Class: {flight.class_type.title()}\n"
                result += f"   • Price: ${flight.price_usd:,.2f}\n"
                result += f"   • Points needed: {rewards.points_needed:,}\n"
                result += f"   • Best redemption: {rewards.best_redemption}\n"
                result += f"   • Points remaining: {rec['points_remaining']:,}\n"
                result += f"   • Duration: {flight.duration} ({flight.stops} stops)\n\n"
        
        # Show expensive options that exceed budget
        expensive_flights = [r for r in recommendations["recommendations"] if not r["affordable"]]
        if expensive_flights:
            result += "💰 **Options exceeding your points budget:**\n"
            for i, rec in enumerate(expensive_flights[:2], 1):
                flight = rec["flight"]
                rewards = rec["rewards_info"]
                points_short = rewards.points_needed - chase_points
                
                result += f"{i}. **{flight.airline}** - ${flight.price_usd:,.2f}\n"
                result += f"   • Points needed: {rewards.points_needed:,}\n"
                result += f"   • Short by: {points_short:,} points\n\n"
        
        result += f"\n**Your Points Summary:**\n"
        result += f"• Available: {chase_points:,} Chase Ultimate Rewards points\n"
        result += f"• Cash equivalent: ${recommendations['budget_analysis']['cash_equivalent']:,.2f}\n"
        
        return result

    async def stream_chat(
        self, request: ChatRequest, context: str = ""
    ) -> AsyncGenerator[str, None]:
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
                      display_name="Claude Sonnet 4.6", context_length=200000,
                      supports_vision=True, description="Latest Claude Sonnet – fast & powerful"),
            ModelInfo(name="claude-opus-4-6", provider=ModelProvider.CLAUDE,
                      display_name="Claude Opus 4.6", context_length=200000,
                      supports_vision=True, description="Most capable Claude model"),
            ModelInfo(name="claude-haiku-4-5-20251001", provider=ModelProvider.CLAUDE,
                      display_name="Claude Haiku 4.5", context_length=200000,
                      supports_vision=True, description="Fastest Claude model"),
        ]

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _build_messages(request: ChatRequest, context: str) -> list[dict]:
        history = None
        if request.conversation_id:
            history = await _conv_repo.get(request.conversation_id)
        if history is None:
            history = ConversationHistory(
                conversation_id=request.conversation_id or "",
                messages=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

        msgs: list[dict] = []
        for msg in history.messages[-20:]:  # keep last 20 turns
            msgs.append({"role": msg.role.value, "content": msg.content})

        # Inject document context
        user_content = request.message
        if context:
            user_content = (
                f"<document_context>\n{context}\n</document_context>\n\n{request.message}"
            )

        msgs.append({"role": "user", "content": user_content})
        return msgs

    @staticmethod
    def _build_response(
        request: ChatRequest, content: str, model: str, usage: dict
    ) -> ChatResponse:
        conv_id = request.conversation_id or str(uuid.uuid4())
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=content)
        return ChatResponse(
            conversation_id=conv_id,
            message=msg,
            model_used=model,
            usage=usage,
        )


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
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
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
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
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=content)
        return ChatResponse(
            conversation_id=conv_id, message=msg, model_used=model, usage=usage
        )

    async def stream_chat(
        self, request: ChatRequest, context: str = ""
    ) -> AsyncGenerator[str, None]:
        model = request.model_name or settings.OLLAMA_DEFAULT_MODEL
        messages = await self._build_messages(request, context)

        ctx_window = _CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
        check_context_fits(
            estimated_input_tokens=estimate_tokens(messages),
            model_name=model,
            context_window=ctx_window,
            max_output_tokens=request.max_tokens,
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self.BASE}/api/chat",
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
                size_gb = m.get("size", 0) / 1e9
                models.append(ModelInfo(
                    name=name,
                    provider=ModelProvider.OLLAMA,
                    display_name=name,
                    context_length=8192,
                    supports_vision="llava" in name or "vision" in name,
                    is_local=True,
                    size_gb=round(size_gb, 1),
                ))
            return models
        except Exception as exc:
            logger.warning(f"Could not reach Ollama: {exc}")
            return []

    @staticmethod
    async def _build_messages(request: ChatRequest, context: str) -> list[dict]:
        msgs: list[dict] = [
            {"role": "system", "content": request.system_prompt or EHR_SYSTEM_PROMPT}
        ]
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


# ── TinyLlama Provider ────────────────────────────────────────────────────────

class TinyLlamaProvider(BaseLLMProvider):
    """TinyLlama provider for local medical AI with fine-tuning capabilities"""
    
    def __init__(self):
        self.service = tinyllama_service
    
    async def chat(self, request: ChatRequest, context: str = "") -> ChatResponse:
        if not self.service.is_available():
            # Initialize if not done yet
            if not await self.service.initialize():
                raise RuntimeError("TinyLlama service failed to initialize")
        
        # Build prompt with context
        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"
        
        # Generate response
        response_text = await self.service.generate_response(
            prompt=user_content,
            max_length=request.max_tokens or 512,
            temperature=request.temperature or 0.7
        )
        
        conv_id = request.conversation_id or str(uuid.uuid4())
        model_name = "TinyLlama-1.1B-Chat"
        if self.service.fine_tuned_model:
            model_name += "-Medical"
        
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
        usage = {"input_tokens": len(user_content.split()), "output_tokens": len(response_text.split())}
        
        return ChatResponse(
            conversation_id=conv_id,
            message=msg,
            model_used=model_name,
            usage=usage
        )
    
    async def stream_chat(
        self, request: ChatRequest, context: str = ""
    ) -> AsyncGenerator[str, None]:
        if not self.service.is_available():
            if not await self.service.initialize():
                raise RuntimeError("TinyLlama service failed to initialize")
        
        # Build prompt with context
        user_content = request.message
        if context:
            user_content = f"<document_context>\n{context}\n</document_context>\n\n{request.message}"
        
        # Stream response
        async for chunk in self.service.stream_response(
            prompt=user_content,
            max_length=request.max_tokens or 512,
            temperature=request.temperature or 0.7
        ):
            yield chunk
    
    async def list_models(self) -> list[ModelInfo]:
        models = []
        
        # Base TinyLlama model
        models.append(ModelInfo(
            name="TinyLlama-1.1B-Chat",
            provider=ModelProvider.TINYLLAMA,
            display_name="TinyLlama 1.1B Chat",
            context_length=2048,
            supports_vision=False,
            is_local=True,
            size_gb=2.2,
            description="Lightweight 1.1B parameter chat model, fast inference on 8GB systems"
        ))
        
        # Fine-tuned medical model if available
        if self.service.is_available() and self.service.fine_tuned_model:
            models.append(ModelInfo(
                name="TinyLlama-1.1B-Medical",
                provider=ModelProvider.TINYLLAMA,
                display_name="TinyLlama Medical (Fine-tuned)",
                context_length=2048,
                supports_vision=False,
                is_local=True,
                size_gb=2.3,
                description="TinyLlama fine-tuned on medical data for EHR tasks"
            ))
        
        return models


# ── LLM Service (router) ──────────────────────────────────────────────────────

class LLMService:
    def __init__(self):
        self._providers: dict[ModelProvider, BaseLLMProvider] = {}

    def _get_provider(self, provider: ModelProvider) -> BaseLLMProvider:
        if provider not in self._providers:
            if provider == ModelProvider.CLAUDE:
                self._providers[provider] = ClaudeProvider()
            elif provider == ModelProvider.OLLAMA:
                self._providers[provider] = OllamaProvider()
            elif provider == ModelProvider.TINYLLAMA:
                self._providers[provider] = TinyLlamaProvider()
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._providers[provider]

    async def chat(
        self,
        request: ChatRequest,
        document_context: str = "",
    ) -> ChatResponse:
        provider = self._get_provider(request.model_provider)
        response = await provider.chat(request, document_context)

        # Persist conversation
        await self._update_history(request, response)
        return response

    async def stream_chat(
        self,
        request: ChatRequest,
        document_context: str = "",
    ) -> AsyncGenerator[str, None]:
        provider = self._get_provider(request.model_provider)
        collected = []
        async for chunk in provider.stream_chat(request, document_context):
            collected.append(chunk)
            yield chunk

        # Persist after stream completes
        full_content = "".join(collected)
        conv_id = request.conversation_id or str(uuid.uuid4())
        synthetic_response = ChatResponse(
            conversation_id=conv_id,
            message=ChatMessage(role=MessageRole.ASSISTANT, content=full_content),
            model_used=request.model_name or "unknown",
        )
        await self._update_history(request, synthetic_response)

    async def list_all_models(self) -> list[ModelInfo]:
        claude = await self._get_provider(ModelProvider.CLAUDE).list_models()
        ollama = await self._get_provider(ModelProvider.OLLAMA).list_models()
        tinyllama = await self._get_provider(ModelProvider.TINYLLAMA).list_models()
        return claude + ollama + tinyllama

    async def get_conversation(self, conv_id: str) -> Optional[ConversationHistory]:
        return await _conv_repo.get(conv_id)

    async def clear_conversation(self, conv_id: str) -> bool:
        return await _conv_repo.delete(conv_id)

    # ── private ───────────────────────────────────────────────────────────────

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
