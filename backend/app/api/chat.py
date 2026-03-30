"""Chat API endpoints with streaming support."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.documents import get_document_context
from app.core.audit import get_audit_logger
# from app.core.rate_limit import limiter  # temporarily disabled
from app.core.user_auth import get_current_user
from app.models.schemas import (
    AvailableModels,
    ChatRequest,
    ChatResponse,
    ConversationHistory,
)
from app.services.llm_service import LLMService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(get_current_user)],
)
_llm = LLMService()


@router.post("/", response_model=ChatResponse)
# @limiter.limit("30/minute")  # temporarily disabled
async def chat(request: ChatRequest, http_request: Request):
    """
    Send a message and receive a response.

    Attach document IDs via `document_ids` to include their extracted text as context.
    """
    # Gather document context
    doc_context = ""
    if request.document_ids:
        doc_context = await get_document_context(request.document_ids)
        get_audit_logger().log(
            event="CHAT",
            resource="document",
            resource_id=",".join(request.document_ids),
            client_ip=http_request.client.host if http_request.client else "unknown",
            user_agent=http_request.headers.get("user-agent", ""),
            detail={"model": request.model_name, "doc_count": len(request.document_ids)},
        )

    try:
        response = await _llm.chat(request, document_context=doc_context)
        if doc_context:
            response.document_context = request.document_ids
        return response
    except Exception as exc:
        logger.exception(f"Chat error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stream")
# @limiter.limit("20/minute")  # temporarily disabled
async def stream_chat(request: Request, body: ChatRequest):
    """
    Server-Sent Events streaming chat endpoint.
    Set `stream: true` in the request body.
    """
    doc_context = ""
    if body.document_ids:
        doc_context = await get_document_context(body.document_ids)

    async def event_generator():
        try:
            async for chunk in _llm.stream_chat(body, document_context=doc_context):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception(f"Stream error: {exc}")
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str):
    """Retrieve the full message history for a conversation."""
    history = await _llm.get_conversation(conversation_id)
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return history


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Delete a conversation's history."""
    deleted = await _llm.clear_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": f"Conversation {conversation_id} cleared"}


@router.get("/models", response_model=AvailableModels)
async def list_models():
    """Return all available LLM models (Claude + local Ollama models)."""
    models = await _llm.list_all_models()
    return AvailableModels(models=models)
