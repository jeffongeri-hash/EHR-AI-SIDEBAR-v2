"""
Lightweight EHR AI Sidebar – Optimized for older macOS systems
Minimal dependencies, reduced memory usage
"""

import os
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Simple configuration
class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"

class ChatResponse(BaseModel):
    response: str
    model_used: str

# Create lightweight app
app = FastAPI(
    title="EHR AI Sidebar - Lightweight",
    description="Optimized for older macOS systems",
    version="1.0.0-lite"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "version": "1.0.0-lite",
        "optimized_for": "macOS 8GB RAM"
    }

@app.post("/api/chat/simple", response_model=ChatResponse)
async def simple_chat(request: ChatRequest):
    """Simple chat endpoint without heavy ML dependencies"""
    
    # Mock response for testing - replace with actual Claude API call
    mock_responses = [
        "Based on the patient symptoms, I recommend initial assessment including vital signs and ECG.",
        "For chest pain evaluation, consider cardiac enzymes and chest X-ray.",
        "Patient history suggests reviewing medication interactions and recent labs.",
        "Recommend monitoring blood pressure and scheduling follow-up in 48 hours."
    ]
    
    import random
    response = random.choice(mock_responses)
    
    return ChatResponse(
        response=response,
        model_used=request.model
    )

@app.get("/api/models")
async def list_models():
    """List available lightweight models"""
    return {
        "lightweight_models": [
            {
                "id": "distilgpt2",
                "name": "DistilGPT-2",
                "size": "82M parameters",
                "memory": "~300MB",
                "speed": "Fast"
            },
            {
                "id": "microsoft/DialoGPT-small", 
                "name": "DialoGPT Small",
                "size": "117M parameters", 
                "memory": "~500MB",
                "speed": "Fast"
            },
            {
                "id": "microsoft/DialoGPT-medium",
                "name": "DialoGPT Medium",
                "size": "345M parameters",
                "memory": "~1.5GB", 
                "speed": "Medium"
            }
        ],
        "recommended": "distilgpt2"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)