"""
Claude-only EHR Assistant - Zero local ML dependencies
Perfect for 8GB macOS systems
"""

import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Load API key
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="EHR Claude Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str
    document_context: Optional[str] = None

@app.get("/api/health")
async def health():
    return {"status": "healthy", "mode": "claude-only"}

@app.post("/api/chat")
async def chat_with_claude(request: ChatRequest):
    """Chat using only Claude API - no local models"""
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return {"error": "No API key configured"}
    
    # Construct prompt with medical context
    prompt = f"""You are an EHR AI assistant. Help with medical queries.

User Question: {request.message}

{f'Document Context: {request.document_context}' if request.document_context else ''}

Provide a helpful, accurate medical response:"""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data['content'][0]['text'],
                    "model": "claude-3-sonnet",
                    "usage": data.get('usage', {})
                }
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)