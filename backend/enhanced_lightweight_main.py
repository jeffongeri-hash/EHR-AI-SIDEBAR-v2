"""
Enhanced Lightweight EHR AI with HuggingFace Integration
Optimized for 8GB macOS systems with the most compatible models
"""

import os
import asyncio
from typing import Optional, Dict, Any
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import transformers, fall back gracefully
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available, using mock responses")

# Configuration for lightweight models
COMPATIBLE_MODELS = {
    "distilgpt2": {
        "name": "DistilGPT-2",
        "size": "82M parameters", 
        "memory": "~300MB",
        "description": "Fastest, most compatible model for 8GB systems"
    },
    "gpt2": {
        "name": "GPT-2 Small", 
        "size": "124M parameters",
        "memory": "~500MB", 
        "description": "Small GPT-2 model, good balance of speed and quality"
    },
    "microsoft/DialoGPT-small": {
        "name": "DialoGPT Small",
        "size": "117M parameters",
        "memory": "~400MB",
        "description": "Conversational model, good for medical dialogue"
    }
}

class ChatRequest(BaseModel):
    message: str
    model: str = "distilgpt2"
    max_length: int = 150
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    model_used: str
    memory_usage: Optional[str] = None
    generation_time: Optional[float] = None

class ModelDownload(BaseModel):
    model_name: str

# Create app
app = FastAPI(
    title="EHR AI Sidebar - Enhanced Lightweight",
    description="Optimized for 8GB macOS with HuggingFace support",
    version="1.0.0-enhanced"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache = {}

async def load_lightweight_model(model_name: str):
    """Load a lightweight model with memory optimization"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    if model_name in model_cache:
        return model_cache[model_name]
    
    try:
        print(f"🔄 Loading {model_name}...")
        
        # Use CPU-only, minimal memory configuration
        if model_name == "distilgpt2":
            # Most compatible option
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            # For other models, use pipeline for simplicity
            generator = pipeline(
                "text-generation",
                model=model_name,
                device=-1,  # Use CPU
                torch_dtype="auto"
            )
            model_cache[model_name] = generator
            return generator
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model_cache[model_name] = {"tokenizer": tokenizer, "model": model}
        print(f"✅ {model_name} loaded successfully")
        return model_cache[model_name]
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None

@app.get("/api/health")
async def health_check():
    """Health check with system info"""
    import psutil
    
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "version": "1.0.0-enhanced", 
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "system_memory_gb": round(memory.total / (1024**3), 1),
        "available_memory_gb": round(memory.available / (1024**3), 1),
        "memory_usage_percent": memory.percent,
        "loaded_models": list(model_cache.keys())
    }

@app.get("/api/models")
async def list_compatible_models():
    """List models compatible with 8GB macOS systems"""
    return {
        "compatible_models": COMPATIBLE_MODELS,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "recommended": "distilgpt2",
        "download_note": "Use /api/models/download to download a model"
    }

@app.post("/api/models/download")
async def download_model(request: ModelDownload):
    """Download and cache a lightweight model"""
    if not TRANSFORMERS_AVAILABLE:
        raise HTTPException(400, "Transformers not available. Install with: pip install transformers torch")
    
    if request.model_name not in COMPATIBLE_MODELS:
        raise HTTPException(400, f"Model {request.model_name} not in compatible list")
    
    try:
        model = await load_lightweight_model(request.model_name)
        if model:
            return {
                "status": "success",
                "message": f"Model {request.model_name} downloaded and cached",
                "model_info": COMPATIBLE_MODELS[request.model_name]
            }
        else:
            raise HTTPException(500, f"Failed to download {request.model_name}")
    except Exception as e:
        raise HTTPException(500, f"Download error: {str(e)}")

@app.post("/api/chat/local", response_model=ChatResponse)
async def chat_with_local_model(request: ChatRequest):
    """Chat using local HuggingFace model"""
    import time
    start_time = time.time()
    
    if not TRANSFORMERS_AVAILABLE:
        # Fallback to medical mock responses
        medical_responses = [
            "Based on the symptoms described, I recommend obtaining vital signs and performing a focused physical examination.",
            "For chest pain evaluation, consider 12-lead ECG, chest X-ray, and cardiac enzymes (troponin levels).", 
            "Patient presentation suggests need for comprehensive history including onset, quality, radiation of symptoms.",
            "Recommend monitoring blood pressure, heart rate, and oxygen saturation. Consider medication review.",
            "Based on clinical findings, suggest laboratory studies including CBC, BMP, and inflammatory markers."
        ]
        import random
        response = random.choice(medical_responses)
        
        return ChatResponse(
            response=response + f" [Mock response - install transformers for AI models]",
            model_used="mock_medical",
            generation_time=time.time() - start_time
        )
    
    # Load model if not cached
    if request.model not in model_cache:
        model = await load_lightweight_model(request.model)
        if not model:
            raise HTTPException(500, f"Could not load model {request.model}")
    
    try:
        # Medical context prompt
        medical_prompt = f"""You are a medical AI assistant. Provide helpful, accurate medical guidance.

Question: {request.message}

Medical Response:"""

        if request.model == "distilgpt2":
            # Use direct model for better control
            model_data = model_cache[request.model]
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            
            inputs = tokenizer.encode(medical_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            response = response[len(medical_prompt):].strip()
            
        else:
            # Use pipeline for other models
            generator = model_cache[request.model]
            result = generator(
                medical_prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                truncation=True
            )
            response = result[0]['generated_text'][len(medical_prompt):].strip()
        
        generation_time = time.time() - start_time
        
        return ChatResponse(
            response=response,
            model_used=request.model,
            memory_usage=f"~{COMPATIBLE_MODELS.get(request.model, {}).get('memory', 'Unknown')}",
            generation_time=round(generation_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(500, f"Generation error: {str(e)}")

@app.get("/api/memory")
async def check_memory_usage():
    """Check current memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2), 
            "used_gb": round(memory.used / (1024**3), 2),
            "usage_percent": memory.percent,
            "loaded_models": len(model_cache),
            "models": list(model_cache.keys())
        }
    except ImportError:
        return {"error": "psutil not available"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Lightweight EHR AI...")
    print("📊 Optimized for 8GB macOS systems")
    print("🤖 Compatible models: DistilGPT-2, GPT-2 Small, DialoGPT")
    uvicorn.run(app, host="0.0.0.0", port=8004)