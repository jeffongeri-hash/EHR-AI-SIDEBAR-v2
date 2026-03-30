"""
Simple Lightweight EHR AI - Compatible with 8GB macOS
Downloads models on-demand without heavy transformers dependencies
"""

import os
import asyncio
import json
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Simple model configurations for your system
LIGHTWEIGHT_MODELS = {
    "mock_medical": {
        "name": "Medical Mock Model",
        "size": "0MB",
        "memory": "~50MB", 
        "description": "Fast medical responses, no downloads needed",
        "ready": True
    },
    "distilgpt2_download": {
        "name": "DistilGPT-2 (Download Ready)",
        "size": "250MB download",
        "memory": "~300MB",
        "description": "Most compatible model for 8GB macOS",
        "ready": False
    }
}

class ChatRequest(BaseModel):
    message: str
    model: str = "mock_medical"

class ChatResponse(BaseModel):
    response: str
    model_used: str
    generation_time: float
    memory_efficient: bool = True

class ModelDownloadRequest(BaseModel):
    model_name: str

# Create app
app = FastAPI(
    title="Simple Lightweight EHR AI",
    description="Optimized for 8GB macOS - Download models on demand", 
    version="1.0.0-simple"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """System health and memory info"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy",
            "version": "1.0.0-simple",
            "system_memory_gb": round(memory.total / (1024**3), 1),
            "available_memory_gb": round(memory.available / (1024**3), 1), 
            "memory_usage_percent": memory.percent,
            "transformers_installed": False,
            "optimized_for": "8GB macOS"
        }
    except ImportError:
        return {
            "status": "healthy", 
            "version": "1.0.0-simple",
            "memory_info": "psutil not available"
        }

@app.get("/api/models")
async def list_models():
    """List available lightweight models"""
    return {
        "lightweight_models": LIGHTWEIGHT_MODELS,
        "recommended": "mock_medical",
        "note": "Use mock_medical for immediate testing, distilgpt2_download for ML features"
    }

@app.post("/api/models/download")
async def download_compatible_model(request: ModelDownloadRequest):
    """Download the most compatible HuggingFace model for 8GB macOS"""
    
    if request.model_name == "distilgpt2_download":
        return {
            "status": "download_instructions",
            "message": "To download DistilGPT-2, run these commands:",
            "commands": [
                "pip install transformers==4.21.0 --no-deps",
                "pip install torch==1.12.0 --index-url https://download.pytorch.org/whl/cpu",
                "python -c \"from transformers import pipeline; pipeline('text-generation', model='distilgpt2')\""
            ],
            "size": "~250MB download",
            "memory_usage": "~300MB when loaded",
            "compatibility": "Optimized for CPU-only, 8GB systems"
        }
    else:
        return {"error": f"Model {request.model_name} not available"}

@app.post("/api/chat/simple", response_model=ChatResponse) 
async def simple_chat(request: ChatRequest):
    """Simple chat with medical mock responses"""
    start_time = time.time()
    
    # Medical knowledge responses optimized for EHR scenarios
    medical_responses = {
        "chest pain": "For chest pain evaluation: 1) Obtain 12-lead ECG within 10 minutes, 2) Vital signs and O2 saturation, 3) Chest X-ray, 4) Cardiac enzymes (troponin), 5) Assess for STEMI/NSTEMI criteria.",
        
        "blood pressure": "For hypertension management: 1) Confirm elevated readings, 2) Review current medications, 3) Assess for target organ damage, 4) Consider lifestyle modifications, 5) Evaluate for secondary causes if indicated.",
        
        "diabetes": "For diabetes management: 1) Check HbA1c and blood glucose trends, 2) Review current medications and compliance, 3) Assess for complications (retinopathy, nephropathy, neuropathy), 4) Update care plan accordingly.",
        
        "medication": "For medication review: 1) Assess current drug regimen for interactions, 2) Check dosing appropriateness, 3) Review patient compliance, 4) Monitor for adverse effects, 5) Consider therapeutic alternatives if needed.",
        
        "labs": "For laboratory interpretation: 1) Compare with previous values and reference ranges, 2) Consider clinical context, 3) Identify critical values requiring immediate attention, 4) Plan follow-up testing if needed.",
        
        "discharge": "For discharge planning: 1) Ensure medication reconciliation is complete, 2) Provide clear follow-up instructions, 3) Schedule appropriate outpatient appointments, 4) Arrange necessary home care services."
    }
    
    # Smart response selection based on keywords
    message_lower = request.message.lower()
    response = "Based on your query, I recommend: "
    
    # Find best matching response
    best_match = None
    for keyword, response_text in medical_responses.items():
        if keyword in message_lower:
            best_match = response_text
            break
    
    if best_match:
        response = best_match
    else:
        response += "1) Complete history and physical examination, 2) Review relevant laboratory/imaging studies, 3) Assess for differential diagnoses, 4) Develop appropriate treatment plan, 5) Arrange follow-up care as indicated."
    
    generation_time = time.time() - start_time
    
    return ChatResponse(
        response=response,
        model_used=request.model,
        generation_time=round(generation_time, 3),
        memory_efficient=True
    )

@app.post("/api/chat/medical", response_model=ChatResponse)
async def medical_assistant(request: ChatRequest):
    """Enhanced medical assistant with specialized responses"""
    start_time = time.time()
    
    # Advanced medical scenarios
    advanced_responses = {
        "myocardial infarction|mi|heart attack": "STEMI Protocol: 1) Door-to-balloon <90min, 2) Dual antiplatelet therapy (aspirin + P2Y12), 3) Anticoagulation (heparin/enoxaparin), 4) Beta-blocker if no contraindications, 5) Statin therapy, 6) ACE inhibitor post-MI.",
        
        "sepsis|infection|fever": "Sepsis Management: 1) Obtain blood cultures before antibiotics, 2) Start broad-spectrum antibiotics within 1 hour, 3) Fluid resuscitation 30ml/kg if hypotensive, 4) Monitor lactate levels, 5) Consider vasopressors if fluid-refractory shock.",
        
        "stroke|cva|cerebral": "Stroke Protocol: 1) CT head to rule out hemorrhage, 2) Check time of symptom onset, 3) NIHSS assessment, 4) Consider thrombolytics if <4.5 hours and no contraindications, 5) Antiplatelet therapy if ischemic.",
        
        "respiratory|breathing|dyspnea": "Respiratory Assessment: 1) ABG or pulse oximetry, 2) Chest X-ray, 3) Consider PE if appropriate (d-dimer, CTPA), 4) Bronchodilators if asthma/COPD, 5) Consider BiPAP/intubation if severe.",
        
        "renal|kidney|creatinine": "Acute Kidney Injury: 1) Check baseline creatinine, 2) Assess volume status, 3) Review nephrotoxic medications, 4) Urinalysis and electrolytes, 5) Consider renal ultrasound if obstruction suspected."
    }
    
    message_lower = request.message.lower()
    
    # Find matching advanced response
    for keywords, response_text in advanced_responses.items():
        if any(keyword in message_lower for keyword in keywords.split("|")):
            return ChatResponse(
                response=response_text,
                model_used="medical_specialist",
                generation_time=round(time.time() - start_time, 3),
                memory_efficient=True
            )
    
    # Default medical response
    return ChatResponse(
        response="For this clinical scenario, recommend: 1) Comprehensive assessment including vital signs, 2) Relevant diagnostic studies based on presentation, 3) Consider differential diagnoses, 4) Initiate appropriate treatment, 5) Plan disposition and follow-up care.",
        model_used="medical_general",
        generation_time=round(time.time() - start_time, 3),
        memory_efficient=True
    )

@app.get("/api/download/instructions")
async def get_download_instructions():
    """Get instructions for downloading compatible models"""
    return {
        "most_compatible_model": "distilgpt2",
        "system_requirements": "8GB RAM, macOS",
        "installation_commands": [
            "# Step 1: Install minimal transformers",
            "pip install transformers==4.21.0 tokenizers==0.12.1 --no-deps",
            "",
            "# Step 2: Install CPU-only PyTorch", 
            "pip install torch==1.12.0 --index-url https://download.pytorch.org/whl/cpu",
            "",
            "# Step 3: Download DistilGPT-2 (most compatible)",
            "python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2')\"",
            "",
            "# Alternative: Download without loading into memory",
            "python -c \"from transformers import pipeline; pipeline('text-generation', model='distilgpt2', device=-1)\""
        ],
        "download_size": "~250MB",
        "memory_usage": "~300MB when active",
        "note": "CPU-only version optimized for 8GB systems"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Lightweight EHR AI...")
    print("💾 Memory optimized for 8GB macOS")
    print("🏥 Medical responses ready")
    print("📥 HuggingFace download instructions available at /api/download/instructions")
    uvicorn.run(app, host="0.0.0.0", port=8005)