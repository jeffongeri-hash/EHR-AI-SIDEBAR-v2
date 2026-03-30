"""
Fine-tuning API routes for TinyLlama medical specialization
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import json
import logging
from pathlib import Path

from app.services.tinyllama_service import tinyllama_service
from app.models.schemas import (
    FineTuneStatus,
    ModelInfo,
    ModelProvider
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/finetune", tags=["fine-tuning"])

# ── Schemas ────────────────────────────────────────────────────────────────────

class FineTuningRequest(BaseModel):
    training_data: List[Dict[str, str]]
    model_name: str = "TinyLlama-1.1B-Chat"
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1

class FineTuningStatus(BaseModel):
    job_id: str
    status: FineTuneStatus
    progress: float = 0.0
    model_name: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

class FineTuningResponse(BaseModel):
    success: bool
    job_id: str
    message: str
    status: FineTuningStatus

# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/models")
async def list_fine_tunable_models() -> Dict[str, Any]:
    """List available models for fine-tuning"""
    if not tinyllama_service.is_available():
        await tinyllama_service.initialize()
    
    models = tinyllama_service.get_model_info()
    
    return {
        "success": True,
        "models": [
            {
                "name": "TinyLlama-1.1B-Chat",
                "provider": "tinyllama",
                "size_gb": 2.2,
                "parameters": "1.1B",
                "fine_tunable": True,
                "description": "Lightweight chat model perfect for medical fine-tuning"
            }
        ],
        "current_model": models
    }

@router.post("/start")
async def start_fine_tuning(request: FineTuningRequest) -> FineTuningResponse:
    """Start fine-tuning with medical data"""
    try:
        # Validate training data
        if not request.training_data:
            raise HTTPException(status_code=400, detail="Training data is required")
        
        # Validate data format
        for i, item in enumerate(request.training_data):
            if not all(key in item for key in ["question", "answer"]) and not all(key in item for key in ["input", "output"]):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Training item {i} must contain 'question'/'answer' or 'input'/'output' fields"
                )
        
        # Initialize TinyLlama if needed
        if not tinyllama_service.is_available():
            logger.info("Initializing TinyLlama for fine-tuning...")
            if not await tinyllama_service.initialize():
                raise HTTPException(status_code=500, detail="Failed to initialize TinyLlama model")
        
        # Prepare training dataset
        dataset = await tinyllama_service.prepare_fine_tuning_data(request.training_data)
        logger.info(f"Prepared dataset with {len(dataset)} training examples")
        
        # Start fine-tuning
        job_id = f"ft-{tinyllama_service.config.model_name.replace('/', '-')}-{int(datetime.now().timestamp())}"
        logger.info(f"Starting fine-tuning job {job_id}")
        
        # Run fine-tuning (in background - simplified for demo)
        success = await tinyllama_service.start_fine_tuning(
            training_data=dataset,
            learning_rate=request.learning_rate,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size
        )
        
        if success:
            status = FineTuningStatus(
                job_id=job_id,
                status=FineTuneStatus.COMPLETED,
                progress=100.0,
                model_name=request.model_name,
                created_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat()
            )
            
            return FineTuningResponse(
                success=True,
                job_id=job_id,
                message="Fine-tuning completed successfully! Medical model is now available.",
                status=status
            )
        else:
            raise HTTPException(status_code=500, detail="Fine-tuning failed")
            
    except Exception as e:
        logger.error(f"Fine-tuning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-dataset")
async def upload_training_dataset(
    file: UploadFile = File(...),
    description: str = Form("")
) -> Dict[str, Any]:
    """Upload a training dataset file (JSON/JSONL)"""
    try:
        # Validate file type
        if not file.filename.endswith(('.json', '.jsonl')):
            raise HTTPException(status_code=400, detail="File must be .json or .jsonl format")
        
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Parse data
        training_data = []
        if file.filename.endswith('.jsonl'):
            # JSONL format - one JSON object per line
            for line in text_content.strip().split('\n'):
                if line.strip():
                    training_data.append(json.loads(line))
        else:
            # JSON format
            data = json.loads(text_content)
            if isinstance(data, list):
                training_data = data
            else:
                training_data = [data]
        
        # Validate data format
        processed_count = 0
        for item in training_data:
            if isinstance(item, dict) and (
                ("question" in item and "answer" in item) or
                ("input" in item and "output" in item) or
                ("instruction" in item and "response" in item)
            ):
                processed_count += 1
        
        if processed_count == 0:
            raise HTTPException(
                status_code=400, 
                detail="No valid training examples found. Each item should have question/answer, input/output, or instruction/response fields."
            )
        
        # Save dataset temporarily
        dataset_dir = Path("./models/datasets")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / f"uploaded_{file.filename}"
        
        with open(dataset_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        return {
            "success": True,
            "message": f"Dataset uploaded successfully",
            "filename": file.filename,
            "total_examples": len(training_data),
            "valid_examples": processed_count,
            "dataset_path": str(dataset_path),
            "description": description
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Dataset upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets")
async def list_uploaded_datasets() -> Dict[str, Any]:
    """List uploaded training datasets"""
    try:
        dataset_dir = Path("./models/datasets")
        if not dataset_dir.exists():
            return {"success": True, "datasets": []}
        
        datasets = []
        for file_path in dataset_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                datasets.append({
                    "filename": file_path.name,
                    "size_mb": round(file_path.stat().st_size / 1024 / 1024, 2),
                    "examples_count": len(data) if isinstance(data, list) else 1,
                    "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    "path": str(file_path)
                })
            except Exception as e:
                logger.warning(f"Could not read dataset {file_path}: {e}")
        
        return {
            "success": True,
            "datasets": sorted(datasets, key=lambda x: x["created_at"], reverse=True)
        }
        
    except Exception as e:
        logger.error(f"List datasets error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_fine_tuning_status() -> Dict[str, Any]:
    """Get current fine-tuning status and model information"""
    try:
        model_info = tinyllama_service.get_model_info()
        
        return {
            "success": True,
            "model_info": model_info,
            "fine_tuning_available": tinyllama_service.is_available(),
            "has_fine_tuned_model": model_info.get("has_fine_tuned", False),
            "cache_dir": model_info.get("cache_dir", ""),
            "hf_available": model_info.get("hf_available", False)
        }
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sample-data")
async def generate_sample_medical_data() -> Dict[str, Any]:
    """Generate sample medical training data for testing"""
    sample_data = [
        {
            "question": "What are the symptoms of pneumonia?",
            "answer": "Common symptoms of pneumonia include: fever, chills, cough with phlegm or pus, shortness of breath, chest pain when breathing or coughing, nausea, vomiting, diarrhea, fatigue, and confusion (especially in older adults). Seek medical attention if symptoms persist or worsen."
        },
        {
            "question": "How is diabetes diagnosed?",
            "answer": "Diabetes is diagnosed through blood tests including: Fasting plasma glucose ≥126 mg/dL, Random plasma glucose ≥200 mg/dL with symptoms, Hemoglobin A1c ≥6.5%, or 2-hour glucose ≥200 mg/dL during oral glucose tolerance test. Diagnosis should be confirmed with repeat testing."
        },
        {
            "question": "What should I do for a patient with chest pain?",
            "answer": "For chest pain assessment: 1) Check vital signs and obtain ECG immediately, 2) Assess pain characteristics (location, quality, radiation, triggers), 3) Order cardiac enzymes (troponin), 4) Consider chest X-ray, 5) Provide oxygen if SpO2 <94%, 6) Prepare for emergency intervention if acute MI suspected, 7) Monitor continuously."
        },
        {
            "question": "What are the normal vital signs for adults?",
            "answer": "Normal adult vital signs: Heart rate 60-100 bpm, Blood pressure <120/80 mmHg (normal), Respiratory rate 12-20 breaths/min, Temperature 97-99°F (36-37°C), Oxygen saturation >95% on room air. Values may vary based on age, fitness level, and medical conditions."
        },
        {
            "question": "How do you interpret a CBC with differential?",
            "answer": "CBC interpretation includes: WBC count (4.5-11.0 K/µL) for infection/inflammation, RBC count and hemoglobin for anemia, Platelet count (150-450 K/µL) for bleeding risk, MCV for anemia type classification, and differential for specific white cell abnormalities. Always correlate with clinical presentation."
        }
    ]
    
    return {
        "success": True,
        "message": "Sample medical training data generated",
        "data": sample_data,
        "count": len(sample_data),
        "description": "Sample medical Q&A data for fine-tuning demonstration"
    }

# Import datetime for timestamps
from datetime import datetime