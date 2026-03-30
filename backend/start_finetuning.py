#!/usr/bin/env python3
"""
Start fine-tuning with the uploaded dataset
"""

import json
import requests
from pathlib import Path

def load_training_data():
    """Load and format training data"""
    dataset_path = "oncology_training.jsonl"
    
    training_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                example = json.loads(line.strip())
                training_data.append(example)
    
    return training_data

def start_fine_tuning():
    """Start the fine-tuning process"""
    print("🏥 Loading oncology training data...")
    training_data = load_training_data()
    print(f"📊 Loaded {len(training_data)} training examples")
    
    # Prepare the fine-tuning request
    request_data = {
        "training_data": training_data,
        "model_name": "TinyLlama-1.1B-Chat", 
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 1  # Small batch size for 8GB memory
    }
    
    print("🚀 Starting fine-tuning...")
    print(f"📋 Model: {request_data['model_name']}")
    print(f"📚 Training examples: {len(training_data)}")
    print(f"🔧 Learning rate: {request_data['learning_rate']}")
    print(f"🔄 Epochs: {request_data['num_epochs']}")
    print(f"📦 Batch size: {request_data['batch_size']}")
    
    # Send request to fine-tuning API
    try:
        response = requests.post(
            "http://localhost:8002/api/finetune/start",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Fine-tuning started successfully!")
            print(f"🆔 Job ID: {result.get('job_id', 'N/A')}")
            print(f"📝 Status: {result.get('message', 'N/A')}")
            return result
        else:
            print(f"❌ Error starting fine-tuning: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

if __name__ == "__main__":
    result = start_fine_tuning()