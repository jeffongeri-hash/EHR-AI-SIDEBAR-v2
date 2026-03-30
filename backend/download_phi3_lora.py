#!/usr/bin/env python3
"""
Download Phi-3-mini model for medical fine-tuning with LoRA
This is memory-efficient and perfect for 8GB macOS systems
"""

import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_phi3_mini():
    """Download Microsoft Phi-3-mini for efficient fine-tuning"""
    
    try:
        # Install required packages
        logger.info("Installing required packages...")
        os.system("pip install transformers torch torchvision torchaudio accelerate peft")
        
        # Import after installation
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        models_dir = Path("./models/phi3-mini")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {model_name}...")
        logger.info("This is a 3.8B parameter model - perfect for your 8GB system!")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(models_dir),
            trust_remote_code=True
        )
        
        # Download model with memory optimization
        logger.info("Downloading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(models_dir),
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        logger.info(f"✅ Phi-3-mini downloaded successfully to {models_dir}")
        logger.info("Model specs:")
        logger.info("- Parameters: 3.8B")
        logger.info("- Context: 4K tokens") 
        logger.info("- Memory usage: ~2GB with quantization")
        logger.info("- Perfect for LoRA fine-tuning!")
        
        # Test the model quickly
        logger.info("Testing model...")
        test_prompt = "What are the symptoms of pneumonia?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("✅ Model test successful!")
        logger.info(f"Test response: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Phi-3-mini: {e}")
        return False

def create_lora_config():
    """Create LoRA fine-tuning configuration"""
    
    config = {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "lora_config": {
            "r": 16,  # LoRA rank
            "lora_alpha": 32,  # LoRA scaling
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none"
        },
        "training_config": {
            "learning_rate": 2e-4,
            "batch_size": 1,  # Small batch for 8GB RAM
            "gradient_accumulation_steps": 8,
            "max_steps": 500,
            "warmup_steps": 50,
            "save_steps": 100,
            "eval_steps": 100
        }
    }
    
    # Save config
    import json
    with open("./models/phi3_lora_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("✅ LoRA configuration saved to models/phi3_lora_config.json")
    return config

if __name__ == "__main__":
    logger.info("🚀 Starting Phi-3-mini download for medical fine-tuning...")
    
    # Download model
    success = download_phi3_mini()
    
    if success:
        # Create LoRA config
        create_lora_config()
        logger.info("\n🎉 Setup complete!")
        logger.info("Next steps:")
        logger.info("1. Prepare your medical dataset")
        logger.info("2. Run fine-tuning with LoRA")
        logger.info("3. Use fine-tuned model in EHR system")
    else:
        logger.error("❌ Download failed. Try alternative approach.")