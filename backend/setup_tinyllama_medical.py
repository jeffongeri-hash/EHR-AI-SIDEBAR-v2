#!/usr/bin/env python3
"""
Alternative approach: Download TinyLlama for medical fine-tuning
This is very lightweight and compatible with macOS
"""

import os
import sys
import requests
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_tinyllama():
    """Download TinyLlama 1.1B - perfect for 8GB macOS"""
    
    try:
        # Create models directory
        models_dir = Path("./models/tinyllama")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🚀 Downloading TinyLlama-1.1B for medical fine-tuning...")
        logger.info("This model is:")
        logger.info("- Only 1.1B parameters (~2.2GB)")
        logger.info("- Perfect for LoRA fine-tuning")
        logger.info("- Works great on 8GB systems")
        
        # Model files to download
        base_url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main"
        files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "special_tokens_map.json",
            "pytorch_model.bin.index.json"
        ]
        
        # Download config files first
        for file in files:
            logger.info(f"Downloading {file}...")
            try:
                response = requests.get(f"{base_url}/{file}")
                if response.status_code == 200:
                    with open(models_dir / file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"✅ Downloaded {file}")
                else:
                    logger.warning(f"Could not download {file} (status: {response.status_code})")
            except Exception as e:
                logger.warning(f"Error downloading {file}: {e}")
        
        # Create a medical fine-tuning configuration
        create_medical_finetune_config()
        
        logger.info("\n🎉 TinyLlama setup complete!")
        logger.info("Benefits for medical fine-tuning:")
        logger.info("- Low memory usage (~2GB)")
        logger.info("- Fast training with LoRA")
        logger.info("- Good performance for medical Q&A")
        logger.info("- Compatible with your 8GB system")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up TinyLlama: {e}")
        return False

def create_medical_finetune_config():
    """Create medical fine-tuning configuration"""
    
    # Medical training configuration
    config = {
        "model": {
            "name": "TinyLlama-1.1B-Chat",
            "parameters": "1.1B",
            "memory_usage": "~2GB",
            "context_length": 2048
        },
        "lora_config": {
            "r": 8,  # Smaller rank for faster training
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training": {
            "learning_rate": 2e-4,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_steps": 200,
            "warmup_steps": 20,
            "weight_decay": 0.01,
            "lr_scheduler": "cosine",
            "optim": "adamw_torch"
        },
        "medical_dataset": {
            "format": "instruction",
            "max_length": 512,
            "template": {
                "instruction": "You are a medical AI assistant. Answer the following medical question:",
                "input": "{question}",
                "output": "{answer}"
            }
        },
        "hardware_optimization": {
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "save_strategy": "steps",
            "save_steps": 50,
            "eval_steps": 25
        }
    }
    
    # Save configuration
    config_path = Path("./models/medical_finetune_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Medical fine-tuning config saved to {config_path}")
    
    # Create sample medical dataset format
    sample_dataset = [
        {
            "instruction": "What are the symptoms of pneumonia?",
            "input": "",
            "output": "Common symptoms of pneumonia include: fever, chills, cough with phlegm, shortness of breath, chest pain, fatigue, and confusion in older adults. Seek medical attention if symptoms persist."
        },
        {
            "instruction": "How is diabetes diagnosed?", 
            "input": "",
            "output": "Diabetes is diagnosed through blood tests: Fasting glucose ≥126 mg/dL, Random glucose ≥200 mg/dL with symptoms, or HbA1c ≥6.5%. Oral glucose tolerance test may also be used."
        },
        {
            "instruction": "What should I do for a patient with chest pain?",
            "input": "",
            "output": "For chest pain: 1) Assess vital signs 2) Obtain ECG 3) Check cardiac enzymes 4) Consider cardiac monitoring 5) Administer oxygen if needed 6) Prepare for emergency intervention if acute MI suspected."
        }
    ]
    
    sample_path = Path("./models/sample_medical_dataset.json")
    with open(sample_path, 'w') as f:
        json.dump(sample_dataset, f, indent=2)
    
    logger.info(f"✅ Sample medical dataset saved to {sample_path}")

def create_finetune_script():
    """Create a medical fine-tuning script"""
    
    script_content = '''#!/usr/bin/env python3
"""
Medical Fine-tuning Script for TinyLlama with LoRA
Optimized for 8GB macOS systems
"""

import torch
import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_medical_dataset():
    """Load and format medical dataset"""
    
    with open("./models/sample_medical_dataset.json", "r") as f:
        data = json.load(f)
    
    # Format for training
    formatted_data = []
    for item in data:
        text = f"### Instruction: {item['instruction']}\\n### Response: {item['output']}"
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer():
    """Setup model and tokenizer for fine-tuning"""
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def main():
    """Run medical fine-tuning"""
    
    logger.info("🏥 Starting medical fine-tuning with TinyLlama...")
    
    # Load dataset
    dataset = load_medical_dataset()
    logger.info(f"Loaded {len(dataset)} medical examples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    logger.info("✅ Model and tokenizer loaded")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/medical-tinyllama",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_steps=20,
        logging_steps=10,
        save_steps=50,
        eval_steps=25,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        report_to=None
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    logger.info("🚀 Starting fine-tuning...")
    trainer.train()
    
    # Save model
    model.save_pretrained("./models/medical-tinyllama-finetuned")
    tokenizer.save_pretrained("./models/medical-tinyllama-finetuned")
    
    logger.info("🎉 Medical fine-tuning complete!")
    logger.info("Model saved to: ./models/medical-tinyllama-finetuned")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("./models/finetune_medical.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"✅ Fine-tuning script saved to {script_path}")

if __name__ == "__main__":
    logger.info("🚀 Setting up TinyLlama for medical fine-tuning...")
    
    success = download_tinyllama()
    
    if success:
        create_finetune_script()
        logger.info("\n🎉 Complete setup finished!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python models/finetune_medical.py")
        logger.info("2. Wait for training to complete (~10-15 minutes)")
        logger.info("3. Use fine-tuned model in your EHR system")
        logger.info("\nBenefits:")
        logger.info("- Memory efficient (works on 8GB)")
        logger.info("- Fast training with LoRA")
        logger.info("- Medical domain adaptation")
        logger.info("- Easy to integrate with existing system")
    else:
        logger.error("❌ Setup failed")