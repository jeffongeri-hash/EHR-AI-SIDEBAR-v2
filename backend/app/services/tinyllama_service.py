"""
TinyLlama service for EHR AI Sidebar
Provides local AI inference and fine-tuning capabilities
"""

import os
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from pathlib import Path
import logging
from dataclasses import dataclass

# Core inference dependencies — required for chat
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    import torch
    HF_AVAILABLE = True
except ImportError as e:
    HF_AVAILABLE = False
    logging.warning(f"Hugging Face transformers/torch not available for TinyLlama: {e}")

# Fine-tuning dependencies — optional, only needed for LoRA training
try:
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except (ImportError, TypeError) as e:
    PEFT_AVAILABLE = False
    logging.warning(f"peft not available — fine-tuning disabled: {e}")

# datasets — optional, only needed for fine-tuning data prep
try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    class Dataset:  # type: ignore[no-redef]
        @classmethod
        def from_list(cls, data):
            return None

logger = logging.getLogger(__name__)

@dataclass
class TinyLlamaConfig:
    """Configuration for TinyLlama model"""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 512
    temperature: float = 0.7
    device: str = "auto"
    cache_dir: str = "./models/tinyllama"
    
class TinyLlamaService:
    """TinyLlama service for medical AI inference and fine-tuning"""
    
    def __init__(self, config: Optional[TinyLlamaConfig] = None):
        self.config = config or TinyLlamaConfig()
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.fine_tuned_model: Optional[Any] = None
        self.is_initialized = False
        
        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
    async def initialize(self) -> bool:
        """Initialize the TinyLlama model"""
        if not HF_AVAILABLE:
            logger.error("Hugging Face transformers not available")
            return False
            
        try:
            logger.info("Initializing TinyLlama model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with memory optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16,
                device_map=self.config.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Check for existing fine-tuned model
            fine_tuned_path = Path(self.config.cache_dir) / "medical-tinyllama-finetuned"
            if fine_tuned_path.exists() and PEFT_AVAILABLE:
                logger.info("Loading fine-tuned medical model...")
                try:
                    self.fine_tuned_model = PeftModel.from_pretrained(
                        self.model,
                        str(fine_tuned_path)
                    )
                    logger.info("✅ Fine-tuned medical model loaded")
                except Exception as e:
                    logger.warning(f"Could not load fine-tuned model: {e}")
            
            self.is_initialized = True
            logger.info("✅ TinyLlama initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TinyLlama: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if TinyLlama is available for use"""
        return HF_AVAILABLE and self.is_initialized
    
    async def generate_response(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        use_fine_tuned: bool = True
    ) -> str:
        """Generate a response using TinyLlama"""
        if not self.is_available():
            raise RuntimeError("TinyLlama not initialized or available")
        
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        
        # Choose model (fine-tuned if available)
        model_to_use = self.fine_tuned_model if (use_fine_tuned and self.fine_tuned_model) else self.model
        
        # Format prompt for medical context
        formatted_prompt = self._format_medical_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        if hasattr(model_to_use, 'device'):
            inputs = {k: v.to(model_to_use.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    async def stream_response(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response generation (simplified for demo)"""
        response = await self.generate_response(prompt, max_length, temperature)
        
        # Simulate streaming by yielding chunks
        words = response.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield f" {word}"
            
            # Small delay to simulate streaming
            import asyncio
            await asyncio.sleep(0.05)
    
    def _format_medical_prompt(self, prompt: str) -> str:
        """Format prompt for medical context"""
        return f"""### Medical AI Assistant

You are a helpful medical AI assistant. Provide accurate, professional medical information while always recommending consultation with healthcare professionals for specific medical advice.

### Question: {prompt}

### Response:"""
    
    async def prepare_fine_tuning_data(self, conversations: List[Dict[str, str]]) -> Dataset:
        """Prepare medical conversation data for fine-tuning"""
        training_data = []
        
        for conv in conversations:
            # Format for instruction following
            instruction = "You are a medical AI assistant. Answer the following medical question:"
            input_text = conv.get("question", conv.get("input", ""))
            output_text = conv.get("answer", conv.get("output", ""))
            
            if input_text and output_text:
                formatted_text = f"### Instruction: {instruction}\n### Input: {input_text}\n### Response: {output_text}"
                training_data.append({"text": formatted_text})
        
        return Dataset.from_list(training_data)
    
    async def start_fine_tuning(
        self, 
        training_data,
        output_dir: str = "./models/medical-tinyllama-finetuned",
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 1
    ) -> bool:
        """Start fine-tuning process with LoRA"""
        if not self.is_available():
            raise RuntimeError("TinyLlama not initialized")

        if not PEFT_AVAILABLE:
            raise RuntimeError("peft library not available — fine-tuning disabled. Install with: pip install peft")

        try:
            logger.info("Starting LoRA fine-tuning...")
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=8,  # Low rank for memory efficiency
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # Get PEFT model
            peft_model = get_peft_model(self.model, lora_config)
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length
                )
            
            tokenized_dataset = training_data.map(tokenize_function, batched=True)
            
            # Training arguments optimized for 8GB macOS
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=8,  # Increase for small batch
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                warmup_steps=50,
                logging_steps=10,
                save_steps=100,
                fp16=True,  # Memory optimization
                gradient_checkpointing=True,
                dataloader_num_workers=0,  # Avoid multiprocessing issues on macOS
                report_to=None,  # Disable wandb
                push_to_hub=False,
                remove_unused_columns=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Trainer
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Start training
            logger.info("🚀 Starting fine-tuning...")
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Load the fine-tuned model
            self.fine_tuned_model = PeftModel.from_pretrained(
                self.model,
                output_dir
            )
            
            logger.info("🎉 Fine-tuning completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "model_name": self.config.model_name,
            "initialized": self.is_initialized,
            "hf_available": HF_AVAILABLE,
            "peft_available": PEFT_AVAILABLE,
            "has_fine_tuned": self.fine_tuned_model is not None,
            "cache_dir": self.config.cache_dir
        }
        
        if self.is_initialized and self.model:
            try:
                info["parameters"] = sum(p.numel() for p in self.model.parameters())
                info["device"] = str(next(self.model.parameters()).device)
            except Exception:
                pass
        
        return info

# Global service instance
tinyllama_service = TinyLlamaService()