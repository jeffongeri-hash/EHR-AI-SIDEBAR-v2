#!/usr/bin/env python3
"""
Download DeepSeek model for medical fine-tuning
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

def download_deepseek_model():
    """Download DeepSeek-Coder model optimized for medical use cases"""
    
    # Create models directory
    models_dir = Path("./models/deepseek")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Model name - using DeepSeek-Coder-V2-Lite which is optimized for 8GB systems
    model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    
    print(f"🔽 Downloading {model_name}...")
    print("This model is optimized for code and structured data analysis")
    print("Perfect for medical record processing and clinical decision support")
    print("Size: ~1.3B parameters - ideal for 8GB macOS systems")
    
    try:
        # Download tokenizer
        print("📝 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(models_dir)
        )
        
        # Download model with optimizations for macOS
        print("🤖 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",
            cache_dir=str(models_dir)
        )
        
        # Save locally
        model_path = models_dir / "deepseek-coder-v2-lite"
        print(f"💾 Saving model to {model_path}")
        
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        # Test the model
        print("🧪 Testing model...")
        inputs = tokenizer.encode("Analyze patient symptoms:", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Test successful: {response}")
        
        print(f"""
✅ DeepSeek model downloaded successfully!

📊 Model Details:
- Name: DeepSeek-Coder-V2-Lite-Instruct  
- Size: ~1.3B parameters (~2.6GB disk space)
- Optimized for: Code analysis, structured data, medical records
- Memory usage: ~3-4GB during inference
- Perfect for: EHR analysis, clinical coding, medical NLP

📁 Location: {model_path}
🎯 Use case: Medical fine-tuning and clinical decision support
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("💡 Trying alternative approach...")
        
        # Fallback to smaller model  
        try:
            alt_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            print(f"🔄 Trying {alt_model}...")
            
            tokenizer = AutoTokenizer.from_pretrained(alt_model, cache_dir=str(models_dir))
            model = AutoModelForCausalLM.from_pretrained(
                alt_model, 
                torch_dtype=torch.float16,
                cache_dir=str(models_dir)
            )
            
            model_path = models_dir / "deepseek-r1-1.5b"
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            
            print(f"✅ Alternative model downloaded: {model_path}")
            return True
            
        except Exception as e2:
            print(f"❌ Fallback failed: {e2}")
            return False

if __name__ == "__main__":
    success = download_deepseek_model()
    if success:
        print("🎉 Ready for medical fine-tuning!")
    else:
        print("❌ Download failed - check internet connection")