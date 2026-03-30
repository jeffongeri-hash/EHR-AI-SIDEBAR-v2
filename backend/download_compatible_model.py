#!/usr/bin/env python3
"""
Download the most compatible HuggingFace model for 8GB macOS
This script downloads DistilGPT-2 using direct HTTP requests
"""

import os
import requests
from pathlib import Path
import json

def download_file(url, filename):
    """Download file with progress"""
    print(f"📥 Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\r  Progress: {progress:.1f}% ({downloaded/1024/1024:.1f}MB)", end="")
    print()

def download_distilgpt2():
    """Download DistilGPT-2 model files directly"""
    
    model_dir = Path("./models/distilgpt2")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("🤖 Downloading DistilGPT-2 - Most compatible model for 8GB macOS")
    print("📊 Model size: 250MB")
    print("💾 Memory usage: ~300MB when loaded")
    
    # Base URL for DistilGPT-2
    base_url = "https://huggingface.co/distilgpt2/resolve/main"
    
    # Files to download
    files_to_download = {
        "config.json": f"{base_url}/config.json",
        "pytorch_model.bin": f"{base_url}/pytorch_model.bin", 
        "tokenizer.json": f"{base_url}/tokenizer.json",
        "tokenizer_config.json": f"{base_url}/tokenizer_config.json",
        "vocab.json": f"{base_url}/vocab.json",
        "merges.txt": f"{base_url}/merges.txt"
    }
    
    try:
        for filename, url in files_to_download.items():
            filepath = model_dir / filename
            if not filepath.exists():
                download_file(url, filepath)
            else:
                print(f"✅ {filename} already exists")
        
        # Create a simple usage script
        usage_script = f'''
# DistilGPT-2 Usage Example (Optimized for 8GB macOS)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model from local directory
model_path = "{model_dir.absolute()}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

def generate_medical_response(prompt, max_length=100):
    """Generate medical response with memory optimization"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Example usage
medical_prompt = "For a patient with chest pain, the first assessment should include"
response = generate_medical_response(medical_prompt)
print(f"Prompt: {{medical_prompt}}")
print(f"Response: {{response}}")
'''
        
        with open(model_dir / "usage_example.py", "w") as f:
            f.write(usage_script)
        
        print("\n🎉 DistilGPT-2 downloaded successfully!")
        print(f"📁 Model location: {model_dir.absolute()}")
        print(f"📝 Usage example: {model_dir / 'usage_example.py'}")
        print("\n💡 To use with minimal dependencies:")
        print("   pip install transformers==4.21.0 torch==1.12.0 --no-deps")
        print("   python models/distilgpt2/usage_example.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    download_distilgpt2()