# EHR AI Sidebar — Windows Setup Guide

This guide gets the full stack running on a Windows machine with TinyLlama support.

---

## Prerequisites

Install these first (all free):

1. **Python 3.11** — https://www.python.org/downloads/  
   ✅ Tick "Add Python to PATH" during install
2. **Node.js 20 LTS** — https://nodejs.org/
3. **Git** — https://git-scm.com/download/win
4. **Ollama** (optional, for Ollama models) — https://ollama.com/download
5. **VS Code** (recommended) — https://code.visualstudio.com/

---

## Step 1 — Clone the repo

Open **Command Prompt** or **PowerShell**:

```bat
git clone https://github.com/YOUR_USERNAME/EHR-AI-SIDEBAR-.git
cd EHR-AI-SIDEBAR-
```

---

## Step 2 — Backend setup

```bat
cd backend

:: Create virtual environment
python -m venv venv

:: Activate it
venv\Scripts\activate

:: Install all dependencies (including torch + transformers for TinyLlama)
pip install -r requirements.txt

:: Install TinyLlama / HuggingFace deps (Windows-compatible versions)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.36.0 accelerate peft datasets
```

> 💡 If you have an NVIDIA GPU, replace the torch install line with:
> ```bat
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Step 3 — Verify TinyLlama can load

```bat
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; print('✅ transformers OK')"
python -c "import torch; print('✅ torch', torch.__version__)"
python -c "from peft import LoraConfig; print('✅ peft OK')"
```

All three should print ✅. If any fail, re-run the pip install step.

---

## Step 4 — Environment keys

The `.env` file is already included in the repo with the API keys.  
It lives at `backend/.env` — **do not share this file publicly**.

If keys need refreshing:
- `ANTHROPIC_API_KEY` — get from https://console.anthropic.com
- `HF_TOKEN` — get from https://huggingface.co/settings/tokens

---

## Step 5 — Start the backend

```bat
:: From the backend folder with venv active:
python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8002
INFO:     Application startup complete.
```

Test it:
```bat
curl http://localhost:8002/api/health
```

---

## Step 6 — Frontend setup

Open a **new** terminal window:

```bat
cd frontend
npm install
npm run dev
```

Frontend will be at: **http://localhost:5173**

---

## Step 7 — Using TinyLlama in the chat

1. Open http://localhost:5173 in your browser
2. In the **Model** dropdown, select **⚡ TinyLlama (Local)**
3. On first use, TinyLlama will download the model (~650 MB) from HuggingFace  
   — this only happens once, cached in `backend/models/tinyllama/`
4. Send a message — TinyLlama will answer locally, no API key needed

> ⚠️ First response takes 30–60 seconds while the model loads into memory.  
> Subsequent responses are faster.

---

## Step 8 — Using Ollama (optional)

```bat
:: Install a model (e.g. llama3.2)
ollama pull llama3.2

:: Ollama runs automatically as a service on Windows
:: Backend connects to it at http://localhost:11434
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: transformers` | Run `pip install transformers` in the venv |
| `TinyLlama service failed to initialize` | Run the pip installs in Step 2 again |
| `Port 8002 already in use` | Run `netstat -ano \| findstr :8002` then `taskkill /PID <pid> /F` |
| `CORS error in browser` | Make sure backend is running on port 8002 |
| Claude returns errors | Check `ANTHROPIC_API_KEY` in `backend/.env` |

---

## Project Structure

```
EHR-AI-SIDEBAR-/
├── backend/
│   ├── app/
│   │   ├── api/routes/      # FastAPI routes (chat, documents, finetune)
│   │   ├── services/
│   │   │   ├── llm_service.py        # Routes to Claude / Ollama / TinyLlama
│   │   │   ├── tinyllama_service.py  # TinyLlama inference + fine-tuning
│   │   │   ├── document_service.py   # Document upload & text extraction
│   │   │   └── ocr_service.py        # PDF/image OCR
│   │   └── models/schemas.py         # Pydantic request/response models
│   ├── main.py              # FastAPI app entry point
│   ├── requirements.txt     # All Python dependencies
│   └── .env                 # API keys (Anthropic, HuggingFace)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ModelSelector.tsx     # Dropdown with ⚡ TinyLlama option
│   │   │   └── ChatInterface.tsx     # Main chat UI
│   │   └── types/index.ts            # TypeScript types (ModelProvider)
│   └── package.json
└── WINDOWS_SETUP.md         # This file
```

---

## Fine-tuning TinyLlama (Windows)

The oncology training data is at `backend/oncology_training.jsonl` (12 examples).

To fine-tune:
```bat
:: Make sure peft and datasets are installed
pip install peft datasets

:: Use the fine-tune API endpoint
curl -X POST http://localhost:8002/api/finetune/start ^
  -H "Content-Type: application/json" ^
  -d "{\"dataset_path\": \"oncology_training.jsonl\", \"epochs\": 3}"
```

Fine-tuned model saves to `backend/models/tinyllama/medical-tinyllama-finetuned/`  
and is automatically loaded on next startup.
