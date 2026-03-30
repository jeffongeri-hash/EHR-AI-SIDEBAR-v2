# EHR AI Sidebar

An AI-powered sidebar for Electronic Health Records (EHR) featuring:
- **Multi-model chat** (Claude, Llama, DeepSeek, Mistral via Ollama)
- **Computer vision document processing** – OCR + layout extraction for PDFs and images, including low-quality scans
- **LLM fine-tuning** – LoRA/QLoRA fine-tuning of Llama/DeepSeek on domain-specific EHR datasets

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                  │
│  ┌──────────┐  ┌───────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │  Chat UI │  │  Doc Upload   │  │  Fine-Tune  │  │Settings │ │
│  └──────────┘  └───────────────┘  └─────────────┘  └─────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST + SSE
┌──────────────────────────▼──────────────────────────────────────┐
│                     Backend (FastAPI)                           │
│  ┌───────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ /api/chat │  │ /api/documents  │  │ /api/fine-tune       │  │
│  └───────────┘  └─────────────────┘  └──────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ LLMService  |  DocumentProcessor  |  FineTuneService        ││
│  │ OCRService  |  LayoutService                                ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌──────────┐     ┌──────────────┐    ┌───────────┐
  │  Claude  │     │    Ollama    │    │  PyMuPDF  │
  │  API     │     │ (Llama/DS)  │    │  EasyOCR  │
  └──────────┘     └──────────────┘    │  Tesseract│
                                        └───────────┘
```

---

## Quick Start

### 1. Clone & configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Docker (recommended)

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/api/docs

Pull a local model for offline use:
```bash
docker exec -it ehr-ai-ollama ollama pull llama3.2
docker exec -it ehr-ai-ollama ollama pull deepseek-r1:1.5b
```

### 3. Manual setup

**Backend**
```bash
sudo apt-get install tesseract-ocr libgl1   # Ubuntu/Debian
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

---

## Document Processing Pipeline

### Supported formats
| Format | Native text | OCR | Tables | Layout |
|--------|------------|-----|--------|--------|
| PDF (native) | Yes | - | Yes | Yes |
| PDF (scanned) | - | Yes | - | Yes |
| PNG / JPEG / TIFF | - | Yes | - | Yes |

### Low-quality scan enhancement (automatic)
1. **Denoising** – non-local means filtering
2. **Deskewing** – auto-detect and correct rotation
3. **Shadow removal** – background normalisation
4. **CLAHE** – contrast-limited adaptive histogram equalisation
5. **Binarisation** – adaptive thresholding (for very poor quality)
6. **2x upscaling** – bicubic super-resolution for tiny text

### OCR engines
| Engine | Best for |
|--------|----------|
| EasyOCR (auto default for scans) | Degraded images, handwriting |
| Tesseract | Clean printed documents |
| Auto | Selects based on image quality score |

---

## Fine-Tuning

### Via the UI
1. Open the **Fine-Tune** tab
2. Set base model (e.g. `meta-llama/Llama-3.2-3B-Instruct`)
3. Add training examples or set a HuggingFace dataset name
4. Click **Start Training**

### Via CLI
```bash
# YAML config
python fine_tuning/train.py --config fine_tuning/configs/llama3_ehr.yaml

# Inline args
python fine_tuning/train.py \
  --base_model meta-llama/Llama-3.2-3B-Instruct \
  --dataset medalpaca/medical_meadow_medqa \
  --output_name ehr-llama-3b

# Prepare a dataset from CSV
python fine_tuning/prepare_dataset.py \
  --input data/ehr_notes.csv \
  --output datasets/ehr_train.jsonl \
  --format csv
```

### Recommended configs
| Model | VRAM | Config |
|-------|------|--------|
| Llama 3.2-3B QLoRA | 8 GB | `configs/llama3_ehr.yaml` |
| DeepSeek-R1-1.5B QLoRA | 6 GB | `configs/deepseek_ehr.yaml` |
| Llama 3.1-8B QLoRA | 16 GB | increase r/batch |

---

## API Reference

Full OpenAPI docs: http://localhost:8000/api/docs

```http
POST /api/chat/                    # Chat (with optional document context)
POST /api/chat/stream              # Streaming SSE chat
POST /api/documents/upload         # Upload + OCR + layout extraction
GET  /api/documents/{id}/text      # Extracted text
GET  /api/documents/{id}/tables    # Extracted tables
POST /api/fine-tune/jobs           # Start a training job
GET  /api/fine-tune/jobs/{id}      # Job status + live logs
GET  /api/fine-tune/templates/ehr  # Sample EHR training data
```

---

## Project Structure

```
EHR-AI-SIDEBAR-/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routers (chat, documents, fine-tuning)
│   │   ├── services/       # Business logic (OCR, layout, LLM, training)
│   │   ├── models/         # Pydantic schemas
│   │   └── config.py
│   └── main.py
├── frontend/
│   └── src/
│       ├── components/     # React UI components
│       ├── contexts/       # Global state
│       ├── hooks/          # useChat streaming hook
│       └── services/       # API client
├── fine_tuning/
│   ├── train.py            # Standalone CLI training script
│   ├── prepare_dataset.py  # Dataset converter (CSV/JSONL/text/HF)
│   └── configs/            # YAML training configs
├── docker-compose.yml
└── .env.example
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OLLAMA_BASE_URL` | Ollama server URL (default: http://localhost:11434) |
| `HF_TOKEN` | HuggingFace token (required for gated models like Llama 3) |
| `UPLOAD_DIR` | Document storage path |
| `MODELS_DIR` | Fine-tuned model storage |

---

## Tech Stack

**Frontend**: React 18, TypeScript, Tailwind CSS, Vite, Zustand
**Backend**: FastAPI, Pydantic v2, uvicorn
**OCR**: EasyOCR, Tesseract, OpenCV, Pillow
**PDF**: PyMuPDF (fitz), PDFPlumber
**LLM**: Anthropic SDK, Ollama, HuggingFace Transformers
**Fine-tuning**: PEFT, TRL (SFTTrainer), BitsAndBytes (QLoRA)
**Infrastructure**: Docker, Redis