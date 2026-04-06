# ── EHR AI Sidebar — Backend ──────────────────────────────────────────────────
#
# CPU build (default):
#   docker build -t ehr-ai-backend .
#
# GPU build (NVIDIA CUDA 12.1):
#   docker build --build-arg BASE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
#                -t ehr-ai-backend-gpu .
#
ARG BASE=python:3.11-slim
FROM ${BASE}

# Install Python 3.11 when using CUDA base (which is Ubuntu, not Python slim)
RUN if ! command -v python3 > /dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip && \
      ln -sf python3.11 /usr/bin/python3 && \
      ln -sf python3 /usr/bin/python && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # PDF rendering (PyMuPDF / pdfplumber)
    libmupdf-dev \
    poppler-utils \
    # OpenCV / EasyOCR native libs
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # Build tools (needed for some pip packages)
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

# Install PyTorch first so other packages resolve the right CUDA variant.
# For GPU builds, replace with the CUDA wheel URL:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

# ── HuggingFace cache directory (models downloaded at runtime) ────────────────
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
