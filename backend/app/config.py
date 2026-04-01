from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "EHR AI Sidebar"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./ehr_ai.db"

    # Anthropic Claude
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-sonnet-4-6"
    CLAUDE_MAX_TOKENS: int = 4096

    # Ollama (local LLM)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "llama3.2"

    # OCR Settings
    TESSERACT_PATH: Optional[str] = None  # e.g. /usr/bin/tesseract
    OCR_LANGUAGES: List[str] = ["en"]
    OCR_CONFIDENCE_THRESHOLD: float = 0.6

    # Document Processing
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    SUPPORTED_FORMATS: List[str] = [".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]

    # Fine-tuning
    MODELS_DIR: str = "./models"
    DATASETS_DIR: str = "./datasets"
    HF_TOKEN: Optional[str] = None
    DEFAULT_BASE_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Local HuggingFace models (for multi-LLM and fine-tuning)
    LOCAL_MODEL_DIR: str = "./models/local"
    DEFAULT_LOCAL_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Redis (for task queue)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security — HIPAA compliance
    # MUST be overridden in production via environment variable
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # API key protecting all PHI endpoints (set in .env)
    EHR_API_KEY: Optional[str] = None

    # AES-256-GCM encryption of uploaded documents at rest
    # Generate: python -c "import secrets,base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
    EHR_ENCRYPTION_KEY: Optional[str] = None
    ENCRYPT_UPLOADS: bool = True   # Encryption is ON by default (HIPAA requirement)

    # HIPAA audit log path (append-only JSONL)
    AUDIT_LOG_PATH: str = "./audit.jsonl"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.DATASETS_DIR, exist_ok=True)
os.makedirs(settings.LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs(f"{settings.LOCAL_MODEL_DIR}/fine-tuned", exist_ok=True)
