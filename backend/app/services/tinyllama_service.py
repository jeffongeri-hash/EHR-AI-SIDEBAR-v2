"""
tinyllama_service.py  —  backward-compatibility shim
=====================================================
All new code should import from local_model_service instead.
This module keeps existing imports working without changes.
"""
from app.services.local_model_service import (
    LocalModelService,
    LocalModelConfig,
    local_model_service,
    MODEL_CATALOGUE,
)
from typing import Optional, Dict, Any, List, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class TinyLlamaService:
    """Thin wrapper kept for backward-compat. Delegates to local_model_service."""

    def __init__(self):
        self._svc = local_model_service
        # Keep expected attributes
        self.fine_tuned_model = None   # set when adapter loaded

    class _FakeConfig:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        cache_dir = "./models/tinyllama"

    config = _FakeConfig()

    def is_available(self) -> bool:
        return self._svc.is_available()

    async def initialize(self) -> bool:
        result = await self._svc.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if result:
            self.fine_tuned_model = self._svc._peft_model
        return result

    async def generate_response(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        use_fine_tuned: bool = True,
    ) -> str:
        return await self._svc.generate(
            prompt, max_new_tokens=max_length, temperature=temperature,
            use_fine_tuned=use_fine_tuned
        )

    async def stream_response(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        async for chunk in self._svc.stream_generate(prompt, max_length, temperature):
            yield chunk

    async def prepare_fine_tuning_data(self, conversations: List[Dict[str, str]]):
        """Convert conversation list to dataset format."""
        try:
            from datasets import Dataset
            texts = [self._svc._format_training_example(c) for c in conversations]
            return Dataset.from_list([{"text": t} for t in texts])
        except ImportError:
            raise RuntimeError("pip install datasets")

    async def start_fine_tuning(
        self,
        training_data,
        output_dir: str = "./models/medical-tinyllama-finetuned",
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 1,
    ) -> bool:
        """Start fine-tuning. training_data may be a Dataset or list of dicts."""
        try:
            if hasattr(training_data, "to_list"):
                examples = training_data.to_list()
            elif hasattr(training_data, "__iter__"):
                examples = list(training_data)
            else:
                examples = training_data

            await self._svc.fine_tune(
                examples=examples,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            self.fine_tuned_model = self._svc._peft_model
            return True
        except Exception as exc:
            logger.error(f"Fine-tuning failed: {exc}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        info = self._svc.get_info()
        info["model_name"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        info["initialized"] = self._svc.is_available()
        info["has_fine_tuned"] = self._svc._peft_model is not None
        return info


# Global singleton kept for backward-compat
tinyllama_service = TinyLlamaService()
