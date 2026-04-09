"""InferBit — BitNet-level inference for any open LLM."""

from .model import InferbitModel
from .convert import convert, convert_pretrained
from .ollama import resolve_ollama_model, list_ollama_models

__version__ = "0.1.0"
__all__ = [
    "InferbitModel",
    "convert",
    "convert_pretrained",
    "resolve_ollama_model",
    "list_ollama_models",
]
