"""InferBit — BitNet-level inference for any open LLM."""

from .model import InferbitModel
from .convert import convert, convert_pretrained
from .ollama import resolve_ollama_model, list_ollama_models
from .eval import EvalGates, EvalResult, evaluate_model_gates, load_token_samples
from .calibrate import QuantProfile, CalibrateResult, search_quantization_profile
from .pq_session import Session, STREAMING, L1_ONLY, L2SKIP, SPARSE, INT8
from .pack import pack, unpack

__version__ = "0.4.1"
__all__ = [
    "InferbitModel",
    "convert",
    "convert_pretrained",
    "resolve_ollama_model",
    "list_ollama_models",
    "EvalGates",
    "EvalResult",
    "evaluate_model_gates",
    "load_token_samples",
    "QuantProfile",
    "CalibrateResult",
    "search_quantization_profile",
    "Session",
    "STREAMING", "L1_ONLY", "L2SKIP", "SPARSE", "INT8",
    "pack",
    "unpack",
]
