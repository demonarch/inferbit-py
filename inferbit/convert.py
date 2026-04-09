"""Model conversion: HuggingFace / local safetensors -> .ibf"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Optional

from ._ffi import _get_lib, ConvertConfig, ProgressCallback


def convert(
    source: str,
    output: str,
    *,
    bits: int = 4,
    sensitive_bits: int = 8,
    sparsity: float = 0.0,
    kv_bits: int = 8,
    threads: int = 0,
    progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Convert a local safetensors file to .ibf format.

    Args:
        source: Path to .safetensors file
        output: Path for output .ibf file
        bits: Default quantization bits (2, 4, 8)
        sensitive_bits: Bits for attention/embeddings (4, 8)
        sparsity: Target structured sparsity (0.0-0.6)
        kv_bits: KV cache quantization bits
        threads: CPU threads (0 = auto)
        progress: Optional callback(percent, stage)

    Returns:
        Path to the output .ibf file.
    """
    # Resolve ollama:// sources
    if source.startswith("ollama://") or (not os.path.exists(source) and ":" in source and "/" not in source):
        from .ollama import resolve_ollama_model
        resolved = resolve_ollama_model(source)
        if resolved:
            source = resolved
        elif source.startswith("ollama://"):
            raise RuntimeError(f"Ollama model not found: {source}")

    lib = _get_lib()

    cfg = lib.inferbit_default_convert_config()
    cfg.default_bits = bits
    cfg.sensitive_bits = sensitive_bits
    cfg.sparsity = sparsity
    cfg.kv_bits = kv_bits
    cfg.threads = threads

    if progress:
        @ProgressCallback
        def _cb(pct, stage, ctx):
            stage_str = stage.decode() if stage else ""
            progress(pct, stage_str)

        # Store reference to prevent GC
        cfg.progress = ctypes.cast(_cb, ctypes.c_void_p).value
        _progress_ref = _cb  # noqa: F841
    else:
        cfg.progress = None

    rc = lib.inferbit_convert(
        source.encode(), output.encode(), ctypes.byref(cfg)
    )
    if rc != 0:
        err = lib.inferbit_last_error()
        msg = err.decode() if err else "unknown error"
        raise RuntimeError(f"Conversion failed: {msg}")

    return output


def convert_pretrained(
    model_id: str,
    *,
    bits: int = 4,
    sensitive_bits: int = 8,
    cache_dir: str = "./models",
    progress: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Download a model from HuggingFace Hub and convert to .ibf.

    If the .ibf already exists in cache_dir, skip conversion.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-2-7b-hf")
        bits: Default quantization bits
        sensitive_bits: Bits for attention/embeddings
        cache_dir: Directory to store .ibf files
        progress: Optional callback(percent, stage)

    Returns:
        Path to the .ibf file.
    """
    from huggingface_hub import snapshot_download

    # Derive output filename
    safe_name = model_id.replace("/", "--")
    ibf_name = f"{safe_name}-int{bits}.ibf"
    os.makedirs(cache_dir, exist_ok=True)
    ibf_path = os.path.join(cache_dir, ibf_name)

    # Skip if already converted
    if os.path.isfile(ibf_path):
        return ibf_path

    if progress:
        progress(0.0, "downloading")

    # Download model files
    local_dir = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json"],
    )

    # Find safetensors file(s)
    st_files = list(Path(local_dir).glob("*.safetensors"))
    if not st_files:
        raise RuntimeError(f"No .safetensors files found in {local_dir}")

    # Use the single file, or model.safetensors if multiple
    if len(st_files) == 1:
        source = str(st_files[0])
    else:
        # Prefer model.safetensors (non-sharded)
        single = Path(local_dir) / "model.safetensors"
        if single.exists():
            source = str(single)
        else:
            # Use first shard — TODO: support multi-shard conversion
            source = str(sorted(st_files)[0])

    if progress:
        progress(0.1, "converting")

    convert(
        source,
        ibf_path,
        bits=bits,
        sensitive_bits=sensitive_bits,
        progress=progress,
    )

    return ibf_path
