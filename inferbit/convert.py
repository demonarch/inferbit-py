"""Model conversion: HuggingFace / local safetensors -> .ibf"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Optional

from ._ffi import (
    _get_lib, ConvertConfig, ProgressCallback,
    INFERBIT_CONVERT_INT4, INFERBIT_CONVERT_PQV2_FLAT,
    INFERBIT_CONVERT_PQV2_PYRAMID,
    INFERBIT_TENSOR_CLASS_FFN_GATE, INFERBIT_TENSOR_CLASS_FFN_UP,
    INFERBIT_TENSOR_CLASS_FFN_DOWN, INFERBIT_TENSOR_CLASS_ATTN_Q,
    INFERBIT_TENSOR_CLASS_ATTN_K, INFERBIT_TENSOR_CLASS_ATTN_V,
    INFERBIT_TENSOR_CLASS_ATTN_O, INFERBIT_TENSOR_CLASS_EMBED,
    INFERBIT_TENSOR_CLASS_LM_HEAD,
    INFERBIT_RESIDENCY_AUTO, INFERBIT_RESIDENCY_RAM,
    INFERBIT_RESIDENCY_DRIVE,
)


# Valid values for the ``format`` argument of ``convert``.
#   "int4"    — INT4-blk32 + INT8 (the v0.4.1 default; uses C convert.c)
#   "pqv2"    — flat PQv2 (n_levels=1, format string ``pq2d_v1_l1``)
#   "pyramid" — PQv2 with L2 additive residual codebook (n_levels=2,
#               format string ``pq2d_v1_pyramid``). This is the target
#               format of Stage 1 in docs/v2/00_CORRECTION.md.
_VALID_FORMATS = ("int4", "pqv2", "pyramid")


# String -> inferbit_convert_format enum (see include/inferbit.h).
_FORMAT_ENUM = {
    "int4":    INFERBIT_CONVERT_INT4,
    "pqv2":    INFERBIT_CONVERT_PQV2_FLAT,
    "pyramid": INFERBIT_CONVERT_PQV2_PYRAMID,
}


# String -> inferbit_residency enum.
_RESIDENCY_ENUM = {
    "auto":  INFERBIT_RESIDENCY_AUTO,
    "ram":   INFERBIT_RESIDENCY_RAM,
    "drive": INFERBIT_RESIDENCY_DRIVE,
}


# Tensor-class groupings for the per_class_format[] / per_class_residency[]
# arrays. Each high-level CLI knob ("ffn", "attn", "embed", "lm_head") maps
# to one or more inferbit_tensor_class slots.
_CLASS_GROUPS = {
    "ffn":     (INFERBIT_TENSOR_CLASS_FFN_GATE,
                INFERBIT_TENSOR_CLASS_FFN_UP,
                INFERBIT_TENSOR_CLASS_FFN_DOWN),
    "attn":    (INFERBIT_TENSOR_CLASS_ATTN_Q,
                INFERBIT_TENSOR_CLASS_ATTN_K,
                INFERBIT_TENSOR_CLASS_ATTN_V,
                INFERBIT_TENSOR_CLASS_ATTN_O),
    "embed":   (INFERBIT_TENSOR_CLASS_EMBED,),
    "lm_head": (INFERBIT_TENSOR_CLASS_LM_HEAD,),
}


def _apply_per_class_format(cfg: ConvertConfig, group: str, value: str) -> None:
    """Set per_class_format[] slots for a high-level group.

    Empty/None value leaves slots at 0 (= use cfg.format). Unknown
    format strings raise ValueError.
    """
    if not value:
        return
    if value not in _FORMAT_ENUM:
        raise ValueError(
            f"per-class format for {group!r} must be one of "
            f"{tuple(_FORMAT_ENUM)!r}; got {value!r}"
        )
    enum_val = _FORMAT_ENUM[value]
    for slot in _CLASS_GROUPS[group]:
        cfg.per_class_format[slot] = enum_val


def _apply_per_class_residency(cfg: ConvertConfig, group: str, value: str) -> None:
    """Set per_class_residency[] slots for a high-level group.

    Empty/None or "auto" leaves slots at 0 (= AUTO, loader picks).
    """
    if not value or value == "auto":
        return
    if value not in _RESIDENCY_ENUM:
        raise ValueError(
            f"per-class residency for {group!r} must be one of "
            f"{tuple(_RESIDENCY_ENUM)!r}; got {value!r}"
        )
    enum_val = _RESIDENCY_ENUM[value]
    for slot in _CLASS_GROUPS[group]:
        cfg.per_class_residency[slot] = enum_val


def convert(
    source: str,
    output: str,
    *,
    format: str = "int4",
    bits: int = 4,
    sensitive_bits: int = 8,
    sparsity: float = 0.0,
    kv_bits: int = 16,
    threads: int = 0,
    progress: Optional[Callable[[float, str], None]] = None,
    # Stage 3a — MoME row-slice expert count for FFN tensors. 1 = off.
    mome_experts: int = 1,
    # Stage 5k — scale precision. 0 = legacy fp16/fp16, 2 = int8 row +
    # fp8 codebook scale.
    scale_precision: int = 0,
    # Stage 5j — codebook pool dedup scaffolding. False = off (legacy).
    codebook_dedup: bool = False,
    # Stage 5b — per-tensor-class format overrides. Empty string leaves
    # the slot at the global cfg.format default.
    format_ffn: str = "",
    format_attn: str = "",
    format_embed: str = "",
    format_lm_head: str = "",
    # Stage 5c — per-tensor-class residency hints. "auto" leaves the
    # slot at AUTO (loader picks).
    residency_ffn: str = "auto",
    residency_attn: str = "auto",
    residency_embed: str = "auto",
    residency_lm_head: str = "auto",
) -> str:
    """
    Convert a local safetensors file (or HuggingFace id) to .ibf format.

    Args:
        source: Path to .safetensors file / directory, or HF model id
        output: Path for output .ibf file
        format: Output quantization family:
            * ``"int4"``    — INT4-blk32 FFN + INT8 attention (v0.4.1
                              default; uses the C convert pipeline)
            * ``"pqv2"``    — flat PQv2 (n_levels=1, ``pq2d_v1_l1``)
            * ``"pyramid"`` — PQv2 pyramid (n_levels=2,
                              ``pq2d_v1_pyramid``); FFN and attention
                              encoded with K=256/half=2 codebooks plus an
                              additive L2 residual layer. Stage 1 target
                              per docs/v2/00_CORRECTION.md.
        bits: Default quantization bits (INT4 path only; ignored when
            ``format != "int4"``)
        sensitive_bits: Bits for attention/embeddings (INT4 path only)
        sparsity: Target structured sparsity (INT4 path only)
        kv_bits: KV cache quantization bits. 16 = fp32 KV (default):
            quality-safe and on the GPU fast prefill path. 8 = INT8 KV:
            smaller, but takes a slower per-position attention fallback —
            only worth it at long context where KV memory traffic dominates.
        threads: CPU threads (0 = auto)
        progress: Optional callback(percent, stage)

    Returns:
        Path to the output .ibf file.
    """
    if format not in _VALID_FORMATS:
        raise ValueError(
            f"format must be one of {_VALID_FORMATS!r}; got {format!r}"
        )

    # Resolve ollama:// sources
    if source.startswith("ollama://") or (not os.path.exists(source) and ":" in source and "/" not in source):
        from .ollama import resolve_ollama_model
        resolved = resolve_ollama_model(source)
        if resolved:
            source = resolved
        elif source.startswith("ollama://"):
            raise RuntimeError(f"Ollama model not found: {source}")

    # PQv2 / pyramid formats are produced by libinferbit's C encoder
    # (src/pqv2_encode.c). The Python wrapper only routes the format flag
    # through to the FFI entrypoint; no logic lives here. This keeps the
    # Python and Node packages in lockstep — both are thin consumers of
    # the same libinferbit API. See docs/v2/00_CORRECTION.md.
    _format_enum = _FORMAT_ENUM[format]

    if kv_bits == 8:
        import sys
        print(
            "warning: kv_bits=8 uses the slower per-position GPU attention "
            "path; kv_bits=16 is recommended for throughput",
            file=sys.stderr,
        )

    lib = _get_lib()

    cfg = lib.inferbit_default_convert_config()
    cfg.default_bits = bits
    cfg.sensitive_bits = sensitive_bits
    cfg.sparsity = sparsity
    cfg.kv_bits = kv_bits
    cfg.threads = threads
    cfg.format = _format_enum

    # Stage 3a / 5j / 5k scalar knobs. Defaults preserve v0.4.1 behavior.
    cfg.mome_experts = int(mome_experts)
    cfg.scale_precision = int(scale_precision)
    cfg.codebook_dedup = 1 if codebook_dedup else 0

    # Stage 5b — per-tensor-class format overrides. Empty string leaves
    # the slot at 0 (= use cfg.format).
    _apply_per_class_format(cfg, "ffn",     format_ffn)
    _apply_per_class_format(cfg, "attn",    format_attn)
    _apply_per_class_format(cfg, "embed",   format_embed)
    _apply_per_class_format(cfg, "lm_head", format_lm_head)

    # Stage 5c — per-tensor-class residency hints. "auto" / empty leaves
    # the slot at 0 (= AUTO).
    _apply_per_class_residency(cfg, "ffn",     residency_ffn)
    _apply_per_class_residency(cfg, "attn",    residency_attn)
    _apply_per_class_residency(cfg, "embed",   residency_embed)
    _apply_per_class_residency(cfg, "lm_head", residency_lm_head)

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
    format: str = "int4",
    bits: int = 4,
    sensitive_bits: int = 8,
    cache_dir: str = "./models",
    output: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    # Stage 3a / 5j / 5k / 5b / 5c knobs — forwarded verbatim to convert().
    mome_experts: int = 1,
    scale_precision: int = 0,
    codebook_dedup: bool = False,
    format_ffn: str = "",
    format_attn: str = "",
    format_embed: str = "",
    format_lm_head: str = "",
    residency_ffn: str = "auto",
    residency_attn: str = "auto",
    residency_embed: str = "auto",
    residency_lm_head: str = "auto",
) -> str:
    """
    Download a model from HuggingFace Hub and convert to .ibf.

    If the .ibf already exists at the resolved output path, skip
    conversion.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-2-7b-hf")
        format: Output format ("int4", "pqv2", or "pyramid"). See
            ``convert()`` for semantics.
        bits: Default quantization bits (INT4 path only)
        sensitive_bits: Bits for attention/embeddings (INT4 path only)
        cache_dir: Directory to store .ibf files when ``output`` is not
            given. Ignored when ``output`` is provided.
        output: Explicit output .ibf path. When provided, takes
            precedence over ``cache_dir`` + derived filename. The CLI
            forwards its ``-o`` argument here so the user's chosen path
            wins.
        progress: Optional callback(percent, stage)

    Returns:
        Path to the .ibf file.
    """
    from huggingface_hub import snapshot_download

    if format not in _VALID_FORMATS:
        raise ValueError(
            f"format must be one of {_VALID_FORMATS!r}; got {format!r}"
        )

    if output is not None:
        ibf_path = output
        os.makedirs(os.path.dirname(os.path.abspath(ibf_path)) or ".",
                    exist_ok=True)
    else:
        # Derive output filename. PQv2 / pyramid get distinct names so the
        # INT4 baseline file doesn't get overwritten by a PQ run.
        safe_name = model_id.replace("/", "--")
        if format == "int4":
            ibf_name = f"{safe_name}-int{bits}.ibf"
        elif format == "pqv2":
            ibf_name = f"{safe_name}-pqv2.ibf"
        else:  # pyramid
            ibf_name = f"{safe_name}-pyramid.ibf"
        os.makedirs(cache_dir, exist_ok=True)
        ibf_path = os.path.join(cache_dir, ibf_name)

    # Skip if already converted (idempotent — caller can rm to force
    # a re-convert).
    if os.path.isfile(ibf_path):
        return ibf_path

    if progress:
        progress(0.0, "downloading")

    # Download safetensors + configs locally. The C encoder reads from
    # disk; this is identical for INT4 and PQv2/pyramid paths.
    local_dir = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json", "tokenizer.model"],
    )

    # Pick the source path to hand to the C encoder.
    st_files = list(Path(local_dir).glob("*.safetensors"))
    if not st_files:
        raise RuntimeError(f"No .safetensors files found in {local_dir}")

    if format in ("pqv2", "pyramid"):
        # PQv2 encoder handles single-file and multi-shard directories
        # via libinferbit's safetensors reader. Pass the directory when
        # multi-shard, single file otherwise.
        single = Path(local_dir) / "model.safetensors"
        source = str(single) if single.exists() else local_dir
    elif len(st_files) == 1:
        source = str(st_files[0])
    else:
        # INT4 multi-shard: prefer model.safetensors, else first shard.
        single = Path(local_dir) / "model.safetensors"
        if single.exists():
            source = str(single)
        else:
            source = str(sorted(st_files)[0])

    if progress:
        progress(0.1, "converting")

    convert(
        source,
        ibf_path,
        format=format,
        bits=bits,
        sensitive_bits=sensitive_bits,
        progress=progress,
        mome_experts=mome_experts,
        scale_precision=scale_precision,
        codebook_dedup=codebook_dedup,
        format_ffn=format_ffn,
        format_attn=format_attn,
        format_embed=format_embed,
        format_lm_head=format_lm_head,
        residency_ffn=residency_ffn,
        residency_attn=residency_attn,
        residency_embed=residency_embed,
        residency_lm_head=residency_lm_head,
    )

    return ibf_path
