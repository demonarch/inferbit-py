"""Quantization profile search with INT2-first fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .convert import convert
from .eval import EvalGates, EvalResult, evaluate_model_gates, load_token_samples
from .model import InferbitModel


@dataclass
class QuantProfile:
    name: str
    bits: int
    sensitive_bits: int = 8
    sparsity: float = 0.0
    kv_bits: int = 8


@dataclass
class CalibrateResult:
    selected: QuantProfile
    model_path: str
    eval_result: EvalResult
    tried: List[tuple[QuantProfile, EvalResult]]


DEFAULT_PROFILES: List[QuantProfile] = [
    QuantProfile(name="int2_aggressive", bits=2, sensitive_bits=8, sparsity=0.0, kv_bits=8),
    QuantProfile(name="int4_balanced", bits=4, sensitive_bits=8, sparsity=0.0, kv_bits=8),
    QuantProfile(name="int8_safe", bits=8, sensitive_bits=8, sparsity=0.0, kv_bits=8),
]


def search_quantization_profile(
    source: str,
    *,
    output_dir: str,
    token_dataset: Optional[str] = None,
    prompt_tokens: Optional[List[int]] = None,
    profiles: Optional[Iterable[QuantProfile]] = None,
    gates: Optional[EvalGates] = None,
    threads: int = 0,
    output_tokens: int = 128,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    progress: Optional[Callable[[str], None]] = None,
) -> CalibrateResult:
    """
    Try quantization profiles in order (INT2 -> INT4 -> INT8) and pick first that passes gates.

    This is an orchestration-level fallback search. It does not yet do per-layer mixed bits.
    """
    gates = gates or EvalGates()
    profiles = list(profiles or DEFAULT_PROFILES)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    token_samples = load_token_samples(token_dataset) if token_dataset else None
    tried: List[tuple[QuantProfile, EvalResult]] = []

    for profile in profiles:
        out_path = str(Path(output_dir) / f"{profile.name}.ibf")
        if progress:
            progress(f"converting {profile.name}")

        convert(
            source,
            out_path,
            bits=profile.bits,
            sensitive_bits=profile.sensitive_bits,
            sparsity=profile.sparsity,
            kv_bits=profile.kv_bits,
            threads=threads,
        )

        if progress:
            progress(f"evaluating {profile.name}")

        model = InferbitModel.load(out_path, threads=threads)
        result = evaluate_model_gates(
            model,
            token_samples=token_samples,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            gates=gates,
        )
        tried.append((profile, result))

        if result.passes:
            return CalibrateResult(
                selected=profile,
                model_path=out_path,
                eval_result=result,
                tried=tried,
            )

    # If none pass, return the best-effort fallback (last profile)
    fallback_profile, fallback_result = tried[-1]
    return CalibrateResult(
        selected=fallback_profile,
        model_path=str(Path(output_dir) / f"{fallback_profile.name}.ibf"),
        eval_result=fallback_result,
        tried=tried,
    )
