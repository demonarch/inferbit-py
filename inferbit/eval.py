"""Calibration/evaluation harness with explicit quality/performance gates."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .model import InferbitModel


@dataclass
class EvalGates:
    max_perplexity: Optional[float] = None
    min_tokens_per_sec: Optional[float] = None
    max_memory_mb: Optional[float] = None


@dataclass
class EvalResult:
    perplexity: Optional[float]
    tokens_per_sec: float
    latency_ms_per_token: float
    memory_mb: float
    passes: bool
    failed_gates: List[str]


def load_token_samples(path: str) -> List[List[int]]:
    """
    Load tokenized samples from JSONL.

    Supported per-line formats:
      {"tokens": [1, 2, 3, ...]}
      [1, 2, 3, ...]
    """
    samples: List[List[int]] = []
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                tokens = obj
            elif isinstance(obj, dict) and "tokens" in obj:
                tokens = obj["tokens"]
            else:
                raise ValueError(f"Line {ln}: expected list or object with 'tokens'")

            if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
                raise ValueError(f"Line {ln}: 'tokens' must be a list of integers")
            if len(tokens) >= 2:
                samples.append(tokens)

    if not samples:
        raise ValueError("No valid token samples (need at least 2 tokens each)")
    return samples


def _logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def perplexity_from_token_samples(model: InferbitModel, samples: List[List[int]]) -> float:
    """Compute token-level perplexity using teacher forcing over token IDs."""
    nll = 0.0
    count = 0

    for tokens in samples:
        model.kv_clear()

        logits = model.forward([tokens[0]])
        for tok in tokens[1:]:
            lse = _logsumexp(logits)
            nll += lse - logits[tok]
            count += 1
            logits = model.forward([tok])

    if count == 0:
        raise ValueError("No next-token targets available for perplexity")
    return math.exp(nll / count)


def throughput_benchmark(
    model: InferbitModel,
    prompt_tokens: List[int],
    *,
    output_tokens: int = 128,
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> tuple[float, float]:
    """Return (tokens_per_sec, latency_ms_per_token)."""
    durations: List[float] = []
    total = warmup_runs + measured_runs

    for i in range(total):
        model.kv_clear()
        t0 = time.perf_counter()
        out = model.generate_tokens(prompt_tokens, max_tokens=output_tokens, temperature=0.0)
        dt = time.perf_counter() - t0
        if i >= warmup_runs:
            # use actual generated count to avoid skew on early EOS
            produced = max(1, len(out))
            durations.append(dt / produced)

    avg_sec_per_token = statistics.mean(durations)
    tok_s = 1.0 / avg_sec_per_token
    return tok_s, avg_sec_per_token * 1000.0


def evaluate_model_gates(
    model: InferbitModel,
    *,
    token_samples: Optional[List[List[int]]] = None,
    prompt_tokens: Optional[List[int]] = None,
    output_tokens: int = 128,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    gates: Optional[EvalGates] = None,
) -> EvalResult:
    """Run evaluation and gate checks."""
    gates = gates or EvalGates()
    if prompt_tokens is None:
        prompt_tokens = [1, 2, 3, 4, 5]

    ppl = None
    if token_samples:
        ppl = perplexity_from_token_samples(model, token_samples)

    tok_s, latency_ms = throughput_benchmark(
        model,
        prompt_tokens,
        output_tokens=output_tokens,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )

    memory_mb = model.total_memory_mb

    failed: List[str] = []
    if gates.max_perplexity is not None and ppl is not None and ppl > gates.max_perplexity:
        failed.append(f"perplexity {ppl:.3f} > max {gates.max_perplexity}")
    if gates.min_tokens_per_sec is not None and tok_s < gates.min_tokens_per_sec:
        failed.append(f"tokens/sec {tok_s:.3f} < min {gates.min_tokens_per_sec}")
    if gates.max_memory_mb is not None and memory_mb > gates.max_memory_mb:
        failed.append(f"memory_mb {memory_mb:.1f} > max {gates.max_memory_mb}")

    return EvalResult(
        perplexity=ppl,
        tokens_per_sec=tok_s,
        latency_ms_per_token=latency_ms,
        memory_mb=memory_mb,
        passes=(len(failed) == 0),
        failed_gates=failed,
    )
