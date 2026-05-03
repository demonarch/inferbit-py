# Changelog

## v0.3.0 — INT4 conversion pipeline + spec-decode bindings

Tracks libinferbit v0.3.0. See `modules/libinferbit/CHANGELOG.md` for
the runtime-side details.

### Bindings

- `ib_pq_session_matmul_batched` — batched matmul for B inputs.
- `ib_pq_speculative_step` — caller-supplied-draft spec decoding.
- `IB_RAW_U16` raw dtype recognized in `_pq_format`.
- IBF JSON-header reserve now scales with `len(raw_tensors)` for
  raw-heavy bundles (INT4 with all-matmul + outlier sidecar can hit
  500+ raw entries).

### Conversion

- `scripts/convert/int4_encode.py` — group-G symmetric INT4 with
  optional AWQ activation scaling.
- `scripts/convert/int4_gptq.py` — blocked GPTQ encoder with
  iterative damping fallback for rank-deficient Hessians.
- `scripts/convert/int4_quickstart.py` — one-shot pipeline:
  capture act-scales → bundled_convert with INT4 + AWQ + outliers
  defaults. Produces a +0.12% rel PPL bundle in one command.
- `scripts/convert/bundled_convert.py` gains:
  - `--int4-tensors substr1,substr2,…` (encode matched matmul tensors
    as INT4 instead of PQ pyramid)
  - `--int4-G {32,64,128}` and `--int4-G-attn` for per-tensor-class G
  - `--int4-outlier-pct` (0.002 default in quickstart)
  - `--gptq-acts <npz|dir>` (blocked GPTQ via per-tensor X or
    streaming Hessian directory)
  - `--fp16-tensors substr1,substr2,…` (raw fp16 weight matrix)

### Calibration

- `scripts/calibration/capture_act_scales.py` — per-tensor input
  RMS for AWQ.
- `scripts/calibration/capture_activations.py` — per-tensor X for
  GPTQ (in-memory NPZ).
- `scripts/calibration/capture_hessians.py` — per-tensor H = X^T X / N
  (in-memory NPZ).
- `scripts/calibration/capture_hessians_stream.py` — streaming H
  written per-tensor to a directory. Avoids the monolithic-save
  disk explosion for large calibration sets (≥4k tokens × full
  N=5632 fp32 Hessians).

### Validation / bench

New scripts (numbered 60+) cover: INT4 PPL, INT4 session smoke test,
INT4 inspection, KV pyramid PPL, F1.c batched bench, F1.c forward
bench, F1.c per-matmul diff, spec-decode bench.

### Critical fix

- Matches libinferbit's AWQ scratch sizing fix. Forward-step-batch
  output for tall tensors (down_proj N=inter > hidden) was silently
  wrong on prior versions; now bit-identical to single-token forward.

## v0.2.3 (and earlier)

See git history.
