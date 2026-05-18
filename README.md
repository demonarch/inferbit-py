# InferBit

**v0.4.1** — Run any open LLM on CPU. One command.

```bash
pip install inferbit[cli]
inferbit quantize mistralai/Mistral-7B-Instruct-v0.3 -o model.ibf
inferbit chat model.ibf
```

InferBit converts HuggingFace models to a compact 4-bit PQv2 codebook format (`.ibf`) and runs them on CPU everywhere, with optional Apple Metal GPU acceleration on Apple Silicon and a drive mode that streams weights from disk for sub-GB peak RAM on 8B models. No GPU required, no Docker, no complex setup.

## Install

```bash
# Library only
pip install inferbit

# Library + CLI
pip install inferbit[cli]

# Everything (library + CLI + server)
pip install inferbit[all]
```

Requires Python 3.9+. Prebuilt wheels for **macOS (ARM64, x86_64)**, **Linux (x86_64, ARM64)**, and **Windows (x64, ARM64)** — six platforms, all CPU-only by default. Apple Metal GPU is a build-from-source option (see [Platform support](#platform-support) below).

## Quickstart

### Command line

```bash
# Convert any HuggingFace model to PQv2 4-bit
inferbit quantize meta-llama/Llama-3.2-1B -o llama.ibf

# Convert a local safetensors file
inferbit quantize ./model.safetensors -o model.ibf

# Convert from Ollama (if installed)
inferbit quantize ollama://llama3:8b -o llama3.ibf

# Auto-calibrate: try INT2/INT4/INT8 and keep the first that hits the gate
inferbit quantize meta-llama/Llama-3.2-1B -o llama.ibf \
    --auto-calibrate --max-perplexity 12.0 --min-tokens-per-sec 30

# Quality-gated eval against a JSONL token dataset
inferbit eval-gates model.ibf --dataset tokens.jsonl \
    --max-perplexity 12.0 --min-tokens-per-sec 30

# Interactive chat
inferbit chat model.ibf

# Benchmark
inferbit bench model.ibf --tokens 128 --runs 3

# Model info
inferbit info model.ibf

# Serve OpenAI-compatible API (requires: pip install inferbit[server])
inferbit serve model.ibf --port 8000
```

### Python API

```python
from inferbit import InferbitModel

# Load from HuggingFace (downloads, converts, and loads automatically)
model = InferbitModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    bits=4,
)

# Generate text
output = model.generate("Explain gravity in one sentence:")
print(output)
# "Gravity is the force that attracts objects with mass towards each other."

# Stream tokens
for token in model.stream("Write a haiku about mountains:"):
    print(token, end="", flush=True)

# Or load a pre-converted model
model = InferbitModel.load("model.ibf")
```

### Convert separately

```python
from inferbit import convert

# Convert safetensors to IBF
convert("model.safetensors", "model.ibf", bits=4, sensitive_bits=8)

# Convert a HuggingFace directory (with config.json + sharded safetensors)
convert("./model_dir/", "model.ibf", bits=4)

# Convert with progress callback
convert("model.safetensors", "model.ibf", progress=lambda pct, stage: print(f"{pct:.0%} {stage}"))
```

### Token-level API

```python
from inferbit import InferbitModel

model = InferbitModel.load("model.ibf")

# Work with raw token IDs
token_ids = model.generate_tokens([1, 2, 3, 4, 5], max_tokens=20, temperature=0.7)

# Get raw logits
logits = model.forward([1, 2, 3])

# KV cache control
model.kv_clear()
model.kv_truncate(512)
print(model.kv_length)
```

### Model info

```python
model = InferbitModel.load("model.ibf")
print(model.architecture)   # "llama"
print(model.num_layers)      # 32
print(model.hidden_size)     # 4096
print(model.vocab_size)      # 32768
print(model.max_context)     # 32768
print(model.bits)            # 4
print(model.total_memory_mb) # 3971.0
```

### Quality-gated quantization

```python
from inferbit import search_quantization_profile, EvalGates

# Automatically find the most aggressive quantization that meets quality targets
result = search_quantization_profile(
    "model.safetensors",
    output_dir="./models",
    gates=EvalGates(max_perplexity=10.0, min_tokens_per_sec=5.0),
)
print(f"Selected: {result.selected.name} ({result.selected.bits}-bit)")
print(f"Speed: {result.eval_result.tokens_per_sec:.1f} tok/s")
```

## Supported Sources

| Source | Example |
|--------|---------|
| HuggingFace Hub | `inferbit quantize mistralai/Mistral-7B-Instruct-v0.3` |
| Local safetensors | `inferbit quantize model.safetensors` |
| Sharded safetensors directory | `inferbit quantize ./model_dir/` |
| Local GGUF | `inferbit quantize model.gguf` |
| Ollama models | `inferbit quantize ollama://llama3:8b` |

## Supported Models

Any LLaMA-family architecture with public weights:

- LLaMA 2, LLaMA 3, LLaMA 3.2
- Mistral, Mixtral
- TinyLlama
- Code Llama
- And any model with the same architecture (GQA/MQA/MHA, RMSNorm, SiLU, RoPE)

## Benchmarks

> **Correction (2026-05-17):** the "InferBit PQv2" rows below
> were actually measured against `InferBit INT4-blk32 + INT8`,
> not PQv2. The production convert path didn't emit PQ-format
> tensors in v0.4.1. Real PQv2 (incl. pyramid) is being wired
> for v0.4.2 — see [`docs/v2/00_CORRECTION.md`](https://github.com/inferbit/inferbit/blob/main/docs/v2/00_CORRECTION.md).

Apple M4, full v0.4.1 cross-engine matrix. Perplexity measured on
the same tokenized 2048-token wikitext window for both engines
(llama.cpp's tokenization fed to both `bench_ppl_run` and
`llama-perplexity`, so quality is compared byte-for-byte over the
identical sequence). Prefill via `bench_compare --prompt-tokens 64`
and `llama-bench -p 64`; decode `--gen-tokens 128` / `-n 128`. Peak
RAM from `getrusage` (InferBit) and `/usr/bin/time -l` (llama.cpp).

**TinyLlama 1.1B-Chat**

| Engine / mode | File | Prefill | Decode | Peak RAM | PPL |
|---|---:|---:|---:|---:|---:|
| InferBit PQv2 — Metal | 528 MiB | 437 t/s | 55.5 t/s | 1205 MB | **13.06** |
| InferBit PQv2 — CPU | 528 MiB | 27 t/s | 24.9 t/s | 627 MB | 13.06 |
| InferBit PQv2 — drive | 528 MiB | 287 t/s | 9.4 t/s | **297 MB** | 13.06 |
| llama.cpp Q4_K_M — Metal | 638 MiB | **1347 t/s** | **121.3 t/s** | 704 MB | **13.89** |
| llama.cpp Q4_K_M — CPU | 638 MiB | 130 t/s | 74.2 t/s | 1293 MB | 13.89 |

**Llama-3.2-1B Instruct**

| Engine / mode | File | Prefill | Decode | Peak RAM | PPL |
|---|---:|---:|---:|---:|---:|
| InferBit PQv2 — Metal | 718 MiB | 435 t/s | 48.1 t/s | 1258 MB | **11.29** |
| InferBit PQv2 — CPU | 718 MiB | 28 t/s | 22.7 t/s | 847 MB | 11.37 |
| InferBit PQv2 — drive | 718 MiB | 257 t/s | 9.3 t/s | **602 MB** | 11.29 |
| llama.cpp Q4_K_M — Metal | 770 MiB | **1359 t/s** | **104.3 t/s** | 888 MB | **12.33** |
| llama.cpp Q4_K_M — CPU | 770 MiB | 132 t/s | 64.3 t/s | 1644 MB | 12.33 |

**Llama-3.1-8B Instruct**

| Engine / mode | File | Prefill | Decode | Peak RAM | PPL |
|---|---:|---:|---:|---:|---:|
| InferBit PQv2 — Metal | 3.75 GiB | 65 t/s | 8.5 t/s | 3203 MB | **6.34** |
| InferBit PQv2 — CPU | 3.75 GiB | 4.5 t/s | 4.2 t/s | 4306 MB | 6.36 |
| InferBit PQv2 — drive | 3.75 GiB | 34.3 t/s | 0.70 t/s | **1359 MB** | 6.34 |
| llama.cpp Q4_K_M — Metal | 4.58 GiB | **216 t/s** | **20.1 t/s** | 4784 MB | **6.77** |
| llama.cpp Q4_K_M — CPU | 4.58 GiB | 4.2 t/s | 2.4 t/s | 6755 MB | 6.77 |

**What the numbers say:**

- **Quality** — InferBit PQv2 perplexity is 6–8% **lower** than the
  same-bit-budget Q4_K_M on all three models (over the identical
  token stream).
- **File size** — InferBit `.ibf` is 7–18% smaller than the
  equivalent Q4_K_M GGUF.
- **Speed** — On M4 Metal, llama.cpp is 2–3× faster on decode and
  3–6× on prefill; on pure CPU the engines are closer. Closing the
  Metal decode gap is active work (the PQv2 random-access codebook
  reads).
- **Memory** — InferBit drive mode holds the 8B model in **1.36 GB
  peak RAM** at the same PPL as the in-memory path (3.20 GB) —
  −58% RAM at zero quality cost. Throughput drops at long contexts
  (re-streams weights every position); useful when RAM is the
  binding constraint.

Full methodology + tooling notes in
[`docs/34_METRICS_SNAPSHOT.md`](https://github.com/inferbit/inferbit/blob/main/docs/34_METRICS_SNAPSHOT.md).

## How it works

1. **Convert**: reads safetensors/GGUF weights, quantizes the MLP weights with **PQv2** (K=256 per-(chunk, subchunk) codebook + uint8 indices, 4-bit-equivalent) and attention/embeddings with INT8, packs everything into a single mmap-friendly `.ibf` binary.
2. **Load**: memory-maps the `.ibf` file for instant loading.
3. **Run**: hand-tuned C kernels with multi-threaded matmul and parallel attention heads. On Apple Silicon, an optional Metal GPU backend ([build from source](#platform-support)) routes both prefill and decode through the GPU; the same `.ibf` works in both modes.
4. **Drive mode** (`IB_RESIDENCY_MODE=drive`, macOS/Linux): weights stream from disk through a bounded GPU/CPU scratch ring instead of being resident. Bit-identical perplexity; the 8B model holds in 1.36 GB peak RAM (see [Benchmarks](#benchmarks)).

The `.ibf` format is 64-byte aligned, no parsing at load time, and tracks the same K=256 codebooks the GPU kernels consume — there is no quality difference between CPU and GPU.

## Configuration

### Quantization

| Flag | Default | Description |
|------|---------|-------------|
| `--bits` | 4 | Weight quantization (2, 4, 8) |
| `--sensitive-bits` | 8 | Attention/embedding bits |
| `--sparsity` | 0.0 | Structured sparsity (0.0-0.6) |

### Generation

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 0.7 | Sampling temperature |
| `--top-k` | 40 | Top-K sampling |
| `--top-p` | 0.9 | Nucleus sampling |
| `--max-tokens` | 512 | Max tokens to generate |
| `--threads` | auto | CPU threads |

## Platform support

| Platform | Wheel | CPU SIMD | Metal GPU | Drive mode |
|---|---|---|---|---|
| macOS Apple Silicon (arm64) | `macosx_11_0_arm64` | NEON | opt-in (build from source) | ✓ |
| macOS Intel (x86_64) | `macosx_10_15_x86_64` | portable C | — | ✓ |
| Linux x86_64 | `manylinux_2_17_x86_64` | portable C | — | ✓ |
| Linux ARM64 (aarch64) | `manylinux_2_17_aarch64` | NEON + dotprod | — | ✓ |
| Windows x64 | `win_amd64` | portable C (MSVC) | — | — |
| Windows ARM64 | `win_arm64` | NEON (MSVC) | — | — |

**Build with Metal GPU** (Apple Silicon, recommended for best M-series throughput):

```bash
# clone the engine repo, then:
cmake -B build -DIB_ENABLE_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# point InferBit at the freshly-built dylib:
export INFERBIT_LIB_PATH="$PWD/build/libinferbit.dylib"
python -c "import inferbit; print(inferbit.__version__)"
```

Drive mode is currently macOS/Linux only (uses POSIX `madvise`/`fcntl(F_NOCACHE)`); on Windows the runtime keeps weights resident.

## Architecture

```
libinferbit (C shared library)
    |
    +-- Python: pip install inferbit
    +-- Node.js: npm install @inferbit/{core,node,cli}
```

Single C engine, multiple language bindings. Same `.ibf` model file, same numerics, any language.

## License

MIT
