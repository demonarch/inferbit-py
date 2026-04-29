# InferBit

**v0.2.0** — Run any open LLM on CPU. One command.

```bash
pip install inferbit[cli]
inferbit quantize mistralai/Mistral-7B-Instruct-v0.3 -o model.ibf
inferbit chat model.ibf
```

InferBit converts HuggingFace models to optimized INT4 and runs them on any CPU (Apple Silicon, x86) with no GPU, no Docker, and no complex setup.

## Install

```bash
# Library only
pip install inferbit

# Library + CLI
pip install inferbit[cli]

# Everything (library + CLI + server)
pip install inferbit[all]
```

Requires Python 3.9+. Works on macOS (ARM/Intel) and Linux (x86_64).

## Quickstart

### Command line

```bash
# Convert any HuggingFace model to INT4
inferbit quantize meta-llama/Llama-3.2-1B -o llama.ibf

# Convert a local safetensors file
inferbit quantize ./model.safetensors -o model.ibf

# Convert from Ollama (if installed)
inferbit quantize ollama://llama3:8b -o llama3.ibf

# Interactive chat
inferbit chat model.ibf

# Benchmark
inferbit bench model.ibf --tokens 128 --runs 3

# Model info
inferbit info model.ibf

# Serve with OpenAI-compatible API
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

Apple Silicon, INT4 + INT8 attention, 8 threads:

| Model | File size | Decode speed | Quality |
|-------|-----------|-------------|---------|
| TinyLlama 1.1B | 643 MB | 34.6 tok/s | Good |
| Mistral 7B | 3,971 MB | 6.8 tok/s | Excellent |

Compression: 3.5x vs FP16 source. No retraining required.

## How it works

1. **Convert**: reads safetensors/GGUF weights, quantizes to INT4 (MLP layers) and INT8 (attention/embeddings), packs into an optimized `.ibf` binary format
2. **Load**: memory-maps the `.ibf` file for instant loading
3. **Run**: SIMD-optimized kernels (NEON on ARM, AVX2 on x86) with multi-threaded matmul and parallel attention heads

The `.ibf` format is designed for fast loading: 64-byte aligned, mmap-friendly, no parsing at load time.

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

## Architecture

```
libinferbit (C shared library)
    |
    +-- Python: pip install inferbit
    +-- Node.js: npm install @inferbit/node (coming soon)
```

Single C engine, multiple language bindings. Same model, same results, any language.

## License

MIT
