"""Test the inferbit Python library against a fake safetensors model."""

import os
import struct
import json
import tempfile
import pytest

# Ensure we can find libinferbit from development build
os.environ.setdefault(
    "INFERBIT_LIB_PATH",
    os.path.join(
        os.path.dirname(__file__), "..", "..", "libinferbit", "build", "libinferbit.dylib"
    ),
)

from inferbit import InferbitModel, convert

# Model dimensions
HIDDEN = 64
HEADS = 2
KV_HEADS = 2
HEAD_DIM = 32
INTER = 128
VOCAB = 256
NUM_LAYERS = 2


def _fp16(val: float) -> bytes:
    """Convert float to FP16 bytes."""
    import struct as s
    # Simple conversion via struct
    return s.pack("<e", val)


def _write_fake_safetensors(path: str):
    """Write a fake safetensors file with LLaMA-style naming."""
    tensors = {}

    def add(name, shape):
        elems = 1
        for d in shape:
            elems *= d
        tensors[name] = {"dtype": "F16", "shape": shape, "elems": elems}

    add("model.embed_tokens.weight", [VOCAB, HIDDEN])
    for l in range(NUM_LAYERS):
        add(f"model.layers.{l}.self_attn.q_proj.weight", [HIDDEN, HIDDEN])
        add(f"model.layers.{l}.self_attn.k_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN])
        add(f"model.layers.{l}.self_attn.v_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN])
        add(f"model.layers.{l}.self_attn.o_proj.weight", [HIDDEN, HIDDEN])
        add(f"model.layers.{l}.mlp.gate_proj.weight", [INTER, HIDDEN])
        add(f"model.layers.{l}.mlp.up_proj.weight", [INTER, HIDDEN])
        add(f"model.layers.{l}.mlp.down_proj.weight", [HIDDEN, INTER])
        add(f"model.layers.{l}.input_layernorm.weight", [HIDDEN])
        add(f"model.layers.{l}.post_attention_layernorm.weight", [HIDDEN])
    add("model.norm.weight", [HIDDEN])
    add("lm_head.weight", [VOCAB, HIDDEN])

    # Compute offsets
    offset = 0
    header = {}
    for name, info in tensors.items():
        size = info["elems"] * 2  # FP16
        header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size

    total_data = offset
    header_json = json.dumps(header).encode()

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        # Write FP16 data
        for i in range(total_data // 2):
            val = 0.01 * ((i % 100) - 50)
            f.write(_fp16(val))


@pytest.fixture
def fake_model(tmp_path):
    """Create a fake safetensors file and convert to .ibf."""
    st_path = str(tmp_path / "model.safetensors")
    ibf_path = str(tmp_path / "model.ibf")
    _write_fake_safetensors(st_path)
    convert(st_path, ibf_path, bits=8, sensitive_bits=8)
    return ibf_path


class TestConvert:
    def test_convert_creates_file(self, tmp_path):
        st_path = str(tmp_path / "model.safetensors")
        ibf_path = str(tmp_path / "model.ibf")
        _write_fake_safetensors(st_path)
        result = convert(st_path, ibf_path, bits=4, sensitive_bits=8)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_convert_progress(self, tmp_path):
        st_path = str(tmp_path / "model.safetensors")
        ibf_path = str(tmp_path / "model.ibf")
        _write_fake_safetensors(st_path)
        stages = []

        def on_progress(pct, stage):
            stages.append((pct, stage))

        convert(st_path, ibf_path, progress=on_progress)
        assert len(stages) > 0


class TestModel:
    def test_load(self, fake_model):
        model = InferbitModel.load(fake_model)
        assert model.architecture == "llama"
        assert model.num_layers == NUM_LAYERS
        assert model.hidden_size == HIDDEN
        assert model.vocab_size == VOCAB

    def test_info(self, fake_model):
        model = InferbitModel.load(fake_model)
        assert model.max_context > 0
        assert model.weight_memory_mb > 0
        assert model.total_memory_mb > 0

    def test_forward(self, fake_model):
        model = InferbitModel.load(fake_model)
        logits = model.forward([1, 2, 3])
        assert len(logits) == VOCAB
        assert all(isinstance(v, float) for v in logits)

    def test_generate_tokens(self, fake_model):
        model = InferbitModel.load(fake_model)
        out = model.generate_tokens([1, 2, 3], max_tokens=5, temperature=0.0)
        assert len(out) > 0
        assert all(0 <= t < VOCAB for t in out)

    def test_kv_cache(self, fake_model):
        model = InferbitModel.load(fake_model)
        assert model.kv_length == 0
        model.forward([1, 2])
        assert model.kv_length == 2
        model.kv_clear()
        assert model.kv_length == 0

    def test_kv_truncate(self, fake_model):
        model = InferbitModel.load(fake_model)
        model.forward([1, 2, 3, 4, 5])
        assert model.kv_length == 5
        model.kv_truncate(3)
        assert model.kv_length == 3
