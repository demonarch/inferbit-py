"""InferbitModel — main user-facing class."""

from __future__ import annotations

import ctypes
from ctypes import c_int, c_int32, c_float, POINTER
from pathlib import Path
from typing import Iterator, Optional

from ._ffi import _get_lib, SampleParams, StreamCallback


class InferbitModel:
    """Load and run inference on an .ibf model."""

    def __init__(self, ptr, lib, tokenizer=None):
        self._ptr = ptr
        self._lib = lib
        self._tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        path: str,
        *,
        threads: int = 0,
        context_length: int = 0,
        kv_dynamic: bool = False,
        kv_window: int = 0,
        tokenizer=None,
    ) -> "InferbitModel":
        """Load a pre-converted .ibf model.

        kv_window: rotating KV-cache window (doc 36 phase 2.2). 0 = full
        causal cache (default). >0 = ring buffer of `kv_window` token
        slots — peak KV RAM becomes O(kv_window) instead of
        O(context_length), at the cost of a sliding attention horizon.
        """
        lib = _get_lib()

        config = lib.inferbit_config_create()
        if not config:
            raise RuntimeError("Failed to create config")

        if threads > 0:
            lib.inferbit_config_set_threads(config, threads)
        if context_length > 0:
            lib.inferbit_config_set_context_length(config, context_length)
        if kv_dynamic:
            lib.inferbit_config_set_kv_cache_dynamic(config, 1)
        if kv_window > 0:
            lib.inferbit_config_set_kv_window(config, kv_window)

        ptr = lib.inferbit_load(path.encode(), config)
        lib.inferbit_config_free(config)

        if not ptr:
            err = lib.inferbit_last_error()
            msg = err.decode() if err else "unknown error"
            raise RuntimeError(f"Failed to load model: {msg}")

        return cls(ptr, lib, tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        bits: int = 4,
        sensitive_bits: int = 8,
        cache_dir: str = "./models",
        threads: int = 0,
        tokenizer=None,
        **kwargs,
    ) -> "InferbitModel":
        """Download from HuggingFace, convert to .ibf, and load."""
        from .convert import convert_pretrained

        ibf_path = convert_pretrained(
            model_id,
            bits=bits,
            sensitive_bits=sensitive_bits,
            cache_dir=cache_dir,
        )

        # Try to load tokenizer if not provided
        if tokenizer is None:
            try:
                from tokenizers import Tokenizer as HFTokenizer
                from huggingface_hub import hf_hub_download
                tok_path = hf_hub_download(model_id, "tokenizer.json")
                tokenizer = HFTokenizer.from_file(tok_path)
            except Exception:
                pass

        return cls.load(ibf_path, threads=threads, tokenizer=tokenizer, **kwargs)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            self._lib.inferbit_free(self._ptr)
            self._ptr = None

    # ── Generation ──────────────────────────────────────────────

    def _make_params(
        self,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        seed: int = -1,
    ) -> SampleParams:
        p = self._lib.inferbit_default_sample_params()
        p.temperature = temperature
        p.top_k = top_k
        p.top_p = top_p
        p.repeat_penalty = repeat_penalty
        p.max_tokens = max_tokens
        p.seed = seed
        return p

    def _encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError(
                "No tokenizer loaded. Pass tokenizer= to load() or use generate_tokens() with raw IDs."
            )
        enc = self._tokenizer.encode(text)
        return enc.ids if hasattr(enc, "ids") else list(enc)

    def _decode(self, ids: list[int]) -> str:
        if self._tokenizer is None:
            return " ".join(str(i) for i in ids)
        return self._tokenizer.decode(ids)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a string prompt."""
        input_ids = self._encode(prompt)
        out_ids = self.generate_tokens(input_ids, **kwargs)
        return self._decode(out_ids)

    def generate_tokens(
        self, input_tokens: list[int], max_tokens: int = 256, **kwargs
    ) -> list[int]:
        """Generate from token IDs. Returns output token IDs."""
        n_in = len(input_tokens)
        in_arr = (c_int32 * n_in)(*input_tokens)
        out_arr = (c_int32 * max_tokens)()
        params = self._make_params(max_tokens=max_tokens, **kwargs)

        n = self._lib.inferbit_generate(
            self._ptr, in_arr, n_in, out_arr, max_tokens, params
        )
        if n < 0:
            err = self._lib.inferbit_last_error()
            msg = err.decode() if err else "unknown error"
            raise RuntimeError(f"Generation failed: {msg}")

        return list(out_arr[:n])

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream generated text token by token."""
        input_ids = self._encode(prompt)
        for token_id in self.stream_tokens(input_ids, **kwargs):
            yield self._decode([token_id])

    def stream_tokens(
        self, input_tokens: list[int], max_tokens: int = 256, **kwargs
    ) -> Iterator[int]:
        """Stream generated token IDs one at a time."""
        import queue
        import threading

        q: queue.Queue = queue.Queue()
        n_in = len(input_tokens)
        in_arr = (c_int32 * n_in)(*input_tokens)
        params = self._make_params(max_tokens=max_tokens, **kwargs)

        @StreamCallback
        def callback(token_id, ctx):
            q.put(token_id)
            return 1

        def run():
            self._lib.inferbit_generate_stream(
                self._ptr, in_arr, n_in, callback, None, params
            )
            q.put(None)  # Sentinel

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        while True:
            token_id = q.get()
            if token_id is None:
                break
            yield token_id

        thread.join()

    def forward(self, tokens: list[int]) -> list[float]:
        """Run a forward pass and return logits."""
        n = len(tokens)
        in_arr = (c_int32 * n)(*tokens)
        vocab = self.vocab_size
        out_arr = (c_float * vocab)()

        rc = self._lib.inferbit_forward(self._ptr, in_arr, n, out_arr, vocab)
        if rc != 0:
            err = self._lib.inferbit_last_error()
            msg = err.decode() if err else "unknown error"
            raise RuntimeError(f"Forward pass failed: {msg}")

        return list(out_arr)

    @staticmethod
    def build_target_layer_ids(n_target_layers: int, n_draft_layers: int) -> list[int]:
        """Evenly-spaced target-layer indices for hidden-state capture
        (doc 36 phase 4.1, mirrors dflash's build_target_layer_ids).
        Skips the first/last 3 layers; returns ascending indices."""
        lib = _get_lib()
        buf = (c_int * max(n_draft_layers, 1))()
        count = lib.inferbit_build_target_layer_ids(
            n_target_layers, n_draft_layers, buf
        )
        return [buf[i] for i in range(count)]

    def forward_with_hiddens(self, tokens: list[int], layer_ids: list[int]):
        """Prefill forward that also captures per-layer hidden states
        (doc 36 phase 4.1). Runs on the Metal backend; the GPU context is
        created lazily on first call and cached on the model.

        Returns (logits, hiddens) where:
          logits  — numpy array [n_tokens, vocab_size]
          hiddens — dict {layer_id: numpy array [n_tokens, hidden_size]}

        Intended for the DFlash hybrid orchestrator: the draft model
        conditions its K-token predictions on the target's mid-stack
        hidden states.
        """
        import numpy as np

        n = len(tokens)
        if n == 0:
            raise ValueError("tokens must be non-empty")
        n_layers = len(layer_ids)
        if n_layers == 0:
            raise ValueError("layer_ids must be non-empty")
        hidden = self.hidden_size
        vocab = self.vocab_size

        tok_arr = (c_int32 * n)(*tokens)
        lid_arr = (c_int * n_layers)(*layer_ids)
        # Caller-allocated output buffers; the C side fills them in place.
        hiddens = np.zeros((n_layers, n, hidden), dtype=np.float32)
        logits = np.zeros((n, vocab), dtype=np.float32)

        rc = self._lib.inferbit_forward_with_hiddens(
            self._ptr, tok_arr, n, lid_arr, n_layers,
            hiddens.ctypes.data_as(POINTER(c_float)),
            logits.ctypes.data_as(POINTER(c_float)),
        )
        if rc != 0:
            err = self._lib.inferbit_last_error()
            msg = err.decode() if err else "unknown error"
            raise RuntimeError(f"forward_with_hiddens failed: {msg}")

        hidden_map = {lid: hiddens[i] for i, lid in enumerate(layer_ids)}
        return logits, hidden_map

    # ── KV cache ────────────────────────────────────────────────

    def kv_clear(self):
        self._lib.inferbit_kv_clear(self._ptr)

    def kv_truncate(self, length: int):
        self._lib.inferbit_kv_truncate(self._ptr, length)

    @property
    def kv_length(self) -> int:
        return self._lib.inferbit_kv_length(self._ptr)

    # ── Speculative decoding ────────────────────────────────────

    def set_draft_model(self, draft, draft_tokens: int = 4):
        """Attach a sibling model as draft for speculative decoding.

        Requires matching vocab and greedy sampling (temperature < 0.01).
        Call unset_draft_model() to detach. Takes precedence over
        set_prompt_lookup() when both configured.
        """
        if draft is None:
            self._lib.inferbit_unset_draft_model(self._ptr)
            return
        self._lib.inferbit_set_draft_model(self._ptr, draft._ptr, draft_tokens)

    def unset_draft_model(self):
        self._lib.inferbit_unset_draft_model(self._ptr)

    def set_prompt_lookup(self, ngram: int, k: int):
        """Enable draft-less speculation via n-gram match over running history.

        ngram=0 disables. Typical: ngram=2 or 3, k=4 to 10. Greedy-only.
        """
        self._lib.inferbit_set_prompt_lookup(self._ptr, ngram, k)

    # ── Model info ──────────────────────────────────────────────

    @property
    def architecture(self) -> str:
        return self._lib.inferbit_model_architecture(self._ptr).decode()

    @property
    def num_layers(self) -> int:
        return self._lib.inferbit_model_num_layers(self._ptr)

    @property
    def hidden_size(self) -> int:
        return self._lib.inferbit_model_hidden_size(self._ptr)

    @property
    def vocab_size(self) -> int:
        return self._lib.inferbit_model_vocab_size(self._ptr)

    @property
    def max_context(self) -> int:
        return self._lib.inferbit_model_max_context(self._ptr)

    @property
    def bits(self) -> int:
        return self._lib.inferbit_model_default_bits(self._ptr)

    @property
    def weight_memory_mb(self) -> float:
        return self._lib.inferbit_model_weight_memory(self._ptr) / (1024 * 1024)

    @property
    def kv_memory_mb(self) -> float:
        return self._lib.inferbit_model_kv_memory(self._ptr) / (1024 * 1024)

    @property
    def total_memory_mb(self) -> float:
        return self._lib.inferbit_model_total_memory(self._ptr) / (1024 * 1024)
