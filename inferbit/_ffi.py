"""ctypes bindings to libinferbit."""

import ctypes
from ctypes import (
    c_char_p, c_int, c_int32, c_float, c_size_t, c_void_p,
    POINTER, Structure, CFUNCTYPE,
)

from ._binary import find_library

# ── Load library ────────────────────────────────────────────────

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        path = find_library()
        _lib = ctypes.CDLL(path)
        _setup_signatures(_lib)
    return _lib


# ── Structs ─────────────────────────────────────────────────────

class SampleParams(Structure):
    _fields_ = [
        ("temperature", c_float),
        ("top_k", c_int),
        ("top_p", c_float),
        ("repeat_penalty", c_float),
        ("max_tokens", c_int),
        ("seed", c_int),
    ]


class IbPqPolicy(Structure):
    _fields_ = [
        ("variant", c_int),
        ("skip_threshold", c_float),
        ("act_threshold", c_float),
    ]


class IbPqSampleParams(Structure):
    _fields_ = [
        ("temperature", c_float),
        ("top_k", c_int),
        ("top_p", c_float),
        ("seed", ctypes.c_uint32),
    ]


# Variant constants (mirror ib_pq_variant in pq_decode.h)
VARIANT_STREAMING = 0
VARIANT_L1_ONLY = 1
VARIANT_L2SKIP = 2
VARIANT_SPARSE = 3
VARIANT_INT8 = 4


class ConvertConfig(Structure):
    _fields_ = [
        ("default_bits", c_int),
        ("sensitive_bits", c_int),
        ("sparsity", c_float),
        ("block_size", c_int),
        ("kv_bits", c_int),
        ("threads", c_int),
        ("progress", c_void_p),  # Function pointer — set separately
        ("progress_ctx", c_void_p),
    ]


# Callback types
StreamCallback = CFUNCTYPE(c_int, c_int32, c_void_p)
ProgressCallback = CFUNCTYPE(None, c_float, c_char_p, c_void_p)


# ── Setup function signatures ───────────────────────────────────

def _setup_signatures(lib):
    # Version
    lib.inferbit_version.restype = c_char_p
    lib.inferbit_version.argtypes = []
    lib.inferbit_version_major.restype = c_int
    lib.inferbit_version_minor.restype = c_int
    lib.inferbit_version_patch.restype = c_int

    # Error
    lib.inferbit_last_error.restype = c_char_p
    lib.inferbit_last_error.argtypes = []

    # Config
    lib.inferbit_config_create.restype = c_void_p
    lib.inferbit_config_create.argtypes = []
    lib.inferbit_config_free.restype = None
    lib.inferbit_config_free.argtypes = [c_void_p]
    lib.inferbit_config_set_threads.argtypes = [c_void_p, c_int]
    lib.inferbit_config_set_context_length.argtypes = [c_void_p, c_int]
    lib.inferbit_config_set_kv_cache_dynamic.argtypes = [c_void_p, c_int]

    # Model lifecycle
    lib.inferbit_load.restype = c_void_p
    lib.inferbit_load.argtypes = [c_char_p, c_void_p]
    lib.inferbit_free.restype = None
    lib.inferbit_free.argtypes = [c_void_p]

    # Sample params
    lib.inferbit_default_sample_params.restype = SampleParams
    lib.inferbit_default_sample_params.argtypes = []

    # Generate
    lib.inferbit_generate.restype = c_int
    lib.inferbit_generate.argtypes = [
        c_void_p, POINTER(c_int32), c_int,
        POINTER(c_int32), c_int, SampleParams,
    ]

    lib.inferbit_generate_stream.restype = c_int
    lib.inferbit_generate_stream.argtypes = [
        c_void_p, POINTER(c_int32), c_int,
        StreamCallback, c_void_p, SampleParams,
    ]

    lib.inferbit_forward.restype = c_int
    lib.inferbit_forward.argtypes = [
        c_void_p, POINTER(c_int32), c_int,
        POINTER(c_float), c_int,
    ]

    # KV cache
    lib.inferbit_kv_clear.restype = None
    lib.inferbit_kv_clear.argtypes = [c_void_p]
    lib.inferbit_kv_truncate.restype = None
    lib.inferbit_kv_truncate.argtypes = [c_void_p, c_int]
    lib.inferbit_kv_length.restype = c_int
    lib.inferbit_kv_length.argtypes = [c_void_p]

    # Model info
    lib.inferbit_model_architecture.restype = c_char_p
    lib.inferbit_model_architecture.argtypes = [c_void_p]
    lib.inferbit_model_num_layers.restype = c_int
    lib.inferbit_model_num_layers.argtypes = [c_void_p]
    lib.inferbit_model_hidden_size.restype = c_int
    lib.inferbit_model_hidden_size.argtypes = [c_void_p]
    lib.inferbit_model_vocab_size.restype = c_int
    lib.inferbit_model_vocab_size.argtypes = [c_void_p]
    lib.inferbit_model_max_context.restype = c_int
    lib.inferbit_model_max_context.argtypes = [c_void_p]
    lib.inferbit_model_default_bits.restype = c_int
    lib.inferbit_model_default_bits.argtypes = [c_void_p]
    lib.inferbit_model_weight_memory.restype = c_size_t
    lib.inferbit_model_weight_memory.argtypes = [c_void_p]
    lib.inferbit_model_kv_memory.restype = c_size_t
    lib.inferbit_model_kv_memory.argtypes = [c_void_p]
    lib.inferbit_model_total_memory.restype = c_size_t
    lib.inferbit_model_total_memory.argtypes = [c_void_p]

    # Speculative
    lib.inferbit_set_draft_model.restype = None
    lib.inferbit_set_draft_model.argtypes = [c_void_p, c_void_p, c_int]
    lib.inferbit_unset_draft_model.restype = None
    lib.inferbit_unset_draft_model.argtypes = [c_void_p]
    lib.inferbit_set_prompt_lookup.restype = None
    lib.inferbit_set_prompt_lookup.argtypes = [c_void_p, c_int, c_int]

    # Convert
    lib.inferbit_default_convert_config.restype = ConvertConfig
    lib.inferbit_default_convert_config.argtypes = []
    lib.inferbit_detect_format.restype = c_int
    lib.inferbit_detect_format.argtypes = [c_char_p]
    lib.inferbit_convert.restype = c_int
    lib.inferbit_convert.argtypes = [c_char_p, c_char_p, POINTER(ConvertConfig)]

    # ── PQ session (pq_decode.h) ──
    lib.ib_pq_session_open.restype = c_int
    lib.ib_pq_session_open.argtypes = [c_char_p, POINTER(c_void_p)]
    lib.ib_pq_session_close.restype = None
    lib.ib_pq_session_close.argtypes = [c_void_p]
    lib.ib_pq_session_set_default_policy.restype = c_int
    lib.ib_pq_session_set_default_policy.argtypes = [c_void_p, IbPqPolicy]
    lib.ib_pq_session_set_policy.restype = c_int
    lib.ib_pq_session_set_policy.argtypes = [c_void_p, c_char_p, IbPqPolicy]
    lib.ib_pq_session_matmul.restype = c_int
    lib.ib_pq_session_matmul.argtypes = [c_void_p, c_char_p,
                                           POINTER(c_float), POINTER(c_float)]
    lib.ib_pq_session_lm_head_topk.restype = c_int
    lib.ib_pq_session_lm_head_topk.argtypes = [c_void_p, c_char_p,
                                                 POINTER(c_float), c_int,
                                                 POINTER(c_float), POINTER(c_int32)]
    lib.ib_pq_session_tensor_shape.restype = c_int
    lib.ib_pq_session_tensor_shape.argtypes = [c_void_p, c_char_p,
                                                 POINTER(c_int), POINTER(c_int)]
    lib.ib_pq_session_tensor_count.restype = c_int
    lib.ib_pq_session_tensor_count.argtypes = [c_void_p]
    lib.ib_pq_session_tensor_name.restype = c_char_p
    lib.ib_pq_session_tensor_name.argtypes = [c_void_p, c_int]

    lib.ib_pq_session_raw_count.restype = c_int
    lib.ib_pq_session_raw_count.argtypes = [c_void_p]
    lib.ib_pq_session_raw_name.restype = c_char_p
    lib.ib_pq_session_raw_name.argtypes = [c_void_p, c_int]
    lib.ib_pq_session_raw_get.restype = c_int
    lib.ib_pq_session_raw_get.argtypes = [c_void_p, c_char_p,
                                           POINTER(c_void_p), POINTER(c_int),
                                           POINTER(c_int), POINTER(c_int)]
    lib.ib_pq_session_config_json.restype = c_char_p
    lib.ib_pq_session_config_json.argtypes = [c_void_p]

    lib.ib_pq_kv_cache_create.restype = c_int
    lib.ib_pq_kv_cache_create.argtypes = [c_void_p, c_int, POINTER(c_void_p)]
    lib.ib_pq_kv_cache_free.restype = None
    lib.ib_pq_kv_cache_free.argtypes = [c_void_p]
    lib.ib_pq_kv_cache_clear.restype = None
    lib.ib_pq_kv_cache_clear.argtypes = [c_void_p]
    lib.ib_pq_kv_cache_length.restype = c_int
    lib.ib_pq_kv_cache_length.argtypes = [c_void_p]
    lib.ib_pq_forward_step.restype = c_int
    lib.ib_pq_forward_step.argtypes = [c_void_p, c_void_p, c_int, c_int,
                                         POINTER(c_float)]
    lib.ib_pq_generate_greedy.restype = c_int
    lib.ib_pq_generate_greedy.argtypes = [c_void_p, c_void_p,
                                            POINTER(c_int), c_int, c_int, c_int,
                                            POINTER(c_int), POINTER(c_int),
                                            c_void_p, c_void_p]
    lib.ib_pq_generate_sample.restype = c_int
    lib.ib_pq_generate_sample.argtypes = [c_void_p, c_void_p,
                                            POINTER(c_int), c_int, c_int, c_int,
                                            IbPqSampleParams,
                                            POINTER(c_int), POINTER(c_int),
                                            c_void_p, c_void_p]
