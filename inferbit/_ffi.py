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


# Stage 5b — per-tensor class taxonomy (mirrors inferbit_tensor_class
# in include/inferbit.h). Used to index per_class_format[] and
# per_class_residency[].
INFERBIT_TENSOR_CLASS_FFN_GATE = 0
INFERBIT_TENSOR_CLASS_FFN_UP   = 1
INFERBIT_TENSOR_CLASS_FFN_DOWN = 2
INFERBIT_TENSOR_CLASS_ATTN_Q   = 3
INFERBIT_TENSOR_CLASS_ATTN_K   = 4
INFERBIT_TENSOR_CLASS_ATTN_V   = 5
INFERBIT_TENSOR_CLASS_ATTN_O   = 6
INFERBIT_TENSOR_CLASS_EMBED    = 7
INFERBIT_TENSOR_CLASS_LM_HEAD  = 8
INFERBIT_TENSOR_CLASS_COUNT    = 9


# Stage 5c — residency hint enum (mirrors inferbit_residency in
# include/inferbit.h).
INFERBIT_RESIDENCY_AUTO  = 0
INFERBIT_RESIDENCY_RAM   = 1
INFERBIT_RESIDENCY_DRIVE = 2


class ConvertConfig(Structure):
    # Field order MUST exactly match `inferbit_convert_config` in
    # modules/libinferbit/include/inferbit.h (ctypes Structure is
    # positional). Any drift here writes values into the wrong slots.
    _fields_ = [
        ("default_bits", c_int),
        ("sensitive_bits", c_int),
        ("sparsity", c_float),
        ("block_size", c_int),
        ("kv_bits", c_int),
        ("threads", c_int),
        ("progress", c_void_p),       # Function pointer — set separately
        ("progress_ctx", c_void_p),
        # inferbit_convert_format enum (see include/inferbit.h).
        ("format", c_int),
        # Stage 3a — post-hoc Mixture-of-Mini-Experts. Row-slice expert
        # count for FFN tensors. Default 1 = no MoME.
        ("mome_experts", c_int),
        # Stage 5b — per-tensor-class format override. Index by
        # INFERBIT_TENSOR_CLASS_*. Entry == 0 means "use cfg.format".
        ("per_class_format", c_int * INFERBIT_TENSOR_CLASS_COUNT),
        # Stage 5c — per-tensor-class residency hint. Index by
        # INFERBIT_TENSOR_CLASS_*. Entry == 0 means AUTO (loader picks).
        ("per_class_residency", c_int * INFERBIT_TENSOR_CLASS_COUNT),
        # Stage 5k — lower-precision row/codebook scales.
        # 0 = legacy fp16/fp16 (default). 2 = int8 row + fp8 cb_scale.
        ("scale_precision", c_int),
        # Stage 5j — codebook + scale dedup. 0 = off (default).
        # 1 = emit identity pool_id mapping (scaffolding).
        ("codebook_dedup", c_int),
    ]


# inferbit_convert_format enum mirrors include/inferbit.h.
INFERBIT_CONVERT_INT4 = 0
INFERBIT_CONVERT_PQV2_FLAT = 1
INFERBIT_CONVERT_PQV2_PYRAMID = 2


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
    lib.inferbit_config_set_kv_window.argtypes = [c_void_p, c_int]

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

    # Hidden-state capture (doc 36 phase 4.1)
    lib.inferbit_forward_with_hiddens.restype = c_int
    lib.inferbit_forward_with_hiddens.argtypes = [
        c_void_p,            # model
        POINTER(c_int32),    # tokens
        c_int,               # n_tokens
        POINTER(c_int),      # layer_ids
        c_int,               # n_layer_ids
        POINTER(c_float),    # hiddens_out [n_layer_ids][n_tokens][hidden]
        POINTER(c_float),    # logits_out  [n_tokens][vocab]
    ]
    lib.inferbit_build_target_layer_ids.restype = c_int
    lib.inferbit_build_target_layer_ids.argtypes = [
        c_int,               # n_target_layers
        c_int,               # n_draft_layers
        POINTER(c_int),      # out_ids
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

    # Distribution-time compression (Stage 5i — docs/v2/00_CORRECTION.md).
    # Both return 0 on success, -1 on failure (use inferbit_last_error()).
    lib.inferbit_pack.restype = c_int
    lib.inferbit_pack.argtypes = [c_char_p, c_char_p, c_int]
    lib.inferbit_unpack.restype = c_int
    lib.inferbit_unpack.argtypes = [c_char_p, c_char_p]

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
