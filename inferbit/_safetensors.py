"""ctypes binding for libinferbit's multi-shard safetensors loader.

Wraps `ib_st_multi_open / find / tensor_data / tensor_shape / tensor_dtype`
so the converter can stream-read tensors from a model directory without
ever holding the full FP16 model in process memory. The shards are
mmap'd by the C side; the OS evicts pages we're not touching.

Used by the streaming converter to do 70B-on-16GB-consumer.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from ._binary import find_library


_LIB = None


_DTYPE_MAP = {
    "F32":  (np.float32, 4),
    "F16":  (np.float16, 2),
    "BF16": ("bfloat16", 2),  # numpy 2 doesn't have native bf16; we'll cast
    "I32":  (np.int32, 4),
    "I16":  (np.int16, 2),
    "I8":   (np.int8, 1),
    "U8":   (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    _LIB = ctypes.CDLL(find_library())

    # Single-shard
    _LIB.ib_st_open.restype = ctypes.c_void_p
    _LIB.ib_st_open.argtypes = [ctypes.c_char_p]
    _LIB.ib_st_close.restype = None
    _LIB.ib_st_close.argtypes = [ctypes.c_void_p]
    _LIB.ib_st_find.restype = ctypes.c_int
    _LIB.ib_st_find.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    _LIB.ib_st_tensor_data.restype = ctypes.c_void_p
    _LIB.ib_st_tensor_data.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _LIB.ib_st_tensor_dtype_at.restype = ctypes.c_char_p
    _LIB.ib_st_tensor_dtype_at.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _LIB.ib_st_tensor_shape_at.restype = ctypes.c_int
    _LIB.ib_st_tensor_shape_at.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    # Multi-shard
    _LIB.ib_st_multi_open.restype = ctypes.c_void_p
    _LIB.ib_st_multi_open.argtypes = [ctypes.c_char_p]
    _LIB.ib_st_multi_close.restype = None
    _LIB.ib_st_multi_close.argtypes = [ctypes.c_void_p]
    _LIB.ib_st_multi_find.restype = ctypes.c_int
    _LIB.ib_st_multi_find.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    _LIB.ib_st_multi_tensor_data.restype = ctypes.c_void_p
    _LIB.ib_st_multi_tensor_data.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _LIB.ib_st_multi_tensor_dtype.restype = ctypes.c_char_p
    _LIB.ib_st_multi_tensor_dtype.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _LIB.ib_st_multi_tensor_shape.restype = ctypes.c_int
    _LIB.ib_st_multi_tensor_shape.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    _LIB.ib_st_multi_num_shards.restype = ctypes.c_int
    _LIB.ib_st_multi_num_shards.argtypes = [ctypes.c_void_p]
    return _LIB


class SingleShardModel:
    """Lazy single-shard safetensors reader (mmap'd via libinferbit).

    Used by the incremental converter that downloads / processes / deletes
    one shard at a time. Mirrors MultiShardModel's get_tensor signature.
    """

    def __init__(self, file_path: str | Path):
        self._path = str(file_path)
        self._handle = _lib().ib_st_open(self._path.encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"ib_st_open failed for {self._path}")

    def close(self) -> None:
        if self._handle:
            _lib().ib_st_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_tensor(self, name: str, max_ndim: int = 4) -> np.ndarray:
        idx = _lib().ib_st_find(self._handle, name.encode("utf-8"))
        if idx < 0:
            raise KeyError(f"tensor not found: {name}")

        dtype_str = _lib().ib_st_tensor_dtype_at(self._handle, idx)
        if not dtype_str:
            raise RuntimeError(f"no dtype for {name}")
        dtype_name = dtype_str.decode("ascii")
        if dtype_name not in _DTYPE_MAP:
            raise NotImplementedError(f"unsupported dtype: {dtype_name}")
        np_dtype, _ = _DTYPE_MAP[dtype_name]

        shape = []
        for d in range(max_ndim):
            sz = _lib().ib_st_tensor_shape_at(self._handle, idx, d)
            if sz <= 0:
                break
            shape.append(sz)
        if not shape:
            raise RuntimeError(f"empty shape for {name}")

        n = 1
        for s in shape:
            n *= s

        ptr = _lib().ib_st_tensor_data(self._handle, idx)
        if not ptr:
            raise RuntimeError(f"null data for {name}")

        if dtype_name == "BF16":
            buf = (ctypes.c_uint16 * n).from_address(ptr)
            arr = np.frombuffer(buf, dtype=np.uint16).reshape(shape).copy()
            arr_fp32 = (arr.astype(np.uint32) << 16).view(np.float32)
            return arr_fp32

        ArrayType = (ctypes.c_byte * (n * np.dtype(np_dtype).itemsize))
        buf = ArrayType.from_address(ptr)
        arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape)
        return arr


class MultiShardModel:
    """Lazy multi-shard safetensors reader. Opens all shards as mmap; the
    OS pages in only what we touch."""

    def __init__(self, dir_path: str | Path):
        self._dir = str(dir_path)
        self._handle = _lib().ib_st_multi_open(self._dir.encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"ib_st_multi_open failed for {self._dir}")

    def close(self) -> None:
        if self._handle:
            _lib().ib_st_multi_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def num_shards(self) -> int:
        return _lib().ib_st_multi_num_shards(self._handle)

    def get_tensor(self, name: str, max_ndim: int = 4) -> np.ndarray:
        """Return a numpy array view over the mmap'd tensor data.

        Lifetime: the returned array shares memory with the mmap. Drop
        all references to it before calling close().
        """
        shard = ctypes.c_int(-1)
        tensor = ctypes.c_int(-1)
        rc = _lib().ib_st_multi_find(
            self._handle, name.encode("utf-8"),
            ctypes.byref(shard), ctypes.byref(tensor),
        )
        if rc != 0:
            raise KeyError(f"tensor not found: {name}")

        dtype_str = _lib().ib_st_multi_tensor_dtype(self._handle, shard, tensor)
        if not dtype_str:
            raise RuntimeError(f"no dtype for {name}")
        dtype_name = dtype_str.decode("ascii")

        if dtype_name not in _DTYPE_MAP:
            raise NotImplementedError(f"unsupported dtype: {dtype_name}")
        np_dtype, _ = _DTYPE_MAP[dtype_name]

        # Read shape until we hit zero (loader returns 0 past ndim).
        shape = []
        for d in range(max_ndim):
            sz = _lib().ib_st_multi_tensor_shape(self._handle, shard, tensor, d)
            if sz <= 0:
                break
            shape.append(sz)
        if not shape:
            raise RuntimeError(f"empty shape for {name}")

        n = 1
        for s in shape:
            n *= s

        ptr = _lib().ib_st_multi_tensor_data(self._handle, shard, tensor)
        if not ptr:
            raise RuntimeError(f"null data for {name}")

        # bf16: numpy can't natively view as bf16. Read as uint16 then
        # caller can convert with a manual unpack. For our converter we
        # only care about F16 / F32 weights.
        if dtype_name == "BF16":
            buf = (ctypes.c_uint16 * n).from_address(ptr)
            arr = np.frombuffer(buf, dtype=np.uint16).reshape(shape).copy()
            # bf16 → fp32 via shifting up to fp32 high bits
            arr_fp32 = (arr.astype(np.uint32) << 16).view(np.float32)
            return arr_fp32

        # Zero-copy view over mmap'd memory.
        ArrayType = (ctypes.c_byte * (n * np.dtype(np_dtype).itemsize))
        buf = ArrayType.from_address(ptr)
        arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape)
        # Returned view aliases mmap memory. Caller should consume it
        # (e.g., by quantizing to fp32 numpy then dropping the view).
        return arr
