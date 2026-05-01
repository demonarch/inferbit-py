"""Thin Python shell over ib_pq_session_* in libinferbit.

All dispatch decisions (which matmul variant to use for which tensor)
live in C. This module just owns the session handle's lifetime and
gives idiomatic Python access.
"""
from __future__ import annotations

import ctypes
import json as _json
from typing import Optional

import numpy as np

from . import _ffi


_RAW_DTYPE_FROM_INT = {
    0: np.float32,
    1: np.float16,
    2: np.int32,
    3: np.int16,
    4: np.int8,
    5: np.uint8,
}


class Session:
    """One IBF + cache fleet + per-tensor policies.

    Open once at model load, call matmul-by-name from the forward
    loop. Codebook decode, inner_cols build, transposed-indices, and
    optional INT8 quant happen exactly once at open time.
    """

    def __init__(self, ibf_path: str):
        self._lib = _ffi._get_lib()
        self._handle = ctypes.c_void_p()
        rc = self._lib.ib_pq_session_open(
            ibf_path.encode(), ctypes.byref(self._handle)
        )
        if rc != 0 or not self._handle:
            raise RuntimeError(f"ib_pq_session_open failed (rc={rc}, path={ibf_path})")
        self._n = self._lib.ib_pq_session_tensor_count(self._handle)

    def close(self) -> None:
        if self._handle:
            self._lib.ib_pq_session_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self._n

    def names(self) -> list[str]:
        out = []
        for i in range(self._n):
            p = self._lib.ib_pq_session_tensor_name(self._handle, i)
            out.append(p.decode() if p else "")
        return out

    def shape(self, name: str) -> tuple[int, int]:
        M, N = ctypes.c_int(), ctypes.c_int()
        rc = self._lib.ib_pq_session_tensor_shape(
            self._handle, name.encode(), ctypes.byref(M), ctypes.byref(N)
        )
        if rc != 0:
            raise KeyError(name)
        return M.value, N.value

    def set_default_policy(
        self,
        variant: int = _ffi.VARIANT_STREAMING,
        skip_threshold: float = 0.0,
        act_threshold: float = 0.0,
    ) -> None:
        p = _ffi.IbPqPolicy(variant, float(skip_threshold), float(act_threshold))
        rc = self._lib.ib_pq_session_set_default_policy(self._handle, p)
        if rc != 0:
            raise RuntimeError(f"set_default_policy rc={rc}")

    def set_policy(
        self,
        name: str,
        variant: int,
        skip_threshold: float = 0.0,
        act_threshold: float = 0.0,
    ) -> None:
        p = _ffi.IbPqPolicy(variant, float(skip_threshold), float(act_threshold))
        rc = self._lib.ib_pq_session_set_policy(self._handle, name.encode(), p)
        if rc != 0:
            raise KeyError(name)

    def matmul(self, name: str, x: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute weights[name] @ x. C picks the kernel based on policy."""
        if x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x, dtype=np.float32)
        M, N = self.shape(name)
        if x.shape[-1] != N:
            raise ValueError(f"{name}: x.shape[-1]={x.shape[-1]} but N={N}")
        if out is None:
            out = np.zeros(M, dtype=np.float32)
        elif out.dtype != np.float32 or out.size < M or not out.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous float32 with at least M elements")
        rc = self._lib.ib_pq_session_matmul(
            self._handle, name.encode(),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"matmul {name} rc={rc}")
        return out

    def raw_names(self) -> list[str]:
        n = self._lib.ib_pq_session_raw_count(self._handle)
        out = []
        for i in range(n):
            p = self._lib.ib_pq_session_raw_name(self._handle, i)
            out.append(p.decode() if p else "")
        return out

    def raw(self, name: str) -> np.ndarray:
        """Zero-copy view of a raw tensor stored in the IBF.

        The returned array borrows from the session — do NOT use after
        session.close(). Make a .copy() if you need it to outlive.
        """
        data = ctypes.c_void_p()
        dtype = ctypes.c_int()
        ndim = ctypes.c_int()
        shape = (ctypes.c_int * 4)()
        rc = self._lib.ib_pq_session_raw_get(
            self._handle, name.encode(),
            ctypes.byref(data), ctypes.byref(dtype),
            shape, ctypes.byref(ndim),
        )
        if rc != 0:
            raise KeyError(f"raw tensor not found: {name}")
        np_dtype = np.dtype(_RAW_DTYPE_FROM_INT[dtype.value])
        shp = tuple(shape[d] for d in range(ndim.value))
        nelem = 1
        for d in shp:
            nelem *= d
        nbytes = nelem * np_dtype.itemsize
        buf = (ctypes.c_uint8 * nbytes).from_address(data.value)
        return np.frombuffer(buf, dtype=np_dtype).reshape(shp)

    def config(self) -> dict:
        p = self._lib.ib_pq_session_config_json(self._handle)
        if not p:
            return {}
        return _json.loads(p.decode())

    def lm_head_topk(
        self,
        name: str,
        x: np.ndarray,
        k_top: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-K logits + token ids for an lm_head pyramid tensor."""
        if x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x, dtype=np.float32)
        logits = np.zeros(k_top, dtype=np.float32)
        ids = np.zeros(k_top, dtype=np.int32)
        rc = self._lib.ib_pq_session_lm_head_topk(
            self._handle, name.encode(),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(k_top),
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if rc != 0:
            raise RuntimeError(f"lm_head_topk {name} rc={rc}")
        return logits, ids


# Re-export variant constants
STREAMING = _ffi.VARIANT_STREAMING
L1_ONLY = _ffi.VARIANT_L1_ONLY
L2SKIP = _ffi.VARIANT_L2SKIP
SPARSE = _ffi.VARIANT_SPARSE
INT8 = _ffi.VARIANT_INT8
