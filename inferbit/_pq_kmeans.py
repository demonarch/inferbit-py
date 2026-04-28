"""ctypes binding to libinferbit's ib_kmeans_fit / ib_kmeans_assign.

Used by the PQ converter as a faster, parallel alternative to
sklearn.cluster.KMeans. Only depends on numpy + libinferbit.so/dylib.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from ._binary import find_library


_LIB = None
_POOL_CACHE: dict[int, ctypes.c_void_p] = {}


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    _LIB = ctypes.CDLL(find_library())

    # ib_thread_pool* ib_pool_create(int n_threads)
    _LIB.ib_pool_create.restype = ctypes.c_void_p
    _LIB.ib_pool_create.argtypes = [ctypes.c_int]
    # void ib_pool_destroy(ib_thread_pool*)
    _LIB.ib_pool_destroy.restype = None
    _LIB.ib_pool_destroy.argtypes = [ctypes.c_void_p]

    # int ib_kmeans_fit(const float* X, int N, const ib_kmeans_config* cfg,
    #                   float* centers_out, int32_t* indices_out, double* inertia_out)
    _LIB.ib_kmeans_fit.restype = ctypes.c_int
    _LIB.ib_kmeans_fit.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
    ]

    # int ib_kmeans_assign(const float* X, int N, int D,
    #                      const float* centers, int K,
    #                      int32_t* indices_out, ib_thread_pool*)
    _LIB.ib_kmeans_assign.restype = ctypes.c_int
    _LIB.ib_kmeans_assign.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p,
    ]
    return _LIB


# Mirror of `ib_kmeans_config` in pq_kmeans.h.
class _KMeansConfig(ctypes.Structure):
    _fields_ = [
        ("K", ctypes.c_int),
        ("D", ctypes.c_int),
        ("max_iter", ctypes.c_int),
        ("tol", ctypes.c_float),
        ("n_init", ctypes.c_int),
        ("subsample", ctypes.c_int),
        ("seed", ctypes.c_uint32),
        ("pool", ctypes.c_void_p),
    ]


def _get_pool(n_threads: int) -> ctypes.c_void_p:
    """Cache one pool per thread count for the process lifetime."""
    if n_threads <= 1:
        return ctypes.c_void_p(0)
    p = _POOL_CACHE.get(n_threads)
    if p is None:
        p = _lib().ib_pool_create(n_threads)
        _POOL_CACHE[n_threads] = p
    return ctypes.c_void_p(p)


def fit(X: np.ndarray, K: int, *, max_iter: int = 20, tol: float = 1e-4,
        n_init: int = 1, subsample: int = 0, seed: int = 0,
        n_threads: int = 0,
        return_indices: bool = False) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Fit K-means on X (N x D fp32). Returns (centers, indices_or_None, inertia).

    Mirrors sklearn.cluster.KMeans interface:
      - centers: [K, D] fp32
      - indices: [N] int32 (only if return_indices)
      - inertia: SSE objective at convergence
    """
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    N, D = X.shape
    if n_threads <= 0:
        n_threads = max(1, (os.cpu_count() or 1))

    centers = np.empty((K, D), dtype=np.float32)
    indices = np.empty(N, dtype=np.int32) if return_indices else None
    inertia = ctypes.c_double(0.0)

    cfg = _KMeansConfig(
        K=K, D=D, max_iter=max_iter, tol=tol,
        n_init=n_init, subsample=subsample, seed=seed,
        pool=_get_pool(n_threads),
    )

    rc = _lib().ib_kmeans_fit(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N,
        ctypes.byref(cfg),
        centers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if indices is not None
            else ctypes.POINTER(ctypes.c_int32)(),
        ctypes.byref(inertia),
    )
    if rc != 0:
        raise RuntimeError(f"ib_kmeans_fit failed rc={rc}")

    return centers, indices, float(inertia.value)


def assign(X: np.ndarray, centers: np.ndarray, *, n_threads: int = 0) -> np.ndarray:
    """Assign each row of X to the nearest center. Returns int32 [N]."""
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    if centers.dtype != np.float32:
        centers = centers.astype(np.float32, copy=False)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X)
    if not centers.flags["C_CONTIGUOUS"]:
        centers = np.ascontiguousarray(centers)
    N, D = X.shape
    K, D2 = centers.shape
    if D != D2:
        raise ValueError(f"X dim {D} != centers dim {D2}")
    if n_threads <= 0:
        n_threads = max(1, (os.cpu_count() or 1))

    indices = np.empty(N, dtype=np.int32)
    rc = _lib().ib_kmeans_assign(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, D,
        centers.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K,
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _get_pool(n_threads),
    )
    if rc != 0:
        raise RuntimeError(f"ib_kmeans_assign failed rc={rc}")
    return indices
