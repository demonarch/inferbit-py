"""IBF v5 — stacked 2D PQ pyramid tensor encoding.

Reference Python writer + reader. Spec: docs/26_IBF_V5_PQ_FORMAT.md.

This module is intentionally standalone (numpy-only). It writes a
single-tensor IBF v5 file given quantizer outputs, and reads it back
into the same arrays. Used to validate the schema and as the oracle
for the C decoder.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

IBF_MAGIC = b"INFERBIT"
IBF_VERSION = 5
IBF_PREAMBLE = 32
IBF_ALIGNMENT = 64


def _align(x: int, a: int = IBF_ALIGNMENT) -> int:
    return (x + a - 1) & ~(a - 1)


@dataclass
class PQTensor:
    """Decoded PQ tensor — what the writer takes in and the reader returns."""

    shape: tuple[int, int]
    G: int
    K: int
    n_levels: int
    rotate: bool

    codebook_l1_l: np.ndarray  # [K, G/2] fp16
    codebook_l1_r: np.ndarray  # [K, G/2] fp16
    indices_l1_l: np.ndarray   # [M, C] uint8
    indices_l1_r: np.ndarray   # [M, C] uint8

    row_scale: np.ndarray      # [M] fp16

    codebook_l2_l: Optional[np.ndarray] = None
    codebook_l2_r: Optional[np.ndarray] = None
    indices_l2_l:  Optional[np.ndarray] = None
    indices_l2_r:  Optional[np.ndarray] = None

    outlier_cols:    Optional[np.ndarray] = None  # [n_outlier] int32
    outlier_sidecar: Optional[np.ndarray] = None  # [M, n_outlier] int8
    outlier_scale:   Optional[np.ndarray] = None  # [n_outlier] fp16

    def format_str(self) -> str:
        has_outlier = self.outlier_cols is not None
        if self.n_levels == 1:
            return "pq2d_v1_l1"
        if self.n_levels == 2 and not has_outlier:
            return "pq2d_v1_l2"
        if self.n_levels == 2 and has_outlier:
            return "pq2d_v1_l1_l2"
        raise ValueError(f"unsupported (n_levels={self.n_levels}, outlier={has_outlier})")


def _check(t: PQTensor) -> None:
    M, N = t.shape
    n_outlier = 0 if t.outlier_cols is None else int(t.outlier_cols.shape[0])
    n_inner = N - n_outlier
    assert n_inner % t.G == 0, f"inner cols {n_inner} not multiple of G={t.G}"
    C = n_inner // t.G

    assert t.codebook_l1_l.shape == (t.K, t.G // 2) and t.codebook_l1_l.dtype == np.float16
    assert t.codebook_l1_r.shape == (t.K, t.G // 2) and t.codebook_l1_r.dtype == np.float16
    assert t.indices_l1_l.shape == (M, C) and t.indices_l1_l.dtype == np.uint8
    assert t.indices_l1_r.shape == (M, C) and t.indices_l1_r.dtype == np.uint8
    assert t.row_scale.shape == (M,) and t.row_scale.dtype == np.float16

    if t.n_levels == 2:
        for arr in (t.codebook_l2_l, t.codebook_l2_r, t.indices_l2_l, t.indices_l2_r):
            assert arr is not None, "n_levels=2 requires level-2 arrays"
        assert t.codebook_l2_l.shape == (t.K, t.G // 2) and t.codebook_l2_l.dtype == np.float16
        assert t.codebook_l2_r.shape == (t.K, t.G // 2) and t.codebook_l2_r.dtype == np.float16
        assert t.indices_l2_l.shape == (M, C) and t.indices_l2_l.dtype == np.uint8
        assert t.indices_l2_r.shape == (M, C) and t.indices_l2_r.dtype == np.uint8

    if t.outlier_cols is not None:
        assert t.outlier_sidecar is not None and t.outlier_scale is not None
        assert t.outlier_cols.dtype == np.int32 and t.outlier_cols.shape == (n_outlier,)
        assert t.outlier_sidecar.dtype == np.int8 and t.outlier_sidecar.shape == (M, n_outlier)
        assert t.outlier_scale.dtype == np.float16 and t.outlier_scale.shape == (n_outlier,)


def reconstruct(t: PQTensor) -> np.ndarray:
    """Reference reconstruction: PQTensor → fp16 W of shape [M, N].

    No FWHT applied here — that's the caller's job (rotation merges into
    the prior projection at runtime). When `t.rotate` is True the
    returned matrix is in the rotated basis.
    """
    _check(t)
    M, N = t.shape
    G = t.G
    n_outlier = 0 if t.outlier_cols is None else int(t.outlier_cols.shape[0])
    n_inner = N - n_outlier
    C = n_inner // G

    W = np.zeros((M, N), dtype=np.float16)

    inner_mask = np.ones(N, dtype=bool)
    if t.outlier_cols is not None:
        inner_mask[t.outlier_cols] = False
    inner_idx = np.flatnonzero(inner_mask)

    # Decode contract (see docs/26): all PQ arithmetic happens in fp32;
    # the only fp16 rounding step is the final cast to fp16 below.
    chunks = np.empty((M, C, G), dtype=np.float32)
    chunks[:, :, : G // 2] = t.codebook_l1_l[t.indices_l1_l].astype(np.float32)
    chunks[:, :, G // 2 :] = t.codebook_l1_r[t.indices_l1_r].astype(np.float32)
    if t.n_levels == 2:
        chunks[:, :, : G // 2] += t.codebook_l2_l[t.indices_l2_l].astype(np.float32)
        chunks[:, :, G // 2 :] += t.codebook_l2_r[t.indices_l2_r].astype(np.float32)
    chunks *= t.row_scale[:, None, None].astype(np.float32)

    W[:, inner_idx] = chunks.reshape(M, n_inner).astype(np.float16)

    if t.outlier_cols is not None:
        out = t.outlier_sidecar.astype(np.float32) * t.outlier_scale.astype(np.float32)[None, :]
        W[:, t.outlier_cols] = out.astype(np.float16)

    return W


def write_multi_tensor(path: str | Path, tensors: dict[str, PQTensor]) -> None:
    """Write a multi-tensor IBF v5 file.

    Each tensor's blocks are written as `<name>::<key>` in the JSON header
    so block keys remain unique. Block layout per tensor matches
    write_single_tensor; offsets are still relative to weight_data_start.
    """
    blocks: list[tuple[str, bytes]] = []
    tensor_order: list[str] = []

    def add(qkey: str, arr: np.ndarray) -> None:
        blocks.append((qkey, np.ascontiguousarray(arr).tobytes()))

    for name, t in tensors.items():
        _check(t)
        tensor_order.append(name)

        def keyed(k: str) -> str:
            return f"{name}::{k}"

        add(keyed("codebook_l1_l"), t.codebook_l1_l)
        add(keyed("codebook_l1_r"), t.codebook_l1_r)
        add(keyed("indices_l1_l"), t.indices_l1_l)
        add(keyed("indices_l1_r"), t.indices_l1_r)
        add(keyed("row_scale"), t.row_scale)
        if t.n_levels == 2:
            add(keyed("codebook_l2_l"), t.codebook_l2_l)
            add(keyed("codebook_l2_r"), t.codebook_l2_r)
            add(keyed("indices_l2_l"), t.indices_l2_l)
            add(keyed("indices_l2_r"), t.indices_l2_r)
        if t.outlier_cols is not None:
            add(keyed("outlier_cols"), t.outlier_cols)
            add(keyed("outlier_sidecar"), t.outlier_sidecar)
            add(keyed("outlier_scale"), t.outlier_scale)

    # Reserve a generous JSON header for multi-tensor models.
    json_reserve = max(32 * 1024, 2048 * len(tensor_order))
    weight_data_start = _align(IBF_PREAMBLE + json_reserve)

    offsets: dict[str, tuple[int, int]] = {}
    cur = weight_data_start
    for qkey, blob in blocks:
        cur = _align(cur)
        offsets[qkey] = (cur, len(blob))
        cur += len(blob)
    file_end = _align(cur)

    tensors_json: dict[str, dict] = {}
    for name, t in tensors.items():
        meta: dict = {
            "format": t.format_str(),
            "shape": list(t.shape),
            "G": t.G,
            "K": t.K,
            "n_levels": t.n_levels,
            "rotate": bool(t.rotate),
        }
        if t.outlier_cols is not None:
            meta["outlier"] = {"n_cols": int(t.outlier_cols.shape[0])}

        keys = ["codebook_l1_l", "codebook_l1_r", "indices_l1_l", "indices_l1_r", "row_scale"]
        if t.n_levels == 2:
            keys += ["codebook_l2_l", "codebook_l2_r", "indices_l2_l", "indices_l2_r"]
        if t.outlier_cols is not None:
            keys += ["outlier_cols", "outlier_sidecar", "outlier_scale"]
        for k in keys:
            off, sz = offsets[f"{name}::{k}"]
            meta[k] = {"offset": off - weight_data_start, "size": sz}
        tensors_json[name] = meta

    header = {
        "ibf_version": IBF_VERSION,
        "tensors": tensors_json,
        "weight_data_start": weight_data_start,
    }
    header_bytes = json.dumps(header).encode("utf-8")
    if len(header_bytes) > json_reserve:
        raise RuntimeError(f"json header {len(header_bytes)} > reserve {json_reserve}")

    with open(path, "wb") as f:
        f.write(IBF_MAGIC)
        f.write(struct.pack("<I", IBF_VERSION))
        f.write(struct.pack("<I", json_reserve))
        f.write(struct.pack("<I", 0))
        f.write(b"\x00" * 12)
        f.write(header_bytes)
        f.write(b"\x00" * (json_reserve - len(header_bytes)))
        last_end = weight_data_start
        for qkey, blob in blocks:
            off, sz = offsets[qkey]
            f.seek(off)
            f.write(blob)
            last_end = off + sz
        if file_end > last_end:
            f.seek(file_end - 1)
            f.write(b"\x00")


def read_multi_tensor(path: str | Path) -> dict[str, PQTensor]:
    """Read a multi-tensor IBF v5 file written by `write_multi_tensor`."""
    out: dict[str, PQTensor] = {}
    with open(path, "rb") as f:
        preamble = f.read(IBF_PREAMBLE)
        assert preamble[:8] == IBF_MAGIC
        version = struct.unpack("<I", preamble[8:12])[0]
        assert version == IBF_VERSION
        json_reserve = struct.unpack("<I", preamble[12:16])[0]
        header_bytes = f.read(json_reserve).rstrip(b"\x00")
        header = json.loads(header_bytes)
        weight_data_start = header["weight_data_start"]

        for name, m in header["tensors"].items():
            def load(key: str, dtype, shape) -> np.ndarray:
                off = m[key]["offset"] + weight_data_start
                sz = m[key]["size"]
                f.seek(off)
                buf = f.read(sz)
                return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

            M, N = m["shape"]
            G = m["G"]; K = m["K"]; n_levels = m["n_levels"]
            n_outlier = m.get("outlier", {}).get("n_cols", 0)
            n_inner = N - n_outlier
            assert n_inner % G == 0
            C = n_inner // G

            t = PQTensor(
                shape=(M, N), G=G, K=K, n_levels=n_levels, rotate=bool(m["rotate"]),
                codebook_l1_l=load("codebook_l1_l", np.float16, (K, G // 2)),
                codebook_l1_r=load("codebook_l1_r", np.float16, (K, G // 2)),
                indices_l1_l=load("indices_l1_l", np.uint8, (M, C)),
                indices_l1_r=load("indices_l1_r", np.uint8, (M, C)),
                row_scale=load("row_scale", np.float16, (M,)),
            )
            if n_levels == 2:
                t.codebook_l2_l = load("codebook_l2_l", np.float16, (K, G // 2))
                t.codebook_l2_r = load("codebook_l2_r", np.float16, (K, G // 2))
                t.indices_l2_l  = load("indices_l2_l", np.uint8, (M, C))
                t.indices_l2_r  = load("indices_l2_r", np.uint8, (M, C))
            if n_outlier > 0:
                t.outlier_cols    = load("outlier_cols", np.int32, (n_outlier,))
                t.outlier_sidecar = load("outlier_sidecar", np.int8, (M, n_outlier))
                t.outlier_scale   = load("outlier_scale", np.float16, (n_outlier,))
            out[name] = t
    return out


def write_single_tensor(path: str | Path, name: str, t: PQTensor) -> None:
    """Write a single-tensor IBF v5 file. Used for roundtrip tests."""
    _check(t)
    blocks: list[tuple[str, bytes]] = []

    def add(key: str, arr: np.ndarray) -> None:
        blocks.append((key, np.ascontiguousarray(arr).tobytes()))

    add("codebook_l1_l", t.codebook_l1_l)
    add("codebook_l1_r", t.codebook_l1_r)
    add("indices_l1_l", t.indices_l1_l)
    add("indices_l1_r", t.indices_l1_r)
    add("row_scale", t.row_scale)

    if t.n_levels == 2:
        add("codebook_l2_l", t.codebook_l2_l)
        add("codebook_l2_r", t.codebook_l2_r)
        add("indices_l2_l", t.indices_l2_l)
        add("indices_l2_r", t.indices_l2_r)

    if t.outlier_cols is not None:
        add("outlier_cols", t.outlier_cols)
        add("outlier_sidecar", t.outlier_sidecar)
        add("outlier_scale", t.outlier_scale)

    json_reserve = 32 * 1024
    weight_data_start = _align(IBF_PREAMBLE + json_reserve)

    offsets: dict[str, tuple[int, int]] = {}
    cur = weight_data_start
    for key, blob in blocks:
        cur = _align(cur)
        offsets[key] = (cur, len(blob))
        cur += len(blob)
    file_end = _align(cur)

    tensor_meta: dict = {
        "format": t.format_str(),
        "shape": list(t.shape),
        "G": t.G,
        "K": t.K,
        "n_levels": t.n_levels,
        "rotate": bool(t.rotate),
    }
    if t.outlier_cols is not None:
        tensor_meta["outlier"] = {"n_cols": int(t.outlier_cols.shape[0])}

    for key, (off, sz) in offsets.items():
        tensor_meta[key] = {"offset": off - weight_data_start, "size": sz}

    header = {
        "ibf_version": IBF_VERSION,
        "tensors": {name: tensor_meta},
        "weight_data_start": weight_data_start,
    }
    header_bytes = json.dumps(header).encode("utf-8")
    if len(header_bytes) > json_reserve:
        raise RuntimeError(f"json header {len(header_bytes)} > reserve {json_reserve}")

    with open(path, "wb") as f:
        f.write(IBF_MAGIC)
        f.write(struct.pack("<I", IBF_VERSION))
        f.write(struct.pack("<I", json_reserve))
        f.write(struct.pack("<I", 0))      # flags
        f.write(b"\x00" * 12)              # reserved
        f.write(header_bytes)
        f.write(b"\x00" * (json_reserve - len(header_bytes)))
        # weight blocks
        last_end = weight_data_start
        for key, blob in blocks:
            off, sz = offsets[key]
            f.seek(off)
            f.write(blob)
            last_end = off + sz
        # Pad to file_end only if alignment leaves a tail
        if file_end > last_end:
            f.seek(file_end - 1)
            f.write(b"\x00")


def read_single_tensor(path: str | Path) -> tuple[str, PQTensor]:
    """Read a single-tensor IBF v5 file written by `write_single_tensor`."""
    with open(path, "rb") as f:
        preamble = f.read(IBF_PREAMBLE)
        assert preamble[:8] == IBF_MAGIC, "bad magic"
        version = struct.unpack("<I", preamble[8:12])[0]
        assert version == IBF_VERSION, f"version {version} != {IBF_VERSION}"
        json_reserve = struct.unpack("<I", preamble[12:16])[0]
        header_bytes = f.read(json_reserve).rstrip(b"\x00")
        header = json.loads(header_bytes)
        weight_data_start = header["weight_data_start"]

        names = list(header["tensors"].keys())
        assert len(names) == 1, "this reader is single-tensor"
        name = names[0]
        m = header["tensors"][name]

        def load(key: str, dtype, shape) -> np.ndarray:
            off = m[key]["offset"] + weight_data_start
            sz = m[key]["size"]
            f.seek(off)
            buf = f.read(sz)
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

        M, N = m["shape"]
        G = m["G"]; K = m["K"]; n_levels = m["n_levels"]
        n_outlier = m.get("outlier", {}).get("n_cols", 0)
        n_inner = N - n_outlier
        assert n_inner % G == 0
        C = n_inner // G

        t = PQTensor(
            shape=(M, N), G=G, K=K, n_levels=n_levels, rotate=bool(m["rotate"]),
            codebook_l1_l=load("codebook_l1_l", np.float16, (K, G // 2)),
            codebook_l1_r=load("codebook_l1_r", np.float16, (K, G // 2)),
            indices_l1_l=load("indices_l1_l", np.uint8, (M, C)),
            indices_l1_r=load("indices_l1_r", np.uint8, (M, C)),
            row_scale=load("row_scale", np.float16, (M,)),
        )

        if n_levels == 2:
            t.codebook_l2_l = load("codebook_l2_l", np.float16, (K, G // 2))
            t.codebook_l2_r = load("codebook_l2_r", np.float16, (K, G // 2))
            t.indices_l2_l  = load("indices_l2_l", np.uint8, (M, C))
            t.indices_l2_r  = load("indices_l2_r", np.uint8, (M, C))

        if n_outlier > 0:
            t.outlier_cols    = load("outlier_cols", np.int32, (n_outlier,))
            t.outlier_sidecar = load("outlier_sidecar", np.int8, (M, n_outlier))
            t.outlier_scale   = load("outlier_scale", np.float16, (n_outlier,))

    return name, t
