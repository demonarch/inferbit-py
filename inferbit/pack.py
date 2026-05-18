"""Distribution-time .ibf compression (Stage 5i, docs/v2/00_CORRECTION.md).

Thin wrappers over libinferbit's `inferbit_pack` / `inferbit_unpack`.
These are *out-of-band* utilities — the runtime loader still mmaps a
plain `.ibf`. Compress at distribution; decompress once at install (or
before `inferbit_load`). No per-access decompression, ever.

If libinferbit was built without zstd, both helpers raise RuntimeError
with the underlying ``inferbit_last_error()`` message.
"""

from __future__ import annotations

import os
from typing import Optional

from ._ffi import _get_lib


DEFAULT_LEVEL = 19  # high-compression default; matches the C-side fallback.


def pack(source: str, output: Optional[str] = None, level: int = DEFAULT_LEVEL) -> str:
    """Compress an ``.ibf`` to ``.ibf.zst`` for distribution.

    Args:
        source: Path to the input ``.ibf`` file.
        output: Output path. Defaults to ``source + ".zst"``.
        level: Zstd compression level (1..22). 19 is the default;
            higher = smaller + slower at pack time, no difference at
            unpack time.

    Returns:
        The output path.
    """
    if output is None:
        output = source + ".zst"

    lib = _get_lib()
    rc = lib.inferbit_pack(source.encode(), output.encode(), int(level))
    if rc != 0:
        err = lib.inferbit_last_error()
        msg = err.decode() if err else "unknown error"
        raise RuntimeError(f"inferbit_pack failed: {msg}")
    return output


def unpack(source: str, output: Optional[str] = None) -> str:
    """Decompress an ``.ibf.zst`` back to ``.ibf``.

    Args:
        source: Path to the input ``.ibf.zst`` file.
        output: Output path. Defaults to ``source`` with the trailing
            ``.zst`` removed.

    Returns:
        The output path.
    """
    if output is None:
        if source.endswith(".zst"):
            output = source[: -len(".zst")]
        else:
            # No .zst suffix to strip — refuse to overwrite the input.
            output = source + ".ibf"

    lib = _get_lib()
    rc = lib.inferbit_unpack(source.encode(), output.encode())
    if rc != 0:
        err = lib.inferbit_last_error()
        msg = err.decode() if err else "unknown error"
        raise RuntimeError(f"inferbit_unpack failed: {msg}")
    return output


__all__ = ["pack", "unpack", "DEFAULT_LEVEL"]
