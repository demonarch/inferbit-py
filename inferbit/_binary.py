"""Locate the libinferbit shared library."""

import os
import sys
import platform
from pathlib import Path


def _lib_name() -> str:
    s = platform.system()
    if s == "Linux":
        return "libinferbit.so"
    elif s == "Darwin":
        return "libinferbit.dylib"
    elif s == "Windows":
        return "inferbit.dll"
    raise RuntimeError(f"Unsupported platform: {s}")


def find_library() -> str:
    name = _lib_name()

    # 1. INFERBIT_LIB_PATH environment variable
    env = os.environ.get("INFERBIT_LIB_PATH")
    if env and os.path.isfile(env):
        return env

    # 2. Next to this package (bundled in wheel)
    pkg_dir = Path(__file__).parent
    bundled = pkg_dir / name
    if bundled.exists():
        return str(bundled)

    # 3. In the build directory (development)
    # Submodule: inferbit-py/libinferbit/build/
    repo_dir = pkg_dir.parent  # inferbit-py/
    build_lib = repo_dir / "libinferbit" / "build" / name
    if build_lib.exists():
        return str(build_lib)

    # Parent monorepo: modules/libinferbit/build/
    modules_dir = repo_dir.parent
    build_lib2 = modules_dir / "libinferbit" / "build" / name
    if build_lib2.exists():
        return str(build_lib2)

    # Monorepo root: inferbit/modules/libinferbit/build/
    repo_root = modules_dir.parent
    build_lib3 = repo_root / "modules" / "libinferbit" / "build" / name
    if build_lib3.exists():
        return str(build_lib3)

    # 4. System library path
    for d in ["/usr/local/lib", "/usr/lib"]:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p

    raise RuntimeError(
        f"Could not find {name}. Set INFERBIT_LIB_PATH or install libinferbit.\n"
        f"For development, build with: cd modules/libinferbit && cmake -B build && cmake --build build"
    )
