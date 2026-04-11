#!/usr/bin/env python3
"""
Build script: compile libinferbit and bundle into the Python package.

Usage:
    python build_wheel.py          # Build library + wheel
    python build_wheel.py --skip-build  # Wheel only (library already built)
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LIBINFERBIT_DIR = ROOT.parent / "libinferbit"
INFERBIT_PKG = ROOT / "inferbit"


def lib_name():
    s = platform.system()
    if s == "Darwin":
        return "libinferbit.dylib"
    elif s == "Linux":
        return "libinferbit.so"
    elif s == "Windows":
        return "inferbit.dll"
    raise RuntimeError(f"Unsupported platform: {s}")


def build_libinferbit():
    print("Building libinferbit...")
    build_dir = LIBINFERBIT_DIR / "build"

    subprocess.run(
        ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", str(LIBINFERBIT_DIR)],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir)],
        check=True,
    )

    lib = build_dir / lib_name()
    if not lib.exists():
        raise RuntimeError(f"Build succeeded but {lib} not found")

    return lib


def bundle_library(lib_path):
    dest = INFERBIT_PKG / lib_path.name
    print(f"Bundling {lib_path.name} into {dest}")
    shutil.copy2(str(lib_path), str(dest))


def build_wheel():
    print("Building wheel...")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", str(ROOT)],
        check=True,
    )


def main():
    skip_build = "--skip-build" in sys.argv

    if not skip_build:
        lib = build_libinferbit()
    else:
        lib = LIBINFERBIT_DIR / "build" / lib_name()
        if not lib.exists():
            print(f"ERROR: {lib} not found. Run without --skip-build first.")
            sys.exit(1)

    bundle_library(lib)

    # Ensure the library is included in the package
    manifest = ROOT / "MANIFEST.in"
    manifest_line = f"include inferbit/{lib_name()}\n"
    if not manifest.exists() or manifest_line not in manifest.read_text():
        with open(manifest, "a") as f:
            f.write(manifest_line)

    build_wheel()

    # Show output
    dist = ROOT / "dist"
    if dist.exists():
        for whl in dist.glob("*.whl"):
            size = whl.stat().st_size / 1024 / 1024
            print(f"\nBuilt: {whl.name} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
