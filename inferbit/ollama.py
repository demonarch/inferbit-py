"""Resolve Ollama model names to local GGUF file paths.

Ollama stores models at:
  ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>
  ~/.ollama/models/blobs/sha256-<hash>

The manifest JSON contains layer entries; the one with
mediaType "application/vnd.ollama.image.model" is the GGUF weights.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def ollama_models_dir() -> Path:
    """Return the Ollama models directory."""
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def resolve_ollama_model(model_spec: str) -> Optional[str]:
    """
    Resolve an Ollama model spec to a local GGUF file path.

    Args:
        model_spec: Model name like "llama3:8b", "mistral:latest",
                    or "ollama://llama3:8b"

    Returns:
        Absolute path to the GGUF blob file, or None if not found.
    """
    # Strip ollama:// prefix if present
    if model_spec.startswith("ollama://"):
        model_spec = model_spec[len("ollama://"):]

    # Parse model:tag
    if ":" in model_spec:
        model_name, tag = model_spec.rsplit(":", 1)
    else:
        model_name = model_spec
        tag = "latest"

    # Handle namespace (e.g., "library/llama3" vs just "llama3")
    if "/" not in model_name:
        namespace = "library"
        name = model_name
    else:
        namespace, name = model_name.split("/", 1)

    models_dir = ollama_models_dir()
    manifest_path = (
        models_dir / "manifests" / "registry.ollama.ai" / namespace / name / tag
    )

    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Find the model layer (GGUF weights)
    layers = manifest.get("layers", [])
    for layer in layers:
        media_type = layer.get("mediaType", "")
        if media_type == "application/vnd.ollama.image.model":
            digest = layer.get("digest", "")
            if digest:
                # Digest format: "sha256:abc123..."
                blob_name = digest.replace(":", "-")
                blob_path = models_dir / "blobs" / blob_name
                if blob_path.exists():
                    return str(blob_path)
            break

    return None


def list_ollama_models() -> list[dict]:
    """
    List locally available Ollama models.

    Returns:
        List of dicts with 'name', 'tag', and 'path' keys.
    """
    models_dir = ollama_models_dir()
    manifests_dir = models_dir / "manifests" / "registry.ollama.ai" / "library"

    if not manifests_dir.exists():
        return []

    results = []
    for model_dir in sorted(manifests_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for tag_file in sorted(model_dir.iterdir()):
            if tag_file.is_dir():
                continue
            tag = tag_file.name
            gguf_path = resolve_ollama_model(f"{model_name}:{tag}")
            if gguf_path:
                results.append({
                    "name": model_name,
                    "tag": tag,
                    "spec": f"{model_name}:{tag}",
                    "path": gguf_path,
                })

    return results
