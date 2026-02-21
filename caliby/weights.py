"""Utilities for downloading model weights from HuggingFace."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

HF_REPO_ID = "ProteinDesignLab/caliby-weights"

# Registry: model name → relative path within HF repo / weights dir
MODEL_REGISTRY = {
    "caliby": "caliby/caliby.ckpt",
    "soluble_caliby": "caliby/soluble_caliby.ckpt",
    "caliby_packer_000": "caliby/caliby_packer_000.ckpt",
    "caliby_packer_010": "caliby/caliby_packer_010.ckpt",
    "caliby_packer_030": "caliby/caliby_packer_030.ckpt",
}


def resolve_ckpt_path(ckpt_name_or_path: str) -> str:
    """Resolve a checkpoint specifier to a local file path.

    If ``ckpt_name_or_path`` does **not** end with ``.ckpt``, it is treated as a
    model name and looked up in :data:`MODEL_REGISTRY`.  The resolved file is
    downloaded from HuggingFace if it does not already exist under
    ``$MODEL_PARAMS_DIR``.

    If it **does** end with ``.ckpt``, it is treated as a literal file path and
    returned as-is.
    """
    if not ckpt_name_or_path.endswith(".ckpt"):
        weights_dir = os.environ["MODEL_PARAMS_DIR"]
        rel_path = MODEL_REGISTRY[ckpt_name_or_path]
        full_path = os.path.join(weights_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"Downloading {rel_path} from HuggingFace ({HF_REPO_ID})...")
            hf_hub_download(repo_id=HF_REPO_ID, filename=rel_path, local_dir=weights_dir)
        return full_path
    else:
        return ckpt_name_or_path


def ensure_dir(path: str) -> str:
    """Ensure a directory of weight files exists locally, downloading from HuggingFace if needed.

    The HF subdirectory is derived by stripping the ``$MODEL_PARAMS_DIR`` prefix
    from ``path``.
    """
    if os.path.isdir(path):
        return path

    weights_dir = os.environ["MODEL_PARAMS_DIR"]
    subdir = str(Path(path).relative_to(weights_dir))
    print(f"Downloading {subdir}/ from HuggingFace ({HF_REPO_ID})...")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=weights_dir, allow_patterns=f"{subdir}/**")
    return path
