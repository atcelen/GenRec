"""Weight resolution for GenRec checkpoints.

Resolves a checkpoint spec — a registry name (e.g. ``"genrec"``), an ``http(s)``
URL, or a local filesystem path — to a local file path, downloading from
HuggingFace on first use and caching it locally.
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

# Registry of named GenRec checkpoints -> HuggingFace (repo_id, filename).
# Downloads go through huggingface_hub, so they share the HF cache and use your
# `huggingface-cli login` token if the repo requires authentication.
WEIGHTS_REGISTRY = {
    "genrec": {"repo_id": "atcelen/GenRec", "filename": "genrec.safetensors"},
}

# Per-dataset observation masks used for the paper's observed/unobserved metric
# split (boolean masks packed as .npz, keyed by scene name).
OBS_MASKS_REGISTRY = {
    "re10k": {"repo_id": "atcelen/GenRec", "filename": "obs_masks/re10k.npz"},
    "dl3dv": {"repo_id": "atcelen/GenRec", "filename": "obs_masks/dl3dv.npz"},
    "mipnerf360": {"repo_id": "atcelen/GenRec", "filename": "obs_masks/mipnerf360.npz"},
}


def resolve_obs_masks(dataset_type: str, cache_dir: Optional[str] = None) -> str:
    """Download the paper's observation masks for ``dataset_type``; return the
    local path of the ``.npz`` file."""
    if dataset_type not in OBS_MASKS_REGISTRY:
        raise ValueError(
            f"No published observation masks for dataset '{dataset_type}' "
            f"(available: {sorted(OBS_MASKS_REGISTRY)}). Pass a local "
            f"--obs_masks_dir or 'none'."
        )
    entry = OBS_MASKS_REGISTRY[dataset_type]
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=entry["repo_id"], filename=entry["filename"], cache_dir=cache_dir
    )


def resolve_weights(name_or_path: str, cache_dir: Optional[str] = None) -> str:
    """Resolve a checkpoint spec to a local file path.

    Accepts, in order:
      * a local filesystem path        -> returned unchanged
      * a registry name (e.g. "genrec")-> downloaded from HuggingFace (cached)
      * an ``http(s)`` URL             -> downloaded (cached under ``TORCH_HOME/hub``)

    Args:
        name_or_path: registry name, URL, or local path.
        cache_dir: optional download cache directory (defaults to the HF /
            torch hub cache).

    Returns:
        Local path to the checkpoint file.
    """
    # Local path wins — lets users point at their own checkpoint.
    if os.path.exists(name_or_path):
        return name_or_path

    if name_or_path in WEIGHTS_REGISTRY:
        entry = WEIGHTS_REGISTRY[name_or_path]
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            cache_dir=cache_dir,
        )

    if name_or_path.startswith("http://") or name_or_path.startswith("https://"):
        return _download(name_or_path, cache_dir)

    raise FileNotFoundError(
        f"Could not resolve checkpoint '{name_or_path}': it is not a local path, a "
        f"known registry name {sorted(WEIGHTS_REGISTRY)}, or an http(s) URL."
    )


def _download(url: str, cache_dir: Optional[str]) -> str:
    """Download ``url`` to the torch hub cache (or ``cache_dir``); return local path."""
    from torch.hub import download_url_to_file, get_dir

    if cache_dir is None:
        cache_dir = os.path.join(get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)

    filename = os.path.basename(urlparse(url).path) or "checkpoint.pth"
    dst = os.path.join(cache_dir, filename)
    if not os.path.exists(dst):
        print(f"[weights] downloading {url} -> {dst}")
        download_url_to_file(url, dst)
    return dst
