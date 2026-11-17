"""Pinned layout-model assets for the development runtime (ADR 0006).

The release runtime passes explicit local paths and never downloads. The
torch weights' revision is kept because the label table and the Core ML
export both derive from it.
"""

from __future__ import annotations

from pathlib import Path

TORCH_LAYOUT_REPOSITORY = "PaddlePaddle/PP-DocLayoutV3_safetensors"
TORCH_LAYOUT_REVISION = "97d101e6db2642e162a1d05392d1b0231c91033e"
COREML_LAYOUT_REPOSITORY = "hugging-mac/pp-doclayout-v3-coreml"
COREML_LAYOUT_REVISION = "270f90d8404e9ef88ce6293da80d4cf731586353"


def default_cache_root() -> Path:
    return Path.home() / "Library/Caches/ai.trillionlabs.gravity/layout"


def resolve_coreml_layout(cache_root: Path | None = None, *, download=None) -> Path | None:
    """The pinned Core ML package as real files, or None when unavailable.

    `local_dir` materializes real files: the Core ML compiler cannot follow
    the HF cache's symlinked blobs. Any failure (offline, network, partial
    download) returns None.
    """
    target = (cache_root or default_cache_root()) / COREML_LAYOUT_REVISION
    package = target / "model.mlpackage"
    if not (package / "Manifest.json").is_file():
        if download is None:
            from huggingface_hub import snapshot_download as download
        try:
            download(
                COREML_LAYOUT_REPOSITORY,
                revision=COREML_LAYOUT_REVISION,
                local_dir=target,
                allow_patterns=["model.mlpackage/*", "model.mlpackage/**"],
                token=False,
            )
        except Exception:
            return None
    if not (package / "Manifest.json").is_file():
        return None
    if any(path.is_symlink() for path in package.rglob("*")):
        return None
    return package
