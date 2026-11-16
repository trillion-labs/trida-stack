"""Import the mlx_vlm modules we use without paying for the package's eager __init__.

``mlx_vlm/__init__.py`` imports convert/generate/utils, and
``mlx_vlm/models/glm_ocr/__init__.py`` imports a processor we never call. Between them
they pull transformers, torch and scipy.optimize into the process, which costs about
2.9 seconds of start-up for code the MLX path never touches. The modules we actually
use — cache, base, turboquant, rope_utils and the glm_ocr model — depend only on mlx,
numpy and PIL.

Registering unexecuted package objects in ``sys.modules`` lets the submodule imports
resolve while their parent ``__init__`` bodies stay unrun. If the upstream layout ever
changes, ``load_glm_ocr_symbols`` discards the stubs and falls back to the ordinary
import, so a mlx_vlm upgrade degrades start-up time instead of breaking the engine.
"""

import importlib
import importlib.util
import sys

_STUB_PACKAGES = (
    "mlx_vlm",
    "mlx_vlm.models",
    "mlx_vlm.models.glm_ocr",
)


def _install_stub_packages() -> None:
    for name in _STUB_PACKAGES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.submodule_search_locations is None:
            raise ImportError(f"{name} is not an importable package")
        module = importlib.util.module_from_spec(spec)
        module.__path__ = spec.submodule_search_locations
        sys.modules[name] = module


def _purge_mlx_vlm_modules() -> None:
    for name in [key for key in sys.modules if key.split(".")[0] == "mlx_vlm"]:
        del sys.modules[name]


def _load_via_stubs() -> tuple[type, type, type]:
    _install_stub_packages()
    cache = importlib.import_module("mlx_vlm.models.cache")
    glm_ocr = importlib.import_module("mlx_vlm.models.glm_ocr.glm_ocr")
    return cache.KVCache, glm_ocr.Model, glm_ocr.ModelConfig


def _load_normally() -> tuple[type, type, type]:
    cache = importlib.import_module("mlx_vlm.models.cache")
    glm_ocr = importlib.import_module("mlx_vlm.models.glm_ocr")
    return cache.KVCache, glm_ocr.Model, glm_ocr.ModelConfig


def load_glm_ocr_symbols() -> tuple[type, type, type]:
    """Return ``(KVCache, Model, ModelConfig)`` from mlx_vlm.

    Uses the cheap path when mlx_vlm has not been imported yet, and the ordinary import
    when something else in the process already loaded the full package.
    """
    if "mlx_vlm" in sys.modules:
        return _load_normally()
    try:
        return _load_via_stubs()
    except Exception:
        _purge_mlx_vlm_modules()
        return _load_normally()
