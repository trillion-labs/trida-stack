import hashlib
import os
import platform
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy
from PIL import Image

from ocr.layout.processing import (
    LAYOUT_INPUT_SIZE,
    postprocess_layout,
    preprocess_layout,
)
from ocr.layout.regions import PP_DOCLAYOUT_V3_LABELS, LayoutRegion, build_regions

COMPILED_CACHE_ENV = "GRAVITY_LAYOUT_COMPILED_CACHE"


def default_compiled_cache() -> Path:
    override = os.environ.get(COMPILED_CACHE_ENV)
    if override:
        return Path(override)
    return Path.home() / "Library/Caches/ai.trillionlabs.gravity/layout-compiled"


def compiled_layout_model(
    package: Path,
    cache: Path,
    compile_model: Callable[[str, str], object],
) -> Path:
    """The package compiled once into `cache`, reused on every later start.

    Loading an `.mlpackage` makes Core ML compile it (~88 MB) into the
    system temporary folder on every engine start, and a killed engine
    never removes it. The compiled copy is keyed by the package manifest and
    the macOS version (Core ML recompiles across OS updates), and the cache
    keeps one entry, so it stays at one model's size.
    """
    manifest = (package / "Manifest.json").read_bytes()
    key = hashlib.sha256(manifest + platform.mac_ver()[0].encode()).hexdigest()[:16]
    target = cache / f"{key}.mlmodelc"
    if not target.is_dir():
        cache.mkdir(parents=True, exist_ok=True)
        staging = cache / f".{key}-{uuid.uuid4().hex}.mlmodelc"
        try:
            compile_model(str(package), str(staging))
            try:
                staging.rename(target)
            except OSError:
                if not target.is_dir():  # another engine won the race otherwise
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    for entry in cache.iterdir():
        if entry != target and not entry.name.startswith("."):
            shutil.rmtree(entry, ignore_errors=True)
    return target


class CoreMLDocLayoutDetector:
    """PP-DocLayoutV3 through the Apple Core ML export (ADR 0006).

    Pre- and post-processing are numpy ports of the transformers processor
    (`ocr.layout.processing`), so the runtime needs neither torch nor
    transformers. `compute_units` names a `coremltools.ComputeUnit`: ALL lets
    Core ML use the Neural Engine, CPU_ONLY is the fallback when that fails.
    """

    def __init__(
        self,
        coreml_package: str | Path,
        *,
        threshold: float = 0.3,
        compute_units: str = "ALL",
    ) -> None:
        import coremltools

        self.id2label = dict(enumerate(PP_DOCLAYOUT_V3_LABELS))
        self.compute_units = compute_units
        units = coremltools.ComputeUnit[compute_units]
        try:
            compiled = compiled_layout_model(
                Path(coreml_package),
                default_compiled_cache(),
                coremltools.models.utils.compile_model,
            )
            self.model = coremltools.models.CompiledMLModel(str(compiled), compute_units=units)
        except Exception:
            # A cache that cannot be written costs speed and temp space, not layout.
            self.model = coremltools.models.MLModel(str(coreml_package), compute_units=units)
        self.threshold = threshold
        # Core ML defers compile errors to the first predict. Surface them
        # here, where the caller can still choose other compute units.
        blank = numpy.zeros((1, 3, LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE), dtype=numpy.float32)
        self.model.predict({"pixel_values": blank})

    def detect(self, image: Image.Image) -> list[LayoutRegion]:
        return self.detect_profiled(image)[0]

    def detect_profiled(
        self,
        image: Image.Image,
    ) -> tuple[list[LayoutRegion], dict[str, float]]:
        preprocess_started = time.perf_counter()
        pixel_values = preprocess_layout(image)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1_000
        forward_started = time.perf_counter()
        raw = self.model.predict({"pixel_values": pixel_values})
        forward_ms = (time.perf_counter() - forward_started) * 1_000
        postprocess_started = time.perf_counter()
        result = postprocess_layout(
            numpy.asarray(raw["logits"]),
            numpy.asarray(raw["pred_boxes"]),
            numpy.asarray(raw["order_logits"]),
            threshold=self.threshold,
            image_size=image.size,
        )
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1_000
        region_started = time.perf_counter()
        regions = build_regions(result, self.id2label, image)
        region_ms = (time.perf_counter() - region_started) * 1_000
        return regions, {
            "layout_preprocess_ms": preprocess_ms,
            "layout_transfer_ms": 0.0,
            "layout_forward_ms": forward_ms,
            "layout_postprocess_ms": postprocess_ms,
            "layout_region_ms": region_ms,
        }
