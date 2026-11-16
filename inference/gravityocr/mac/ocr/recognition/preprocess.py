"""GLM-OCR image preprocessing in numpy.

The shipping runtime has no torch. This reproduces, bit for bit, what the
previous torch/torchvision pipeline produced: torchvision's uint8 antialiased
bicubic resize (`ocr.resample`), float32 `(x / 255 - mean) / std` in the same
operation order, and the same (grid, merge, temporal) patch layout.
`tests/test_glm_ocr_image_processor.py` pins parity against torchvision
whenever it is installed (the development environment).
"""

import math
from dataclasses import dataclass

import numpy as np

from ocr.resample import resize_bicubic_uint8

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_MEAN = np.asarray(CLIP_MEAN, dtype=np.float32)[:, None, None]
_STD = np.asarray(CLIP_STD, dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class ProcessedImage:
    pixel_values: np.ndarray  # float32 [patches, channels * temporal * patch * patch]
    image_grid_thw: np.ndarray  # int64 [[1, grid_height, grid_width]]
    resized_height: int
    resized_width: int


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 28 * 28 * 15_000,
    temporal_factor: int = 2,
) -> tuple[int, int]:
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Absolute image aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if temporal_factor * resized_height * resized_width > max_pixels:
        beta = math.sqrt((temporal_factor * height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif temporal_factor * resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (temporal_factor * height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def pil_to_hwc_uint8(image) -> np.ndarray:
    # convert() copies even when the mode already matches, so skip it when it is a no-op.
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)


def resize_bicubic_antialias(hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    """torchvision's uint8 bicubic resize with antialias=True (see `ocr.resample`)."""
    return resize_bicubic_uint8(hwc, height, width, antialias=True)


def patchify(
    image: np.ndarray,
    *,
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
) -> tuple[np.ndarray, int, int]:
    if image.ndim != 3:
        raise ValueError("Expected a CHW image array")
    channel, height, width = image.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    if grid_height % merge_size or grid_width % merge_size:
        raise ValueError("Patch grid must be divisible by merge size")
    patches = image.reshape(
        channel,
        grid_height // merge_size,
        merge_size,
        patch_size,
        grid_width // merge_size,
        merge_size,
        patch_size,
    ).transpose(1, 4, 2, 5, 0, 3, 6)
    repeated = np.broadcast_to(
        patches[:, :, :, :, :, None],
        (*patches.shape[:5], temporal_patch_size, patch_size, patch_size),
    )
    flattened = repeated.reshape(
        grid_height * grid_width,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    return flattened, grid_height, grid_width


def preprocess_image(
    image,
    *,
    patch_size: int = 14,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 112 * 112,
    max_pixels: int = 28 * 28 * 15_000,
) -> ProcessedImage:
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        temporal_factor=temporal_patch_size,
    )
    resized = resize_bicubic_antialias(pil_to_hwc_uint8(image), resized_height, resized_width)
    normalized = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
    normalized /= np.float32(255.0)
    normalized -= _MEAN
    normalized /= _STD
    flattened, grid_height, grid_width = patchify(
        normalized,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )
    return ProcessedImage(
        pixel_values=flattened,
        image_grid_thw=np.asarray([[1, grid_height, grid_width]], dtype=np.int64),
        resized_height=resized_height,
        resized_width=resized_width,
    )
