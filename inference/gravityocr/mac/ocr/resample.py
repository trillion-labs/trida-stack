"""uint8 bicubic resizing, bit-identical to torch's CPU kernel, in numpy.

The shipping runtime has no torch, but both models were trained and
validated on torch-resized inputs, so this reproduces ATen's uint8 path
exactly rather than approximating it (Pillow's bicubic differs by up to 2
levels with antialias, and far more without):

- antialias=True (GLM-OCR recognition): the Pillow-style filter (a = -0.5)
  whose support widens with the downscale factor, normalized per output.
- antialias=False (PP-DocLayoutV3 layout): the classic 4-tap bicubic
  (a = -0.75) around floor(source), with border pixels replicated.

Both use int16 fixed-point weights (the largest precision below 22 bits that
keeps the biggest weight under 2**15, rounding half away from zero) and a
width pass before the height pass with a uint8 intermediate.

Speed: each axis runs as banded float32 matmuls (Accelerate) over only the
input span each block of outputs touches. Weights are split as
`w = 256 * high + low` (0 <= low < 256) so every float32 sum stays an integer
below 2**24 and is exact in any summation order. The fixed-point result
`floor((sum(w * x) + 2**(p-1)) / 2**p)` equals
`floor((H + floor((L + 2**(p-1)) / 256)) / 2**(p-8))` for H = sum(high * x),
L = sum(low * x), since p >= 8 and a sub-unit remainder never changes the
floor of an integer quotient.
"""

from functools import lru_cache

import numpy as np

_BLOCK = 128


def _cubic(distance: np.ndarray, a: float) -> np.ndarray:
    x = np.abs(distance)
    return np.where(
        x < 1,
        ((a + 2) * x - (a + 3)) * x * x + 1,
        np.where(x < 2, (((x - 5) * x + 8) * x - 4) * a, 0.0),
    )


def _antialias_weights(in_size: int, out_size: int) -> np.ndarray:
    scale = in_size / out_size
    support = 2.0 * scale if scale >= 1 else 2.0
    invscale = 1 / scale if scale >= 1 else 1.0
    centers = scale * (np.arange(out_size) + 0.5)
    starts = np.maximum((centers - support + 0.5).astype(np.int64), 0)
    stops = np.minimum((centers + support + 0.5).astype(np.int64), in_size)
    source = np.arange(in_size)[:, None]
    inside = (source >= starts[None, :]) & (source < stops[None, :])
    weights = np.where(inside, _cubic((source - centers[None, :] + 0.5) * invscale, -0.5), 0.0)
    totals = weights.sum(axis=0, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals != 0)


def _classic_weights(in_size: int, out_size: int) -> np.ndarray:
    scale = in_size / out_size
    weights = np.zeros((in_size, out_size))
    source = scale * (np.arange(out_size) + 0.5) - 0.5
    base = np.floor(source).astype(np.int64)
    fraction = source - base
    columns = np.arange(out_size)
    for offset in range(-1, 3):
        rows = np.clip(base + offset, 0, in_size - 1)
        np.add.at(weights, (rows, columns), _cubic(offset - fraction, -0.75))
    return weights


@lru_cache(maxsize=256)
def _resample_blocks(in_size: int, out_size: int, antialias: bool) -> tuple[tuple, int]:
    """Banded split fixed-point weights for one axis: (blocks, precision).

    Each block is `(out_start, out_stop, in_start, in_stop, high, low)` with
    float32 `high`/`low` of shape [in span, out span].
    """
    weights = (_antialias_weights if antialias else _classic_weights)(in_size, out_size)
    precision = 0
    for bits in range(22):
        if int(0.5 + weights.max() * (1 << (bits + 1))) >= (1 << 15):
            break
        precision = bits + 1
    if precision < 8:  # a largest weight <= 1 implies >= 14; the identity needs >= 8
        raise ValueError("resampling precision below 8 bits")
    scaled = weights * (1 << precision)
    fixed = np.where(scaled < 0, np.trunc(scaled - 0.5), np.trunc(scaled + 0.5)).astype(np.int64)
    high, low = np.divmod(fixed, 256)
    used = np.flatnonzero(fixed.any(axis=1))
    blocks = []
    for out_start in range(0, out_size, _BLOCK):
        out_stop = min(out_start + _BLOCK, out_size)
        touched = np.flatnonzero(fixed[:, out_start:out_stop].any(axis=1))
        in_start, in_stop = (
            (int(touched[0]), int(touched[-1]) + 1) if touched.size else (int(used[0]), int(used[0]) + 1)
        )
        blocks.append(
            (
                out_start,
                out_stop,
                in_start,
                in_stop,
                np.ascontiguousarray(high[in_start:in_stop, out_start:out_stop], dtype=np.float32),
                np.ascontiguousarray(low[in_start:in_stop, out_start:out_stop], dtype=np.float32),
            )
        )
    return tuple(blocks), precision


def _resample_last_axis(values: np.ndarray, out_size: int, antialias: bool) -> np.ndarray:
    blocks, precision = _resample_blocks(values.shape[-1], out_size, antialias)
    source = np.ascontiguousarray(values, dtype=np.float32)
    bias = np.float32(1 << (precision - 1))
    shift = np.float32(1 << (precision - 8))
    result = np.empty((*values.shape[:-1], out_size), dtype=np.float32)
    for out_start, out_stop, in_start, in_stop, high, low in blocks:
        span = source[..., in_start:in_stop]
        carry = span @ low
        carry += bias
        carry /= 256
        np.floor(carry, out=carry)
        carry += span @ high
        carry /= shift
        np.floor(carry, out=carry)
        result[..., out_start:out_stop] = carry
    np.clip(result, 0, 255, out=result)
    return result.astype(np.uint8)


def resize_bicubic_uint8(hwc: np.ndarray, height: int, width: int, *, antialias: bool) -> np.ndarray:
    """Resize an HWC uint8 image exactly like torch's CPU uint8 bicubic kernel."""
    chw = hwc.transpose(2, 0, 1)
    if chw.shape[2] != width:
        chw = _resample_last_axis(chw, width, antialias)
    if chw.shape[1] != height:
        chw = _resample_last_axis(chw.transpose(0, 2, 1), height, antialias).transpose(0, 2, 1)
    return chw.transpose(1, 2, 0)
