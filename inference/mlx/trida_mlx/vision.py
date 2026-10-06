"""Vision input for Trida checkpoints that ship a Qwen3.5 (Qwen3-VL family) vision encoder.

Three pieces, all MLX / numpy / PIL only:

* ``preprocess``      Qwen2/3-VL image preprocessing (smart resize to a multiple of
                      patch*merge, normalize, flatten into (C, T, P, P) patches) -> pixel rows
                      + ``grid_thw``.
* ``VisionTower``     the Qwen3-VL vision encoder: Conv3d patch embed, bilinearly
                      interpolated learned position embedding, 2-D rotary ViT blocks and the
                      2x2 patch merger/projector into the LM hidden size.
                      Adapted from mlx-vlm's ``qwen3_vl/vision.py`` (MIT, Apple Inc. and
                      contributors) so the parameter names match HF/mlx-vlm checkpoints.
* ``mrope_*``         interleaved multimodal RoPE positions for prompts that contain images
                      (text tokens get equal t/h/w positions, i.e. ordinary RoPE).
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ----------------------------------------------------------------------------- config
@dataclass
class VisionConfig:
    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    out_hidden_size: int = 2560
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "VisionConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class ImageProcessorConfig:
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    min_pixels: int = 64 * 32 * 32
    max_pixels: int = 1024 * 1024
    image_mean: tuple = (0.5, 0.5, 0.5)
    image_std: tuple = (0.5, 0.5, 0.5)

    @classmethod
    def from_files(cls, path, max_pixels: Optional[int] = None) -> "ImageProcessorConfig":
        import json
        c = cls()
        for name in ("preprocessor_config.json", "processor_config.json"):
            f = path / name
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            d = d.get("image_processor", d)
            c.patch_size = int(d.get("patch_size", c.patch_size))
            c.temporal_patch_size = int(d.get("temporal_patch_size", c.temporal_patch_size))
            c.merge_size = int(d.get("merge_size", c.merge_size))
            size = d.get("size") or {}
            c.min_pixels = int(size.get("shortest_edge", d.get("min_pixels", c.min_pixels)))
            c.max_pixels = int(size.get("longest_edge", d.get("max_pixels", c.max_pixels)))
            c.image_mean = tuple(d.get("image_mean", c.image_mean))
            c.image_std = tuple(d.get("image_std", c.image_std))
            break
        if max_pixels:  # on-device budget: every 32x32 pixels is one LM token
            c.max_pixels = min(c.max_pixels, max_pixels)
            c.min_pixels = min(c.min_pixels, c.max_pixels)
        return c


# ----------------------------------------------------------------------------- preprocessing
def load_image(src):
    """PIL image from bytes, a data: URL, an http(s) URL or a local path."""
    from PIL import Image

    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    elif isinstance(src, str) and src.startswith("data:"):
        data = base64.b64decode(src.split(",", 1)[1])
    elif isinstance(src, str) and src.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(src, timeout=20) as r:
            data = r.read()
    elif isinstance(src, str):
        with open(src.removeprefix("file://"), "rb") as fh:
            data = fh.read()
    else:
        raise ValueError(f"unsupported image source {type(src)}")
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB"), data


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    if max(height, width) / max(1, min(height, width)) > 200:
        raise ValueError("image aspect ratio must be < 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess(img, cfg: ImageProcessorConfig):
    """-> (pixel_rows float32 [N, C*T*P*P], grid_thw (t, h, w))."""
    from PIL import Image

    P, T, m = cfg.patch_size, cfg.temporal_patch_size, cfg.merge_size
    h, w = smart_resize(img.height, img.width, P * m, cfg.min_pixels, cfg.max_pixels)
    x = np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0  # H W C
    x = (x - np.array(cfg.image_mean, np.float32)) / np.array(cfg.image_std, np.float32)
    x = x.transpose(2, 0, 1)[None]  # 1 C H W
    x = np.repeat(x, T, axis=0)  # a still image fills the temporal patch
    C = x.shape[1]
    gt, gh, gw = 1, h // P, w // P
    x = x.reshape(gt, T, C, gh // m, m, P, gw // m, m, P)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gt * gh * gw, C * T * P * P), (gt, gh, gw)


def image_key(data: bytes, cfg: ImageProcessorConfig) -> int:
    """Stable negative id standing for one image's tokens in cache keys (never a real token)."""
    h = hashlib.sha1(data + f"{cfg.max_pixels}/{cfg.min_pixels}".encode()).hexdigest()
    return -(int(h[:12], 16) % (1 << 46)) - 1


# ----------------------------------------------------------------------------- encoder
def _rotate_half(x):
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class PatchEmbed(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        k = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
        self.proj = nn.Conv3d(cfg.in_channels, cfg.hidden_size, kernel_size=k, stride=k, bias=True)

    def __call__(self, x):
        c = self.cfg
        x = x.reshape(-1, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size).moveaxis(1, 4)
        return self.proj(x).reshape(-1, c.hidden_size)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, cu_seqlens: list, cos, sin):
        L = x.shape[0]
        q, k, v = mx.split(self.qkv(x).reshape(L, 3, self.num_heads, -1).transpose(1, 0, 2, 3), 3)
        q, k, v = q[0], k[0], v[0]  # [L, H, D]
        dt = q.dtype
        q = (q.astype(mx.float32) * cos + _rotate_half(q.astype(mx.float32)) * sin).astype(dt)
        k = (k.astype(mx.float32) * cos + _rotate_half(k.astype(mx.float32)) * sin).astype(dt)
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))  # [1, H, L, D]
        outs = []
        for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):  # attention within each image
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], scale=self.scale))
        o = mx.concatenate(outs, axis=2)
        return self.proj(o[0].transpose(1, 0, 2).reshape(L, -1))


class _MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden, bias=True)
        self.linear_fc2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class _Block(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = _Attention(cfg.hidden_size, cfg.num_heads)
        self.mlp = _MLP(cfg.hidden_size, cfg.intermediate_size)

    def __call__(self, x, cu, cos, sin):
        x = x + self.attn(self.norm1(x), cu, cos, sin)
        return x + self.mlp(self.norm2(x))


class _Merger(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.hidden = cfg.hidden_size * cfg.spatial_merge_size ** 2
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden, self.hidden)
        self.linear_fc2 = nn.Linear(self.hidden, cfg.out_hidden_size)

    def __call__(self, x):
        x = self.norm(x).reshape(-1, self.hidden)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        if cfg.deepstack_visual_indexes:
            raise NotImplementedError("deepstack vision features are not supported")
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.num_grid_per_side = int(cfg.num_position_embeddings ** 0.5)
        self.blocks = [_Block(cfg) for _ in range(cfg.depth)]
        self.merger = _Merger(cfg)

    # 2-D rotary (rows, cols of the full-resolution patch grid, in merge-window order)
    def _rotary(self, grid):
        m = self.cfg.spatial_merge_size
        head_dim = self.cfg.hidden_size // self.cfg.num_heads
        dim = head_dim // 2
        inv = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        rows, cols = [], []
        for t, h, w in grid:
            r = (np.arange(h // m)[:, None, None, None] * m + np.arange(m)[None, None, :, None])
            c = (np.arange(w // m)[None, :, None, None] * m + np.arange(m)[None, None, None, :])
            r = np.broadcast_to(r, (h // m, w // m, m, m)).reshape(-1)
            c = np.broadcast_to(c, (h // m, w // m, m, m)).reshape(-1)
            rows.append(np.tile(r, t)); cols.append(np.tile(c, t))
        r, c = np.concatenate(rows), np.concatenate(cols)
        freqs = np.concatenate([np.outer(r, inv), np.outer(c, inv)], axis=-1)  # [L, head_dim/2]
        emb = np.concatenate([freqs, freqs], axis=-1)[:, None, :]  # [L, 1, head_dim]
        return mx.array(np.cos(emb)), mx.array(np.sin(emb))

    def _pos_embed(self, grid):
        n, m = self.num_grid_per_side, self.cfg.spatial_merge_size
        outs = []
        for t, h, w in grid:
            hi, wi = np.linspace(0, n - 1, h), np.linspace(0, n - 1, w)
            hf, wf = hi.astype(np.int32), wi.astype(np.int32)
            hc, wc = np.minimum(hf + 1, n - 1), np.minimum(wf + 1, n - 1)
            dh, dw = (hi - hf)[:, None], (wi - wf)[None, :]
            idx = [hf[:, None] * n + wf[None], hf[:, None] * n + wc[None],
                   hc[:, None] * n + wf[None], hc[:, None] * n + wc[None]]
            wts = [(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]
            e = None
            for i, wt in zip(idx, wts):
                term = self.pos_embed(mx.array(i.reshape(-1))) * mx.array(wt.reshape(-1, 1).astype(np.float32))
                e = term if e is None else e + term
            D = e.shape[-1]
            e = mx.tile(e, (t, 1)).reshape(t, h // m, m, w // m, m, D).transpose(0, 1, 3, 2, 4, 5).reshape(-1, D)
            outs.append(e)
        return mx.concatenate(outs, axis=0)

    def __call__(self, pixels: mx.array, grid: list) -> mx.array:
        """pixels [N, C*T*P*P], grid [(t, h, w), ...] -> merged features [N / merge^2, out_hidden]."""
        x = self.patch_embed(pixels.astype(self.patch_embed.proj.weight.dtype))
        x = x + self._pos_embed(grid).astype(x.dtype)
        cos, sin = self._rotary(grid)
        cu = [0]
        for t, h, w in grid:
            for _ in range(t):
                cu.append(cu[-1] + h * w)
        for blk in self.blocks:
            x = blk(x, cu, cos, sin)
        return self.merger(x)


def sanitize_vision_weights(weights: dict) -> dict:
    """{any HF/mlx-vlm vision key: array} -> {VisionTower key: array}."""
    out = {}
    for k, v in weights.items():
        for pre in ("model.visual.", "visual.", "vision_tower.", "model.language_model.visual."):
            if k.startswith(pre):
                k = k[len(pre):]
                break
        else:
            continue
        if "position_ids" in k:
            continue
        if k == "patch_embed.proj.weight" and v.ndim == 5 and v.shape[1] in (1, 3) and v.shape[-1] not in (1, 3):
            v = v.transpose(0, 2, 3, 4, 1)  # PyTorch NCDHW -> MLX NDHWC
        out[k] = v
    return out


def is_vision_key(k: str) -> bool:
    return k.startswith(("model.visual.", "visual.", "vision_tower.", "model.language_model.visual."))


# ----------------------------------------------------------------------------- mrope
def mrope_cos_sin(pos3: np.ndarray, dims: int, base: float, section) -> tuple:
    """pos3 [3, L] (t, h, w) -> cos, sin [L, dims] for interleaved MRoPE (Qwen3-VL/3.5):
    frequency j takes the h position if j % 3 == 1 and j < 3*section[1], the w position if
    j % 3 == 2 and j < 3*section[2], else the t position."""
    inv = 1.0 / (base ** (np.arange(0, dims, 2, dtype=np.float64) / dims))  # [dims/2]
    n = inv.shape[0]
    axis = np.zeros(n, dtype=np.int64)
    axis[1: section[1] * 3: 3] = 1
    axis[2: section[2] * 3: 3] = 2
    pos = pos3.astype(np.float64)[axis]  # [n, L]
    freqs = (pos * inv[:, None]).T  # [L, n]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return mx.array(np.cos(emb).astype(np.float32)), mx.array(np.sin(emb).astype(np.float32))


def apply_mrope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x [B, H, L, D]; rotates the first cos.shape[-1] dims (non-traditional / rotate-half)."""
    d = cos.shape[-1]
    xr, xp = x[..., :d], x[..., d:]
    xf = xr.astype(mx.float32)
    out = (xf * cos + _rotate_half(xf) * sin).astype(x.dtype)
    return mx.concatenate([out, xp], axis=-1) if xp.shape[-1] else out
