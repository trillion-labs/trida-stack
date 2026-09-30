"""trida_mlx — on-device (Apple Silicon / MLX) self-speculative decoding for Trida2.0-4B."""
from .model import DEFAULT_MODEL, SeqCache, TridaRuntime, load_model, mask_id_of
from .decode import DecodeStats, SamplingParams, ar_generate, selfspec_generate

__all__ = [
    "DEFAULT_MODEL", "SeqCache", "TridaRuntime", "load_model", "mask_id_of",
    "DecodeStats", "SamplingParams", "ar_generate", "selfspec_generate",
]
