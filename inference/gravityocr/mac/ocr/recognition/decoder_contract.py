from dataclasses import dataclass


@dataclass(frozen=True)
class BlockDiffusionConfig:
    block_size: int = 32
    mask_token_id: int = 59282
    threshold: float = 0.99
    masked_only_lm_head: bool = False
