import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    block_size: int
    mask_token_id: int
    eos_token_ids: frozenset[int]
    max_new_tokens: dict[str, int]

    def token_limit(self, task: str) -> int:
        return self.max_new_tokens.get(task, self.max_new_tokens["text"])


def product_token_limits(product_config: Path | None = None) -> dict[str, int]:
    """Per-task generation limits from the product config, the single source."""
    path = product_config or Path(__file__).with_name("product_generation.json")
    return dict(json.loads(path.read_text(encoding="utf-8"))["maxNewTokens"])


def load_generation_settings(
    checkpoint: Path,
    product_config: Path | None = None,
) -> GenerationSettings:
    checkpoint_config = checkpoint / "block_diffusion.json"
    if not checkpoint_config.is_file():
        raise FileNotFoundError(f"Missing checkpoint generation config: {checkpoint_config}")
    if product_config is None:
        checkpoint_override = checkpoint / "gravity_generation.json"
        product_config = (
            checkpoint_override
            if checkpoint_override.is_file()
            else Path(__file__).with_name("product_generation.json")
        )

    block = json.loads(checkpoint_config.read_text(encoding="utf-8"))
    product = json.loads(product_config.read_text(encoding="utf-8"))
    if product.get("schemaVersion") != 1:
        raise ValueError("Unsupported product generation schema")
    block_size = product.get("blockSize")
    mask_token_id = block.get("mask_id")
    eos_token_ids = product.get("eosTokenIDs")
    limits = product.get("maxNewTokens")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("blockSize must be a positive integer")
    if not isinstance(mask_token_id, int) or mask_token_id < 0:
        raise ValueError("checkpoint mask_id must be a non-negative integer")
    if (
        not isinstance(eos_token_ids, list)
        or not eos_token_ids
        or any(not isinstance(value, int) or value < 0 for value in eos_token_ids)
    ):
        raise ValueError("eosTokenIDs must contain non-negative integers")
    if (
        not isinstance(limits, dict)
        or "text" not in limits
        or any(not isinstance(value, int) or value <= 0 for value in limits.values())
    ):
        raise ValueError("maxNewTokens must contain positive task limits")
    return GenerationSettings(
        block_size=block_size,
        mask_token_id=mask_token_id,
        eos_token_ids=frozenset(eos_token_ids),
        max_new_tokens=dict(limits),
    )
