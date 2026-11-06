"""Deprecated: moved to ``train.data.jev`` (data) and ``train.models.jev`` (pointer head, loss).

Kept so existing ``from train.jev import ...`` keeps working; it will be removed in a future release.
"""
import warnings

from train.data.jev import (  # noqa: F401
    DISTRACTORS,
    MAX_BRANCH,
    MAX_STATE,
    MIN_PACK,
    NONE_OPTIONS,
    SPECIAL,
    ContextOverflow,
    JevRowDataset,
    augment,
    encode_rows,
    load_jev_records,
    materialize,
    none_pair,
    option_text,
    pack_jev,
    question_keys,
    record_variants,
    render,
    training_context,
    user_tokens,
)
from train.models.jev import PointerHead, jev_loss  # noqa: F401

warnings.warn(
    "train.jev is deprecated; import from train.data.jev and train.models.jev instead",
    DeprecationWarning,
    stacklevel=2,
)
