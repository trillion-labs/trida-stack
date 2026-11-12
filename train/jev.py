"""Deprecated training import; removed in 0.2.0. Use train.data.decision / train.models.decision."""
import warnings
from train.decision import *  # noqa: F403
from train.data.decision import (  # noqa: F401
    DecisionRowDataset as JevRowDataset,
    load_decision_records as load_jev_records,
    pack_decision as pack_jev,
)
from train.models.decision import decision_loss as jev_loss  # noqa: F401

warnings.warn("train.jev will be removed in 0.2.0; use train.data.decision and train.models.decision",
              DeprecationWarning, stacklevel=2)
