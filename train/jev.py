"""Deprecated compatibility import; use train.decision or its data/models modules."""
import sys
import warnings
from train import decision as _implementation

warnings.warn("train.jev is deprecated; use train.data.decision and train.models.decision",
              DeprecationWarning, stacklevel=2)
sys.modules[__name__] = _implementation
