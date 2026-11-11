"""Compatibility import; use train.data.decision for new code."""
import sys
from train.data import decision as _implementation

sys.modules[__name__] = _implementation
