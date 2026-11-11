"""Compatibility import; use train.models.decision for new code."""
import sys
from train.models import decision as _implementation

sys.modules[__name__] = _implementation
