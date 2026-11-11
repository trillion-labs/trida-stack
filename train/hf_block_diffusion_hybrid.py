"""Compatibility import, removed in 0.2.0; use ``train.models.hybrid``."""
import sys
from train.models import hybrid as _implementation

# Preserve private helper imports and module-level patches used by existing tools.
sys.modules[__name__] = _implementation
