"""Compatibility import for the model now in ``train.models.hybrid``."""
import sys
from train.models import hybrid as _implementation

# Preserve private helper imports and module-level patches used by existing tools.
sys.modules[__name__] = _implementation
