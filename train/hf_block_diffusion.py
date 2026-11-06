"""Compatibility import for the model now in ``train.models.block_diffusion``."""
import sys
from train.models import block_diffusion as _implementation

# Preserve private helper imports and module-level patches used by existing tools.
sys.modules[__name__] = _implementation
