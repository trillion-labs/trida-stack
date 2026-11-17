"""Gravity OCR package.

Import concrete runtime or reference modules explicitly. The package root is
deliberately side-effect free so metadata tools, protocol tests, and the native
launcher do not import torch or construct reference-model symbols accidentally.
"""

__all__: list[str] = []
