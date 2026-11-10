# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""The Jev pointer head: the module a Trida checkpoint's ``jev_head.pt`` loads into.

``jev_head.pt`` holds ``{"head": state_dict, "head_dim": dp}``; build ``PointerHead(hidden_size, dp)`` and load
``head`` into it. The rows it scores come from ``inference/zoro-1/jev.py`` (kept torch-free, so this lives apart).

Derived from kev's ``PointerHead`` (https://github.com/jaredpalmer/kev, ``kev/model.py``, Copyright Jared Palmer and
the kev contributors, Apache-2.0). Modified: the multi-question scoring is ``forward``.
"""
import math

import torch.nn as nn


class PointerHead(nn.Module):
    """Scores each option's </opt> state against the <decide> state. Trained from scratch."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts, owner):   # [Q, d], [sum K, d], question of each option -> logits [sum K]
        """kev's PointerHead.many. Called through forward so the head can be its own FSDP unit."""
        return (self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale
