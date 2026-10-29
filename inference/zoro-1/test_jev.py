# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""Jev encoding + pointer head — no GPU, no network. The token-level check against golden rows needs the pinned
tokenizer and lives in test_jev_golden.py."""
import subprocess
import sys
from pathlib import Path

import torch

from jev_head import PointerHead

HERE = Path(__file__).parent


def test_encoding_imports_without_torch():
    # torch (and anything else training-only) blocked: importing the encoder must still work.
    code = ("import sys\n"
            "for m in ('torch', 'transformers', 'numpy'):\n"
            "    sys.modules[m] = None\n"
            "import jev\n"
            "assert jev.encode_rows and jev.materialize\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_pointer_head_scores_options_against_decide():
    # Hand-worked: dp=4 so scale = 1/sqrt(4) = 0.5. q reads h_decide[0], k reads h_opts[1] (into the same first
    # projection dim; the other dims are zero). Question 0 has h_decide [2, 0] -> q = 2; its options have k = 3 and 5
    # -> logits 2*3*0.5 = 3.0 and 2*5*0.5 = 5.0. Question 1 has q = -1 and one option k = 4 -> -2.0.
    head = PointerHead(2, dp=4)
    q_w, k_w = torch.zeros(4, 2), torch.zeros(4, 2)
    q_w[0, 0], k_w[0, 1] = 1.0, 1.0
    head.load_state_dict({"q.weight": q_w, "q.bias": torch.zeros(4), "k.weight": k_w, "k.bias": torch.zeros(4)})
    h_decide = torch.tensor([[2.0, 0.0], [-1.0, 7.0]])
    h_opts = torch.tensor([[9.0, 3.0], [9.0, 5.0], [9.0, 4.0]])
    owner = torch.tensor([0, 0, 1])
    assert head(h_decide, h_opts, owner).tolist() == [3.0, 5.0, -2.0]


def test_pointer_head_state_dict_keys():
    # The keys jev_head.pt's "head" entry holds (scripts/export_kev_checkpoint.py copies them as they are).
    assert sorted(PointerHead(8, dp=4).state_dict()) == ["k.bias", "k.weight", "q.bias", "q.weight"]
