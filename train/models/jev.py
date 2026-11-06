"""Jev pointer head and training loss, vendored from kev (Apache-2.0).

``PointerHead`` (``kev/model.py``) scores each option's ``</opt>`` hidden state against its row's
``<decide>`` hidden state: ``z = k(h_opt) · q(h_decide) / sqrt(dp)``. ``jev_loss`` (``kev/train.py``)
is cross-entropy over each question's options (or against a soft target), averaged per variant.
Modified for packed multi-row batches; see NOTICE. The input encoding is in ``train/data/jev.py``.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerHead(nn.Module):
    """Scores each option's </opt> state against the <decide> state. Trained from scratch."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts, owner):   # [Q, d], [sum K, d], question of each option -> logits [sum K]
        """kev's PointerHead.many. Called through forward so the head can be its own FSDP unit."""
        return (self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale


def jev_loss(z, batch):
    """Mean over variants of the mean over each variant's questions of the per-question CE (or soft-target CE), as
    kev.train averages. z: logits [sum K] in batch option order. Returns (loss, accuracy over hard-label questions).

    ``batch["questions"]`` comes from ``train.data.jev.pack_jev``; each entry needs ``start`` and ``k`` (its slice
    of z), ``variant``, and either ``target`` (soft probabilities, length k) or ``label`` (hard index). Change both
    sides together."""
    losses, correct, hard = [], 0, 0
    for q in batch["questions"]:
        lp = F.log_softmax(z[q["start"]:q["start"] + q["k"]].float(), -1)
        if q.get("target") is not None:
            losses.append(-(torch.tensor(q["target"], device=lp.device) * lp).sum())
        else:
            losses.append(-lp[q["label"]])
            correct += int(lp.argmax()) == q["label"]
            hard += 1
    per_variant = {}
    for q, loss in zip(batch["questions"], losses):
        per_variant.setdefault(q["variant"], []).append(loss)
    loss = torch.stack([torch.stack(v).mean() for v in per_variant.values()]).mean()
    return loss, (correct / hard if hard else float("nan"))


