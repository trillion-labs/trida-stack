"""Jev-style decision training (kev's recipe) for the block-diffusion trainer.

A Jev request is a ``state`` plus typed questions (``noul`` yes/no, ``choice``, ``score``); the model
returns one probability per option. Following kev (https://github.com/jaredpalmer/kev, Apache-2.0),
each question is a causal row ``<state> … <q> instructions <opt> o1 </opt> … <decide>`` (reused Qwen
special tokens as delimiters) and a small pointer head scores every ``</opt>`` hidden state against
the ``<decide>`` hidden state: ``z = k(h_opt) · q(h_decide) / sqrt(dp)``; the loss is cross-entropy
over the options (or against a soft target). The encoder, request mapping, augmentation and loss
below are vendored from kev (``kev/api.py``, ``kev/data.py``, ``kev/model.py``, ``kev/train.py``; modified: packed
multi-row batches, this trainer's tokenizer handling, see NOTICE) so
training text is byte-identical to what kev's serving path builds; ``train/test_jev.py`` checks that.

Rows run through the trainer's document-isolated causal stack (``forward_jev`` in
``hf_block_diffusion_hybrid.py``): every question row is one packed document.
"""
import hashlib
import json
import math
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

# ---- request mapping (kev/api.py) ---------------------------------------------------------------

def render(v, indent=0):
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def option_text(name, desc):
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype, criteria):
    """Keys a question's probabilities are reported under, in option order."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def materialize(req):
    """Labelled request -> internal record {"state", "questions": [{instr, options, label, qtype, keys, target?}]}
    (kev.data.materialize over kev.api.to_record)."""
    qs = []
    for qid, q in req["questions"].items():
        keys = question_keys(q["type"], q.get("criteria"))
        if q["type"] == "noul":
            c = q.get("criteria") or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q["type"] == "choice":
            opts = [option_text(k, v) for k, v in q["criteria"].items()]
        else:
            opts = [render(x) for x in q["criteria"]]
        y = q["label"]
        rq = {"instr": render(q.get("instructions")), "options": opts, "qid": qid, "qtype": q["type"], "keys": keys,
              "label": keys.index(y) if q["type"] == "choice" else int(y)}
        if q.get("target") is not None:
            t = [float(q["target"].get(k, 0.0)) for k in keys]
            if sum(t) <= 0:
                raise ValueError(f"target for {qid} puts no mass on any option")
            rq["target"] = [x / sum(t) for x in t]
        qs.append(rq)
    return {"state": render(req["state"]), "questions": qs}


# ---- augmentation (kev/data.py) -------------------------------------------------------------------

NONE_OPTIONS = [("other", "None of the above"), ("other", "A reason that fits none of the above"), ("none", "None of these"),
                ("other", "Something else"), ("not_listed", "Not listed here"), ("none_of_the_above", None),
                ("other", "A category that fits none of the above"), ("other", "None of the listed options apply"),
                ("unknown", "Cannot be determined from the options given"), ("other", "Other"), ("none", None),
                ("other", "An answer not covered by the other options"), ("no_match", "No option matches")]
DISTRACTORS = {"weather": "Bad weather caused it", "purple": "The colour purple", "pancakes": "A recipe for pancakes",
               "taxes": "Unrelated: quarterly tax filing"}


def augment(req, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Choice only: permute option order (always); sometimes add a 'none of the above' option, either as the correct
    answer (true option removed) or as a wrong alternative; sometimes add an irrelevant distractor."""
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q
            continue
        crit, y = dict(q["criteria"]), q["label"]
        if q.get("target") is not None:   # soft targets: permute only
            keys = list(crit)
            rng.shuffle(keys)
            out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}}
            continue
        r = rng.random()
        none_options = [(k, v) for k, v in NONE_OPTIONS if k not in crit]
        distractors = [k for k in DISTRACTORS if k not in crit]
        if len(crit) > 2 and r < p_none and none_options:
            nk, nd = rng.choice(none_options)
            crit.pop(y)
            crit[nk] = nd
            y = nk
        elif p_none <= r < p_none + p_none_distract and len(crit) < 255 and none_options:
            nk, nd = rng.choice(none_options)
            crit[nk] = nd
        elif p_none + p_none_distract <= r < p_none + p_none_distract + p_distract and len(crit) < 255 and distractors:
            k = rng.choice(distractors)
            crit[k] = DISTRACTORS[k]
        keys = list(crit)
        rng.shuffle(keys)
        out["questions"][qid] = {**q, "criteria": {k: crit[k] for k in keys}, "label": y}
    return out


def none_pair(req, rng):
    """Minimal pair for the none-of-the-above shortcut: the same question with the true option present (a none option
    is wrong) and removed (the same none option is right). [] when no Choice is eligible."""
    eligible = [(qid, q) for qid, q in req["questions"].items()
                if q["type"] == "choice" and len(q["criteria"]) >= 3 and q.get("target") is None]
    if not eligible:
        return []
    qid, q = rng.choice(eligible)
    nk, nd = rng.choice([o for o in NONE_OPTIONS if o[0] not in q["criteria"]] or [("none_of_these", None)])
    keys = list(q["criteria"]) + [nk]
    rng.shuffle(keys)
    present = {**q, "criteria": {k: (nd if k == nk else q["criteria"][k]) for k in keys}}
    absent = {**present, "criteria": {k: v for k, v in present["criteria"].items() if k != q["label"]}, "label": nk}
    return [{"state": req["state"], "questions": {qid: present}}, {"state": req["state"], "questions": {qid: absent}}]


def record_variants(req, rng, p_none_pair=0.25):
    """The requests one record trains this epoch: a freshly augmented copy, plus its none-pair siblings with
    probability p_none_pair (kev.train.record_variants)."""
    variants = [augment(req, rng)]
    if p_none_pair > 0 and rng.random() < p_none_pair:
        variants += none_pair(req, rng)
    return variants


# ---- encoder (kev/model.py) -----------------------------------------------------------------------

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]  # state q opt /opt decide
MAX_STATE, MAX_BRANCH = 384, 1024
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class ContextOverflow(ValueError):
    pass


def training_context(max_state=MAX_STATE):
    """Encoder limits with the state limit lifted to max_state; the row limit grows by the same amount."""
    if max_state < MAX_STATE:
        raise ValueError(f"max_state must be >= {MAX_STATE}")
    return {"max_state": max_state, "max_branch": MAX_BRANCH + max_state - MAX_STATE}


def user_tokens(tok, text):
    """Caller text can never produce delimiter/control tokens: `<|name|>` is rewritten to `<¦name¦>` first."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode_rows(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH):
    """One record -> its question rows, each the causal row [<state> state] + [<q> instr <opt>o</opt>... <decide>]
    (kev.model.encode + rows_of, non-isolated options). Row k: {"ids", "decide": index of <decide>, "opts": index of
    each </opt>, "label", "target"?}. The state is cut to its first max_state tokens (kev's non-strict encode); a
    branch over its row budget raises ContextOverflow."""
    s_id, q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL)
    state = [s_id] + user_tokens(tok, rec["state"])[: max_state - 1]
    rows = []
    for q in rec["questions"]:
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        branch = instr + [t for sp in spans for t in sp] + [d_id]
        if len(branch) > max_branch - len(state):
            raise ContextOverflow(f"branch too long: {len(branch)} tokens with a {len(state)}-token state")
        ends, cursor = [], len(state) + len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        row = {"ids": state + branch, "decide": len(state) + len(branch) - 1, "opts": ends, "label": q["label"]}
        if q.get("target") is not None:
            row["target"] = q["target"]
        rows.append(row)
    return rows


# ---- pointer head + loss (kev/model.py, kev/train.py) ----------------------------------------------

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
    kev.train averages. z: logits [sum K] in batch option order. Returns (loss, accuracy over hard-label questions)."""
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


# ---- data ---------------------------------------------------------------------------------------

def load_jev_records(paths):
    """Labelled requests (kev's training JSONL: API request shape + `label` per question) from files or dirs."""
    files = []
    for p in paths:
        files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".jsonl")) if os.path.isdir(p) else [p]
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    records.append({"state": r["state"], "questions": r["questions"]})
    return records


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


class JevRowDataset(IterableDataset):
    """Endless stream of Jev micro-batches. Each epoch reshuffles, augments every record afresh (+ none pairs), encodes
    it to question rows and packs whole variants into micro-batches of at most `max_tokens` row tokens. A variant longer
    than that gets a micro-batch of its own, up to `max_variant_tokens` (longer ones are skipped, counted). Records
    are sharded by (rank, worker)."""

    def __init__(self, records, tokenizer, rank=0, world=1, seed=0, max_tokens=32768, max_state=7552, p_none_pair=0.25,
                 max_variant_tokens=None):
        self.records, self.tok, self.rank, self.world, self.seed = records, tokenizer, rank, world, seed
        self.max_tokens, self.ctx, self.p_none_pair = max_tokens, training_context(max_state), p_none_pair
        self.max_variant_tokens = max(max_tokens, max_variant_tokens or 0)
        self.skipped = 0

    def _variants(self, epoch, idx):
        rng = random.Random(_seed(self.seed, epoch, idx))
        for v in record_variants(self.records[idx], rng, self.p_none_pair):
            try:
                rows = encode_rows(self.tok, materialize(v), **self.ctx)
            except ContextOverflow:
                self.skipped += 1
                continue
            if sum(len(r["ids"]) for r in rows) > self.max_variant_tokens:
                self.skipped += 1
                continue
            yield rows

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        shard, nshard = self.rank * nw + wid, self.world * nw
        n = len(self.records)
        if n == 0:
            raise ValueError("JevRowDataset has no records")
        epoch = 0
        while True:
            order = list(range(n))
            random.Random(_seed(self.seed, "order", epoch)).shuffle(order)
            batch, used, emitted = [], 0, False
            # fewer records than shards: wrap around so every shard still gets one (an empty shard never yields)
            for idx in (order[i % n] for i in range(shard, max(n, nshard), nshard)):
                for rows in self._variants(epoch, idx):
                    t = sum(len(r["ids"]) for r in rows)
                    if t > self.max_tokens:                      # oversize: its own micro-batch
                        emitted = True
                        yield pack_jev([rows], self.tok)
                        continue
                    if batch and used + t > self.max_tokens:
                        emitted = True
                        yield pack_jev(batch, self.tok)
                        batch, used = [], 0
                    batch.append(rows)
                    used += t
            if batch:
                emitted = True
                yield pack_jev(batch, self.tok)
            if not emitted:
                raise RuntimeError(f"Jev shard {shard}/{nshard} produced no micro-batch in epoch {epoch}: every variant "
                                   f"was skipped (over --jev_max_variant_tokens or the context limit)")
            epoch += 1


MIN_PACK = 128   # below this many query tokens inductor picks its flex *decoding* kernel, which has no config for head_dim 256


def pack_jev(variants, tok):
    """Variants (each a list of question rows) -> one packed causal row: input_ids / seg_id [1, T] (each question row
    is its own document, positions restart per row; a row shorter than MIN_PACK is padded with a dummy document), flat
    indices of every <decide> and </opt>, and per-question {start, k, label, target?, variant} for the loss."""
    ids, seg, decide, opts, owner, questions = [], [], [], [], [], []
    for vi, rows in enumerate(variants):
        for row in rows:
            base, qi = len(ids), len(questions)
            ids += row["ids"]
            seg += [qi] * len(row["ids"])
            decide.append(base + row["decide"])
            q = {"start": len(opts), "k": len(row["opts"]), "label": row["label"], "variant": vi}
            if "target" in row:
                q["target"] = row["target"]
            opts += [base + o for o in row["opts"]]
            owner += [qi] * len(row["opts"])
            questions.append(q)
    if len(ids) < MIN_PACK:   # pad with a dummy row the loss never reads: flex attention can't compile shorter rows
        pad = MIN_PACK - len(ids)
        ids += [getattr(tok, "pad_token_id", None) or 0] * pad
        seg += [len(questions)] * pad
    return {"input_ids": torch.tensor([ids]), "seg_id": torch.tensor([seg]), "decide": torch.tensor(decide),
            "opts": torch.tensor(opts), "owner": torch.tensor(owner), "questions": questions}
