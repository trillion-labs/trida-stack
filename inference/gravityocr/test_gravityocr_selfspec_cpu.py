# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# Portions (marked "verbatim from SGLang v0.5.10") are copied from SGLang v0.5.10 (sgl-project/sglang, commit 1519acf),
# Copyright 2023-2024 SGLang Team, licensed under the Apache License, Version 2.0, and are used here as reference behaviour.
"""GravityOCR SGLang patch: CPU unit tests of the self-spec host rules and sampling semantics.

Runs on CPU with torch and pytest only (no GPU, no SGLang install). The two dependency-free modules of the patch are loaded
by path from sglang/python/sglang/srt/speculative/:
  selfspec_commit_rules.py  host commit of a round (cut_index, commit_fast / commit_slow, host_commit, cannot_finish)
  selfspec_sampling.py      per-row penalties, penalizer cumulation, AR target distribution, custom processors, grammar bitmask
and checked against SGLang v0.5.10 behaviour, driven one token at a time the way autoregressive decoding drives it:
  Req.check_finished (verbatim copy below)                   -> the per-token commit loop
  the four penalizers and their orchestrator (verbatim copies) -> AR decoding's penalty state and logits
  spec_utils.traverse_tree (verbatim copy)                    -> the n-gram grammar bitmask
When SGLang itself is importable (the patched server venv), the same properties also run against the real classes
(real Req, real penalizers, real XGrammarGrammar), and the patched worker's plain-batch gate, a0 sampler and the full-draft
round's output log probabilities are checked.
"""

import ast
import copy
import importlib.util
import itertools
import random
import re
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
SPEC_DIR = HERE / "sglang" / "python" / "sglang" / "srt" / "speculative"


def _load(name):
    path = SPEC_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_gravityocr_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rules = _load("selfspec_commit_rules")
sampling = _load("selfspec_sampling")


def _have_sglang():
    try:
        import sglang.srt.managers.schedule_batch  # noqa: F401
    except Exception:
        return False
    return True


HAVE_SGLANG = _have_sglang()
needs_sglang = pytest.mark.skipif(not HAVE_SGLANG, reason="SGLang (the patched server venv) is not installed")


# =============================================================================================== reference: Req.check_finished
class BaseFinishReason:  # verbatim from SGLang v0.5.10 managers/schedule_batch.py (to_json kept)
    def __init__(self, is_error: bool = False):
        self.is_error = is_error


class FINISH_MATCHED_TOKEN(BaseFinishReason):  # verbatim from SGLang v0.5.10
    def __init__(self, matched):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {"type": "stop", "matched": self.matched}


class FINISH_MATCHED_STR(BaseFinishReason):  # verbatim from SGLang v0.5.10
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {"type": "stop", "matched": self.matched}


class FINISHED_MATCHED_REGEX(BaseFinishReason):  # verbatim from SGLang v0.5.10
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {"type": "stop", "matched": self.matched}


class FINISH_LENGTH(BaseFinishReason):  # verbatim from SGLang v0.5.10
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {"type": "length", "length": self.length}


class FINISH_ABORT(BaseFinishReason):  # verbatim from SGLang v0.5.10 (to_json shortened)
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {"type": "abort", "message": self.message}


class MiniReq:
    """The fields of SGLang's Req that the commit rules and check_finished read; check_finished and its helpers are
    verbatim from SGLang v0.5.10 (managers/schedule_batch.py, class Req)."""

    def __init__(self, sampling_params, eos_token_ids, tokenizer, vocab_size, output_ids, grammar=None, to_finish=None):
        self.sampling_params = sampling_params
        self.eos_token_ids = eos_token_ids
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.output_ids = output_ids
        self.origin_input_ids = [1, 2, 3]
        self.grammar = grammar
        self.to_finish = to_finish
        self.finished_reason = None
        self.finished_len = None
        self.decoded_text = ""
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0
        self.spec_acceptance_histogram = []

    def update_spec_acceptance_histogram(self, accepted_draft_tokens: int):  # verbatim
        if len(self.spec_acceptance_histogram) <= accepted_draft_tokens:
            self.spec_acceptance_histogram.extend(
                [0] * (accepted_draft_tokens - len(self.spec_acceptance_histogram) + 1)
            )
        self.spec_acceptance_histogram[accepted_draft_tokens] += 1

    def finished(self) -> bool:  # verbatim
        return self.finished_reason is not None

    def tail_str(self) -> str:  # verbatim
        if (
            len(self.sampling_params.stop_strs) == 0
            and len(self.sampling_params.stop_regex_strs) == 0
        ):
            return ""
        max_len_tail_str = max(
            self.sampling_params.stop_str_max_len + 1,
            self.sampling_params.stop_regex_max_len + 1,
        )
        tail_len = min(max_len_tail_str, len(self.output_ids))
        return self.tokenizer.decode(self.output_ids[-tail_len:])

    def _check_token_based_finish(self, new_accepted_tokens) -> bool:  # verbatim
        if self.sampling_params.ignore_eos:
            return False
        matched_eos = False
        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True
        return False

    def _check_str_based_finish(self):  # verbatim
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str()
            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    if stop_str in tail_str or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        return True
            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:
                    if re.search(stop_regex_str, tail_str):
                        self.finished_reason = FINISHED_MATCHED_REGEX(matched=stop_regex_str)
                        return True
        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens=None):  # verbatim
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id > self.vocab_size or token_id < 0:
                offset = len(self.output_ids) - len(new_accepted_tokens) + i
                if self.sampling_params.stop_token_ids:
                    self.output_ids[offset] = next(iter(self.sampling_params.stop_token_ids))
                if self.eos_token_ids:
                    self.output_ids[offset] = next(iter(self.eos_token_ids))
                self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                self.finished_len = offset + 1
                return True
        return False

    def check_finished(self, new_accepted_len: int = 1):  # verbatim
        if self.finished():
            return
        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return
        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(length=self.sampling_params.max_new_tokens)
            self.finished_len = self.sampling_params.max_new_tokens
            return
        if self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
                return
        new_accepted_tokens = self.output_ids[-new_accepted_len:]
        if self._check_token_based_finish(new_accepted_tokens):
            return
        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return
        if self._check_str_based_finish():
            return


V_COMMIT = 40


class ToyTokenizer:
    def __init__(self, eos, extra):
        self.eos_token_id = eos
        self.additional_stop_token_ids = extra

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) if 0 <= i < V_COMMIT else "?" for i in ids)


class ToyGrammar:
    """terminates after accepting `end`; rejects token `bad` (ValueError, as XGrammarGrammar.accept_token does)."""

    def __init__(self, end, bad, acc):
        self.end, self.bad, self.acc, self.term = end, bad, list(acc), False

    def accept_token(self, t):
        if self.term:
            return
        if t == self.bad:
            raise ValueError(f"Tokens not accepted: {t}")
        self.acc.append(t)
        if t == self.end:
            self.term = True

    def is_terminated(self):
        return self.term


def _mini_params(rng, max_new):
    stop = rng.choice(["ab", "zz", "c"]) if rng.random() < 0.15 else None
    return types.SimpleNamespace(
        max_new_tokens=max_new,
        ignore_eos=rng.random() < 0.15,
        stop_token_ids=set(rng.sample(range(V_COMMIT), rng.randint(0, 2))) or None,
        stop_strs=[stop] if stop else [],
        stop_regex_strs=[],
        stop_str_max_len=len(stop) if stop else 0,
        stop_regex_max_len=0,
    )


def _make_mini(rng):
    max_new = rng.randint(1, 40)
    req = MiniReq(_mini_params(rng, max_new),
                  eos_token_ids=set(rng.sample(range(V_COMMIT), rng.randint(0, 2))) or None,
                  tokenizer=ToyTokenizer(rng.randrange(V_COMMIT), set(rng.sample(range(V_COMMIT), rng.randint(0, 1))) or None),
                  vocab_size=V_COMMIT,
                  output_ids=[rng.randrange(V_COMMIT) for _ in range(rng.randint(0, max_new - 1))])
    if rng.random() < 0.25:
        req.grammar = ToyGrammar(end=rng.randrange(V_COMMIT), bad=rng.randrange(V_COMMIT), acc=req.output_ids)
    if rng.random() < 0.05:
        req.to_finish = FINISH_ABORT("x")
    return req


def _make_real(rng):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT as RealAbort
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    tok = ToyTokenizer(rng.randrange(V_COMMIT), set(rng.sample(range(V_COMMIT), rng.randint(0, 1))) or None)
    tok.encode = lambda s, add_special_tokens=False: [ord(c) - 97 for c in s]
    sp = SamplingParams(max_new_tokens=rng.randint(1, 40), ignore_eos=rng.random() < 0.15,
                        stop_token_ids=rng.sample(range(V_COMMIT), rng.randint(0, 2)) or None,
                        stop=(rng.choice(["ab", "zz", "c"]) if rng.random() < 0.15 else None))
    sp.normalize(tok)
    req = Req("r", "", [1, 2, 3], sp, eos_token_ids=set(rng.sample(range(V_COMMIT), rng.randint(0, 2))) or None,
              vocab_size=V_COMMIT)
    req.tokenizer = tok
    req.output_ids = [rng.randrange(V_COMMIT) for _ in range(rng.randint(0, sp.max_new_tokens - 1))]
    if rng.random() < 0.25:
        req.grammar = ToyGrammar(end=rng.randrange(V_COMMIT), bad=rng.randrange(V_COMMIT), acc=req.output_ids)
    if rng.random() < 0.05:
        req.to_finish = RealAbort("x")
    return req


def per_token_loop(req, T, abort_factory):
    """The old per-token commit loop of the full-draft round (worker _FullDraftVerifyInput._fill_requests + the scheduler's
    check_finished re-check), with SGLang's AR handling of a grammar rejection: the request is aborted, never the server."""
    def accept(t):
        try:
            req.grammar.accept_token(t)
            return True
        except ValueError:
            req.to_finish = abort_factory(f"grammar rejected token {t}")
            req.check_finished()
            return False

    req.output_ids.append(T[0])
    req.check_finished()
    a0_fin = req.finished() or (req.grammar is not None and not accept(T[0]))
    kept = 1
    if not a0_fin:
        for t in T[1:]:
            req.output_ids.append(t)
            req.check_finished()
            kept += 1
            if req.finished() or (req.grammar is not None and not accept(t)):
                break
    req.check_finished()
    req.spec_verify_ct += 1
    adt = 0 if a0_fin else len(T) - 1        # convention: the untruncated verify count (acc + 1), 0 if a0 ended the request
    req.spec_accepted_tokens += adt
    req.update_spec_acceptance_histogram(adt)
    return kept


def _state(r):
    fr = r.finished_reason
    g = r.grammar
    return (list(r.output_ids), None if fr is None else (type(fr).__name__, fr.to_json()), getattr(r, "finished_len", None),
            r.spec_verify_ct, r.spec_accepted_tokens, list(getattr(r, "spec_acceptance_histogram", None) or []),
            None if g is None else (list(g.acc), g.term))


def _random_T(rng):
    K = rng.randint(1, 8)
    return [rng.randrange(-2, V_COMMIT + 3) if rng.random() < 0.05 else rng.randrange(V_COMMIT) for _ in range(K + 1)]


@pytest.fixture
def abort_factory():
    old = rules._abort_reason_factory
    rules.set_abort_reason_factory(lambda msg: FINISH_ABORT(message=msg))
    yield rules._abort_reason_factory
    rules._abort_reason_factory = old


def test_commit_rules_module_is_dependency_free():
    tree = ast.parse((SPEC_DIR / "selfspec_commit_rules.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"logging", "typing"}, mods


@pytest.mark.parametrize("kind", ["mini", pytest.param("real", marks=needs_sglang)])
def test_host_commit_equals_per_token_loop(kind, abort_factory):
    """host_commit (one extend + one check_finished for fast requests, the per-token loop for slow ones) leaves every request in
    exactly the state the per-token loop leaves it: tokens, finish reason / length, vocab-boundary replacement, spec counters,
    grammar state; and cannot_finish is sound (always) and exact (fast requests)."""
    make = _make_mini if kind == "mini" else _make_real
    if kind == "real":
        from sglang.srt.managers.schedule_batch import FINISH_ABORT as RealAbort
        rules.set_abort_reason_factory(lambda msg: RealAbort(message=msg))
    rng = random.Random(1234)
    n_rej = 0
    for it in range(4000):
        r0 = make(rng)
        T = _random_T(rng)
        a, b = copy.deepcopy(r0), copy.deepcopy(r0)
        row = [T[0], len(T) - 2] + T[1:]
        pre = rules.cannot_finish(copy.deepcopy(r0), T)
        kept = per_token_loop(a, T, rules._abort_reason_factory)
        toks, js, accs = rules.host_commit([b], [row])
        assert _state(a) == _state(b), (it, T)
        assert kept == js[0] + 1 and toks[0] == T[: js[0] + 1] and accs[0] == len(T) - 2, (it, T)
        exact = js[0] == len(T) - 1 and not b.finished()
        assert not pre or exact, ("cannot_finish unsound", it, T)
        if rules.is_fast(r0):
            assert pre == exact, ("cannot_finish inexact for a fast request", it, T)
        n_rej += b.finished_reason is not None and type(b.finished_reason).__name__ == "FINISH_ABORT" and r0.to_finish is None
    assert n_rej > 0, "the randomized cases never exercised a grammar rejection"


def test_grammar_rejection_aborts_instead_of_raising(abort_factory):
    req = MiniReq(_mini_params(random.Random(0), 30), eos_token_ids=None, tokenizer=ToyTokenizer(39, None), vocab_size=V_COMMIT,
                  output_ids=[], grammar=ToyGrammar(end=38, bad=7, acc=[]))
    req.sampling_params.ignore_eos, req.sampling_params.stop_token_ids, req.sampling_params.stop_strs = True, None, []
    toks, js, accs = rules.host_commit([req], [[5, 3, 6, 7, 9, 10]])   # a0 = 5, verified 6 7 9 10 (7 is rejected)
    assert toks == [[5, 6, 7]] and js == [2]
    assert type(req.finished_reason).__name__ == "FINISH_ABORT"
    assert req.grammar.acc == [5, 6]
    # a0 itself rejected: the request ends at a0, no verify token is appended
    req2 = MiniReq(_mini_params(random.Random(0), 30), eos_token_ids=None, tokenizer=ToyTokenizer(39, None), vocab_size=V_COMMIT,
                   output_ids=[], grammar=ToyGrammar(end=38, bad=5, acc=[]))
    req2.sampling_params.ignore_eos, req2.sampling_params.stop_strs = True, []
    toks, js, _ = rules.host_commit([req2], [[5, 1, 6, 8]])
    assert toks == [[5]] and js == [0] and req2.spec_accepted_tokens == 0
    assert type(req2.finished_reason).__name__ == "FINISH_ABORT"


def test_cut_index_examples():
    stop = frozenset({9})
    assert rules.cut_index([1, 2, 3], budget=10, stop=stop, vocab=40) == 2
    assert rules.cut_index([1, 9, 3], budget=10, stop=stop, vocab=40) == 1
    assert rules.cut_index([1, 2, 3], budget=2, stop=stop, vocab=40) == 1
    assert rules.cut_index([1, 2, 3], budget=0, stop=stop, vocab=40) == 0
    assert rules.cut_index([1, 41, 3], budget=10, stop=stop, vocab=40) == 1
    assert rules.cut_index([1, -1, 3], budget=10, stop=stop, vocab=40) == 1
    assert rules.cut_index([1, 2, 9], budget=10, stop=stop, vocab=40) == 2   # the last token is cut by check_finished itself


# =============================================================================================== reference: penalizers
def apply_scaling_penalties(logits, scaling_penalties):  # verbatim from SGLang v0.5.10 (without @torch.compile)
    logits[:] = torch.where(logits < 0, logits * scaling_penalties, logits / scaling_penalties)


class _BatchedPenalizer:  # verbatim from SGLang v0.5.10 sampling/penaltylib/orchestrator.py (the parts used here)
    is_multiplicative = False

    def __init__(self, orchestrator):
        self._orchestrator = orchestrator
        self._is_prepared = False

    @property
    def orchestrator(self):
        return self._orchestrator

    def is_prepared(self):
        return self._is_prepared

    def prepare(self):
        if not self._is_prepared:
            self._prepare()
            self._is_prepared = True

    def prepare_if_required(self):
        if self._is_required():
            self.prepare()
            return True
        return False

    def cumulate_output_tokens(self, output_ids):
        if not self._is_prepared:
            return
        self._cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits):
        if not self._is_prepared:
            return
        self._apply(logits=logits)


class BatchedRepetitionPenalizer(_BatchedPenalizer):  # verbatim from SGLang v0.5.10
    is_multiplicative = True

    def _is_required(self):
        return any(req.sampling_params.repetition_penalty != 1.0 for req in self.orchestrator.reqs())

    def _prepare(self):
        self.cumulated_repetition_penalties = torch.ones(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size), dtype=torch.float32)
        self.repetition_penalties = torch.tensor(
            [req.sampling_params.repetition_penalty for req in self.orchestrator.reqs()], dtype=torch.float32).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids):
        self.cumulated_repetition_penalties.scatter_(dim=1, index=output_ids.unsqueeze(1), src=self.repetition_penalties)

    def _apply(self, logits):
        apply_scaling_penalties(logits, self.cumulated_repetition_penalties)

    def get_scaling_penalties(self):
        return self.cumulated_repetition_penalties


class BatchedFrequencyPenalizer(_BatchedPenalizer):  # verbatim from SGLang v0.5.10
    def _is_required(self):
        return any(req.sampling_params.frequency_penalty != 0.0 for req in self.orchestrator.reqs())

    def _prepare(self):
        self.cumulated_frequency_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size), dtype=torch.float32)
        self.frequency_penalties = torch.tensor(
            [req.sampling_params.frequency_penalty for req in self.orchestrator.reqs()], dtype=torch.float32).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids):
        self.cumulated_frequency_penalties.scatter_add_(dim=1, index=output_ids.unsqueeze(1), src=self.frequency_penalties)

    def _apply(self, logits):
        logits.sub_(self.cumulated_frequency_penalties)


class BatchedPresencePenalizer(_BatchedPenalizer):  # verbatim from SGLang v0.5.10
    def _is_required(self):
        return any(req.sampling_params.presence_penalty != 0.0 for req in self.orchestrator.reqs())

    def _prepare(self):
        self.cumulated_presence_penalties = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size), dtype=torch.float32)
        self.presence_penalties = torch.tensor(
            [req.sampling_params.presence_penalty for req in self.orchestrator.reqs()], dtype=torch.float32).unsqueeze_(1)

    def _cumulate_output_tokens(self, output_ids):
        self.cumulated_presence_penalties.scatter_(dim=1, index=output_ids.unsqueeze(1), src=self.presence_penalties)

    def _apply(self, logits):
        logits.sub_(self.cumulated_presence_penalties)


class BatchedMinNewTokensPenalizer(_BatchedPenalizer):  # verbatim from SGLang v0.5.10
    def _is_required(self):
        return any(req.sampling_params.min_new_tokens > 0 for req in self.orchestrator.reqs())

    def _prepare(self):
        V = self.orchestrator.vocab_size
        self.min_new_tokens = torch.tensor(
            [req.sampling_params.min_new_tokens for req in self.orchestrator.reqs()], dtype=torch.int32).unsqueeze_(1)
        padded_stop_token_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(list((req.sampling_params.stop_token_ids or set())
                               | (req.tokenizer.additional_stop_token_ids or set())
                               | {req.tokenizer.eos_token_id}), dtype=torch.int64)
             for req in self.orchestrator.reqs()],
            batch_first=True, padding_value=V)
        self.stop_token_penalties = torch.zeros((len(self.orchestrator.reqs()), V + 1), dtype=torch.float32).scatter_add_(
            dim=1, index=padded_stop_token_ids,
            src=torch.full_like(padded_stop_token_ids, dtype=torch.float32, fill_value=float("-inf")))[:, :V]
        self.len_output_tokens = torch.zeros((len(self.orchestrator.reqs()), 1), dtype=torch.int32)

    def _cumulate_output_tokens(self, output_ids):
        self.len_output_tokens += 1

    def _apply(self, logits):
        mask = (self.len_output_tokens < self.min_new_tokens).expand_as(logits)
        logits[mask] += self.stop_token_penalties[mask]


class BatchedPenalizerOrchestrator:  # verbatim from SGLang v0.5.10 (init, cumulate_output_tokens, apply incl. repeat)
    def __init__(self, vocab_size, batch, penalizers):
        self.vocab_size = vocab_size
        self.batch = batch
        self.device = "cpu"
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}
        is_required = False
        for penalizer in self.penalizers.values():
            is_required |= penalizer.prepare_if_required()
        self.is_required = is_required

    def reqs(self):
        return self.batch.reqs

    def cumulate_output_tokens(self, output_ids):
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits, repeat=None):
        if repeat is None:
            for penalizer in self.penalizers.values():
                penalizer.apply(logits)
        else:   # the relaxed speculative variant (the old self-spec verify rows used this)
            bs = logits.shape[0] // repeat
            additive = torch.zeros((bs, logits.shape[1]), dtype=torch.float32)
            for penalizer in self.penalizers.values():
                if not penalizer.is_multiplicative:
                    penalizer.apply(additive)
            logits.add_(torch.repeat_interleave(additive, repeat, dim=0))
            acc = None
            for penalizer in self.penalizers.values():
                if penalizer._is_prepared and penalizer.is_multiplicative:
                    acc = penalizer.get_scaling_penalties().clone() if acc is None else acc * penalizer.get_scaling_penalties()
            if acc is not None:
                apply_scaling_penalties(logits, torch.repeat_interleave(acc, repeat, dim=0))


MINI_PENALIZERS = {"rep": BatchedRepetitionPenalizer, "freq": BatchedFrequencyPenalizer,
                   "pres": BatchedPresencePenalizer, "minnew": BatchedMinNewTokensPenalizer}
V_PEN = 50


def _penalty_lib(kind):
    if kind == "mini":
        return MINI_PENALIZERS, BatchedPenalizerOrchestrator, apply_scaling_penalties
    import sglang.srt.sampling.penaltylib as pl
    from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties as real_scaling
    return ({"rep": pl.BatchedRepetitionPenalizer, "freq": pl.BatchedFrequencyPenalizer, "pres": pl.BatchedPresencePenalizer,
             "minnew": pl.BatchedMinNewTokensPenalizer}, pl.BatchedPenalizerOrchestrator, real_scaling)


_KEEP_ALIVE = []


def _make_orch(rng, kind, which, reqs=None, bs=None):
    classes, Orch, _ = _penalty_lib(kind)
    if reqs is None:
        reqs = []
        for _ in range(bs):
            sp = types.SimpleNamespace(repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, min_new_tokens=0,
                                       stop_token_ids={rng.randrange(V_PEN)})
            if "rep" in which:
                sp.repetition_penalty = rng.choice([1.0, 1.3, 0.8])
            if "freq" in which:
                sp.frequency_penalty = rng.choice([0.0, 0.4, -0.2])
            if "pres" in which:
                sp.presence_penalty = rng.choice([0.0, 0.5])
            if "minnew" in which:
                sp.min_new_tokens = rng.randrange(0, 12)
            tok = types.SimpleNamespace(additional_stop_token_ids={rng.randrange(V_PEN)}, eos_token_id=rng.randrange(V_PEN))
            reqs.append(types.SimpleNamespace(sampling_params=sp, tokenizer=tok))

    class _Batch:   # weakref-able stand-in for ScheduleBatch
        pass

    batch = _Batch()
    batch.reqs, batch.device = reqs, "cpu"
    _KEEP_ALIVE.append(batch)
    return Orch(vocab_size=V_PEN, batch=batch, penalizers=set(classes[w] for w in MINI_PENALIZERS)), reqs


def _row_orchs(rng, kind, which, reqs, hist):
    """one single-request orchestrator per request, fed its history one token at a time with the upstream API (= AR). Every
    penalizer is prepared (as the upstream merge() does for an unprepared one), so a request with neutral parameters has the
    neutral state a batched orchestrator keeps for it."""
    out = []
    for req, h in zip(reqs, hist):
        o, _ = _make_orch(rng, kind, which, reqs=[req])
        for p in o.penalizers.values():
            p.prepare()
        for t in h:
            o.cumulate_output_tokens(torch.tensor([t], dtype=torch.int64))
        out.append(o)
    return out


STATE_NAMES = ("cumulated_repetition_penalties", "cumulated_presence_penalties", "cumulated_frequency_penalties",
               "len_output_tokens")
SUBSETS = [set(c) for n in range(1, 5) for c in itertools.combinations(sorted(MINI_PENALIZERS), n)]


@pytest.mark.parametrize("kind", ["mini", pytest.param("real", marks=needs_sglang)])
def test_cumulate_block_equals_per_token(kind):
    rng = random.Random(7)
    torch.manual_seed(7)
    checked = 0
    for which in SUBSETS:
        for _ in range(12):
            bs = rng.randint(1, 4)
            orch, reqs = _make_orch(rng, kind, which, bs=bs)
            if not orch.is_required:
                continue
            hist = [[rng.randrange(V_PEN) for _ in range(rng.randint(0, 8))] for _ in range(bs)]
            L = max(1, max(map(len, hist)))
            # entries where valid is False hold garbage (out-of-range ids too): they must never be used as an index
            tok = torch.tensor([h + [rng.choice([-5, 10**6, 3])] * (L - len(h)) for h in hist], dtype=torch.int64)
            valid = torch.tensor([[j < len(h) for j in range(L)] for h in hist])
            sampling.cumulate_block(orch, tok, valid)
            refs = _row_orchs(rng, kind, which, reqs, hist)
            for (cls, p) in orch.penalizers.items():
                if not p.is_prepared():
                    continue
                for b, ro in enumerate(refs):
                    q = ro.penalizers[cls]
                    for name in STATE_NAMES:
                        if hasattr(p, name):
                            assert torch.equal(getattr(p, name)[b], getattr(q, name)[0]), (which, name, hist[b])
                            checked += 1
    assert checked > 100


@pytest.mark.parametrize("kind", ["mini", pytest.param("real", marks=needs_sglang)])
def test_penalize_rows_equal_ar_rows(kind):
    """Every verify row r == AR decoding's logits for the token after cand[r]: the request's history plus cand[1..r] fed one token
    at a time, then the orchestrator's (non-speculative) apply on that row. Bitwise. The relaxed speculative apply that the old
    verify used (window-start state for all K rows, additive before scaling) fails the same comparison."""
    _, _, scaling = _penalty_lib(kind)
    rng = random.Random(11)
    torch.manual_seed(11)
    K = 6
    rows = relaxed_bad = 0
    for which in SUBSETS:
        for _ in range(6):
            bs = rng.randint(1, 3)
            orch, reqs = _make_orch(rng, kind, which, bs=bs)
            if not orch.is_required:
                continue
            hist = [[rng.randrange(V_PEN) for _ in range(rng.randint(0, 6))] for _ in range(bs)]
            L = max(1, max(map(len, hist)))
            tok = torch.tensor([h + [0] * (L - len(h)) for h in hist], dtype=torch.int64)
            sampling.cumulate_block(orch, tok, torch.tensor([[j < len(h) for j in range(L)] for h in hist]))
            # draft chains with repeats, so in-block repetition is exercised
            cand = torch.tensor([[rng.choice([rng.randrange(V_PEN), *(hist[b][-2:] or [1])]) for _ in range(K)] for b in range(bs)])
            logits = torch.randn(bs * K, V_PEN) * 3
            ours = logits.clone()
            sampling.penalize_rows(orch, ours, cand, scaling)
            relaxed = logits.clone()
            orch.apply(relaxed, repeat=K)
            for b in range(bs):
                for r in range(K):
                    ro = _row_orchs(rng, kind, which, [reqs[b]], [hist[b] + cand[b, 1 : r + 1].tolist()])[0]
                    ref = logits[b * K + r].clone().view(1, -1)
                    ro.apply(ref)
                    assert torch.equal(ours[b * K + r], ref[0]), (which, b, r, hist[b], cand[b].tolist())
                    relaxed_bad += not torch.equal(relaxed[b * K + r], ref[0])
                    rows += 1
    assert rows > 300
    assert relaxed_bad > 0, "the relaxed (old) apply should differ from AR on in-block repeats"


def _reorder(orch, classes, order):
    """the orchestrator's penalizers in the given order (SGLang builds the dict from a set of classes, so the order the
    non-speculative apply uses differs between processes)."""
    orch.penalizers = {classes[n]: orch.penalizers[classes[n]] for n in order}
    return orch


@pytest.mark.parametrize("kind", ["mini", pytest.param("real", marks=needs_sglang)])
def test_penalize_rows_follow_the_orchestrator_order(kind):
    """Repetition (scaling) and frequency / presence (subtraction) do not commute, and SGLang applies them in the order of the
    orchestrator's dict, which comes from a set of classes. For every one of the 24 orders, each verify row equals AR decoding
    with an orchestrator in that same order, bitwise; and the order changes the result (so the check has teeth)."""
    classes, _, scaling = _penalty_lib(kind)
    rng = random.Random(23)
    torch.manual_seed(23)
    K, bs = 5, 2
    differs = 0
    for _ in range(4):
        orch0, reqs = _make_orch(rng, kind, set(MINI_PENALIZERS), bs=bs)
        for req in reqs:   # every penalty active, non-neutral
            req.sampling_params.repetition_penalty, req.sampling_params.frequency_penalty = 1.3, 0.4
            req.sampling_params.presence_penalty, req.sampling_params.min_new_tokens = 0.5, 3
        hist = [[rng.randrange(V_PEN) for _ in range(rng.randint(1, 5))] for _ in range(bs)]
        cand = torch.tensor([[rng.choice(hist[b] + [rng.randrange(V_PEN)]) for _ in range(K)] for b in range(bs)])
        logits = torch.randn(bs * K, V_PEN) * 3
        results = []
        for order in itertools.permutations(sorted(MINI_PENALIZERS)):
            orch, _ = _make_orch(rng, kind, set(MINI_PENALIZERS), reqs=reqs)
            _reorder(orch, classes, order)
            L = max(map(len, hist))
            tok = torch.tensor([h + [0] * (L - len(h)) for h in hist], dtype=torch.int64)
            sampling.cumulate_block(orch, tok, torch.tensor([[j < len(h) for j in range(L)] for h in hist]))
            ours = logits.clone()
            sampling.penalize_rows(orch, ours, cand, scaling)
            for b in range(bs):
                for r in range(K):
                    ro = _reorder(_row_orchs(rng, kind, set(MINI_PENALIZERS), [reqs[b]], [hist[b] + cand[b, 1 : r + 1].tolist()])[0],
                                  classes, order)
                    ref = logits[b * K + r].clone().view(1, -1)
                    ro.apply(ref)
                    assert torch.equal(ours[b * K + r], ref[0]), (order, b, r)
            results.append(ours)
        differs += any(not torch.equal(results[0], x) for x in results[1:])
    assert differs > 0, "the penalizer order never changed a row: the check would not catch an order mismatch"


def test_cumulate_block_skips_unrequired_orchestrator():
    orch = types.SimpleNamespace(is_required=False, penalizers=None)
    sampling.cumulate_block(orch, torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1, dtype=torch.bool))
    sampling.penalize_rows(orch, torch.zeros(1, 4), torch.zeros(1, 1, dtype=torch.int64), None)
    sampling.cumulate_block(None, torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1, dtype=torch.bool))


# =============================================================================================== target distribution
def _brute_joint(p, k, top_p):
    keep = torch.zeros_like(p, dtype=torch.bool)
    for q in range(p.numel()):
        gt = p > p[q]
        keep[q] = bool(int(gt.sum()) < k and (float(p[gt].sum()) < top_p or top_p >= 1.0))
    out = torch.where(keep, p, torch.zeros_like(p))
    return out / out.sum()


def test_joint_top_k_top_p_matches_definition():
    """flashinfer's joint filter: token q kept iff #{p > p_q} < k and sum{p > p_q} < top_p (ties kept together)."""
    torch.manual_seed(0)
    for _ in range(60):
        V = random.Random(_).randint(2, 30)
        p = torch.rand(V) ** 3
        if _ % 3 == 0:   # ties
            p[: V // 2] = p[0]
        p = p / p.sum()
        k = random.Random(_ + 100).choice([1, 2, 3, 5, V, 1 << 30])
        tp = random.Random(_ + 200).choice([0.3, 0.6, 0.9, 1.0])
        got = sampling.joint_top_k_top_p(p.view(1, -1), torch.tensor([k], dtype=torch.int32), torch.tensor([tp]))[0]
        torch.testing.assert_close(got, _brute_joint(p, k, tp), rtol=0, atol=1e-6)


def _sglang_torch_sampler_mask(probs, top_ks, top_ps, min_ps, need_min_p):
    """the filtering part of SGLang v0.5.10 layers/sampler.top_k_top_p_min_p_sampling_from_probs_torch (verbatim)."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[torch.arange(0, probs.shape[-1]).view(1, -1) >= top_ks.view(-1, 1)] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    if need_min_p:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0
    out = torch.zeros_like(probs).scatter_(1, probs_idx, probs_sort)
    return out / out.sum(-1, keepdim=True)


def test_torch_backend_filter_matches_sglang_torch_sampler():
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(16, 40) * 2, dim=-1)
    top_ks = torch.randint(1, 41, (16,), dtype=torch.int32)
    top_ps = torch.rand(16) * 0.9 + 0.1
    min_ps = torch.rand(16) * 0.2
    for need in (False, True):
        got = sampling.torch_backend_filter(probs, top_ks, top_ps, min_ps, need)
        torch.testing.assert_close(got, _sglang_torch_sampler_mask(probs.clone(), top_ks, top_ps, min_ps, need), rtol=0, atol=1e-6)


def _si(**kw):
    base = dict(temperatures=torch.ones(1, 1), top_ks=torch.tensor([1 << 30], dtype=torch.int32), top_ps=torch.ones(1),
                min_ps=torch.zeros(1), need_top_p_sampling=False, need_top_k_sampling=False, need_min_p_sampling=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_verify_target_probs_branches():
    torch.manual_seed(2)
    K = 3
    logits = torch.randn(2 * K, 20)
    T = torch.tensor([[0.7], [1.3]])
    # temperature only: softmax(logits / T_row), the request's temperature on each of its K rows
    si = _si(temperatures=T, top_ks=torch.full((2,), 1 << 30, dtype=torch.int32), top_ps=torch.ones(2), min_ps=torch.zeros(2))
    got = sampling.verify_target_probs(logits.clone(), si, K, "flashinfer")
    torch.testing.assert_close(got, torch.softmax(logits / T.repeat_interleave(K, 0), dim=-1))
    # flashinfer + min_p: top-k renorm -> top-p renorm -> keep p >= min_p * max (the renorm kernels are the caller's)
    calls = []

    def tk(p, k):
        calls.append("k")
        return p

    def tp(p, q):
        calls.append("p")
        return p

    si = _si(temperatures=torch.ones(2, 1), top_ks=torch.full((2,), 1 << 30, dtype=torch.int32), top_ps=torch.ones(2),
             min_ps=torch.tensor([0.5, 0.0]), need_min_p_sampling=True)
    got = sampling.verify_target_probs(logits.clone(), si, K, "flashinfer", tk, tp)
    assert calls == ["k", "p"]
    p = torch.softmax(logits, dim=-1)
    keep = p >= p.max(-1, keepdim=True).values * torch.tensor([0.5, 0.0]).repeat_interleave(K).view(-1, 1)
    want = torch.where(keep, p, torch.zeros_like(p))
    torch.testing.assert_close(got, want / want.sum(-1, keepdim=True))
    # flashinfer, top-k / top-p only: the joint filter
    si = _si(temperatures=torch.ones(2, 1), top_ks=torch.tensor([3, 1 << 30], dtype=torch.int32), top_ps=torch.tensor([1.0, 0.5]),
             min_ps=torch.zeros(2), need_top_k_sampling=True, need_top_p_sampling=True)
    got = sampling.verify_target_probs(logits.clone(), si, K, "flashinfer")
    torch.testing.assert_close(got, sampling.joint_top_k_top_p(torch.softmax(logits, -1), torch.tensor([3, 3, 3] + [1 << 30] * 3,
                               dtype=torch.int32), torch.tensor([1.0] * 3 + [0.5] * 3)))
    with pytest.raises(ValueError):
        sampling.verify_target_probs(logits.clone(), si, K, "ascend")


def test_acceptance_coins_are_in_open_closed_unit_interval():
    c = sampling.acceptance_coins((4096, 33), "cpu")
    assert float(c.min()) > 0.0 and float(c.max()) <= 1.0


def test_is_plain_truth_table():
    def si(greedy=True, req=False, bias=None, custom=False):
        return types.SimpleNamespace(is_all_greedy=greedy, penalizer_orchestrator=types.SimpleNamespace(is_required=req),
                                     logit_bias=bias, has_custom_logit_processor=custom)

    assert sampling.is_plain(si(), False)
    assert not sampling.is_plain(si(greedy=False), False)
    assert not sampling.is_plain(si(req=True), False)
    assert not sampling.is_plain(si(bias=torch.zeros(1, 2)), False)
    assert not sampling.is_plain(si(custom=True), False)
    assert not sampling.is_plain(si(), True)


# =============================================================================================== custom logit processors
class _RecordingNoRepeat:
    """bans every token already in origin_input_ids + output_ids (a function of the request's ids, like SGLang's processors)."""

    def __call__(self, logits, params):
        for i, prm in enumerate(params):
            req = prm["__req__"]
            logits[i, req.origin_input_ids + req.output_ids] = -float("inf")
        return logits


@pytest.mark.parametrize("start", [0, 1])
def test_apply_custom_rows_matches_per_row_ar(start):
    torch.manual_seed(3)
    K, V, bs = 4, 30, 3
    reqs = [types.SimpleNamespace(origin_input_ids=[1], output_ids=[2 + i, 5]) for i in range(bs)]
    cand = [[7 + i, 9, 11, 13 + i] for i in range(bs)]
    si = types.SimpleNamespace(custom_logit_processor={"h": (_RecordingNoRepeat(), torch.tensor([True, False, True]))},
                               custom_params=[{"__req__": r, "x": 1} for r in reqs])
    logits = torch.randn(bs * K, V)
    ours = logits.clone()
    sampling.apply_custom_rows(ours, si, cand, start)
    for i in range(bs):
        for j in range(K):
            ref = logits[i * K + j].clone().view(1, -1)
            if si.custom_logit_processor["h"][1][i]:
                view = types.SimpleNamespace(origin_input_ids=reqs[i].origin_input_ids,
                                             output_ids=reqs[i].output_ids + cand[i][start : j + 1])
                ref = _RecordingNoRepeat()(ref, [{"__req__": view, "x": 1}])
            assert torch.equal(ours[i * K + j], ref[0]), (i, j)
    assert reqs[0].output_ids == [2, 5], "the real request must not be modified"


def test_request_view_is_read_only():
    req = types.SimpleNamespace(output_ids=[1], rid="a")
    v = sampling.RequestView(req, [2, 3])
    assert v.output_ids == [1, 2, 3] and v.rid == "a" and req.output_ids == [1]
    with pytest.raises(AttributeError):
        v.output_ids = []


# =============================================================================================== grammar bitmask
def traverse_tree(retrieve_next_token, retrieve_next_sibling, draft_tokens, grammar, allocate_token_bitmask, vocab_size=None):
    """verbatim from SGLang v0.5.10 speculative/spec_utils.py (logging removed)."""
    assert retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape

    def dfs(curr, retrieve_next_token, retrieve_next_sibling, parent_pos):
        if curr == 0:
            accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            if vocab_size and curr_token_id >= vocab_size:
                accepted = False
            else:
                accepted = (parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))) != 0
        if accepted:
            if curr != 0:
                grammar.accept_token(draft_tokens[curr])
            if not grammar.is_terminated():
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    dfs(retrieve_next_token[curr], retrieve_next_token, retrieve_next_sibling, curr)
            if curr != 0:
                grammar.rollback(1)
        if retrieve_next_sibling[curr] != -1:
            dfs(retrieve_next_sibling[curr], retrieve_next_token, retrieve_next_sibling, parent_pos)

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


V_G = 70


class PackedToyGrammar:
    """An xgrammar-like grammar object (packed int32 bitmask, accept_token raises on a rejected token, rollback, termination).
    Language: tokens alternate between even and odd ids, starting with whatever the first token is; token 1 ends it."""

    def __init__(self):
        self.acc = []
        self.finished = False

    def _allowed(self):
        if not self.acc:
            return set(range(V_G))
        want = 1 - (int(self.acc[-1]) % 2)
        return {t for t in range(V_G) if t % 2 == want} | {1}

    def accept_token(self, t):
        t = int(t)
        if self.is_terminated():
            return
        if t not in self._allowed():
            raise ValueError(f"Tokens not accepted: {t}")
        self.acc.append(t)

    def rollback(self, k):
        del self.acc[len(self.acc) - k:]

    def is_terminated(self):
        return bool(self.acc) and self.acc[-1] == 1

    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        return torch.full((batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32)

    def fill_vocab_mask(self, vocab_mask, idx):
        words = [0] * ((V_G + 31) // 32)
        for t in self._allowed():
            words[t // 32] |= 1 << (t % 32)
        vocab_mask[idx] = torch.tensor([w - (1 << 32) if w >= (1 << 31) else w for w in words], dtype=torch.int32)


def _fresh_mask(prefix):
    g = PackedToyGrammar()
    for t in prefix:
        g.accept_token(t)
    m = g.allocate_vocab_mask(V_G, 1, "cpu")
    if not g.is_terminated():
        g.fill_vocab_mask(m, 0)
    return m[0]


def _allowed_by(mask_row, t):
    return (int(mask_row[t // 32]) & (1 << (t % 32))) != 0


def test_fulldraft_bitmask_rows_match_sequential_grammar():
    """row j of the full-draft bitmask == the mask of a fresh grammar after (history, a0, d1..dj), for every j up to the first
    draft token the grammar rejects; the grammar leaves the call in the state it entered it."""
    rng = random.Random(5)
    K = 6
    for _ in range(300):
        reqs, rows_ok = [], []
        bs = rng.randint(1, 3)
        vt = torch.zeros((bs, K), dtype=torch.int64)
        for i in range(bs):
            if rng.random() < 0.2:
                reqs.append(types.SimpleNamespace(grammar=None))
                vt[i] = torch.tensor([rng.randrange(V_G) for _ in range(K)])
                continue
            g = PackedToyGrammar()
            hist = []
            for _ in range(rng.randint(0, 3)):
                t = rng.choice(sorted(g._allowed() - {1}))
                g.accept_token(t)
                hist.append(t)
            a0 = rng.choice(sorted(g._allowed() - ({1} if rng.random() < 0.9 else set())))
            seq, probe = [a0], PackedToyGrammar()
            for t in hist + [a0]:
                probe.accept_token(t)
            for _ in range(K - 1):
                allowed = probe._allowed() if not probe.is_terminated() else set()
                t = rng.choice(sorted(allowed - {1})) if (allowed and rng.random() < 0.85) else rng.randrange(V_G)
                seq.append(t)
                if t in allowed and not probe.is_terminated():
                    probe.accept_token(t)
            vt[i] = torch.tensor(seq)
            reqs.append(types.SimpleNamespace(grammar=g))
            rows_ok.append((i, hist))
        before = [(list(r.grammar.acc) if r.grammar else None) for r in reqs]
        mask, first = sampling.fulldraft_bitmask(reqs, vt, V_G, traverse_tree)
        assert [(list(r.grammar.acc) if r.grammar else None) for r in reqs] == before
        if not rows_ok:
            assert mask is None and first is None
            continue
        for i, hist in rows_ok:
            prefix = hist + [int(vt[i, 0])]
            for j in range(K):
                if j:
                    tok = int(vt[i, j])
                    g = PackedToyGrammar()
                    for t in prefix:
                        g.accept_token(t)
                    if g.is_terminated() or not _allowed_by(_fresh_mask(prefix), tok):
                        break   # rows after a rejected draft token are never reached by the verify
                    prefix = prefix + [tok]
                assert torch.equal(mask[i * K + j], _fresh_mask(prefix)), (i, j, prefix)


def test_fulldraft_bitmask_rejected_a0_and_terminated_grammar_leave_rows_open():
    g_bad = PackedToyGrammar()
    g_bad.accept_token(2)                    # next must be odd
    g_term = PackedToyGrammar()
    g_term.accept_token(1)                   # terminated
    reqs = [types.SimpleNamespace(grammar=g_bad), types.SimpleNamespace(grammar=g_term)]
    vt = torch.tensor([[4, 3, 5, 7], [2, 3, 4, 5]])
    mask, first = sampling.fulldraft_bitmask(reqs, vt, V_G, traverse_tree)
    assert first is g_bad and g_bad.acc == [2] and g_term.acc == [1]
    assert bool((mask == -1).all()), "rows of a rejected a0 / terminated grammar stay all-allowed (as in AR)"


class CachedBufferToyGrammar(PackedToyGrammar):
    """PackedToyGrammar with llguidance's allocator: one buffer cached on the grammar object, handed back with the previous
    call's rows still in it (SGLang v0.5.10 constrained/llguidance_backend.py, GuidanceGrammar.allocate_vocab_mask)."""

    _buf = None

    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        if self._buf is None or self._buf.shape[0] < batch_size:
            self._buf = torch.full((batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32)
            return self._buf
        return self._buf[:batch_size]


def test_fulldraft_bitmask_starts_from_all_allowed_rows_with_a_reused_buffer():
    """Rows the traversal does not fill (a request without a grammar, rows past a rejected draft token) are all-allowed even
    when the backend returns a reused, dirty buffer."""
    g = CachedBufferToyGrammar()
    g._buf = torch.zeros((8, (V_G + 31) // 32), dtype=torch.int32)   # left over from an earlier call: every token disallowed
    reqs = [types.SimpleNamespace(grammar=None), types.SimpleNamespace(grammar=g)]
    vt = torch.tensor([[5, 6, 7, 8], [2, 4, 5, 7]])   # request 1: a0 = 2, then d1 = 4 is rejected (after an even id: odd or 1)
    mask, first = sampling.fulldraft_bitmask(reqs, vt, V_G, traverse_tree)
    assert first is g and g.acc == []
    assert bool((mask[0:4] == -1).all()), "a request without a grammar is unconstrained"
    assert torch.equal(mask[4], _fresh_mask([2]))
    assert bool((mask[5:8] == -1).all()), "rows past the rejected draft token are left all-allowed"


@needs_sglang
def test_fulldraft_bitmask_with_real_xgrammar():
    """The same property with SGLang's XGrammarGrammar and its real traverse_tree (regex grammar over a toy vocabulary)."""
    xgr = pytest.importorskip("xgrammar")
    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
    from sglang.srt.speculative.spec_utils import traverse_tree as real_traverse

    vocab = ["<eos>", "a", "b", "c", "ab", "ba", "{", "}", '"', "x", "1", "2"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[0])
    compiled = xgr.GrammarCompiler(info).compile_regex(r"(ab|ba)+c{0,2}")

    def fresh():
        return XGrammarGrammar(xgr.GrammarMatcher(compiled, max_rollback_tokens=200), len(vocab), compiled, None)

    def mask_after(prefix):
        g = fresh()
        for t in prefix:
            g.accept_token(t)
        m = g.allocate_vocab_mask(len(vocab), 1, "cpu")
        if not g.is_terminated():
            g.fill_vocab_mask(m, 0)
        return m[0], g

    rng = random.Random(9)
    K = 5
    for _ in range(200):
        g = fresh()
        hist = []
        for _ in range(rng.randint(0, 2)):
            m, _g = mask_after(hist)
            ok = [t for t in range(len(vocab)) if (int(m[t // 32]) >> (t % 32)) & 1 and t != 0]
            if not ok:
                break
            t = rng.choice(ok)
            g.accept_token(t)
            hist.append(t)
        m, _g = mask_after(hist)
        ok = [t for t in range(len(vocab)) if (int(m[t // 32]) >> (t % 32)) & 1]
        a0 = rng.choice(ok)
        vt = torch.tensor([[a0] + [rng.randrange(len(vocab)) for _ in range(K - 1)]])
        req = types.SimpleNamespace(grammar=g)
        before = list(g.accepted_tokens)
        mask, first = sampling.fulldraft_bitmask([req], vt, len(vocab), real_traverse)
        assert list(g.accepted_tokens) == before
        prefix = hist + [a0]
        for j in range(K):
            if j:
                m_prev, gp = mask_after(prefix)
                tok = int(vt[0, j])
                if gp.is_terminated() or not (int(m_prev[tok // 32]) >> (tok % 32)) & 1:
                    break
                prefix = prefix + [tok]
            m_j, gj = mask_after(prefix)
            if gj.is_terminated():
                break
            assert torch.equal(mask[j], m_j), (j, prefix)


# =============================================================================================== the patched worker (SGLang venv)
@needs_sglang
def test_plain_batch_verify_runs_only_the_greedy_kernel(monkeypatch):
    """A plain batch (all greedy, no penalty / bias / custom processor / grammar) runs exactly the greedy verify on the raw
    logits: none of the new per-row processing, sampling verify or cumulation is reached."""
    from sglang.srt.speculative import selfspec_fast_commit as fc

    def boom(*a, **k):
        raise AssertionError("non-plain machinery reached on a plain batch")

    for name in ("process_verify_logits", "sampling_verify", "cumulate_block", "penalize_rows", "apply_custom_rows"):
        monkeypatch.setattr(fc, name, boom)
    calls = []
    vi = types.SimpleNamespace(_greedy_verify=lambda batch, lo: calls.append("greedy"))
    fc.verify_logits(vi, object(), object(), None, plain=True)
    assert calls == ["greedy"]


@needs_sglang
def test_worker_plain_gate_and_a0_sampler(monkeypatch):
    """_plain is the host-only gate (a batch that asks for log probabilities is not plain); a non-plain batch draws a0 through
    model_runner._preprocess_logits + model_runner.sampler on the draft's row 0 and feeds it to the penalizers."""
    from sglang.srt.speculative.selfspec_diffusion_worker import SelfSpecDiffusionWorker

    W = SelfSpecDiffusionWorker
    plain_si = types.SimpleNamespace(is_all_greedy=True, penalizer_orchestrator=types.SimpleNamespace(is_required=False),
                                     logit_bias=None, has_custom_logit_processor=False)
    assert W._plain(types.SimpleNamespace(sampling_info=plain_si, has_grammar=False, return_logprob=False))
    assert not W._plain(types.SimpleNamespace(sampling_info=plain_si, has_grammar=True, return_logprob=False))
    assert not W._plain(types.SimpleNamespace(sampling_info=plain_si, has_grammar=False, return_logprob=True))
    events = []

    class Orch:
        is_required = True

        def cumulate_output_tokens(self, ids):
            events.append(("cumulate", ids.tolist()))

    si = types.SimpleNamespace(penalizer_orchestrator=Orch())

    class MR:
        def _preprocess_logits(self, lo, s):
            assert s is si
            events.append(("preprocess", tuple(lo.next_token_logits.shape)))
            lo.next_token_logits[:, 3] += 100.0

        def sampler(self, lo, s, return_logprob, top_n, ids_n, positions):
            events.append(("sample", return_logprob, top_n, ids_n, positions.tolist()))
            if return_logprob:
                lo.next_token_logprobs = torch.tensor([-0.5, -0.25])
            return lo.next_token_logits.argmax(-1).to(torch.int32)

    w = object.__new__(W)
    w.model_runner = MR()
    logits = torch.zeros(2, 5, 8)
    logits[:, 0, 6] = 1.0
    batch = types.SimpleNamespace(sampling_info=si, seq_lens=torch.tensor([10, 20]), return_logprob=False)
    a0 = w._sample_a0(batch, logits)
    assert a0.tolist() == [3, 3] and a0.dtype == torch.int64
    assert events == [("preprocess", (2, 8)), ("sample", False, None, None, [10, 20]), ("cumulate", [3, 3])]
    assert float(logits[0, 0, 3]) == 0.0, "the draft logits are not modified (a copy is processed)"
    assert w._a0_lo is None
    # with log probabilities the sampler computes a0's (AR's own code path) and the worker keeps them for the verify step
    events.clear()
    batch = types.SimpleNamespace(sampling_info=si, seq_lens=torch.tensor([10, 20]), return_logprob=True,
                                  top_logprobs_nums=[2, 0], token_ids_logprobs=[None, [5]])
    a0 = w._sample_a0(batch, logits)
    assert a0.tolist() == [3, 3]
    assert events[1] == ("sample", True, [2, 0], [None, [5]], [10, 20])
    assert w._a0_lo is not None and w._a0_lo.next_token_logprobs.tolist() == [-0.5, -0.25]


def _logprob_req(top=0, ids=None, on=True):
    lists = {} if not on else {k: [] for k in (
        "output_token_logprobs_val", "output_token_logprobs_idx", "output_top_logprobs_val", "output_top_logprobs_idx",
        "output_token_ids_logprobs_val", "output_token_ids_logprobs_idx")}
    return types.SimpleNamespace(return_logprob=on, top_logprobs_num=top, token_ids_logprob=ids, **lists)


def _fulldraft_logprob_case(K=4, V=11, accept=(1, -1, 0), seed=0):
    """A full-draft round's state after verify(): per request a0 + (accept + 1) verified tokens (accept -1: the request ended
    at a0), the verify logits sliced to the accepted rows, the sampler's a0 output, as _fulldraft_verify hands them over."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    g = torch.Generator().manual_seed(seed)
    bs = len(accept)
    reqs = [_logprob_req(top=2), _logprob_req(top=0, ids=[1, 4]), _logprob_req(top=3, ids=[0])]
    rows = [i * K + j for i, a in enumerate(accept) for j in range(a + 1)]
    full = torch.randn(bs * K, V, generator=g)
    a0 = torch.tensor([7, 2, 9])
    a0_logits = torch.randn(bs, V, generator=g)
    a0_lp = torch.log_softmax(a0_logits, -1)
    tv, ti = a0_lp.topk(3, -1)
    a0_lo = LogitsProcessorOutput(next_token_logits=None)
    a0_lo.next_token_logprobs = a0_lp[torch.arange(bs), a0]
    a0_lo.next_token_top_logprobs_val = [tv[i][:r.top_logprobs_num] for i, r in enumerate(reqs)]
    a0_lo.next_token_top_logprobs_idx = [ti[i][:r.top_logprobs_num] for i, r in enumerate(reqs)]
    a0_lo.next_token_token_ids_logprobs_val = [a0_lp[i, r.token_ids_logprob] if r.token_ids_logprob else [] for i, r in
                                               enumerate(reqs)]
    a0_lo.next_token_token_ids_logprobs_idx = [r.token_ids_logprob or [] for r in reqs]
    verified = torch.randint(0, V, (len(rows),), generator=g)
    vi = types.SimpleNamespace(accepted_indices=torch.tensor(rows, dtype=torch.int32), accept_length=torch.tensor(accept),
                               verified_id=verified)
    lo = LogitsProcessorOutput(next_token_logits=full[rows].clone())
    temps = torch.tensor([[1.0], [0.7], [1.3]])
    batch = types.SimpleNamespace(reqs=reqs, top_logprobs_nums=[r.top_logprobs_num for r in reqs],
                                  token_ids_logprobs=[r.token_ids_logprob for r in reqs],
                                  sampling_info=types.SimpleNamespace(temperatures=temps, device="cpu"),
                                  spec_info=types.SimpleNamespace(draft_token_num=K))
    return batch, a0, a0_lo, a0_lp, vi, lo, full, rows, temps


@needs_sglang
def test_fulldraft_output_logprobs_put_a0_first_then_the_verified_tokens():
    """NEW-1: one entry per committed token, in commit order: a0's from the sampler output, then each verified token's from the
    verify logits at AR's temperature (add_output_logprobs_for_spec_v1); a request that ends at a0 gets a0's entry only."""
    from sglang.srt.speculative.selfspec_diffusion_worker import fulldraft_output_logprobs

    batch, a0, a0_lo, a0_lp, vi, lo, full, rows, temps = _fulldraft_logprob_case()
    fulldraft_output_logprobs(batch, a0, a0_lo, vi, lo, n_verified=len(rows))
    K, pt = 4, 0
    for i, (req, acc) in enumerate(zip(batch.reqs, (1, -1, 0))):
        ver = vi.verified_id[pt : pt + acc + 1].tolist()
        assert req.output_token_logprobs_idx == [int(a0[i])] + ver
        want = [float(a0_lp[i, a0[i]])] + [
            float(torch.log_softmax(full[i * K + j] / temps[i], -1)[t]) for j, t in enumerate(ver)]
        assert req.output_token_logprobs_val == pytest.approx(want, abs=1e-6)
        n = 1 + acc + 1
        if req.top_logprobs_num:
            assert len(req.output_top_logprobs_val) == len(req.output_top_logprobs_idx) == n
            assert req.output_top_logprobs_idx[0] == a0_lp[i].topk(req.top_logprobs_num).indices.tolist()
            for j in range(1, n):
                lp = torch.log_softmax(full[i * K + j - 1] / temps[i], -1)
                assert req.output_top_logprobs_idx[j] == lp.topk(req.top_logprobs_num).indices.tolist()
        else:
            assert req.output_top_logprobs_val == []
        if req.token_ids_logprob:
            assert req.output_token_ids_logprobs_idx == [req.token_ids_logprob] * n
            assert req.output_token_ids_logprobs_val[0] == pytest.approx(a0_lp[i, req.token_ids_logprob].tolist())
        pt += acc + 1


@needs_sglang
def test_fulldraft_output_logprobs_when_every_request_ends_at_a0():
    """Every request ends at a0: no verified token at all (upstream's top-k helper cannot take an empty batch); a0's entries
    only, and a request without return_logprob is left alone."""
    from sglang.srt.speculative.selfspec_diffusion_worker import fulldraft_output_logprobs

    batch, a0, a0_lo, a0_lp, vi, lo, full, rows, temps = _fulldraft_logprob_case(accept=(-1, -1, -1))
    assert rows == []
    batch.reqs[1] = _logprob_req(on=False)
    fulldraft_output_logprobs(batch, a0, a0_lo, vi, lo, n_verified=0)
    for i in (0, 2):
        assert batch.reqs[i].output_token_logprobs_idx == [int(a0[i])]
        assert batch.reqs[i].output_token_logprobs_val == pytest.approx([float(a0_lp[i, a0[i]])])
        assert len(batch.reqs[i].output_top_logprobs_idx) == 1
    assert not hasattr(batch.reqs[1], "output_token_logprobs_val")


def test_worker_never_refuses_log_probabilities():
    """NEW-1: the full-draft round used to raise NotImplementedError for return_logprob inside the scheduler, which stopped the
    server. The vendored worker raises it nowhere, routes log-probability batches off the single-readback commit, and keeps
    them off the plain (greedy draft-argmax) gate."""
    src = (SPEC_DIR / "selfspec_diffusion_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    raised = [ast.unparse(n.exc) for n in ast.walk(tree) if isinstance(n, ast.Raise) and n.exc is not None]
    assert not [r for r in raised if "NotImplementedError" in r], raised
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "and not batch.return_logprob" in ast.get_source_segment(src, funcs["_plain"])
    fv = ast.get_source_segment(src, funcs["_fulldraft_verify"])
    assert "self._fast_commit and not batch.return_hidden_states and not batch.return_logprob" in fv
    assert "fulldraft_output_logprobs(batch, a0, a0_lo, verify_input, logits_output" in fv


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
