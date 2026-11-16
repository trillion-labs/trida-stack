# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# Portions (the UPSTREAM_* blocks, marked "verbatim from SGLang v0.5.10") are copied from SGLang v0.5.10
# (sgl-project/sglang, commit 1519acf), Copyright 2023-2024 SGLang Team, licensed under the Apache License,
# Version 2.0, and are used here as reference behaviour.
"""GravityOCR SGLang patch: every change on a shared code path is confined to self-spec (SELFSPEC_DIFFUSION).

The patched SGLang is a general-purpose build. Other models, the other speculative algorithms (EAGLE, EAGLE3,
NEXTN, STANDALONE, NGRAM) and SGLang's diffusion LLMs (--dllm-algorithm: LLaDA2, SDAR) must run exactly as
upstream v0.5.10 does. Two groups of checks:

- On the patch text (stdlib + pytest; run in CI): the patch leaves the dLLM modules, the EAGLE-family runners and
  workers and the other upstream-only files untouched, adds no code that branches on dLLM state, carries none of the
  removed "dLLM self-spec" experiment, and reads its DLLM_* diagnostics only on a self-spec server.
- On the patched server (sglang/setup.sh's venv; skipped where SGLang is not installed), differential tests: the
  functions the patch changes on shared paths -- CudaGraphRunner.capture and can_run,
  FlashInferAttnBackend.forward_extend, FlashInferIndicesUpdaterPrefill.call_begin_forward and
  ForwardBatch._compute_spec_mrope_positions -- run on stubs next to verbatim copies of the v0.5.10 functions and
  must make exactly the same calls and return exactly the same values for every non-self-spec configuration
  (self-spec diagnostics switched on in the module flags, to show they stay off). Also: the EAGLE-family draft
  runners capture CUDA graphs with upstream's call (the first PR version raised AttributeError there), the files
  restored to upstream are byte-identical to v0.5.10, --dllm-algorithm is refused together with self-spec, and
  every remaining self-spec hook in the shared modules sits behind a self-spec condition (AST), and nothing of the
  removed SELFSPEC_NOMASK path (an older mask-free dual-graph experiment) is left.

Run the second group with that venv's Python, from the repository root:

    CUDA_VISIBLE_DEVICES="" "$VBIN/python" -m pytest -q inference/gravityocr/test_gravityocr_gating.py

(`$VBIN` as printed by setup.sh; pytest must be importable by that Python.)
"""

import ast
import contextlib
import hashlib
import importlib
import inspect
import itertools
import random
import re
import textwrap
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PATCH = HERE / "sglang" / "gravityocr_selfspec.patch"
ns = types.SimpleNamespace

# Upstream modules the patch must leave byte-identical (prefixes of patch paths): the diffusion-LLM package and the
# files only diffusion LLMs use, the EAGLE-family (EAGLE / EAGLE3 / NEXTN / STANDALONE) workers, infos and draft
# CUDA-graph runners, and the GLM-4V model (GLM-OCR's model file subclasses it).
UPSTREAM_ONLY = (
    "python/sglang/srt/dllm/",
    "python/sglang/srt/mem_cache/radix_cache.py",
    "python/sglang/srt/managers/schedule_policy.py",
    "python/sglang/srt/layers/logits_processor.py",
    "python/sglang/srt/models/glm4v.py",
    "python/sglang/srt/models/llada2.py",
    "python/sglang/srt/models/sdar.py",
    "python/sglang/srt/models/sdar_moe.py",
    "python/sglang/srt/speculative/eagle_",
    "python/sglang/srt/speculative/multi_layer_eagle_",
    "python/sglang/srt/speculative/standalone_worker",
)
# Identifiers of SGLang's diffusion-LLM state: code the patch adds must not branch on them.
DLLM_STATE = re.compile(
    r"\b(is_dllm|is_dllm_model|dllm_config|is_dllm_extend|is_dllm_prefill|DllmConfig|dllm_block_offsets"
    r"|DLLM_EXTEND|dllm_algorithm_config)\b"
)
# Fingerprints of the removed "SLIDE self-spec via --dllm-algorithm" experiment.
REMOVED_EXPERIMENT = ("DUMP_POS", "SELFSPEC_SLIDE", "DLLM_PREFILL_CHUNK", "return_full_logits", "_mrope_delta_dev")

# sha256 of the v0.5.10 files the patch used to modify and now leaves alone (the installed copies must match).
UPSTREAM_SHA256 = {
    "sglang.srt.dllm.config": "df4efbf252a3a403ce92d6e4cc3599246dc91ff53e1a63f5de4c196034682cbf",
    "sglang.srt.dllm.mixin.req": "097655a52170a7792b2444ad9ef2bb822c493f57e50c7bf3dfa64f1907797719",
    "sglang.srt.dllm.mixin.scheduler": "f9b792d119bdb723eeb2fc0050cdad41bf5268d8095f8861fb7b143a16259ffc",
    "sglang.srt.managers.schedule_policy": "50e847eafefcb994cefe8c740c204cdbb382dbfa1a09918e9959e313825b7adb",
    "sglang.srt.mem_cache.radix_cache": "52c73d1077101fd178f6a85f2703fa9818e74aaf6ea7f0aaaa9df08696ca7d59",
    "sglang.srt.models.glm4v": "1b875d45cdc9b9fc76908b6e67ffa6db603de3fbe550537842c43d181ff8af57",
    "sglang.srt.speculative.eagle_draft_cuda_graph_runner": "c76050a5e077cd9865bda844580b47c9ac96471e2ba08ae06af5dbf8861c18a8",
}


def _patch_text():
    return PATCH.read_text(encoding="utf-8")


def _patch_files(text):
    return re.findall(r"^diff --git a/(\S+) b/", text, flags=re.M)


def _added_lines(text):
    """(file, line) for every line the patch adds, inside hunks only (the +++ headers are not added lines)."""
    out, cur, in_hunk = [], None, False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            cur, in_hunk = line.split()[2][2:], False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            out.append((cur, line[1:]))
    return out


def _code(line):
    """The line without a trailing comment (good enough for the patch's own code: no '#' inside its strings)."""
    s = line.strip()
    return "" if s.startswith("#") else re.sub(r"\s+#.*$", "", s)


# ================================================================================================ the patch text
def test_patch_text_is_parsed():
    files = _patch_files(_patch_text())
    assert len(files) >= 5 and len(_added_lines(_patch_text())) > 100


def test_patch_leaves_upstream_dllm_and_eagle_modules_alone():
    bad = [f for f in _patch_files(_patch_text()) if f.startswith(UPSTREAM_ONLY)]
    assert not bad, f"the patch modifies upstream-only files: {bad}"


def test_patch_adds_no_code_that_depends_on_dllm_state():
    hits = [(f, line.strip()) for f, line in _added_lines(_patch_text()) if DLLM_STATE.search(_code(line))]
    assert not hits, f"added code reads diffusion-LLM state (must run as upstream for dLLMs): {hits}"


def test_patch_carries_none_of_the_removed_dllm_experiment():
    hits = [(f, line.strip()) for f, line in _added_lines(_patch_text()) if any(w in line for w in REMOVED_EXPERIMENT)]
    assert not hits, hits


def test_patch_reads_dllm_diagnostics_only_on_selfspec_servers():
    reads = [
        (f, _code(line))
        for f, line in _added_lines(_patch_text())
        if re.search(r"environ(\.get\(|\[)\s*['\"]DLLM_", _code(line))
    ]
    assert reads, "expected the gated DLLM_PROF / DLLM_GRAPH_DBG reads"
    ungated = [(f, c) for f, c in reads if "is_selfspec" not in c]
    assert not ungated, f"DLLM_* diagnostics read outside a self-spec condition: {ungated}"


# ====================================================================================== the patched server (SGLang)
def _importable(name):
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


HAVE_SGLANG = _importable("sglang.srt.model_executor.cuda_graph_runner")
needs_sglang = pytest.mark.skipif(not HAVE_SGLANG, reason="SGLang (the patched server venv) is not installed")


def _have_selfspec():
    if not HAVE_SGLANG:
        return False
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return hasattr(SpeculativeAlgorithm, "SELFSPEC_DIFFUSION")


HAVE_SELFSPEC = _have_selfspec()
needs_selfspec = pytest.mark.skipif(not HAVE_SELFSPEC, reason="needs the patched SGLang (SELFSPEC_DIFFUSION)")


def _upstream(module, src, name):
    """The v0.5.10 function `name` from `src`, compiled against the (possibly monkeypatched) globals of `module`."""
    g = dict(vars(module))
    code = "from __future__ import annotations\n" + textwrap.dedent(src)  # the modules' annotations are lazy too
    exec(compile(code, f"<SGLang v0.5.10 {module.__name__}.{name}>", "exec"), g)
    return g[name]


def _canon(x):
    """Comparable form of call arguments: tensors by dtype / shape / values."""
    import torch

    if isinstance(x, torch.Tensor):
        return ("tensor", str(x.dtype), tuple(x.shape), x.detach().cpu().flatten().tolist())
    if isinstance(x, dict):
        return tuple(sorted((k, _canon(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_canon(v) for v in x)
    return x


# ----------------------------------------------------------------------------- restored files == v0.5.10 (G8, G11)
@needs_sglang
@pytest.mark.parametrize("module", sorted(UPSTREAM_SHA256))
def test_restored_module_is_byte_identical_to_upstream(module):
    path = Path(importlib.import_module(module).__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == UPSTREAM_SHA256[module], f"{path} differs from v0.5.10"


# ------------------------------------------------------------------------------------- CUDA-graph capture (G8)
EAGLE_RUNNERS = (
    ("sglang.srt.speculative.eagle_draft_cuda_graph_runner", "EAGLEDraftCudaGraphRunner"),
    ("sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner", "EAGLEDraftExtendCudaGraphRunner"),
    ("sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner", "MultiLayerEagleDraftExtendCudaGraphRunner"),
)


def _stub_capture_env(monkeypatch, cgr):
    """capture() without a GPU: memory probe, tqdm, torch.compile patching, gc freezing and graph streams stubbed."""
    monkeypatch.setattr(cgr, "get_available_gpu_memory", lambda *a, **k: 0.0)
    monkeypatch.setattr(cgr, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(cgr, "patch_model", lambda *a, **k: contextlib.nullcontext("forward"))
    monkeypatch.setattr(cgr, "freeze_gc", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(cgr, "graph_capture", lambda *a, **k: contextlib.nullcontext(ns(stream=None)))


def _capture_stub(cls, **attrs):
    """A runner object as capture() sees it; like the EAGLE-family runners, CudaGraphRunner.__init__ never ran."""
    obj = object.__new__(cls)
    obj.capture_bs, obj.compile_bs = [1, 2, 4], [2]
    obj.enable_profile_cuda_graph = False
    obj.enable_pdmux = False
    obj.num_tokens_per_bs = 1
    obj.graphs, obj.output_buffers = {}, {}
    obj.model_runner = ns(device="cuda", gpu_id=0, model=None, tp_group=None, server_args=ns(enable_cudagraph_gc=False))
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _record_capture_calls(monkeypatch, cls):
    """Replace cls.capture_one_batch_size with a recorder that binds every call against cls's own signature."""
    sig = inspect.signature(cls.capture_one_batch_size)
    calls = []

    def rec(self, *args, **kwargs):
        sig.bind(self, *args, **kwargs)  # TypeError on a call this runner's signature rejects
        bs, _forward, *rest = args
        calls.append((bs, tuple(rest), dict(kwargs)))
        return ("graph", bs, tuple(sorted(kwargs.items()))), ("out", bs, tuple(sorted(kwargs.items())))

    monkeypatch.setattr(cls, "capture_one_batch_size", rec)
    return calls


@needs_sglang
@pytest.mark.parametrize("module, name", EAGLE_RUNNERS)
def test_eagle_family_draft_runner_captures_like_upstream(monkeypatch, module, name):
    import sglang.srt.model_executor.cuda_graph_runner as cgr

    cls = getattr(importlib.import_module(module), name)
    _stub_capture_env(monkeypatch, cgr)
    upstream_capture = _upstream(cgr, UPSTREAM_CAPTURE, "capture")
    calls = _record_capture_calls(monkeypatch, cls)
    patched = _capture_stub(cls)
    cgr.CudaGraphRunner.capture(patched)  # the first PR version raised AttributeError ('ss_fast') here
    patched_calls = list(calls)
    calls.clear()
    reference = _capture_stub(cls)
    upstream_capture(reference)
    assert patched_calls == calls and all(kw == {} for _, _, kw in patched_calls)
    assert patched.graphs == reference.graphs and patched.output_buffers == reference.output_buffers


@needs_sglang
def test_base_runner_capture_is_upstream_unless_selfspec_variants(monkeypatch):
    import sglang.srt.model_executor.cuda_graph_runner as cgr

    cls = cgr.CudaGraphRunner
    _stub_capture_env(monkeypatch, cgr)
    upstream_capture = _upstream(cgr, UPSTREAM_CAPTURE, "capture")
    calls = _record_capture_calls(monkeypatch, cls)
    reference = _capture_stub(cls)
    upstream_capture(reference)
    reference_calls = list(calls)
    # a runner of any other server: no self-spec attributes at all, or the values __init__ gives them there
    for flags in ({}, {"is_selfspec": False, "ss_fast": False}):
        calls.clear()
        runner = _capture_stub(cls, **flags)
        cls.capture(runner)
        assert calls == reference_calls and runner.graphs == reference.graphs
    if not HAVE_SELFSPEC:
        return
    # a self-spec runner on the custom-mask path (SELFSPEC_FASTATTN=0): upstream's calls; a stale is_dual_causal_graph attribute
    # (the removed SELFSPEC_NOMASK path's flag) changes nothing
    for flags in ({"ss_fast": False}, {"ss_fast": False, "is_dual_causal_graph": True}):
        calls.clear()
        runner = _capture_stub(cls, is_selfspec=True, **flags)
        cls.capture(runner)
        assert calls == reference_calls and runner.graphs == reference.graphs
    # the fast path keeps the graphs it had before this change: one per (bs, role)
    for flags in ({"ss_fast": True}, {"ss_fast": True, "is_dual_causal_graph": True}):
        calls.clear()
        runner = _capture_stub(cls, is_selfspec=True, **flags)
        cls.capture(runner)
        assert set(runner.graphs) == {(bs, v) for bs in (1, 2, 4) for v in ("draft", "verify")}
        assert [kw for _, _, kw in calls] == [{"ss_role": v} for _ in (4, 2, 1) for v in ("draft", "verify")]
        assert all(rest == (None,) for _, rest, _ in calls)


@needs_selfspec
def test_selfspec_capture_variant_keys():
    import sglang.srt.model_executor.cuda_graph_runner as cgr

    variants = cgr._selfspec_capture_variants
    assert variants(ns(), 8) == [(8, {})]
    assert variants(ns(ss_fast=False), "1_8") == [("1_8", {})]
    assert variants(ns(ss_fast=True), "1_8") == [
        (("1_8", "draft"), {"ss_role": "draft"}),
        (("1_8", "verify"), {"ss_role": "verify"}),
    ]
    # the removed SELFSPEC_NOMASK path's flag is not read any more
    assert variants(ns(is_dual_causal_graph=True, ss_fast=False), 8) == [(8, {})]
    assert variants(ns(is_dual_causal_graph=True, ss_fast=True), 8) == variants(ns(ss_fast=True), 8)


# --------------------------------------------------------------------------------------- CudaGraphRunner.can_run
def _can_run_runner(cgr, rng, alg, is_dllm, selfspec=False):
    import torch

    r = object.__new__(cgr.CudaGraphRunner)
    r.require_mlp_tp_gather = rng.random() < 0.3
    r.enable_pdmux = False
    r.disable_padding = rng.random() < 0.5
    r.capture_bs = [1, 2, 4, 8]
    r.max_bs = 8
    r.graphs = {bs: None for bs in r.capture_bs}
    r.require_mlp_sync = rng.random() < 0.3
    r.is_encoder_decoder = rng.random() < 0.2
    r.capture_hidden_mode = rng.choice(list(cgr.CaptureHiddenMode))
    r.enable_two_batch_overlap = rng.random() < 0.2
    r.num_tokens_per_bs = 1 if alg.is_none() else 4
    r.seq_len_fill_value = 1
    r.model_runner = ns(spec_algorithm=alg, attn_backend=ns(cuda_graph_custom_mask=torch.zeros(64, dtype=torch.uint8)))
    r.is_dllm, r.dllm_config = is_dllm, None
    # what CudaGraphRunner.__init__ sets on this server
    r.is_selfspec = selfspec
    r.ss_fast = False
    r.ss_padfix = selfspec
    return r


def _can_run_batch(cgr, rng, runner, modes):
    import torch

    bs = rng.choice([1, 2, 3, 4, 8, 9])
    n = bs * runner.num_tokens_per_bs + (1 if rng.random() < 0.2 else 0)
    spec = None
    if not runner.model_runner.spec_algorithm.is_none():
        # custom masks small and large (the large ones overflow the 64-byte graph mask buffer above)
        spec = ns(
            capture_hidden_mode=rng.choice([None, *cgr.CaptureHiddenMode]),
            custom_mask=torch.ones(rng.choice([8, 4096]), dtype=torch.bool),
        )
    mm = rng.random() < 0.5
    return ns(
        batch_size=bs,
        global_num_tokens_cpu=[bs * runner.num_tokens_per_bs, 1],
        can_run_dp_cuda_graph=rng.random() < 0.8,
        encoder_lens=torch.tensor([rng.choice([0, 3]) for _ in range(bs)]),
        capture_hidden_mode=rng.choice(list(cgr.CaptureHiddenMode)),
        spec_info=spec,
        can_run_tbo=rng.random() < 0.5,
        input_ids=torch.zeros(n, dtype=torch.long),
        forward_mode=rng.choice(modes),
        contains_mm_inputs=lambda: mm,
        seq_lens_cpu=None,
    )


@needs_sglang
def test_can_run_equals_upstream_on_every_other_server():
    import sglang.srt.model_executor.cuda_graph_runner as cgr
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as SA

    upstream_can_run = _upstream(cgr, UPSTREAM_CAN_RUN, "can_run")
    rng = random.Random(0)
    cases = (
        (SA.NONE, False, (ForwardMode.DECODE, ForwardMode.IDLE)),
        (SA.NONE, True, (ForwardMode.DLLM_EXTEND, ForwardMode.DECODE)),  # LLaDA2 / SDAR, multimodal or not
        (SA.EAGLE, False, (ForwardMode.TARGET_VERIFY, ForwardMode.DECODE, ForwardMode.IDLE)),
        (SA.STANDALONE, False, (ForwardMode.TARGET_VERIFY, ForwardMode.IDLE)),
        (SA.NGRAM, False, (ForwardMode.TARGET_VERIFY, ForwardMode.IDLE)),  # with masks above the graph buffer
    )
    seen = set()
    for alg, is_dllm, modes in cases:
        for _ in range(1500):
            runner = _can_run_runner(cgr, rng, alg, is_dllm)
            fb = _can_run_batch(cgr, rng, runner, modes)
            got, want = bool(runner.can_run(fb)), bool(upstream_can_run(runner, fb))
            assert got == want, (alg, is_dllm, fb.forward_mode, vars(runner), vars(fb))
            seen.add((alg, want))
    assert all((alg, v) in seen for alg, _, _ in cases for v in (False, True))  # both outcomes exercised


@needs_selfspec
def test_can_run_mask_guard_stays_on_selfspec_servers():
    import torch

    import sglang.srt.model_executor.cuda_graph_runner as cgr
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as SA

    rng = random.Random(1)
    runner = _can_run_runner(cgr, rng, SA.SELFSPEC_DIFFUSION, False, selfspec=True)
    runner.require_mlp_tp_gather = runner.require_mlp_sync = runner.is_encoder_decoder = False
    runner.enable_two_batch_overlap, runner.disable_padding = False, False
    runner.capture_hidden_mode = cgr.CaptureHiddenMode.NULL
    fb = _can_run_batch(cgr, rng, runner, (ForwardMode.TARGET_VERIFY,))
    fb.batch_size, fb.input_ids, fb.capture_hidden_mode = 2, torch.zeros(8, dtype=torch.long), cgr.CaptureHiddenMode.NULL
    fb.spec_info = ns(capture_hidden_mode=None, custom_mask=torch.ones(8, dtype=torch.bool))
    assert runner.can_run(fb)
    fb.spec_info.custom_mask = torch.ones(4096, dtype=torch.bool)  # overflows the graph's mask buffer -> eager
    assert not runner.can_run(fb)


# ------------------------------------------------------------------------- FlashInferAttnBackend.forward_extend (G11)
class _Wrapper:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def _out(self, q):
        import torch

        return torch.full_like(q, float(len(self.log)))

    def forward(self, q, *rest, **kw):
        self.log.append((self.name, "forward", len(rest), _canon(kw)))
        return self._out(q)

    def forward_return_lse(self, q, *rest, **kw):
        import torch

        self.log.append((self.name, "forward_return_lse", len(rest), _canon(kw)))
        return self._out(q), torch.zeros(q.shape[0], q.shape[1])


class _KVPool:
    def __init__(self, log):
        self.log = log

    def set_kv_buffer(self, layer, loc, k, v, k_scale, v_scale):
        self.log.append(("set_kv_buffer", _canon(loc), _canon(k), k_scale, v_scale))

    def get_kv_buffer(self, layer_id):
        import torch

        return torch.zeros(16, 2, 4), torch.ones(16, 2, 4)


def _run_forward_extend(fn, fbm, case, selfspec=False):
    import torch

    is_dllm, use_ragged, no_prefix, cross, attn_type, causal_attr, save = case
    log = []
    be = object.__new__(fbm.FlashInferAttnBackend)
    be.num_wrappers = 1
    be.forward_metadata = fbm.PrefillMetadata([_Wrapper(log, "paged")], use_ragged, no_prefix)
    be.prefill_wrapper_ragged = _Wrapper(log, "ragged")
    be.is_dllm_model = is_dllm
    be.is_selfspec, be.ss_planner = selfspec, None
    layer = ns(
        is_cross_attention=cross, attn_type=attn_type, layer_id=0, tp_q_head_num=2, tp_k_head_num=2, tp_v_head_num=2,
        head_dim=4, scaling=0.5, sliding_window_size=-1, logit_cap=0.0, k_scale=None, v_scale=None,
        k_scale_float=None, v_scale_float=None,
    )
    fb = ns(out_cache_loc=torch.arange(6), encoder_out_cache_loc=torch.arange(6) + 100, token_to_kv_pool=_KVPool(log))
    if causal_attr is not None:  # a stale attribute on the batch must not matter (the removed SELFSPEC_NOMASK channel)
        fb.dllm_ragged_causal = causal_attr
    q, k, v = (torch.arange(48, dtype=torch.float32).view(6, 8) + i for i in range(3))
    out = fn(be, q, k, v, layer, fb, save_kv_cache=save)
    return log, _canon(out)


@needs_sglang
def test_forward_extend_equals_upstream_without_selfspec(monkeypatch):
    import sglang.srt.layers.attention.flashinfer_backend as fbm

    # (flashinfer is not importable on a CPU-only host, so the module may not define merge_state)
    monkeypatch.setattr(fbm, "merge_state", lambda o1, s1, o2, s2: (o1 + o2, s1), raising=False)
    upstream_forward_extend = _upstream(fbm, UPSTREAM_FORWARD_EXTEND, "forward_extend")
    types_ = (fbm.AttentionType.DECODER, fbm.AttentionType.ENCODER_ONLY)
    n = 0
    for case in itertools.product((False, True), (False, True), (False, True), (False, True), types_, (None, False, True),
                                  (True, False)):
        got = _run_forward_extend(fbm.FlashInferAttnBackend.forward_extend, fbm, case)
        want = _run_forward_extend(upstream_forward_extend, fbm, case)
        assert got == want, case
        n += 1
    assert n == 192


@needs_selfspec
def test_forward_extend_on_selfspec_ignores_a_stale_causal_flag(monkeypatch):
    # The removed SELFSPEC_NOMASK path took the paged causal flag from forward_batch.dllm_ragged_causal. A self-spec server
    # without the fast path's metadata now uses upstream's flag (not cross attention) whatever the batch carries.
    import sglang.srt.layers.attention.flashinfer_backend as fbm

    for causal in (None, False, True):
        case = (False, False, False, False, fbm.AttentionType.DECODER, causal, True)
        log, _ = _run_forward_extend(fbm.FlashInferAttnBackend.forward_extend, fbm, case, selfspec=True)
        assert [dict(c[3])["causal"] for c in log if c[0] == "paged"] == [True]


# -------------------------------------------------------------- FlashInferIndicesUpdaterPrefill.call_begin_forward
class _PlanRecorder:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def begin_forward(self, *args, **kwargs):
        self.log.append((self.name, _canon(args), _canon(kwargs)))


def _run_call_begin_forward(fn, fbm, spec, selfspec, fixed_split, use_ragged):
    import torch

    from sglang.srt.speculative.ngram_info import NgramVerifyInput

    log = []
    up = object.__new__(fbm.FlashInferIndicesUpdaterPrefill)
    up.attn_backend = ns(is_selfspec=selfspec)
    up.num_qo_heads, up.num_kv_heads, up.head_dim, up.page_size = 4, 2, 8, 1
    up.q_data_type = up.data_type = torch.bfloat16
    up.kv_last_page_len = torch.ones(16, dtype=torch.int32)
    up.req_to_token = torch.arange(4 * 64, dtype=torch.int32).view(4, 64)
    seq_lens = torch.tensor([7, 9], dtype=torch.int32)
    prefix_lens = torch.tensor([3, 5], dtype=torch.int32)
    spec_info = None
    if spec:
        spec_info = object.__new__(NgramVerifyInput)  # what EAGLE / n-gram verify hands the planner
        spec_info.generate_attn_arg_prefill = lambda rpi, lens, lens_sum, r2t: (
            torch.arange(16, dtype=torch.int32), torch.tensor([0, 7, 16], dtype=torch.int32),
            torch.tensor([0, 4, 8], dtype=torch.int32), torch.ones(40, dtype=torch.bool),
        )
    fn(
        up, _PlanRecorder(log, "ragged"), _PlanRecorder(log, "paged"), torch.tensor([0, 1]), seq_lens, int(seq_lens.sum()),
        seq_lens, prefix_lens, None, torch.zeros(17, dtype=torch.int32), torch.zeros(17, dtype=torch.int32), use_ragged,
        spec_info, fixed_split_size=fixed_split,
    )
    return log


@needs_sglang
def test_call_begin_forward_equals_upstream_without_selfspec(monkeypatch):
    import sglang.srt.layers.attention.flashinfer_backend as fbm

    class _NoKernel:  # create_flashinfer_kv_indices_triton[grid](..., kv_indices, stride): fill the torch.empty buffer
        def __getitem__(self, grid):
            return lambda *a, **k: a[5].fill_(7)

    monkeypatch.setattr(fbm, "create_flashinfer_kv_indices_triton", _NoKernel())
    monkeypatch.setattr(fbm, "get_forward_context", lambda: None)
    # the self-spec split-KV knobs switched ON (SELFSPEC_VERIFY_FIXED_SPLIT=64, SELFSPEC_VERIFY_NO_SPLIT=1)
    monkeypatch.setattr(fbm, "_SELFSPEC_VERIFY_FIXED_SPLIT", 64, raising=False)
    monkeypatch.setattr(fbm, "_SELFSPEC_VERIFY_NO_SPLIT", True, raising=False)
    upstream_cbf = _upstream(fbm, UPSTREAM_CALL_BEGIN_FORWARD, "call_begin_forward")
    for spec, fixed_split, use_ragged in itertools.product((False, True), (None, 1024), (False, True)):
        got = _run_call_begin_forward(fbm.FlashInferIndicesUpdaterPrefill.call_begin_forward, fbm, spec, False,
                                      fixed_split, use_ragged)
        want = _run_call_begin_forward(upstream_cbf, fbm, spec, False, fixed_split, use_ragged)
        assert got == want, (spec, fixed_split, use_ragged)
        assert "disable_split_kv" not in dict(got[-1][2])


@needs_selfspec
def test_call_begin_forward_split_knobs_apply_to_selfspec_verify_only(monkeypatch):
    import sglang.srt.layers.attention.flashinfer_backend as fbm

    cbf = fbm.FlashInferIndicesUpdaterPrefill.call_begin_forward
    upstream_cbf = _upstream(fbm, UPSTREAM_CALL_BEGIN_FORWARD, "call_begin_forward")
    monkeypatch.setattr(fbm, "_SELFSPEC_VERIFY_FIXED_SPLIT", 0)
    monkeypatch.setattr(fbm, "_SELFSPEC_VERIFY_NO_SPLIT", False)
    assert _run_call_begin_forward(cbf, fbm, True, True, None, False) == _run_call_begin_forward(
        upstream_cbf, fbm, True, True, None, False)  # knobs unset: the self-spec plan is upstream's too
    monkeypatch.setattr(fbm, "_SELFSPEC_VERIFY_FIXED_SPLIT", 64)
    monkeypatch.setattr(fbm, "_SELFSPEC_VERIFY_NO_SPLIT", True)
    kw = dict(_run_call_begin_forward(cbf, fbm, True, True, None, False)[-1][2])
    assert kw["fixed_split_size"] == 64 and kw["disable_split_kv"] is True


# --------------------------------------------------------------- ForwardBatch._compute_spec_mrope_positions (G11)
def _mrope_case(fbi, alg, mode, deltas, n):
    import torch

    bs = len(deltas)
    fb = object.__new__(fbi.ForwardBatch)
    fb.seq_lens = torch.full((bs,), 11)
    mm = [None if d is None else ns(mrope_position_delta=torch.tensor([[d]])) for d in deltas]
    batch = ns(
        forward_mode=mode,
        spec_info=ns(positions=torch.arange(bs * n) * 3 + 2),
        multimodal_inputs=mm,
        extend_seq_lens=[n] * bs,
    )
    return fb, ns(device="cpu", spec_algorithm=alg), batch


@needs_sglang
def test_spec_mrope_positions_equal_upstream_without_selfspec(monkeypatch):
    import torch

    import sglang.srt.model_executor.forward_batch_info as fbi
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as SA

    def selfspec_only(*a, **k):
        raise AssertionError("the SELFSPEC_MROPE_FIX branch ran outside a self-spec server")

    monkeypatch.setattr(fbi, "_SS_MROPE_FIX", True, raising=False)  # the self-spec switch ON in the module
    monkeypatch.setattr(fbi, "_ss_spec_mrope_positions", selfspec_only, raising=False)
    upstream_fn = _upstream(fbi, UPSTREAM_SPEC_MROPE, "_compute_spec_mrope_positions")
    modes = (fbi.ForwardMode.TARGET_VERIFY, fbi.ForwardMode.DRAFT_EXTEND, fbi.ForwardMode.DECODE)
    for alg, mode, deltas, n in itertools.product(
        (SA.EAGLE, SA.EAGLE3, SA.STANDALONE, SA.NGRAM), modes, ([None], [None, None], [5], [-3, None, 7]), (1, 4)
    ):
        fb, mr, batch = _mrope_case(fbi, alg, mode, deltas, n)
        fbi.ForwardBatch._compute_spec_mrope_positions(fb, mr, batch)
        ref, mr2, batch2 = _mrope_case(fbi, alg, mode, deltas, n)
        upstream_fn(ref, mr2, batch2)
        assert fb.mrope_positions.dtype == ref.mrope_positions.dtype
        assert torch.equal(fb.mrope_positions, ref.mrope_positions), (alg, mode, deltas, n)


@needs_selfspec
def test_spec_mrope_positions_selfspec_paths(monkeypatch):
    import torch

    import sglang.srt.model_executor.forward_batch_info as fbi
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as SA

    upstream_fn = _upstream(fbi, UPSTREAM_SPEC_MROPE, "_compute_spec_mrope_positions")
    mode = fbi.ForwardMode.TARGET_VERIFY
    monkeypatch.setattr(fbi, "_SS_MROPE_FIX", False)  # FAST=0: the self-spec server runs upstream's branch
    for deltas in ([None], [5], [-3, None, 7]):
        fb, mr, batch = _mrope_case(fbi, SA.SELFSPEC_DIFFUSION, mode, deltas, 33)
        fbi.ForwardBatch._compute_spec_mrope_positions(fb, mr, batch)
        ref, mr2, batch2 = _mrope_case(fbi, SA.SELFSPEC_DIFFUSION, mode, deltas, 33)
        upstream_fn(ref, mr2, batch2)
        assert torch.equal(fb.mrope_positions, ref.mrope_positions)
    monkeypatch.setattr(fbi, "_SS_MROPE_FIX", True)  # FAST=1: the self-spec branch
    hit = []
    monkeypatch.setattr(fbi, "_ss_spec_mrope_positions", lambda *a: hit.append(1) or torch.zeros(3, 33, dtype=torch.long))
    fb, mr, batch = _mrope_case(fbi, SA.SELFSPEC_DIFFUSION, mode, [5], 33)
    fbi.ForwardBatch._compute_spec_mrope_positions(fb, mr, batch)
    assert hit == [1]


# ------------------------------------------------------------------------------------- ServerArgs (G11)
def _server_args_stub(**overrides):
    import dataclasses

    from sglang.srt.server_args import ServerArgs

    values = {}
    for f in dataclasses.fields(ServerArgs):
        if f.default is not dataclasses.MISSING:
            values[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            values[f.name] = f.default_factory()
    values.update(model_path="/nonexistent", **overrides)
    return ns(**values)


@needs_selfspec
def test_selfspec_refuses_dllm_algorithm_and_dllm_alone_is_untouched():
    from sglang.srt.server_args import ServerArgs

    with pytest.raises(ValueError, match="--dllm-algorithm"):
        ServerArgs._handle_speculative_decoding(
            _server_args_stub(speculative_algorithm="SELFSPEC_DIFFUSION", dllm_algorithm="LowConfidence")
        )
    sa = _server_args_stub(speculative_algorithm="SELFSPEC_DIFFUSION")
    ServerArgs._handle_speculative_decoding(sa)
    assert sa.disable_overlap_schedule and sa.speculative_eagle_topk == 1
    sa = _server_args_stub(dllm_algorithm="LowConfidence")
    before = dict(vars(sa))
    ServerArgs._handle_speculative_decoding(sa)  # a dLLM server: only upstream's own default is filled in
    assert {k for k, v in vars(sa).items() if before[k] != v} <= {"speculative_moe_runner_backend"}
    assert sa.dllm_algorithm == "LowConfidence" and sa.speculative_algorithm is None


# ------------------------------------------------------------- remaining self-spec hooks sit behind self-spec (AST)
def _unguarded(module, match, guards):
    """Nodes of `module`'s source selected by match(node) that have no enclosing if / conditional expression / boolean
    operation mentioning one of `guards` (in its condition, or anywhere in the boolean operation)."""
    path = Path(importlib.import_module(module).__file__)
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    found, bad = 0, []
    for node in ast.walk(tree):
        if not match(node):
            continue
        found += 1
        a, ok = node, False
        while a in parent and not ok:
            a = parent[a]
            seg = None
            if isinstance(a, (ast.If, ast.IfExp, ast.While)):
                seg = ast.get_source_segment(src, a.test)
            elif isinstance(a, ast.BoolOp):
                seg = ast.get_source_segment(src, a)
            ok = seg is not None and any(g in seg for g in guards)
        if not ok:
            bad.append(f"{path.name}:{node.lineno}: {ast.get_source_segment(src, node)}")
    return found, bad


def _env_read(prefix):
    def match(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).startswith(prefix)
        )

    return match


def _getattr_of(*names):
    def match(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in names
        )

    return match


def _name_use(*names):
    def match(node):
        if isinstance(node, ast.Attribute) and node.attr in names:
            return True
        if isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, ast.Load):
            return True
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in names
        )

    return match


SELFSPEC_GUARDS = (
    ("sglang.srt.model_executor.model_runner", _env_read("DLLM_"), ("is_selfspec",)),
    ("sglang.srt.model_executor.cuda_graph_runner", _env_read("DLLM_"), ("is_selfspec",)),
    ("sglang.srt.model_executor.cuda_graph_runner", _env_read("SELFSPEC_"), ("is_selfspec",)),
    ("sglang.srt.layers.attention.flashinfer_backend", _getattr_of("ss_causal", "ss_seed_causal", "ss_shadow"),
     ("is_selfspec",)),
    ("sglang.srt.layers.attention.flashinfer_backend", _name_use("_SELFSPEC_VERIFY_FIXED_SPLIT", "_SELFSPEC_VERIFY_NO_SPLIT"),
     ("is_selfspec",)),
    ("sglang.srt.model_executor.forward_batch_info", _name_use("_ss_spec_mrope_positions"), ("is_selfspec_diffusion",)),
    ("sglang.srt.managers.tokenizer_manager", _name_use("_first_chunk_tokens", "get_decode_latency"), ("SELFSPEC_DIFFUSION",)),
)


@needs_selfspec
@pytest.mark.parametrize("module, match, guards", SELFSPEC_GUARDS, ids=[f"{m.rsplit('.', 1)[1]}-{i}" for i, (m, _, _) in
                                                                         enumerate(SELFSPEC_GUARDS)])
def test_selfspec_hooks_in_shared_modules_are_selfspec_gated(module, match, guards):
    found, bad = _unguarded(module, match, guards)
    assert found, f"no hook found in {module}: the test no longer matches the code"
    assert not bad, "\n".join(bad)


@needs_selfspec
@pytest.mark.parametrize("module", ["sglang.srt.model_executor.cuda_graph_runner",
                                    "sglang.srt.layers.attention.flashinfer_backend",
                                    "sglang.srt.model_executor.forward_batch_info",
                                    "sglang.srt.speculative.selfspec_diffusion_worker"])
def test_the_nomask_path_is_gone(module):
    # SELFSPEC_NOMASK (an older mask-free dual-graph path, default off and refused with the shipped settings) was removed:
    # no flag, dual graph key or causal channel from the worker to the attention backend is left.
    import importlib.util

    src = Path(importlib.util.find_spec(module).origin).read_text(encoding="utf-8")
    for name in ("SELFSPEC_NOMASK", "selfspec_nomask", "_nomask", "is_dual_causal_graph", "dllm_causal",
                 "dllm_ragged_causal", "selfspec_causal"):
        assert name not in src, (module, name)


# =========================================================== verbatim from SGLang v0.5.10 (commit 1519acf), reference

# CudaGraphRunner.capture (sglang/srt/model_executor/cuda_graph_runner.py), verbatim from SGLang v0.5.10
UPSTREAM_CAPTURE = r'''
    def capture(self) -> None:
        profile_context = empty_context()
        if self.enable_profile_cuda_graph:
            profile_context = self._init_profile_context_and_memory_record()

        def _capture_one_stream(stream_idx: Optional[int] = None):
            avail_mem = get_available_gpu_memory(
                self.model_runner.device,
                self.model_runner.gpu_id,
                empty_cache=False,
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            for i, bs in enumerate(capture_range):
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                    )

                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    num_tokens=bs * self.num_tokens_per_bs,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs, forward, stream_idx)
                    # For pd_multiplexing, we need to save the graph and output buffers
                    key = bs if stream_idx is None else f"{stream_idx}_{bs}"
                    self.graphs[key] = graph
                    self.output_buffers[key] = output_buffers

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            if not self.enable_pdmux:
                with graph_capture() as graph_capture_context, profile_context as prof:
                    self.stream = graph_capture_context.stream
                    _capture_one_stream()
            else:
                set_pdmux_status(False)
                for i, sg in enumerate(self.stream_groups):
                    with graph_capture(
                        stream=sg[1]
                    ) as graph_capture_context, profile_context as prof:
                        self.stream = graph_capture_context.stream
                        _capture_one_stream(i)

        if self.enable_profile_cuda_graph:
            self._post_process_after_profile(prof)
'''

# CudaGraphRunner.can_run (sglang/srt/model_executor/cuda_graph_runner.py), verbatim from SGLang v0.5.10
UPSTREAM_CAN_RUN = r'''
    def can_run(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.batch_size

        graph_key = cuda_graph_bs
        if self.enable_pdmux:
            graph_key = f"{get_current_stream_idx()}_{cuda_graph_bs}"

        is_bs_supported = (
            graph_key in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )
        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        is_ngram_supported = (
            (
                forward_batch.batch_size * self.num_tokens_per_bs
                == forward_batch.input_ids.numel()
            )
            if self.model_runner.spec_algorithm.is_ngram()
            else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and capture_hidden_mode_matches
            and is_ngram_supported
        )
'''

# FlashInferAttnBackend.forward_extend (sglang/srt/layers/attention/flashinfer_backend.py), verbatim from SGLang v0.5.10
UPSTREAM_FORWARD_EXTEND = r'''
    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()
        if not self.forward_metadata.use_ragged:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

            o = prefill_wrapper_paged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=not layer.is_cross_attention,
                sm_scale=layer.scaling,
                # Disable sliding window attention for multi-item scoring:
                # - Sliding window could cut across item boundaries, breaking semantic coherence
                # - Multi-item sequences need full attention to properly handle delimiter tokens
                # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
                #   provide more precise attention control than simple sliding windows
                # - Item-aware masking takes precedence over window-based masking
                window_left=(
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                ),
                logits_soft_cap=logits_soft_cap,
                # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
        else:
            # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
            # `forward_batch.token_to_kv_pool` for this layer. This enables attention over
            # previously cached context without re-materializing KV tensors (e.g., the
            # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
            if k is None and v is None:
                k = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
                v = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]
            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False
            if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
                save_kv_cache = False

            if self.forward_metadata.extend_no_prefix:
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

            else:
                if not self.is_dllm_model:
                    # TODO: design a better interface
                    # For other models, use causal attention for the ragged part as previously
                    causal = True

                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

                o, _ = merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
'''

# FlashInferIndicesUpdaterPrefill.call_begin_forward (sglang/srt/layers/attention/flashinfer_backend.py), verbatim from SGLang v0.5.10
UPSTREAM_CALL_BEGIN_FORWARD = r'''
    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            # Reserve extra space in kv_indices for a potential piecewise CUDA graph
            # dummy request (see below). Worst case: static_num_tokens extra pages.
            fwd_ctx = get_forward_context()
            pcg_num_tokens = fwd_ctx.num_tokens if fwd_ctx is not None else None
            extra_kv = pcg_num_tokens if pcg_num_tokens is not None else 0
            kv_indices = torch.empty(
                paged_kernel_lens_sum + extra_kv + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            # Piecewise CUDA graph padding: input_ids are padded to static_num_tokens,
            # so q.shape[0] == static_num_tokens but qo_indptr[-1] == actual tokens.
            # Append a dummy request for the padding tokens so that
            # qo_indptr[-1] == static_num_tokens, satisfying flashinfer's shape check
            # without corrupting the causal masks of real requests.
            # The dummy request's KV indices all point to slot 0 (a scratch location);
            # its attention output is discarded via the [:raw_num_tokens] slice in replay.
            bs_eff = bs
            # extend_num_tokens is a Python int (== sum of seq_lens - prefix_lens),
            # and paged_kernel_lens_sum is also a Python int (== kv_indptr[-1]),
            # so this block requires no CPU-GPU synchronisation.
            actual_qo_tokens = (
                fwd_ctx.forward_batch.extend_num_tokens if fwd_ctx is not None else None
            )
            if (
                pcg_num_tokens is not None
                and actual_qo_tokens is not None
                and pcg_num_tokens > actual_qo_tokens
            ):
                pad_tokens = pcg_num_tokens - actual_qo_tokens
                num_dummy_pages = (pad_tokens + self.page_size - 1) // self.page_size
                kv_start = (
                    paged_kernel_lens_sum  # equals kv_indptr[-1], no .item() needed
                )
                kv_indices[kv_start : kv_start + num_dummy_pages] = 0
                qo_indptr = torch.cat(
                    [qo_indptr, qo_indptr.new_tensor([pcg_num_tokens])]
                )
                kv_indptr = torch.cat(
                    [kv_indptr, kv_indptr.new_tensor([kv_start + num_dummy_pages])]
                )
                bs_eff = bs + 1

            custom_mask = None
        else:
            assert isinstance(spec_info, SpecInput)
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )
            bs_eff = bs

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters
        if multi_item_params is not None and multi_item_params.is_enabled():
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            # No multi-item scoring - use standard parameters
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs_eff],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=use_custom_mask,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
        )
'''

# ForwardBatch._compute_spec_mrope_positions (sglang/srt/model_executor/forward_batch_info.py), verbatim from SGLang v0.5.10
UPSTREAM_SPEC_MROPE = r'''
    def _compute_spec_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        # TODO support batched deltas
        batch_size = self.seq_lens.shape[0]
        device = model_runner.device
        mm_inputs = batch.multimodal_inputs

        if batch.forward_mode.is_draft_extend():  # draft_extend_after_decode
            mrope_deltas = []
            extend_lens = []
            for batch_idx in range(batch_size):
                extend_seq_len = batch.extend_seq_lens[batch_idx]
                extend_lens.append(extend_seq_len)
                mrope_delta = (
                    torch.zeros(1, dtype=torch.int64)
                    if mm_inputs[batch_idx] is None
                    else mm_inputs[batch_idx].mrope_position_delta.squeeze(0)
                )
                mrope_deltas.append(mrope_delta.to(device=device))
            position_chunks = torch.split(batch.spec_info.positions, extend_lens)
            mrope_positions_list = [
                pos_chunk + delta
                for pos_chunk, delta in zip(position_chunks, mrope_deltas)
            ]
            next_input_positions = (
                torch.cat(mrope_positions_list, dim=0).unsqueeze(0).repeat(3, 1)
            )

        else:  # target_verify or draft_decode
            seq_positions = batch.spec_info.positions.view(batch_size, -1)
            # Split text-only and mixed batches here because SpecV2 text-only batches can avoid an extra D2H.
            if all(mm_input is None for mm_input in mm_inputs):
                mrope_delta_tensor = torch.zeros(
                    (batch_size, 1), dtype=torch.int64, device=device
                )
            else:
                mrope_deltas = [
                    (
                        torch.zeros(1, dtype=torch.int64)
                        if mm_inputs[i] is None
                        else mm_inputs[i].mrope_position_delta.squeeze(0)
                    )
                    for i in range(batch_size)
                ]
                mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)
            next_input_positions = (
                (seq_positions + mrope_delta_tensor).flatten().unsqueeze(0).repeat(3, 1)
            )

        self.mrope_positions = next_input_positions
'''
