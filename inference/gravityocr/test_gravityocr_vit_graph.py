# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""GravityOCR SGLang patch: CPU unit tests of the GLM-OCR ViT CUDA-graph runner.

Runs on CPU with torch and pytest only (no GPU, no SGLang install). The runner,
sglang/python/sglang/srt/multimodal/glm_vit_graph_runner.py, imports only the standard library and
torch at module level, so it is loaded by path; a fake ViT and a prefilled graphs dict stand in for
the captured model. Covered:
  - the settings parser (SS_GLM_VIT_GRAPH_MAXM, SS_GLM_VIT_EAGER_LIST, SS_GLM_VIT_GRAPH_CHECK) and
    the settings line;
  - run() dispatch (graph replay / eager-list fallback / original eager forward) and its counters,
    the replay's static-input loads and output copy;
  - the single-caller guard: a call from another thread never touches the graphs;
  - setup(): a failing capture releases every partial graph and buffer and falls back to the
    eager forward, an unusable CUDA context re-raises the original error, a MAXM beyond the rotary
    table is refused before anything is allocated;
  - the module docstring states the launcher default.
With SGLang installed (the patched server venv) three more checks run: GlmVitGraphRunner._body equals
GlmOcrVisionModel._ss_forward_orig after its prologue (AST), layers/attention/vision.py parses the
variable with the same strict parser, and GlmOcrForConditionalGeneration falls back to its eager
forward when the capture fails.
"""

import ast
import importlib.util
import os
import subprocess
import sys
import threading
import types
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "sglang" / "python" / "sglang" / "srt" / "multimodal" / "glm_vit_graph_runner.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vg = _load(RUNNER_PATH, "_gravityocr_glm_vit_graph_runner")
MAXM, EAGER, CHECK = "SS_GLM_VIT_GRAPH_MAXM", "SS_GLM_VIT_EAGER_LIST", "SS_GLM_VIT_GRAPH_CHECK"


def _sglang_root():
    try:
        spec = importlib.util.find_spec("sglang")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(list(spec.submodule_search_locations)[0])
    return root if (root / "srt" / "models" / "glm_ocr.py").is_file() else None


SGLANG_ROOT = _sglang_root()
needs_sglang = pytest.mark.skipif(SGLANG_ROOT is None, reason="SGLang (the patched server venv) is not installed")


# =============================================================================================== fakes
class FakeRotary:
    def __init__(self, rows=8192):
        self.cos_sin_cache = torch.zeros(rows, 2)


class FakeVis:
    """The attributes of GlmOcrVisionModel the runner reads, on CPU."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, rows=8192):
        self.rotary_pos_emb = FakeRotary(rows)
        self.blocks = []
        self.orig_calls = []

    def rot_pos_emb(self, grid_thw):
        n = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
        cos = torch.arange(n * 3, dtype=torch.float32).view(n, 3) / 7.0
        return cos, -cos, None

    def _ss_forward_orig(self, x, grid_thw):
        self.orig_calls.append((threading.get_ident(), tuple(x.shape), grid_thw.tolist()))
        return torch.full((x.shape[0] // 4, 5), 3.0)


class FakeGraph:
    """Stands in for torch.cuda.CUDAGraph: replay() runs `fn` (which reads the runner's static inputs)."""

    def __init__(self, fn=None):
        self.fn = fn
        self.n_replay = 0

    def replay(self):
        self.n_replay += 1
        if self.fn is not None:
            self.fn()


def _settings(maxm=64, eager_list=True, check=True):
    return vg.VitGraphSettings(maxm=maxm, requested=maxm, eager_list=eager_list, check=check)


def _runner(maxm=64, eager_list=True, captured=(16, 64), rows=8192):
    r = vg.GlmVitGraphRunner(FakeVis(rows), _settings(maxm, eager_list))
    for M in captured:
        r.graphs[M] = (FakeGraph(), torch.zeros(M // 4, 5), torch.tensor([0, M], dtype=torch.int32))
    return r


def _spy(r):
    """Replace the three execution paths with recorders; run() only has to pick one."""
    log = []
    r._replay = lambda x, g, M: log.append(("replay", M)) or torch.zeros(1)
    r._eager_list = lambda x, g: log.append(("eager_list", tuple(g.shape))) or torch.zeros(1)
    r._eager_orig = lambda x, g: log.append(("eager_orig", tuple(g.shape))) or torch.zeros(1)
    return log


def _x(rows):
    return torch.randn(rows, 6)


def _g(*rows):
    return torch.tensor([list(r) for r in rows])


# =============================================================================================== settings (U12)
@pytest.mark.parametrize(
    "raw,maxm,requested",
    [(None, 0, 0), ("", 0, 0), ("0", 0, 0), ("2048", 2048, 2048), ("2050", 2048, 2050), ("3", 0, 3),
     ("0002048", 2048, 2048), ("4", 4, 4), ("16384", 16384, 16384)],
)
def test_maxm_parses_digits_and_rounds_down_to_a_multiple_of_4(raw, maxm, requested):
    env = {} if raw is None else {MAXM: raw}
    s = vg.vit_graph_settings(env)
    assert (s.maxm, s.requested, s.eager_list, s.check) == (maxm, requested, True, True)
    assert vg.glm_vit_graph_maxm(env) == maxm


@pytest.mark.parametrize("raw", ["2k", "-4", " 8", "8 ", "+8", "1e3", "0x10", "2048.0", "２０４８", "true", "off"])
def test_malformed_maxm_is_an_error_naming_the_variable(raw):
    with pytest.raises(ValueError, match=MAXM):
        vg.vit_graph_settings({MAXM: raw})
    with pytest.raises(ValueError, match=MAXM):
        vg.glm_vit_graph_maxm({MAXM: raw})


@pytest.mark.parametrize("name", [EAGER, CHECK])
@pytest.mark.parametrize("raw,value", [(None, True), ("", True), ("1", True), ("0", False)])
def test_flags_accept_exactly_0_and_1(name, raw, value):
    env = {MAXM: "2048"}
    if raw is not None:
        env[name] = raw
    s = vg.vit_graph_settings(env)
    assert getattr(s, "eager_list" if name == EAGER else "check") is value


@pytest.mark.parametrize("name", [EAGER, CHECK])
@pytest.mark.parametrize("raw", ["true", "yes", "on", "2", " 1", "1 ", "01", "-1"])
def test_malformed_flag_is_an_error_naming_the_flag(name, raw):
    with pytest.raises(ValueError, match=name):
        vg.vit_graph_settings({MAXM: "2048", name: raw})


def test_glm_vit_graph_maxm_reads_only_maxm():
    # layers/attention/vision.py calls it at import for every vision model: a bad flag must not break those
    assert vg.glm_vit_graph_maxm({MAXM: "2048", EAGER: "true", CHECK: "yes"}) == 2048


def test_settings_default_to_the_process_environment(monkeypatch):
    monkeypatch.setenv(MAXM, "2k")
    with pytest.raises(ValueError, match="'2k'"):
        vg.vit_graph_settings()
    monkeypatch.setenv(MAXM, "1024")
    monkeypatch.setenv(EAGER, "0")
    monkeypatch.delenv(CHECK, raising=False)
    assert vg.vit_graph_settings() == vg.VitGraphSettings(1024, 1024, False, True)


def test_settings_line_is_printed_once_and_says_what_was_parsed(monkeypatch, capsys):
    monkeypatch.setattr(vg, "_settings_reported", False)
    vg.report_settings(vg.vit_graph_settings({MAXM: "2050", CHECK: "0"}))
    vg.report_settings(vg.vit_graph_settings({MAXM: "8"}))
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "[ss-vit-graph] settings: SS_GLM_VIT_GRAPH_MAXM=2048 (rounded down from 2050 to a multiple of 4) -> graphs for "
        "M = 4..2048 step 4 (512); SS_GLM_VIT_EAGER_LIST=1; SS_GLM_VIT_GRAPH_CHECK=0"
    ]
    assert "-> off" in vg.describe_settings(vg.vit_graph_settings({}))
    assert "3 rounds down to 0 -> off" in vg.describe_settings(vg.vit_graph_settings({MAXM: "3"}))


# =============================================================================================== run() dispatch (U13)
def test_single_image_with_a_captured_m_replays():
    r = _runner()
    log = _spy(r)
    r.run(_x(16), _g((1, 4, 4)))
    r.run(_x(64), _g((1, 8, 8)))
    r.run(_x(64), _g((1, 2, 32)))  # another grid with the same M: cos / sin are graph inputs
    assert log == [("replay", 16), ("replay", 64), ("replay", 64)]
    assert (r.n_replay, r.n_eager_list, r.n_eager_orig) == (3, 0, 0)


@pytest.mark.parametrize(
    "rows,grid,why",
    [
        (32, ((2, 4, 4),), "t > 1"),
        (24, ((1, 4, 6),), "M not captured"),
        (128, ((1, 8, 16),), "M above MAXM"),
        (20, ((1, 4, 4),), "row count != M"),
        (20, ((1, 4, 4), (1, 2, 2)), "two images"),
    ],
)
@pytest.mark.parametrize("eager_list", [True, False])
def test_everything_else_takes_the_eager_paths(rows, grid, why, eager_list):
    r = _runner(eager_list=eager_list)
    log = _spy(r)
    r.run(_x(rows), _g(*grid))
    want = "eager_list" if eager_list else "eager_orig"
    assert log == [(want, (len(grid), 3))], why
    assert (r.n_replay, r.n_eager_list, r.n_eager_orig) == (0, int(eager_list), int(not eager_list))


def test_no_graphs_left_means_eager_list_for_every_call():
    r = _runner(captured=())
    log = _spy(r)
    r.run(_x(16), _g((1, 4, 4)))
    assert log == [("eager_list", (1, 3))]


def test_usage_line_after_the_1st_10th_100th_and_every_1000th_call(capsys):
    r = _runner()
    _spy(r)
    for _ in range(1001):
        r.run(_x(16), _g((1, 4, 4)))
    lines = [line for line in capsys.readouterr().out.splitlines() if "usage" in line]
    assert [line.split("calls=")[1].split()[0] for line in lines] == ["1", "10", "100", "1000"]
    assert lines[-1].endswith("calls=1000 replay=1000 eager_list=0 eager_orig=0")


def test_replay_loads_the_static_inputs_and_returns_a_copy_of_the_graph_output():
    r = _runner(captured=())
    r.X, r.C, r.S = torch.zeros(64, 6), torch.zeros(64, 6), torch.zeros(64, 6)
    out = torch.zeros(4, 6)
    seen = {}

    def fn():  # what the "graph" computes from its static inputs
        seen["X"], seen["C"], seen["S"] = r.X[:16].clone(), r.C[:16].clone(), r.S[:16].clone()
        out.copy_(r.X[:16].view(4, 4, 6).sum(1))

    graph = FakeGraph(fn)
    r.graphs[16] = (graph, out, torch.tensor([0, 16], dtype=torch.int32))
    x, g = _x(16), _g((1, 4, 4))
    y = r.run(x, g)
    cos, sin, _ = r.vis.rot_pos_emb(g)
    assert graph.n_replay == 1 and r.n_replay == 1
    assert torch.equal(seen["X"], x)
    assert torch.equal(seen["C"], torch.cat([cos, cos], -1)) and torch.equal(seen["S"], torch.cat([sin, sin], -1))
    assert torch.equal(y, x.view(4, 4, 6).sum(1))
    out.fill_(-1)  # the next replay (any M: one shared pool) overwrites the graph output ...
    assert torch.equal(y, x.view(4, 4, 6).sum(1))  # ... but not what run() returned


def test_eager_orig_is_the_models_original_forward():
    r = _runner(eager_list=False, captured=())
    y = r.run(_x(24), _g((1, 4, 6)))
    assert r.vis.orig_calls == [(threading.get_ident(), (24, 6), [[1, 4, 6]])]
    assert torch.equal(y, torch.full((6, 5), 3.0))


# =============================================================================================== single caller (U7)
def test_a_call_from_another_thread_never_touches_the_graphs(capsys):
    r = _runner()
    graph = r.graphs[16][0]
    r.X, r.C, r.S = torch.zeros(64, 6), torch.zeros(64, 6), torch.zeros(64, 6)
    res = {}

    def other():
        res["tid"] = threading.get_ident()
        res["y"] = r.run(_x(16), _g((1, 4, 4)))
        res["y2"] = r.run(_x(16), _g((1, 4, 4)))

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert graph.n_replay == 0 and r.n_replay == 0
    assert torch.count_nonzero(r.X) == 0  # static inputs untouched
    assert [c[0] for c in r.vis.orig_calls] == [res["tid"], res["tid"]]
    assert (r.n_eager_orig, r.n_other_thread) == (2, 2)
    warn = [line for line in capsys.readouterr().out.splitlines() if "not the capture thread" in line]
    assert len(warn) == 1  # said once
    r.run(_x(16), _g((1, 4, 4)))  # the owner thread still replays
    assert graph.n_replay == 1


def test_the_owner_is_the_thread_that_built_the_runner():
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", _runner()))
    t.start()
    t.join()
    r = box["r"]
    _spy(r)
    r.run(_x(16), _g((1, 4, 4)))  # this (main) thread did not build it
    assert (r.n_replay, r.n_eager_orig, r.n_other_thread) == (0, 1, 1)


# =============================================================================================== setup(): capture failure (U3)
class FakeCudaGraphObj:
    pass


@pytest.fixture
def fake_cuda(monkeypatch):
    """torch.cuda.synchronize / empty_cache recorders (CPU torch has no CUDA)."""
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: calls.append("empty_cache"))
    return calls


def _failing_capture(r, exc, after=3, live=None):
    """_capture_all that allocates the static buffers and `after` graphs, then fails while capturing the next one."""

    def capture_all(dev):
        r.X, r.C, r.S = torch.zeros(r.maxm, 6), torch.zeros(r.maxm, 6), torch.zeros(r.maxm, 6)
        r.pool = ("pool", 1)
        for M in range(r.maxm, r.maxm - 4 * after, -4):
            r.graphs[M] = (FakeCudaGraphObj(), torch.zeros(M // 4, 5), torch.tensor([0, M]))
        gr = FakeCudaGraphObj()  # the half-captured graph: referenced only by this frame when the error leaves it
        if live is not None:
            live.append(weakref.ref(gr))
        raise exc

    r._capture_all = capture_all


def test_capture_failure_releases_everything_and_falls_back(fake_cuda, capsys, monkeypatch):
    r = vg.GlmVitGraphRunner(FakeVis(), _settings(maxm=2048))
    monkeypatch.setattr(r, "_fa3_backend_name", lambda: None)
    live = []
    partial = []
    _failing_capture(r, torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB.\nsecond line"),
                     live=live)
    real_empty = fake_cuda.append

    def empty_cache_check(*a, **k):
        # by the time the cache is emptied, nothing may hold the half-captured graph, and no exception may be in flight: the
        # release runs after the except block (inside it the handled exception's traceback still pins capture()'s frames)
        partial.append((live[0]() is None, sys.exc_info()[0] is None))
        real_empty("empty_cache")

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache_check)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        assert r.setup() is False
        assert torch.get_default_dtype() == torch.float64  # capture() restores the caller's default dtype
    finally:
        torch.set_default_dtype(prev)
    assert r.graphs == {} and r.X is None and r.C is None and r.S is None and r.pool is None
    assert fake_cuda == ["synchronize", "empty_cache"] and partial == [(True, True)]
    out = capsys.readouterr().out
    assert ("[ss-vit-graph] OFF: capture failed (OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.); "
            "released the partial graphs and buffers") in out
    assert "using the original eager forward" in out and "second line" not in out


@pytest.mark.parametrize("exc", [RuntimeError("CUDA error: operation not permitted when stream is capturing"),
                                 AssertionError("x"), ImportError("no fa3")])
def test_any_capture_error_falls_back(fake_cuda, exc, monkeypatch, capsys):
    r = vg.GlmVitGraphRunner(FakeVis(), _settings(maxm=64))
    monkeypatch.setattr(r, "_fa3_backend_name", lambda: None)
    _failing_capture(r, exc, after=1)
    assert r.setup() is False and r.graphs == {}
    assert f"OFF: capture failed ({type(exc).__name__}: " in capsys.readouterr().out


def test_unusable_cuda_context_reraises_the_original_error(monkeypatch):
    r = vg.GlmVitGraphRunner(FakeVis(), _settings(maxm=64))
    monkeypatch.setattr(r, "_fa3_backend_name", lambda: None)
    oom = torch.cuda.OutOfMemoryError("CUDA out of memory")
    _failing_capture(r, oom)

    def broken(*a, **k):
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(torch.cuda, "synchronize", broken)
    with pytest.raises(torch.cuda.OutOfMemoryError) as info:
        r.setup()
    assert info.value is oom and "device-side assert" in str(info.value.__context__)


def test_successful_capture_installs_the_runner(fake_cuda, monkeypatch):
    r = vg.GlmVitGraphRunner(FakeVis(), _settings(maxm=64))
    monkeypatch.setattr(r, "_fa3_backend_name", lambda: None)
    monkeypatch.setattr(r, "_capture_all", lambda dev: r.graphs.update({64: (FakeGraph(), torch.zeros(16, 5), None)}))
    assert r.setup() is True and list(r.graphs) == [64]
    assert fake_cuda == []  # nothing released


def test_nothing_usable_after_the_self_check_falls_back(fake_cuda, monkeypatch, capsys):
    r = vg.GlmVitGraphRunner(FakeVis(), _settings(maxm=64))
    monkeypatch.setattr(r, "_fa3_backend_name", lambda: None)

    def all_dropped(dev):
        r.X = torch.zeros(64, 6)
        r.eager_list_ok = False

    monkeypatch.setattr(r, "_capture_all", all_dropped)
    assert r.setup() is False and r.X is None
    assert "OFF: no graph passed the self-check" in capsys.readouterr().out


def test_non_fa3_backend_is_off_without_a_failure_message(fake_cuda, monkeypatch, capsys):
    r = vg.GlmVitGraphRunner(FakeVis(), _settings(maxm=64))
    monkeypatch.setattr(r, "_fa3_backend_name", lambda: "VisionSdpaAttention")
    monkeypatch.setattr(r, "_capture_all", lambda dev: pytest.fail("must not capture"))
    assert r.setup() is False
    out = capsys.readouterr().out
    assert "OFF: ViT attention backend is VisionSdpaAttention, graphs need fa3" in out
    assert "capture failed" not in out and "self-check" not in out
    assert fake_cuda == []  # nothing was allocated, nothing to release


@pytest.mark.parametrize("maxm,ok", [(16384, True), (16388, False), (40000000, False)])
def test_maxm_beyond_the_rotary_table_is_refused_before_any_allocation(fake_cuda, monkeypatch, maxm, ok):
    r = vg.GlmVitGraphRunner(FakeVis(rows=8192), _settings(maxm=maxm))
    monkeypatch.setattr(r, "_fa3_backend_name", lambda: None)
    monkeypatch.setattr(r, "_capture_all", lambda dev: r.graphs.update({4: (FakeGraph(), torch.zeros(1, 5), None)}))
    if ok:
        assert r.setup() is True
    else:
        with pytest.raises(ValueError, match="SS_GLM_VIT_GRAPH_MAXM=.*at most 16384"):
            r.setup()
        assert r.graphs == {} and fake_cuda == []


# =============================================================================================== docs (U4) and leftovers (U14)
def test_docstring_states_the_launcher_default():
    doc = vg.__doc__
    assert "Off unless SS_GLM_VIT_GRAPH_MAXM > 0" in doc and "2048 under FAST=1" in doc
    assert "default OFF" not in doc
    for name in (MAXM, EAGER, CHECK):
        assert name in doc


def test_no_unused_d_in_default():
    assert not hasattr(vg.GlmVitGraphRunner, "D_IN_DEFAULT")


# =============================================================================================== with SGLang installed
def _method(tree, cls, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


class _Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node


@needs_sglang
def test_body_matches_the_models_forward_after_its_prologue():
    """KEEP IN SYNC guard: _body must be GlmOcrVisionModel._ss_forward_orig from patch_embed on."""
    body = _method(ast.parse(RUNNER_PATH.read_text()), "GlmVitGraphRunner", "_body")
    orig = _method(ast.parse((SGLANG_ROOT / "srt/models/glm_ocr.py").read_text()), "GlmOcrVisionModel",
                   "_ss_forward_orig")
    ren = _Rename({"vis": "self", "cu_list": "cu_seqlens", "cos": "rotary_pos_emb_cos", "sin": "rotary_pos_emb_sin"})
    b_stmts = [ast.dump(ren.visit(s)) for s in body.body
               if not (isinstance(s, ast.Assign) and ast.unparse(s) == "vis = self.vis")]
    prologue = {"rotary_pos_emb_cos", "rotary_pos_emb_sin", "image_type_ids", "cu_seqlens"}
    o_stmts = []
    for s in orig.body[1:]:  # [0] = x.to(device, dtype): the callers' cast
        targets = {n.id for t in getattr(s, "targets", []) for n in ast.walk(t) if isinstance(n, ast.Name)}
        if targets & prologue:
            continue
        o_stmts.append(ast.dump(s))
    assert ast.unparse(orig.body[0]) == "x = x.to(device=self.device, dtype=self.dtype)"
    assert b_stmts == o_stmts


def _python_with_sglang(code, env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SS_GLM_")}
    env.update(env_extra, CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)


@needs_sglang
@pytest.mark.parametrize("raw,want", [("2048", "True"), ("2", "False"), ("", "False"), ("2k", None)])
def test_vision_module_uses_the_strict_parser(raw, want):
    p = _python_with_sglang("import sglang.srt.layers.attention.vision as v; print(v._SS_VIT_LIST)", {MAXM: raw})
    if want is None:
        assert p.returncode != 0 and "ValueError: SS_GLM_VIT_GRAPH_MAXM='2k'" in p.stderr, p.stderr[-2000:]
    else:
        assert p.returncode == 0 and p.stdout.strip().splitlines()[-1] == want, p.stderr[-2000:]


@pytest.fixture
def glm_ocr_model(fake_cuda, monkeypatch):
    if SGLANG_ROOT is None:
        pytest.skip("SGLang (the patched server venv) is not installed")
    glm_ocr = pytest.importorskip("sglang.srt.models.glm_ocr")
    runner_mod = sys.modules.get("sglang.srt.multimodal.glm_vit_graph_runner")
    if runner_mod is None:
        runner_mod = pytest.importorskip("sglang.srt.multimodal.glm_vit_graph_runner")
    monkeypatch.setattr(runner_mod, "_settings_reported", False, raising=False)
    visual = types.SimpleNamespace(_ss_vit=None, device=types.SimpleNamespace(type="cuda"),
                                   rotary_pos_emb=FakeRotary(), blocks=[])
    model = types.SimpleNamespace(visual=visual, use_data_parallel=False)
    return glm_ocr.GlmOcrForConditionalGeneration._ss_maybe_capture_vit_graphs, model, runner_mod


def test_model_keeps_the_eager_forward_when_the_capture_fails(glm_ocr_model, monkeypatch, capsys):
    maybe_capture, model, runner_mod = glm_ocr_model
    monkeypatch.setenv(MAXM, "2048")

    def oom(self):
        self.X = torch.zeros(4, 4)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 94.08 GiB.")

    monkeypatch.setattr(runner_mod.GlmVitGraphRunner, "capture", oom)
    maybe_capture(model)  # must not raise: the graphs are optional
    assert model.visual._ss_vit is None
    out = capsys.readouterr().out
    assert "[ss-vit-graph] settings: SS_GLM_VIT_GRAPH_MAXM=2048" in out
    assert "[ss-vit-graph] OFF: capture failed (OutOfMemoryError: CUDA out of memory." in out


def test_model_refuses_a_malformed_setting(glm_ocr_model, monkeypatch):
    maybe_capture, model, _ = glm_ocr_model
    monkeypatch.setenv(MAXM, "2k")
    with pytest.raises(ValueError, match="SS_GLM_VIT_GRAPH_MAXM='2k'"):
        maybe_capture(model)
    monkeypatch.setenv(MAXM, "2048")
    monkeypatch.setenv(EAGER, "true")
    with pytest.raises(ValueError, match="SS_GLM_VIT_EAGER_LIST='true'"):
        maybe_capture(model)
    assert model.visual._ss_vit is None
