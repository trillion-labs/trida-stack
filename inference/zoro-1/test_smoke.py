#!/usr/bin/env python3
"""Zoro-1 nano-inference smoke tests — no GPU, no network. Run: python test_smoke.py

A safety net for contributors: checks the wiring (command construction, config
parsing, mode registry) without needing a served model.
"""
import sys
from pathlib import Path

import serve

HERE = Path(__file__).parent
CONFIGS = HERE / "configs"


class Args:
    def __init__(self, **kw):
        d = dict(model="dummy/model", mode="self-spec", port=30000, host="0.0.0.0",
                 tp=1, mem_fraction=0.8, max_running=32, cuda_graph_bs="1 2 4 8",
                 dtype="bfloat16", config=None, reasoning_parser=None,
                 dry_run=True, passthrough=[])
        d.update(kw)
        self.__dict__.update(d)


def test_modes_registry():
    # Zoro-1 only: no pure diffusion mode (the model does not support it), no other model families
    assert set(serve.MODES) == {"causal", "self-spec"}
    assert serve.DEFAULT_MODEL == "trillionlabs/Zoro-1"


def test_self_spec_is_the_card_setting():
    text = (CONFIGS / serve.MODES["self-spec"][1]).read_text()
    assert "block_size: 15" in text and "gen_block_size: 8" in text and "mask_id: 248077" in text


def test_causal_has_no_dllm_args():
    cmd = serve.build_command(Args(mode="causal"))
    assert "--dllm-algorithm" not in cmd
    assert "--model-path" in cmd and "--trust-remote-code" in cmd


def test_all_modes_wire_and_configs_exist():
    for mode, (algo, cfg, _extra) in serve.MODES.items():
        cmd = serve.build_command(Args(mode=mode))
        if algo is None:                      # causal: no dllm args
            assert "--dllm-algorithm" not in cmd
            continue
        assert algo in cmd, mode
        assert any(cfg in c for c in cmd), mode
        assert (CONFIGS / cfg).exists(), f"missing config {cfg} for mode {mode}"


def test_overrides_pass_through():
    cmd = serve.build_command(Args(port=31234, tp=2))
    assert "31234" in cmd and "2" in cmd


def test_reasoning_parser_optional():
    assert "--reasoning-parser" not in serve.build_command(Args())
    cmd = serve.build_command(Args(reasoning_parser="qwen3"))
    assert "--reasoning-parser" in cmd and "qwen3" in cmd


def test_decide_mode_launches_serve_decide():
    cmd = serve.build_command(Args(mode="decide", model="/ckpt", port=30001))
    assert cmd[1].endswith("serve_decide.py") and cmd[2] == "/ckpt"
    assert "sglang.launch_server" not in cmd and "--dllm-algorithm" not in cmd
    assert "30001" in cmd


def test_configs_have_required_keys():
    for y in CONFIGS.glob("*.yaml"):
        text = y.read_text()
        assert "mask_id:" in text, f"{y.name} missing mask_id"
        assert "block_size:" in text, f"{y.name} missing block_size"


def _raw_ckpt(root):
    import json
    raw = root / "zoro-1"
    raw.mkdir()
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "vision_config": {}}
    (raw / "config.json").write_text(json.dumps(cfg))
    (raw / "model-00001-of-00002.safetensors").write_bytes(b"w")
    (raw / "tokenizer.json").write_text("{}")
    return raw


def test_for_sglang_shapes_once_and_reuses(tmp_path):
    import json
    raw = _raw_ckpt(tmp_path)
    cache = tmp_path / "cache"
    out = Path(serve.for_sglang(str(raw), cache))
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3_5DLLMForConditionalGeneration"] and "vision_config" in cfg
    assert (out / "model-00001-of-00002.safetensors").is_symlink() and (out / "tokenizer.json").exists()
    assert serve.for_sglang(str(raw), cache) == str(out)          # same dir next time, no re-shape
    assert serve.for_sglang(str(out), cache) == str(out)          # an already-shaped dir is served as is


def test_dry_run_and_no_shape_leave_the_model_alone(monkeypatch):
    called = []
    monkeypatch.setattr(serve, "for_sglang", lambda m, cache=None: called.append(m) or m)
    for extra in (["--dry-run"], ["--no-shape", "--dry-run"]):
        monkeypatch.setattr(sys, "argv", ["serve.py", "trillionlabs/Zoro-1", *extra])
        serve.main()
    monkeypatch.setattr(sys, "argv", ["serve.py", "trillionlabs/Zoro-1", "--mode", "decide", "--dry-run"])
    serve.main()
    assert called == []


def _launch(monkeypatch, argv):
    """Run serve.main() up to the exec and return the command it would exec."""
    import pytest
    launched = []

    def fake_exec(prog, cmd):           # the real execvp never returns
        launched.append(cmd)
        raise SystemExit(0)
    monkeypatch.setattr(serve.os, "execvp", fake_exec)
    monkeypatch.setattr(sys, "argv", ["serve.py", *argv])
    with pytest.raises(SystemExit):
        serve.main()
    assert len(launched) == 1
    return launched[0]


def test_no_shape_skips_shaping_without_dry_run(monkeypatch):
    called = []
    monkeypatch.setattr(serve, "for_sglang", lambda m, cache=None: called.append(m) or m)
    _launch(monkeypatch, ["trillionlabs/Zoro-1", "--no-shape"])
    assert called == []
    _launch(monkeypatch, ["trillionlabs/Zoro-1"])
    assert called == ["trillionlabs/Zoro-1"]


def test_decide_default_model_reaches_serve_decide_as_a_local_dir(monkeypatch, tmp_path):
    # serve_decide.py opens <ckpt>/jev_head.pt, so it needs a directory, not the HF id
    import types
    snap = tmp_path / "snapshot"
    snap.mkdir()
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=lambda repo: str(snap)))
    monkeypatch.delenv("TRIDA_MODEL", raising=False)
    monkeypatch.delenv("TRIDA_DECIDE_MODEL", raising=False)
    cmd = _launch(monkeypatch, ["--mode", "decide"])
    ckpt = cmd[[i for i, c in enumerate(cmd) if c.endswith("serve_decide.py")][0] + 1]
    assert Path(ckpt).is_dir() and Path(ckpt) == snap.resolve()


def test_for_sglang_shapes_a_flat_causal_lm_checkpoint(tmp_path):
    import json
    raw = _raw_ckpt(tmp_path)
    (raw / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"],
                                                 "model_type": "qwen3_5_text", "hidden_size": 8}))
    cfg = json.loads((Path(serve.for_sglang(str(raw), tmp_path / "cache")) / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3_5DLLMForConditionalGeneration"]
    assert cfg["text_config"]["model_type"] == "qwen3_5_text" and cfg["text_config"]["hidden_size"] == 8
    assert "architectures" not in cfg["text_config"]


def test_an_incomplete_shim_is_rebuilt(tmp_path):
    # a crash between writing config.json and the weight links must not leave a shim that is reused forever
    raw = _raw_ckpt(tmp_path)
    cache = tmp_path / "cache"
    out = Path(serve.for_sglang(str(raw), cache))
    (out / "model-00001-of-00002.safetensors").unlink()
    assert serve.for_sglang(str(raw), cache) == str(out)
    assert (out / "model-00001-of-00002.safetensors").is_symlink()


def test_a_dangling_shim_link_is_rebuilt(tmp_path):
    raw = _raw_ckpt(tmp_path)
    cache = tmp_path / "cache"
    out = Path(serve.for_sglang(str(raw), cache))
    (raw / "tokenizer.json").unlink()                    # e.g. the HF cache was cleaned
    assert serve.for_sglang(str(raw), cache) == str(out)
    assert not (out / "tokenizer.json").is_symlink()     # rebuilt from what the checkpoint now has


def test_a_changed_local_checkpoint_gets_a_fresh_shim(tmp_path):
    import json
    raw = _raw_ckpt(tmp_path)
    cfg = {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "hidden_size": 8}
    (raw / "config.json").write_text(json.dumps(cfg))
    cache = tmp_path / "cache"
    first = serve.for_sglang(str(raw), cache)
    (raw / "config.json").write_text(json.dumps({**cfg, "hidden_size": 16}))   # re-saved in place
    second = serve.for_sglang(str(raw), cache)
    assert second != first
    assert json.loads((Path(second) / "config.json").read_text())["text_config"]["hidden_size"] == 16


def test_shim_build_leaves_no_temp_dirs(tmp_path):
    raw = _raw_ckpt(tmp_path)
    cache = tmp_path / "cache"
    out = Path(serve.for_sglang(str(raw), cache))
    assert [p.name for p in cache.iterdir()] == [out.name]


def test_a_shim_another_replica_finished_first_is_kept(monkeypatch, tmp_path):
    # Two replicas start together and both miss the cache. The one that finishes second must use the
    # winner's shim, never delete it: the winner's SGLang may already be loading from it.
    import types
    raw = _raw_ckpt(tmp_path)
    name = Path(serve.for_sglang(str(raw), tmp_path / "probe")).name
    cache = tmp_path / "cache"
    winner = cache / name
    real = serve._shape_tool()
    seen = {}

    def shape_while_another_replica_wins(raw_dir, out_dir, copy=False):
        if not winner.exists():
            real.shape(raw_dir, winner)                       # the other replica's finished shim
            seen["inode"] = (winner / "config.json").stat().st_ino
        return real.shape(raw_dir, out_dir)

    monkeypatch.setattr(serve, "_shape_tool", lambda: types.SimpleNamespace(
        is_shaped=real.is_shaped, carried_names=real.carried_names, shape=shape_while_another_replica_wins))
    assert serve.for_sglang(str(raw), cache) == str(winner)
    assert (winner / "config.json").stat().st_ino == seen["inode"]   # not deleted and rebuilt
    assert [p.name for p in cache.iterdir()] == [name]               # no temp dirs left behind


def test_temp_dirs_are_unique_per_build_not_per_pid(monkeypatch, tmp_path):
    # PID 1 in two containers sharing one cache must not build into the same temp dir
    raw = _raw_ckpt(tmp_path)
    monkeypatch.setattr(serve.os, "getpid", lambda: 1)
    made = []
    real_mkdtemp = serve.tempfile.mkdtemp
    monkeypatch.setattr(serve.tempfile, "mkdtemp", lambda **kw: made.append(real_mkdtemp(**kw)) or made[-1])
    serve.for_sglang(str(raw), tmp_path / "a")
    serve.for_sglang(str(raw), tmp_path / "b")
    assert len(made) == 2 and Path(made[0]).name != Path(made[1]).name


def main():
    import inspect
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fixtured = [t for t in tests if inspect.signature(t).parameters]   # need pytest fixtures: run under pytest
    tests = [t for t in tests if t not in fixtured]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed ({len(fixtured)} more under `pytest test_smoke.py`)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
