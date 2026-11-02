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
