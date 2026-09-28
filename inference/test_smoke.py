#!/usr/bin/env python3
"""nano-inference smoke tests — no GPU, no network. Run: python test_smoke.py

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
        d = dict(model="dummy/model", mode="diffusion", port=30000, host="0.0.0.0",
                 tp=1, mem_fraction=0.8, max_running=32, cuda_graph_bs="1 2 4 8",
                 dtype="bfloat16", config=None, reasoning_parser=None,
                 dry_run=True, passthrough=[])
        d.update(kw)
        self.__dict__.update(d)


def test_modes_registry():
    assert {"causal", "diffusion", "self-spec", "sdar", "llada2-0",
            "llada2-1-speed", "llada2-1-quality"} <= set(serve.MODES)


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


def test_configs_have_required_keys():
    for y in CONFIGS.glob("*.yaml"):
        text = y.read_text()
        assert "mask_id:" in text, f"{y.name} missing mask_id"
        assert "block_size:" in text, f"{y.name} missing block_size"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
