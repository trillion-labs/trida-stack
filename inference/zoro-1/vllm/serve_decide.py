#!/usr/bin/env python3
"""Serve /v1/decide and chat on stock vLLM 0.32.x, without the diffusion plugin."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve_decide import create_app  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="local checkpoint directory with jev_head.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--context-length", type=int, default=16384)
    ap.add_argument("--mem-fraction", type=float, default=0.5,
                    help="GPU memory fraction for the single generation runner (default: 0.5)")
    ap.add_argument("--max-concurrency", type=int, choices=(1, 2, 4), default=4,
                    help="shared generation/capture limit (default: 4; 1 retains serial execution)")
    ap.add_argument("--decision-max-state", type=int, default=7552)
    ap.add_argument("--decision-max-row", type=int, default=8192)
    ap.add_argument("--reasoning-token-limit", type=int, default=4096,
                    help="maximum decision reasoning budget (1-8192; default: 4096)")
    a = ap.parse_args()
    import uvicorn
    from vllm_decide_engine import VllmDecideEngine
    engine = VllmDecideEngine(a.model, a.mem_fraction, a.context_length, max_concurrency=a.max_concurrency,
                                decision_max_state=a.decision_max_state, decision_max_row=a.decision_max_row,
                                reasoning_token_limit=a.reasoning_token_limit)
    server = uvicorn.Server(uvicorn.Config(create_app(engine, cancel_disconnected_decisions=True),
                                         host=a.host, port=a.port, loop="none"))
    try:
        engine.loop.run_until_complete(server.serve())
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
