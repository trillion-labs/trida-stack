#!/usr/bin/env python3
"""Round-robin router in front of N Trida replica servers (data-parallel throughput).

An OpenAI client points at ONE base_url, so this fans `/v1/completions` and `/v1/chat/completions`
requests across the per-GPU replicas started by `serve_trida_openai.py`. Blocking (GPU-bound)
generate on each replica is serialized there; the router just distributes concurrent requests
round-robin, failing over to a healthy replica on error/non-2xx/non-JSON.

Usage:
    python router.py --port 8000 --upstreams http://localhost:8001,http://localhost:8002,...
"""
import argparse
import itertools
import threading

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--upstreams", required=True, help="comma-separated base URLs of replica servers")
    p.add_argument("--timeout", type=float, default=1200.0)
    return p.parse_args()


def build_app(upstreams, timeout):
    cycle = itertools.cycle(upstreams)
    lock = threading.Lock()
    client = httpx.AsyncClient(timeout=timeout)
    app = FastAPI()

    def next_upstream():
        with lock:
            return next(cycle)

    @app.get("/v1/models")
    async def models():
        r = await client.get(f"{upstreams[0]}/v1/models")
        return JSONResponse(r.json(), status_code=r.status_code)

    @app.get("/health")
    async def health():
        return {"status": "ok", "upstreams": upstreams}

    async def proxy(path: str, body: bytes):
        last_err = None
        # Try replicas round-robin, failing over to a healthy one if a replica errors, returns
        # non-2xx (e.g. OOM), or sends a non-JSON body — so one bad replica doesn't 500 the request.
        for _ in range(len(upstreams)):
            up = next_upstream()
            try:
                r = await client.post(f"{up}{path}", content=body,
                                      headers={"content-type": "application/json"})
            except Exception as e:  # connection refused / timeout / dead replica
                last_err = f"{up}: {e}"
                continue
            if r.status_code // 100 != 2:
                last_err = f"{up}: HTTP {r.status_code}"
                continue
            try:
                return JSONResponse(r.json(), status_code=200)
            except Exception as e:  # 2xx but not JSON
                last_err = f"{up}: bad JSON ({e})"
                continue
        return JSONResponse({"error": {"message": f"all upstreams failed: {last_err}"}},
                            status_code=502)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy("/v1/completions", await request.body())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await proxy("/v1/chat/completions", await request.body())

    return app


if __name__ == "__main__":
    args = parse_args()
    ups = [u.strip().rstrip("/") for u in args.upstreams.split(",") if u.strip()]
    uvicorn.run(build_app(ups, args.timeout), host=args.host, port=args.port, log_level="warning")
