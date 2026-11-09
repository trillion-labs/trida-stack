#!/usr/bin/env python3
# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""Serve GravityOCR on an Apple Silicon Mac, with the MLX engine of the installed Gravity app.

serve.sh is the real server: upstream SGLang plus the GravityOCR patch, on an NVIDIA L40 or H100.
This script is the Mac option for trying the model locally and for running getting_started.ipynb
without an NVIDIA GPU. It answers the same OpenAI chat-completions requests as serve.sh (one crop, one task
prompt, a data: image URL), so ocr_request.py and the notebook work against either.

It does not ship an engine. It imports the MLX engine inside Gravity.app
(https://github.com/trillion-labs/GravityOCR-macOS, a separate download from Trillion Labs) and runs
it with Gravity's bundled Python, so run it with that interpreter:

    /Applications/Gravity.app/Contents/Resources/runtime/bin/python -I -B serve_mac.py
    .../runtime/bin/python -I -B serve_mac.py --mode causal --port 30001
    .../runtime/bin/python -I -B serve_mac.py --model /path/to/checkpoint

How it differs from serve.sh, so its outputs and timings are not the L40 / H100 numbers:
  * MLX on the Mac GPU instead of SGLang on CUDA; text-decoder layers quantized to 6 bits (Gravity's
    setting), the vision encoder in bf16;
  * self-spec drafts blocks of 16 tokens (Gravity's setting) instead of 32;
  * crops are downscaled to a 1344 px long side; generation stops at Gravity's per-task limits
    (text 2048, table 8192, formula 1024 tokens) and max_tokens is ignored;
  * --mode causal swaps Gravity's speculative decoder for the autoregressive one bundled with it,
    over the same prefilled cache.

It reaches into Gravity's internal Python package (ocr.engine), which is not a public API. It was
written against Gravity 0.3.4 and warns on any other version; an update may break it.
"""

from __future__ import annotations

import sys

# Never write __pycache__ into Gravity.app: the app is signed, and new files inside the bundle break its
# seal (codesign --verify fails, and macOS may refuse to open it). Set before any import from Gravity's
# runtime; -B on the command line does the same.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import base64
import hashlib
import io
import ipaddress
import json
import os
import plistlib
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DEFAULT_APP = "/Applications/Gravity.app"
TESTED_GRAVITY = "0.3.4"
DEFAULT_MODEL = "trillionlabs/GravityOCR"
CACHE = Path.home() / ".cache" / "gravityocr-mac"
CKPT_PATTERNS = (".json", ".jinja", ".safetensors")  # what the engine reads; skips README, assets
MAX_BODY = 64 * 1024 * 1024
TASK_BY_PROMPT = {
    "Text Recognition:": "text",
    "Table Recognition:": "table",
    "Formula Recognition:": "formula",
}


def log(msg: str) -> None:
    print(f"[serve_mac] {msg}", file=sys.stderr, flush=True)


# ---- checkpoint ---------------------------------------------------------------------------------
def fetch_checkpoint(repo: str, cache: Path = CACHE, revision: str = "main") -> Path:
    """Download a Hub checkpoint once, pinned to a commit, checking every size and LFS SHA-256."""
    api = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(api, timeout=60) as r:
        info = json.loads(r.read())
    sha = info["sha"]
    dest = cache / repo.replace("/", "--") / sha
    files = [s for s in info["siblings"] if s["rfilename"].endswith(CKPT_PATTERNS)
             and "/" not in s["rfilename"]]
    for s in files:
        path = dest / s["rfilename"]
        if path.is_file() and path.stat().st_size == s["size"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.{os.getpid()}.part")  # two servers may fetch at once
        url = f"https://huggingface.co/{repo}/resolve/{sha}/{s['rfilename']}"
        log(f"downloading {s['rfilename']} ({s['size'] / 1e9:.2f} GB) from {repo}@{sha[:7]}")
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
        want = (s.get("lfs") or {}).get("sha256")
        if part.stat().st_size != s["size"] or (want and digest.hexdigest() != want):
            part.unlink()
            raise RuntimeError(f"{s['rfilename']}: size or SHA-256 does not match the Hub")
        part.replace(path)
    return dest


# ---- Gravity's engine ---------------------------------------------------------------------------
def gravity_version(app: Path) -> str:
    with open(app / "Contents" / "Info.plist", "rb") as f:
        return plistlib.load(f).get("CFBundleShortVersionString", "unknown")


def load_engine(app: Path, checkpoint: Path, mode: str):
    """Gravity's LocalOcrEngine on `checkpoint`; in causal mode with its AR decoder swapped in."""
    sys.path.insert(0, str(app / "Contents" / "Resources" / "project"))
    from ocr.engine import pipeline
    from ocr.recognition.model import decode_ar_mlx_from_prefix

    if mode == "causal":
        def decode_ar_many(model, prefixes, *, eos_token_ids, **_):
            return [decode_ar_mlx_from_prefix(model, p.input_ids, p.cache, p.seed_logits,
                                              p.rope_delta, max_new_tokens=p.max_new_tokens,
                                              eos_token_ids=eos_token_ids)
                    for p in prefixes]
        pipeline.decode_speculative_mlx_many_from_prefix = decode_ar_many
    return pipeline.LocalOcrEngine(checkpoint, structured=False, max_long_side=1344)


# ---- HTTP ---------------------------------------------------------------------------------------
class BadRequest(ValueError):
    pass


def parse_chat_request(body: bytes) -> tuple[bytes, str]:
    """An OpenAI chat request for one crop -> (image bytes, task). Accepts data: URLs only."""
    try:
        req = json.loads(body)
        content = req["messages"][-1]["content"]
        url = next(c["image_url"]["url"] for c in content if c.get("type") == "image_url")
        prompt = next(c["text"] for c in content if c.get("type") == "text").strip()
    except (ValueError, KeyError, TypeError, IndexError, StopIteration):
        raise BadRequest("expected a user message with an image_url part and a text part") from None
    if not url.startswith("data:") or "," not in url:
        raise BadRequest("send the image as a data: URL")
    if prompt not in TASK_BY_PROMPT:
        raise BadRequest(f"the text part must be one of {list(TASK_BY_PROMPT)}")
    try:
        image = base64.b64decode(url.split(",", 1)[1], validate=True)
    except ValueError:
        raise BadRequest("the data: URL is not valid base64") from None
    return image, TASK_BY_PROMPT[prompt]


def chat_response(model: str, text: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def make_handler(recognize, model_name: str):
    """recognize(image_bytes, task) -> (text, visual_tokens, output_tokens)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                return self._send(200, {"object": "list",
                                        "data": [{"id": model_name, "object": "model"}]})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found"}})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                return self._send(413, {"error": {"message": f"body must be 1..{MAX_BODY} bytes"}})
            try:
                image, task = parse_chat_request(self.rfile.read(length))
                text, prompt_tokens, completion_tokens = recognize(image, task)
            except BadRequest as e:
                return self._send(400, {"error": {"message": str(e)}})
            except Exception as e:  # keep serving after a bad image or an engine error
                return self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            self._send(200, chat_response(model_name, text, prompt_tokens, completion_tokens))

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hub id or local checkpoint directory")
    ap.add_argument("--mode", choices=("self-spec", "causal"), default="self-spec")
    ap.add_argument("--host", default="127.0.0.1", help="loopback addresses only")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--served-name", default="glm-ocr")
    ap.add_argument("--gravity-app", type=Path, default=Path(DEFAULT_APP))
    args = ap.parse_args(argv)

    host = "127.0.0.1" if args.host == "localhost" else args.host
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        ap.error("--host must be a loopback address: this server has no authentication")
    if not (args.gravity_app / "Contents" / "Resources" / "project" / "ocr").is_dir():
        ap.error(f"Gravity's engine not found in {args.gravity_app}: install Gravity "
                 "(https://gravityocr.trillionlabs.co) or pass --gravity-app")
    version = gravity_version(args.gravity_app)
    if version != TESTED_GRAVITY:
        log(f"warning: Gravity {version}; this script was tested with {TESTED_GRAVITY}")

    ckpt = Path(args.model).expanduser()
    if not ckpt.is_dir():
        ckpt = fetch_checkpoint(args.model)
    if not (ckpt / "block_diffusion.json").is_file():
        ap.error(f"{ckpt} has no block_diffusion.json (see MODEL_CONTRACT.md)")

    t0 = time.time()
    engine = load_engine(args.gravity_app, ckpt, args.mode)
    gen = engine.generation
    log(f"mode={args.mode} block_size={gen.block_size} mask_id={gen.mask_token_id} "
        f"(from {ckpt}/block_diffusion.json); Gravity {version} MLX engine, text layers 6-bit; "
        f"loaded in {time.time() - t0:.1f} s")

    from PIL import Image

    def recognize(image_bytes: bytes, task: str):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text, profile = engine.recognize_profiled(image, task)
        return text, profile["visual_tokens"], profile["output_tokens"]

    server = HTTPServer((host, args.port), make_handler(recognize, args.served_name))
    log(f"serving {args.served_name} on http://{host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
