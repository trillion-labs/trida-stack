#!/usr/bin/env python3
# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""Send one image crop to a running GravityOCR server (serve.sh) and print the recognised text.

GravityOCR reads one layout region per request: crop the region first (a paragraph, a table or a
formula) and pick the matching task prompt:

    Text Recognition:      plain text in reading order
    Table Recognition:     the table as HTML
    Formula Recognition:   the formula as LaTeX

The image travels inside the request as a base64 data URL; the client never asks the server to
fetch a URL or read a file. Standard library only (Pillow is needed only for --synthetic).

    python ocr_request.py crop.png
    python ocr_request.py table.png --prompt "Table Recognition:"
    python ocr_request.py --synthetic "Hello GravityOCR"         # renders the text into a PNG first
    API_KEY=... python ocr_request.py crop.png --url http://gpu-box:30000
"""

from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}
DEFAULT_PROMPT = TASK_PROMPTS["text"]
DEFAULT_URL = "http://127.0.0.1:30000"
DEFAULT_MODEL = "glm-ocr"  # serve.sh SERVED_NAME default
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 300.0
MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def guess_mime(path: str | os.PathLike) -> str:
    """MIME type for an image path, from its suffix (PNG when unknown)."""
    return MIME_BY_SUFFIX.get(Path(path).suffix.lower(), "image/png")


def build_payload(image_bytes: bytes, mime: str, prompt: str, model: str, max_tokens: int) -> dict:
    """OpenAI chat-completions body for one crop: [image_url (data URL), text], greedy decoding."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def render_text_png(text: str, font_size: int = 32, margin: int = 24) -> bytes:
    """Render text (black on white) into PNG bytes, for a smoke test without a sample image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - exercised only without Pillow
        raise RuntimeError("--synthetic needs Pillow (pip install pillow)") from exc
    try:
        font = ImageFont.load_default(size=font_size)  # scalable default font, Pillow >= 10.1
    except TypeError:  # older Pillow: fixed-size bitmap font
        font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=font)
    width = max(right - left, 1) + 2 * margin
    height = max(bottom - top, 1) + 2 * margin
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).multiline_text(
        (margin - left, margin - top), text, fill="black", font=font
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Send one image crop to a GravityOCR server and print the recognised text.",
        epilog="Task prompts: " + ", ".join(f"'{p}'" for p in TASK_PROMPTS.values()),
    )
    ap.add_argument("image", nargs="?", help="image file of one layout region (crop)")
    ap.add_argument(
        "--synthetic", metavar="TEXT", help="render TEXT into a PNG and send that instead of a file"
    )
    ap.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help=f"task prompt (default: '{DEFAULT_PROMPT}')"
    )
    ap.add_argument("--url", default=DEFAULT_URL, help=f"server root URL (default: {DEFAULT_URL})")
    ap.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"served model name (default: {DEFAULT_MODEL})"
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="bearer key for a server started with API_KEY (default: $API_KEY)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    args = ap.parse_args(argv)
    if (args.image is None) == (args.synthetic is None):
        ap.error("give exactly one of IMAGE or --synthetic TEXT")
    if not args.url.startswith(("http://", "https://")):
        ap.error("--url must start with http:// or https://")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.synthetic is not None:
        image_bytes, mime = render_text_png(args.synthetic), "image/png"
    else:
        image_bytes, mime = Path(args.image).read_bytes(), guess_mime(args.image)
    payload = build_payload(image_bytes, mime, args.prompt, args.model, args.max_tokens)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    url = request.full_url
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        print(f"[ocr_request] HTTP {exc.code} from {url}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"[ocr_request] cannot reach {url}: {exc.reason}", file=sys.stderr)
        return 1
    except TimeoutError:  # the server accepted the connection but did not answer in time
        print(f"[ocr_request] no reply from {url} within {args.timeout:g} s (--timeout)", file=sys.stderr)
        return 1
    except (ConnectionError, http.client.HTTPException) as exc:  # e.g. the server dropped the connection
        print(f"[ocr_request] connection to {url} failed: {exc!r}", file=sys.stderr)
        return 1
    try:
        out = json.loads(raw)
        content = out["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError(f"content is {type(content).__name__}")
    except (ValueError, LookupError, TypeError) as exc:
        snippet = raw[:300].decode("utf-8", "replace")
        print(f"[ocr_request] unexpected reply from {url} ({exc!r}): {snippet}", file=sys.stderr)
        return 1
    print(content)
    print(f"[ocr_request] usage: {json.dumps(out.get('usage'))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
