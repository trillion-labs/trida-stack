#!/usr/bin/env python3
# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""Send one image crop to a running GravityOCR server (serve.sh) and print the recognised text.

GravityOCR reads one layout region per request: crop the region first (a paragraph, a table or a
formula) and pick the matching task with --task (or pass a prompt of your own with --prompt):

    --task text       Text Recognition:      plain text in reading order (the default)
    --task table      Table Recognition:     the table as HTML
    --task formula    Formula Recognition:   the formula as LaTeX

The image travels inside the request as a base64 data URL; the client never asks the server to
fetch a URL or read a file. Standard library only (Pillow is needed only for --synthetic).

    python ocr_request.py crop.png
    python ocr_request.py table.png --task table
    python ocr_request.py --synthetic "Hello GravityOCR"         # renders the text into a PNG first
    API_KEY=... python ocr_request.py crop.png --url http://gpu-box:30000

The bearer key is read from $API_KEY only, never from the command line (arguments show up in the
process list). The request goes to --url and nowhere else: redirects are refused, because urllib
would resend the request, Authorization header included, to whatever host the redirect names; and
proxy environment variables (http_proxy, https_proxy, ...) are ignored, because the proxy would
receive the key.
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
DEFAULT_TASK = "text"
DEFAULT_PROMPT = TASK_PROMPTS[DEFAULT_TASK]
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


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Turn every 3xx into an HTTPError instead of following it. urllib would otherwise copy the
    Authorization header to whatever host Location names (and turn the POST into a GET on 301-303)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib then raises HTTPError for the 3xx response


# An empty ProxyHandler: an exported http_proxy / https_proxy would otherwise receive the request,
# key included, even for the loopback default URL.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects)


def send(request: urllib.request.Request, timeout: float):
    """Open `request` with the opener above: no redirects followed, no proxies."""
    return _OPENER.open(request, timeout=timeout)


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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="tasks (--task) and their prompts:\n"
        + "".join(f"  {task:<9} {prompt}\n" for task, prompt in TASK_PROMPTS.items())
        + "\nThe bearer key, if the server needs one, is read from $API_KEY. Redirects are not\n"
        "followed and proxy variables (http_proxy, https_proxy, ...) are ignored.",
    )
    ap.add_argument("image", nargs="?", help="image file of one layout region (crop)")
    ap.add_argument(
        "--synthetic", metavar="TEXT", help="render TEXT into a PNG and send that instead of a file"
    )
    task = ap.add_mutually_exclusive_group()
    task.add_argument(
        "--task",
        choices=sorted(TASK_PROMPTS),
        help=f"region type; selects the task prompt (default: {DEFAULT_TASK})",
    )
    task.add_argument("--prompt", help="a task prompt of your own, instead of --task")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"server root URL (default: {DEFAULT_URL})")
    ap.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"served model name (default: {DEFAULT_MODEL})"
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
    if args.prompt is None:
        args.prompt = TASK_PROMPTS[args.task or DEFAULT_TASK]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.synthetic is not None:
        image_bytes, mime = render_text_png(args.synthetic), "image/png"
    else:
        image_bytes, mime = Path(args.image).read_bytes(), guess_mime(args.image)
    payload = build_payload(image_bytes, mime, args.prompt, args.model, args.max_tokens)
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    url = request.full_url
    try:
        with send(request, timeout=args.timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            where = exc.headers.get("Location") if exc.headers is not None else None
            print(
                f"[ocr_request] HTTP {exc.code} from {url}: redirect to {where or '(no Location)'} not"
                " followed (the request carries the bearer key); pass the final URL with --url",
                file=sys.stderr,
            )
            return 1
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
