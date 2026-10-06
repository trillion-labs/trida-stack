"""OpenAI-compatible server for on-device Trida (stdlib only: http.server + SSE).

    python -m trida_mlx.server --model ./Trida2.0-4B-mlx-q8 --port 8080
    curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'

Endpoints: GET /v1/models, GET /health, GET /stats, POST /v1/chat/completions,
POST /v1/completions. Chat supports ``tools`` (rendered by the model's chat template,
parsed from ``<tool_call>`` blocks into OpenAI ``tool_calls``), ``stream``, reasoning split
(``message.reasoning_content``), ``stop``, ``seed`` and per-request
``chat_template_kwargs`` (e.g. ``{"enable_thinking": false}``). Extra body field
``trida_mode`` = ``self-spec`` | ``causal`` overrides the server's decode mode.
One device, one stream: requests are serialized.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from typing import Optional
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .decode import DecodeStats, SamplingParams
from .engine import Engine, IncrementalDetokenizer
from .parsing import StreamSplitter, parse_tool_calls, split_reasoning, to_openai_tool_calls

ENGINE: Engine = None  # set in main()
DEFAULTS: dict = {}


def _sampling(body: dict) -> SamplingParams:
    t = body.get("temperature", DEFAULTS["temperature"])
    return SamplingParams(
        temperature=float(t if t is not None else DEFAULTS["temperature"]),
        top_k=int(body.get("top_k") or DEFAULTS["top_k"]),
        top_p=float(body.get("top_p") or DEFAULTS["top_p"]),
        seed=body.get("seed"),
    )


def _image_source(part: dict):
    """The image reference of an OpenAI / Responses / plain content part, or None."""
    if not isinstance(part, dict):
        return None
    t = part.get("type")
    if t in ("image_url", "input_image") or "image_url" in part:
        u = part.get("image_url")
        return u.get("url") if isinstance(u, dict) else u
    if t == "image" or "image" in part:
        u = part.get("image")
        return u.get("url") if isinstance(u, dict) else u
    return None


def _content(content, images: Optional[list]):
    """Template content for one message. With ``images`` (a list to append sources to) image
    parts stay as ``{"type": "image"}`` items so the chat template emits its vision
    placeholder; without it they become a short text marker (no vision encoder)."""
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)
    items, has_image = [], False
    for p in content:
        src = _image_source(p)
        if src is not None:
            if images is not None:
                images.append(src)
                items.append({"type": "image"})
                has_image = True
            else:
                items.append({"type": "text", "text": "[image omitted: this model has no vision encoder]"})
        elif isinstance(p, dict) and "text" in p:
            items.append({"type": "text", "text": p.get("text") or ""})
    if not has_image:
        return "".join(i["text"] for i in items)
    return items


def _normalize_messages(messages: list, vision: bool = False):
    """OpenAI -> the Trida (Qwen3.5) chat template's contract. Returns (messages, image_sources).
    * text parts joined; image parts kept as template image items (in order) when ``vision``
    * ``developer`` -> ``system``; all leading system messages merged into one; a system message
      later in the conversation (harness nudges, compaction notes) becomes a user turn, because
      the template only accepts a system message first (images in system messages are dropped)
    * assistant ``tool_calls[].function.arguments`` as dicts (the template iterates them)
    """
    out, sys_parts, seen_non_system, images = [], [], False, []
    for m in messages:
        m = dict(m)
        role = m.get("role")
        if role == "developer":
            role = m["role"] = "system"
        if role in ("system", "assistant"):
            c = _content(m.get("content"), None)
            m["content"] = c if isinstance(c, str) else ""
        else:
            m["content"] = _content(m.get("content"), images if vision else None)
        if role == "system":
            if not seen_non_system:
                sys_parts.append(m["content"])
                continue
            m = {"role": "user", "content": f"[System note]\n{m['content']}"}
        seen_non_system = True
        if role == "assistant":
            if isinstance(m.get("reasoning"), str) and not isinstance(m.get("reasoning_content"), str):
                m["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    fn = dict(tc.get("function", tc))
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except Exception:
                            args = {"_raw": args}
                    fn["arguments"] = args if isinstance(args, dict) else {"value": args}
                    tcs.append({"type": "function", "id": tc.get("id"), "function": fn})
                m["tool_calls"] = tcs
        out.append(m)
    if sys_parts:
        out.insert(0, {"role": "system", "content": "\n\n".join(p for p in sys_parts if p)})
    return out, images


def _thinking(body: dict) -> bool:
    kw = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in kw:
        return bool(kw["enable_thinking"])
    if "enable_thinking" in body:
        return bool(body["enable_thinking"])
    eff = body.get("reasoning_effort")
    if eff is None and isinstance(body.get("reasoning"), dict):
        eff = body["reasoning"].get("effort")
    if isinstance(eff, str) and eff.lower() in ("none", "off", "disabled", "minimal"):
        return False
    return DEFAULTS["enable_thinking"]


class ContextOverflow(Exception):
    pass


class BadRequest(Exception):
    pass


class _StopMatcher:
    def __init__(self, stops):
        self.stops = [s for s in ([stops] if isinstance(stops, str) else (stops or [])) if s]
        self.text = ""

    def check(self, full_text: str):
        """Returns index to cut the full text at, or None."""
        for s in self.stops:
            i = full_text.find(s)
            if i >= 0:
                return i
        return None


class _Job:
    def __init__(self, prompt, body, max_tokens, want_progress):
        self.prompt, self.body, self.max_tokens, self.want_progress = prompt, body, max_tokens, want_progress
        self.out: queue.Queue = queue.Queue()
        self.cancelled = False

    def run(self):
        """Runs on the worker thread (the only thread that touches MLX)."""
        body = self.body
        stop = _StopMatcher(body.get("stop"))
        stats = DecodeStats()
        text = ""
        detok = IncrementalDetokenizer(ENGINE.tokenizer)
        progress = (lambda done, total: self.out.put(("progress", done, total))) if self.want_progress else None
        gen = ENGINE.generate_ids(self.prompt, max_tokens=self.max_tokens, sampling=_sampling(body),
                                  mode=body.get("trida_mode"), stats=stats, progress=progress)
        try:
            for ids in gen:
                if self.cancelled:
                    stats.finish_reason = "cancelled"
                    break
                d = detok.add(ids)
                if not d:
                    continue
                cut = stop.check(text + d) if stop.stops else None
                if cut is not None:
                    d = (text + d)[len(text):cut] if cut >= len(text) else ""
                    text += d
                    if d:
                        self.out.put(("delta", d))
                    stats.finish_reason = "stop"
                    break
                text += d
                self.out.put(("delta", d))
        finally:
            gen.close()
        d = stats.as_dict()
        sys.stderr.write(f"[trida-mlx] prompt {d['prompt_tokens']} (cached {d['reused_tokens']}, prefill "
                         f"{d['prefill_s']:.1f}s) -> {d['new_tokens']} tok @ {d['decode_tok_s']} tok/s, "
                         f"{d['tokens_per_forward']} tok/fwd [{d['finish_reason']}]\n")
        self.out.put(("done", stats))


class _Worker(threading.Thread):
    """MLX streams are per-thread, so the model is loaded and every forward runs on this one
    thread; HTTP handler threads only render/tokenize and relay deltas."""

    def __init__(self, engine_kwargs: dict):
        super().__init__(daemon=True, name="trida-mlx-worker")
        self.engine_kwargs = engine_kwargs
        self.jobs: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.error = None

    def submit(self, job: "_Job"):
        self.jobs.put(job)

    def run(self):
        global ENGINE
        try:
            ENGINE = Engine(**self.engine_kwargs)
            ENGINE.warmup()
        except BaseException as e:  # noqa: BLE001
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.jobs.get()
            if job.cancelled:
                job.out.put(("done", DecodeStats(finish_reason="cancelled")))
                continue
            try:
                job.run()
            except BaseException as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                job.out.put(("error", e))


WORKER: "_Worker" = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("[trida-mlx] " + (fmt % args) + "\n")

    # -- helpers ----------------------------------------------------------------
    def _json(self, code: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj):
        payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _model_card(self):
        ctx = DEFAULTS["context_length"]
        return {"id": DEFAULTS["served_name"], "object": "model", "owned_by": "trillionlabs", "created": 0,
                "context_length": ctx, "max_model_len": ctx, "root": str(ENGINE.path),
                "supports_vision": ENGINE.supports_vision,
                "input_modalities": ["text", "image"] if ENGINE.supports_vision else ["text"]}

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [self._model_card()]})
        if path.startswith(("/v1/models/", "/models/")):
            return self._json(200, self._model_card())
        if path in ("/health", "/v1/health", ""):
            return self._json(200, {"status": "ok", "mode": ENGINE.mode})
        if path == "/stats":
            return self._json(200, {"last": ENGINE.last_stats, "cache_slots": ENGINE.cached_tokens()})
        self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})
        route = self.path.split("?")[0].rstrip("/")
        if DEFAULTS.get("log_requests"):
            import pathlib
            d = pathlib.Path(DEFAULTS["log_requests"]); d.mkdir(parents=True, exist_ok=True)
            (d / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.json").write_text(
                json.dumps({"path": route, "body": body}, ensure_ascii=False, indent=1))
        try:
            if route in ("/v1/chat/completions", "/chat/completions"):
                return self._chat(body)
            if route in ("/v1/completions", "/completions"):
                return self._completion(body)
            return self._json(404, {"error": {"message": f"no route {self.path}"}})
        except BadRequest as e:
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
        except ContextOverflow as e:
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error",
                                              "param": "messages", "code": "context_length_exceeded"}})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            try:
                self._json(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            except Exception:
                pass

    # -- generation core --------------------------------------------------------
    @staticmethod
    def _budget(prompt: list, body: dict) -> int:
        ctx = DEFAULTS["context_length"]
        if len(prompt) >= ctx:
            raise ContextOverflow(
                f"This model's maximum context length is {ctx} tokens. However, your messages resulted in "
                f"{len(prompt)} tokens. Please reduce the length of the messages.")
        want = int(body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULTS["max_tokens"])
        return max(1, min(want, ctx - len(prompt)))

    def _run(self, prompt: list, body: dict, on_delta, keepalive=None):
        """Generate on the MLX worker thread; call on_delta(text) here per decoded delta.
        Returns (full_text, stats). A client disconnect cancels the generation."""
        max_tokens = self._budget(prompt, body)
        job = _Job(prompt, body, max_tokens, want_progress=keepalive is not None)
        WORKER.submit(job)
        text = ""
        try:
            while True:
                kind, *val = job.out.get()
                if kind == "delta":
                    text += val[0]
                    on_delta(val[0])
                elif kind == "progress":
                    keepalive(*val)
                elif kind == "error":
                    raise val[0]
                elif kind == "done":
                    return text, val[0]
        except BaseException:
            job.cancelled = True
            raise

    def _usage(self, stats: DecodeStats):
        return {"prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.new_tokens,
                "total_tokens": stats.prompt_tokens + stats.new_tokens,
                "prompt_tokens_details": {"cached_tokens": stats.reused_tokens},
                "trida": stats.as_dict()}

    def _chat(self, body: dict):
        messages, image_srcs = _normalize_messages(body.get("messages") or [], vision=ENGINE.supports_vision)
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        kw = {k: v for k, v in (body.get("chat_template_kwargs") or {}).items() if k != "enable_thinking"}
        prompt_text = ENGINE.render(messages, tools=tools, enable_thinking=_thinking(body), **kw)
        try:
            images = [ENGINE.add_image(src) for src in image_srcs]
        except Exception as e:  # noqa: BLE001
            raise BadRequest(f"could not load image: {type(e).__name__}: {e}")
        prompt = ENGINE.encode(prompt_text, images)
        self._budget(prompt, body)  # 400 before any streaming starts
        started_in_think = prompt_text.rstrip().endswith("<think>")
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        model = body.get("model") or DEFAULTS["served_name"]

        if not body.get("stream"):
            text, stats = self._run(prompt, body, lambda d: None)
            reasoning, content = split_reasoning(text, started_in_think)
            calls = []
            if tools:
                content, calls = parse_tool_calls(content, tools)
            msg = {"role": "assistant", "content": content or (None if calls else "")}
            if reasoning:
                msg["reasoning_content"] = reasoning
            finish = stats.finish_reason
            if calls:
                msg["tool_calls"] = to_openai_tool_calls(calls)
                finish = "tool_calls"
            return self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": model,
                                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                    "usage": self._usage(stats)})

        self._sse_start()
        splitter = StreamSplitter(started_in_think, parse_tools=bool(tools))

        def chunk(delta: dict, finish=None, **extra):
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            obj.update(extra)
            self._sse(obj)

        def emit(parts):
            for kind, t in parts:
                chunk({"reasoning_content": t} if kind == "reasoning" else {"content": t})

        def keepalive(done, total):
            self.wfile.write(f": prefill {done}/{total}\n\n".encode())
            self.wfile.flush()

        chunk({"role": "assistant", "content": ""})
        _, stats = self._run(prompt, body, lambda d: emit(splitter.feed(d)), keepalive=keepalive)
        rest, calls = splitter.finish(tools)
        emit(rest)
        finish = stats.finish_reason
        if calls:
            tcs = to_openai_tool_calls(calls)
            chunk({"tool_calls": [dict(tc, index=i) for i, tc in enumerate(tcs)]})
            finish = "tool_calls"
        chunk({}, finish)
        if (body.get("stream_options") or {}).get("include_usage", True):
            self._sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [], "usage": self._usage(stats)})
        self._sse("[DONE]")

    def _completion(self, body: dict):
        prompt_text = body.get("prompt") or ""
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0]
        rid, created = f"cmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        model = body.get("model") or DEFAULTS["served_name"]
        prompt = ENGINE.encode(prompt_text)
        self._budget(prompt, body)
        if not body.get("stream"):
            text, stats = self._run(prompt, body, lambda d: None)
            return self._json(200, {"id": rid, "object": "text_completion", "created": created, "model": model,
                                    "choices": [{"index": 0, "text": text, "finish_reason": stats.finish_reason}],
                                    "usage": self._usage(stats)})
        self._sse_start()
        _, stats = self._run(prompt, body, lambda d: self._sse(
            {"id": rid, "object": "text_completion", "created": created, "model": model,
             "choices": [{"index": 0, "text": d, "finish_reason": None}]}))
        self._sse({"id": rid, "object": "text_completion", "created": created, "model": model,
                   "choices": [{"index": 0, "text": "", "finish_reason": stats.finish_reason}],
                   "usage": self._usage(stats)})
        self._sse("[DONE]")


def main(argv=None):
    global WORKER
    ap = argparse.ArgumentParser(description="OpenAI-compatible on-device Trida server (MLX)")
    ap.add_argument("--model", default="trillionlabs/Trida2.0-4B",
                    help="HF id or local dir (raw checkpoint or convert.py output)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mode", default="self-spec", choices=["self-spec", "causal"])
    ap.add_argument("--gen-block", type=int, default=None, help="N tokens per self-spec step (canvas 2N-1); default 4 (fastest on Apple Silicon)")
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--no-prompt-cache", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192, help="default output budget (capped by context)")
    ap.add_argument("--context-length", type=int, default=65536,
                    help="advertised + enforced context window (Hermes needs >= 64000)")
    ap.add_argument("--served-model-name", default="trida2.0-4b")
    ap.add_argument("--log-requests", default=None, metavar="DIR", help="dump every POST body as JSON (debugging)")
    ap.add_argument("--max-image-pixels", type=int, default=1024 * 1024,
                    help="resize budget per image (every 32x32 px = 1 LM token; 1024*1024 -> <=1024 tokens)")
    ap.add_argument("--cache-slots", type=int, default=3,
                    help="conversations whose prompt cache is kept (main loop + side requests)")
    ap.add_argument("--no-think", action="store_true", help="default enable_thinking=false")
    ap.add_argument("--no-fused-gdn", action="store_true")
    a = ap.parse_args(argv)
    DEFAULTS.update(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, max_tokens=a.max_tokens,
                    enable_thinking=not a.no_think, context_length=a.context_length,
                    served_name=a.served_model_name, log_requests=a.log_requests)
    print(f"[trida-mlx] loading {a.model} ...", flush=True)
    WORKER = _Worker(dict(model=a.model, mode=a.mode, gen_block=a.gen_block, prefill_chunk=a.prefill_chunk,
                          prompt_cache=not a.no_prompt_cache, fused_gdn=not a.no_fused_gdn,
                          cache_slots=a.cache_slots, max_image_pixels=a.max_image_pixels))
    WORKER.start()
    WORKER.ready.wait()
    if WORKER.error is not None:
        raise WORKER.error
    print(f"[trida-mlx] ready in {ENGINE.load_s:.1f}s  mode={a.mode} N={ENGINE.n} mask_id={ENGINE.mask_id} "
          f"eos={sorted(ENGINE.eos_ids)} ctx={a.context_length} vision={ENGINE.supports_vision} "
          f"model={a.served_model_name!r}  ->  "
          f"http://{a.host}:{a.port}/v1", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
