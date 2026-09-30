"""End-to-end (offline) checks on a tiny raw-HF checkpoint: loader, convert (q8), engine
prompt cache, OpenAI server (non-stream + SSE + tools), and output parsing."""
import json
import os
import socket
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx

import tiny_ckpt
from test_tiny import tiny_runtime
from trida_mlx import convert
from trida_mlx.decode import DecodeStats, SamplingParams
from trida_mlx.engine import Engine
from trida_mlx.model import TridaRuntime, load_model
from trida_mlx.parsing import StreamSplitter, parse_tool_calls, split_reasoning

TMP = Path(tempfile.mkdtemp(prefix="trida_e2e_"))
CKPT, MASK_ID = tiny_ckpt.build(TMP / "raw")


def test_raw_hf_loader_matches_reference():
    model, cfg, _ = load_model(str(CKPT), dtype=mx.float32)
    rt = TridaRuntime(model)
    ref = tiny_runtime(seed=0)
    p = [5, 17, 99, 3, 250, 8, 41]
    a = rt.prefill(rt.make_cache(), p)
    b = ref.prefill(ref.make_cache(), p)
    # bf16 round trip of the weights -> small differences only
    assert float(mx.max(mx.abs(a - b))) < 0.25 * float(mx.max(mx.abs(b))), (a[:5], b[:5])
    assert int(mx.argmax(a)) == int(mx.argmax(b))


def test_convert_q8_and_selfspec_lossless_on_quantized():
    out = TMP / "q8"
    convert.main(["--model", str(CKPT), "--out", str(out), "--bits", "8", "--group-size", "32"])
    eng = Engine(str(out))
    assert eng.mask_id == MASK_ID
    p = eng.encode(eng.render([{"role": "user", "content": "hello world"}]))
    sp = SamplingParams(temperature=0.0)
    a = [t for c in eng.generate_ids(p, max_tokens=80, sampling=sp, mode="causal") for t in c]
    eng.reset_cache()
    b = [t for c in eng.generate_ids(p, max_tokens=80, sampling=sp, mode="self-spec") for t in c]
    assert a == b


def test_prompt_cache_reuse():
    eng = Engine(str(CKPT))
    sp = SamplingParams(temperature=0.0)
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello world"}]
    p1 = eng.encode(eng.render(msgs))
    out1 = [t for c in eng.generate_ids(p1, max_tokens=30, sampling=sp) for t in c]
    msgs2 = msgs + [{"role": "assistant", "content": eng.tokenizer.decode(out1)},
                    {"role": "user", "content": "and the weather?"}]
    p2 = eng.encode(eng.render(msgs2))
    st = DecodeStats()
    out2 = [t for c in eng.generate_ids(p2, max_tokens=30, sampling=sp, stats=st) for t in c]
    assert st.reused_tokens > len(p1) // 2, st.as_dict()
    # same result as a cold engine
    eng2 = Engine(str(CKPT), prompt_cache=False)
    ref2 = [t for c in eng2.generate_ids(p2, max_tokens=30, sampling=sp) for t in c]
    assert out2 == ref2


def test_parsing():
    c, calls = parse_tool_calls('ok <tool_call>{"name": "f", "arguments": {"x": 1}}}</tool_call>')
    assert c == "ok" and calls == [{"name": "f", "arguments": {"x": 1}}]
    _, calls = parse_tool_calls("<tool_call>\n<function=g>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>")
    assert calls == [{"name": "g", "arguments": {"city": "Seoul"}}]
    assert split_reasoning("abc</think>\n\nhi", True) == ("abc", "hi")
    s = StreamSplitter(True, parse_tools=True)
    text = 'let me think</think>\n\nSure.<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    parts = []
    for ch in text:
        parts += s.feed(ch)
    rest, calls = s.finish()
    parts += rest
    reasoning = "".join(t for k, t in parts if k == "reasoning")
    content = "".join(t for k, t in parts if k == "content")
    assert reasoning == "let me think" and content == "Sure." and calls[0]["name"] == "f", (parts, calls)


def test_normalize_messages_against_real_trida_template():
    """Hermes-style histories must render with the real Trida chat template."""
    from transformers import AutoTokenizer
    from trida_mlx.server import _normalize_messages
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    tok.chat_template = (Path(__file__).parent / "fixtures" / "trida_chat_template.jinja").read_text()
    tools = [{"type": "function", "function": {"name": "terminal", "description": "run",
              "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
    msgs = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "developer", "content": "Be terse."},
        {"role": "user", "content": [{"type": "text", "text": "list files"}]},
        {"role": "assistant", "content": None, "reasoning": "use terminal",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.py\nb.py"},
        {"role": "system", "content": "Context was compacted."},
        {"role": "user", "content": "thanks, now count them"},
    ]
    norm = _normalize_messages(msgs)
    assert [m["role"] for m in norm] == ["system", "user", "assistant", "tool", "user", "user"]
    text = tok.apply_chat_template(norm, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    assert "You are Hermes.\n\nBe terse." in text
    assert "<function=terminal>\n<parameter=command>\nls\n</parameter>" in text
    assert "[System note]\nContext was compacted." in text and text.endswith("<think>\n")


def test_tool_param_coercion():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "n": {"type": "integer"}, "flags": {"type": "array"}}}}}]
    _, calls = parse_tool_calls("<tool_call>\n<function=f>\n<parameter=path>\n007\n</parameter>\n"
                                "<parameter=n>\n5\n</parameter>\n<parameter=flags>\n[\"-a\"]\n</parameter>\n"
                                "</function>\n</tool_call>", tools)
    assert calls == [{"name": "f", "arguments": {"path": "007", "n": 5, "flags": ["-a"]}}], calls


def test_multi_slot_cache_survives_side_requests():
    eng = Engine(str(CKPT), cache_slots=3)
    sp = SamplingParams(temperature=0.0)
    run = lambda msgs, st=None: [t for c in eng.generate_ids(eng.encode(eng.render(msgs)), max_tokens=12,
                                                              sampling=sp, stats=st) for t in c]
    main = [{"role": "system", "content": "main agent " * 30}, {"role": "user", "content": "hello world"}]
    out = run(main)
    run([{"role": "user", "content": "write a title for: hello"}])        # side request (title)
    run([{"role": "user", "content": "summarize: the weather in seoul"}])  # another side request
    st = DecodeStats()
    run(main + [{"role": "assistant", "content": eng.tokenizer.decode(out)},
                {"role": "user", "content": "and then?"}], st)
    assert st.reused_tokens > 100, st.as_dict()
    assert len(eng.slots) == 3


def test_system_block_snapshot_reused_when_user_turn_changes():
    """Hermes rewrites the last user message on a truncation retry; the (long) system block
    must still be reused."""
    eng = Engine(str(CKPT), cache_slots=1)
    sp = SamplingParams(temperature=0.0)
    sys_msg = {"role": "system", "content": "tool schemas " * 60}
    p1 = eng.encode(eng.render([sys_msg, {"role": "user", "content": "list the files"}]))
    for _ in eng.generate_ids(p1, max_tokens=5, sampling=sp):
        pass
    st = DecodeStats()
    p2 = eng.encode(eng.render([sys_msg, {"role": "user", "content": "list the files [System: continue]"}]))
    out = [t for c in eng.generate_ids(p2, max_tokens=5, sampling=sp, stats=st) for t in c]
    n_sys = p1.index(eng.im_start, 1)  # tokens up to the end of the system block
    assert st.reused_tokens == n_sys, (st.reused_tokens, n_sys)
    ref = Engine(str(CKPT), prompt_cache=False)
    assert out == [t for c in ref.generate_ids(p2, max_tokens=5, sampling=sp) for t in c]


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_server_roundtrip():
    from trida_mlx import server
    port = _free_port()
    t = threading.Thread(target=server.main, args=(["--model", str(CKPT), "--port", str(port),
                                                    "--max-tokens", "40", "--prefill-chunk", "16",
                                                    "--context-length", "2048"],), daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}/v1"
    import time
    for _ in range(200):
        try:
            urllib.request.urlopen(f"{base}/models", timeout=1); break
        except Exception:
            time.sleep(0.1)
    tools = [{"type": "function", "function": {"name": "now", "description": "time",
                                                "parameters": {"type": "object", "properties": {}}}}]
    body = {"messages": [{"role": "user", "content": "hello"}], "tools": tools, "temperature": 0}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})).read())
    assert r["choices"][0]["message"]["role"] == "assistant"
    assert r["usage"]["trida"]["new_tokens"] > 0
    body["stream"] = True
    lines = urllib.request.urlopen(urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})).read().decode().splitlines()
    datas = [l[6:] for l in lines if l.startswith("data: ")]
    assert datas[-1] == "[DONE]"
    usage = json.loads(datas[-2])["usage"]
    assert usage["prompt_tokens_details"]["cached_tokens"] > 0  # second identical request hits the cache
    assert any(l.startswith(": prefill") for l in lines) or usage["prompt_tokens_details"]["cached_tokens"] > 0
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{base}/completions", data=json.dumps({"prompt": "hello", "max_tokens": 10}).encode(),
        headers={"Content-Type": "application/json"})).read())
    assert r["usage"]["completion_tokens"] <= 10
    models = json.loads(urllib.request.urlopen(f"{base}/models").read())
    assert models["data"][0]["context_length"] == 2048 and models["data"][0]["id"] == "trida2.0-4b"
    # context overflow -> OpenAI-style 400 that harnesses recognise (and compress)
    big = {"messages": [{"role": "user", "content": "hello world " * 3000}], "stream": True}
    try:
        urllib.request.urlopen(urllib.request.Request(f"{base}/chat/completions", data=json.dumps(big).encode(),
                                                      headers={"Content-Type": "application/json"}))
        assert False, "expected 400"
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())["error"]
        assert e.code == 400 and err["code"] == "context_length_exceeded" and "maximum context length" in err["message"]
    # keep-alive comments during a long (uncached) prefill
    body2 = {"messages": [{"role": "user", "content": "the weather in seoul is sunny " * 40}], "stream": True,
             "max_tokens": 4}
    lines = urllib.request.urlopen(urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body2).encode(),
        headers={"Content-Type": "application/json"})).read().decode().splitlines()
    assert any(l.startswith(": prefill") for l in lines), lines[:5]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"  ok   {name}")
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                fails += 1; print(f"  FAIL {name}: {type(e).__name__}: {e}")
    sys.stdout.flush()
    os._exit(1 if fails else 0)
