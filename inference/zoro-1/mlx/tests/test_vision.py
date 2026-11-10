"""Image-input tests on a tiny random multimodal checkpoint (see tiny_vl.py; CPU is fine).

Pinned: image expansion into per-image keys; greedy self-spec = greedy AR with an image in the
prompt; chunked prefill across an image = one-shot; MRoPE position advance after an image; the
prompt cache never reuses across different images but does reuse past the same one; convert keeps
the vision tower bf16 (bit-identical features); server image_url round trip and 400 on bad data.
"""
import base64, json, os, socket, sys, tempfile, threading, time, urllib.request
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx, numpy as np
import tiny_vl
from trida_mlx.engine import Engine
from trida_mlx.decode import SamplingParams, DecodeStats
from trida_mlx import convert

TMP = Path(tempfile.mkdtemp(prefix="trida_vl_"))
CKPT, IDS = tiny_vl.build(TMP / "raw")
G = SamplingParams(temperature=0.0)

def gen(eng, prompt, mode="self-spec", n=40, st=None):
    return [t for c in eng.generate_ids(prompt, max_tokens=n, sampling=G, mode=mode, stats=st) for t in c]

def vl_prompt(eng, img_bytes, text="describe this image"):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    key_n = eng.add_image(img_bytes)
    return eng.encode(eng.render(msgs, enable_thinking=False), [key_n]), key_n

def test_load_and_expand():
    eng = Engine(str(CKPT))
    assert eng.supports_vision
    p, (key, n) = vl_prompt(eng, tiny_vl.png(1))
    assert n > 0 and p.count(key) == n
    assert IDS["<|image_pad|>"] not in p

def test_greedy_lossless_with_image():
    eng = Engine(str(CKPT), prompt_cache=False)
    p, _ = vl_prompt(eng, tiny_vl.png(2))
    a = gen(eng, p, "causal"); b = gen(eng, p, "self-spec")
    assert a == b, (a[:10], b[:10])

def test_chunked_prefill_across_image():
    e1 = Engine(str(CKPT), prompt_cache=False)
    e2 = Engine(str(CKPT), prompt_cache=False, prefill_chunk=5)
    p, _ = vl_prompt(e1, tiny_vl.png(3)); e2.add_image(tiny_vl.png(3))
    assert gen(e1, p, "causal", 20) == gen(e2, p, "causal", 20)

def test_image_changes_output_and_positions():
    eng = Engine(str(CKPT), prompt_cache=False)
    p1, (k1, n1) = vl_prompt(eng, tiny_vl.png(4))
    p2, _ = vl_prompt(eng, tiny_vl.png(5))
    c = eng.rt.make_cache(); l1 = eng.rt.prefill(c, p1)
    m = 2; t, h, w = eng.rt.images[k1]["grid"]
    assert c.pos == len(p1) - n1 + max(t, h // m, w // m), (c.pos, len(p1), n1)
    l2 = eng.rt.prefill(eng.rt.make_cache(), p2)
    assert float(mx.max(mx.abs(l1 - l2))) > 1e-3  # different image -> different logits

def test_prompt_cache_distinguishes_images():
    eng = Engine(str(CKPT), cache_slots=1)
    p1, _ = vl_prompt(eng, tiny_vl.png(6)); gen(eng, p1, n=8)
    st = DecodeStats()
    p2, _ = vl_prompt(eng, tiny_vl.png(7)); out = gen(eng, p2, n=8, st=st)
    first_img = next(i for i, t in enumerate(p2) if t < 0)
    assert st.reused_tokens <= first_img, (st.reused_tokens, first_img)
    ref = Engine(str(CKPT), prompt_cache=False); ref.add_image(tiny_vl.png(7))
    assert out == gen(ref, p2, n=8)
    # same image again, follow-up turn -> reuses past the image
    st2 = DecodeStats()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe this image"}]},
            {"role": "assistant", "content": eng.tokenizer.decode(out)}, {"role": "user", "content": "more"}]
    k = eng.add_image(tiny_vl.png(7))
    p3 = eng.encode(eng.render(msgs, enable_thinking=False), [k]); gen(eng, p3, n=4, st=st2)
    assert st2.reused_tokens > first_img + k[1], (st2.reused_tokens, first_img, k)

def test_convert_keeps_vision():
    out = TMP / "q8"
    convert.main(["--model", str(CKPT), "--out", str(out), "--bits", "8", "--group-size", "32"])
    eng = Engine(str(out), prompt_cache=False)
    assert eng.supports_vision and (out / "preprocessor_config.json").exists()
    raw = Engine(str(CKPT), prompt_cache=False)
    img = tiny_vl.png(8)
    ka, _ = eng.add_image(img); kb, _ = raw.add_image(img)
    fa = eng.vision(mx.array(eng.rt.images[ka]["pixels"]), [eng.rt.images[ka]["grid"]])
    fb = raw.vision(mx.array(raw.rt.images[kb]["pixels"]), [raw.rt.images[kb]["grid"]])
    assert float(mx.max(mx.abs(fa - fb))) == 0.0  # vision tower kept bf16, bit-identical
    p, _ = vl_prompt(eng, img)
    assert gen(eng, p, "causal", 20) == gen(eng, p, "self-spec", 20)

def test_server_image_request():
    from trida_mlx import server
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=server.main, args=(["--model", str(CKPT), "--port", str(port), "--max-tokens", "12"],), daemon=True).start()
    base = f"http://127.0.0.1:{port}/v1"
    for _ in range(300):
        try: urllib.request.urlopen(f"{base}/models", timeout=1); break
        except Exception: time.sleep(0.1)
    card = json.loads(urllib.request.urlopen(f"{base}/models").read())["data"][0]
    assert card["supports_vision"] is True
    url = "data:image/png;base64," + base64.b64encode(tiny_vl.png(9)).decode()
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "what is in the picture"},
                                                      {"type": "image_url", "image_url": {"url": url}}]}],
            "temperature": 0}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                                                 headers={"Content-Type": "application/json"})).read())
    assert r["usage"]["prompt_tokens"] > 20 and r["usage"]["completion_tokens"] > 0, r["usage"]
    bad = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    try:
        urllib.request.urlopen(urllib.request.Request(f"{base}/chat/completions", data=json.dumps(bad).encode(),
                                                      headers={"Content-Type": "application/json"})); assert False
    except urllib.error.HTTPError as e:
        assert e.code == 400
