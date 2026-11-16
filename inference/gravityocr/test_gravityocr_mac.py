# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for serve_mac.py: request parsing, the HTTP handler and argument checks.

MLX is not needed: the engine is replaced by a stub. Also checks the vendored engine in mac/ocr against
mac/VENDORED.sha256.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name, filename):
    # Load by path under a unique name, like the other gravityocr tests: the pytest pythonpath holds
    # other models' sibling modules.
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve_mac = _load("gravityocr_serve_mac", "serve_mac.py")
ocr_request = _load("gravityocr_ocr_request_for_mac", "ocr_request.py")

PNG = b"\x89PNG\r\n\x1a\nfake"


def _body(prompt="Text Recognition:", url=None):
    url = url or "data:image/png;base64," + base64.b64encode(PNG).decode()
    return json.dumps({"model": "glm-ocr", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": prompt}]}]}).encode()


@pytest.mark.parametrize("prompt,task", [("Text Recognition:", "text"),
                                         ("Table Recognition:", "table"),
                                         ("Formula Recognition:", "formula")])
def test_parse_maps_prompt_to_task_and_decodes_image(prompt, task):
    assert serve_mac.parse_chat_request(_body(prompt)) == (PNG, task)


def test_parse_accepts_what_ocr_request_sends():
    payload = ocr_request.build_payload(PNG, "image/png", "Table Recognition:", "glm-ocr", 4096)
    assert serve_mac.parse_chat_request(json.dumps(payload).encode()) == (PNG, "table")


@pytest.mark.parametrize("body", [
    b"not json",
    json.dumps({"messages": []}).encode(),
    _body(url="https://example.com/a.png"),
    _body(url="file:///etc/passwd"),
    _body(url="data:image/png;base64,!!!"),
    _body(prompt="Describe this image"),
])
def test_parse_rejects_bad_requests(body):
    with pytest.raises(serve_mac.BadRequest):
        serve_mac.parse_chat_request(body)


@pytest.fixture
def server():
    calls = []

    def recognize(image, task):
        calls.append((image, task))
        if task == "formula":
            raise RuntimeError("engine failed")
        return f"read {task}", 33, 5

    httpd = HTTPServer(("127.0.0.1", 0), serve_mac.make_handler(recognize, "glm-ocr"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", calls
    httpd.shutdown()


def _post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_models_lists_the_served_name(server):
    url, _ = server
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as r:
        assert json.loads(r.read())["data"][0]["id"] == "glm-ocr"


def test_chat_returns_openai_shape(server):
    url, calls = server
    status, out = _post(url, _body("Table Recognition:"))
    assert status == 200
    assert out["choices"][0]["message"]["content"] == "read table"
    assert out["usage"] == {"prompt_tokens": 33, "completion_tokens": 5, "total_tokens": 38}
    assert calls == [(PNG, "table")]


def test_bad_request_is_400_and_engine_error_is_500_and_server_survives(server):
    url, _ = server
    assert _post(url, _body(url="https://example.com/a.png"))[0] == 400
    assert _post(url, _body("Formula Recognition:"))[0] == 500
    assert _post(url, _body())[0] == 200


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com"])
def test_non_loopback_host_is_refused(host, tmp_path):
    with pytest.raises(SystemExit) as e:
        serve_mac.main(["--host", host, "--engine", str(tmp_path)])
    assert e.value.code == 2


def test_missing_engine_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        serve_mac.main(["--engine", str(tmp_path)])
    assert e.value.code == 2


def test_default_engine_is_the_vendored_copy():
    assert serve_mac.DEFAULT_ENGINE == HERE / "mac"
    assert (serve_mac.DEFAULT_ENGINE / "ocr" / "engine" / "pipeline.py").is_file()


def test_vendored_engine_matches_its_checksums():
    # mac/ocr is a byte-for-byte copy of the Gravity app's engine (mac/VENDORED.md): no file edited,
    # added or removed.
    mac = HERE / "mac"
    want = {}
    for line in (mac / "VENDORED.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        want[name] = digest
    have = {str(f.relative_to(mac)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((mac / "ocr").rglob("*"))
            if f.is_file() and "__pycache__" not in f.parts}
    assert have == want


def test_engine_import_writes_no_bytecode_into_the_app(tmp_path, monkeypatch):
    # Gravity.app is signed: a __pycache__ written inside it breaks the bundle's seal. Write bytecode
    # next to the sources, as Gravity's runtime does (Apple's system Python redirects it elsewhere).
    import sys
    monkeypatch.setattr(sys, "pycache_prefix", None)
    project = tmp_path / "Gravity.app" / "Contents" / "Resources" / "project"
    pkg = project / "ocr"
    for sub in ("engine", "recognition"):
        (pkg / sub).mkdir(parents=True)
        (pkg / sub / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "engine" / "pipeline.py").write_text(
        "class LocalOcrEngine:\n"
        "    def __init__(self, checkpoint, **kw):\n"
        "        self.checkpoint = checkpoint\n")
    (pkg / "recognition" / "model.py").write_text("def decode_ar_mlx_from_prefix(*a, **k):\n    pass\n")
    try:
        engine = serve_mac.load_engine(project, tmp_path, "causal")
        assert engine.checkpoint == tmp_path
    finally:
        for name in [m for m in sys.modules if m == "ocr" or m.startswith("ocr.")]:
            del sys.modules[name]
        sys.path.remove(str(project))
    assert not list((tmp_path / "Gravity.app").rglob("__pycache__"))


def test_licence_header():
    head = Path(serve_mac.__file__).read_text().splitlines()[:3]
    assert "# SPDX-License-Identifier: Apache-2.0" in head


def test_concurrent_requests_queue_instead_of_resetting():
    # The notebook's page section sends 8 crops at once; the default listen backlog (5) reset some.
    import time
    from concurrent.futures import ThreadPoolExecutor

    def recognize(image, task):
        time.sleep(0.05)  # one at a time, like the engine
        return "ok", 1, 1

    httpd = serve_mac.Server(("127.0.0.1", 0), serve_mac.make_handler(recognize, "glm-ocr"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with ThreadPoolExecutor(32) as pool:
            codes = list(pool.map(lambda _: _post(url, _body())[0], range(32)))
    finally:
        httpd.shutdown()
    assert codes == [200] * 32


# ---- checkpoint fetch: revision, offline, token ----------------------------------------------------
SHA = "a" * 40


def _snapshot(cache, repo, sha, complete=True):
    d = cache / repo.replace("/", "--") / sha
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    if complete:
        (d / serve_mac.COMPLETE).write_text(sha)
    return d


def _no_network(*_a, **_k):
    raise urllib.error.URLError("offline")


def test_cached_full_commit_needs_no_network(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path, "org/m", SHA)
    monkeypatch.setattr(serve_mac, "_open", _no_network)
    assert serve_mac.fetch_checkpoint("org/m", cache=tmp_path, revision=SHA) == snap


def test_offline_falls_back_to_the_newest_complete_snapshot(tmp_path, monkeypatch):
    import os

    old, new = _snapshot(tmp_path, "org/m", "b" * 40), _snapshot(tmp_path, "org/m", "c" * 40)
    _snapshot(tmp_path, "org/m", "d" * 40, complete=False)  # an interrupted download is never used
    os.utime(old / serve_mac.COMPLETE, (1, 1))
    monkeypatch.setattr(serve_mac, "_open", _no_network)
    assert serve_mac.fetch_checkpoint("org/m", cache=tmp_path) == new


def test_offline_with_nothing_cached_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(serve_mac, "_open", _no_network)
    with pytest.raises(RuntimeError, match="nothing is cached"):
        serve_mac.fetch_checkpoint("org/m", cache=tmp_path)


def test_hub_refusal_is_not_hidden_by_the_cache(tmp_path, monkeypatch):
    _snapshot(tmp_path, "org/m", SHA)

    def refused(url, timeout):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(serve_mac, "_open", refused)
    with pytest.raises(urllib.error.HTTPError):
        serve_mac.fetch_checkpoint("org/m", cache=tmp_path)


def test_token_is_sent_to_the_hub_but_not_across_a_redirect(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert serve_mac._hf_token() == "hf_secret"
    req = urllib.request.Request("https://huggingface.co/org/m/resolve/x/model.safetensors",
                                 headers={"Authorization": "Bearer hf_secret"})
    handler = serve_mac._DropAuthOnOtherHost()
    cdn = handler.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.org/blob")
    assert not cdn.has_header("Authorization")
    same = handler.redirect_request(req, None, 302, "Found", {}, "https://huggingface.co/other")
    assert same.get_header("Authorization") == "Bearer hf_secret"


# ---- the vendored engine's import boundary ----------------------------------------------------------
def test_vendored_engine_imports_no_torch_or_transformers():
    # GravityOCR-macOS enforces this with tests/test_import_boundaries.py, which is not copied. The lock has
    # no torch or transformers, so an engine update that needs them would only fail on a Mac; this catches it
    # on any CI host.
    import ast

    banned = {"torch", "torchvision", "transformers"}
    found = []
    for path in sorted((HERE / "mac" / "ocr").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            found += [f"{path.relative_to(HERE)}: {n}" for n in names if n.split(".")[0] in banned]
    assert not found, found
