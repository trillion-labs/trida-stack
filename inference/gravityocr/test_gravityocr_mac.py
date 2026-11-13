# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for serve_mac.py: request parsing, the HTTP handler and argument checks.

Gravity.app and MLX are not needed: the engine is replaced by a stub.
"""

from __future__ import annotations

import base64
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
        serve_mac.main(["--host", host, "--gravity-app", str(tmp_path)])
    assert e.value.code == 2


def test_missing_gravity_app_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        serve_mac.main(["--gravity-app", str(tmp_path / "Gravity.app")])
    assert e.value.code == 2


def test_engine_import_writes_no_bytecode_into_the_app(tmp_path, monkeypatch):
    # Gravity.app is signed: a __pycache__ written inside it breaks the bundle's seal. Write bytecode
    # next to the sources, as Gravity's runtime does (Apple's system Python redirects it elsewhere).
    import sys
    monkeypatch.setattr(sys, "pycache_prefix", None)
    pkg = tmp_path / "Gravity.app" / "Contents" / "Resources" / "project" / "ocr"
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
        engine = serve_mac.load_engine(tmp_path / "Gravity.app", tmp_path, "causal")
        assert engine.checkpoint == tmp_path
    finally:
        for name in [m for m in sys.modules if m == "ocr" or m.startswith("ocr.")]:
            del sys.modules[name]
        sys.path.remove(str(tmp_path / "Gravity.app" / "Contents" / "Resources" / "project"))
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
