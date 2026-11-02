# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""ocr_request.py: request encoding and CLI, CPU only, no network (urlopen is replaced)."""

import base64
import http.client
import importlib.util
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def ocr_request():
    # Loaded from this directory under a unique module name, so it cannot collide with another
    # model's sibling modules on sys.path.
    spec = importlib.util.spec_from_file_location("gravityocr_ocr_request", HERE / "ocr_request.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeServer:
    """Stands in for urllib.request.urlopen and records the request it gets."""

    def __init__(self, content="ok", error=None, raw=None):
        self.content, self.error, self.raw, self.requests = content, error, raw, []

    def __call__(self, request, timeout=None):
        self.requests.append((request, timeout))
        if self.error is not None:
            raise self.error
        if self.raw is not None:
            return io.BytesIO(self.raw)
        body = {
            "choices": [{"message": {"content": self.content}}],
            "usage": {"completion_tokens": 1},
        }
        return io.BytesIO(json.dumps(body).encode())

    @property
    def body(self):
        return json.loads(self.requests[-1][0].data)


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.delenv("API_KEY", raising=False)
    return fake


def _decode_data_url(url):
    head, b64 = url.split(",", 1)
    assert head.startswith("data:") and head.endswith(";base64")
    return head[len("data:") : -len(";base64")], base64.b64decode(b64)


def test_payload_shape(ocr_request):
    image = bytes(range(256)) * 3
    body = ocr_request.build_payload(image, "image/jpeg", "Table Recognition:", "glm-ocr", 123)
    assert body["model"] == "glm-ocr" and body["max_tokens"] == 123 and body["temperature"] == 0.0
    (message,) = body["messages"]
    assert message["role"] == "user"
    image_part, text_part = message["content"]  # image first, then the task prompt
    assert image_part["type"] == "image_url"
    assert text_part == {"type": "text", "text": "Table Recognition:"}
    assert _decode_data_url(image_part["image_url"]["url"]) == ("image/jpeg", image)
    json.dumps(body)  # serialisable as is


def test_task_prompts(ocr_request):
    assert ocr_request.TASK_PROMPTS == {
        "text": "Text Recognition:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    }
    assert ocr_request.DEFAULT_PROMPT == "Text Recognition:"
    assert ocr_request.DEFAULT_MODEL == "glm-ocr"
    assert ocr_request.DEFAULT_URL == "http://127.0.0.1:30000"


@pytest.mark.parametrize(
    "name,mime",
    [
        ("a.png", "image/png"),
        ("b.JPG", "image/jpeg"),
        ("c.jpeg", "image/jpeg"),
        ("d.webp", "image/webp"),
        ("e", "image/png"),
        ("f.unknown", "image/png"),
    ],
)
def test_guess_mime(ocr_request, name, mime):
    assert ocr_request.guess_mime(name) == mime


def test_main_sends_one_request(ocr_request, server, tmp_path, capsys):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"\xff\xd8not-really-a-jpeg")
    rc = ocr_request.main(
        [
            str(crop),
            "--url",
            "http://gpu-box:1234/",
            "--prompt",
            "Formula Recognition:",
            "--model",
            "gravityocr",
            "--max-tokens",
            "77",
            "--api-key",
            "k3y",
            "--timeout",
            "5",
        ]
    )
    assert rc == 0
    request, timeout = server.requests[-1]
    assert request.full_url == "http://gpu-box:1234/v1/chat/completions"
    assert request.get_method() == "POST" and timeout == 5.0
    assert request.get_header("Authorization") == "Bearer k3y"
    assert request.get_header("Content-type") == "application/json"
    body = server.body
    assert body["model"] == "gravityocr" and body["max_tokens"] == 77 and body["temperature"] == 0.0
    image_part, text_part = body["messages"][0]["content"]
    assert text_part["text"] == "Formula Recognition:"
    assert _decode_data_url(image_part["image_url"]["url"]) == ("image/jpeg", crop.read_bytes())
    out = capsys.readouterr()
    assert out.out == "ok\n" and "usage" in out.err


def test_defaults(ocr_request, server, tmp_path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(PNG_SIGNATURE)
    assert ocr_request.main([str(crop)]) == 0
    request, timeout = server.requests[-1]
    assert request.full_url == "http://127.0.0.1:30000/v1/chat/completions"
    assert request.get_header("Authorization") is None  # no key, no header
    assert server.body["model"] == "glm-ocr" and server.body["max_tokens"] == 4096
    assert server.body["messages"][0]["content"][1]["text"] == "Text Recognition:"
    assert timeout == 300.0


def test_api_key_from_environment(ocr_request, server, tmp_path, monkeypatch):
    crop = tmp_path / "crop.png"
    crop.write_bytes(PNG_SIGNATURE)
    monkeypatch.setenv("API_KEY", "from-env")
    assert ocr_request.main([str(crop)]) == 0
    assert server.requests[-1][0].get_header("Authorization") == "Bearer from-env"


def test_http_error_is_reported(ocr_request, server, tmp_path, capsys):
    crop = tmp_path / "crop.png"
    crop.write_bytes(PNG_SIGNATURE)
    server.error = urllib.error.HTTPError(
        "http://x", 401, "Unauthorized", {}, io.BytesIO(b'{"error": "bad key"}')
    )
    assert ocr_request.main([str(crop)]) == 1
    err = capsys.readouterr().err
    assert "HTTP 401" in err and "bad key" in err


def test_unreachable_server_is_reported(ocr_request, server, tmp_path, capsys):
    crop = tmp_path / "crop.png"
    crop.write_bytes(PNG_SIGNATURE)
    server.error = urllib.error.URLError("connection refused")
    assert ocr_request.main([str(crop)]) == 1
    assert "connection refused" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error,needle",
    [
        (TimeoutError("timed out"), "no reply from"),  # read timeout after the connection opened
        (http.client.RemoteDisconnected("closed"), "connection to"),
        (ConnectionResetError(104, "reset"), "connection to"),
        (http.client.IncompleteRead(b"partial"), "connection to"),
    ],
)
def test_timeouts_and_dropped_connections_are_reported(
    ocr_request, server, tmp_path, capsys, error, needle
):
    crop = tmp_path / "crop.png"
    crop.write_bytes(PNG_SIGNATURE)
    server.error = error
    assert ocr_request.main([str(crop), "--timeout", "7"]) == 1
    err = capsys.readouterr().err
    assert needle in err and "Traceback" not in err


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b'{"error": "oops"}',
        b'{"choices": []}',
        b'{"choices": [{"message": {"content": null}}]}',
        b"[]",
    ],
)
def test_unexpected_replies_are_reported(ocr_request, server, tmp_path, capsys, raw):
    crop = tmp_path / "crop.png"
    crop.write_bytes(PNG_SIGNATURE)
    server.raw = raw
    assert ocr_request.main([str(crop)]) == 1
    captured = capsys.readouterr()
    assert "unexpected reply" in captured.err and captured.out == ""


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["a.png", "--synthetic", "x"],
        ["a.png", "--url", "file:///etc"],
        ["a.png", "--url", "127.0.0.1:30000"],
        ["a.png", "--max-tokens", "0"],
    ],
)
def test_bad_arguments_exit_2(ocr_request, server, argv):
    with pytest.raises(SystemExit) as exc:
        ocr_request.main(argv)
    assert exc.value.code == 2
    assert not server.requests


def test_synthetic_renders_a_png(ocr_request, server):
    image_mod = pytest.importorskip("PIL.Image")
    png = ocr_request.render_text_png("Hello GravityOCR")
    assert png.startswith(PNG_SIGNATURE)
    with image_mod.open(io.BytesIO(png)) as img:
        assert img.format == "PNG" and img.width > 50 and img.height > 10
        assert img.convert("L").getextrema()[0] < 128  # some text was actually drawn
    assert ocr_request.main(["--synthetic", "Hello GravityOCR"]) == 0
    mime, data = _decode_data_url(server.body["messages"][0]["content"][0]["image_url"]["url"])
    assert mime == "image/png" and data.startswith(PNG_SIGNATURE)
