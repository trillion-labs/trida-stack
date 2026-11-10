"""Start trida_mlx.server in a subprocess for HTTP tests, so its flags (module globals) never leak into the
test process."""
import contextlib
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


@contextlib.contextmanager
def server(checkpoint, *args):
    """Yields (base URL ending in /v1, the /v1/models card)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "trida_mlx.server", "--model", str(checkpoint),
                             "--port", str(port), *map(str, args)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}/v1"
    try:
        for _ in range(300):
            if proc.poll() is not None:
                pytest.fail(proc.stderr.read().decode())
            try:
                with urllib.request.urlopen(f"{base}/models", timeout=1) as response:
                    card = json.load(response)["data"][0]
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            pytest.fail("server did not start")
        yield base, card
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def post(url, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)
