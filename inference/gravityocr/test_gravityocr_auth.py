# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""GravityOCR SGLang recipe: the patched server's API-key handling, on CPU (no GPU, no model).

These tests import the patched SGLang server (sglang/setup.sh's venv), so trida CI, which has no
SGLang, skips them. Run them with that venv's Python, from the repository root:

    CUDA_VISIBLE_DEVICES="" "$VBIN/python" -m pytest -q inference/gravityocr/test_gravityocr_auth.py

(`$VBIN` as printed by setup.sh; pytest must be importable by that Python.)

What they check:
- No secret (api_key, admin_api_key, ssl_keyfile_password) is returned by /server_info or
  /get_server_info (top level and the scheduler's internal states), by Engine.get_server_info or by
  /set_internal_state, appears in repr(ServerArgs) (the startup log line), or is written to the
  request / crash dumps; reading the internal state no longer changes the live server arguments.
- Every tokenizer worker enforces the keys: the auth middleware sits on the module-level app that
  each uvicorn worker process imports, the keys reach each worker with its pickled uvicorn target
  (never through the /dev/shm args segment), a worker without keys answers 503 instead of serving
  open, and a worker started with an API key no longer fails its startup assertion.
- The multi-tokenizer args segment in /dev/shm is created exclusively and only read back when this
  user created it, so another local user cannot read it or plant a pickle for the workers.

Names added by the patch are looked up when a test runs, so on an unpatched build the tests fail on
their assertions instead of failing to import.
"""

import asyncio
import functools
import http.client
import multiprocessing
import os
import pickle
import signal
import socket
import subprocess
import sys
import textwrap
import time
import types
import uuid

import pytest

pytest.importorskip("fastapi")
hs = pytest.importorskip("sglang.srt.entrypoints.http_server")

from fastapi.testclient import TestClient  # noqa: E402

import sglang.srt.managers.multi_tokenizer_mixin as mixin  # noqa: E402
import sglang.srt.utils.auth as auth  # noqa: E402

try:  # added by the patch: the keys each HTTP worker process enforces
    import sglang.srt.entrypoints.http_worker_keys as worker_keys  # noqa: E402
except ImportError:
    worker_keys = None
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)

API = "gravityocr-test-api-key-3f9c1e"
ADMIN = "gravityocr-test-admin-key-a71d04"
SSL_PASSWORD = "gravityocr-test-ssl-password-5b2e"
SECRETS = (API, ADMIN, SSL_PASSWORD)


# ---------------------------------------------------------------- helpers
def _new(name, module=auth):
    """A name the patch adds (fails the test on an unpatched build)."""
    value = getattr(module, name, None) if module is not None else None
    assert value is not None, f"{name} is missing: unpatched SGLang"
    return value


def _keys_module():
    assert worker_keys is not None, "sglang.srt.entrypoints.http_worker_keys is missing: unpatched SGLang"
    return worker_keys


def _set_keys(monkeypatch, api_key, admin_api_key):
    """Configure the keys the module-level middleware enforces in this process (restored after the
    test). An unpatched build has no such middleware, so there is nothing to configure."""
    if worker_keys is not None:
        monkeypatch.setattr(worker_keys, "_auth_keys", None)
        worker_keys.set_auth_keys(api_key, admin_api_key)


def _server_args(**kw):
    sa = ServerArgs(model_path="dummy", **kw)
    sa.ssl_keyfile_password = SSL_PASSWORD  # set after __post_init__: it requires --ssl-keyfile
    return sa


class _StubScheduler(types.SimpleNamespace):
    """What Scheduler.get_internal_state / set_internal_state read from `self`."""

    def __init__(self):
        mr = types.SimpleNamespace(weight_load_mem_usage=1.0, graph_mem_usage=0.5)
        kv = types.SimpleNamespace(get_kvcache=lambda: types.SimpleNamespace(mem_usage=2.0))
        super().__init__(
            last_gen_throughput=0.0,
            tp_worker=types.SimpleNamespace(model_runner=mr),
            token_to_kv_pool_allocator=kv,
            max_total_num_tokens=1,
            max_running_requests=8,
            pp_size=1,
            spec_algorithm=types.SimpleNamespace(is_none=lambda: True),
            spec_total_num_forward_ct=0,
            spec_total_num_accepted_tokens=0,
        )


def _scheduler_internal_state():
    """The real Scheduler.get_internal_state, as run in the scheduler process (global args)."""
    from sglang.srt.managers.io_struct import GetInternalStateReq
    from sglang.srt.managers.scheduler import Scheduler

    return Scheduler.get_internal_state(_StubScheduler(), GetInternalStateReq()).internal_state


class _TokenizerManager:
    """The parts of TokenizerManager the routes below use."""

    def __init__(self, server_args):
        self.server_args = server_args
        self.served_model_name = "glm-ocr"
        self.model_config = types.SimpleNamespace(context_len=16384)
        self.calls = []

    async def get_internal_state(self):
        return [_scheduler_internal_state()]

    def configure_logging(self, obj):
        self.calls.append("configure_logging")


@pytest.fixture
def server(monkeypatch):
    """The real module-level app with a stub tokenizer manager; global server args = the stub's."""
    sa = _server_args(api_key=API, admin_api_key=ADMIN)
    tm = _TokenizerManager(sa)
    monkeypatch.setattr(hs, "_global_state", None)
    hs.set_global_state(hs._GlobalState(tokenizer_manager=tm, template_manager=None, scheduler_info={}))
    previous = get_global_server_args() if _has_global_args() else None
    set_global_server_args_for_scheduler(sa)
    yield types.SimpleNamespace(args=sa, tm=tm)
    if previous is not None:
        set_global_server_args_for_scheduler(previous)


def _has_global_args():
    try:
        get_global_server_args()
        return True
    except Exception:
        return False


def _bearer(key):
    return {"Authorization": f"Bearer {key}"} if key else {}


def _assert_no_secret(text, where):
    for secret in SECRETS:
        assert secret not in text, f"{where} contains a secret ({secret[:20]}...)"


# ---------------------------------------------------------------- G2: secrets never echoed
def test_redact_server_args_copies_and_blanks_every_secret():
    redact = _new("redact_server_args")
    src = {"api_key": API, "admin_api_key": ADMIN, "ssl_keyfile_password": SSL_PASSWORD, "port": 1}
    out = redact(src)
    assert out == {"api_key": None, "admin_api_key": None, "ssl_keyfile_password": None, "port": 1}
    assert src["api_key"] == API and out is not src  # the input is untouched
    assert redact({"port": 2}) == {"port": 2}  # missing fields are not added
    assert set(_new("SECRET_SERVER_ARG_FIELDS")) == {"api_key", "admin_api_key", "ssl_keyfile_password"}


def test_server_args_repr_hides_the_secrets_but_keeps_them_usable():
    sa = _server_args(api_key=API, admin_api_key=ADMIN)
    _assert_no_secret(repr(sa), "repr(ServerArgs)")  # the startup "server_args=..." log line
    _assert_no_secret(f"{sa=}", "the f-string the startup log uses")
    assert (sa.api_key, sa.admin_api_key, sa.ssl_keyfile_password) == (API, ADMIN, SSL_PASSWORD)
    assert "tokenizer_worker_num=" in repr(sa)  # the rest of the line is unchanged


@pytest.mark.parametrize("path", ["/server_info", "/get_server_info"])
def test_server_info_returns_no_secret(server, monkeypatch, path):
    _set_keys(monkeypatch, API, ADMIN)
    r = TestClient(hs.app, raise_server_exceptions=False).get(path, headers=_bearer(API))
    assert r.status_code == 200
    _assert_no_secret(r.text, f"GET {path}")  # the raw body, wherever it might be
    body = r.json()
    for field in ("api_key", "admin_api_key", "ssl_keyfile_password"):
        assert body[field] is None
        for state in body["internal_states"]:
            assert state.get(field) is None
    assert body["model_path"] == "dummy" and body["internal_states"][0]["last_gen_throughput"] == 0.0


def test_server_info_leaves_the_live_server_args_alone(server, monkeypatch):
    _set_keys(monkeypatch, API, ADMIN)
    before = dict(vars(get_global_server_args()))
    TestClient(hs.app, raise_server_exceptions=False).get("/server_info", headers=_bearer(API))
    after = dict(vars(get_global_server_args()))
    assert after.keys() == before.keys(), f"added {sorted(after.keys() - before.keys())}"
    assert get_global_server_args().admin_api_key == ADMIN  # the server keeps its keys


def test_set_internal_state_returns_no_secret_and_logs_none(server, caplog):
    from sglang.srt.managers.io_struct import SetInternalStateReq
    from sglang.srt.managers.scheduler import Scheduler

    caplog.set_level("INFO")
    req = SetInternalStateReq(server_args={"speculative_accept_threshold_single": 1.0})
    out = Scheduler.set_internal_state(_StubScheduler(), req)
    assert out.updated
    assert out.server_args["api_key"] is None and out.server_args["admin_api_key"] is None
    _assert_no_secret(repr(out.server_args), "set_internal_state's reply")
    assert "Global server args updated" in caplog.text
    _assert_no_secret(caplog.text, "the scheduler log")


def test_engine_get_server_info_returns_no_secret():
    from sglang.srt.entrypoints.engine import Engine

    sa = _server_args(api_key=API, admin_api_key=ADMIN)
    set_global_server_args_for_scheduler(sa)
    tm = _TokenizerManager(sa)
    stub = types.SimpleNamespace(
        loop=asyncio.new_event_loop(),
        tokenizer_manager=tm,
        _scheduler_init_result=types.SimpleNamespace(scheduler_infos=[{"max_total_num_tokens": 1}]),
    )
    try:
        info = Engine.get_server_info(stub)
    finally:
        stub.loop.close()
    _assert_no_secret(repr(info), "Engine.get_server_info")
    assert info["api_key"] is None and info["internal_states"][0]["admin_api_key"] is None


def test_request_and_crash_dumps_carry_no_secret(tmp_path):
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    sa = _server_args(api_key=API, admin_api_key=ADMIN)

    async def dump_requests():
        stub = types.SimpleNamespace(server_args=sa)
        TokenizerManager._dump_data_to_file(stub, [("req", {}, 0.0, 0.0)], str(tmp_path / "d" / "r.pkl"), "dump")
        for _ in range(200):
            await asyncio.sleep(0.01)
            if (tmp_path / "d" / "r.pkl").exists() and (tmp_path / "d" / "r.pkl").stat().st_size:
                break

    asyncio.run(dump_requests())
    crash = types.SimpleNamespace(
        server_args=sa, crash_dump_folder=str(tmp_path / "c"), crash_dump_performed=False,
        crash_dump_request_list=[("req", {}, 0.0, 0.0)], rid_to_state={},
    )
    crash_file = TokenizerManager.dump_requests_before_crash(crash, hostname="h")
    time.sleep(0.2)
    for f in (tmp_path / "d" / "r.pkl", crash_file):
        raw = open(f, "rb").read()
        _assert_no_secret(raw.decode("latin-1"), f"dump {f}")
        dumped = pickle.loads(raw)["server_args"]
        assert dumped.model_path == "dummy" and dumped.api_key is None
    assert sa.api_key == API  # the server keeps its keys


# ---------------------------------------------------------------- G5: keys in every tokenizer worker
def _auth_middleware_entries(app):
    return [m for m in app.user_middleware if m.cls.__name__ == "_ApiKeyASGIMiddleware"]


def test_module_level_app_carries_the_auth_middleware():
    # A multi-tokenizer worker serves a freshly imported `app`; only middleware added at import reaches it.
    entries = _auth_middleware_entries(hs.app)
    assert len(entries) == 1, [m.cls.__name__ for m in hs.app.user_middleware]
    assert entries[0].kwargs["keys"] is _keys_module().get_auth_keys


def test_worker_without_keys_fails_closed(server, monkeypatch):
    monkeypatch.setattr(_keys_module(), "_auth_keys", None)
    c = TestClient(hs.app, raise_server_exceptions=False)
    assert c.get("/v1/models", headers=_bearer(API)).status_code == 503
    assert c.post("/configure_logging", json={}, headers=_bearer(ADMIN)).status_code == 503
    assert server.tm.calls == []
    assert c.options("/v1/models").status_code != 503  # CORS preflight still passes
    assert c.get("/health").status_code != 503  # liveness probes still reach the handler


@pytest.mark.parametrize(
    "keys, path, header, expect",
    [
        # API key + admin key (the launcher's configuration when API_KEY is set)
        ((API, ADMIN), "/v1/models", None, 401),
        ((API, ADMIN), "/v1/models", API, 200),
        ((API, ADMIN), "/v1/models", ADMIN, 401),
        ((API, ADMIN), "/configure_logging", None, 401),
        ((API, ADMIN), "/configure_logging", API, 401),
        ((API, ADMIN), "/configure_logging", ADMIN, 200),
        # admin key only (the launcher's configuration without API_KEY)
        ((None, ADMIN), "/v1/models", None, 200),
        ((None, ADMIN), "/configure_logging", None, 401),
        ((None, ADMIN), "/configure_logging", "wrong-key", 401),
        ((None, ADMIN), "/configure_logging", ADMIN, 200),
        # /set_trace_level is an admin route too (upstream v0.5.10 left it at the normal level)
        ((None, ADMIN), "/set_trace_level?level=0", None, 401),
        ((API, ADMIN), "/set_trace_level?level=0", API, 401),
        ((None, ADMIN), "/set_trace_level?level=0", ADMIN, 200),
        # no keys: upstream behaviour (no auth)
        ((None, None), "/v1/models", None, 200),
        ((None, None), "/configure_logging", None, 200),
    ],
)
def test_keys_are_enforced_per_auth_level(server, monkeypatch, keys, path, header, expect):
    _set_keys(monkeypatch, *keys)
    c = TestClient(hs.app, raise_server_exceptions=False)
    if path == "/configure_logging":
        r = c.post(path, json={}, headers=_bearer(header))
        assert (server.tm.calls == ["configure_logging"]) == (expect == 200)
    else:
        r = c.get(path, headers=_bearer(header))
    assert r.status_code == expect


def test_fixed_key_helper_keeps_its_upstream_behaviour():
    # add_api_key_middleware (fixed keys) is no longer used by the server but stays for other callers.
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/plain")
    async def plain():
        return {}

    @app.post("/admin")
    @auth.auth_level(auth.AuthLevel.ADMIN_OPTIONAL)
    async def admin():
        return {}

    auth.add_api_key_middleware(app, api_key=API, admin_api_key=ADMIN)
    c = TestClient(app)
    assert [c.get("/plain", headers=_bearer(k)).status_code for k in (None, API, ADMIN)] == [401, 200, 401]
    assert [c.post("/admin", headers=_bearer(k)).status_code for k in (None, API, ADMIN)] == [401, 401, 200]


def _fake_shm_reader(server_args):
    """read_from_shared_memory as a worker sees it: an unpickled copy of what the parent wrote."""
    blob = pickle.dumps((types.SimpleNamespace(tokenizer_ipc_name=None), server_args, {"max_req_input_len": 1}))
    return lambda name: pickle.loads(blob)


def _stub_worker_classes(monkeypatch):
    class _Worker:
        def __init__(self, server_args, port_args):
            self.server_args, self.worker_id = server_args, 0

    class _Templates:
        def initialize_templates(self, **kw):
            pass

    monkeypatch.setattr(hs, "TokenizerWorker", _Worker)
    monkeypatch.setattr(hs, "TemplateManager", _Templates)
    monkeypatch.setattr(hs, "get_main_process_id", lambda: 1)
    monkeypatch.setattr(hs, "_global_state", None)


def test_worker_startup_accepts_an_api_key_and_restores_the_keys(monkeypatch):
    sa = _server_args(api_key=API, admin_api_key=ADMIN)
    strip = getattr(auth, "server_args_without_secrets", None)
    shm_args = strip(sa) if strip else sa  # what the parent writes to /dev/shm
    _stub_worker_classes(monkeypatch)
    monkeypatch.setattr(hs, "read_from_shared_memory", _fake_shm_reader(shm_args))
    _set_keys(monkeypatch, API, ADMIN)
    # unpatched: AssertionError "API key is not supported in multi-tokenizer mode"
    args = asyncio.run(hs.init_multi_tokenizer())
    # restored from the worker's target, for its own warmup request and the handlers that read server_args
    assert (args.api_key, args.admin_api_key) == (API, ADMIN)
    assert (shm_args.api_key, shm_args.admin_api_key) == (None, None)


def test_worker_startup_without_keys_is_refused(monkeypatch):
    monkeypatch.setattr(_keys_module(), "_auth_keys", None)
    _stub_worker_classes(monkeypatch)
    monkeypatch.setattr(hs, "read_from_shared_memory", _fake_shm_reader(_server_args()))
    with pytest.raises(RuntimeError, match="without its API keys"):
        asyncio.run(hs.init_multi_tokenizer())


def _patch_launch(monkeypatch, captured):
    monkeypatch.setattr(hs, "set_uvicorn_logging_configs", lambda server_args: None)
    monkeypatch.setattr(hs, "monkey_patch_uvicorn_multiprocessing", lambda: None)
    monkeypatch.setattr(hs, "_global_state", None)
    for attr in ("is_single_tokenizer_mode", "server_args", "warmup_thread_kwargs"):
        monkeypatch.setattr(hs.app, attr, getattr(hs.app, attr, None), raising=False)
    monkeypatch.setattr(hs.app, "user_middleware", list(hs.app.user_middleware))

    def fake_write(port_args, server_args, scheduler_info):
        captured["shm_args"] = pickle.loads(pickle.dumps(server_args))
        return types.SimpleNamespace(unlink=lambda: captured.setdefault("unlinked", True))

    def fake_workers(app_path, auth_keys, **kwargs):
        captured["workers"] = (app_path, auth_keys, kwargs)

    def fake_uvicorn_run(app, **kwargs):
        captured["uvicorn.run"] = (app, kwargs)
        captured["keys_at_start"] = worker_keys.get_auth_keys() if worker_keys else None

    monkeypatch.setattr(hs, "write_data_for_multi_tokenizer", fake_write)
    monkeypatch.setattr(hs, "_run_uvicorn_workers", fake_workers, raising=False)
    monkeypatch.setattr(hs.uvicorn, "run", fake_uvicorn_run)


def _tokenizer_manager_stub():
    return types.SimpleNamespace(socket_mapping=types.SimpleNamespace(clear_all_sockets=lambda: None))


def test_multi_worker_launch_keeps_secrets_out_of_shared_memory(monkeypatch):
    captured = {}
    _patch_launch(monkeypatch, captured)
    sa = _server_args(api_key=API, admin_api_key=ADMIN, tokenizer_worker_num=2)
    hs._setup_and_run_http_server(sa, _tokenizer_manager_stub(), None, None, [{}], None)
    shm_args = captured["shm_args"]
    assert (shm_args.api_key, shm_args.admin_api_key, shm_args.ssl_keyfile_password) == (None, None, None)
    _assert_no_secret(pickle.dumps(shm_args).decode("latin-1"), "the /dev/shm args segment")
    assert "workers" in captured, "the workers were started with plain uvicorn.run (no keys)"
    app_path, keys, kwargs = captured["workers"]
    assert app_path == "sglang.srt.entrypoints.http_server:app" and kwargs["workers"] == 2
    assert (keys.api_key, keys.admin_api_key) == (API, ADMIN)
    assert API not in repr(keys) and ADMIN not in repr(keys)
    assert (sa.api_key, sa.admin_api_key) == (API, ADMIN)  # the parent's args keep them


def test_single_worker_launch_sets_the_keys_before_serving(monkeypatch):
    captured = {}
    _patch_launch(monkeypatch, captured)
    if worker_keys is not None:
        monkeypatch.setattr(worker_keys, "_auth_keys", None)
    sa = _server_args(api_key=API, admin_api_key=ADMIN, tokenizer_worker_num=1)
    hs._setup_and_run_http_server(sa, _tokenizer_manager_stub(), None, None, [{}], None)
    app, _ = captured["uvicorn.run"]
    assert app is hs.app
    keys = captured["keys_at_start"]
    assert keys is not None and (keys.api_key, keys.admin_api_key) == (API, ADMIN)
    assert len(_auth_middleware_entries(hs.app)) == 1  # set once at import, not added again


def _probe_worker(out_q, sockets=None):
    """Runs in a freshly spawned process, as a uvicorn worker would: imports the server module anew
    and probes its app. The keys come only from the pickled AuthKeysWorkerTarget."""
    import sglang.srt.entrypoints.http_server as fresh
    from fastapi.testclient import TestClient as Client

    tm = _TokenizerManager(_server_args())
    fresh.set_global_state(fresh._GlobalState(tokenizer_manager=tm, template_manager=None, scheduler_info={}))
    c = Client(fresh.app, raise_server_exceptions=False)
    out_q.put(
        {
            "pid": os.getpid(),
            "models_no_key": c.get("/v1/models").status_code,
            "models_api": c.get("/v1/models", headers=_bearer(API)).status_code,
            "admin_no_key": c.post("/configure_logging", json={}).status_code,
            "admin_api": c.post("/configure_logging", json={}, headers=_bearer(API)).status_code,
            "admin_admin": c.post("/configure_logging", json={}, headers=_bearer(ADMIN)).status_code,
            "calls": tm.calls,
        }
    )


def test_keys_reach_freshly_spawned_workers_through_the_pickled_target():
    target_cls = _keys_module().AuthKeysWorkerTarget
    keys = worker_keys.AuthKeys(api_key=API, admin_api_key=ADMIN)
    ctx = multiprocessing.get_context("spawn")  # what uvicorn uses for its workers
    q = ctx.Queue()
    procs = [ctx.Process(target=target_cls(functools.partial(_probe_worker, q), keys)) for _ in range(2)]
    for p in procs:
        p.start()
    try:
        results = [q.get(timeout=300) for _ in procs]
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()
    assert len({r["pid"] for r in results}) == 2
    for r in results:
        assert (r["models_no_key"], r["models_api"]) == (401, 200), r
        assert (r["admin_no_key"], r["admin_api"], r["admin_admin"]) == (401, 401, 200), r
        assert r["calls"] == ["configure_logging"], r


def test_worker_target_unpickles_without_heavy_imports():
    """uvicorn pings a new worker after 0.5 s and kills it unless the worker's pong thread answers
    within 5 s; that thread starts only after the worker has unpickled its target. Unpickling the
    keyed target must therefore not import torch or sglang.srt.utils (several seconds)."""
    keys = _keys_module().AuthKeys(api_key=API, admin_api_key=ADMIN)
    blob = pickle.dumps(worker_keys.AuthKeysWorkerTarget(functools.partial(print, "run"), keys))
    probe = (
        "import pickle, sys; t = pickle.loads(sys.stdin.buffer.read()); "
        "print(sorted(m for m in ('torch', 'sglang.srt.utils', 'sglang.srt.server_args') if m in sys.modules)); "
        "print(type(t).__name__, repr(t.keys))"
    )
    out = subprocess.run([sys.executable, "-c", probe], input=blob, capture_output=True, timeout=120)
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    heavy, rest = out.stdout.decode().splitlines()[:2]
    assert heavy == "[]", heavy
    assert rest.startswith("AuthKeysWorkerTarget AuthKeys(") and API not in rest and ADMIN not in rest


_UVICORN_PARENT = textwrap.dedent(
    """
    import sys
    import sglang.srt.entrypoints.http_server as hs
    from sglang.srt.entrypoints.http_worker_keys import AuthKeys
    api, admin, port = sys.stdin.read().split()
    hs._run_uvicorn_workers(
        "sglang.srt.entrypoints.http_server:app", AuthKeys(api_key=api, admin_api_key=admin),
        host="127.0.0.1", port=int(port), workers=2, lifespan="off", log_level="warning",
        # uvicorn kills a worker that does not answer its health ping in time; give a cold import room
        timeout_worker_healthcheck=120,
    )
    """
)


def _status(port, method, path, key=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)  # a fresh connection each time
    try:
        conn.request(method, path, body=b"{}" if method == "POST" else None,
                     headers={"Content-Type": "application/json", **_bearer(key)})
        return conn.getresponse().status
    finally:
        conn.close()


def test_uvicorn_workers_all_enforce_the_keys(tmp_path):
    """Real uvicorn with two worker processes (spawned by its Multiprocess supervisor), the real
    module app, no model (lifespan off, so a request that passes auth fails later with 500)."""
    _keys_module()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    log = open(tmp_path / "uvicorn.log", "wb")  # a file, not a pipe: nobody drains a pipe while the test runs
    proc = subprocess.Popen([sys.executable, "-c", _UVICORN_PARENT], stdin=subprocess.PIPE,
                            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    proc.stdin.write(f"{API} {ADMIN} {port}".encode())
    proc.stdin.close()
    try:
        deadline = time.time() + 300
        while True:
            assert proc.poll() is None, (tmp_path / "uvicorn.log").read_text(errors="replace")[-3000:]
            try:
                _status(port, "GET", "/health")
                break
            except OSError:
                assert time.time() < deadline, "the uvicorn workers did not start"
                time.sleep(1)
        time.sleep(5)  # let the second worker finish importing, so both take connections
        seen = {}
        for method, path, key in [
            ("GET", "/v1/models", None), ("GET", "/v1/models", API), ("GET", "/v1/models", ADMIN),
            ("POST", "/configure_logging", None), ("POST", "/configure_logging", API),
            ("POST", "/flush_cache", API), ("POST", "/update_weights_from_disk", API),
            ("POST", "/configure_logging", ADMIN),
        ]:
            seen[(method, path, key)] = {_status(port, method, path, key) for _ in range(30)}
        refused = {401}
        for k, codes in seen.items():
            method, path, key = k
            passes = (path == "/v1/models" and key == API) or (path == "/configure_logging" and key == ADMIN)
            if passes:  # reached the handler, which has no model behind it
                assert codes.isdisjoint({401, 403, 503}), (k, codes)
            else:
                assert codes == refused, (k, codes)
        import psutil

        assert len(psutil.Process(proc.pid).children()) >= 2  # two worker processes
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
        time.sleep(1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # anything left in the group (none expected)
        except ProcessLookupError:
            pass
        log.close()


# ---------------------------------------------------------------- the /dev/shm args segment
@pytest.fixture
def segment_name():
    name = f"gravityocr_test_{uuid.uuid4().hex}"
    yield name
    try:
        os.unlink(os.path.join("/dev/shm", name))
    except FileNotFoundError:
        pass


def _plant(name, payload, mode):
    """Create segment `name` the way another local user would (here: as this user, with `mode`)."""
    path = os.path.join("/dev/shm", name)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, mode)
    try:
        os.fchmod(fd, mode)
        os.write(fd, payload.ljust(1 << 16, b"\0"))
    finally:
        os.close(fd)
    return path


_UNPICKLED = []


def _record_unpickle(tag):
    _UNPICKLED.append(tag)
    return tag


class _Planted:
    def __reduce__(self):
        return (_record_unpickle, ("planted pickle ran",))


_linux_only = pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs Linux /dev/shm")


@_linux_only
def test_args_segment_round_trip_is_private(segment_name):
    shm = mixin.write_to_shared_memory({"x": 1}, segment_name)
    try:
        st = os.stat(os.path.join("/dev/shm", segment_name))
        assert st.st_uid == os.geteuid() and st.st_mode & 0o777 == 0o600
        assert mixin.read_from_shared_memory(segment_name) == {"x": 1}
    finally:
        shm.close()


@_linux_only
def test_args_segment_replaces_our_own_stale_leftover(segment_name):
    _plant(segment_name, pickle.dumps("stale"), 0o600)  # a killed server with the same pid
    shm = mixin.write_to_shared_memory({"fresh": True}, segment_name)
    try:
        assert mixin.read_from_shared_memory(segment_name) == {"fresh": True}
    finally:
        shm.close()


@_linux_only
def test_args_segment_planted_by_another_user_is_refused(segment_name, monkeypatch):
    path = _plant(segment_name, b"planted", 0o666)
    real_euid = os.geteuid()
    monkeypatch.setattr(mixin.os, "geteuid", lambda: real_euid + 1)  # the segment is not ours
    with pytest.raises(PermissionError):
        mixin.write_to_shared_memory({"api_key": API}, segment_name)
    monkeypatch.undo()
    with open(path, "rb") as f:
        assert f.read(7) == b"planted"  # nothing was written into it


@_linux_only
def test_args_segment_writable_by_others_is_never_unpickled(segment_name):
    _UNPICKLED.clear()
    _plant(segment_name, pickle.dumps(_Planted()), 0o666)
    with pytest.raises(PermissionError):
        mixin.read_from_shared_memory(segment_name)
    assert _UNPICKLED == []
