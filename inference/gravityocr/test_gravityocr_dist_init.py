# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""GravityOCR SGLang patch: a single-node server's torch.distributed rendezvous stays on loopback.

Upstream SGLang v0.5.10 connects a single-node server's torch.distributed rendezvous to `--host`,
the HTTP bind address, and torch's tcp:// rendezvous starts its TCPStore server on every interface
whatever address it names. So every server had a store port open on all interfaces, at the
`SERVE_HOST` address too. The patch changes two things:

- model_runner.py `_torch_dist_init_method`: a single-node server (nnodes 1, no --dist-init-addr,
  no SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE) rendezvouses on 127.0.0.1, whatever `--host` is;
- parallel_state.py `_loopback_rendezvous_store`, used by `init_distributed_environment`: a
  tcp://<loopback IP>:<port> rendezvous gets a store that listens on that address only.

Two groups of checks:

- On the patch text and CPU torch (run in CI): both changes are in the patch, and the store
  function as the patch adds it, run on CPU torch, listens on loopback only (libuv and legacy
  TCPStore backends, IPv4 and IPv6), serves a group of two ranks, closes its port with the store,
  takes the backend's default timeout, and returns None (torch's own rendezvous) for every other
  init method.
- On the patched server (sglang/setup.sh's venv; skipped where SGLang is not installed): the init
  method for every `--host`, with the override and --dist-init-addr still taking precedence and
  multi-node unchanged; `init_distributed_environment` end to end with gloo, one rank and two:
  the store listens on loopback only, collectives work, and every other init method reaches
  torch.distributed.init_process_group unchanged.

Run the second group with that venv's Python, from the repository root:

    CUDA_VISIBLE_DEVICES="" "$VBIN/python" -m pytest -q \
        inference/gravityocr/test_gravityocr_dist_init.py

(`$VBIN` as printed by setup.sh; pytest must be importable by that Python.)
"""

import ast
import gc
import importlib
import inspect
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import textwrap
import threading
import types
import urllib.parse
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pytest

HERE = Path(__file__).resolve().parent
PATCH = HERE / "sglang" / "gravityocr_selfspec.patch"
PARALLEL_STATE = "python/sglang/srt/distributed/parallel_state.py"
MODEL_RUNNER = "python/sglang/srt/model_executor/model_runner.py"
TIMEOUT = timedelta(seconds=30)

needs_proc_net = pytest.mark.skipif(
    not Path("/proc/net/tcp").exists(), reason="reads listening sockets from /proc/net (Linux)"
)


# ----------------------------------------------------------------------------------- helpers
def _listeners(port):
    """Local addresses of the TCP sockets in LISTEN state on `port` (this network namespace)."""
    found = set()
    for family, path in ((socket.AF_INET, "/proc/net/tcp"), (socket.AF_INET6, "/proc/net/tcp6")):
        try:
            rows = Path(path).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            addr, hexport = fields[1].split(":")
            if fields[3] != "0A" or int(hexport, 16) != port:  # 0A = TCP_LISTEN
                continue
            raw = bytes.fromhex(addr)  # 32-bit words in host (little-endian) order
            raw = b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4))
            found.add(socket.inet_ntop(family, raw))
    return found


def _free_port():
    """A port that is free on every interface (so it is free on loopback too)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _have_ipv6_loopback():
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


def _patch_runs(path):
    """The runs of consecutive added lines the patch makes to `path` (one list of lines per run)."""
    runs, cur, in_file, in_hunk = [], [], False, False
    for line in PATCH.read_text(encoding="utf-8").splitlines():
        if line.startswith("diff --git "):
            in_file, in_hunk = line.split()[2][2:] == path, False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_file and in_hunk and line.startswith("+"):
            cur.append(line[1:])
            continue
        if cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _added_text(path):
    return "\n".join("\n".join(run) for run in _patch_runs(path))


def _patch_function(path, name):
    """Function `name` exactly as the patch adds it to `path` (one run of added lines)."""
    for run in _patch_runs(path):
        source = "\n".join(run)
        if re.search(rf"^def {name}\(", source, flags=re.M):
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
            assert node.name == name
            return ast.get_source_segment(source, node)
    raise AssertionError(f"the patch adds no function {name} to {path}")


@pytest.fixture(scope="module")
def store_fn():
    """The patch's _loopback_rendezvous_store, compiled with parallel_state.py's names."""
    torch = pytest.importorskip("torch")
    import torch.distributed  # noqa: F401

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is not available")
    import urllib  # the module imports urllib.parse, which binds `urllib`

    g = {
        "ipaddress": ipaddress,
        "os": os,
        "socket": socket,
        "urllib": urllib,
        "torch": torch,
        "Backend": torch.distributed.Backend,
        "Optional": Optional,
        "timedelta": timedelta,
    }
    src = _patch_function(PARALLEL_STATE, "_loopback_rendezvous_store")
    exec(compile(src, "<patch: parallel_state._loopback_rendezvous_store>", "exec"), g)
    return g["_loopback_rendezvous_store"]


# ======================================================== the patch text + CPU torch (CI)
def test_patch_rendezvouses_a_single_node_server_on_loopback():
    added = _added_text(MODEL_RUNNER)
    fn = _patch_function(MODEL_RUNNER, "_torch_dist_init_method")
    # upstream precedence first, then loopback for one node; --host only for the multi-node fallback
    order = [
        "SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE",
        "server_args.dist_init_addr",
        "if server_args.nnodes == 1:",
        'return NetworkAddress("127.0.0.1", dist_port).to_tcp()',
        'return NetworkAddress(server_args.host or "127.0.0.1", dist_port).to_tcp()',
    ]
    pos = [fn.find(s) for s in order]
    assert -1 not in pos and pos == sorted(pos), dict(zip(order, pos))
    assert "dist_init_method = _torch_dist_init_method(self.server_args, self.dist_port)" in added


def test_patch_gives_a_loopback_rendezvous_its_own_store():
    added = _added_text(PARALLEL_STATE)
    assert "store = _loopback_rendezvous_store(" in added
    assert "init_method=distributed_init_method if store is None else None," in added
    assert "store=store," in added


@needs_proc_net
@pytest.mark.parametrize("use_libuv", ["1", "0"], ids=["libuv", "legacy"])
def test_patch_store_listens_on_loopback_only(store_fn, monkeypatch, use_libuv):
    monkeypatch.setenv("USE_LIBUV", use_libuv)
    port = _free_port()
    store = store_fn(f"tcp://127.0.0.1:{port}", 0, 1, "gloo", TIMEOUT)
    assert type(store).__name__ == "PrefixStore"
    assert _listeners(port) == {"127.0.0.1"}  # torch alone: every interface ("0.0.0.0" / "::")
    store.set("k", "v")
    assert store.get("k") == b"v"
    tcp = store.underlying_store
    assert type(tcp).__name__ == "TCPStore" and (tcp.host, tcp.port) == ("127.0.0.1", port)
    assert tcp.libuvBackend == (use_libuv == "1")
    del store, tcp
    gc.collect()
    assert _listeners(port) == set(), "the store's port must close with the store"


@needs_proc_net
@pytest.mark.skipif(not _have_ipv6_loopback(), reason="no IPv6 loopback")
def test_patch_store_listens_on_ipv6_loopback_only(store_fn):
    port = _free_port()
    store = store_fn(f"tcp://[::1]:{port}", 0, 1, "gloo", TIMEOUT)
    assert _listeners(port) == {"::1"}
    store.set("k", "v")
    assert store.get("k") == b"v"


@needs_proc_net
def test_patch_store_serves_two_ranks(store_fn):
    port = _free_port()
    url = f"tcp://127.0.0.1:{port}"
    master = []
    t = threading.Thread(target=lambda: master.append(store_fn(url, 0, 2, "gloo", TIMEOUT)))
    t.start()  # rank 0 waits in the constructor until rank 1 has connected
    worker = store_fn(url, 1, 2, "gloo", TIMEOUT)
    t.join(timeout=60)
    assert master, "rank 0's store did not come up"
    worker.set("from-rank-1", "x")
    assert master[0].get("from-rank-1") == b"x"
    master[0].set("from-rank-0", "y")
    assert worker.get("from-rank-0") == b"y"
    assert _listeners(port) == {"127.0.0.1"}


@pytest.mark.parametrize(
    "method",
    [
        "env://",
        "file:///tmp/gravityocr-rendezvous",
        "tcp://10.1.2.3:{p}",
        "tcp://0.0.0.0:{p}",
        "tcp://[::]:{p}",
        "tcp://localhost:{p}",
        "tcp://127.0.0.1:{p}?rank=0&world_size=1",
        "tcp://127.0.0.1",
        "tcp://127.0.0.1:{p}/x",
    ],
)
def test_patch_store_leaves_other_init_methods_to_torch(store_fn, method):
    port = _free_port()
    assert store_fn(method.format(p=port), 0, 1, "gloo", TIMEOUT) is None
    assert store_fn(method.format(p=port), 0, 1, "nccl", None) is None


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_patch_store_takes_the_backends_default_timeout(store_fn, backend):
    torch = pytest.importorskip("torch")
    c10d = torch.distributed.distributed_c10d
    store = store_fn(f"tcp://127.0.0.1:{_free_port()}", 0, 1, backend, None)
    expected = c10d._get_default_timeout(torch.distributed.Backend(backend))
    assert store.timeout == expected and store.underlying_store.timeout == expected


def test_patch_store_leaves_unknown_ranks_and_torchelastic_to_torch(store_fn, monkeypatch):
    port = _free_port()
    url = f"tcp://127.0.0.1:{port}"
    assert store_fn(url, -1, 1, "gloo", TIMEOUT) is None
    assert store_fn(url, 0, -1, "gloo", TIMEOUT) is None
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    assert store_fn(url, 0, 1, "gloo", TIMEOUT) is None


def test_patch_store_refuses_a_port_in_use(store_fn):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        with pytest.raises(OSError):
            store_fn(f"tcp://127.0.0.1:{s.getsockname()[1]}", 0, 1, "gloo", TIMEOUT)


# ===================================================== the patched server (SGLang venv)
def _importable(name):
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


HAVE_SGLANG = _importable("sglang.srt.distributed.parallel_state") and _importable(
    "sglang.srt.model_executor.model_runner"
)
needs_sglang = pytest.mark.skipif(
    not HAVE_SGLANG, reason="SGLang (the patched server venv) is not installed"
)


def _init_method(monkeypatch, override=None, **args):
    mr = importlib.import_module("sglang.srt.model_executor.model_runner")
    fn = getattr(mr, "_torch_dist_init_method", None)
    assert fn is not None, "model_runner has no _torch_dist_init_method (unpatched build)"
    if override is None:
        monkeypatch.delenv("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE", override)
    server_args = types.SimpleNamespace(
        **{"host": "127.0.0.1", "nnodes": 1, "dist_init_addr": None, **args}
    )
    return fn(server_args, 4321)


@needs_sglang
@pytest.mark.parametrize(
    "host", ["127.0.0.1", "10.1.2.3", "0.0.0.0", "::", "fd00::7", "gpu-host.example", None]
)
def test_single_node_rendezvous_is_loopback_whatever_the_host(monkeypatch, host):
    assert _init_method(monkeypatch, host=host) == "tcp://127.0.0.1:4321"


@needs_sglang
def test_override_and_dist_init_addr_keep_upstream_precedence(monkeypatch):
    assert _init_method(monkeypatch, override="env://", host="10.1.2.3") == "env://"
    assert (
        _init_method(monkeypatch, override="tcp://10.9.9.9:1", dist_init_addr="10.0.0.5:5000")
        == "tcp://10.9.9.9:1"
    )
    assert (
        _init_method(monkeypatch, host="10.1.2.3", dist_init_addr="10.0.0.5:5000")
        == "tcp://10.0.0.5:5000"
    )
    assert _init_method(monkeypatch, dist_init_addr="[fd00::5]:5000") == "tcp://[fd00::5]:5000"
    # multi-node without --dist-init-addr: upstream's --host fallback, unchanged
    assert _init_method(monkeypatch, nnodes=2, host="10.1.2.3") == "tcp://10.1.2.3:4321"
    assert _init_method(monkeypatch, nnodes=2, host=None) == "tcp://127.0.0.1:4321"


_RANK_SCRIPT = textwrap.dedent(
    """
    import json, socket, sys, torch, torch.distributed as dist
    from pathlib import Path
    from sglang.srt.distributed import parallel_state as ps
    exec(sys.argv[4])  # _listeners
    rank, world, port = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ps.init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, backend="gloo",
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    t = torch.tensor([rank + 1.0])
    dist.all_reduce(t)
    # torch's layers: PrefixStore("0/", PrefixStore("default_pg", TCPStore))
    chain = [dist.distributed_c10d._get_default_store()]
    while hasattr(chain[-1], "underlying_store"):
        chain.append(chain[-1].underlying_store)
    tcp = chain[-1]
    cpu = ps.get_world_group().cpu_group
    dist.barrier(group=cpu)
    report = {
        "listeners": sorted(_listeners(port)),
        "sum": t.item(),
        "chain": [type(x).__name__ for x in chain],
        "tcp": [type(tcp).__name__, getattr(tcp, "host", None), getattr(tcp, "port", None)],
        "world": ps.get_world_group().world_size,
    }
    print(json.dumps(report), flush=True)
    dist.barrier(group=cpu)  # rank 0 keeps the store up until every rank has reported
    """
)


def _run_ranks(world):
    port = _free_port()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _RANK_SCRIPT,
                str(r),
                str(world),
                str(port),
                inspect.getsource(_listeners),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for r in range(world)
    ]
    outs = [p.communicate(timeout=300) for p in procs]
    for p, (out, err) in zip(procs, outs):
        assert p.returncode == 0, err[-3000:]
    return port, [json.loads(out.strip().splitlines()[-1]) for out, _ in outs]


@needs_sglang
@needs_proc_net
@pytest.mark.parametrize("world", [1, 2])
def test_init_distributed_environment_store_listens_on_loopback_only(world):
    port, reports = _run_ranks(world)
    assert [r["sum"] for r in reports] == [world * (world + 1) / 2] * world
    assert all(r["world"] == world for r in reports)
    assert reports[0]["listeners"] == ["127.0.0.1"], reports[0]  # upstream: every interface
    for r in reports:
        # the same layers as torch's own rendezvous: the group's prefix, "default_pg", the TCPStore
        assert r["chain"] == ["PrefixStore", "PrefixStore", "TCPStore"], r
        assert r["tcp"] == ["TCPStore", "127.0.0.1", port], r


@needs_sglang
@pytest.mark.parametrize(
    "method",
    [
        "env://",
        "tcp://10.1.2.3:{p}",
        "tcp://0.0.0.0:{p}",
        "tcp://localhost:{p}",
        "file:///tmp/gravityocr-rdzv",
        "tcp://127.0.0.1:{p}",
    ],
)
def test_init_distributed_environment_gives_torch_the_store_or_the_method(monkeypatch, method):
    import torch.distributed as dist

    ps = importlib.import_module("sglang.srt.distributed.parallel_state")
    seen = {}

    class Stop(Exception):
        pass

    def fake_init_process_group(**kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", fake_init_process_group)
    url = method.format(p=_free_port())
    with pytest.raises(Stop):
        ps.init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="gloo", distributed_init_method=url
        )
    if urllib.parse.urlsplit(url).hostname == "127.0.0.1":
        assert seen["init_method"] is None and type(seen["store"]).__name__ == "PrefixStore"
    else:
        assert seen["init_method"] == url and seen.get("store") is None
    assert (seen["backend"], seen["world_size"], seen["rank"]) == ("gloo", 1, 0)
