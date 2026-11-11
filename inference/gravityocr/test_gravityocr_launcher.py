# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""serve.sh dry runs: CPU only, no GPU, no network, no SGLang install.

Each test runs the launcher with bash against a stub VBIN whose `python` prints the argv and
environment it was exec'd with (and the --config file, if any) instead of starting a server. PATH is
reduced to a few coreutils so a real nvidia-smi / nvcc on the machine never leaks in.
"""

import json
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "serve.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="serve.sh needs bash")

COREUTILS = ("sed", "mktemp", "sleep", "rm", "od")
STUB_PYTHON = """#!/bin/sh
exec "$STUB_PY" -I -c '
import json, os, stat, sys
argv = sys.argv[1:]
if argv[:1] == ["-c"] and "snapshot_download" in argv[1]:
    with open(os.environ["STUB_DL_LOG"], "w") as f:
        json.dump({"repo": argv[2], "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}, f)
    print(os.environ["STUB_SNAPSHOT"])
    raise SystemExit(0)
rec = {"argv": argv, "env": dict(os.environ)}
if "--config" in argv:
    path = argv[argv.index("--config") + 1]
    with open(path) as f:
        rec["config"] = f.read()
    rec["config_path"] = path
    rec["config_mode"] = stat.S_IMODE(os.stat(path).st_mode)
    rec["config_dir_mode"] = stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode)
print(json.dumps(rec))
' "$@"
"""
NVCC_OUT = "Cuda compilation tools, release {rel}, V{rel}.85\\nBuild cuda_{rel}.r{rel}/compiler.0_0"
SELFSPEC_FAST_ENV = (
    "SELFSPEC_FASTATTN",
    "SELFSPEC_MROPE_FIX",
    "SELFSPEC_FAST_COMMIT",
    "SELFSPEC_HOSTLEN",
    "SELFSPEC_DEV_POS",
    "SELFSPEC_DRAFT_AHEAD",
)


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class Run:
    def __init__(self, proc):
        self.returncode, self.stdout, self.stderr = proc.returncode, proc.stdout, proc.stderr
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        self.rec = json.loads(last) if last.startswith("{") else None

    @property
    def launched(self):
        return self.rec is not None

    @property
    def argv(self):
        return self.rec["argv"]

    @property
    def env(self):
        return self.rec["env"]

    def opt(self, flag):
        """Value of the last occurrence of `flag` (argparse: last one wins)."""
        idx = [i for i, a in enumerate(self.argv) if a == flag]
        assert idx, f"{flag} not in {self.argv}"
        return self.argv[idx[-1] + 1]

    @property
    def output(self):
        """What serve.sh itself printed (stdout without the stub's record line, then stderr)."""
        lines = self.stdout.splitlines()
        return "\n".join(lines[:-1] if self.launched else lines) + "\n" + self.stderr

    @property
    def keys(self):
        """The --config YAML as {key: value} (the launcher writes one 'key: 'value'' per line)."""
        out = {}
        for line in self.rec["config"].splitlines():
            key, _, value = line.partition(": ")
            assert value.startswith("'") and value.endswith("'"), line
            out[key] = value[1:-1].replace("''", "'")
        return out


@pytest.fixture
def launcher(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in COREUTILS:
        found = shutil.which(name)
        assert found, f"{name} is needed on PATH"
        (tools / name).symlink_to(found)
    vbin = tmp_path / "venv" / "bin"
    _script(vbin / "python", STUB_PYTHON)
    _script(vbin / "ninja", "#!/bin/sh\nexit 0\n")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (tmp_path / "home").mkdir()
    (tmp_path / "tmp").mkdir()

    def run(*kv, env=None, bd_json=False, preflight=False, cwd=None):
        if bd_json:
            (ckpt / "block_diffusion.json").write_text('{"bd_size": 32, "mask_id": 59282}')
        full_env = {
            "PATH": str(tools),
            "HOME": str(tmp_path / "home"),
            "TMPDIR": str(tmp_path / "tmp"),
            "STUB_PY": sys.executable,
            "STUB_DL_LOG": str(tmp_path / "download.json"),
            "VBIN": str(vbin),
            "CKPT": str(ckpt),
            "CUDA_HOME": str(tmp_path / "cuda"),
        }
        if not preflight:
            full_env["SKIP_PREFLIGHT"] = "1"
        for key, value in (env or {}).items():
            if value is None:
                full_env.pop(key, None)
            else:
                full_env[key] = value
        proc = subprocess.run(
            [BASH, str(LAUNCHER), *kv],
            env=full_env,
            cwd=cwd or tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return Run(proc)

    run.tmp, run.vbin, run.ckpt = tmp_path, vbin, ckpt
    return run


# ---------------------------------------------------------------- modes
def test_self_spec_is_the_default(launcher):
    r = launcher()
    assert r.returncode == 0, r.stderr
    assert r.argv[:2] == ["-m", "sglang.launch_server"]
    assert r.opt("--model-path") == str(launcher.ckpt)
    assert r.opt("--speculative-algorithm") == "SELFSPEC_DIFFUSION"
    assert r.opt("--speculative-num-draft-tokens") == "33"  # K = bd_size + 1
    assert r.opt("--page-size") == "1"
    assert r.opt("--attention-backend") == "flashinfer"
    assert r.opt("--host") == "127.0.0.1"
    assert r.opt("--port") == "30000"
    assert r.opt("--served-model-name") == "glm-ocr"
    assert r.opt("--context-length") == "16384"
    assert r.opt("--mem-fraction-static") == "0.6"
    assert r.opt("--tokenizer-worker-num") == "2"  # FAST=1 without API_KEY
    for absent in (
        "--dp-size",
        "--config",
        "--api-key",
        "--trust-remote-code",
        "--max-running-requests",
        "--cuda-graph-max-bs",
    ):
        assert absent not in r.argv
    env = r.env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    # the position layout GravityOCR's published results were produced with (0 = AR-aligned)
    assert env["SELFSPEC_POS_OFFSET"] == "1"
    assert env["SELFSPEC_FULLDRAFT"] == "1"
    assert env["SELFSPEC_MASK_ID"] == "59282"  # the checkpoint dir has no block_diffusion.json
    assert all(env[k] == "1" for k in SELFSPEC_FAST_ENV)
    assert env["SS_GLM_VIT_GRAPH_MAXM"] == "2048" and env["SS_GLM_PV_BF16"] == "1"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["TORCH_NUM_THREADS"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1" and env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["HF_HOME"] == str(launcher.tmp / "home" / ".cache" / "huggingface")
    assert env["PATH"].split(":")[:2] == [str(launcher.vbin), str(launcher.tmp / "cuda" / "bin")]
    assert "API_KEY" not in env
    # two tokenizer workers: SGLang enforces no keys, and the launcher says so
    assert "management endpoints are open" in r.stderr


def test_self_spec_turns_grammar_backends_off(launcher):
    # The self-spec worker builds no grammar vocab mask, so a token a grammar forbids raises inside the
    # scheduler and takes the whole server down. Until it does, self-spec serves no constrained decoding.
    assert launcher().opt("--grammar-backend") == "none"
    assert "--grammar-backend" not in launcher("MODE=causal").argv


@pytest.mark.parametrize("from_recipe_folder", [False, True])
def test_python_never_puts_the_cwd_on_sys_path(launcher, from_recipe_folder):
    # From inference/gravityocr/, `python -m sglang.launch_server` would import the recipe's own sglang/
    # folder (a namespace package, first on sys.path) instead of the installed SGLang.
    r = launcher(cwd=HERE if from_recipe_folder else None)
    assert r.returncode == 0, r.stderr
    assert r.env["PYTHONSAFEPATH"] == "1"


def test_echo_line(launcher):
    r = launcher()
    assert r.stdout.splitlines()[0] == (
        f"[serve] MODE=self-spec FAST=1 GPU=0 DP=1 SERVE_HOST=127.0.0.1 PORT=30000 "
        f"SERVED_NAME=glm-ocr CKPT={launcher.ckpt}"
    )


def test_mask_id_comes_from_block_diffusion_json(launcher):
    r = launcher(bd_json=True)
    assert r.returncode == 0, r.stderr
    assert "SELFSPEC_MASK_ID" not in r.env
    assert r.env["SELFSPEC_POS_OFFSET"] == "1" and r.env["SELFSPEC_FULLDRAFT"] == "1"


def test_explicit_mask_id_is_kept(launcher):
    r = launcher(env={"SELFSPEC_MASK_ID": "151329"})
    assert r.env["SELFSPEC_MASK_ID"] == "151329"


def test_causal_mode_has_no_self_spec_args_or_env(launcher):
    r = launcher("MODE=causal", env={"SELFSPEC_FASTATTN": "1", "SELFSPEC_POS_OFFSET": "1"})
    assert r.returncode == 0, r.stderr
    assert not [a for a in r.argv if a.startswith("--speculative")]
    assert "--page-size" not in r.argv
    assert not [k for k in r.env if k.startswith("SELFSPEC_")]  # inherited switches are dropped too
    assert r.opt("--tokenizer-worker-num") == "2" and r.env["SS_GLM_VIT_GRAPH_MAXM"] == "2048"
    assert r.stdout.startswith("[serve] MODE=causal ")


def test_fast0_keeps_only_the_required_self_spec_env(launcher):
    r = launcher("FAST=0")
    assert r.opt("--speculative-algorithm") == "SELFSPEC_DIFFUSION"
    assert r.env["SELFSPEC_POS_OFFSET"] == "1" and r.env["SELFSPEC_FULLDRAFT"] == "1"
    assert not [k for k in SELFSPEC_FAST_ENV if k in r.env]
    assert r.opt("--tokenizer-worker-num") == "1"  # SGLang's own default, made explicit
    assert not [k for k in r.env if k.startswith("SS_GLM_")]


def test_causal_fast0_is_plain(launcher):
    r = launcher("MODE=causal", "FAST=0")
    assert r.opt("--tokenizer-worker-num") == "1" and "--speculative-algorithm" not in r.argv
    assert not [k for k in r.env if k.startswith(("SELFSPEC_", "SS_GLM_"))]


@pytest.mark.parametrize("mode", ["spec", "ar", "diffusion", "Self-Spec", "causal "])
def test_unknown_mode_exits_2(launcher, mode):
    r = launcher(env={"MODE": mode})
    assert r.returncode == 2 and not r.launched
    assert "MODE must be self-spec | causal" in r.stderr


@pytest.mark.parametrize("fast", ["2", "yes", "true"])
def test_bad_fast_exits_2(launcher, fast):
    r = launcher(f"FAST={fast}")
    assert r.returncode == 2 and not r.launched


def test_never_trust_remote_code(launcher):
    for mode in ("self-spec", "causal"):
        r = launcher(f"MODE={mode}")
        assert "--trust-remote-code" not in r.argv


# ---------------------------------------------------------------- knobs
def test_knob_overrides(launcher):
    r = launcher(
        "GPU=3",
        "PORT=30500",
        "SERVED_NAME=gravityocr",
        "CONTEXT=8192",
        "MEMFRAC=0.3",
        "DRAFT=17",
        "MAXREQ=64",
        "GRAPHBS=32",
        "TOKENIZER_WORKERS=4",
        "OMP_NUM_THREADS_SERVER=2",
    )
    assert r.returncode == 0, r.stderr
    assert r.env["CUDA_VISIBLE_DEVICES"] == "3"
    assert (r.opt("--port"), r.opt("--served-model-name"), r.opt("--context-length")) == (
        "30500",
        "gravityocr",
        "8192",
    )
    assert (r.opt("--mem-fraction-static"), r.opt("--speculative-num-draft-tokens")) == (
        "0.3",
        "17",
    )
    assert (r.opt("--max-running-requests"), r.opt("--cuda-graph-max-bs")) == ("64", "32")
    assert r.opt("--tokenizer-worker-num") == "4" and r.env["OMP_NUM_THREADS"] == "2"


def test_host_env_is_ignored(launcher):
    r = launcher(env={"HOST": "0.0.0.0"})  # some shells export HOST=<hostname>
    assert r.returncode == 0, r.stderr
    assert r.opt("--host") == "127.0.0.1"


def test_default_cuda_home(launcher):
    r = launcher(env={"CUDA_HOME": None})
    assert r.env["CUDA_HOME"] == "/usr/local/cuda-12.8"


def test_help_and_bad_arguments(launcher):
    r = launcher("--help")
    assert r.returncode == 0 and "SERVE_HOST" in r.stdout and "MODE" in r.stdout and not r.launched
    r = launcher("not-a-key-value")
    assert r.returncode == 2 and not r.launched


def test_missing_vbin_exits_2(launcher):
    assert launcher(env={"VBIN": None}).returncode == 2
    assert launcher(env={"VBIN": str(launcher.tmp / "nowhere")}).returncode == 2


# ---------------------------------------------------------------- checkpoint resolution
def test_gravityocr_model_env_is_the_fallback(launcher):
    other = launcher.tmp / "other"
    other.mkdir()
    r = launcher(env={"CKPT": None, "GRAVITYOCR_MODEL": str(other)})
    assert r.opt("--model-path") == str(other)
    r = launcher(env={"GRAVITYOCR_MODEL": str(other)})  # CKPT wins
    assert r.opt("--model-path") == str(launcher.ckpt)


@pytest.mark.parametrize("hub_env", [{"CKPT": None}, {"CKPT": "someorg/SomeOCR"}])
def test_hub_id_is_fetched_online_then_served_offline(launcher, hub_env):
    snap = launcher.tmp / "snapshot"
    snap.mkdir()
    (snap / "block_diffusion.json").write_text('{"bd_size": 32, "mask_id": 59282}')
    r = launcher(env={**hub_env, "STUB_SNAPSHOT": str(snap)})
    assert r.returncode == 0, r.stderr
    download = json.loads((launcher.tmp / "download.json").read_text())
    assert download["repo"] == (hub_env["CKPT"] or "trillionlabs/GravityOCR")
    assert download["HF_HUB_OFFLINE"] == "0"
    assert r.opt("--model-path") == str(snap)
    assert r.env["HF_HUB_OFFLINE"] == "1"
    assert "SELFSPEC_MASK_ID" not in r.env  # read from the snapshot's block_diffusion.json


def test_missing_local_checkpoint_path_exits_2(launcher):
    r = launcher(env={"CKPT": str(launcher.tmp / "missing")})
    assert r.returncode == 2 and not r.launched
    assert not (launcher.tmp / "download.json").exists()


# ---------------------------------------------------------------- data parallel
def test_data_parallel(launcher):
    r = launcher("DP=2", "GPU=0,1")
    assert r.returncode == 0, r.stderr
    assert r.opt("--dp-size") == "2" and r.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert " DP=2 " in r.stdout.splitlines()[0]


@pytest.mark.parametrize(
    "dp,gpu",
    [
        ("2", "0"),
        ("1", "0,1"),
        ("3", "0,1"),
        ("2", "0,,1"),
        ("2", "0, 1"),
        ("0", "0"),
        ("x", "0"),
        ("-1", "0"),
    ],
)
def test_data_parallel_gpu_mismatch_exits_2(launcher, dp, gpu):
    r = launcher(env={"DP": dp, "GPU": gpu})
    assert r.returncode == 2 and not r.launched, r.stderr


# ---------------------------------------------------------------- network exposure and keys
@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "::1"])
def test_loopback_needs_no_key(launcher, host):
    r = launcher(env={"SERVE_HOST": host})
    assert r.returncode == 0, r.stderr
    assert r.opt("--host") == host and "--config" not in r.argv


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.1.2.3", "gpu-box", "127.0.0.1.example.com"])
def test_non_loopback_without_api_key_is_refused(launcher, host):
    r = launcher(env={"SERVE_HOST": host})
    assert r.returncode != 0 and not r.launched
    assert "API_KEY" in r.stderr


def _assert_private_key_file(r):
    assert r.rec["config_mode"] == 0o600 and r.rec["config_dir_mode"] == 0o700
    # The watcher removes the key directory once the server process is gone.
    keydir = Path(r.rec["config_path"]).parent
    deadline = time.monotonic() + 20
    while keydir.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not keydir.exists()


@pytest.mark.skipif(
    BASH is not None and subprocess.run([BASH, "-c", "((BASH_VERSINFO[0] >= 4))"]).returncode != 0,
    reason="bash 3.2 (macOS /bin/bash) escapes the quote in the key differently; serve.sh targets Linux",
)
def test_api_key_reaches_sglang_through_a_private_config_file(launcher):
    key = "s3cr'et/Key-1#x"
    r = launcher(env={"SERVE_HOST": "0.0.0.0", "API_KEY": key})
    assert r.returncode == 0, r.stderr
    assert r.opt("--host") == "0.0.0.0"
    # SGLang v0.5.10 enforces keys only with one tokenizer worker, so FAST=1 drops to one
    assert r.opt("--tokenizer-worker-num") == "1"
    assert "--api-key" not in r.argv and not [a for a in r.argv if key in a]
    assert key not in r.output and "API_KEY" not in r.env and key not in json.dumps(r.env)
    assert r.argv[-2] == "--config" and r.argv.count("--config") == 1
    assert r.keys["api-key"] == key
    try:
        import yaml  # SGLang reads the file with yaml.safe_load
    except ImportError:
        yaml = None
    if yaml is not None:
        assert yaml.safe_load(r.rec["config"]) == r.keys
    assert "Authorization: Bearer" in r.stderr
    _assert_private_key_file(r)


def test_api_key_on_loopback_is_also_honoured(launcher):
    r = launcher(env={"API_KEY": "abc123"})
    assert r.returncode == 0, r.stderr
    assert r.keys["api-key"] == "abc123" and "abc123" not in " ".join(r.argv)
    assert r.opt("--tokenizer-worker-num") == "1"


@pytest.mark.parametrize("fast", ["0", "1"])
def test_api_key_always_means_one_tokenizer_worker(launcher, fast):
    r = launcher(f"FAST={fast}", env={"API_KEY": "abc123"})
    assert r.returncode == 0, r.stderr
    nums = [r.argv[i + 1] for i, a in enumerate(r.argv) if a == "--tokenizer-worker-num"]
    assert nums == ["1"]


@pytest.mark.parametrize("workers", ["2", "4"])
def test_api_key_with_more_tokenizer_workers_is_refused(launcher, workers):
    r = launcher(f"TOKENIZER_WORKERS={workers}", env={"API_KEY": "abc123"})
    assert r.returncode == 2 and not r.launched
    assert "single tokenizer worker" in r.stderr


def test_one_tokenizer_worker_locks_the_management_endpoints(launcher):
    # With one worker SGLang enforces keys, so the launcher sets a random admin key it never shows:
    # /update_weights_from_disk, /flush_cache, profiling, LoRA loading etc. refuse every caller.
    runs = [launcher("FAST=0"), launcher("TOKENIZER_WORKERS=1"), launcher(env={"API_KEY": "abc123"})]
    admins = []
    for r in runs:
        assert r.returncode == 0, r.stderr
        assert r.opt("--tokenizer-worker-num") == "1"
        admin = r.keys["admin-api-key"]
        assert len(admin) == 64 and set(admin) <= set("0123456789abcdef")
        assert admin not in r.output and not [a for a in r.argv if admin in a]
        assert admin not in json.dumps(r.env)
        _assert_private_key_file(r)
        admins.append(admin)
    assert len(set(admins)) == len(admins)  # fresh per start
    assert "api-key" not in runs[0].keys  # no API_KEY: inference endpoints stay open on loopback


def test_api_key_as_an_argument_is_refused(launcher):
    r = launcher("API_KEY=abc123", env={"SERVE_HOST": "0.0.0.0"})
    assert r.returncode == 2 and not r.launched


@pytest.mark.parametrize("key", ["has space", "-starts-with-dash", "tab\tkey", "klüssel"])
def test_malformed_api_key_is_refused(launcher, key):
    r = launcher(env={"SERVE_HOST": "0.0.0.0", "API_KEY": key})
    assert r.returncode == 2 and not r.launched


def test_pythonoptimize_is_cleared(launcher):
    r = launcher(env={"PYTHONOPTIMIZE": "1"})  # would strip SGLang's assert statements
    assert r.returncode == 0, r.stderr
    assert "PYTHONOPTIMIZE" not in r.env


# ---------------------------------------------------------------- launcher values
@pytest.mark.parametrize(
    "kv",
    [
        "PORT=abc",
        "PORT=0",
        "PORT=70000",
        "PORT=--config",
        "CONTEXT=16k",
        "CONTEXT=-1",
        "DRAFT=x",
        "MAXREQ=--config",
        "GRAPHBS=1.5",
        "TOKENIZER_WORKERS=--config",
        "TOKENIZER_WORKERS=0",
        "OMP_NUM_THREADS_SERVER=all",
        "MEMFRAC=lots",
        "MEMFRAC=--config",
        "SERVED_NAME=--config",
        "SERVED_NAME=a b",
        "SERVE_HOST=--config",
        "CKPT=--config",
        "GPU=-1",
    ],
)
def test_malformed_launcher_values_are_refused(launcher, kv):
    r = launcher(kv, env={"API_KEY": "abc123"} if kv.startswith("SERVE_HOST") else None)
    assert r.returncode == 2 and not r.launched, r.stdout


def test_config_cannot_be_smuggled_through_launcher_values(launcher):
    # SGLang merges a YAML file wherever the literal token '--config' appears in its argv.
    evil = launcher.tmp / "evil.yaml"
    evil.write_text("enable-custom-logit-processor: true\n")
    r = launcher("TOKENIZER_WORKERS=--config", env={"EXTRA_ARGS": f"{evil} 2"})
    assert r.returncode == 2 and not r.launched


# ---------------------------------------------------------------- EXTRA_ARGS allowlist
FORBIDDEN_EXTRA = [
    # keys, bind address and config files
    "--config other.yaml",
    "--config=other.yaml",
    "--conf=other.yaml",
    "--con x.yaml",
    "--host 0.0.0.0",
    "--host=0.0.0.0",
    "--ho 0.0.0.0",
    "--ho=0.0.0.0",
    "--h=0.0.0.0",
    "--api-key k",
    "--api-key=k",
    "--api k",
    "--ap=k",
    "--admin-api-key k",
    "--admin-api-key=k",
    "--ad k",
    # deserialization sinks, LoRA, code and weight loading
    "--enable-custom-logit-processor",
    "--enable-custom-logit-processor=1",
    "--enable-custom-logit",
    "--enable-cus",
    "--enable-lora",
    "--enable-lora=true",
    "--enable-lora-overlap-loading",
    "--lora-paths a=/x",
    "--lora-paths=a=/x",
    "--lora-p a=/x",
    "--lora-backend triton",
    "--lora-target-modules all",
    "--lora",
    "--trust-remote-code",
    "--load-format pt",
    "--custom-weight-loader x.y",
    "--grammar-backend outlines",
    # other front-ends and transports, which ignore the API key
    "--grpc-mode",
    "--encoder-only",
    "--use-ray",
    "--enable-dp-attention",
    "--dist-init-addr 0.0.0.0:5000",
    "--nccl-init-addr 0.0.0.0:5000",
    "--nnodes 2",
    "--node-rank 1",
    "--kv-events-config {}",
    # what the launcher sets from its own variables
    "--tokenizer-worker-num 4",
    "--context-length 12288",
    "--port 1",
    "--page-size 16",
    "--speculative-algorithm NGRAM",
    # abbreviations of allowed options, and other '-' tokens
    "--log warning",
    "--disable-radix",
    "--",
    "-h",
    "--log-level info --host=0.0.0.0",
    "--log-level info\n--enable-lora",
]


@pytest.mark.parametrize("extra", FORBIDDEN_EXTRA)
def test_extra_args_outside_the_allowlist_are_refused(launcher, extra):
    r = launcher(env={"EXTRA_ARGS": extra})
    assert r.returncode == 2 and not r.launched, r.stdout
    assert "EXTRA_ARGS may not contain" in r.stderr


def test_refused_launcher_flags_name_their_variable(launcher):
    r = launcher(env={"EXTRA_ARGS": "--context-length 12288"})
    assert "(set CONTEXT instead)" in r.stderr


def test_allowed_extra_args_are_appended_last(launcher):
    (launcher.tmp / "decoy.txt").write_text("")
    extra = "--log-level warning --enable-metrics --disable-radix-cache --chunked-prefill-size=4096 *"
    tokens = extra.split()
    r = launcher(env={"EXTRA_ARGS": extra})
    assert r.returncode == 0, r.stderr
    assert (
        r.argv[-len(tokens) :] == tokens
    )  # in order, after the launcher's own flags, no glob expansion
    assert r.argv.index("--tokenizer-worker-num") < len(r.argv) - len(tokens)


def test_extra_args_come_before_the_launchers_config(launcher):
    r = launcher("FAST=0", env={"EXTRA_ARGS": "--log-level warning"})
    assert r.argv[-4:-2] == ["--log-level", "warning"] and r.argv[-2] == "--config"


# ---------------------------------------------------------------- preflight
def _preflight_env(launcher, release="12.8", cap="9.0", smi=True):
    cuda_bin = launcher.tmp / "cuda" / "bin"
    if release is not None:
        _script(cuda_bin / "nvcc", f'#!/bin/sh\nprintf "{NVCC_OUT.format(rel=release)}\\n"\n')
    env = {}
    if smi:
        log = launcher.tmp / "smi.log"
        _script(
            launcher.tmp / "smi" / "nvidia-smi", f'#!/bin/sh\necho "$*" >> "{log}"\necho "{cap}"\n'
        )
        env["PATH"] = f"{launcher.tmp / 'smi'}:{launcher.tmp / 'tools'}"
    return env


def test_preflight_passes_on_hopper_with_cuda12(launcher):
    r = launcher("DP=2", "GPU=0,1", env=_preflight_env(launcher), preflight=True)
    assert r.returncode == 0, r.stderr
    calls = (launcher.tmp / "smi.log").read_text().splitlines()
    assert [c.split()[:2] for c in calls] == [["-i", "0"], ["-i", "1"]]
    assert all("--query-gpu=compute_cap" in c for c in calls)


@pytest.mark.parametrize(
    "kw,needle",
    [
        ({"release": "13.0"}, "CUDA 13.0"),
        ({"release": "11.8"}, "CUDA 11.8"),
        ({"release": None}, "no nvcc"),
        ({"cap": "8.0"}, "compute capability '8.0'"),
        ({"cap": "10.0"}, "compute capability '10.0'"),
        ({"smi": False}, "nvidia-smi not found"),
    ],
)
def test_preflight_failures_exit_1(launcher, kw, needle):
    r = launcher(env=_preflight_env(launcher, **kw), preflight=True)
    assert r.returncode == 1 and not r.launched
    assert needle in r.stderr


def test_preflight_needs_ninja_in_vbin(launcher):
    (launcher.vbin / "ninja").unlink()
    r = launcher(env=_preflight_env(launcher), preflight=True)
    assert r.returncode == 1 and "no ninja" in r.stderr
    assert launcher(env=_preflight_env(launcher)).returncode == 0  # SKIP_PREFLIGHT=1 bypasses it


def test_launcher_is_valid_bash():
    subprocess.run([BASH, "-n", str(LAUNCHER)], check=True)
