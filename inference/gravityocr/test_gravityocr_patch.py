# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""GravityOCR SGLang recipe: CPU-only consistency checks (stdlib + pytest; no GPU, no network).

The SGLang files are parsed, never imported. Nothing here applies the patch or runs setup.sh;
sglang/setup.sh does that against the upstream checkout pinned in sglang/BASE_COMMIT.txt. The checks
below cover what that script relies on: its syntax, the lock it installs and how it installs it.
"""

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SGLANG_DIR = HERE / "sglang"
BASE_COMMIT_FILE = SGLANG_DIR / "BASE_COMMIT.txt"
PATCH = SGLANG_DIR / "gravityocr_selfspec.patch"
SETUP = SGLANG_DIR / "setup.sh"
LOCK = SGLANG_DIR / "requirements-lock.txt"
REQUIREMENTS_IN = SGLANG_DIR / "requirements.in"
PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(\S+)", re.M)
# The four new SGLang files -> whether part of the file is adapted from ngram_info.py
# (its header then says so).
NEW_FILES = {
    "python/sglang/srt/multimodal/glm_vit_graph_runner.py": False,
    "python/sglang/srt/speculative/selfspec_diffusion_worker.py": True,
    "python/sglang/srt/speculative/selfspec_fast_attn.py": False,
    "python/sglang/srt/speculative/selfspec_fast_commit.py": True,
}
# Upstream files the patch modifies (in patch order). The four new files above ship as whole files.
PATCH_TARGETS = [
    "python/pyproject.toml",
    "python/sglang/srt/dllm/config.py",
    "python/sglang/srt/dllm/mixin/req.py",
    "python/sglang/srt/dllm/mixin/scheduler.py",
    "python/sglang/srt/entrypoints/http_server.py",
    "python/sglang/srt/layers/attention/flashinfer_backend.py",
    "python/sglang/srt/layers/attention/vision.py",
    "python/sglang/srt/managers/schedule_batch.py",
    "python/sglang/srt/managers/schedule_policy.py",
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/scheduler_output_processor_mixin.py",
    "python/sglang/srt/managers/scheduler_update_weights_mixin.py",
    "python/sglang/srt/managers/tokenizer_manager.py",
    "python/sglang/srt/managers/utils.py",
    "python/sglang/srt/mem_cache/radix_cache.py",
    "python/sglang/srt/model_executor/cuda_graph_runner.py",
    "python/sglang/srt/model_executor/forward_batch_info.py",
    "python/sglang/srt/model_executor/model_runner.py",
    "python/sglang/srt/models/glm4v.py",
    "python/sglang/srt/models/glm_ocr.py",
    "python/sglang/srt/multimodal/processors/base_processor.py",
    "python/sglang/srt/multimodal/processors/glm4v.py",
    "python/sglang/srt/server_args.py",
    "python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py",
    "python/sglang/srt/speculative/ngram_info.py",
    "python/sglang/srt/speculative/spec_info.py",
]
LICENSE_HEADER = ["# Copyright 2026 Trillion Labs", "# SPDX-License-Identifier: Apache-2.0"]
# A quoted git SHA: 7-40 hex digits with at least one letter and one digit
# (so dates and plain words do not match).
SHA_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b")
SEMVER_RE = re.compile(r"\bv?(\d+\.\d+\.\d+)\b")


def _base_commit() -> str:
    return BASE_COMMIT_FILE.read_text(encoding="utf-8").split()[0]


def _base_tag_version():
    """'X.Y.Z' from the 'tag vX.Y.Z' note in BASE_COMMIT.txt, or None when it records no tag."""
    m = re.search(r"\btag v(\d+\.\d+\.\d+)\b", BASE_COMMIT_FILE.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _header(path: Path) -> list:
    """Leading comment lines of a Python file (the licence / provenance block)."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return lines


def _pin_sources():
    """(label, [lines to scan], whole_file) for every place that quotes the SGLang base commit."""
    sources = []
    for rel, whole in (
        ("sglang/BASE_COMMIT.txt", True),
        ("sglang/setup.sh", True),
        ("README.md", False),
        ("NOTICE", False),
    ):
        path = HERE / rel
        if path.is_file():
            sources.append((rel, path.read_text(encoding="utf-8").splitlines(), whole))
    for rel in NEW_FILES:
        sources.append((f"sglang/{rel} (header)", _header(SGLANG_DIR / rel), True))
    return sources


# ---------------------------------------------------------------- base commit
def test_base_commit_is_a_full_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", _base_commit())
    assert len(SHA_RE.findall(BASE_COMMIT_FILE.read_text(encoding="utf-8"))) == 1


def test_every_quoted_sglang_pin_is_the_base_commit():
    base = _base_commit()
    for label, lines, whole_file in _pin_sources():
        for n, line in enumerate(lines, 1):
            if not whole_file and "sglang" not in line.lower():
                continue
            for sha in SHA_RE.findall(line):
                assert base.startswith(sha), f"{label}:{n} quotes {sha}, but BASE_COMMIT is {base}"


def test_readme_and_notice_quote_the_base_commit():
    short = _base_commit()[:7]
    for rel in ("README.md", "NOTICE"):
        path = HERE / rel
        if path.is_file():
            assert short in path.read_text(encoding="utf-8"), (
                f"{rel} never names the SGLang base {short}"
            )


def test_quoted_sglang_versions_match_the_base_tag():
    tag = _base_tag_version()
    if tag is None:
        pytest.skip("BASE_COMMIT.txt records no tag")
    series = tag.rsplit(".", 1)[0] + "."
    for label, lines, _ in _pin_sources():
        for n, line in enumerate(lines, 1):
            if "sglang" not in line.lower():
                continue
            for version in SEMVER_RE.findall(line):
                if version.startswith(series):
                    assert version == tag, (
                        f"{label}:{n} names SGLang {version}, but the base is tag v{tag}"
                    )


# ---------------------------------------------------------------- patch
def test_patch_targets_exactly_the_expected_upstream_files():
    text = PATCH.read_text(encoding="utf-8")
    pairs = re.findall(r"^diff --git a/(\S+) b/(\S+)$", text, flags=re.M)
    assert all(a == b for a, b in pairs), "a patch entry renames a file"
    assert [a for a, _ in pairs] == PATCH_TARGETS
    assert re.findall(r"^--- a/(\S+)$", text, flags=re.M) == PATCH_TARGETS
    assert re.findall(r"^\+\+\+ b/(\S+)$", text, flags=re.M) == PATCH_TARGETS
    for marker in ("new file mode", "deleted file mode", "rename from", "GIT binary patch"):
        assert marker not in text, f"unexpected '{marker}' in the patch"
    assert not set(NEW_FILES) & set(PATCH_TARGETS)


def test_server_info_never_returns_the_keys():
    # Upstream /server_info returns dataclasses.asdict(server_args), keys included, at the NORMAL auth
    # level: any API-key holder (or any local client, without a key) could read the admin key. The patch
    # redacts both. This is the one hunk not in the GravityOCR release (see NOTICE).
    text = PATCH.read_text(encoding="utf-8")
    start = text.index("diff --git a/python/sglang/srt/entrypoints/http_server.py")
    hunk = text[start:text.index("\ndiff --git ", start + 1)]
    added = "\n".join(line[1:] for line in hunk.splitlines() if line.startswith("+") and not line.startswith("+++"))
    assert 'for key in ("api_key", "admin_api_key"):' in added and '"<redacted>"' in added
    assert "+        **server_args," in hunk and "-        **dataclasses.asdict(" in hunk


def test_a_vit_graph_capture_failure_falls_back_to_eager():
    # Capture runs inside load_weights, on by default with FAST=1: an OOM in any capture used to stop the
    # server from starting. The eager forward is always correct, so a failure must only switch graphs off.
    text = PATCH.read_text(encoding="utf-8")
    start = text.index("diff --git a/python/sglang/srt/models/glm_ocr.py")
    hunk = text[start:text.index("\ndiff --git ", start + 1)]
    body = hunk[hunk.index("+        runner = GlmVitGraphRunner("):]
    assert "+        try:\n+            runner.capture()\n+        except Exception" in body
    assert "capture failed" in body and "+            return" in body


# ---------------------------------------------------------------- the four new SGLang files
@pytest.mark.parametrize("rel", sorted(NEW_FILES))
def test_new_file_parses_on_py310_and_carries_the_header(rel):
    path = SGLANG_DIR / rel
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path), feature_version=(3, 10))
    assert source.splitlines()[:2] == LICENSE_HEADER
    header = "\n".join(_header(path))
    assert _base_commit() in header, "the header must name the full SGLang base commit"
    assert ("ngram_info.py" in header) == NEW_FILES[rel]
    # The header is comments only, so the module docstring survives.
    assert ast.get_docstring(tree)


def test_no_other_files_under_the_sglang_tree():
    shipped = {
        p.relative_to(SGLANG_DIR).as_posix()
        for p in (SGLANG_DIR / "python").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert shipped == set(NEW_FILES)


# ---------------------------------------------------------------- setup.sh and the lock
def test_setup_never_puts_the_cwd_on_sys_path():
    # From inference/gravityocr/, `python -` would find the recipe's own sglang/ folder before the
    # installed SGLang, and the verify step would fail with a misleading "env does not match".
    text = SETUP.read_text()
    first_python = text.index('"$VENV/bin/python"')
    assert "export PYTHONSAFEPATH=1" in text[:first_python]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(path: Path) -> dict:
    return {_norm(n): v for n, v in PIN_RE.findall(path.read_text(encoding="utf-8"))}


def _setup_commands(word: str) -> list:
    """setup.sh's logical lines (backslash continuations joined) that contain `word`."""
    text = SETUP.read_text(encoding="utf-8").replace("\\\n", " ")
    return [line.strip() for line in text.splitlines() if word in line and not line.lstrip().startswith("#")]


def test_setup_sh_is_valid_bash():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("needs bash")
    subprocess.run([bash, "-n", str(SETUP)], check=True)


def test_lock_and_setup_sh_name_the_base_tag():
    tag = _base_tag_version()
    assert tag, "BASE_COMMIT.txt must record the upstream tag"
    assert _pins(LOCK)["sglang"] == tag
    assert _pins(REQUIREMENTS_IN)["sglang"] == tag
    m = re.search(r"^SGLANG_VERSION=(\S+)$", SETUP.read_text(encoding="utf-8"), flags=re.M)
    assert m and m.group(1) == tag


def test_lock_has_what_setup_sh_relies_on():
    pins = _pins(LOCK)
    # setup.sh installs exactly these from the PyTorch cu129 index, everything else from PyPI
    cu129 = sorted(n for n, v in pins.items() if v.endswith("+cu129"))
    assert cu129 == ["torchao", "torchaudio", "torchcodec", "torchvision"]
    assert not [n for n, v in pins.items() if "+" in v and not v.endswith("+cu129")]
    # the versions setup.sh's final check asserts after the install
    checked = dict(re.findall(r'\("([a-z][a-z-]*)", "([0-9][^"]*)"\)', SETUP.read_text(encoding="utf-8")))
    assert checked == {"torch": "2.9.1", "flashinfer-python": "0.6.7.post2", "transformers": "5.3.0"}
    for name, version in checked.items():
        assert pins[name] == version, f"lock pins {name}=={pins[name]}, setup.sh checks {version}"
    assert "ninja" in pins  # FlashInfer's JIT and serve.sh's preflight need it in VBIN
    for name in _pins(REQUIREMENTS_IN):
        assert name in pins, f"requirements.in pins {name}, the lock does not"


def test_setup_sh_installs_the_lock_without_resolving_and_from_fixed_indexes():
    installs = _setup_commands("uv pip install")
    assert len(installs) == 3, installs  # PyPI pins, +cu129 pins, the patched SGLang
    for cmd in installs:
        for flag in ("--no-deps", "--no-config", "--default-index"):
            assert flag in cmd, f"{flag} missing from: {cmd}"
        for flag in ("--extra-index-url", "--index-url", "--find-links", "--index ", "-f "):
            assert flag not in cmd, f"{flag} in: {cmd}"
    lock_installs = [c for c in installs if "-r -" in c]
    assert len(lock_installs) == 2 and all("--no-build" in c for c in lock_installs)
    (unset,) = [line for line in _setup_commands("unset ") if "UV_EXTRA_INDEX_URL" in line]
    for var in ("UV_INDEX", "UV_EXTRA_INDEX_URL", "UV_FIND_LINKS", "UV_INDEX_URL", "UV_DEFAULT_INDEX"):
        assert f" {var} " in f"{unset} ", f"setup.sh does not unset {var}"
