# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""GravityOCR SGLang recipe: CPU-only consistency checks (stdlib + pytest; no GPU, no network).

The SGLang files are parsed, never imported. sglang/setup.sh applies the patch to the upstream
checkout pinned in sglang/BASE_COMMIT.txt; the checks below cover what that script relies on: its
syntax, the lock it installs and how it installs it, and (with git and bash) its steps 1-3 run for
real against a tiny synthetic upstream repository, with a stub uv that stops it before any install.
"""

import ast
import os
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
FREEZE = SGLANG_DIR / "validated-freeze.txt"
LOCK_HEADER_END = "# (MiDashengLM) at startup; OCR never uses torchaudio.\n"
PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(\S+)", re.M)
# The new SGLang files -> the SGLang v0.5.10 sources part of the file is adapted from (empty: none).
# A derived file's header carries SGLang's copyright notice, SGLANG_NOTICE verbatim (the line of
# SGLang's LICENSE, and of models/glm_ocr.py; ngram_info.py, penaltylib/, sampler.py and
# spec_utils.py have no per-file notice), under the Apache License 2.0, and names those sources.
SGLANG_NOTICE = "Copyright 2023-2024 SGLang Team"
NEW_FILES = {
    "python/sglang/srt/entrypoints/http_worker_keys.py": (),
    "python/sglang/srt/multimodal/glm_vit_graph_runner.py": ("sglang/srt/models/glm_ocr.py",),
    "python/sglang/srt/speculative/selfspec_commit_rules.py": ("sglang/srt/speculative/ngram_info.py",),
    "python/sglang/srt/speculative/selfspec_diffusion_worker.py": ("sglang/srt/speculative/ngram_info.py",),
    "python/sglang/srt/speculative/selfspec_fast_attn.py": (),
    "python/sglang/srt/speculative/selfspec_fast_commit.py": ("sglang/srt/speculative/ngram_info.py",),
    "python/sglang/srt/speculative/selfspec_sampling.py": (
        "sglang/srt/sampling/penaltylib/",
        "sglang/srt/layers/sampler.py",
        "sglang/srt/speculative/spec_utils.py",
    ),
}
# Upstream files the patch modifies (in patch order). The new files above ship as whole files.
PATCH_TARGETS = [
    "python/pyproject.toml",
    "python/sglang/srt/distributed/parallel_state.py",
    "python/sglang/srt/entrypoints/engine.py",
    "python/sglang/srt/entrypoints/http_server.py",
    "python/sglang/srt/layers/attention/flashinfer_backend.py",
    "python/sglang/srt/layers/attention/vision.py",
    "python/sglang/srt/managers/multi_tokenizer_mixin.py",
    "python/sglang/srt/managers/schedule_batch.py",
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/scheduler_output_processor_mixin.py",
    "python/sglang/srt/managers/scheduler_update_weights_mixin.py",
    "python/sglang/srt/managers/tokenizer_manager.py",
    "python/sglang/srt/managers/utils.py",
    "python/sglang/srt/model_executor/cuda_graph_runner.py",
    "python/sglang/srt/model_executor/forward_batch_info.py",
    "python/sglang/srt/model_executor/model_runner.py",
    "python/sglang/srt/models/glm_ocr.py",
    "python/sglang/srt/multimodal/processors/base_processor.py",
    "python/sglang/srt/multimodal/processors/glm4v.py",
    "python/sglang/srt/server_args.py",
    "python/sglang/srt/speculative/ngram_info.py",
    "python/sglang/srt/speculative/spec_info.py",
    "python/sglang/srt/utils/auth.py",
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


def _added_lines(target: str) -> str:
    """The lines the patch adds to one upstream file."""
    text = PATCH.read_text(encoding="utf-8")
    start = text.index(f"diff --git a/{target} ")
    end = text.find("\ndiff --git ", start + 1)
    hunk = text[start : end if end != -1 else len(text)]
    return "\n".join(line[1:] for line in hunk.splitlines() if line.startswith("+") and not line.startswith("+++"))


def test_server_info_never_returns_the_keys():
    # Upstream /server_info and /get_server_info return dataclasses.asdict(server_args), keys included, at the NORMAL
    # auth level: any API-key holder (or any local client, without a key) could read the admin key. The patch blanks
    # every secret field there; test_gravityocr_auth.py checks the running app with the patched SGLang.
    auth = _added_lines("python/sglang/srt/utils/auth.py")
    assert 'SECRET_SERVER_ARG_FIELDS = ("api_key", "admin_api_key", "ssl_keyfile_password")' in auth
    server = _added_lines("python/sglang/srt/entrypoints/http_server.py")
    assert "info = redact_server_args(" in server and "redact_server_args(state)" in server


def test_a_vit_graph_capture_failure_falls_back_to_eager():
    # Capture runs inside load_weights, on by default with FAST=1: a failure must only switch the graphs off. The
    # runner's setup() releases a failed capture and returns False (test_gravityocr_vit_graph.py drives it).
    glm = _added_lines("python/sglang/srt/models/glm_ocr.py")
    assert "if runner.setup():" in glm and "self.visual._ss_vit = runner" in glm
    assert "runner.capture()" not in glm


# ---------------------------------------------------------------- the new SGLang files
@pytest.mark.parametrize("rel", sorted(NEW_FILES))
def test_new_file_parses_on_py310_and_carries_the_header(rel):
    path = SGLANG_DIR / rel
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path), feature_version=(3, 10))
    assert source.splitlines()[:2] == LICENSE_HEADER
    header = "\n".join(_header(path))
    assert _base_commit() in header, "the header must name the full SGLang base commit"
    # Apache-2.0 4(c): a file adapted from SGLang keeps SGLang's copyright notice and says what it adapts
    flat = " ".join(line.lstrip("#").strip() for line in _header(path))
    sources = NEW_FILES[rel]
    assert (SGLANG_NOTICE in flat) == bool(sources), (
        f"{rel}: the header must carry '{SGLANG_NOTICE}' exactly when the file adapts SGLang code"
    )
    for src in sources:
        assert src in flat, f"{rel}: the header does not name {src}"
    if sources:
        assert "Apache License, Version 2.0" in flat
    assert ("ngram_info.py" in flat) == ("sglang/srt/speculative/ngram_info.py" in sources)
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
    # setup.sh installs exactly these from the PyTorch cu128 index, everything else from PyPI. They must
    # match torch's CUDA: the PyPI torch 2.9.1 is built for CUDA 12.8, and torchaudio refuses to import
    # when its CUDA differs ("PyTorch and TorchAudio were compiled with different CUDA versions").
    cu128 = sorted(n for n, v in pins.items() if v.endswith("+cu128"))
    assert cu128 == ["torchao", "torchaudio", "torchcodec", "torchvision"]
    assert not [n for n, v in pins.items() if "+" in v and not v.endswith("+cu128")]
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
    assert len(installs) == 3, installs  # PyPI pins, +cu128 pins, the patched SGLang
    for cmd in installs:
        for flag in ("--no-deps", "--no-config", "--default-index"):
            assert flag in cmd, f"{flag} missing from: {cmd}"
        for flag in ("--extra-index-url", "--index-url", "--find-links", "--index ", "-f "):
            assert flag not in cmd, f"{flag} in: {cmd}"
    lock_installs = [c for c in installs if "-r -" in c]
    assert len(lock_installs) == 2 and all("--no-build" in c for c in lock_installs)
    # the patched SGLang is built with the locked build backend (step 4), offline: nothing unpinned is
    # fetched to build it (with build isolation uv would resolve setuptools-scm / wheel from PyPI)
    (editable,) = [c for c in installs if " -e " in c]
    for flag in ("--no-build-isolation", "--offline"):
        assert flag in editable, f"{flag} missing from: {editable}"
    (unset,) = [line for line in _setup_commands("unset ") if "UV_EXTRA_INDEX_URL" in line]
    for var in ("UV_INDEX", "UV_EXTRA_INDEX_URL", "UV_FIND_LINKS", "UV_INDEX_URL", "UV_DEFAULT_INDEX"):
        assert f" {var} " in f"{unset} ", f"setup.sh does not unset {var}"


def test_setup_never_puts_the_cwd_on_sys_path():
    # From inference/gravityocr/, `python -` would find the recipe's own sglang/ folder before the
    # installed SGLang, and the verify step would fail with a misleading "env does not match".
    text = SETUP.read_text(encoding="utf-8")
    first_python = text.index('"$VENV/bin/python"')
    assert "export PYTHONSAFEPATH=1" in text[:first_python]


def test_lock_pins_the_build_backend():
    # setup.sh step 5 builds SGLang --no-build-isolation: python/pyproject.toml [build-system] needs
    # setuptools>=61.0, setuptools-scm>=8.0 and wheel; setuptools-scm 10 adds vcs-versioning and packaging
    pins = _pins(LOCK)
    for name in ("setuptools", "setuptools-scm", "vcs-versioning", "wheel", "packaging"):
        assert name in pins, f"the lock does not pin the build backend package {name}"
    version = pytest.importorskip("packaging.version").Version
    assert version(pins["setuptools"]) >= version("61.0") and version(pins["setuptools-scm"]) >= version("8.0")
    declared = {_norm(m) for m in re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", REQUIREMENTS_IN.read_text(), re.M)}
    assert {"setuptools", "setuptools-scm", "wheel"} <= declared


def test_validated_freeze_replays_the_lock():
    # The lock is `uv pip compile requirements.in -c validated-freeze.txt --exclude-newer <date> ...`
    # (its header). The freeze is committed, so the resolution replays from the repository; offline,
    # this checks the invariants that replay relies on (the generator's LOCK=check re-resolves).
    lock_text = LOCK.read_text(encoding="utf-8")
    freeze = _pins(FREEZE)
    assert len(freeze) > 100
    assert FREEZE.read_text(encoding="utf-8").startswith("# ")  # a header says what the file is
    floored = {_norm(n) for n in re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*>=", REQUIREMENTS_IN.read_text(), re.M)}
    assert floored and not floored & set(freeze), "a floored package is pinned by the freeze"
    lock = _pins(LOCK)
    for name, version in freeze.items():
        assert lock.get(name) == version, f"freeze {name}=={version}, lock {lock.get(name)}"
    # every lock entry the freeze constrained says so, and no other
    annotated = re.findall(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)==\S+\n(?:    #.*\n)*?    #   -c validated-freeze\.txt$", lock_text, re.M
    )
    assert sorted(map(_norm, annotated)) == sorted(freeze)
    header = lock_text[: lock_text.index(LOCK_HEADER_END) + len(LOCK_HEADER_END)]
    cmd = " ".join(line.strip("# \\") for line in header.splitlines() if line.startswith("#  "))
    assert "-c validated-freeze.txt" in cmd
    assert re.search(r"--exclude-newer \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", cmd), "no fixed resolution date"
    assert "not committed" not in header


# ---------------------------------------------------------------- setup.sh steps 1-3, for real
# setup.sh runs against a tiny git repository standing in for upstream SGLang, a one-hunk patch and one
# new file. A stub `uv` stops it at step 4 (the venv), so nothing is installed.
GIT = shutil.which("git")


def _git(cwd, *args):
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_NOSYSTEM": "1", "PATH": os.environ.get("PATH", ""), "HOME": str(cwd),
    }
    return subprocess.run([GIT, "-C", str(cwd), *args], check=True, capture_output=True, text=True, env=env).stdout


@pytest.fixture
def synthetic_recipe(tmp_path):
    if GIT is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    up = tmp_path / "upstream"
    (up / "python/sglang").mkdir(parents=True)
    _git(up, "init", "-q")
    (up / ".gitignore").write_text("__pycache__/\n*.egg-info/\npython/sglang/_version.py\n*.so\nbuild/\n")
    (up / "python/sglang/__init__.py").write_text("X = 1\n")
    (up / "python/sglang/a.py").write_text("".join(f"line_{i} = {i}\n" for i in range(1, 41)))
    (up / "python/sglang/b.py").write_text("B = 2\n")
    _git(up, "add", "-A")
    _git(up, "commit", "-qm", "base")
    base = _git(up, "rev-parse", "HEAD").strip()
    edited = tmp_path / "edited"
    shutil.copytree(up, edited)
    a = edited / "python/sglang/a.py"
    a.write_text(a.read_text().replace("line_5 = 5\n", "line_5 = 50\n"))
    recipe = tmp_path / "recipe"
    (recipe / "python/sglang/srt").mkdir(parents=True)
    shutil.copy(SETUP, recipe / "setup.sh")
    (recipe / "gravityocr_selfspec.patch").write_text(_git(edited, "diff"))
    (recipe / "python/sglang/srt/newfile.py").write_text("# new\n")
    (recipe / "BASE_COMMIT.txt").write_text(f"{base}  synthetic\n")
    (recipe / "requirements-lock.txt").write_text("x==1\n")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "uv").write_text(
        '#!/bin/sh\n[ "$1" = --version ] && { echo "uv 0.11.27"; exit 0; }\necho STOP_AT_STEP4 >&2; exit 97\n'
    )
    (stub / "uv").chmod(0o755)

    def run(src):
        # GIT_CEILING_DIRECTORIES: the recipe copy must not look like part of a repository that
        # happens to contain the temp dir (setup.sh refuses an SGLANG_SRC inside its own repository)
        env = {"PATH": f"{stub}:{os.environ.get('PATH', '')}", "HOME": str(tmp_path), "SKIP_PREFLIGHT": "1",
               "GIT_CONFIG_NOSYSTEM": "1", "GIT_CEILING_DIRECTORIES": str(tmp_path), "TMPDIR": str(tmp_path)}
        p = subprocess.run(["bash", str(recipe / "setup.sh"), f"SGLANG_SRC={src}"], env=env,
                           capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout + p.stderr

    def checkout(name):
        dst = tmp_path / name
        shutil.copytree(up, dst)
        return dst

    return run, checkout


def test_setup_sh_applies_the_patch_and_accepts_a_rerun(synthetic_recipe):
    run, checkout = synthetic_recipe
    src = checkout("c")
    rc, out = run(src)  # applies the patch, copies the new file
    assert rc == 97 and "applied gravityocr_selfspec.patch" in out and "STOP_AT_STEP4" in out, out
    assert (src / "python/sglang/a.py").read_text().count("line_5 = 50\n") == 1
    assert (src / "python/sglang/srt/newfile.py").read_text() == "# new\n"
    # what the editable install leaves behind is ignored by the checkout's .gitignore
    (src / "python/sglang/_version.py").write_text("version = '0.5.10'\n")
    (src / "python/sglang.egg-info").mkdir()
    (src / "python/sglang.egg-info/PKG-INFO").write_text("x\n")
    (src / "python/sglang/__pycache__").mkdir()
    (src / "python/sglang/__pycache__/a.cpython-312.pyc").write_bytes(b"\0")
    rc, out = run(src)  # "already applied", verified again
    assert rc == 97 and "already applied" in out and "checkout verified" in out, out
    assert not list(src.parent.glob("tmp*")), "setup.sh left a scratch directory behind"


@pytest.mark.parametrize(
    "edit,needle",
    [
        # a patched file, outside the hunk: the old file-name check accepted this
        (lambda s: (s / "python/sglang/a.py").write_text(
            (s / "python/sglang/a.py").read_text().replace("line_30 = 30\n", "line_30 = 31\n")), "differ from"),
        (lambda s: (s / "python/sglang/a.py").write_text((s / "python/sglang/a.py").read_text() + "extra = 1\n"),
         "differ from"),
        (lambda s: (s / "python/sglang/b.py").write_text("B = 3\n"), "differ from"),
        (lambda s: (s / "python/sglang/b.py").unlink(), "differ from"),
        (lambda s: (s / "python/sglang/extra.py").write_text("import os\n"), "neither upstream nor ours"),
        (lambda s: (s / "python/sglang/srt/sub").mkdir() or (s / "python/sglang/srt/sub/x.py").write_text(""),
         "neither upstream nor ours"),
        # ignored by the checkout's .gitignore, but an extension module shadows the .py beside it
        (lambda s: (s / "python/sglang/a.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\0"), "ignored files"),
        (lambda s: (s / "python/build/lib").mkdir(parents=True) or (s / "python/build/lib/a.py").write_text(""),
         "ignored files"),
        # an edit git diff would not see: the index entry is marked skip-worktree / assume-unchanged
        (lambda s: _git(s, "update-index", "--skip-worktree", "python/sglang/b.py")
         or (s / "python/sglang/b.py").write_text("B = 3\n"), "skip-worktree"),
        (lambda s: _git(s, "update-index", "--assume-unchanged", "python/sglang/b.py")
         or (s / "python/sglang/b.py").write_text("B = 3\n"), "skip-worktree"),
    ],
    ids=["edit-in-patched-file", "line-appended-to-patched-file", "edit-in-unpatched-file",
         "deleted-unpatched-file", "extra-untracked-file", "extra-untracked-subdir", "ignored-extension-module",
         "ignored-build-dir", "skip-worktree-edit", "assume-unchanged-edit"],
)
def test_setup_sh_refuses_a_checkout_that_is_not_base_plus_patch(synthetic_recipe, edit, needle):
    run, checkout = synthetic_recipe
    src = checkout("c")
    assert run(src)[0] == 97  # the first run applies the patch
    edit(src)
    rc, out = run(src)
    assert rc == 1 and needle in out and "STOP_AT_STEP4" not in out, out
