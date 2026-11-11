#!/usr/bin/env bash
# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# Build the patched SGLang server env that inference/gravityocr/serve.sh runs (its VBIN).
#
#   SGLANG_SRC=$HOME/src/sglang-gravityocr bash inference/gravityocr/sglang/setup.sh
#   bash inference/gravityocr/sglang/setup.sh SGLANG_SRC=$HOME/src/sglang-gravityocr   # same
#
# Steps. Re-running is safe: a finished build is only re-verified.
#   1. Preflight: uv and git on PATH, nvcc 12.x at $CUDA_HOME. FlashInfer and Triton compile kernels
#      when the server starts, so the CUDA toolkit is a runtime requirement, not only a build one.
#   2. Upstream sgl-project/sglang at the commit in BASE_COMMIT.txt (tag v0.5.10): a shallow fetch of
#      exactly that commit into an empty $SGLANG_SRC, or reuse a checkout that is already at it.
#   3. Apply gravityocr_selfspec.patch (skipped when already applied) and copy the new files in python/.
#   4. Create $SGLANG_SRC/.venv-sglang (Python $PY) and install requirements-lock.txt as is: --no-deps,
#      --no-build, nothing re-resolved. PyPI pins first, from https://pypi.org/simple only, then the four
#      +cu129 builds from the PyTorch cu129 index only (the lock header says why it is two steps). uv
#      config files and index variables (UV_INDEX, UV_EXTRA_INDEX_URL, UV_FIND_LINKS, ...) are ignored.
#   5. Install the patched SGLang editable, with version 0.5.10. Upstream derives its version from git
#      tags, which a shallow fetch does not have.
#   6. Check the installed packages against the lock, then print VBIN for serve.sh.
#
# Env, or KEY=VAL arguments:
#   SGLANG_SRC        required. Directory for the SGLang checkout and its venv, outside this repository.
#                     Empty or missing: fetched. A clean SGLang checkout at another commit: the base
#                     commit is fetched into it (keeping its history; it is not made shallow) and checked
#                     out as a detached HEAD, so its branches are left alone. A checkout with local
#                     changes, or a git repository that is not SGLang: refused.
#   PY                Python for the venv (default 3.12; the lock is resolved for 3.12).
#   CUDA_HOME         CUDA toolkit (default /usr/local/cuda-12.8). Must be 12.x.
#   SGLANG_REPO       upstream URL (default https://github.com/sgl-project/sglang.git).
#   SKIP_PREFLIGHT=1  skip the nvcc check, e.g. to build the env on a machine without the toolkit.
#
# The patch and serve.sh were validated on x86_64 Linux, H100 (sm_90), NVIDIA driver 570 (CUDA 12.8),
# in the environment frozen in the GravityOCR repository. requirements-lock.txt raises 11 transitive
# pins above that freeze for security fixes. A fresh run of this script with it (2026-10-30) gave the
# same greedy outputs as that freeze on 100 OCR crops, in self-spec and causal mode.
# The lock was generated with uv 0.11.27.
set -euo pipefail
# Never put the current directory on sys.path: run from inference/gravityocr/, `python -` would find the
# recipe's own sglang/ folder before the installed SGLang, and the verify step would fail.
export PYTHONSAFEPATH=1

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# Package sources are fixed by the flags below; no index or config from the environment may add one.
unset UV_INDEX UV_INDEX_URL UV_DEFAULT_INDEX UV_EXTRA_INDEX_URL UV_FIND_LINKS UV_NO_INDEX \
      UV_INDEX_STRATEGY UV_CONFIG_FILE UV_NO_CONFIG
for kv in "$@"; do
  case "$kv" in
    SGLANG_SRC=*|PY=*|CUDA_HOME=*|SGLANG_REPO=*|SKIP_PREFLIGHT=*) export "${kv?}" ;;
    -h|--help) sed -n '2,/^set -euo pipefail$/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup.sh: unknown argument '$kv' (expected KEY=VAL; see --help)" >&2; exit 2 ;;
  esac
done

: "${SGLANG_SRC:?set SGLANG_SRC to a directory for the SGLang checkout, outside this repository}"
PY=${PY:-3.12}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
SGLANG_REPO=${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}
CU129_INDEX=https://download.pytorch.org/whl/cu129
LOCK=$HERE/requirements-lock.txt
PATCH=$HERE/gravityocr_selfspec.patch
SGLANG_VERSION=0.5.10
UV_TESTED=0.11.27
PYPI_INDEX=https://pypi.org/simple

die() { echo "setup.sh: $*" >&2; exit 1; }
log() { echo "[setup] $*"; }

BASE=$(head -c 40 "$HERE/BASE_COMMIT.txt")
[[ $BASE =~ ^[0-9a-f]{40}$ ]] || die "BASE_COMMIT.txt does not start with a 40-hex commit"
for f in "$LOCK" "$PATCH"; do [ -f "$f" ] || die "missing $f"; done

# ── 1. preflight ──────────────────────────────────────────────────────────────────────────────────
command -v git >/dev/null || die "git not found"
command -v uv >/dev/null || die "uv not found (https://docs.astral.sh/uv/getting-started/installation/)"
uv_version=$(uv --version | awk '{print $2}')
[ "$uv_version" = "$UV_TESTED" ] || log "note: uv $uv_version; the lock was generated and tested with uv $UV_TESTED"
PY_MM=$(echo "$PY" | cut -d. -f1,2)
[ "$PY_MM" = 3.12 ] || log "warning: PY=$PY, but requirements-lock.txt is resolved for Python 3.12"
if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
  [ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc at $CUDA_HOME/bin/nvcc. FlashInfer and Triton compile kernels" \
    "when the server starts, so a CUDA 12.x toolkit is required: set CUDA_HOME (SKIP_PREFLIGHT=1 builds the env anyway)"
  nvcc_release=$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
  case "$nvcc_release" in
    12.*) log "nvcc $nvcc_release at $CUDA_HOME" ;;
    *) die "nvcc at $CUDA_HOME reports release '${nvcc_release:-unknown}'; this env needs CUDA 12.x" \
         "(cu12 wheels, validated with a CUDA 12.8 toolkit and driver)" ;;
  esac
fi

# ── 2. upstream checkout at BASE ──────────────────────────────────────────────────────────────────
mkdir -p "$SGLANG_SRC"
SGLANG_SRC=$(cd "$SGLANG_SRC" && pwd -P)
REPO_TOP=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || true)
if [ -n "$REPO_TOP" ]; then
  case "$SGLANG_SRC/" in "$REPO_TOP"/*) die "SGLANG_SRC must be outside this repository ($REPO_TOP)" ;; esac
fi
if [ "$(git -C "$SGLANG_SRC" rev-parse --show-toplevel 2>/dev/null || true)" = "$SGLANG_SRC" ]; then
  head_commit=$(git -C "$SGLANG_SRC" rev-parse -q --verify HEAD 2>/dev/null || echo none)
  if [ "$head_commit" = "$BASE" ]; then
    log "reusing $SGLANG_SRC (HEAD is $BASE)"
  else
    [ -f "$SGLANG_SRC/python/sglang/__init__.py" ] \
      || die "$SGLANG_SRC is a git repository but not an SGLang checkout (no python/sglang/); use an empty directory"
    [ -z "$(git -C "$SGLANG_SRC" status --porcelain --untracked-files=no)" ] \
      || die "$SGLANG_SRC is a checkout of $head_commit with local changes; refusing to switch it to $BASE." \
             "Use an empty directory."
    log "switching $SGLANG_SRC from $head_commit to $BASE (detached HEAD; branches are left alone)"
    depth=()   # keep a full clone full: --depth would make it shallow
    if [ "$(git -C "$SGLANG_SRC" rev-parse --is-shallow-repository)" = true ]; then depth=(--depth 1); fi
    git -C "$SGLANG_SRC" fetch -q ${depth[@]+"${depth[@]}"} "$SGLANG_REPO" "$BASE"
    git -C "$SGLANG_SRC" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
  fi
else
  [ -z "$(ls -A "$SGLANG_SRC")" ] || die "$SGLANG_SRC exists, is not empty and is not a git checkout"
  log "fetching $SGLANG_REPO at $BASE (shallow)"
  git -C "$SGLANG_SRC" init -q
  git -C "$SGLANG_SRC" fetch -q --depth 1 "$SGLANG_REPO" "$BASE"
  git -C "$SGLANG_SRC" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
fi
[ "$(git -C "$SGLANG_SRC" rev-parse HEAD)" = "$BASE" ] || die "$SGLANG_SRC is not at $BASE"

# ── 3. patch + new files ──────────────────────────────────────────────────────────────────────────
if git -C "$SGLANG_SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
  log "gravityocr_selfspec.patch already applied"
else
  git -C "$SGLANG_SRC" apply --check "$PATCH" \
    || die "gravityocr_selfspec.patch does not apply to $SGLANG_SRC (local changes to the files it touches?)"
  git -C "$SGLANG_SRC" apply "$PATCH"
  log "applied gravityocr_selfspec.patch"
fi
want=$(sed -n 's#^diff --git a/\([^ ]*\) b/.*#\1#p' "$PATCH" | sort)
have=$(git -C "$SGLANG_SRC" diff --name-only | sort)
[ "$want" = "$have" ] || die "the tracked changes in $SGLANG_SRC are not exactly the patch's files:" \
  "$(diff <(echo "$want") <(echo "$have") || true)"
mapfile -t new_files < <(cd "$HERE" && find python -type f -name '*.py' | sort)
[ "${#new_files[@]}" -gt 0 ] || die "no new files under $HERE/python"
for f in "${new_files[@]}"; do
  if git -C "$SGLANG_SRC" ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    die "$f is an upstream file; the new files must not overwrite one"
  fi
  install -D -m 644 "$HERE/$f" "$SGLANG_SRC/$f"
done
log "copied ${#new_files[@]} new files into $SGLANG_SRC/python"
exclude_file=$(git -C "$SGLANG_SRC" rev-parse --git-path info/exclude)
case "$exclude_file" in /*) ;; *) exclude_file=$SGLANG_SRC/$exclude_file ;; esac
mkdir -p "$(dirname "$exclude_file")"
grep -qxF '/.venv-sglang/' "$exclude_file" 2>/dev/null || echo '/.venv-sglang/' >> "$exclude_file"

# ── 4. venv + locked dependencies ─────────────────────────────────────────────────────────────────
VENV=$SGLANG_SRC/.venv-sglang
if [ -x "$VENV/bin/python" ]; then
  venv_mm=$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [ "$venv_mm" = "$PY_MM" ] || die "$VENV is Python $venv_mm, not $PY_MM: remove it, or set PY=$venv_mm"
  log "reusing $VENV (Python $venv_mm)"
else
  uv venv --no-config -q --python "$PY" "$VENV"
  log "created $VENV"
fi
pins=$(grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' "$LOCK" | grep -v '^sglang==')
pypi_pins=$(grep -v '+cu129$' <<<"$pins")
cu129_pins=$(grep '+cu129$' <<<"$pins" || true)
log "installing $(wc -l <<<"$pypi_pins") locked pins from $PYPI_INDEX"
uv pip install --no-config --python "$VENV/bin/python" --no-deps --no-build \
  --default-index "$PYPI_INDEX" -r - <<<"$pypi_pins"
if [ -n "$cu129_pins" ]; then
  log "installing $(wc -l <<<"$cu129_pins") locked +cu129 pins from $CU129_INDEX"
  uv pip install --no-config --python "$VENV/bin/python" --no-deps --no-build \
    --default-index "$CU129_INDEX" -r - <<<"$cu129_pins"
fi

# ── 5. the patched SGLang, editable ───────────────────────────────────────────────────────────────
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=$SGLANG_VERSION \
  uv pip install --no-config -q --python "$VENV/bin/python" --no-deps --default-index "$PYPI_INDEX" \
  -e "$SGLANG_SRC/python"
log "installed sglang $SGLANG_VERSION (editable) from $SGLANG_SRC/python"

# ── 6. verify ─────────────────────────────────────────────────────────────────────────────────────
FREEZE=$(mktemp)
trap 'rm -f "$FREEZE"' EXIT
uv pip freeze --python "$VENV/bin/python" > "$FREEZE"
"$VENV/bin/python" - "$LOCK" "$FREEZE" "$SGLANG_SRC/python" "$SGLANG_VERSION" <<'PY' \
  || die "the installed env does not match requirements-lock.txt (see above)"
import importlib.util
import re
import sys
from importlib import metadata
from pathlib import Path

lock_path, freeze_path, src_python, sglang_version = sys.argv[1:5]
pin = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")


def pins(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = pin.match(line)
        if m:
            out[re.sub(r"[-_.]+", "-", m[1]).lower()] = m[2]
    return out


lock, have = pins(lock_path), pins(freeze_path)
lock.pop("sglang")
have.pop("sglang", None)  # the editable install appears as "-e file://..." anyway
problems = [f"missing {n}=={v}" for n, v in lock.items() if n not in have]
problems += [f"{n}: installed {have[n]}, lock {v}" for n, v in lock.items() if have.get(n, v) != v]
problems += [f"not in the lock: {n}=={v}" for n, v in have.items() if n not in lock]
if metadata.version("sglang") != sglang_version:
    problems.append(f"sglang metadata version {metadata.version('sglang')}, want {sglang_version}")
origin = Path(importlib.util.find_spec("sglang").origin).resolve()
if not origin.is_relative_to(Path(src_python).resolve()):
    problems.append(f"sglang resolves to {origin}, not the patched tree {src_python}")
for name, want in (("torch", "2.9.1"), ("flashinfer-python", "0.6.7.post2"), ("transformers", "5.3.0")):
    if metadata.version(name) != want:
        problems.append(f"{name} {metadata.version(name)}, want {want}")
for p in problems:
    print(f"  {p}", file=sys.stderr)
sys.exit(1 if problems else 0)
PY
[ -x "$VENV/bin/ninja" ] || die "ninja missing from $VENV/bin (FlashInfer's JIT needs it)"
log "OK: sglang $SGLANG_VERSION + gravityocr_selfspec.patch at $BASE; $(grep -c . <<<"$pins") locked packages match"
if ! uv pip check -q --python "$VENV/bin/python" >/dev/null 2>&1; then
  log "note: 'uv pip check' reports dependency metadata conflicts (informational; review them):"
  uv pip check --python "$VENV/bin/python" 2>&1 | sed 's/^/  /' || true
fi
echo "VBIN=$VENV/bin"
echo "next: VBIN=$VENV/bin bash $(dirname "$HERE")/serve.sh"
