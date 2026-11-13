#!/usr/bin/env bash
# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup for run.sh: everything the GravityOCR evaluation needs, under one directory (EVAL_HOME).
#
#     bash benchmark/gravityocr/setup.sh                                   # EVAL_HOME=$HOME/.gravityocr-eval
#     EVAL_HOME=/opt/gravityocr-eval bash benchmark/gravityocr/setup.sh    # e.g. in a Dockerfile
#     bash benchmark/gravityocr/setup.sh SKIP_SGLANG=1                     # scoring only (no GPU server)
#
# Steps. Re-running is safe: finished steps are reused.
#   1. Preflight: uv, git, curl, tar and perl on PATH (perl runs the TeX Live installer).
#   2. OmniDocBench (the metric code) at the commit in OMNIDOCBENCH_COMMIT, into $EVAL_HOME/OmniDocBench.
#   3. The scoring venv $EVAL_HOME/venv (Python 3.10): OmniDocBench's pinned requirements plus datasets,
#      sacrebleu and pymupdf. run.sh uses it for the dataset export, the client, scoring and the report.
#   4. TeX Live from upstream (install-tl), into $EVAL_HOME/texlive, with the packages CDM needs. Not apt:
#      CDM colors tokens with \mathcolor (xcolor 2.14+), and the TeX Live in Ubuntu 22.04 (xcolor 2.13) renders
#      every token black, which scores every formula 0.
#   5. The patched SGLang server env, by inference/gravityocr/sglang/setup.sh, into $EVAL_HOME/sglang
#      (needs nvcc 12.x at CUDA_HOME; no GPU needed to build). SKIP_SGLANG=1 skips it.
#   6. $EVAL_HOME/env.sh, which run.sh reads: VBIN, PYTHON, OMNIDOCBENCH and TeX Live on PATH.
#   7. Check: pdflatex renders \mathcolor in color.
#
# Env, or KEY=VALUE arguments:
#   EVAL_HOME           install directory (default $HOME/.gravityocr-eval)
#   SKIP_SGLANG=1       skip step 5 (scoring only, or VBIN comes from elsewhere)
#   SKIP_TEXLIVE=1      skip step 4 and use the pdflatex on PATH (it must be TeX Live 2022.06 or newer)
#   TL_REPO             TeX Live package repository (default https://mirror.ctan.org/systems/texlive/tlnet)
#   CUDA_HOME, PY       passed to the SGLang setup (defaults there: /usr/local/cuda-12.8, Python 3.12)
set -euo pipefail

OMNIDOCBENCH_REPO=https://github.com/opendatalab/OmniDocBench
OMNIDOCBENCH_COMMIT=0604d9843f1dfa3a292bd5f253ae83a1ba47cd24
TEX_PACKAGES="geometry booktabs multirow amsmath amsfonts xcolor was cjk cjk-ko nanumtype1 uhc unfonts-core"
SCORE_PACKAGES="datasets==5.2.0 sacrebleu==2.6.0 pymupdf==1.28.2"

for kv in "$@"; do
  case "$kv" in
    -h|--help) sed -n '/^# One-time /,/^set -euo pipefail$/{/^set /d;s/^# \{0,1\}//;p;}' "$0"; exit 0 ;;
    EVAL_HOME=*|SKIP_SGLANG=*|SKIP_TEXLIVE=*|TL_REPO=*|CUDA_HOME=*|PY=*) export "${kv?}" ;;
    *) echo "setup.sh: unknown argument: $kv" >&2; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
die() { echo "setup.sh: $*" >&2; exit 1; }
log() { echo "[setup] $*"; }
EVAL_HOME=${EVAL_HOME:-$HOME/.gravityocr-eval}
TL_REPO=${TL_REPO:-https://mirror.ctan.org/systems/texlive/tlnet}
mkdir -p "$EVAL_HOME"
EVAL_HOME=$(cd "$EVAL_HOME" && pwd)
log "EVAL_HOME=$EVAL_HOME"

# ── 1. preflight ─────────────────────────────────────────────────────────────────────────────────
for tool in uv git curl tar perl; do
  command -v "$tool" > /dev/null || die "$tool not found on PATH${tool/uv/ (https://docs.astral.sh/uv/getting-started/installation/)}"
done

# ── 2. OmniDocBench ──────────────────────────────────────────────────────────────────────────────
ODB=$EVAL_HOME/OmniDocBench
if [ "$(git -C "$ODB" rev-parse HEAD 2>/dev/null || true)" = "$OMNIDOCBENCH_COMMIT" ]; then
  log "reusing $ODB at ${OMNIDOCBENCH_COMMIT:0:7}"
else
  [ ! -e "$ODB" ] || [ -z "$(ls -A "$ODB")" ] || die "$ODB exists and is not OmniDocBench at ${OMNIDOCBENCH_COMMIT:0:7}; remove it"
  log "fetching OmniDocBench at ${OMNIDOCBENCH_COMMIT:0:7}"
  mkdir -p "$ODB"
  git -C "$ODB" init -q
  git -C "$ODB" fetch -q --depth 1 "$OMNIDOCBENCH_REPO" "$OMNIDOCBENCH_COMMIT"
  git -C "$ODB" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
fi

# ── 3. scoring venv ──────────────────────────────────────────────────────────────────────────────
VENV=$EVAL_HOME/venv
[ -x "$VENV/bin/python" ] || { log "creating $VENV (Python 3.10)"; uv venv -q --python 3.10 "$VENV"; }
log "installing OmniDocBench's requirements and $SCORE_PACKAGES"
# shellcheck disable=SC2086
uv pip install -q --python "$VENV/bin/python" "$ODB" $SCORE_PACKAGES
"$VENV/bin/python" -c "import datasets, sacrebleu, pymupdf, Levenshtein, apted, pylatexenc" \
  || die "the scoring venv is missing a package"

# ── 4. TeX Live ──────────────────────────────────────────────────────────────────────────────────
TL=$EVAL_HOME/texlive
if [ "${SKIP_TEXLIVE:-0}" = 1 ]; then
  command -v pdflatex > /dev/null || die "SKIP_TEXLIVE=1 but pdflatex is not on PATH"
  TEXBIN=$(dirname "$(command -v pdflatex)")
  log "using the TeX Live on PATH ($TEXBIN)"
else
  TEXBIN=$(ls -d "$TL"/bin/*/ 2>/dev/null | head -1 || true)
  if [ -z "$TEXBIN" ] || [ ! -x "$TEXBIN/pdflatex" ]; then
    log "installing TeX Live (scheme-basic) into $TL from $TL_REPO; this takes a few minutes"
    work=$(mktemp -d)
    # mirror.ctan.org redirects each request to a random mirror, and some have broken TLS: retry until one
    # works, then use that same mirror for the installer and tlmgr.
    for _ in 1 2 3 4 5 6; do
      if url=$(curl -fsSL -o "$work/install-tl.tar.gz" -w '%{url_effective}' "$TL_REPO/install-tl-unx.tar.gz"); then
        TL_REPO=${url%/install-tl-unx.tar.gz}; break
      fi
      rm -f "$work/install-tl.tar.gz"
    done
    [ -s "$work/install-tl.tar.gz" ] || die "could not download install-tl from $TL_REPO (set TL_REPO to a mirror)"
    log "TeX Live mirror: $TL_REPO"
    tar xzf "$work/install-tl.tar.gz" -C "$work" --strip-components=1
    cat > "$work/eval.profile" <<EOF
selected_scheme scheme-basic
TEXDIR $TL
TEXMFLOCAL $TL/texmf-local
TEXMFSYSVAR $TL/texmf-var
TEXMFSYSCONFIG $TL/texmf-config
TEXMFVAR $TL/texmf-var
TEXMFCONFIG $TL/texmf-config
TEXMFHOME $TL/texmf-home
instopt_adjustpath 0
tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0
tlpdbopt_autobackup 0
EOF
    perl "$work/install-tl" --no-interaction --profile "$work/eval.profile" --repository "$TL_REPO" > "$EVAL_HOME/texlive-install.log" 2>&1 \
      || { tail -20 "$EVAL_HOME/texlive-install.log" >&2; die "TeX Live install failed (log: $EVAL_HOME/texlive-install.log)"; }
    rm -rf "$work"
    TEXBIN=$(ls -d "$TL"/bin/*/ | head -1)
  fi
  TEXBIN=${TEXBIN%/}
  log "installing TeX packages: $TEX_PACKAGES"
  # shellcheck disable=SC2086
  for try in 1 2 3; do
    if "$TEXBIN/tlmgr" --repository "$TL_REPO" install $TEX_PACKAGES > "$EVAL_HOME/tlmgr.log" 2>&1; then break; fi
    [ "$try" = 3 ] && { tail -20 "$EVAL_HOME/tlmgr.log" >&2; die "tlmgr install failed (log: $EVAL_HOME/tlmgr.log)"; }
    log "tlmgr failed (try $try); retrying"
  done
fi

# ── 5. patched SGLang ────────────────────────────────────────────────────────────────────────────
VBIN=
if [ "${SKIP_SGLANG:-0}" = 1 ]; then
  log "SKIP_SGLANG=1: no server env (set VBIN for run.sh, or use run.sh with ENDPOINT)"
else
  log "building the patched SGLang server env into $EVAL_HOME/sglang (inference/gravityocr/sglang/setup.sh)"
  SGLANG_SRC=$EVAL_HOME/sglang bash "$REPO/inference/gravityocr/sglang/setup.sh"
  VBIN=$EVAL_HOME/sglang/.venv-sglang/bin
  [ -x "$VBIN/python" ] || die "the SGLang setup did not produce $VBIN/python"
fi

# ── 6. env.sh ────────────────────────────────────────────────────────────────────────────────────
# Defaults only: a value already in the environment wins.
{
  echo "# Written by benchmark/gravityocr/setup.sh; run.sh reads it. Defaults only: the environment wins."
  [ -z "$VBIN" ] || echo ": \"\${VBIN:=$VBIN}\""
  echo ": \"\${PYTHON:=$VENV/bin/python}\""
  echo ": \"\${OMNIDOCBENCH:=$ODB}\""
  echo "export VBIN PYTHON OMNIDOCBENCH"
  echo "case \":\$PATH:\" in *\":$TEXBIN:\"*) ;; *) export PATH=\"$TEXBIN:\$PATH\" ;; esac"
} > "$EVAL_HOME/env.sh"

# ── 7. check: \mathcolor renders in color ────────────────────────────────────────────────────────
check=$(mktemp -d)
cat > "$check/t.tex" <<'EOF'
\documentclass{article}
\usepackage{xcolor}
\begin{document}
$\mathcolor[RGB]{0,0,15}{a}$
\end{document}
EOF
(cd "$check" && PATH="$TEXBIN:$PATH" pdflatex -interaction=nonstopmode t.tex > /dev/null 2>&1) || true
grep -q "^! Undefined control sequence" "$check/t.log" 2>/dev/null \
  && die "pdflatex in $TEXBIN has no \\mathcolor (xcolor older than 2.14): CDM would score every formula 0"
[ -s "$check/t.pdf" ] || die "pdflatex in $TEXBIN could not compile a test document (see $check/t.log)"
rm -rf "$check"

log "done. Run the evaluation with:"
log "  EVAL_HOME=$EVAL_HOME MODEL=<checkpoint or Hub id> bash $HERE/run.sh"
