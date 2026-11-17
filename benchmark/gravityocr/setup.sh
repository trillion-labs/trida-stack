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
#   5. The server env. Linux: the patched SGLang env, by inference/gravityocr/sglang/setup.sh, into
#      $EVAL_HOME/sglang (needs nvcc 12.x at CUDA_HOME; no GPU needed to build). Apple Silicon Mac: the MLX
#      engine env, by inference/gravityocr/mac/setup.sh, into $EVAL_HOME/mac-venv (no Gravity app needed).
#      SKIP_SGLANG=1 skips it.
#   6. $EVAL_HOME/env.sh, which run.sh reads: VBIN or MAC_PYTHON, PYTHON, OMNIDOCBENCH and TeX Live on PATH.
#   7. Check: pdflatex renders \mathcolor in color.
#
# Env, or KEY=VALUE arguments:
#   EVAL_HOME           install directory (default $HOME/.gravityocr-eval)
#   SKIP_SGLANG=1       skip step 5 (scoring only, or the server env comes from elsewhere)
#   SKIP_TEXLIVE=1      skip step 4 and use the pdflatex on PATH (it must be TeX Live 2022.06 or newer)
#   TL_REPO             TeX Live package repository (default https://mirror.ctan.org/systems/texlive/tlnet, which
#                       picks a nearby mirror; a failed install is retried there, then on TL_FALLBACK)
#   TL_FALLBACK         mirrors tried after TL_REPO, space-separated (default: JAIST, Japan, then RIT, US)
#   CUDA_HOME, PY       passed to the SGLang setup (defaults there: /usr/local/cuda-12.8, Python 3.12)
set -euo pipefail

OMNIDOCBENCH_REPO=https://github.com/opendatalab/OmniDocBench
OMNIDOCBENCH_COMMIT=0604d9843f1dfa3a292bd5f253ae83a1ba47cd24
TEX_PACKAGES="geometry booktabs multirow amsmath amsfonts xcolor was cjk cjk-ko nanumtype1 uhc unfonts-core"
SCORE_PACKAGES="datasets==5.2.0 sacrebleu==2.6.0 pymupdf==1.28.2"

for kv in "$@"; do
  case "$kv" in
    -h|--help) sed -n '/^# One-time /,/^set -euo pipefail$/{/^set /d;s/^# \{0,1\}//;p;}' "$0"; exit 0 ;;
    EVAL_HOME=*|SKIP_SGLANG=*|SKIP_TEXLIVE=*|TL_REPO=*|TL_FALLBACK=*|CUDA_HOME=*|PY=*) export "${kv?}" ;;
    *) echo "setup.sh: unknown argument: $kv" >&2; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
die() { echo "setup.sh: $*" >&2; exit 1; }
log() { echo "[setup] $*"; }
EVAL_HOME=${EVAL_HOME:-$HOME/.gravityocr-eval}
TL_REPO=${TL_REPO:-https://mirror.ctan.org/systems/texlive/tlnet}
TL_FALLBACK=${TL_FALLBACK:-https://ftp.jaist.ac.jp/pub/CTAN/systems/texlive/tlnet https://mirrors.rit.edu/CTAN/systems/texlive/tlnet}
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
    # mirror.ctan.org redirects each request to a random nearby mirror, and some are broken: bad TLS, or the
    # installer but not every package (from Korea, mirror.kakao.com has no archive/texlive-scripts.tar.xz). So
    # each try runs the whole install on one mirror, and a failed try moves on: TL_REPO (three times if it is
    # the redirector, a new mirror each time), then TL_FALLBACK. The mirror that worked is then used for tlmgr too.
    ok=
    tries="$TL_REPO"
    case "$TL_REPO" in *//mirror.ctan.org/*) tries="$TL_REPO $TL_REPO $TL_REPO" ;; esac
    # shellcheck disable=SC2086
    for repo in $tries $TL_FALLBACK; do
      work=$(mktemp -d)
      if url=$(curl -fsSL -o "$work/install-tl.tar.gz" -w '%{url_effective}' "$repo/install-tl-unx.tar.gz"); then
        mirror=${url%/install-tl-unx.tar.gz}
        log "TeX Live mirror: $mirror"
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
        rm -rf "$TL"  # what a failed try left behind
        # From $work: on failure install-tl writes install-tl.log into its working directory
        if (cd "$work" && perl ./install-tl --no-interaction --profile eval.profile --repository "$mirror") \
             > "$EVAL_HOME/texlive-install.log" 2>&1; then
          TL_REPO=$mirror; ok=1; rm -rf "$work"; break
        fi
        log "install-tl failed from $mirror; trying another mirror (log: $EVAL_HOME/texlive-install.log)"
      fi
      rm -rf "$work"
    done
    [ -n "$ok" ] || { tail -20 "$EVAL_HOME/texlive-install.log" >&2 2>/dev/null || true
      die "TeX Live install failed from every mirror tried; set TL_REPO to a working mirror, e.g. TL_REPO=https://ftp.jaist.ac.jp/pub/CTAN/systems/texlive/tlnet (log: $EVAL_HOME/texlive-install.log)"; }
    TEXBIN=$(ls -d "$TL"/bin/*/ | head -1)
  fi
  TEXBIN=${TEXBIN%/}
  log "installing TeX packages: $TEX_PACKAGES"
  # TL_REPO (twice: the redirector may pick another mirror), then TL_FALLBACK
  ok=
  # shellcheck disable=SC2086
  for repo in "$TL_REPO" "$TL_REPO" $TL_FALLBACK; do
    if "$TEXBIN/tlmgr" --repository "$repo" install $TEX_PACKAGES > "$EVAL_HOME/tlmgr.log" 2>&1; then ok=1; break; fi
    log "tlmgr failed from $repo; trying another mirror"
  done
  [ -n "$ok" ] || { tail -20 "$EVAL_HOME/tlmgr.log" >&2
    die "tlmgr install failed from every mirror tried; set TL_REPO to a working mirror (log: $EVAL_HOME/tlmgr.log)"; }
fi

# ── 5. server env: patched SGLang (Linux) or the MLX engine (Apple Silicon Mac) ─────────────────
VBIN='' MAC_PYTHON=''
if [ "${SKIP_SGLANG:-0}" = 1 ]; then
  log "SKIP_SGLANG=1: no server env (set VBIN or MAC_PYTHON for run.sh, or use run.sh with ENDPOINT)"
elif [ "$(uname -s)-$(uname -m)" = Darwin-arm64 ]; then
  log "building the Mac MLX engine env into $EVAL_HOME/mac-venv (inference/gravityocr/mac/setup.sh)"
  MAC_VENV=$EVAL_HOME/mac-venv bash "$REPO/inference/gravityocr/mac/setup.sh"
  MAC_PYTHON=$EVAL_HOME/mac-venv/bin/python
  [ -x "$MAC_PYTHON" ] || die "the Mac setup did not produce $MAC_PYTHON"
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
  [ -z "$MAC_PYTHON" ] || echo ": \"\${MAC_PYTHON:=$MAC_PYTHON}\"; export MAC_PYTHON"
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
