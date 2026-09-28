#!/usr/bin/env bash
# Fetch the two-stream Gated-DeltaNet + ShortConv Triton kernels into this directory.
# These kernels are PolyForm-Noncommercial (upstream) and are NOT vendored in this
# Apache-2.0 repo — see README.md. Recipe: clone pinned upstream, copy, apply our patch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${HYBRIDDIFFUSION_URL:-https://github.com/yuchen-zhu-zyc/HybridDiffusion}"
COMMIT="${HYBRIDDIFFUSION_COMMIT:-6ca547a}"
SUBDIR="torchtitan/models/qwen3_5/model/block_gated_delta_rule"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git clone --quiet "$URL" "$TMP/hd"
git -C "$TMP/hd" checkout --quiet "$COMMIT"
cp "$TMP/hd/$SUBDIR"/*.py "$HERE/"
( cd "$HERE" && patch -p1 < trillion_mods.patch )
echo "block_gated_delta_rule kernels fetched into $HERE (upstream $COMMIT + trillion_mods.patch)"
