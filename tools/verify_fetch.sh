#!/usr/bin/env bash
# Verify that the third-party kernel fetch recipe still works.
#
# The recipes in train/block_gated_delta_rule/, inference/zoro-1/sglang/ and
# inference/zoro-1/vllm/vllm_native_diffusion/ ARE the install path for both two-stream
# code paths. If upstream moves, renames, force-pushes, or the subpath changes,
# nobody can build -- and we would normally find out from a bug report.
#
# This checks, in order:
#   1. the pinned commit still resolves upstream
#   2. the kernel subpath still exists
#   3. the fetched files match tools/kernel-checksums.txt  (integrity, not just reachability)
#   4. trillion_mods.patch still applies cleanly
#   5. the patched sources parse
#   6-8. GravityOCR: the pinned SGLang commit resolves, gravityocr_selfspec.patch applies, and the
#        patched plus new sources parse
#
# Needs network. Run: bash tools/verify_fetch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/train/block_gated_delta_rule"
URL="${HYBRIDDIFFUSION_URL:-https://github.com/yuchen-zhu-zyc/HybridDiffusion}"
COMMIT="$(sed -n 's/.*HYBRIDDIFFUSION_COMMIT:-\([0-9a-f]*\).*/\1/p' "$HERE/fetch_kernels.sh")"
SUBDIR="torchtitan/models/qwen3_5/model/block_gated_delta_rule"

[ -n "$COMMIT" ] || { echo "FAIL: could not read the pinned commit from fetch_kernels.sh"; exit 1; }
echo "pinned commit: $COMMIT"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "==> 1. clone and resolve the pinned commit"
git clone --quiet "$URL" "$TMP/hd" || { echo "FAIL: upstream unreachable at $URL"; exit 1; }
git -C "$TMP/hd" checkout --quiet "$COMMIT" || {
  echo "FAIL: pinned commit $COMMIT no longer resolves upstream."
  echo "      The recipe is broken for every user. Investigate before bumping the pin."
  exit 1; }

echo "==> 2. kernel subpath present"
[ -d "$TMP/hd/$SUBDIR" ] || { echo "FAIL: $SUBDIR no longer exists upstream"; exit 1; }

echo "==> 3. verify checksums of the upstream sources"
( cd "$TMP/hd/$SUBDIR" && shasum -a 256 -c "$ROOT/tools/kernel-checksums.txt" ) || {
  echo "FAIL: fetched kernel sources do not match tools/kernel-checksums.txt."
  echo "      The pin resolves but the content differs -- the upstream history may have"
  echo "      been rewritten. Do NOT update the checksums without reviewing the diff."
  exit 1; }

echo "==> 4. trillion_mods.patch applies"
cp "$TMP/hd/$SUBDIR"/*.py "$TMP/"
cp "$HERE/trillion_mods.patch" "$TMP/"
( cd "$TMP" && patch -p1 --dry-run < trillion_mods.patch >/dev/null ) || {
  echo "FAIL: trillion_mods.patch no longer applies to the pinned sources."; exit 1; }
( cd "$TMP" && patch -p1 --silent < trillion_mods.patch )

echo "==> 5. patched sources parse"
python3 -m py_compile "$TMP"/*.py || { echo "FAIL: patched sources do not compile"; exit 1; }

echo
echo "OK: fetch recipe verified end to end at $COMMIT (18 sources, patch applies, output parses)."

# ---- GravityOCR: upstream SGLang pin + gravityocr_selfspec.patch (inference/gravityocr/sglang/) ----
G="$ROOT/inference/gravityocr/sglang"
G_URL="${SGLANG_REPO:-https://github.com/sgl-project/sglang}"
G_COMMIT="$(head -c 40 "$G/BASE_COMMIT.txt")"   # as setup.sh reads it
[[ $G_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "FAIL: $G/BASE_COMMIT.txt is not a full commit SHA"; exit 1; }
echo
echo "==> 6. GravityOCR: fetch the pinned SGLang commit $G_COMMIT"
git init --quiet "$TMP/sglang"
git -C "$TMP/sglang" fetch --quiet --depth 1 "$G_URL" "$G_COMMIT" || {
  echo "FAIL: pinned SGLang commit $G_COMMIT no longer resolves at $G_URL"; exit 1; }
git -C "$TMP/sglang" checkout --quiet FETCH_HEAD

echo "==> 7. GravityOCR: gravityocr_selfspec.patch applies"
git -C "$TMP/sglang" apply --check "$G/gravityocr_selfspec.patch" || {
  echo "FAIL: gravityocr_selfspec.patch no longer applies to SGLang $G_COMMIT"; exit 1; }
git -C "$TMP/sglang" apply "$G/gravityocr_selfspec.patch"
cp -R "$G/python/." "$TMP/sglang/python/"

echo "==> 8. GravityOCR: patched and new sources parse"
g_files=()
while IFS= read -r f; do g_files+=("$f"); done < <(
  sed -n 's#^+++ b/\(.*\.py\)$#\1#p' "$G/gravityocr_selfspec.patch"; cd "$G" && find python -name '*.py')
( cd "$TMP/sglang" && python3 -m py_compile "${g_files[@]}" ) || { echo "FAIL: patched GravityOCR sources do not compile"; exit 1; }
echo
echo "OK: GravityOCR recipe verified at SGLang $G_COMMIT (patch applies, ${#g_files[@]} sources parse)."
