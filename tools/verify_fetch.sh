#!/usr/bin/env bash
# Verify that the third-party fetch recipes still work.
#
# The recipes in train/block_gated_delta_rule/, inference/zoro-1/sglang/ and
# inference/zoro-1/vllm/vllm_native_diffusion/ ARE the install path for both two-stream
# code paths, and inference/gravityocr/sglang/setup.sh is the install path for the
# GravityOCR server (upstream SGLang at a pinned commit plus our patch). If upstream moves,
# renames, force-pushes, or the subpath changes, nobody can build -- and we would normally
# find out from a bug report.
#
# This checks, in order:
#   1. the pinned commit still resolves upstream
#   2. the kernel subpath still exists
#   3. the fetched files match tools/kernel-checksums.txt  (integrity, not just reachability)
#   4. trillion_mods.patch still applies cleanly
#   5. the patched sources parse
# and for the GravityOCR SGLang recipe (inference/gravityocr/sglang/):
#   6. the upstream tag named in BASE_COMMIT.txt still points at the pinned commit
#   7. exactly the pinned commit can be fetched
#   8. gravityocr_selfspec.patch applies to it
#   9. the new files shipped next to the patch do not exist upstream (so they add, never replace)
#  10. the patched and the new sources parse
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

# ---- GravityOCR: upstream SGLang at BASE_COMMIT.txt + gravityocr_selfspec.patch ------------------
# The same fetch and patch steps as inference/gravityocr/sglang/setup.sh, without the GPU build.
GO="$ROOT/inference/gravityocr/sglang"
SGLANG_URL="${SGLANG_URL:-${SGLANG_REPO:-https://github.com/sgl-project/sglang}}"   # SGLANG_REPO: as setup.sh names it
SGLANG_BASE="$(head -c 40 "$GO/BASE_COMMIT.txt")"
SGLANG_TAG="$(sed -n 's/^[0-9a-f]* .* tag \(v[0-9][0-9.]*\) .*/\1/p' "$GO/BASE_COMMIT.txt")"
[[ $SGLANG_BASE =~ ^[0-9a-f]{40}$ ]] || { echo "FAIL: $GO/BASE_COMMIT.txt does not start with a 40-hex commit"; exit 1; }
[ -n "$SGLANG_TAG" ] || { echo "FAIL: $GO/BASE_COMMIT.txt names no upstream tag"; exit 1; }
echo
echo "pinned SGLang commit: $SGLANG_BASE (tag $SGLANG_TAG)"

echo "==> 6. the upstream tag still points at the pinned commit"
TAGS="$(git ls-remote --tags "$SGLANG_URL" "refs/tags/$SGLANG_TAG" "refs/tags/$SGLANG_TAG^{}")" || {
  echo "FAIL: upstream unreachable at $SGLANG_URL"; exit 1; }
TAG_SHA="$(awk -v t="refs/tags/$SGLANG_TAG" '$2 == t "^{}" {p = $1} $2 == t {d = $1} END {print (p ? p : d)}' <<<"$TAGS")"
[ "$TAG_SHA" = "$SGLANG_BASE" ] || {
  echo "FAIL: upstream $SGLANG_TAG now resolves to '${TAG_SHA:-nothing}', not the pinned $SGLANG_BASE."
  echo "      The recipe fetches by commit, so it may still build, but every record that says"
  echo "      '$SGLANG_TAG' is now wrong. Investigate before touching the pin."
  exit 1; }

echo "==> 7. fetch exactly the pinned commit"
git init --quiet "$TMP/sglang"
git -C "$TMP/sglang" fetch --quiet --depth 1 "$SGLANG_URL" "$SGLANG_BASE" || {
  echo "FAIL: pinned SGLang commit $SGLANG_BASE no longer resolves upstream."
  echo "      inference/gravityocr/sglang/setup.sh is broken for every user. Investigate before bumping the pin."
  exit 1; }
git -C "$TMP/sglang" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD
[ "$(git -C "$TMP/sglang" rev-parse HEAD)" = "$SGLANG_BASE" ] || {
  echo "FAIL: the fetched commit is not $SGLANG_BASE"; exit 1; }

echo "==> 8. gravityocr_selfspec.patch applies"
git -C "$TMP/sglang" apply --check "$GO/gravityocr_selfspec.patch" || {
  echo "FAIL: gravityocr_selfspec.patch no longer applies to SGLang at $SGLANG_BASE."; exit 1; }
git -C "$TMP/sglang" apply "$GO/gravityocr_selfspec.patch"

echo "==> 9. the new files are new upstream"
mapfile -t NEW_FILES < <(cd "$GO" && find python -type f -name '*.py' | sort)
[ "${#NEW_FILES[@]}" -gt 0 ] || { echo "FAIL: no new SGLang files under $GO/python"; exit 1; }
for f in "${NEW_FILES[@]}"; do
  if git -C "$TMP/sglang" cat-file -e "HEAD:$f" 2>/dev/null; then
    echo "FAIL: $f exists upstream at $SGLANG_BASE; the copy in $GO would replace it"; exit 1; fi
  install -D -m 644 "$GO/$f" "$TMP/sglang/$f"
done

echo "==> 10. patched and new sources parse"
mapfile -t PATCHED < <(sed -n 's#^diff --git a/\([^ ]*\.py\) b/.*#\1#p' "$GO/gravityocr_selfspec.patch")
( cd "$TMP/sglang" && python3 -m py_compile "${PATCHED[@]}" "${NEW_FILES[@]}" ) || {
  echo "FAIL: the patched SGLang sources do not compile"; exit 1; }

echo
echo "OK: GravityOCR SGLang recipe verified at $SGLANG_BASE ($SGLANG_TAG): the patch applies," \
     "${#PATCHED[@]} patched and ${#NEW_FILES[@]} new Python files parse."
