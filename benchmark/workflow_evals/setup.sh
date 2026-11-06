#!/usr/bin/env bash
# Fetch the exact benchmark revision and apply only the Trida provider patch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$HERE/.upstream"
REVISION="$(cat "$HERE/upstream-revision")"
PATCH="$HERE/patches/trida-provider.patch"

if [[ "${1:-}" == "--check" && ! -d "$UPSTREAM/.git" ]]; then
    echo "Run $HERE/setup.sh first." >&2
    exit 1
fi
if [[ ! -e "$UPSTREAM" ]]; then
    git clone --no-checkout https://github.com/typesafe-ai/WorkflowEvals.git "$UPSTREAM"
    git -C "$UPSTREAM" checkout --detach "$REVISION"
fi
[[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$REVISION" ]] || {
    echo "Unexpected upstream revision; preserve any work and remove .upstream before setup." >&2
    exit 1
}
git -C "$UPSTREAM" diff --quiet || {
    echo "Upstream has unstaged edits; refusing to overwrite them." >&2
    exit 1
}
if git -C "$UPSTREAM" diff --cached --quiet; then
    [[ "${1:-}" != "--check" ]] || { echo "Provider patch is missing; run setup.sh." >&2; exit 1; }
    git -C "$UPSTREAM" apply --check --index "$PATCH"
    git -C "$UPSTREAM" apply --index "$PATCH"
else
    # --index records added files too, allowing an exact check on subsequent setup runs.
    git -C "$UPSTREAM" diff --cached --binary | cmp -s "$PATCH" - || {
        echo "Upstream differs from the expected provider patch; refusing to continue." >&2
        exit 1
    }
fi
[[ "${1:-}" != "--check" ]] || exit 0
uv sync --locked --project "$UPSTREAM"
echo "Ready: $HERE/run.sh <workflow> --model trida:<checkpoint-name> --base-url <server-root>"
