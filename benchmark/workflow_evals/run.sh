#!/usr/bin/env bash
# Select the pinned data; all execution, checkpointing and scoring remain upstream.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$HERE/.upstream"
if [[ $# -lt 1 || "$1" == "--help" ]]; then
    echo "Usage: $0 <workflow> --model trida:<checkpoint-name> --base-url <server-root> [upstream options]"
    echo "Workflows: invoice_processing, customer_service, agent_trace_observability, security_incidents"
    exit 0
fi
[[ -x "$UPSTREAM/.venv/bin/python" ]] || { echo "Run $HERE/setup.sh first." >&2; exit 1; }
"$HERE/setup.sh" --check
WORKFLOW="$1"
shift
# Keep the data fixed. Update dataset-revisions.json explicitly to change it.
ARGS=("$@")
MODEL=""
while [[ $# -gt 0 ]]; do
    ARG="$1"
    if [[ "$ARG" == --model ]]; then
        [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 1; }
        MODEL="$2"
        shift
    elif [[ "$ARG" == --model=* ]]; then
        MODEL="${ARG#--model=}"
    fi
    if [[ "$ARG" == --dataset-revision || "$ARG" == --dataset-revision=* ]]; then
        echo "Dataset revisions are pinned in dataset-revisions.json." >&2
        exit 1
    fi
    shift
done
[[ "$MODEL" == trida:?* ]] || { echo "Specify --model trida:<checkpoint-name>." >&2; exit 1; }
REVISION="$("$UPSTREAM/.venv/bin/python" - "$HERE/dataset-revisions.json" "$WORKFLOW" "$UPSTREAM" <<'PY'
import json, sys
pins = json.load(open(sys.argv[1]))
if sys.argv[2] not in pins:
    raise SystemExit(f"Unknown workflow: {sys.argv[2]}")
sys.path.insert(0, sys.argv[3])
from core.dataset import DATASETS
pin = pins[sys.argv[2]]
if pin["repo"] != DATASETS[sys.argv[2]]:
    raise SystemExit("Pinned dataset repository differs from upstream")
if pin.get("license") != "Apache-2.0":
    raise SystemExit(f"License not verified for {pin['repo']}; see docs/oss/intake/0009-workflowevals.md")
print(pin["revision"])
PY
)"
cd "$UPSTREAM"
exec uv run --locked python run.py "$WORKFLOW" --dataset-revision "$REVISION" "${ARGS[@]}"
