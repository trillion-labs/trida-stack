#!/usr/bin/env bash
# Select the pinned data; all execution, checkpointing and scoring remain upstream.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$HERE/.upstream"
if [[ $# -lt 1 || "$1" == "--help" ]]; then
    echo "Usage: $0 <workflow> --model trida:<checkpoint-name> --base-url <server-root> [upstream options]"
    echo "Use --allow-unverified-datasets to opt in to local evaluation of datasets with unverified licenses."
    echo "Workflows: invoice_processing, customer_service, agent_trace_observability, security_incidents"
    echo "For invoice --reasoning, start the decision server with --context-length 32768 (a server flag, not a run.sh flag)."
    echo "This accommodates the observed reasoning prompt; fixed decision-encoder limits still apply."
    exit 0
fi
[[ -x "$UPSTREAM/.venv/bin/python" ]] || { echo "Run $HERE/setup.sh first." >&2; exit 1; }
"$HERE/setup.sh" --check
WORKFLOW="$1"
shift
# Keep the data fixed. Update dataset-revisions.json explicitly to change it.
ARGS=()
MODEL=""
ALLOW_UNVERIFIED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-unverified-datasets)
            ALLOW_UNVERIFIED=1
            shift
            continue
            ;;
        --model)
            [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 1; }
            MODEL="$2"
            ARGS+=("$1" "$2")
            shift 2
            continue
            ;;
        --model=*) MODEL="${1#--model=}" ;;
        --dataset-revision|--dataset-revision=*)
            echo "Dataset revisions are pinned in dataset-revisions.json." >&2
            exit 1
            ;;
    esac
    ARGS+=("$1")
    shift
done
[[ "$MODEL" == trida:?* ]] || { echo "Specify --model trida:<checkpoint-name>." >&2; exit 1; }
REVISION="$("$UPSTREAM/.venv/bin/python" - "$HERE/dataset-revisions.json" "$WORKFLOW" "$UPSTREAM" "$ALLOW_UNVERIFIED" <<'PY'
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
    if sys.argv[4] != "1":
        raise SystemExit(f"License not verified for {pin['repo']}; use --allow-unverified-datasets "
                         "to opt in to local evaluation only")
    print(f"Warning: license unverified for {pin['repo']}; local evaluation opt-in enabled. "
          "Dataset contents must remain outside the repository.", file=sys.stderr)
print(pin["revision"])
PY
)"
cd "$UPSTREAM"
exec uv run --locked python run.py "$WORKFLOW" --dataset-revision "$REVISION" "${ARGS[@]}"
