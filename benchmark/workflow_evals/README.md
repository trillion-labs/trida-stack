# WorkflowEvals

Run [TypeSafe's original workflow benchmark](https://github.com/typesafe-ai/WorkflowEvals)
against our decision server. It covers invoices, customer service, agent traces, and
security incidents. Questions, policies, saved reference answers, and scoring stay
unchanged. No LLM judge or reference-generation calls are needed.

## Setup

Requires Git and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
benchmark/workflow_evals/setup.sh
```

This creates an isolated environment in the ignored `.upstream/` directory.
Upstream code and datasets are pinned. The patch only adds our model connection;
setup and each run reject unexpected changes to the upstream checkout.

## Run

Start a [decision server](../../inference/README.md#decisions) exposing `/v1/decide`, then:

```bash
# Try one case first. Use --dry-run to check data without inference.
benchmark/workflow_evals/run.sh customer_service \
  --model trida:zoro-1-bf16 --base-url http://127.0.0.1:30001 \
  --limit 1 --workers 1 --name zoro-smoke

# Run all four workflows.
for workflow in invoice_processing customer_service agent_trace_observability security_incidents; do
  benchmark/workflow_evals/run.sh "$workflow" \
    --model trida:zoro-1-bf16 --base-url http://127.0.0.1:30001 \
    --workers 1 --name zoro-full
done
```

Pass the server root URL, without `/v1/decide`. Use the actual model and precision
in the model name. Direct decisions and grouped questions are the defaults.
To reason before deciding, add `--reasoning --max-output-tokens 2048` and use a
separate run name, such as `zoro-reasoning`. The token limit caps reasoning generation;
without it, upstream defaults to 64,000. Increase `--timeout` for slower runs.

Repeat a command to resume; use a new run name when changing settings.
Set `TRIDA_API_KEY` only if your server requires authentication.

## Results

Find outputs in `.upstream/runs/<workflow>/<name>/`:

- `results.json`: requests, responses, final actions, and run configuration.
- `scores.json`: reference agreement, counts, failures, timing, and estimated cost.

**Exact agreement** requires the full action set, including arguments, to match.
**Primary agreement** compares the main action where available. Agent traces report
primary agreement only; customer service reports exact agreement only. References are
LLM-generated, so agreement is not human-verified accuracy. Prediction errors count
as wrong; check coverage and failure counts alongside scores. Self-hosted cost is
unknown unless pricing is supplied.

Record the checkpoint, server revision, hardware, and precision when sharing scores.

## Context-length limitation

**The current server cannot faithfully evaluate every full-length benchmark input.**
The adapter sends inputs unchanged, but the decision encoder:

- Truncates state text beyond **7,551 tokens**.
- Rejects rows over **8,192 tokens**, including the state, question, answer options,
  and delimiter tokens.

These are hardcoded encoder limits. Increasing the server's `--context-length`
alone does not remove them. Two invoice cases failed on this limit in the BF16 pilot.

Full-length evaluation requires a separate encoder/context change and
validation with the model.

## Maintenance

```bash
# Offline adapter tests, after setup.
benchmark/workflow_evals/.upstream/.venv/bin/python benchmark/workflow_evals/test_provider.py
```

To update upstream, change `upstream-revision`, review the provider patch, and update
`dataset-revisions.json` as needed. Preserve results before recreating `.upstream/`.
Upstream code is Apache-2.0; datasets have separate licenses.
