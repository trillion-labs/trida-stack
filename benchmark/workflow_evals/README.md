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

## Dataset licenses

The invoice dataset is Apache-2.0. The other three have no verified license
declaration. They require `--allow-unverified-datasets` for local evaluation. This
opt-in is a proposal awaiting maintainer approval, not an approved license exception.
It does not verify their licenses or authorize redistribution. Keep downloaded
data outside Git. See the
[intake record](../../docs/oss/intake/0011-workflowevals.md).

## Run

Start a [decision server](../../inference/zoro-1/README.md#decisions) exposing `/v1/decide`, then:

```bash
# Try one case first. Use --dry-run to check data without inference.
benchmark/workflow_evals/run.sh invoice_processing \
  --model trida:zoro-1-bf16 --base-url http://127.0.0.1:30001 \
  --limit 1 --workers 1 --name zoro-smoke

# Full invoice workflow.
benchmark/workflow_evals/run.sh invoice_processing \
  --model trida:zoro-1-bf16 --base-url http://127.0.0.1:30001 \
  --workers 1 --name zoro-full
```

For local evaluation of customer service, security incidents, or agent traces:

```bash
benchmark/workflow_evals/run.sh customer_service \
  --allow-unverified-datasets \
  --model trida:zoro-1-bf16 --base-url http://127.0.0.1:30001 \
  --limit 1 --workers 1 --name zoro-local
```

The flag is required on each invocation, including dry runs and resumes. Dataset
pins and license metadata stay unchanged; setup and CI do not download these datasets.

Pass the server root URL, without `/v1/decide`. Use the actual model and precision
in the model name. Direct decisions and grouped questions are the defaults.
To reason before deciding, add `--reasoning --max-output-tokens 2048` and use a
separate run name, such as `zoro-reasoning`. The token limit caps reasoning generation;
the Trida adapter default is 2,048 tokens, with an 8,192-token ceiling. The CUDA
server permits up to 4,096 by default; larger requests require the opt-in below. It checks
prompt + budget against its context length. Increase `--timeout` for slower runs.
Unfinished, empty, or state-trimmed reasoning is recorded as a failure, even if the
server returns direct fallback answers. Only complete, untrimmed reasoning is scored.

For invoice reasoning, start the CUDA decision server with a **32,768-token context**
(from the repository root, in its SGLang environment):

```bash
python inference/zoro-1/serve_decide.py /path/to/decision-checkpoint \
  --port 30001 --context-length 32768
```

This is a server flag, not a `run.sh` flag. The H100 pilot's `ap_00002` reasoning
prompt used 21,306 tokens; adding the 2,048-token budget exceeds the default 16,384.
32,768 accommodates that observed prompt and budget. On the H100 pilot, no invoice
case completed reasoning within 8,192 tokens; see [VALIDATION.md](VALIDATION.md).
Other inputs may need more room.
The full benchmark inputs are retained. This setting does **not** raise the default
decision-encoder limits described below.

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

These are the default encoder limits. Increasing the server's `--context-length`
alone does not remove them. Two invoice cases failed on this limit in the BF16 pilot.

Reasoning also shares this state budget. Both servers keep the conclusion when
only a reasoning tail fits and report `reasoning_trimmed: true`. If no reasoning
fits, they return direct answers with `cut: "state_limit"`. This benchmark rejects
both cases in reasoning mode; a normal generation stop alone is insufficient.

For experimental longer-input CUDA evaluation, explicitly raise both decision
limits and the reasoning ceiling:

```bash
python inference/zoro-1/serve_decide.py /path/to/decision-checkpoint \
  --port 30001 --context-length 32768 --reasoning-token-limit 8192 \
  --decision-max-state 15744 --decision-max-row 16384
```

Use benchmark options `--reasoning --max-output-tokens 8192 --questions one`.
Both state and row limits must fit the server context, and the state cap must
leave at least 640 tokens for a question. Larger questions may need a wider gap. The same limits apply when
fitting generated reasoning, and all strict completion/trimming checks stay on.
These options change serving limits, not benchmark inputs. Longer decision inputs
extend beyond the default training-format limits; successful execution alone does
not establish their accuracy or calibration. Full-length coverage still requires
model validation and may need larger limits for other cases.

See [validation notes](VALIDATION.md) for the pilot evidence and review test results.

## Maintenance

CI runs the adapter tests in a separate Python 3.13 job, with the pinned upstream
checkout and locked dependencies. Tests require no model, dataset download, or API key.


```bash
# Offline adapter tests, after setup.
benchmark/workflow_evals/.upstream/.venv/bin/python benchmark/workflow_evals/test_provider.py
```

To update upstream, change `upstream-revision`, review the provider patch, and update
`dataset-revisions.json` as needed. Preserve results before recreating `.upstream/`.
Upstream code is Apache-2.0; datasets have separate licenses.
