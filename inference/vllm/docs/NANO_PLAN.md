# Plan: bring `inference/vllm/` to the same shape as `inference/sglang/`

Written 2026-09-14. The sglang side was already reduced to a nanoGPT-style recipe:
three files, one README you can read in a sitting, no history, nothing that only runs
on our cluster. The vllm side is still the working tree it grew up as: 36 files,
~5,000 lines, six design docs from stages that have shipped or died, twenty scripts
hard-wired to `/path/to`, a serve script that launches the parked mode, and a plugin
whose behaviour is steered by 18 environment variables.

The goal is not to shrink for its own sake. It is to make the vllm backend readable by
someone who has never seen it, to make the one runnable command run the thing we
actually ship, and to stop the repo describing a result we superseded.

## What "nano" means here, concretely

Measured against the sglang folder and against nanoGPT itself:

- **One README, recipe-shaped.** What it is (one paragraph), install, serve, what each
  mode does, known limits, provenance. No history, no dated results tables, no
  roadmap. Numbers appear once, as facts, with the checkpoint they came from.
- **Only the shipped path in the code.** Self-spec (AR-Trust) with the spec-shape
  layout and full cuda graphs is what we serve and what we measured. Pure-diffusion
  denoising is parked and lives in the sglang patch; debug dumps, timers and trace
  hooks were for us, not for a reader.
- **Configuration is a config, not the environment.** Eighteen `TRIDA_*` variables read
  from `os.environ` inside a model file is the opposite of readable. nanoGPT has a
  dataclass. So should we.
- **Files a person can hold in their head.** A 1,950-line model file is not that.
  Three to five files, each with one job, each short enough to read top to bottom.
- **Nothing that only runs on our cluster.** If a reader can't run it, it teaches them
  to distrust the rest.

## Current inventory and disposition

| path | files | lines | disposition |
|---|---|---|---|
| `vllm_native_diffusion/` (the plugin) | 5 | ~3,360 | **keep, slim, split** (section below, pending the code map) |
| `README.md` | 1 | 111 | **rewrite** to recipe shape; absorbs the one-paragraph version of DESIGN.md |
| `DESIGN.md` | 1 | 110 | **fold into README** (one section: "how it fits vLLM"), then delete. Fix the self-conditioning claim while doing it. |
| `serve_diffusion.sh` | 1 | 53 | **replace**: currently serves the parked diffusion-only mode with a 12-line comment saying full graphs don't work |
| `pyproject.toml` | 1 | 17 | keep; fix the three-names problem |
| `docs/WORKLOG.md` | 1 | 443 | **move out of the shipped tree** — it's our lab notebook, not documentation |
| `docs/NOTES.md` | 1 | 406 | **delete** — a snapshot of a milestone WORKLOG already covers |
| `docs/PERF_LEVERS.md`, `PROFILING_FINDINGS.md`, `FULL_CUDAGRAPH_DESIGN.md`, `SELFSPEC_VLLM_PLAN.md` | 4 | ~500 | **delete from the shipped tree**; they document decisions already taken. Anything still true becomes one sentence in the README's "known limits" or the plugin docstring. |
| `tools/comprehensive_run/` | 14 | ~1,400 | **move out** — sweeps, report builder, cluster job scripts. Three pieces have durable value (`sweep_client.py`, `eval_gsm8k_greedy.py`, `build_report_html.py`) and belong in `benchmark/`, parameterised, not here. |
| `tools/agentic_eval/` | 6 | ~600 | **move out**, same reasoning; the committed `summary_fulleval_*.json` result file goes with it |

Net: 36 files → about 8. Roughly 5,000 lines → roughly 1,500, most of it the two kernel files.

Where "move out" goes: a sibling directory outside the shipped tree. Two options,
decide once:

1. `internal/` at the repo root, gitignored, mirrored to the cluster the way
   `luke/scripts/lib/` already is. Simplest; keeps history in git via the move.
2. A separate `trida-lab` repo. Cleaner separation; costs a second remote.

Recommendation: option 1 now, promote to option 2 only if a second person starts
working in it.

## The plugin itself

A line-by-line map of the three source files, classified by whether the shipped
self-spec path needs each block, gives the real picture.

### `qwen3_5_diffusion.py`, 1,947 lines

| what the lines do | ≈ lines | share |
|---|---|---|
| debug: dumps, timers, trace hooks, counters | 490 | 25% |
| shared plumbing: canvas state, prefill seeding, model-state interface | 539 | 28% |
| pure-diffusion denoise loop and its GDN snapshot/restore | 364 | 19% |
| self-spec: verify/accept step, GDN two-stream layer, ring buffers, vLLM shims | 355 | 18% |
| vLLM interface glue | 98 | 5% |
| dead: alternate paths behind flags we never flip | 65 | 3% |

A quarter of the file is instrumentation for us. Another fifth is the parked mode.
The shipped path is 18% of the file, and much of the "shared" plumbing exists to
serve the diffusion lifecycle. Realistic target after the cut: **500 to 650 lines of
Python** plus one Triton kernel.

### `block_causal_readout.py`, 1,172 lines

The header says "ported verbatim from SGLang". That's true of lines 27–654, about
630 lines, and **the shipped path never calls any of them**. They're reachable only
with a flag that has never been set. Lines 657–1020 are the denoise kernel the
parked mode uses. The self-spec path touches lines 1031–1172, about 145 lines, 12%
of the file, and those were written here, not ported. The file shrinks to that
kernel plus the attribution it actually deserves.

### The 18 environment knobs

Of the 18 `TRIDA_*` variables the plugin reads, **exactly one is ever set to a
non-default value in production: `TRIDA_SELFSPEC_N=4`.** Thirteen are debug switches
that are always off. Three exist to reach dead alternate paths. One is a
diffusion-only kernel toggle the self-spec branch never reads. So a config object
with a single field, `spec_n`, and two derived properties, `canvas_length = 2N − 1`
and `draft_len = 2N − 2`, reproduces the shipped behaviour exactly. The three
`SGLANG_*` knobs in the kernel file live in a hard-coded branch for the N=4 shape
that the shipped path doesn't reach either.

### The split

Four files plus the entry point, shipped path only:

| file | ≈ lines | job |
|---|---|---|
| `config.py` | 90 | `SelfSpecConfig` dataclass; the `2N − 2` invariant as a validated property; the vLLM arch-config hook |
| `state.py` | 200 | canvas and per-request self-spec state; GDN layer discovery and mamba-group mapping; ring, window and preconv buffers and their per-step index fill |
| `gdn_layer.py` | 230 | the `_forward_core` patch reduced to self-spec-or-stock, plus the fused Triton layer (~150 of the 230) |
| `sampler.py` | 230 | prefill seeding, the verify/accept/re-draft step, `SamplerOutput` assembly |
| `plugin.py` | 150 | entry point and the three vLLM shims: fake `speculative_config` during builder init, `Scheduler.num_sampled_tokens_per_step`, the `get_uniform_token_count` guard |

Roughly 900 lines including Triton, 650 without. One CPU test under pytest.

Two couplings make the split more than a file move, and the rewrite should fix
rather than preserve them:

- A module-global dictionary acts as a mailbox between `prepare_attn`, the patched
  GDN forward, and the sampler, because the monkeypatch has no reference to the
  model state. It also carries a per-forward layer counter that silently depends on
  layers running in registration order and being reset exactly once. Replace it
  with one context object attached to the patched class.
- The sampler rolls the GDN conv window that the kernel reads on the next step, and
  flips a parity bit whose partner is an index formula in `prepare_attn`. Nothing
  asserts the two stay in lockstep. Put both sides in `state.py` behind one method.

### Do not break

Load-bearing details that a rewrite could silently lose. Each produces wrong output
rather than a crash:

- Two separate `+1`s turn the config's `2N − 2` into the runtime `2N − 1`, on two
  different objects. One derived property, one place.
- vLLM's conv kernel treats state index 0 as a null block and silently skips that
  sequence; every conv-window index is therefore `slot + 1`. Documented once, used
  in four places.
- The self-spec buffers carry extra dummy rows for padded requests under full
  cuda-graph replay, and must be persistent tensors updated by copy, not
  reallocated, or replay reads stale pointers.
- `num_rejected` must be reported as 0 on prefill; reporting 1 rolled the runner
  back a token and corrupted the first step.
- `num_accepted` is clamped to at least 1, or the next step's metadata builder
  misclassifies it.
- A prompt of exactly `2N − 1` tokens looks like a uniform self-spec decode; the
  guard that every scheduled request carries drafts is what stops the decode graph
  replaying over prompt tokens. Removing it produces garbage only at specific
  prompt lengths, which is how it went unnoticed before.
- The kernel's l2norm epsilon, scale, softplus threshold and bf16 round-trips exist
  to bit-match the reference. Any change breaks losslessness silently.
- Mixed batches of a prompt chunk plus canvases fall back to the stock kernel and
  warn once. That is a known correctness hole, not a performance fallback.

Three large comment blocks, including the module docstring, still describe the
diffusion design and call piecewise graphs "the shipped mode". None of that carries
forward.

## Entry point and configuration

Today the two backends have different front doors:

- `inference/serve.py` knows only SGLang. Its `MODES` table maps a mode to an SGLang
  `--dllm-algorithm` id plus a YAML from `configs/`.
- The vllm backend is reached by a separate shell script that takes environment
  variables and builds JSON `--hf-overrides`.
- `configs/trida_self_spec_b7_g4.yaml` carries SGLang-only keys (`variant`,
  `use_spec_verify`, `draft_mode`); the vllm path never reads it.

For the two folders to be genuinely parallel, `serve.py` grows `--backend {sglang,vllm}`
and the vllm branch reads the same YAML and translates it into vLLM's flags. That
means the config schema becomes backend-neutral: `N` (gen block size), `mask_id`,
`threshold`, `max_denoising_steps`, plus a per-backend section for anything that
genuinely differs. `serve_diffusion.sh` then becomes a three-line convenience that
calls `serve.py --backend vllm`, or disappears.

The shipped self-spec command the vllm branch must reproduce (this is what every
measurement this week used):

```
TRIDA_SELFSPEC_N=4 vllm serve <ckpt> \
  --hf-overrides '{"architectures":["Qwen3_5ForBlockDiffusion"],"canvas_length":6,"mask_id":248077,"confidence_threshold":0.90}' \
  --diffusion-config '{"canvas_length":6,"max_denoising_steps":8}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --trust-remote-code
```

with the rule `canvas_length = 2N − 2` (vLLM adds the bonus token). That rule is the
kind of thing a config object exists to encode once.

## Docs: what the README says

Structure mirrors `inference/sglang/README.md`:

1. What this is — three sentences. A stock vLLM 0.27 plugin; no fork, no SGLang;
   same checkpoint as the sglang path.
2. How it fits vLLM — one short section replacing DESIGN.md: a ModelState plugin,
   what vLLM provides, what the plugin adds (the two-stream GDN readout and the
   self-spec sampler), what it deliberately does not do (self-conditioning).
3. Install — two commands.
4. Serve — the self-spec command above, with the `2N − 2` rule stated once.
5. Modes — causal / self-spec, one line each, with the measured numbers for the
   checkpoint they came from (220 vs 351 tok/s at C=1; GSM8K 81.3 vs 81.7).
6. Known limits — the current list is good and stays: no prefix caching, mixed
   prompt+canvas batches crash at N=8 for C≥8, and so on.
7. Test — one command.
8. Provenance — the kernel is a port of the SGLang block-causal kernel; attribute
   by handle, not by first name.

Voice: the register of `eval.py`'s docstring and the plugin's env-knob comment
block. Bold two things per page, not twenty-eight.

## Things the review found that this plan fixes for free

- `--model` documented but positional: goes away when the README is rewritten
  against the real argparse.
- Front-page README advertising 32 tok/s and "FULL not coherent": replaced by the
  self-spec numbers.
- Four names for one thing: pyproject `name`, package dir and entry point align.
  (Package rename deferred until a node is free to verify the plugin loads; the
  plan reserves the name.)
- `SGLANG_*`-prefixed knobs inside the vLLM backend: absorbed into the config object.
- The design doc claiming self-conditioning: the section that replaces it says the
  opposite, which is what the code does.
- Two test runners, two GSM8K harnesses: one test under pytest; the duplicate
  evaluator leaves with `tools/`.
- Internal names and host paths in shipped source: leave with `docs/` and `tools/`,
  and are scrubbed from the kernel header during the split.

## Order of work and safety

The cluster deploys from this tree (`luke/code/vllm-native-dev9` is an rsync of
`inference/vllm`, and the venv has the package installed). Nothing is running on the
plugin while hyungguk's training holds both nodes, which is the right window, but
the sequence still matters:

1. **Tag the current tree** (`vllm-plugin-pre-nano`) so every number in the
   worklog stays reproducible against the code that produced it.
2. **Move, don't delete, first.** `docs/` and `tools/` relocate to `internal/` in
   one commit. No behaviour change; the shipped tree is immediately half its size.
3. **Rewrite the README and replace the serve script** against the shipped command.
   Still no code change. Run the smoke test and one 30-item self-spec eval to confirm
   the documented command is the measured one.
4. **Slim the plugin** in three commits: remove debug branches; remove the parked
   mode and dead flag paths; introduce the config object and delete the env reads.
   After each: the CPU test, then the 30-item eval, and the result must be
   byte-identical to the tagged baseline's token stream, not just equal accuracy.
   We have the identity-check tooling from the pilot evals.
5. **Split the model file** along the seams the map identifies. Pure move; same
   identity check.
6. **Unify the entry point**: `serve.py --backend vllm`, backend-neutral YAML.
   Update the sglang README's `serve.py` lines at the same time so both folders
   describe the same front door.
7. **Fix the install, then rename the package.** Checked on 2026-09-14: the cluster
   venv's editable install of the plugin points at `luke/mv3/…`, a tree that no
   longer exists, so `import vllm_native_diffusion` fails inside the venv. The only
   reason serving works is that the launcher script prepends the source directory to
   `PYTHONPATH`. Anyone running `vllm serve` without that script gets a broken plugin.
   So: uninstall the dangling editable, install once from the canonical tree, drop the
   `PYTHONPATH` line from the launcher, rename the package to match the entry point,
   and confirm `VLLM_PLUGINS=trida_diffusion` loads with nothing on `PYTHONPATH`. Last,
   because it is the only step that touches the deployed environment.

The shipped launcher also pins three things the README's serve section must carry,
because the README snippet currently omits them: `--max-num-seqs 1`,
`--gpu-memory-utilization 0.55`, and a CUDA compat directory on `LD_LIBRARY_PATH` for
the 570-series driver. The self-spec values (`TRIDA_SELFSPEC_N=4`, canvas 6,
threshold 0.90) are passed in by callers; the launcher's own defaults are the parked
diffusion mode's.

Each step is independently shippable and independently revertible. Steps 1–3 are a
day and carry no correctness risk. Steps 4–5 are the real work, probably two to
three days including the identity checks. Steps 6–7 are half a day each.

## Acceptance

- A reader who has never seen the repo can install, serve self-spec, and run the
  test from the README alone, and the first command they copy works.
- `inference/vllm/` is eight files or fewer, none of which reference `/path/to`,
  a hostname, or a colleague by first name.
- The served token stream for the 30-item GSM8K greedy set is byte-identical to the
  `vllm-plugin-pre-nano` tag at N=4.
- Tokens per second at C=1 on the same node is within 3% of the tag (351 tok/s).
- `grep -c 'os.environ' vllm_native_diffusion/*.py` returns zero.
