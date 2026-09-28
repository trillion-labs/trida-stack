# Contributing to trida-stack

Thanks for your interest in `trida-stack` — the training and serving stack for
Trillion Labs' **Trida** two-stream block-diffusion language models. This guide
covers setting up a dev environment, our lint/format/test conventions, and how to
open a pull request.

By contributing you agree that your contributions are licensed under the project's
[Apache License 2.0](LICENSE).

---

## Before you start: the noncommercial kernel caveat

The two-stream Gated-DeltaNet + block-diffusion Triton kernels are **not part of
this repository**. They are **PolyForm Noncommercial 1.0.0** upstream code that is
*fetched and patched via recipe* at setup time (see
[`COMPLIANCE.md`](COMPLIANCE.md) §1b and [`NOTICE`](NOTICE)):

- `train/block_gated_delta_rule/fetch_kernels.sh` + `trillion_mods.patch`
- `inference/sglang/two_stream_diffusion.patch`
- `inference/vllm/vllm_native_diffusion/KERNELS.md`

**The two-stream training and diffusion-serving paths are noncommercial-only.**
Please do not paste, vendor, or otherwise redistribute those kernels (or Trillion's
patches against them) into this Apache-2.0 repository in a PR — patches must stay as
fetch/patch recipes. The rest of the repo (HF/AR model code, data/eval/serving glue,
benchmark wrappers) is Apache-2.0 and contributions there are unrestricted.

---

## First: enable the push guard

This clone talks to the **internal** repository. The public mirror is published
deliberately, never by pushing from here — and a stale `origin` left over from the
repository rename would publish internal work. Turn the guard on once per clone:

```bash
git config core.hooksPath .githooks
git remote -v   # should say trida-stack-internal
```

`.githooks/pre-push` then refuses any push to a remote that is not
`trida-stack-internal`. If a push really is intended, it takes an explicit override:
`TRIDA_ALLOW_PUBLIC_PUSH=1 git push ...`.

## Development environment

Requires **Python 3.10+**.

```bash
# 1. Clone
git clone https://github.com/trillion-labs/trida-stack.git
cd trida-stack

# 2. Virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Install the package (editable) + dev tooling
pip install -e ".[dev]"    # dev extra = pytest, ruff, pre-commit

# 4. (optional) inference/eval client deps
pip install -r inference/requirements.txt
```

> **Note on `torch`:** `requirements.txt` pins the CUDA build to match the
> reference cluster. On a CPU-only dev box (or CI), install a CPU wheel first
> (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) — the
> CPU-safe tests only need `torch` on CPU.

### Fetching the two-stream kernels (only if you touch those paths)

The two-stream training / diffusion-serving paths need the PolyForm-NC kernels
fetched into place first — **for noncommercial use only**:

```bash
bash train/block_gated_delta_rule/fetch_kernels.sh
```

For the SGLang / vLLM diffusion backends, follow the recipes referenced above.
You do **not** need these to work on the Apache-2.0 parts of the codebase.

---

## Lint and format (ruff)

We use [ruff](https://docs.astral.sh/ruff/) (pinned to `0.9.10`, same as pre-commit),
configured in [`pyproject.toml`](pyproject.toml).

```bash
ruff check .          # lint (what CI runs)
ruff check . --fix    # lint + autofix
```

The rule set is intentionally narrow for now — syntax errors, invalid comparisons,
misplaced statements, and undefined names (`E9`, `F63`, `F7`, `F82`) — so CI catches
real bugs without failing on the existing style backlog. The formatter is **not**
enforced yet; please don't reformat whole files in unrelated PRs. We'll widen the
rules and turn on `ruff format --check` in follow-up PRs.

## Pre-commit hooks

We ship a [`.pre-commit-config.yaml`](.pre-commit-config.yaml) (ruff lint,
end-of-file/whitespace fixers, large-file and private-key guards). Install it once:

```bash
pre-commit install                 # run on staged files at every commit
pre-commit run --all-files         # run against the whole tree
```

## Tests

Tests run with [pytest](https://docs.pytest.org/). Most of the stack needs a GPU
and a served model, but there are **CPU-safe** tests you can (and CI does) run with
no GPU:

```bash
pytest -q inference/test_smoke.py \
          inference/vllm/vllm_native_diffusion/test_two_stream_cpu.py
```

- `inference/test_smoke.py` — offline wiring checks (mode registry, command
  construction, config keys); stdlib only.
- `inference/vllm/vllm_native_diffusion/test_two_stream_cpu.py` — pins the
  two-stream GDN snapshot/restore decode algorithm in pure PyTorch on tiny
  tensors; needs `torch` (CPU is fine), no GPU or vLLM.

If you add tests, keep the CPU-safe ones runnable without a GPU or network so they
stay in CI.

---

## Opening a pull request

1. Fork and branch from `main` (e.g. `feat/…`, `fix/…`, `docs/…`).
2. Make focused commits; keep unrelated changes out of the PR.
3. Run `ruff check .`, `pre-commit run` (on your staged files), and the CPU-safe
   tests locally.
4. Update docs (`README.md`, `inference/README.md`, `train/README.md`) when you
   change behavior, flags, or the model contract.
5. If you touch dependencies or third-party code, update
   [`COMPLIANCE.md`](COMPLIANCE.md) and [`NOTICE`](NOTICE) accordingly — and never
   introduce a copyleft or bundle a noncommercial dependency.
6. Push and open a PR against `trillion-labs/trida-stack`; fill in the PR template.

Found a security issue? **Do not open a public issue** — see
[`SECURITY.md`](SECURITY.md).

Thanks for contributing!
