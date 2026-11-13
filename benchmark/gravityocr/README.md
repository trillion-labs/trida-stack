# GravityOCR evaluation (docr-ko-eval)

`run.sh` evaluates a GravityOCR checkpoint on a docr-ko-eval dataset in one command, on one NVIDIA GPU.

- **Input:** a docr-ko-eval dataset of region crops. Each crop is one text block, table or formula with its target.
  There is no layout detection step.
- **Serving:** `run.sh` starts the checkpoint with [`inference/gravityocr/serve.sh`](../../inference/gravityocr/)
  (upstream SGLang plus the GravityOCR self-speculative patch).
- **Timing:** it sends every crop at each concurrency level and times the run.
- **Scoring:** it scores the predictions with OmniDocBench's own metric code and writes one report with speed,
  accuracy and per-source numbers.

| file | what it is |
|---|---|
| `setup.sh` | One-time setup: the scoring venv, OmniDocBench, TeX Live and the patched SGLang server env, all under one directory. |
| `run.sh` | The evaluation: export → serve → warm-up → measured passes → stop server → CDM self-check → score → report. |
| `score.py` | The metric definitions (OmniDocBench Edit_dist / BLEU / TEDS / CDM). `run.sh` calls it; it also runs on its own. |

## 1. Setup (once per machine or image)

```bash
bash benchmark/gravityocr/setup.sh                                  # installs into $HOME/.gravityocr-eval
EVAL_HOME=/opt/gravityocr-eval bash benchmark/gravityocr/setup.sh   # another directory (e.g. in a Dockerfile)
```

### Needs
- **Tools:** `uv`, `git`, `curl`, `tar` and `perl` on PATH.
- **For the server env:** the CUDA 12.x toolkit at `CUDA_HOME` (default `/usr/local/cuda-12.8`).
  A GPU is not needed to build, only to run.
- **Disk:** about 15 GB, mostly the SGLang env. Downloads come from PyPI, the PyTorch index, GitHub and CTAN.
- **Time:** about 10–20 minutes.

### What it installs into `EVAL_HOME`

| step | result |
|---|---|
| OmniDocBench at a pinned commit (`0604d98`) | `OmniDocBench/`: the metric code `score.py` imports |
| Scoring venv, Python 3.10 | `venv/`: OmniDocBench's pinned requirements plus `datasets`, `sacrebleu` and `pymupdf` |
| TeX Live from upstream `install-tl`, scheme-basic plus the packages CDM uses | `texlive/` |
| The patched SGLang server env, via [`inference/gravityocr/sglang/setup.sh`](../../inference/gravityocr/sglang/setup.sh) | `sglang/.venv-sglang/` |
| Defaults for `run.sh` | `env.sh`: sets `VBIN`, `PYTHON` and `OMNIDOCBENCH`, and puts TeX Live first on PATH |

The TeX packages are `geometry booktabs multirow amsmath amsfonts xcolor was cjk cjk-ko nanumtype1 uhc
unfonts-core`; Hangul is set in the `mj` font.

At the end, `setup.sh` checks that pdflatex renders `\mathcolor`. Re-running it is safe: finished steps are
reused.

### Settings

| setting | effect |
|---|---|
| `EVAL_HOME` | install directory (default `$HOME/.gravityocr-eval`) |
| `SKIP_SGLANG=1` | no server env, for a scoring-only machine, or when you run `run.sh` with `ENDPOINT` |
| `SKIP_TEXLIVE=1` | use the `pdflatex` already on PATH; it must be TeX Live 2022.06 or newer |
| `TL_REPO` | TeX Live package mirror (default `https://mirror.ctan.org/systems/texlive/tlnet`) |
| `CUDA_HOME`, `PY` | passed to the SGLang setup |

> **Why not `apt install texlive`?**
> - CDM colors each LaTeX token with `\mathcolor`, which needs xcolor 2.14 or newer.
> - Ubuntu 22.04 ships TeX Live 2021 (xcolor 2.13). There every token renders black, no token is found, and
>   **every formula scores CDM 0 without an error**.
> - `setup.sh` installs upstream TeX Live, and `run.sh` stops before scoring if a CDM self-check fails.

## 2. Run

```bash
hf auth login                                          # once: the default dataset is private (or set HF_TOKEN)
MODEL=trillionlabs/GravityOCR-KO bash benchmark/gravityocr/run.sh
```

`DATASET` defaults to [`trillionlabs/gravityocr-dev`](https://huggingface.co/datasets/trillionlabs/gravityocr-dev),
the 402-crop dev split of docr-ko-eval-v2, downloaded from the Hub on the first run. The Hugging Face account must
have access to the `trillionlabs` dataset. `hf` comes with `huggingface_hub`; `uvx --from huggingface_hub hf auth login`
works without installing it.

More examples:

```bash
# Another checkpoint, autoregressive decoding, three concurrency levels
MODEL=/ckpt/my_checkpoint MODE=causal CONCURRENCY="1 8 64" bash benchmark/gravityocr/run.sh

# Another dataset: a Hub dataset id or a local datasets.save_to_disk directory
MODEL=trillionlabs/GravityOCR-KO DATASET=/data/docr-ko-eval-v2 bash benchmark/gravityocr/run.sh

# Score against a server that is already running (run.sh then starts and stops nothing)
ENDPOINT=http://127.0.0.1:30000/v1 bash benchmark/gravityocr/run.sh
```

`run.sh` reads `$EVAL_HOME/env.sh`; set `EVAL_HOME` if `setup.sh` installed somewhere other than the default.
Anything set in the environment overrides `env.sh`. `bash run.sh --help` lists every setting.

> **Not on a Hopper GPU (for example L40S or L40)?** `serve.sh` refuses GPUs other than compute capability 9.0
> unless you add `SKIP_PREFLIGHT=1`. The reference numbers below were measured that way on an L40S. If the CUDA
> toolkit is not at `/usr/local/cuda-12.8`, set `CUDA_HOME` too:
>
> ```bash
> SKIP_PREFLIGHT=1 CUDA_HOME=/usr/local/cuda MODEL=... bash benchmark/gravityocr/run.sh
> ```

### Settings

| setting | default | meaning |
|---|---|---|
| `MODEL` | – (required) | checkpoint directory or Hub id |
| `DATASET` | `trillionlabs/gravityocr-dev` | Hub dataset id or `datasets.save_to_disk` directory |
| `OUT` | `runs/<model>-<mode>-<time>` | output directory |
| `MODE` | `self-spec` | `self-spec` (block-diffusion draft + AR verify) or `causal` (plain AR) |
| `CONCURRENCY` | `128` | concurrency levels, space-separated |
| `WARMUP` | `1` | one crop per task, then one full pass at the first level, both discarded (see below) |
| `SPLITS` | every split in `DATASET` | splits to score |
| `SCORE_LEVELS` | all levels | levels whose predictions are scored; `none` skips scoring |
| `ENDPOINT` | – | OpenAI base URL of a running server |
| `PORT` | `30500` | server port |
| `SCORE_WORKERS` | CPU count | scoring processes |

Other `serve.sh` settings pass through, for example `GPU`, `MEMFRAC`, `CUDA_HOME`, `SKIP_PREFLIGHT` and `HF_HOME`.

### What one run does
1. **Export:** reads the dataset (Hub or `save_to_disk`) and writes `data/data.jsonl`, `data/images/<id>.png` and
   `data/splits.json`.
2. **Serve:** starts `serve.sh` in its own process group and waits for `/v1/models`. The server is stopped when
   predictions are done, and also if the run fails or is interrupted.
3. **Predict:** sends each crop as an OpenAI chat request: one PNG `data:` URL plus the crop's task prompt
   (`Text Recognition:`, `Table Recognition:` or `Formula Recognition:`).
   - Settings: `temperature` 0; `max_tokens` 512 for text and 4096 for tables and formulas.
   - **Warm-up:** the first high-concurrency pass on a fresh server is about 2× slower, so the warm-up pass
     is discarded.
   - **Measured pass:** one per concurrency level.
4. **Self-check:** scores CDM of two formulas, one with Hangul, against themselves; anything below 1.0 stops
   the run.
5. **Score:** each split at each scored level.
6. **Report:** writes `report.md` and `report.json`.

### Output (`$OUT`)

| file | contents |
|---|---|
| `report.md`, `report.json` | speed, accuracy, per-source accuracy, timing, mean accept length (self-spec) |
| `sweep.json` | per level: wall time, crops/s, latency mean/p50/p90/max, errors, output tokens, tokens/s (and the warm-up pass) |
| `preds_c<N>.jsonl` | per crop: `id`, `pred`, `sec`, `out_tokens`, `error` |
| `c<N>/<split>/` | `summary.json`, `summary.md`, `scores.jsonl` (per-crop metrics) |
| `timing.json`, `serve.log`, `data/` | stage timings, the server log, the exported crops |

## Metrics

All metrics come from OmniDocBench's code; `score.py` has the details.

| task | headline | also reported |
|---|---|---|
| text | 1 − Edit_dist: normalized Levenshtein on OmniDocBench's `normalized_text` | BLEU-4 (sacreBLEU) |
| table | TEDS | TEDS-S (structure only), Edit_dist |
| formula | CDM: both LaTeX strings rendered with pdflatex and the rendered tokens matched (F1) | Edit_dist |

**Overall** = mean(1 − text Edit_dist, table TEDS, formula CDM) × 100. When a dataset lacks a task, the report
averages the tasks it has and says which.

## Dataset format

A Hub dataset or a `datasets.save_to_disk` directory with these columns, in one or more splits (for example `dev`
and `test`; a plain unsplit `Dataset` is scored as one split, `all`):

| column | contents |
|---|---|
| `id` | crop id |
| `image` | the crop |
| `prompt` | `Text Recognition:`, `Table Recognition:` or `Formula Recognition:` |
| `target` | the ground truth |
| `task_type` | `text`, `table` or `formula` |
| `source` | used for the per-source breakdown |

Optional columns (`kind`, `lang`, `degradation`, `source_detail`, ...) feed the extra breakdowns in
`summary.md`. gravityocr-dev, docr-ko-eval-v0, aihub479-ocr-eval-v0 and docr-ko-eval-v2 all have this layout.

Datasets and checkpoints are not stored in this repository. Pass Hub ids as `MODEL` and `DATASET`, or mount local
directories.

## Reference numbers

GravityOCR-KO (`trillionlabs/GravityOCR-KO` @ `bc0e105`) on docr-ko-eval-v2 (402 dev / 402 test crops). The dev
row is the default dataset, `trillionlabs/gravityocr-dev` (the same 402 crops).
Setup: 1 × L40S (`SKIP_PREFLIGHT=1`), self-spec, concurrency 128, from a fresh `setup.sh` and `run.sh` with
the defaults.

| split | Overall | text 1−ED | text BLEU | table TEDS | TEDS-S | formula CDM |
|---|---|---|---|---|---|---|
| dev | 92.57 | 0.902 | 0.740 | 0.939 | 0.990 | 0.936 |
| test | 92.37 | 0.904 | 0.709 | 0.935 | 0.985 | 0.932 |

Speed of the measured pass (804 crops):

| wall time | crops/s | output tokens/s | latency p50 / p90 | mean accept length |
|---|---|---|---|---|
| 12.7 s | 63.1 | 8,163 | 1.64 / 3.00 s | 11.5 |

Whole run: 2.4 minutes (server start 30 s, predictions 41 s, scoring 70 s).

Decoding is greedy, but outputs of batched runs can still differ in a few crops from run to run. Expect Overall
to move by about ±0.1.

## Docker

The image runs `setup.sh` once at build time, and the container then only runs `run.sh`.

The Dockerfile below is a starting point and has not been built yet. `setup.sh` and `run.sh` themselves were
tested on a bare Ubuntu 22.04 L40S host.

```dockerfile
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates perl tar \
 && rm -rf /var/lib/apt/lists/* && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
COPY . /trida-stack
ENV EVAL_HOME=/opt/gravityocr-eval CUDA_HOME=/usr/local/cuda
RUN bash /trida-stack/benchmark/gravityocr/setup.sh
WORKDIR /work
ENTRYPOINT ["bash", "/trida-stack/benchmark/gravityocr/run.sh"]
```

```bash
docker run --gpus 1 -e HF_TOKEN -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $PWD/runs:/work/runs \
  -e MODEL=trillionlabs/GravityOCR-KO gravityocr-eval
# a local dataset instead of the Hub: add -v /data:/data -e DATASET=/data/<save_to_disk dir>
# on a non-Hopper GPU, add: -e SKIP_PREFLIGHT=1
```

## Troubleshooting

- **The dataset download fails with 401, 404 or "Repository not found".** `trillionlabs/gravityocr-dev` is
  private: run `hf auth login` (or set `HF_TOKEN`) with an account that can see it.
- **Every formula scores CDM 0 with `cdm_errors=0`.** pdflatex is too old or a TeX package is missing. Run
  `setup.sh`; `run.sh`'s self-check now stops before scoring in this case.
- **CDM went to 0 after moving `EVAL_HOME`.** The venvs hold absolute paths. Re-run `setup.sh` at the new
  location.
- **The server does not come up.** Look at `$OUT/serve.log`. `SERVE_TIMEOUT` (default 900 s) covers a first
  Hub download; see [`inference/gravityocr/README.md`](../../inference/gravityocr/README.md) for `serve.sh`.
