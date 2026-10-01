# trida-stack — Third-Party Dependency & License Inventory

This project (`trida-stack`) is released under the **Apache License 2.0** (see
[`LICENSE`](LICENSE)). This document inventories the third-party software, benchmark
harnesses, and datasets it depends on, with their licenses. The project's own source
code is Apache-2.0; the **Trida** model weights, when released, carry their own
model-card license.

## 1. Bundled third-party code (redistributed in this repo)

| Component | Path | Upstream | License |
|---|---|---|---|
| IFEval scorer | `inference/ifeval_lib/` | [google-research · instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | **Apache-2.0** |

This is the **only** third-party source code redistributed in the repository (see
[`NOTICE`](NOTICE)). Everything else below is installed or fetched separately.

## 1b. Noncommercial upstream code (fetched via recipe, NOT bundled)

The two-stream Gated-DeltaNet + block-diffusion Triton kernels are **not vendored**
in this repo. They are fetched/patched from upstream at setup time and are
**PolyForm Noncommercial 1.0.0 — a noncommercial license, not permissive.**

| Component | Recipe | Upstream | License |
|---|---|---|---|
| Training kernels | `train/block_gated_delta_rule/` (`fetch_kernels.sh` + `trillion_mods.patch`) | [yuchen-zhu-zyc/HybridDiffusion](https://github.com/yuchen-zhu-zyc/HybridDiffusion) @ `6ca547a` | **PolyForm-NC 1.0.0** |
| sglang two-stream backend | `inference/sglang/two_stream_diffusion.patch` | same upstream @ `6ca547a` | **PolyForm-NC 1.0.0** |
| vLLM diffusion readout | `inference/vllm/vllm_native_diffusion/KERNELS.md` | sglang `block_gdn` backend (above) | **PolyForm-NC 1.0.0** |

Trillion Labs' patches/modifications against these kernels are derivative works and
are likewise PolyForm-NC. **The two-stream training and diffusion-serving paths are
therefore noncommercial-only.** The repo's own Apache-2.0 code (data/eval/serving
glue, the HF/AR model code, benchmark wrappers) does not itself redistribute this
noncommercial code.

## 2. Runtime dependencies (installed separately)

| Component | Role | Upstream | License |
|---|---|---|---|
| PyTorch | tensor / runtime | [pytorch/pytorch](https://github.com/pytorch/pytorch) | **BSD-3-Clause** |
| HF Transformers | model loading (≥ 4.57) | [huggingface/transformers](https://github.com/huggingface/transformers) | **Apache-2.0** |
| HF Datasets | eval-data loading | [huggingface/datasets](https://github.com/huggingface/datasets) | **Apache-2.0** |
| HF Accelerate | training | [huggingface/accelerate](https://github.com/huggingface/accelerate) | **Apache-2.0** |
| SGLang | diffusion serving backend | [sgl-project/sglang](https://github.com/sgl-project/sglang) | **Apache-2.0** |
| vLLM | optional AR serving backend | [vllm-project/vllm](https://github.com/vllm-project/vllm) | **Apache-2.0** |

## 3. Benchmark harnesses (invoked from `benchmark/`, not bundled)

`benchmark/` holds **first-party wrapper scripts** (register/run drivers, handlers,
serving glue); the harnesses themselves are installed separately and executed.
Some run their task suites by executing third-party / untrusted code, so they are
run sandboxed (an operational detail, not a licensing one).

| Harness | Where | Upstream | License |
|---|---|---|---|
| lm-evaluation-harness | `benchmark/tasks/` (Trida task YAML only; not otherwise wired in-repo) | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | **MIT** |
| BFCL (Berkeley Function-Calling) | `benchmark/bfcl_v4/` | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) | **Apache-2.0** |
| tau2-bench | `benchmark/tau2/` | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | **MIT** |
| FunctionChat-Bench | `benchmark/functionchat/` | [kakao/FunctionChat-Bench](https://github.com/kakao/FunctionChat-Bench) | **Apache-2.0** |
| KoAgentBench | `benchmark/ko_agentbench/` | [huggingface-KREW/Ko-AgentBench](https://huggingface.co/datasets/huggingface-KREW/Ko-AgentBench) | **Apache-2.0** |
| SWE-bench | `benchmark/swe_bench/` | [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) | **MIT** |
| Terminal-Bench | `benchmark/terminal_bench/` | [laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench) | **Apache-2.0** |

## 4. Evaluation datasets (pulled on demand, not redistributed)

None are redistributed from this repo — all are downloaded at eval time. See
[`data/DATA_CATALOG.md`](data/DATA_CATALOG.md).

| Dataset | Used by | Source | License |
|---|---|---|---|
| GSM8K | `inference/eval.py` | openai/gsm8k | **MIT** |
| MMLU-Pro | `inference/eval.py` | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | **MIT** |
| IFEval | `inference/eval_ifeval.py` | google/IFEval | **Apache-2.0** |
| benchmark suites | `benchmark/` | each harness's own datasets | per upstream |

## 4b. Model weights — base model and derived checkpoint

Software licences do not carry over to model weights, and the weights this stack serves
are **not** covered by this repository's Apache-2.0 licence. They are also not
redistributed here.

| | component | licence | verified |
|---|---|---|---|
| base | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | **Apache-2.0** | HF model metadata, 2026-10-01 |
| base | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | **Apache-2.0** | HF model metadata, 2026-10-01 |
| base | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | **Apache-2.0** | HF model metadata, 2026-10-01 |
| derived | [trillionlabs/Trida2.0-4B](https://huggingface.co/trillionlabs/Trida2.0-4B) | **`other` — terms not yet published** | HF model card, 2026-10-01 |

**Base models.** All three Qwen checkpoints the trainer targets are Apache-2.0, so no
use restriction propagates from the base model into a derived checkpoint. This matters
because a derived model can inherit the base model's terms — Apache-2.0 bases do not
impose any.

**Derived checkpoint — open item.** The `trillionlabs/Trida2.0-4B` model card declares
`license: other` with no `license_name`, no `license_link` and no LICENSE file, so a
downloader currently has no stated terms. That is a decision for Trillion Labs to publish
on the model repository; this repository can only record the state. Until it is set,
**do not assume the model carries this repository's Apache-2.0 licence.**

**Training data.** Trida's training data is internal and proprietary to Trillion Labs and
is out of scope for this inventory — see [`data/DATA_CATALOG.md`](data/DATA_CATALOG.md).
The datasets listed in §4 above are evaluation data only.

**A note on the noncommercial kernels.** The §1b kernels are *software* used to train and
serve, under a noncommercial licence. Whether that restricts the resulting weights is a
legal question about the licence's scope, not a fact this file can settle. Treat it as
open and route it to legal before any commercial distribution of the weights.

## 5. License summary

All first-party and bundled code is **Apache-2.0**, and the *permissive* runtime
dependencies are **Apache-2.0**, **MIT**, or **BSD-3-Clause** — mutually compatible,
none copyleft (no GPL/AGPL). The one **non-permissive** dependency is the
PolyForm-Noncommercial two-stream kernel set in §1b, which is **not bundled** (fetched
via recipe) but is required for the two-stream training and diffusion-serving paths.

The practical obligations when redistributing:

1. Preserve the `LICENSE` and `NOTICE` files and upstream copyright notices.
2. For Apache-2.0 files you modify, **state significant changes** (Apache-2.0 §4(b)).
3. For MIT components, retain the copyright + license text.
4. **Do not redistribute the §1b PolyForm-NC kernels, and do not use the two-stream
   training / diffusion-serving paths commercially.** Those paths are noncommercial.

No *bundled* code imposes a copyleft or noncommercial obligation. The PolyForm-NC
kernels are noncommercial but are a fetched dependency, not redistributed here.
