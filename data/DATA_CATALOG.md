# Data Catalog

Datasets used by the open-source **evaluation/benchmark** stack, with licenses. **No data files
are committed to this repo** — they are downloaded on demand. Licenses marked **verify** are
best-effort and should be confirmed before use.

> **Scope:** Trida's **training data is internal/proprietary** to Trillion Labs and is **out of
> scope** for this open-source inventory. This catalog covers only the evaluation datasets the
> benchmark stack pulls at eval time (none are redistributed from this repo).

## Benchmark / evaluation data

Pulled by `download_benchmarks.sh` (via Hugging Face / lm-evaluation-harness):

| Dataset | Benchmark | Source | License |
|---------|-----------|--------|---------|
| **GSM8K** | math reasoning | openai/gsm8k | MIT |
| **HumanEval** | code gen | openai/openai_humaneval | MIT |
| **MBPP / sanitized** | code gen | google-research-datasets/mbpp | CC-BY-4.0 |
| **MMLU** | knowledge | cais/mmlu | MIT |
| **GPQA** | knowledge | Idavidrein/gpqa | CC-BY-4.0 |
| **MATH / Minerva Math** | math | hendrycks/competition_math | MIT |
| **IFEval** | instruction following | google/IFEval | Apache-2.0 |
| **SWE-bench Verified** | **agentic software** | princeton-nlp/SWE-bench_Verified | **MIT** |
| **Terminal-Bench (Hard)** | **agentic terminal** | laude-institute/terminal-bench | **Apache-2.0** (tasks run untrusted software in Docker) |

🚩 **Compliance notes:** the single-turn eval sets above are permissive (MIT / CC-BY-4.0 /
Apache-2.0) and are only read at evaluation time. The two **agentic** benchmarks are the exception:
the SWE-bench (MIT) and Terminal-Bench (Apache-2.0) harnesses are permissively licensed, but
their task suites execute third-party / untrusted software — run them Docker-sandboxed.

## Usage

```bash
# Benchmark datasets (cached via the Hugging Face datasets library)
bash download_benchmarks.sh
```
