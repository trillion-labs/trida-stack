#!/bin/bash
# Download benchmark/evaluation datasets via the Hugging Face `datasets` library.
# These are cached under $HF_HOME (or ~/.cache/huggingface). See DATA_CATALOG.md for licenses.
# SWE-bench is handled separately by benchmarking/swe_bench/ (mixed-license task instances).
set -e

python - <<'PY'
from datasets import load_dataset

# (name, config, split) — light pulls just to materialize the cache.
SETS = [
    ("openai/gsm8k", "main", "test"),
    ("openai/openai_humaneval", None, "test"),
    ("google-research-datasets/mbpp", "sanitized", "test"),
    ("cais/mmlu", "all", "test"),
    ("hendrycks/competition_math", None, "test"),
    ("google/IFEval", None, "train"),
]
for name, config, split in SETS:
    try:
        ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        print(f"OK   {name} [{config or '-'}:{split}] -> {len(ds)} rows")
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {name}: {e}")

print("\nNote: GPQA (Idavidrein/gpqa) is gated — request access on Hugging Face.")
print("Note: SWE-bench Verified is pulled by benchmarking/swe_bench/ (see its README + license).")
PY
