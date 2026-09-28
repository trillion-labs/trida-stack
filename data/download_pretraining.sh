#!/bin/bash
# Download pretraining / fine-tuning datasets via Fast-dLLM v2's LMFlow data downloader.
# That downloader is a separate project and is NOT bundled in this repo.
# Usage: FAST_DLLM_DIR=/path/to/fast_dllm_v2 bash download_pretraining.sh <dataset_name|all>
# See DATA_CATALOG.md for the dataset list AND their licenses (several are NON-COMMERCIAL).
set -e

# Point FAST_DLLM_DIR at your own Fast-dLLM v2 checkout (it provides data/download.sh).
FAST_DLLM_DIR="${FAST_DLLM_DIR:?set FAST_DLLM_DIR to your Fast-dLLM v2 checkout (provides data/download.sh)}"
LMFLOW_DL="${FAST_DLLM_DIR}/data/download.sh"

if [ ! -f "$LMFLOW_DL" ]; then
  echo "ERROR: expected LMFlow downloader at $LMFLOW_DL" >&2
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: bash $(basename "$0") <dataset_name|all>"
  echo "Datasets: alpaca, agent_flan, ubpc, dpo-mix-7k, hh_rlhf, ni, imdb, wiki_en_eval, ..."
  echo "⚠  Review licenses in DATA_CATALOG.md first (Alpaca = CC-BY-NC, Wikipedia = share-alike)."
  exit 0
fi

echo "⚠  License reminder: confirm '$1' usage against DATA_CATALOG.md before proceeding."
cd "${FAST_DLLM_DIR}/data"
bash download.sh "$@"
