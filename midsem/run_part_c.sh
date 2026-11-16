#!/bin/bash
# Part C, top to bottom: C1 (direct / zero-shot CoT / self-consistency) and C2 (LLM judge).
# Generator: Qwen/Qwen2.5-1.5B-Instruct (bf16). Judge: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
# (Meta Llama 3.2 3B Instruct, 4-bit NF4), a different model family. No API keys are needed:
# both models are open and ungated on the Hugging Face Hub. ~15 min on one 6 GB laptop GPU.
#
# One-time environment (reuses the system torch / transformers, adds the 4-bit loader):
#   python3 -m venv --system-site-packages midsem/.venv
#   midsem/.venv/bin/pip install -r midsem/requirements.txt
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-$( [ -x midsem/.venv/bin/python ] && echo midsem/.venv/bin/python || echo python3 )}   # a local venv if there is one
mkdir -p midsem/logs midsem/results

echo "[C1.1] the 20 Hindi problems -> midsem/data/c1_problems.jsonl"
$PY midsem/part_c/c1_problems.py
echo "[C1.2-4] three conditions with Qwen2.5-1.5B-Instruct (checker: midsem/part_c/checker.py)"
$PY midsem/part_c/c1_run.py 2>&1 | tee midsem/logs/c1_run.log
echo "[C2] 30 pairs x 2 orders judged by Llama-3.2-3B-Instruct (4-bit NF4, fits a 6 GB GPU)"
$PY midsem/part_c/c2_judge.py 2>&1 | tee midsem/logs/c2_judge.log
# The same pairs in full fp32 precision need ~13 GB of GPU memory (run on Ada by midsem/ada/run.sh):
#   $PY midsem/part_c/c2_judge.py --judge unsloth/Llama-3.2-3B-Instruct --pairs-in midsem/data/c2_pairs_with_verdicts.jsonl \
#       --dtype float32 --device-map auto --tag _fp32
echo "Done: midsem/results/c1_results.json, midsem/results/c2_results.json, midsem/data/c2_pairs_with_verdicts.jsonl"
