#!/bin/bash
# Two more full-finetune seeds (43, 44) with the project's own Phase 3 script, unchanged (seed 42 is the Phase 3
# checkpoint itself), so that full finetuning gets the same 3-seed treatment as LoRA. ~13 min per seed on a laptop GPU.
# The script runs from inside midsem/project/, so its default relative paths (hindi/checkpoints/final_model.pt,
# hindi/data/..., hindi/configs/...) resolve to the copies there; only --seed and --output-dir are set.
set -euo pipefail
cd "$(dirname "$0")/../project"
mkdir -p ../logs ../checkpoints
for s in 43 44; do
  PYTHONPATH=. python3 hindi/train/finetune.py --seed "$s" --output-dir "../checkpoints/fullft_s$s" > "../logs/fullft_s$s.log" 2>&1
done
cd ../..
PYTHONPATH=. python3 midsem/part_a/evaluate_models.py --models fullft_s43 fullft_s44 > midsem/logs/eval_fullft_seeds.log 2>&1
