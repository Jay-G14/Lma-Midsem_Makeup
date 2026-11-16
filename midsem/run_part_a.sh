#!/bin/bash
# Part A, top to bottom, starting from the project checkpoint midsem/project/hindi/checkpoints/final_model.pt.
# Everything it reads is inside midsem/project/ (copied from the individual project; see midsem/project/README.md):
# hindi/checkpoints/{final_model,finetuned_model}.pt, hindi/tokenizer/tokenizer.json,
# hindi/data/reasoning/{train,val,test}.jsonl, hindi/data/processed/{train,val}.jsonl. Adapters go to midsem/checkpoints/.
# Run from anywhere:  bash midsem/run_part_a.sh      (about 1.5 h on one RTX 4050 laptop GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p midsem/logs midsem/results midsem/figures

echo "[A0] three test sets: seen entities, unseen entities (Phase 3 test set), unseen wording"
python3 midsem/part_a/build_test_sets.py

echo "[A1] trainable parameters and the step-0 identity check"
python3 midsem/part_a/lora_sanity.py

echo "[A2] LR probe at r=4 (5 epochs per LR), then the rank sweep r=1, 4, 16 with the chosen LR"
bash midsem/part_a/run_lr_probe.sh
LR=$(python3 midsem/part_a/choose_lr.py | tail -1)      # also installs the r=4 probe run at that LR as lora_r4
echo "     chosen LoRA learning rate: ${LR}"
for r in 1 16; do
  python3 midsem/part_a/lora_sft.py --rank "$r" --lr "$LR" --epochs 5 --resume > "midsem/logs/lora_r${r}.log" 2>&1
done
python3 midsem/part_a/memory_profile.py --lora-lr "$LR" > midsem/logs/memory_profile.log 2>&1
python3 midsem/part_a/evaluate_models.py --models base full_ft lora_r1 lora_r4 lora_r16 --select-best
echo "     two more full-finetune seeds with the project's Phase 3 script (seed spread of full finetuning)"
bash midsem/part_a/run_fullft_seeds.sh
# Control: r=1 with its update scale matched to r=4 (alpha/r x LR equal: 16 x 2.5e-4 = 4 x 1e-3),
# to tell a capacity limit from an LR artefact (validation and unseen-entity sets only).
python3 midsem/part_a/lora_sft.py --rank 1 --lr 2.5e-4 --epochs 5 --resume --tag ablation_r1_lr2.5e-4 \
  --output-dir midsem/checkpoints/ablation_r1_lr2.5e-4 > midsem/logs/ablation_r1_lr2.5e-4.log 2>&1
python3 midsem/part_a/evaluate_models.py --models ablation_r1_lr2.5e-4 --sets val unseen_entities

echo "[A3] preference pairs from the training split"
python3 midsem/part_a/build_preference_pairs.py

echo "[A4] DPO with beta = 0.1 and beta = 0.5 (variant 1), both from the best LoRA model"
for beta in 0.1 0.5; do
  python3 midsem/part_a/dpo.py --beta "$beta" > "midsem/logs/dpo_beta${beta}.log" 2>&1
done
python3 midsem/part_a/evaluate_models.py --models dpo_beta0.1 dpo_beta0.5

echo "[A5] figures and summary tables (drift PPL is in every eval_*.json)"
python3 midsem/part_a/plots.py
python3 midsem/part_a/summarize.py > /dev/null
echo "Done: midsem/results/part_a_summary.md, midsem/figures/"
