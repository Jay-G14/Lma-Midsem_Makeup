#!/bin/bash
# LR probe for LoRA: r = 4, the full 5-epoch recipe, one run per LR. The LR whose best epoch has the
# lowest reasoning-validation loss is then used for every rank (alpha fixed at 16, so alpha / r
# rescales the update with r as in Hu et al.). --resume makes an interrupted probe continue where it
# stopped; data order and dropout are seeded per epoch, so a resumed run equals an uninterrupted one.
set -euo pipefail
cd "$(dirname "$0")/../.."
for lr in ${LRS:-1e-4 3e-4 1e-3 3e-3}; do
  PYTHONPATH=. python3 midsem/part_a/lora_sft.py --rank 4 --lr "$lr" --epochs 5 --resume \
    --tag "lrprobe_r4_lr${lr}" --output-dir "midsem/checkpoints/lrprobe/r4_lr${lr}" \
    >> "midsem/logs/lrprobe_r4_lr${lr}.log" 2>&1
done
