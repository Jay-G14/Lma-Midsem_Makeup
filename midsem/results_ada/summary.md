# Ada study summary (job 84119; mean ± sd over 3 seeds where marked)

Selections: LR per rank {1: '3e-4', 4: '1e-3', 16: '1e-3'}; best LoRA ada_r16_lr1e-3_s43; DPO β=0.1 → ('1e-5', 100), β=0.5 → ('3e-5', 500)

| model | validation | seen entities | unseen entities | unseen wording | PPL (500 docs) | PPL (full split) |
|---|---|---|---|---|---|---|
| Full FT (Phase 3 checkpoint) | 82.9 | 83.9 | 81.0 | 45.3 | 63.5 | 58.62 |
| Full FT, 3 seeds | 79.6 ± 5.0 | 81.1 ± 4.6 | 78.6 ± 5.0 | 44.5 ± 2.1 | 63.35 ± 0.16 | – |
| LoRA r=1 (LR 3e-4) | 35.8 ± 1.8 | 37.1 ± 4.2 | 35.5 ± 2.4 | 31.5 ± 3.0 | 60.58 ± 0.26 | 52.46 ± 0.11 |
| LoRA r=4 (LR 1e-3) | 50.2 ± 3.0 | 48.7 ± 5.1 | 51.9 ± 1.2 | 36.1 ± 2.6 | 60.65 ± 0.06 | 52.89 ± 0.21 |
| LoRA r=16 (LR 1e-3) | 77.6 ± 3.7 | 78.0 ± 4.6 | 76.9 ± 5.9 | 51.4 ± 6.2 | 61.36 ± 0.06 | 53.84 ± 0.07 |
| Best LoRA (ada_r16_lr1e-3_s43) = DPO reference | 80.2 | 80.8 | 80.9 | 56.4 | 61.3 | 53.77 |
| DPO β=0.1 chosen (LR 1e-5, 100 steps) | 80.7 ± 0.2 | 80.9 ± 0.4 | 80.7 ± 0.3 | 55.6 ± 0.4 | 61.30 ± 0.00 | 53.76 ± 0.00 |
| DPO β=0.5 chosen (LR 3e-5, 500 steps) | 81.2 ± 0.3 | 82.3 ± 0.7 | 81.0 ± 0.8 | 49.0 ± 1.4 | 61.34 ± 0.01 | 53.79 ± 0.00 |
| DPO β=0.1 fixed (LR 1e-4, 500 steps) | 43.4 | 52.0 | 46.8 | 29.3 | 61.96 | 54.12 |
| DPO β=0.5 fixed (LR 1e-4, 500 steps) | 79.3 | 81.5 | 79.7 | 42.2 | 61.48 | 53.87 |
| Pretrained base | – | – | 0 (Phase 3) | – | 57.95 | 49.28 |

DPO step-0 loss: ln 2 in all 10 Ada runs (max deviation 1.904654323148236e-09).
