# Part A summary (variant 1: Model H, LoRA on W_Q, W_K, W_V, W_O, DPO beta = 0.5 and 0.1)

## A1 Trainable parameters and step-0 identity

| r | trainable | total | share of model | step-0 max abs logit diff | bit-identical | grad norm A / B at step 0 |
|---|---|---|---|---|---|---|
| 1 | 24,576 | 26,293,248 | 0.0935% | 0.0e+00 | True | 0.0e+00 / 1.313 |
| 4 | 98,304 | 26,366,976 | 0.3728% | 0.0e+00 | True | 0.0e+00 / 0.626 |
| 16 | 393,216 | 26,661,888 | 1.4748% | 0.0e+00 | True | 0.0e+00 / 0.339 |

## LR probe (r = 4, 5 epochs)

| lr | best epoch | best val reasoning loss | PPL after epoch 5 |
|---|---|---|---|
| 0.0001 | 5 | 0.2836 | 59.84 |
| 0.0003 | 5 | 0.1603 | 59.63 |
| 0.001 | 5 | 0.1361 | 60.89 |
| 0.003 | 5 | 0.2484 | 64.67 |

Chosen LR: 0.001

## A2 / A4 Answer EM (chain EM) on the three test sets

| model | Seen entities | Unseen entities | Unseen wording | val | drift PPL |
|---|---|---|---|---|---|
| Full FT (Phase 3) | 83.9 (81.6) | 81.0 (72.9) | 45.3 (3.1) | 82.9 | 63.50 |
| LoRA r=1 | 32.2 (2.5) | 36.4 (2.8) | 31.3 (1.0) | 36.0 | 61.43 |
| LoRA r=4 | 60.9 (45.9) | 55.8 (40.8) | 44.3 (7.4) | 54.1 | 60.89 |
| LoRA r=16 | 63.5 (56.0) | 63.1 (53.5) | 45.3 (5.5) | 65.6 | 61.38 |
| DPO beta=0.1 | 39.4 (28.6) | 31.9 (22.3) | 32.5 (1.9) | 35.4 | 61.91 |
| DPO beta=0.5 | 69.4 (62.2) | 62.4 (52.0) | 42.8 (5.0) | 63.0 | 61.56 |
| Pretrained base | - | 0 (Phase 3) | - | - | 57.95 |

## Per-task answer EM, unseen entities

| model | transitive | numeric_extremum | numeric_pair | verbal |
|---|---|---|---|---|
| Full FT (Phase 3) | 84.8 | 67.7 | 64.8 | 100.0 |
| LoRA r=1 | 28.4 | 32.3 | 40.3 | 51.4 |
| LoRA r=4 | 65.7 | 37.1 | 43.9 | 64.3 |
| LoRA r=16 | 73.4 | 58.6 | 55.6 | 54.6 |
| DPO beta=0.1 | 17.7 | 55.4 | 49.5 | 22.2 |
| DPO beta=0.5 | 70.7 | 67.2 | 57.6 | 47.2 |

## A2 Memory (NVIDIA GeForce RTX 4050 Laptop GPU, batch 8, 300 identical steps)

| config | trainable | weights MB | grads MB | AdamW state MB | peak MB | steps/s |
|---|---|---|---|---|---|---|
| full finetune (Phase 3) | 26,268,672 | 107.1 | 100.21 | 200.41 | 2837.5 | 11.8 |
| LoRA r=1 | 24,576 | 107.2 | 0.09 | 0.19 | 2425.9 | 16.64 |
| LoRA r=4 | 98,304 | 107.5 | 0.38 | 0.75 | 2427.9 | 16.19 |
| LoRA r=16 | 393,216 | 108.6 | 1.5 | 3.0 | 2435.8 | 16.14 |

- LoRA r=1: lr 0.001, best epoch 3, best val reasoning loss 0.5465; per-epoch (val loss, PPL): 0: (6.3951, 57.95), 1: (0.6580, 62.07), 2: (0.5723, 61.83), 3: (0.5465, 61.43), 4: (0.5641, 61.21), 5: (0.5563, 62.12)
- LoRA r=4: lr 0.001, best epoch 5, best val reasoning loss 0.1361; per-epoch (val loss, PPL): 0: (6.3951, 57.95), 1: (0.2427, 61.07), 2: (0.2298, 60.90), 3: (0.1606, 60.65), 4: (0.1546, 60.51), 5: (0.1361, 60.89)
- LoRA r=16: lr 0.001, best epoch 5, best val reasoning loss 0.0786; per-epoch (val loss, PPL): 0: (6.3951, 57.95), 1: (0.1995, 61.20), 2: (0.1166, 61.18), 3: (0.0907, 60.95), 4: (0.0843, 60.89), 5: (0.0786, 61.38)

Best LoRA (validation answer EM): r=16

Control r=1 at LR 2.5e-4 (alpha/r x LR matched to r=4): best val loss 0.4828, val EM 38.0, unseen-entity EM 38.3

## DPO beta=0.1

- step-0 loss: first batch 0.693147, whole training set 0.693147 (log 2 = 0.693147)
- steps 500; final training-set loss 0.2057, margin 24.31, chosen log-ratio -4.85, rejected log-ratio -29.16, preference acc 0.938
- validation pairs: margin 0.00 -> 23.16, loss 0.6931 -> 0.2589, preference acc 0.000 -> 0.885

## DPO beta=0.5

- step-0 loss: first batch 0.693147, whole training set 0.693147 (log 2 = 0.693147)
- steps 500; final training-set loss 0.1117, margin 8.81, chosen log-ratio -0.25, rejected log-ratio -9.06, preference acc 0.960
- validation pairs: margin 0.00 -> 8.02, loss 0.6931 -> 0.2720, preference acc 0.000 -> 0.900

## A3 Preference pairs

- 2000 training pairs (10870 eligible training samples), 400 validation pairs
- task mix {'numeric_pair': 400, 'transitive': 800, 'verbal_comparison': 400, 'numeric_extremum': 400}
- question types {'numeric_pair_yesno': 216, 'transitive_max': 286, 'numeric_pair_which': 184, 'verbal_yesno': 400, 'numeric_extremum_min': 197, 'transitive_min': 264, 'transitive_yesno': 250, 'numeric_extremum_max': 203}
- names in train pool 100.0%, test-pool names 0, val-pool names 0
- prompt sentences with a training template 100.0%, prompts equal to a test question 0

## Test-set audit

| set | size | names in train pool | names in test pool | sentences with a training template | questions in train |
|---|---|---|---|---|---|
| seen_entities | 1000 | 100.0% | 0.0% | 100.0% | 0 |
| unseen_entities | 1000 | 0.0% | 100.0% | 100.0% | 0 |
| unseen_wording | 1000 | 100.0% | 0.0% | 0.0% | 0 |
