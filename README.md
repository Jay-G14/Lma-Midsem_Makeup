# CL3410 Make-up Mid-Semester Assignment (variant 1)

Jayant Gupta, roll 2023113001. Variant = last digit 1, 1 mod 5 = **1**.

| Part | Variant 1 values |
| :--- | :--- |
| A | Model H (Hindi, 26.27M parameters); LoRA on W_Q, W_K, W_V, W_O of every block; DPO β = 0.5 (and β = 0.1) |
| B | LSH b = 10, r = 10; C = 3 × 10^21 FLOPs; LoRA 4096 × 4096, r = 4; 8B model; sampling α = 0.5 |

**Report:** `midsem/report/report.pdf` (LaTeX source in `midsem/report/`). Every number in it is a macro generated from
the result files, and its 17 figures (pages 7–14, after the 6-page main text) are drawn from them too.

## What is where

- **This repository** (GitHub, 26 MB): all code, every result and log, the figures, the report and its source, the
  deliverable JSONL files and the evaluation data. The individual project's code and data that Part A reuses are
  copied into `midsem/project/`, so `midsem/` runs on its own.
- **Google Drive: https://drive.google.com/drive/folders/1o46m2h8MDgc8dHXQR7E3_ddeua4i7eL2** (777 MB): the 72 large
  files, too big for git. They are every checkpoint and adapter (`*.pt`) and the training data
  (`project/hindi/data/reasoning/train.jsonl`, `project/hindi/data/processed/`). The Drive folder mirrors `midsem/`.

| Deliverable | File |
| :--- | :--- |
| Report | `midsem/report/report.pdf` |
| Part A preference pairs | `midsem/data/part_a_preference_pairs.jsonl` (2,000; validation pairs: `..._val.jsonl`) |
| C1 problems | `midsem/data/c1_problems.jsonl` |
| C2 pairs with verdicts | `midsem/data/c2_pairs_with_verdicts.jsonl` (4-bit judge), `midsem/data_ada/c2_pairs_with_verdicts_fp32.jsonl` (fp32 judge) |

## 1. Setup (once)

1. **Get the code.** Clone the repository. Every command below runs from its top folder (the one that contains
   `midsem/`; for the Moodle zip, the unzipped folder), with `PYTHONPATH=.`:
   ```bash
   git clone https://github.com/Jay-G14/Lma-Midsem_Makeup && cd Lma-Midsem_Makeup
   export PYTHONPATH=.
   ```
2. **Get the large files.** Download the Drive folder above, unzip it, copy its contents into `midsem/` (same paths),
   and check that every file arrived intact:
   ```bash
   cp -r drive_upload/project drive_upload/checkpoints drive_upload/checkpoints_ada midsem/   # the downloaded Drive folder
   (cd midsem && sha256sum -c LARGE_FILES.sha256)  # 72 x OK
   (cd midsem/project && sha256sum -c SHA256SUMS)  # the copied project files, 20 x OK
   ```
3. **Python environment.** Python 3.10 or newer with a CUDA build of PyTorch, then
   `pip install -r midsem/requirements.txt` (torch, tokenizers, transformers, bitsandbytes, accelerate, matplotlib,
   seaborn). The reported runs used torch 2.11 and transformers 5.17. No API keys are needed anywhere. The models in
   Part C (Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct) are open and download from the Hugging Face Hub on first use.
4. **For the report:** a TeX distribution with `xelatex`.

Hardware: Part A's single-seed pass and Part C each run on one 6 GB GPU. The multi-seed study takes about 8
GPU-hours (it fills as many GPUs as you give it), and the fp32 judge needs about 13 GB of GPU memory, one large
GPU or several.

## 2. Run everything, top to bottom

Run the steps in this order; each step uses the outputs of the steps before it.

| Step | Command | Time | Writes |
| :--- | :--- | :--- | :--- |
| 1. Part A, single-seed pass | `bash midsem/run_part_a.sh` | ~1.5 h, 1 GPU | test sets and preference pairs in `data/`, `results/`, `checkpoints/`, `figures/fig_a*` |
| 2. Part A, multi-seed study (the report's tables) | `python3 midsem/ada/orchestrate.py --gpus 4` (set `--gpus` to the GPUs you have) | ~8 GPU-h | `results_ada/`, `checkpoints_ada/`, `logs_ada/` |
| 3. Summarise the study | `python3 midsem/part_a/aggregate_ada.py` | seconds | `results_ada/summary.{json,md}`, `figures/fig_ada_*` |
| 4. Part B | `python3 midsem/part_b_calculations.py` | seconds | `results/part_b.json` (re-checks the arithmetic of the written answers in the report) |
| 5. Part C: C1 and the 4-bit judge | `bash midsem/run_part_c.sh` | ~25 min, 1 GPU | `data/c1_*`, `data/c2_pairs_with_verdicts.jsonl`, `results/c1_results.json`, `results/c2_results.json` |
| 6. Part C: the fp32 judge | `MIDSEM_STUDY=1 python3 midsem/part_c/c2_judge.py --judge unsloth/Llama-3.2-3B-Instruct --pairs-in midsem/data/c2_pairs_with_verdicts.jsonl --dtype float32 --device-map auto --tag _fp32` | ~10 min | `results_ada/c2_results_fp32.json`, `data_ada/c2_pairs_with_verdicts_fp32.jsonl` |
| 7. Figures and report | `cd midsem/report && python3 make_numbers.py && python3 make_appendix.py && python3 make_figures.py && xelatex report.tex && xelatex report.tex` | ~1 min | `report/report.pdf`, `figures/fig_eval_*` |
| 8. Moodle zip | `bash midsem/make_zip.sh` | seconds | `midsem/2023113001_alternativemidsem.zip` |

Evaluation, Part B and Part C's decoding are deterministic: rerunning them reproduces the shipped files exactly,
which `git diff` confirms (a second full C1 run reproduced all 340 replies). Training reruns (steps 1–2) match on the
same GPU and software; on a different GPU model the scores can move by a few tenths of a point. Step 2 skips any task
whose output already exists, so with the Drive files in place it only rebuilds a small cache. To recompute the study
from scratch, move `results_ada/` and `checkpoints_ada/` aside first.

## 3. Reproduce one part at a time

### Check the reported numbers without training (needs the Drive files)

Re-score any saved model on validation and the three test sets. The scores are written to the model's
`eval_<name>.json`, which `git diff` then compares with the shipped one.
```bash
python3 midsem/part_a/evaluate_models.py --models base full_ft                    # Phase 2 / Phase 3 checkpoints -> results/
python3 midsem/part_a/evaluate_models.py --models lora_r16 dpo_beta0.5            # single-seed pass -> results/
MIDSEM_STUDY=1 python3 midsem/part_a/evaluate_models.py --models ada_r16_lr1e-3_s43 ada_dpo_b0.5_lr3e-5_s42_step500   # study -> results_ada/
```
A study model's name is its folder in `checkpoints_ada/`: `ada_r<rank>_lr<LR>_s<seed>` for LoRA and
`ada_dpo_b<beta>_lr<LR>_s<seed>_step<step>` for DPO. Drift perplexity on the whole validation split:
`python3 midsem/part_a/ppl_full.py --models base full_ft`.

### Part A, step by step (single-seed pass; `run_part_a.sh` runs these in order)

| Step | Command | Output (under `midsem/`) |
| :--- | :--- | :--- |
| A0 three test sets | `python3 midsem/part_a/build_test_sets.py` | `data/test_seen_entities.jsonl`, `data/test_unseen_wording.jsonl`, `results/test_sets_audit.json` |
| A1 LoRA counts, step-0 identity | `python3 midsem/part_a/lora_sanity.py` | `results/lora_sanity.json` |
| A2 LR probe (r = 4) | `bash midsem/part_a/run_lr_probe.sh; python3 midsem/part_a/choose_lr.py` | `results/lr_choice.json` |
| A2 rank sweep | `python3 midsem/part_a/lora_sft.py --rank {1,16} --lr 1e-3` (r = 4 comes from the probe) | `results/lora_r*_history.json`, `checkpoints/lora_r*/` |
| A2 full-finetune seeds | `bash midsem/part_a/run_fullft_seeds.sh` (the project's Phase 3 script, seeds 43 and 44) | `checkpoints/fullft_s4{3,4}/`, `results/eval_fullft_s4{3,4}.json` |
| A2 memory | `python3 midsem/part_a/memory_profile.py` | `results/memory_profile.json` |
| A2 evaluation | `python3 midsem/part_a/evaluate_models.py --models base full_ft lora_r1 lora_r4 lora_r16 --select-best` | `results/eval_*.json`, `results/best_lora.json` |
| A3 preference pairs | `python3 midsem/part_a/build_preference_pairs.py` | `data/part_a_preference_pairs.jsonl` (+ `_val`), `results/preference_pairs_audit.json` |
| A4 DPO | `python3 midsem/part_a/dpo.py --beta {0.1,0.5}`, then `evaluate_models.py --models dpo_beta0.1 dpo_beta0.5` | `results/dpo_beta*_history.json`, `results/eval_dpo_*.json` |
| A4 DPO unit checks | `python3 midsem/part_a/test_dpo.py` (loads the pretrained checkpoint from Drive) | prints the three checks (step-0 loss = ln 2, batching, shared prefix) |
| A5 figures and tables | `python3 midsem/part_a/plots.py; python3 midsem/part_a/summarize.py` | `figures/fig_a*`, `results/part_a_summary.md` |

### Part A, the multi-seed study

`midsem/ada/orchestrate.py --gpus N` runs the whole study as a task graph on N GPUs, one task per GPU, and makes
every selection on the validation split only:
1. the LR grid (3 ranks × 3 LRs, seed 42) and each rank's LR by validation loss;
2. seeds 43 and 44 at that LR;
3. every evaluation and the best LoRA model;
4. the DPO sweep (2 β × 3 LRs × checkpoints every 100 steps) and each β's (LR, step) by validation accuracy;
5. two more seeds of the chosen DPO configuration;
6. the fixed protocol (LR 1e-4, 500 steps);
7. whole-split drift perplexity.

It needs steps A0 and A3 (test sets and preference pairs) and writes to `results_ada/`, `checkpoints_ada/` and
`logs_ada/`. Its choices are frozen in `results_ada/ada_frozen_*.json` and listed in `results_ada/ada_selections.json`.
`aggregate_ada.py` turns it into `results_ada/summary.md`.

### Part B

All six answers are worked by hand in the report (Part B). `python3 midsem/part_b_calculations.py` re-checks every
number and prints the intermediate steps.

### Part C, step by step (`run_part_c.sh` runs C1.1–C2)

| Step | Command | Output |
| :--- | :--- | :--- |
| C1.1 problems | `python3 midsem/part_c/c1_problems.py` | `data/c1_problems.jsonl` |
| C1.2–4 three conditions | `python3 midsem/part_c/c1_run.py` (add `--out-suffix _rerun` to write a second run beside the first and compare) | `data/c1_outputs.jsonl`, `results/c1_results.json` |
| C1.3 checker self-tests | `python3 -c "import midsem.part_c.checker"` | asserts 0.05 = 1/20 = 5E-02 = ०.०५ = 5% |
| C2 pairs + 4-bit judge | `python3 midsem/part_c/c2_judge.py` (rebuilds the 30 pairs from the C1 replies) | `data/c2_pairs_with_verdicts.jsonl`, `results/c2_results.json` |
| C2 fp32 judge | step 6 of the table above | `results_ada/c2_results_fp32.json`, `data_ada/c2_pairs_with_verdicts_fp32.jsonl` |

### Figures and report only (no GPU, seconds)

`python3 midsem/part_a/aggregate_ada.py`, then step 7. `make_figures.py` reads saved results only.

## Headline results

Part A, multi-seed study (answer EM %, mean ± sd over 3 seeds; full table: `midsem/results_ada/summary.md`):

| Model | Seen entities | Unseen entities | Unseen wording | Drift PPL (500 docs / full split) |
| :--- | ---: | ---: | ---: | ---: |
| Pretrained base | – | 0 (Phase 3) | – | 57.95 / 49.28 |
| Full FT (Phase 3 recipe, 3 seeds) | 81.1 ± 4.6 | 78.6 ± 5.0 | 44.5 ± 2.1 | 63.35 / 58.62 (seed 42) |
| LoRA r = 1 (LR 3e-4) | 37.1 ± 4.2 | 35.5 ± 2.4 | 31.5 ± 3.0 | 60.58 / 52.46 |
| LoRA r = 4 (LR 1e-3) | 48.7 ± 5.1 | 51.9 ± 1.2 | 36.1 ± 2.6 | 60.65 / 52.89 |
| LoRA r = 16 (LR 1e-3) | 78.0 ± 4.6 | 76.9 ± 5.9 | 51.4 ± 6.2 | 61.36 / 53.84 |
| DPO β = 0.1, chosen (LR 1e-5, 100 steps) | 80.9 ± 0.4 | 80.7 ± 0.3 | 55.6 ± 0.4 | 61.30 / 53.76 |
| DPO β = 0.5, chosen (LR 3e-5, 500 steps) | 82.3 ± 0.7 | 81.0 ± 0.8 | 49.0 ± 1.4 | 61.34 / 53.79 |
| DPO β = 0.1, fixed (LR 1e-4, 500 steps) | 52.0 | 46.8 | 29.3 | 61.96 / 54.12 |
| DPO β = 0.5, fixed (LR 1e-4, 500 steps) | 81.5 | 79.7 | 42.2 | 61.48 / 53.87 |

DPO starts from the best LoRA model on validation (r = 16, seed 43: 80.8 / 80.9 / 56.4). The DPO loss at step 0 is
0.693147 = ln 2 in every run.

Part C, with Qwen2.5-1.5B-Instruct on Hindi problems:
- C1, standard problems: direct / CoT / self-consistency = 33.3 / 40.0 / 66.7 % (self-consistency over 3 sampling
  seeds: 68.9 ± 3.8 %).
- C1, the 5 modified problems: 1/5, 1/5 and 0/5 in every seed.
- C2, Llama-3.2-3B judge, 30 pairs × 2 orders:
  - 4-bit: position-bias rate 66.7 %, first-shown answer chosen 80 %, κ = 0.14.
  - fp32: position-bias rate 43.3 %, longer answer wins 78.3 %, κ = 0.00.

## Files

| Path | What it is |
| :--- | :--- |
| `midsem/part_a/lora.py` | LoRA written from scratch (frozen W0, trainable A, B, α/r scaling), injected into W_Q, W_K, W_V, W_O |
| `midsem/part_a/dpo.py` | DPO loss and training loop written from scratch (no peft, no trl) |
| `midsem/part_a/common.py` | paths, model loading, the Phase 3 evaluation on the three test sets, drift perplexity |
| `midsem/part_a/build_test_sets.py` | seen-entity and unseen-wording test sets from the project generator, with a leakage audit |
| `midsem/part_a/build_preference_pairs.py` | 2,000 training + 400 validation pairs (chosen = gold reply, rejected = one-step-wrong reply) |
| `midsem/part_a/ppl_full.py` | drift perplexity on the whole validation split (Phase 2 packing) |
| `midsem/ada/orchestrate.py` | the multi-seed study's task-graph scheduler (any machine with N GPUs) |
| `midsem/ada/run.sh`, `midsem/ada/upload_assets.py` | the batch-job wrapper that ran the study on a GPU cluster; not needed to reproduce |
| `midsem/part_a/aggregate_ada.py` | study results → `results_ada/summary.{json,md}` and `figures/fig_ada_*.png` |
| `midsem/report/make_figures.py` | the evaluation figures (`figures/fig_eval_*.png`) and the example-output tables, from saved results only |
| `midsem/plot_style.py` | one palette for every figure, checked for colour-blind safety |
| `midsem/part_c/checker.py` | numeric answer checker (exact fractions), self-tests run on import |
| `midsem/part_c/c1_problems.py` | the 20 Hindi problems (15 standard, 5 modified; M01 is the class's drifting-boat example) |
| `midsem/part_c/c1_run.py`, `midsem/part_c/c2_judge.py` | C1's three conditions with Qwen2.5-1.5B-Instruct; the C2 judge, Llama-3.2-3B-Instruct |
| `midsem/data/`, `midsem/data_ada/` | test sets, preference pairs, C1 problems and replies, C2 pairs with verdicts |
| `midsem/results/`, `midsem/logs/` | single-seed pass and Part C: every number and log |
| `midsem/results_ada/`, `midsem/logs_ada/` | multi-seed study: every number and log (and the fp32 judge) |
| `midsem/checkpoints/`, `midsem/checkpoints_ada/` | LoRA and DPO adapters (A and B only) and the extra full-finetune seeds (on Drive) |
| `midsem/project/` | copies of the individual project's files that Part A uses; list and checksums in `midsem/project/README.md` |
| `midsem/__init__.py` | puts `midsem/project/` first on `sys.path`, so `import hindi` loads the copies |
| `midsem/LARGE_FILES.sha256` | the 72 files on Drive, with checksums |
