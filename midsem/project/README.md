# midsem/project/ — the individual project's files that the make-up mid-sem uses

Copied unchanged from the individual-project repository (commit `5536fdd`, "project finished") on 16 November 2026, so
that `midsem/` runs on its own. `import midsem` puts this folder first on `sys.path`, so `import hindi...` always loads
these copies. `SHA256SUMS` lists every file (`cd midsem/project && sha256sum -c SHA256SUMS`).

| File | Used for |
| :--- | :--- |
| `hindi/model/{config,transformer}.py` | the Model H decoder (LoRA wraps its W_Q, W_K, W_V, W_O) |
| `hindi/train/finetune.py`, `hindi/train/dataset.py` | the Phase 3 dataset (question masking, `</s>`, 15% replay) and the full-finetune script (`part_a/run_fullft_seeds.sh`) |
| `hindi/eval/eval_phase3.py`, `hindi/eval/evaluate.py` | prompts, greedy generation, answer extraction, perplexity |
| `scripts/generate_reasoning_data.py` | the Phase 3 generator, reused for the seen-entity and unseen-wording test sets |
| `hindi/tokenizer/tokenizer.json`, `hindi/configs/model_config.json` | the 20k BPE tokenizer and the model config |
| `hindi/checkpoints/final_model.pt` | the Phase 2 pretrained Model H, the starting point of Part A |
| `hindi/checkpoints/finetuned_model.pt`, `finetune_history.json` | the Phase 3 full finetune and its training curve |
| `hindi/data/reasoning/{train,val,test}.jsonl`, `dataset_stats.json` | the Phase 3 reasoning data (test = the unseen-entity test set) |
| `hindi/data/processed/val.jsonl` | the pretraining validation split (drift perplexity) |
| `hindi/data/processed/train.jsonl` | the **first 3,000 documents** of the project's 4.8 GB pretraining corpus: the 15% replay buffer reads documents in order and stops after 2,117, so results are identical to using the full file |
| `requirements.txt` | the project's Python requirements |

Not in the Moodle zip (size): `hindi/checkpoints/*.pt` and `hindi/data/processed/`. The project's links in
`midsem/README.md` have them (for `processed/train.jsonl`, `head -n 3000` of the project file is enough).
