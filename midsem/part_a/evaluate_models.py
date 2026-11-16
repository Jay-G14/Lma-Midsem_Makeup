#!/usr/bin/env python3
"""Evaluates Part A models on the three test sets (+ validation) and measures drift perplexity.

Models (pass any subset with --models):
  base          Phase 2 pretrained Model H (drift PPL only: it never writes "उत्तर:", 0% in Phase 3)
  full_ft       Phase 3 full finetune, midsem/project/hindi/checkpoints/finetuned_model.pt
  lora_r{1,4,16} best epoch of each LoRA run (selected on reasoning-validation loss)
  dpo_beta{0.1,0.5} DPO policies started from the best LoRA model

--select-best picks the best LoRA rank on VALIDATION answer accuracy (ties: lower validation loss)
and writes results/best_lora.json, which dpo.py reads. Test sets are never used for selection.
"""

import argparse
from pathlib import Path

import torch

import midsem  # noqa: F401  (puts midsem/project/, the copied project code, first on sys.path)
from hindi.eval.eval_phase3 import chance_accuracy, load_model, load_samples
from midsem.part_a.common import (CKPT_DIR, MIDSEM, FULL_FT_CKPT, LORA_ALPHA, REASONING_DIR, RESULTS_DIR, TEST_SETS, device, evaluate_all_sets,
                                  load_base, load_lora_model, pretrain_ppl, read_json, tokenizer, write_json)

RANKS = (1, 4, 16)


def build(name: str, dev):
    """Returns (model, description) for a model name: base, full_ft, or any run directory under
    midsem/checkpoints/ holding best_adapter.pt (LoRA SFT) or adapter.pt (DPO)."""
    if name == "base":
        return load_base(dev), "Phase 2 pretrained checkpoint"
    if name == "full_ft":
        return load_model(str(FULL_FT_CKPT), dev)[0], "Phase 3 full finetune (all 26.27M parameters)"
    if name.startswith("fullft_s"):          # extra full-finetune seeds, trained by part_a/run_fullft_seeds.sh
        path = CKPT_DIR / name / "finetuned_model.pt"
        return load_model(str(path), dev)[0], f"full finetune, Phase 3 recipe, seed {name[len('fullft_s'):]}"
    for fname in ("best_adapter.pt", "adapter.pt"):
        path = CKPT_DIR / name / fname
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            hp = state.get("hparams", {})
            kind = f"DPO beta={state['beta']}" if "beta" in state else "LoRA"
            desc = f"{kind} r={state['rank']} lr={hp.get('lr', '?')} seed={hp.get('seed', '?')} ({name})"
            return load_lora_model(dev, state["rank"], state["alpha"], path), desc
    raise ValueError(f"no adapter for {name} under {CKPT_DIR}")


def select_best():
    """Best LoRA rank by validation answer EM (then validation reasoning loss)."""
    rows = []
    for r in RANKS:
        ev = read_json(RESULTS_DIR / f"eval_lora_r{r}.json")
        hist = read_json(RESULTS_DIR / f"lora_r{r}_history.json")
        rows.append({"rank": r, "val_answer_em": ev["sets"]["val"]["answer_em"], "val_reasoning_loss": hist["best_val_reasoning_loss"]})
    best = max(rows, key=lambda row: (row["val_answer_em"], -row["val_reasoning_loss"]))
    out = {"rank": best["rank"], "alpha": LORA_ALPHA, "adapter_path": str((CKPT_DIR / f"lora_r{best['rank']}" / "best_adapter.pt").relative_to(MIDSEM)),
           "criterion": "highest validation answer EM, ties broken by lower validation reasoning loss", "candidates": rows}
    write_json(RESULTS_DIR / "best_lora.json", out)
    print(f"Best LoRA: r={best['rank']} (val answer EM {best['val_answer_em']}%)")


def main():
    p = argparse.ArgumentParser(description="Three-test-set evaluation and drift PPL for Part A models")
    p.add_argument("--models", nargs="+", default=["base", "full_ft"] + [f"lora_r{r}" for r in RANKS])
    p.add_argument("--select-best", action="store_true")
    p.add_argument("--keep-generations", type=int, default=6)
    p.add_argument("--sets", nargs="+", default=None, help="subset of val / seen_entities / unseen_entities / unseen_wording (default: all)")
    args = p.parse_args()
    dev, tok = device(), tokenizer()
    for name in args.models:
        model, desc = build(name, dev)
        result = {"model": name, "description": desc, "pretrain_val_ppl_500docs": round(pretrain_ppl(model, tok, dev), 2)}
        if name != "base":
            result["sets"] = evaluate_all_sets(model, tok, dev, include_val=True, keep_generations=args.keep_generations, only=args.sets)
        write_json(RESULTS_DIR / f"eval_{name}.json", result)
        acc = {k: v["answer_em"] for k, v in result.get("sets", {}).items()}
        print(f"{name:12s} PPL {result['pretrain_val_ppl_500docs']:6.2f} | answer EM {acc}")
        del model
        torch.cuda.empty_cache()
    if args.select_best:
        select_best()
    chance = {k: chance_accuracy(load_samples(str(v)))["accuracy"] for k, v in {**TEST_SETS, "val": REASONING_DIR / "val.jsonl"}.items()}
    write_json(RESULTS_DIR / "chance_levels.json", chance)


if __name__ == "__main__":
    main()
