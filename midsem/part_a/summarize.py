#!/usr/bin/env python3
"""Collects every Part A number into midsem/results/part_a_summary.md (tables used in the report)."""

from midsem.part_a.common import RESULTS_DIR, TEST_SET_LABELS, read_json

SETS = list(TEST_SET_LABELS)


def maybe(name):
    path = RESULTS_DIR / name
    return read_json(path) if path.exists() else None


def main():
    lines = ["# Part A summary (variant 1: Model H, LoRA on W_Q, W_K, W_V, W_O, DPO beta = 0.5 and 0.1)", ""]

    sanity = maybe("lora_sanity.json")
    if sanity:
        lines += ["## A1 Trainable parameters and step-0 identity", "",
                  "| r | trainable | total | share of model | step-0 max abs logit diff | bit-identical | grad norm A / B at step 0 |",
                  "|---|---|---|---|---|---|---|"]
        for r, v in sanity["ranks"].items():
            lines.append(f"| {r} | {v['trainable']:,} | {v['total']:,} | {v['trainable_share_pct']:.4f}% | {v['step0_max_abs_logit_diff']:.1e} | "
                         f"{v['step0_logits_bit_identical']} | {v['step0_grad_norm_A']:.1e} / {v['step0_grad_norm_B']:.3f} |")
        lines.append("")

    lr = maybe("lr_choice.json")
    if lr:
        lines += ["## LR probe (r = 4, 5 epochs)", "", "| lr | best epoch | best val reasoning loss | PPL after epoch 5 |", "|---|---|---|---|"]
        lines += [f"| {p['lr']:g} | {p['best_epoch']} | {p['best_val_reasoning_loss']:.4f} | {p['ppl_by_epoch'][-1]:.2f} |" for p in lr["probe"]]
        lines += [f"", f"Chosen LR: {lr['chosen_lr']:g}", ""]

    rows = [("Full FT (Phase 3)", "full_ft")] + [(f"LoRA r={r}", f"lora_r{r}") for r in (1, 4, 16)] + \
           [("DPO beta=0.1", "dpo_beta0.1"), ("DPO beta=0.5", "dpo_beta0.5")]
    lines += ["## A2 / A4 Answer EM (chain EM) on the three test sets", "",
              "| model | " + " | ".join(TEST_SET_LABELS[s] for s in SETS) + " | val | drift PPL |", "|---" * (len(SETS) + 3) + "|"]
    for label, name in rows:
        ev = maybe(f"eval_{name}.json")
        if not ev:
            continue
        cells = [f"{ev['sets'][s]['answer_em']:.1f} ({ev['sets'][s]['chain_em']:.1f})" for s in SETS]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {ev['sets']['val']['answer_em']:.1f} | {ev['pretrain_val_ppl_500docs']:.2f} |")
    base = maybe("eval_base.json")
    if base:
        lines.append(f"| Pretrained base | - | 0 (Phase 3) | - | - | {base['pretrain_val_ppl_500docs']:.2f} |")
    lines.append("")

    lines += ["## Per-task answer EM, unseen entities", "", "| model | transitive | numeric_extremum | numeric_pair | verbal |", "|---|---|---|---|---|"]
    for label, name in rows:
        ev = maybe(f"eval_{name}.json")
        if ev:
            t = ev["sets"]["unseen_entities"]["task_answer_em"]
            lines.append(f"| {label} | {t['transitive']:.1f} | {t['numeric_extremum']:.1f} | {t['numeric_pair']:.1f} | {t['verbal_comparison']:.1f} |")
    lines.append("")

    mem = maybe("memory_profile.json")
    if mem:
        lines += [f"## A2 Memory ({mem['gpu']}, batch 8, {mem['steps']} identical steps)", "",
                  "| config | trainable | weights MB | grads MB | AdamW state MB | peak MB | steps/s |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['config']} | {r['trainable_params']:,} | {r['weights_mb']} | {r['grad_mb']} | {r['optimizer_mb']} | {r['peak_mb']} | {r['steps_per_second']} |"
                  for r in mem["rows"]]
        lines.append("")

    for r in (1, 4, 16):
        h = maybe(f"lora_r{r}_history.json")
        if h:
            lines.append(f"- LoRA r={r}: lr {h['lr']:g}, best epoch {h['best_epoch']}, best val reasoning loss {h['best_val_reasoning_loss']:.4f}; "
                         f"per-epoch (val loss, PPL): " + ", ".join(f"{e['epoch']}: ({e['val_reasoning_loss']:.4f}, {e['pretrain_ppl']:.2f})" for e in h["history"]))
    best = maybe("best_lora.json")
    if best:
        lines += ["", f"Best LoRA (validation answer EM): r={best['rank']}", ""]
    ah, ae = maybe("ablation_r1_lr2.5e-4_history.json"), maybe("eval_ablation_r1_lr2.5e-4.json")
    if ah and ae:
        lines += [f"Control r=1 at LR 2.5e-4 (alpha/r x LR matched to r=4): best val loss {ah['best_val_reasoning_loss']:.4f}, "
                  f"val EM {ae['sets']['val']['answer_em']:.1f}, unseen-entity EM {ae['sets']['unseen_entities']['answer_em']:.1f}", ""]

    for b in ("0.1", "0.5"):
        h = maybe(f"dpo_beta{b}_history.json")
        if h:
            v0, vN = h["val_log"][0], h["val_log"][-1]
            ft = h["final_train_set"]
            lines += [f"## DPO beta={b}", "",
                      f"- step-0 loss: first batch {h['step0']['first_batch_loss']:.6f}, whole training set {h['step0']['train_set_loss']:.6f} (log 2 = {h['step0']['log2']:.6f})",
                      f"- steps {h['steps']}; final training-set loss {ft['loss']:.4f}, margin {ft['margin']:.2f}, chosen log-ratio {ft['chosen_logratio']:.2f}, "
                      f"rejected log-ratio {ft['rejected_logratio']:.2f}, preference acc {ft['preference_accuracy']:.3f}",
                      f"- validation pairs: margin {v0['margin']:.2f} -> {vN['margin']:.2f}, loss {v0['loss']:.4f} -> {vN['loss']:.4f}, "
                      f"preference acc {v0['preference_accuracy']:.3f} -> {vN['preference_accuracy']:.3f}", ""]

    pairs = maybe("preference_pairs_audit.json")
    if pairs:
        lines += ["## A3 Preference pairs", "", f"- {pairs['train_pairs']} training pairs ({pairs['eligible_train_samples']} eligible training samples), "
                  f"{pairs['val_pairs']} validation pairs", f"- task mix {pairs['task_mix']}", f"- question types {pairs['question_types']}",
                  f"- names in train pool {pairs['names_in_train_pool_pct']}%, test-pool names {pairs['names_in_test_pool']}, val-pool names {pairs['names_in_val_pool']}",
                  f"- prompt sentences with a training template {pairs['prompt_sentences_with_training_template_pct']}%, prompts equal to a test question {pairs['prompts_equal_to_a_test_question']}", ""]

    audit = maybe("test_sets_audit.json")
    if audit:
        lines += ["## Test-set audit", "", "| set | size | names in train pool | names in test pool | sentences with a training template | questions in train |", "|---|---|---|---|---|---|"]
        lines += [f"| {s['set']} | {s['size']} | {s['names_in_train_pool_pct']}% | {s['names_in_test_pool_pct']}% | {s['sentences_with_training_template_pct']}% | {s['questions_also_in_train']} |"
                  for s in audit["sets"]]
    (RESULTS_DIR / "part_a_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
