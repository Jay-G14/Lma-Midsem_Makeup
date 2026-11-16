#!/usr/bin/env python3
"""Part A figures (plotting only; all numbers are read from midsem/results/*.json).

  fig_a2_lora_curves.png     reasoning-validation loss and drift PPL per epoch: LoRA r = 1, 4, 16 vs. full FT
  fig_a2_rank_sweep.png      answer EM on the three test sets: full FT vs. LoRA r = 1, 4, 16
  fig_a4_dpo_margin.png      mean chosen-minus-rejected log-ratio margin over DPO training, beta = 0.1 vs. 0.5
  fig_a4_dpo_logratios.png   DPO loss and the chosen / rejected log-ratios separately
  fig_a5_drift.png           pretraining-validation PPL: base, full FT, best LoRA, both DPO models
  fig_a_lr_probe.png         LR probe at r = 4 (appendix)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from midsem.part_a.common import FIG_DIR, PROJECT, RESULTS_DIR, TEST_SET_LABELS, read_json

COLORS = {"full_ft": "#4d4d4d", "lora_r1": "#9ecae1", "lora_r4": "#4292c6", "lora_r16": "#08519c",
          "dpo_beta0.1": "#d95f02", "dpo_beta0.5": "#7570b3", "base": "#a6a6a6"}
LABELS = {"full_ft": "Full FT (Phase 3)", "lora_r1": "LoRA r=1", "lora_r4": "LoRA r=4", "lora_r16": "LoRA r=16",
          "dpo_beta0.1": "DPO β=0.1", "dpo_beta0.5": "DPO β=0.5", "base": "Pretrained base"}


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)
    print("saved", FIG_DIR / name)


def smooth(y, k=10):
    """Trailing k-step moving average (NaN for the first k-1 steps)."""
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.concatenate([np.full(k - 1, np.nan), np.convolve(y, np.ones(k) / k, mode="valid")])


def lora_curves():
    full = read_json(PROJECT / "hindi/checkpoints/finetune_history.json")["epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    runs = [("full_ft", full)] + [(f"lora_r{r}", read_json(RESULTS_DIR / f"lora_r{r}_history.json")["history"]) for r in (1, 4, 16)
                                  if (RESULTS_DIR / f"lora_r{r}_history.json").exists()]
    for name, hist in runs:
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["val_reasoning_loss"] for h in hist], marker="o", color=COLORS[name], label=LABELS[name])
        axes[1].plot(ep, [h["pretrain_ppl"] for h in hist], marker="o", color=COLORS[name], label=LABELS[name])
    axes[0].set_yscale("log")
    axes[0].set(title="Reasoning validation loss", xlabel="Epoch", ylabel="Cross-entropy on reasoning + answer (log scale)")
    axes[1].set(title="Drift: pretraining-validation perplexity", xlabel="Epoch", ylabel="PPL (500 validation documents)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Model H finetuning on the Phase 3 reasoning data: LoRA (W_Q, W_K, W_V, W_O) vs. full finetuning")
    save(fig, "fig_a2_lora_curves.png")


def rank_sweep():
    names = [n for n in ("full_ft", "lora_r1", "lora_r4", "lora_r16") if (RESULTS_DIR / f"eval_{n}.json").exists()]
    sets = list(TEST_SET_LABELS)
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(names)
    for i, n in enumerate(names):
        ev = read_json(RESULTS_DIR / f"eval_{n}.json")["sets"]
        vals = [ev[s]["answer_em"] for s in sets]
        bars = ax.bar(np.arange(len(sets)) + (i - (len(names) - 1) / 2) * width, vals, width, color=COLORS[n], label=LABELS[n])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    chance = read_json(RESULTS_DIR / "chance_levels.json") if (RESULTS_DIR / "chance_levels.json").exists() else None
    if chance:
        for k, s in enumerate(sets):
            ax.hlines(chance[s], k - 0.45, k + 0.45, colors="black", linestyles=":", linewidth=1.2,
                      label="chance (uniform guess over candidates)" if k == 0 else None)
    ax.set_xticks(np.arange(len(sets)), [TEST_SET_LABELS[s] for s in sets])
    ax.set(title="Answer exact match on the three Phase 3 test sets", xlabel="Test set", ylabel="Answer EM (%)", ylim=(0, 105))
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_a2_rank_sweep.png")


def dpo_curves():
    runs = [(b, read_json(RESULTS_DIR / f"dpo_beta{b}_history.json")) for b in ("0.1", "0.5") if (RESULTS_DIR / f"dpo_beta{b}_history.json").exists()]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for b, h in runs:
        name = f"dpo_beta{b}"
        steps = [r["step"] for r in h["train_log"]]
        margin = [r["margin"] for r in h["train_log"]]
        ax.plot(steps, margin, color=COLORS[name], alpha=0.25, linewidth=0.8)
        ax.plot(steps, smooth(margin), color=COLORS[name], linewidth=2, label=f"β={b}: training batches (10-step mean)")
        ax.plot([r["step"] for r in h["val_log"]], [r["margin"] for r in h["val_log"]], "o--", color=COLORS[name],
                markersize=3, linewidth=1, label=f"β={b}: 400 validation pairs")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(title="DPO: mean chosen-minus-rejected log-ratio margin", xlabel="Training step (batch of 8 pairs)",
           ylabel="[log πθ/πref](chosen) − [log πθ/πref](rejected)  (nats)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "fig_a4_dpo_margin.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for b, h in runs:
        name = f"dpo_beta{b}"
        steps = [r["step"] for r in h["train_log"]]
        axes[0].plot(steps, smooth([r["loss"] for r in h["train_log"]]), color=COLORS[name], label=f"β={b}")
        axes[1].plot(steps, smooth([r["chosen_logratio"] for r in h["train_log"]]), color=COLORS[name], label=f"β={b}: chosen")
        axes[1].plot(steps, smooth([r["rejected_logratio"] for r in h["train_log"]]), color=COLORS[name], linestyle="--", label=f"β={b}: rejected")
    axes[0].axhline(np.log(2), color="black", linewidth=0.6, linestyle=":", label="log 2 (step 0)")
    axes[0].set(title="DPO loss (10-step mean)", xlabel="Training step", ylabel="−log σ(β · margin)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set(title="Log-ratios log πθ(y|x) − log πref(y|x) (10-step mean)", xlabel="Training step", ylabel="nats")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    save(fig, "fig_a4_dpo_logratios.png")


def drift():
    best = read_json(RESULTS_DIR / "best_lora.json") if (RESULTS_DIR / "best_lora.json").exists() else None
    names = ["base", "full_ft"] + ([f"lora_r{best['rank']}"] if best else []) + ["dpo_beta0.1", "dpo_beta0.5"]
    names = [n for n in names if (RESULTS_DIR / f"eval_{n}.json").exists()]
    vals = [read_json(RESULTS_DIR / f"eval_{n}.json")["pretrain_val_ppl_500docs"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(names)), vals, color=[COLORS[n] for n in names],
                  label=[LABELS[n] + (" (best LoRA)" if best and n == f"lora_r{best['rank']}" else "") for n in names])
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(names)), [LABELS[n] for n in names], fontsize=8)
    ax.set(title="Drift: Model H pretraining-validation perplexity", xlabel="Model", ylabel="PPL (500 validation documents)",
           ylim=(min(vals) * 0.9, max(vals) * 1.05))
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_a5_drift.png")


def lr_probe():
    files = sorted(RESULTS_DIR.glob("lrprobe_r4_lr*_history.json"), key=lambda p: read_json(p)["lr"])
    if not files:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for f in files:
        h = read_json(f)
        ax.plot([e["epoch"] for e in h["history"]], [e["val_reasoning_loss"] for e in h["history"]], marker="o", label=f"lr={h['lr']:g}")
    ax.set_yscale("log")
    ax.set(title="LoRA LR probe (r=4, α=16, 2 epochs)", xlabel="Epoch", ylabel="Reasoning validation loss (log)")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "fig_a_lr_probe.png")


def main():
    lora_curves()
    rank_sweep()
    dpo_curves()
    drift()
    lr_probe()


if __name__ == "__main__":
    main()
