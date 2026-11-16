#!/usr/bin/env python3
"""Evaluation figures beyond the six that part_a/aggregate_ada.py draws, plus the example-output panels.

Reads only saved results (no model is loaded), so it runs in seconds; run it after aggregate_ada.py:
    python3 midsem/report/make_figures.py
Writes midsem/figures/fig_eval_*.png and midsem/report/qualitative.tex. The example outputs are typeset by
xelatex, because matplotlib cannot shape Devanagari; no figure contains Hindi text.

Part A  fig_eval_a_tasks     answer EM per task type (model x task, for each test set)
        fig_eval_a_outcomes  what every answer was: exact gold chain / right answer, other chain / wrong / none
        fig_eval_a_rank      accuracy vs trainable parameters with every seed; LR grid; memory; throughput
        fig_eval_a_dpo       DPO's own metric (held-out pairs ranked correctly) vs validation answer EM
        fig_eval_a_drift     drift PPL vs unseen-entity accuracy
Part B  fig_eval_b           LSH S-curve, sampling shares vs alpha, loss at fixed compute vs N, 8B memory
Part C  fig_eval_c1_tokens   accuracy vs generated tokens; what the model answers on the modified problems
        fig_eval_c1_grid     every C1 reply (problem x condition / sample), right or wrong
        fig_eval_c2_judge    the judge's P(prefers the CoT answer) in both orders, for every pair
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from midsem import plot_style as ps  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

plt = ps.plt
ps.apply()

MIDSEM = REPO / "midsem"
RA, RL, FIG, REPORT, DATA = MIDSEM / "results_ada", MIDSEM / "results", MIDSEM / "figures", MIDSEM / "report", MIDSEM / "data"
SETS = ["seen_entities", "unseen_entities", "unseen_wording"]
SET_LABEL = {"seen_entities": "Seen entities", "unseen_entities": "Unseen entities", "unseen_wording": "Unseen wording"}
SET_FILE = {"seen_entities": DATA / "test_seen_entities.jsonl", "unseen_entities": MIDSEM / "project/hindi/data/reasoning/test.jsonl",
            "unseen_wording": DATA / "test_unseen_wording.jsonl"}
TASKS = ["verbal_comparison", "transitive", "numeric_pair", "numeric_extremum"]
TASK_LABEL = {"verbal_comparison": "verbal", "transitive": "transitive", "numeric_pair": "pair", "numeric_extremum": "max/min"}
CHANCE = 40.3


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jsonl(p):
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


SUMMARY = jload(RA / "summary.json")
SEL = SUMMARY["selections"]


# ----------------------------------------------------------------------------------------------- Part A
def part_a_models():
    """(label, colour, eval files, hollow) for every Part A model, in the order of the report's tables."""
    rows = [("Full FT (3 seeds)", ps.FULL, [RA / "eval_full_ft.json", RL / "eval_fullft_s43.json", RL / "eval_fullft_s44.json"], False)]
    for r in (1, 4, 16):
        lr = SEL[f"lr_r{r}"]
        rows.append((f"LoRA r={r}", ps.RANK[r], [RA / f"eval_ada_r{r}_lr{lr}_s{s}.json" for s in (42, 43, 44)], False))
    for b in ("0.1", "0.5"):
        rows.append((f"DPO β={b} (chosen)", ps.BETA[b], [RA / f"eval_{m}.json" for m in SUMMARY["dpo_chosen"][b]["models"]], False))
    for b in ("0.1", "0.5"):
        rows.append((f"DPO β={b} (fixed, 1 seed)", ps.BETA[b], [RA / f"eval_{SUMMARY['dpo_fixed_protocol'][b]['name']}.json"], True))
    return rows


def seed_means(files):
    """Mean over the seeds' eval files of answer EM, chain EM, answered share and per-task EM, per test set."""
    ds = [jload(f)["sets"] for f in files]
    out = {}
    for s in SETS:
        out[s] = {k: float(np.mean([d[s][k] for d in ds])) for k in ("answer_em", "chain_em", "answered")}
        out[s]["task"] = {t: float(np.mean([d[s]["task_answer_em"][t] for d in ds])) for t in TASKS}
    return out


def fig_a_tasks():
    models = part_a_models()
    stats = [seed_means(f) for _, _, f, _ in models]
    counts = {s: {t: sum(1 for q in jsonl(SET_FILE[s]) if q["task"] == t) for t in TASKS} for s in SETS}
    fig, axes = plt.subplots(1, 4, figsize=(ps.TEXT_WIDTH_IN, 3.0), gridspec_kw={"width_ratios": [1, 1, 1, 0.045], "wspace": 0.07})
    for ax, s in zip(axes[:3], SETS):
        m = np.array([[st[s]["task"][t] for t in TASKS] for st in stats])
        im = ax.imshow(m, cmap=ps.SEQ, vmin=0, vmax=100, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color=ps.text_on(ps.SEQ(m[i, j] / 100)))
        ax.hlines([0.5, 3.5, 5.5], -0.5, len(TASKS) - 0.5, color=ps.SURFACE, linewidth=2.5)
        ax.set_xticks(range(len(TASKS)), [f"{TASK_LABEL[t]}\nn={counts[s][t]}" for t in TASKS], fontsize=6.0)
        ax.set_title(SET_LABEL[s])
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_yticks(range(len(models)), [lab for lab, *_ in models] if ax is axes[0] else [])
    cb = fig.colorbar(im, cax=axes[3])
    cb.set_label("Answer EM (%)", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=0)
    ps.save(fig, FIG / "fig_eval_a_tasks.png")


def fig_a_outcomes():
    models = part_a_models()
    stats = [seed_means(f) for _, _, f, _ in models]
    cats = [("exact gold chain", ps.CORRECT), ("right answer, other chain", ps.CORRECT_LIGHT),
            ("wrong answer", ps.WRONG), ("no answer line", ps.NONE)]
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, 2.9), sharey=True, gridspec_kw={"wspace": 0.2})
    y = np.arange(len(models))
    for ax, s in zip(axes, SETS):
        for i, st in enumerate(stats):
            ch, an, aw = st[s]["chain_em"], st[s]["answer_em"], st[s]["answered"]
            left = 0.0
            for (_, col), v in zip(cats, [ch, an - ch, aw - an, 100 - aw]):
                ax.barh(i, v, left=left, height=0.66, color=col, edgecolor=ps.SURFACE, linewidth=0.7)
                left += v
            under = ps.CORRECT_LIGHT if an - ch >= 7 else ps.CORRECT
            ax.text(an - 1.2, i, f"{an:.0f}", ha="right", va="center", fontsize=6.2, color=ps.text_on(under))
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100], ["0", "50", "100"])
        ax.grid(axis="y", visible=False)
        ax.set_title(SET_LABEL[s])
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [lab for lab, *_ in models])
    axes[0].invert_yaxis()
    handles = [Patch(facecolor=c, label=lab) for lab, c in cats]
    axes[1].set_xlabel("Share of the 1,000 questions in the set, % (answer EM printed inside)")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.96), ncol=4, fontsize=7)
    ps.save(fig, FIG / "fig_eval_a_outcomes.png")


def fig_a_rank():
    mem = jload(RL / "memory_profile.json")["rows"]
    params = {r["rank"] if r["rank"] else "full": r["trainable_params"] for r in mem}
    total = params["full"]
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.62, wspace=0.32)
    # top row: accuracy vs trainable parameters, every seed
    for k, s in enumerate(SETS):
        ax = fig.add_subplot(gs[0, k])
        xs = {1: 0, 4: 1, 16: 2, "full": 3.4}
        means = []
        for r in (1, 4, 16):
            v = SUMMARY["lora"][str(r)][s]
            for yv in v["values"]:
                ps.point(ax, xs[r], yv, ps.RANK[r], size=4.6)
            ax.plot([xs[r] - 0.28, xs[r] + 0.28], [v["mean"]] * 2, color=ps.RANK[r], linewidth=2.2, zorder=4)
            means.append(v["mean"])
        ax.plot([0, 1, 2], means, color=ps.MUTED, linewidth=0.8, zorder=1)
        f = SUMMARY["full_ft_seeds"][s]
        for yv in f["values"]:
            ps.point(ax, xs["full"], yv, ps.FULL, size=4.6)
        ax.plot([xs["full"] - 0.28, xs["full"] + 0.28], [f["mean"]] * 2, color=ps.FULL, linewidth=2.2, zorder=4)
        ax.axhline(CHANCE, color=ps.INK2, linestyle=":", linewidth=0.9)
        ax.set_xlim(-0.5, 3.9)
        ax.set_xticks([0, 1, 2, 3.4], [f"r=1\n{100 * params[1] / total:.2f}%", f"r=4\n{100 * params[4] / total:.2f}%",
                                       f"r=16\n{100 * params[16] / total:.1f}%", "full\n100%"], fontsize=6.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(20, 95)
        ax.set_title(SET_LABEL[s])
        if k == 0:
            ax.set_ylabel("Answer EM (%)")
        if k == 1:
            ax.set_xlabel("Trainable parameters (% of the 26.3M-parameter model)", fontsize=7)
    # bottom left: LR grid (seed 42)
    ax = fig.add_subplot(gs[1, 0])
    lrs = ["3e-4", "1e-3", "3e-3"]
    grid = {(g["rank"], g["lr"]): g for g in SUMMARY["lr_grid"]}
    em = np.array([[grid[(r, lr)]["val_em"] for lr in lrs] for r in (1, 4, 16)])
    ax.imshow(em, cmap=ps.SEQ, vmin=0, vmax=100, aspect="auto")
    for i, r in enumerate((1, 4, 16)):
        for j, lr in enumerate(lrs):
            g = grid[(r, lr)]
            ax.text(j, i, f"{g['val_em']:.1f}%\nloss {g['best_val_loss']:.3f}", ha="center", va="center", fontsize=5.8,
                    color=ps.text_on(ps.SEQ(g["val_em"] / 100)))
        j = lrs.index(SEL[f"lr_r{r}"])
        ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=ps.INK, linewidth=1.5))
    ax.set_xticks(range(3), lrs)
    ax.set_yticks(range(3), ["r=1", "r=4", "r=16"])
    ax.set_xlabel("Learning rate")
    ax.set_title("LR choice (seed 42)", fontsize=8)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # bottom middle: what the peak memory is made of
    ax = fig.add_subplot(gs[1, 1])
    order = ["full finetune (Phase 3)", "LoRA r=1", "LoRA r=4", "LoRA r=16"]
    rows = {r["config"]: r for r in mem}
    parts = [("weights", ps.BLUE), ("gradients", ps.ORANGE), ("AdamW state", ps.AQUA), ("activations, workspace", "#eda100")]
    for i, name in enumerate(order):
        r = rows[name]
        vals = [r["weights_mb"], r["grad_mb"], r["optimizer_mb"], r["peak_mb"] - r["weights_mb"] - r["grad_mb"] - r["optimizer_mb"]]
        left = 0.0
        for (_, col), v in zip(parts, vals):
            ax.barh(i, v, left=left, height=0.62, color=col, edgecolor=ps.SURFACE, linewidth=0.6)
            left += v
        ax.text(left + 40, i, f"{r['peak_mb']:,.0f}", va="center", fontsize=6.2, color=ps.INK2)
    ax.set_yticks(range(4), ["Full FT", "r=1", "r=4", "r=16"])
    ax.invert_yaxis()
    ax.set_xlim(0, 3500)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Peak GPU memory (MB)")
    ax.set_title("Training memory, batch 8", fontsize=8)
    ax.legend(handles=[Patch(facecolor=c, label=lab) for lab, c in parts], fontsize=5.8, loc="upper center",
              bbox_to_anchor=(0.5, -0.33), ncol=2, handlelength=1.0, columnspacing=0.8)
    # bottom right: throughput
    ax = fig.add_subplot(gs[1, 2])
    cols = [ps.FULL, ps.RANK[1], ps.RANK[4], ps.RANK[16]]
    for i, (name, col) in enumerate(zip(order, cols)):
        tps = rows[name]["tokens_per_second"]
        ax.barh(i, tps, height=0.62, color=col)
        ax.text(tps + 400, i, f"{tps / 1000:.1f}k", va="center", fontsize=6.2, color=ps.INK2)
    ax.set_yticks(range(4), ["Full FT", "r=1", "r=4", "r=16"])
    ax.invert_yaxis()
    ax.set_xlim(0, 35000)
    ax.set_xticks([0, 10000, 20000, 30000], ["0", "10k", "20k", "30k"])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Training tokens per second")
    ax.set_title("Training speed", fontsize=8)
    ps.save(fig, FIG / "fig_eval_a_rank.png")


def fig_a_dpo():
    ref = SUMMARY["best_lora_row"]["val"]["answer_em"]
    lrs, steps = ["1e-5", "3e-5", "1e-4"], [100, 200, 300, 400, 500]
    fig, axes = plt.subplots(2, 3, figsize=(ps.TEXT_WIDTH_IN, 3.9), sharex=True, sharey=True)
    for i, b in enumerate(("0.1", "0.5")):
        chosen = SEL[f"dpo_b{b}"]
        for j, lr in enumerate(lrs):
            ax = axes[i, j]
            h = jload(RA / f"ada_dpo_b{b}_lr{lr}_s42_history.json")
            pv = [(r["step"], 100 * r["preference_accuracy"]) for r in h["val_log"] if r["step"] > 0]
            ax.axhline(ref, color=ps.INK2, linestyle=":", linewidth=1)
            ax.plot(*zip(*pv), color=ps.MUTED, linewidth=1.2, marker="o", markersize=2.6, markeredgewidth=0)
            em = SUMMARY["dpo_sweep"][f"{b}|{lr}"]["val_em_by_step"]
            ax.plot([0] + steps, [ref] + em, color=ps.BETA[b], marker="o", markersize=4.4, markeredgecolor=ps.SURFACE,
                    markeredgewidth=0.8, linewidth=1.6)
            if chosen["lr"] == lr:
                ax.plot([chosen["step"]], [chosen["val_answer_em"]], marker="o", markersize=9, markerfacecolor="none",
                        markeredgecolor=ps.BETA[b], markeredgewidth=1.2, linestyle="none")
            ax.set_title(f"β={b}, LR {lr}", fontsize=8)
            ax.set_ylim(38, 100)
            if i == 1:
                ax.set_xlabel("DPO step")
            if j == 0:
                ax.set_ylabel("%")
    handles = [Line2D([], [], color=ps.MUTED, marker="o", markersize=3, label="held-out pairs ranked correctly (DPO's own metric)"),
               Line2D([], [], color=ps.BETA["0.1"], marker="o", markersize=4, label="validation answer EM, β=0.1"),
               Line2D([], [], color=ps.BETA["0.5"], marker="o", markersize=4, label="validation answer EM, β=0.5"),
               Line2D([], [], color=ps.INK2, linestyle=":", label="best LoRA (step 0)"),
               Line2D([], [], color=ps.INK2, marker="o", markersize=8, markerfacecolor="none", linestyle="none", label="chosen on validation")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=6.6)
    fig.tight_layout()
    ps.save(fig, FIG / "fig_eval_a_dpo.png")


def fig_a_drift():
    s = "unseen_entities"
    base = jload(RL / "eval_base.json")["pretrain_val_ppl_500docs"]
    pts = []  # (label, x, y, xsd, ysd, colour, hollow, marker)
    f = SUMMARY["full_ft_seeds"]
    pts.append(("Full FT (3 seeds)", f["ppl_500"]["mean"], f[s]["mean"], f["ppl_500"]["sd"], f[s]["sd"], ps.FULL, False, "o"))
    for r in (1, 4, 16):
        v = SUMMARY["lora"][str(r)]
        pts.append((f"LoRA r={r} (3 seeds)", v["ppl_500"]["mean"], v[s]["mean"], v["ppl_500"]["sd"], v[s]["sd"], ps.RANK[r], False, "o"))
    br = SUMMARY["best_lora_row"]
    pts.append(("best LoRA (DPO start)", br["ppl_500"], br[s]["answer_em"], 0, 0, ps.RANK[16], True, "D"))
    for b in ("0.1", "0.5"):
        c = SUMMARY["dpo_chosen"][b]
        pts.append((f"DPO β={b}, chosen (3 seeds)", c["ppl_500"]["mean"], c[s]["mean"], c["ppl_500"]["sd"], c[s]["sd"], ps.BETA[b], False, "s"))
    for b in ("0.1", "0.5"):
        c = SUMMARY["dpo_fixed_protocol"][b]
        pts.append((f"DPO β={b}, fixed protocol", c["ppl_500"], c[s]["answer_em"], 0, 0, ps.BETA[b], True, "s"))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.axvline(base, color=ps.INK2, linestyle=":", linewidth=1)
    ax.text(base + 0.08, 35, "pretrained base\n(PPL 57.95)", fontsize=6.3, color=ps.INK2, va="top")
    ax.axhline(CHANCE, color=ps.MUTED, linestyle=":", linewidth=0.9)
    ax.text(64.4, CHANCE + 1, "chance", fontsize=6.3, color=ps.MUTED, ha="right")
    handles = []
    for lab, x, y, xsd, ysd, col, hollow, mk in pts:
        if xsd or ysd:
            ax.errorbar(x, y, xerr=xsd, yerr=ysd, color=col, linewidth=0.9, capsize=1.5, zorder=2, fmt="none")
        ps.point(ax, x, y, col, size=6, marker=mk, hollow=hollow, zorder=3)
        handles.append(Line2D([], [], marker=mk, linestyle="none", markersize=5.5, markerfacecolor=ps.SURFACE if hollow else col,
                              markeredgecolor=col if hollow else ps.SURFACE, markeredgewidth=1.2 if hollow else 0.8, label=lab))
    fx = SUMMARY["dpo_fixed_protocol"]["0.1"]
    ax.annotate("", xy=(fx["ppl_500"], fx[s]["answer_em"] + 1.5), xytext=(br["ppl_500"], br[s]["answer_em"] - 1.8),
                arrowprops=dict(arrowstyle="->", color=ps.INK2, linewidth=0.9, shrinkA=2, shrinkB=2))
    d_acc = f"{fx[s]['answer_em'] - br[s]['answer_em']:+.0f}".replace("-", "−")
    ax.text(62.15, 62, f"over-optimised DPO:\n{d_acc} points of accuracy,\n{fx['ppl_500'] - br['ppl_500']:+.2f} PPL",
            fontsize=6.3, color=ps.INK2, va="center")
    ax.annotate("best LoRA, both chosen DPO\nmodels and fixed β=0.5 overlap\n(79.7–81.0%)", xy=(61.38, 82.2), xytext=(59.0, 86.5),
                fontsize=6.2, color=ps.INK2, va="center", arrowprops=dict(arrowstyle="-", color=ps.MUTED, linewidth=0.7))
    ax.set(xlim=(57.5, 64.5), ylim=(28, 93), xlabel="Pretraining-validation PPL (first 500 documents)", ylabel="Unseen-entity answer EM (%)")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5)
    ps.save(fig, FIG / "fig_eval_a_drift.png")


# ----------------------------------------------------------------------------------------------- Part B
def fig_b():
    pb = jload(RL / "part_b.json")
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXT_WIDTH_IN, 5.0), gridspec_kw={"hspace": 0.5, "wspace": 0.28})
    # B1: LSH S-curve
    ax = axes[0, 0]
    s = np.linspace(0, 1, 501)
    b1 = pb["B1"]
    p10 = 1 - (1 - s ** b1["r"]) ** b1["b"]
    p20 = 1 - (1 - s ** 5) ** 20
    ax.axvspan(0.6, 0.8, color="#f0efec", zorder=0)
    ax.text(0.665, 0.52, "typical\nreposts", ha="center", fontsize=6.0, color=ps.INK2)
    ax.plot(s, p20, color=ps.MUTED, linewidth=1.3)
    ax.plot(s, p10, color=ps.BLUE, linewidth=1.9)
    t = b1["threshold_(1/b)^(1/r)"]
    ax.axvline(t, color=ps.INK2, linestyle=":", linewidth=0.9)
    ax.text(t + 0.015, 0.1, f"t ≈ (1/b)$^{{1/r}}$\n= {t:.3f}", fontsize=6.2, color=ps.INK2)
    for sv in (0.3, 0.8):
        pv = b1[f"s{sv}"]["P"]
        ps.point(ax, sv, pv, ps.BLUE, size=5.5)
    ax.annotate(f"P(0.8) = {b1['s0.8']['P']:.3f}", (0.8, b1["s0.8"]["P"]), xytext=(0.37, 0.72), fontsize=6.4, color=ps.INK,
                arrowprops=dict(arrowstyle="-", color=ps.MUTED, linewidth=0.7))
    ax.annotate(f"P(0.3) = {b1['s0.3']['P']:.1e}", (0.3, b1["s0.3"]["P"]), xytext=(0.03, 0.2), fontsize=6.4, color=ps.INK,
                arrowprops=dict(arrowstyle="-", color=ps.MUTED, linewidth=0.7))
    ax.text(0.47, 0.93, "b=20, r=5 (the fix)", fontsize=6.4, color=ps.INK2, ha="right")
    ax.text(0.865, 0.6, "b=10,\nr=10\n(variant)", fontsize=6.2, color=ps.INK, ha="left")
    ax.set(xlim=(0, 1), ylim=(-0.02, 1.02), xlabel="Jaccard similarity s", ylabel="P(pair becomes a candidate)", title="B1. LSH with 100 MinHash values")
    # B2: sampling share vs alpha
    ax = axes[0, 1]
    b2 = pb["B2"]
    ph, pl = b2["by_size"]["hindi"], b2["by_size"]["nepali"]
    a = np.linspace(0, 1, 201)
    qh = ph ** a / (ph ** a + pl ** a)
    ax.plot(a, qh, color=ps.BLUE, linewidth=1.9)
    ax.plot(a, 1 - qh, color=ps.ORANGE, linewidth=1.9)
    for av, hollow in ((0.5, False), (1.0, True)):
        h = ph ** av / (ph ** av + pl ** av)
        ps.point(ax, av, h, ps.BLUE, size=5.5, hollow=hollow)
        ps.point(ax, av, 1 - h, ps.ORANGE, size=5.5, hollow=hollow)
        ax.text(av - 0.02, h + 0.012, f"{h:.3f}", fontsize=6.3, ha="right" if av == 1.0 else "center", va="bottom")
        ax.text(av - 0.02, 1 - h - 0.014, f"{1 - h:.3f}", fontsize=6.3, ha="right" if av == 1.0 else "center", va="top")
    ax.text(0.03, 0.535, "Hindi", fontsize=6.6, color=ps.INK)
    ax.text(0.03, 0.452, "Nepali", fontsize=6.6, color=ps.INK, va="top")
    d = b2.get("tokens_drawn_alpha", {})
    if d:
        ax.text(0.03, 0.62, f"500M tokens at α=0.5: Nepali gets {d['nepali'] / 1e6:.1f}M\n"
                            f"= {d['nepali'] / b2['tokens']['nepali']:.2f} epochs of its {b2['tokens']['nepali'] / 1e6:.0f}M-token corpus",
                fontsize=6.2, color=ps.INK2)
    ax.set(xlim=(0, 1.04), ylim=(0.38, 0.66), xlabel="α (0 = uniform, 1 = by size; hollow = by size)", ylabel="Sampling share",
           title="B2. Sampling the two corpora")
    # B3: loss at fixed compute vs N
    ax = axes[1, 0]
    b3 = pb["B3"]
    C = b3["C"]
    E, A, al, B, be = 1.82, 482.01, 0.3478, 2085.43, 0.3658   # Besiroglu et al. (2024) refit, as in part_b_calculations.py
    n = np.logspace(9, 11, 300)
    dd = C / (6 * n)
    pt, dt = A / n ** al, B / dd ** be
    ax.plot(n, pt, color=ps.ORANGE, linewidth=1.4)
    ax.plot(n, dt, color=ps.AQUA, linewidth=1.4)
    ax.plot(n, pt + dt, color=ps.BLUE, linewidth=2.0)
    for nv, lab in ((b3["N"], "N = 5B, D = 100B\n(20 tokens/param)"), (b3["colleague"]["N"], "4× larger: N = 20B,\nD = 25B (1.25 tokens/param)")):
        tot = A / nv ** al + B / (C / (6 * nv)) ** be
        ax.axvline(nv, color=ps.INK2, linestyle=":", linewidth=0.8)
        ps.point(ax, nv, tot, ps.BLUE, size=5.5)
        ax.text(nv * 1.08, 0.98 if nv < 1e10 else 0.83, lab, fontsize=6.1, color=ps.INK2, va="top")
    ax.text(1.85e10, 0.478, f"+{b3['loss_colleague']['L'] - b3['loss_compute_optimal']['L']:.3f} nats", fontsize=6.3, color=ps.INK, ha="right")
    ax.text(1.12e9, 0.505, "total", fontsize=6.5, color=ps.INK)
    ax.text(1.12e9, 0.27, "parameter term A/N$^α$", fontsize=6.3, color=ps.INK2)
    ax.text(1.12e9, 0.03, "data term B/D$^β$", fontsize=6.3, color=ps.INK2)
    ax.set_xscale("log")
    ax.set(xlim=(1e9, 1e11), ylim=(0, 1.0), xlabel="Model size N (parameters), D = C/6N", ylabel="Loss above L$_∞$ = 1.82 (nats)",
           title="B3. Fixed compute C = 3×10$^{21}$ FLOPs")
    # B4: memory for the 8B model
    ax = axes[1, 1]
    b4 = pb["B4"]
    gb = 8e9 / 1e9
    bars = [("Run (fp32)", [4 * gb, 0, 0]), ("Full FT, fp32\n+ Adam", [4 * gb, 4 * gb, 8 * gb]),
            ("Full FT, mixed\nprecision + Adam", [(2 + 4) * gb, 2 * gb, 8 * gb])]
    parts = [("weights (+ fp32 master)", ps.BLUE), ("gradients", ps.ORANGE), ("Adam m, v", ps.AQUA)]
    for i, (lab, vals) in enumerate(bars):
        left = 0.0
        for (_, col), v in zip(parts, vals):
            if v:
                ax.barh(i, v, left=left, height=0.6, color=col, edgecolor=ps.SURFACE, linewidth=0.7)
            left += v
        ax.text(left + 2, i, f"{left:.0f} GB", va="center", fontsize=6.6, color=ps.INK)
    assert abs(b4["inference_fp32_GB"] - 4 * gb) < 1e-6 and abs(b4["full_ft_adam_GB"] - 16 * gb) < 1e-6
    ax.set_yticks(range(3), [lab for lab, _ in bars], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="GB (10$^9$ bytes), before activations", title="B4. Memory for an 8B model")
    ax.legend(handles=[Patch(facecolor=c, label=lab) for lab, c in parts], fontsize=5.8, loc="upper right", handlelength=1.0,
              bbox_to_anchor=(1.02, 1.02))
    ps.save(fig, FIG / "fig_eval_b.png")


# ----------------------------------------------------------------------------------------------- Part C
def c1_rows():
    return jsonl(DATA / "c1_outputs.jsonl")


def outcome(x):
    """correct / template / wrong / none for one C1 reply or vote (x has extracted|majority, correct, gave_template_answer)."""
    if x.get("correct"):
        return "correct"
    if x.get("gave_template_answer"):
        return "template"
    ans = x.get("extracted", x.get("majority"))
    return "none" if ans in (None, "") else "wrong"


OUT_STYLE = {"correct": (ps.CORRECT_CELL, "✓"), "template": (ps.WRONG, "T"), "wrong": ("#ec835a", "✗"), "none": (ps.NONE, "–")}
OUT_LABEL = {"correct": "correct", "template": "familiar template answer", "wrong": "other wrong answer", "none": "no number"}


def fig_c1_tokens():
    res = jload(RL / "c1_results.json")["results"]
    rows = c1_rows()
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 2.6), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.42})
    ax = axes[0]
    seeds = res["sc_seeds"]["per_seed"]
    xs = [res["direct"]["mean_generated_tokens"], res["cot"]["mean_generated_tokens"], res["sc_seeds"]["mean_generated_tokens_mean"]]
    ys_std = [res["direct"]["standard_accuracy"], res["cot"]["standard_accuracy"], res["sc_seeds"]["standard_accuracy_mean"]]
    ys_mod = [res["direct"]["modified_accuracy"], res["cot"]["modified_accuracy"], res["sc_seeds"]["modified_accuracy_mean"]]
    ax.plot(xs, ys_std, color=ps.BLUE, linewidth=1.2, zorder=1)
    ax.plot(xs, ys_mod, color=ps.ORANGE, linewidth=1.2, zorder=1)
    for x, y in zip(xs[:2], ys_std[:2]):
        ps.point(ax, x, y, ps.BLUE, size=6)
    for x, y in zip(xs, ys_mod):
        ps.point(ax, x, y, ps.ORANGE, size=6, marker="s")
    ax.errorbar(xs[2], ys_std[2], yerr=res["sc_seeds"]["standard_accuracy_sd"], color=ps.BLUE, capsize=2.5, linewidth=0.9, fmt="none")
    ps.point(ax, xs[2], ys_std[2], ps.BLUE, size=6)
    for v in seeds.values():
        ps.point(ax, v["mean_generated_tokens"] * 1.35, v["standard_accuracy"], ps.BLUE, size=3.6, hollow=True)
    for x, y, lab, dy in zip(xs, ys_std, ["direct", "zero-shot CoT", "self-consistency, 5 samples\n(mean ± sd; hollow: each seed)"], [6, 6, 9]):
        ax.text(x, y + dy, lab, ha="center", fontsize=6.4, color=ps.INK)
    ax.set_xscale("log")
    ax.set(xlim=(2.5, 5000), ylim=(-4, 100), xlabel="Mean generated tokens per problem (log scale)", ylabel="Accuracy (%)",
           title="C1. Accuracy vs. tokens spent")
    ax.legend(handles=[Line2D([], [], color=ps.BLUE, marker="o", markersize=4, label="15 standard problems"),
                       Line2D([], [], color=ps.ORANGE, marker="s", markersize=4, label="5 modified problems")],
              loc="upper left", fontsize=6.5)
    # what the model answers on the modified problems
    ax = axes[1]
    mods = [r for r in rows if r["type"] == "modified"]
    groups = [("direct (5)", [r["direct"] for r in mods]),
              ("CoT (5)", [r["cot"] for r in mods]),
              ("SC vote (3 seeds, 15)", [r["sc"] for r in mods] + [r["sc_other_seeds"][k] for r in mods for k in r["sc_other_seeds"]]),
              ("SC single chains (75)", [smp for r in mods for blk in [r["sc"], *r["sc_other_seeds"].values()] for smp in blk["samples"]])]
    keys = ["correct", "template", "wrong", "none"]
    for i, (lab, items) in enumerate(groups):
        outs = [outcome(x) for x in items]
        left = 0.0
        for k in keys:
            v = 100 * outs.count(k) / len(outs)
            if v:
                ax.barh(i, v, left=left, height=0.62, color=OUT_STYLE[k][0], edgecolor=ps.SURFACE, linewidth=0.7)
                if v >= 9:
                    ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.0, color=ps.text_on(OUT_STYLE[k][0]))
            left += v
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="Share of answers (%)", title="Modified problems: what is answered")
    ax.legend(handles=[Patch(facecolor=OUT_STYLE[k][0], label=OUT_LABEL[k]) for k in keys], fontsize=6.0, ncol=2,
              loc="upper center", bbox_to_anchor=(0.4, -0.24), handlelength=1.0, columnspacing=0.8)
    ps.save(fig, FIG / "fig_eval_c1_tokens.png")


def fig_c1_grid():
    rows = c1_rows()
    probs = {p["id"]: p for p in jsonl(DATA / "c1_problems.jsonl")}
    seeds = ["1234"] + sorted(rows[0]["sc_other_seeds"])
    cols = [("Direct", lambda r: r["direct"]), ("CoT", lambda r: r["cot"])]
    for sd in seeds:
        blk = (lambda r, sd=sd: r["sc"]) if sd == "1234" else (lambda r, sd=sd: r["sc_other_seeds"][sd])
        cols += [(f"{k + 1}", (lambda r, blk=blk, k=k: blk(r)["samples"][k])) for k in range(5)]
        cols.append(("vote", blk))
    xpos, width, x = [], [], 0.0
    for j, (name, _) in enumerate(cols):
        if j in (2, 8, 14):
            x += 0.45
        w = 1.5 if j < 2 else 1.0
        xpos.append(x)
        width.append(w)
        x += w
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH_IN, 4.6))
    for i, r in enumerate(rows):
        yv = i + (0.4 if r["type"] == "modified" else 0)
        for j, (name, get) in enumerate(cols):
            k = outcome(get(r))
            col, sym = OUT_STYLE[k]
            ax.add_patch(Rectangle((xpos[j] + 0.04, yv + 0.06), width[j] - 0.08, 0.88, facecolor=col, edgecolor="none"))
            ax.text(xpos[j] + width[j] / 2, yv + 0.5, sym, ha="center", va="center", fontsize=6.6, color=ps.text_on(col),
                    fontweight="bold" if name == "vote" else "normal")
        topic = probs[r["id"]]["topic"]
        ax.text(-0.3, yv + 0.5, f"{r['id']}  {topic}", ha="right", va="center", fontsize=6.4, color=ps.INK)
        ax.text(xpos[-1] + 1.3, yv + 0.5, f"gold {r['gold']}", ha="left", va="center", fontsize=6.0, color=ps.INK2)
    for j, (name, _) in enumerate(cols):
        ax.text(xpos[j] + width[j] / 2, -0.35, name, ha="center", va="bottom", fontsize=6.2, color=ps.INK2)
    for g, sd in enumerate(seeds):
        x0 = xpos[2 + 6 * g]
        ax.text(x0 + 3.0, -1.35, f"SC, seed {sd}", ha="center", va="bottom", fontsize=6.5, color=ps.INK)
        ax.plot([x0 + 0.05, x0 + 5.95], [-1.05, -1.05], color=ps.AXIS, linewidth=0.8)
    ax.text(xpos[0] + 1.5, -1.35, "greedy", ha="center", va="bottom", fontsize=6.5, color=ps.INK)
    ax.plot([xpos[0] + 0.05, xpos[1] + 1.45], [-1.05, -1.05], color=ps.AXIS, linewidth=0.8)
    n_std = sum(r["type"] == "standard" for r in rows)
    ax.plot([-6.5, xpos[-1] + 1.0], [n_std + 0.2, n_std + 0.2], color=ps.INK2, linewidth=0.6, clip_on=False)
    ax.text(xpos[-1] + 1.3, n_std + 0.2, "modified ↓", fontsize=6.0, color=ps.INK2, va="center")
    ax.set_xlim(-6.6, xpos[-1] + 3.2)
    ax.set_ylim(len(rows) + 0.5, -1.6)
    ax.axis("off")
    handles = [Patch(facecolor=OUT_STYLE[k][0], label=f"{OUT_STYLE[k][1]}  {OUT_LABEL[k]}") for k in ("correct", "template", "wrong", "none")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.6, handlelength=1.2)
    ps.save(fig, FIG / "fig_eval_c1_grid.png")


def fig_c2_judge():
    runs = [("4-bit NF4 judge (laptop)", jsonl(DATA / "c2_pairs_with_verdicts.jsonl")),
            ("fp32 judge (Ada)", jsonl(MIDSEM / "data_ada" / "c2_pairs_with_verdicts_fp32.jsonl"))]
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 3.6), gridspec_kw={"wspace": 0.3})
    style = {"cot": (ps.BLUE, "o", "long CoT answer is right"), "direct": (ps.ORANGE, "^", "short direct answer is right")}
    quad = {(0, 1): "picks the answer\nshown first", (1, 0): "picks the answer\nshown second",
            (1, 1): "picks the CoT\nin both orders", (0, 0): "picks the direct\nin both orders"}
    # pair counts per quadrant: ● = pairs where the CoT answer is right, ▲ = pairs where the direct answer is right
    for ax, (title, pairs) in zip(axes, runs):
        ax.axhline(0.5, color=ps.AXIS, linewidth=0.8)
        ax.axvline(0.5, color=ps.AXIS, linewidth=0.8)
        counts = {q: {"cot": 0, "direct": 0} for q in quad}
        for p in pairs:
            x = p["verdict_direct_first"]["p_B"]       # CoT is answer B when the direct answer is shown first
            y = p["verdict_cot_first"]["p_A"]          # CoT is answer A when it is shown first
            col, mk, _ = style[p["checker_winner"]]
            ps.point(ax, x, y, col, size=5.6, marker=mk)
            qx = int(p["verdict_direct_first"]["prefers"] == "cot")
            qy = int(p["verdict_cot_first"]["prefers"] == "cot")
            counts[(qx, qy)][p["checker_winner"]] += 1
        for (qx, qy), lab in quad.items():
            c = counts[(qx, qy)]
            ax.text(0.02 if qx == 0 else 0.98, 1.285 if qy == 1 else 0.02,
                    f"{lab}\n{c['cot'] + c['direct']} pair{'' if c['cot'] + c['direct'] == 1 else 's'}: {c['cot']} ●  {c['direct']} ▲",
                    ha="left" if qx == 0 else "right", va="top" if qy == 1 else "bottom", fontsize=5.8, color=ps.INK2)
        ax.axhline(1.0, color=ps.AXIS, linewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1.3), title=title,
               xlabel="P(picks CoT) when the direct answer is shown first", ylabel="P(picks CoT) when the CoT is shown first")
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
    handles = [Line2D([], [], marker=mk, color=col, linestyle="none", markersize=5, label=lab) for col, mk, lab in style.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=7)
    ps.save(fig, FIG / "fig_eval_c2_judge.png")


# ----------------------------------------------------------------------------------------------- examples
def tex(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def dn(s):
    return r"\dn{" + tex(s) + "}"


def flat(s):
    return " ".join(s.replace("\n", " ").split())


def head(s, n):
    """The first n characters of s, cut back to the last space so that no Devanagari word is split."""
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0]


def clip(s, n):
    s = flat(s)
    return dn(s) if len(s) <= n else dn(head(s, n)) + r"\,\ldots"


def mark(ok):
    return r"\checkmark" if ok else r"$\times$"


ANS = "\nउत्तर:"


def qualitative_a():
    """Example generations, chosen from the six questions per test set whose outputs every evaluation saves."""
    models = [("Full FT (Phase 3)", RA / "eval_full_ft.json"),     # display names are LaTeX; seeds are in the caption
              (r"Best LoRA ($r{=}16$)", RA / f"eval_{SEL['best_lora']}.json"),
              (r"DPO $\beta{=}0.1$, chosen", RA / f"eval_{SUMMARY['dpo_chosen']['0.1']['models'][0]}.json"),
              (r"DPO $\beta{=}0.5$, chosen", RA / f"eval_{SUMMARY['dpo_chosen']['0.5']['models'][0]}.json"),
              (r"DPO $\beta{=}0.1$, fixed", RA / f"eval_{SUMMARY['dpo_fixed_protocol']['0.1']['name']}.json")]
    picks = [("unseen_entities", 1, "four names, a chain of three relations"),
             ("unseen_entities", 2, "two numbers"),
             ("unseen_wording", 0, "three numbers, held-out phrasing")]
    ex = {name: jload(path)["sets"] for name, path in models}
    lines = [r"\begin{figure}[H]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}p{3.3cm}p{13.25cm}c@{}}", r"\toprule"]
    for k, (s, idx, what) in enumerate(picks):
        q = ex[models[0][0]][s]["examples"][idx]
        question = q["question"].split(":", 1)[1].strip() if q["question"].startswith("प्रश्न:") else q["question"]
        lines.append(r"\multicolumn{3}{@{}p{17.1cm}@{}}{\textbf{(" + "abc"[k] + ") " + SET_LABEL[s] + f" ({what}).}} " + dn(question)
                     + r" \hfill gold: " + dn(q["gold"]) + r"}\\")
        lines.append(r"\midrule")
        for name, _ in models:
            e = ex[name][s]["examples"][idx]
            gen = e["generation"]
            if ANS in gen:
                chain, ans = gen.rsplit(ANS, 1)
                body = clip(chain, 175) + r" \quad$\rightarrow$\," + dn("उत्तर: " + ans.strip())
            else:
                body = clip(gen, 175) + r" \quad \textit{(no answer line before the 128-token limit)}"
            lines.append(name + " & " + body + " & " + mark(e["correct"]) + r"\\")
        lines.append(r"\midrule" if k < len(picks) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}"]
    return lines


def qualitative_c():
    rows = {r["id"]: r for r in c1_rows()}
    probs = {p["id"]: p for p in jsonl(DATA / "c1_problems.jsonl")}
    m = rows["M01"]
    q4 = {p["pair_id"]: p for p in jsonl(DATA / "c2_pairs_with_verdicts.jsonl")}
    q32 = {p["pair_id"]: p for p in jsonl(MIDSEM / "data_ada" / "c2_pairs_with_verdicts_fp32.jsonl")}
    # The example pair: the short direct reply is right, the 4-bit judge follows the order (picks whichever reply it
    # reads first) and the fp32 judge picks the long reply in both orders. Among such pairs, the longest CoT reply.
    def follows_order(p):
        return p["verdict_direct_first"]["prefers"] == "direct" and p["verdict_cot_first"]["prefers"] == "cot"

    def picks_cot(p):
        return p["verdict_direct_first"]["prefers"] == "cot" and p["verdict_cot_first"]["prefers"] == "cot"

    cands = [k for k in q4 if q4[k]["checker_winner"] == "direct" and follows_order(q4[k]) and picks_cot(q32[k])]
    pid = max(cands, key=lambda k: q4[k]["cot_chars"]) if cands else next(iter(q4))
    p4, p32 = q4[pid], q32[pid]
    assert p4["cot_reply"] == p32["cot_reply"] and p4["direct_reply"] == p32["direct_reply"]

    def verdicts(p):
        d, c = p["verdict_direct_first"], p["verdict_cot_first"]
        pick = {"direct": "direct", "cot": "CoT"}
        return (f"direct shown first (as A): P(A) = {d['p_A']:.2f}, picks the {pick[d['prefers']]} answer; "
                f"CoT shown first (as A): P(A) = {c['p_A']:.2f}, picks the {pick[c['prefers']]} answer")

    cot = flat(m["cot"]["reply"])
    template = probs["M01"]["template_answer"]

    def what(x):
        """How a reply relates to the gold and the template answer, from the checker's fields."""
        if x["correct"]:
            return "right"
        if x["gave_template_answer"]:
            return "the template answer"
        return "neither the answer nor the template" if x.get("extracted") else "no number"

    cot_note = (f"loops until the 1,024-token limit; last number {m['cot']['extracted']}" if m["cot"]["truncated"]
                else f"answer {m['cot']['extracted']}: {what(m['cot'])}")
    votes = ", ".join(v if v else "--" for v in m["sc"]["votes"])
    sample = flat(p4["cot_reply"])
    lines = [r"\begin{figure}[H]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}p{3.3cm}p{13.25cm}c@{}}", r"\toprule",
             r"\multicolumn{3}{@{}p{17.1cm}@{}}{\textbf{(a) C1, problem M01 (modified: the boat drifts).} " + dn(m["problem"])
             + r" \hfill gold: " + tex(m["gold"]) + ", familiar template: " + tex(template) + r"}\\",
             r"\midrule",
             r"Direct (greedy) & " + dn(flat(m["direct"]["reply"])) + r" \quad (" + what(m["direct"]) + r") & " + mark(m["direct"]["correct"]) + r"\\",
             r"Zero-shot CoT (greedy) & " + dn(head(cot, 120)) + r"\,\ldots\," + dn(cot[-60:].split(" ", 1)[-1]) + r" \quad (" + tex(cot_note)
             + r") & " + mark(m["cot"]["correct"]) + r"\\",
             r"SC (seed 1234) & votes of the 5 chains: " + tex(votes) + r"; plurality vote " + tex(str(m["sc"]["majority"]))
             + r" & " + mark(m["sc"]["correct"]) + r"\\",
             r"\midrule",
             r"\multicolumn{3}{@{}p{17.1cm}@{}}{\textbf{(b) C2, pair " + pid + " (problem " + p4["problem_id"] + r"; the short answer is right).} "
             + dn(p4["problem"]) + r" \hfill gold: " + tex(p4["gold"]) + r"}\\",
             r"\midrule",
             r"Answer 1: direct & " + dn(flat(p4["direct_reply"])) + r" & " + mark(p4["direct_correct"]) + r"\\",
             r"Answer 2: CoT & (" + tex(p4["cot_source"].replace("sc_seed", "SC seed ").replace("_sample_", ", sample ")) + r") "
             + dn(head(sample, 120)) + r"\,\ldots\," + dn(head(sample[sample.find("उत्तर:"):], 16))
             + r"\,\ldots\ (no number is ever computed; the checker reads the list number " + tex(p4["cot_extracted"]) + r") & "
             + mark(p4["cot_correct"]) + r"\\",
             r"Judge, 4-bit NF4 & " + tex(verdicts(p4)) + r" & --\\",
             r"Judge, fp32 & " + tex(verdicts(p32)) + r" & --\\",
             r"\bottomrule", r"\end{tabular}"]
    return lines


def write_qualitative():
    a = qualitative_a()
    a += [r"\caption{A4: example generations on the test sets (3 of the 6 questions per set whose outputs every evaluation saves, "
          r"picked to show the error types; best LoRA: $r=16$, seed " + SEL["best_lora"].rsplit("_s", 1)[1]
          + r"; chosen DPO: seed 42; fixed: LR $10^{-4}$, 500 steps). (a) Full FT writes names from the training set "
          r"(\dn{चंचल}, \dn{श्रुति}: never in the test set) that are not in the question; the chosen $\beta=0.5$ "
          r"model and the over-optimised $\beta=0.1$ model loop without ever writing the answer line. (b) Every LoRA and DPO model "
          r"writes the right comparison in the chain (55 \textgreater{} 27, so the first name) and then names the other person. "
          r"(c) Under held-out wording the chains bind numbers to the wrong names.}", r"\label{fig:qualA}", r"\end{figure}"]
    c = qualitative_c()
    c += [r"\caption{C1/C2: two examples in full. (a) The drifting boat defeats every condition, and no reply treats `drifting' as "
          r"moving at the current's 3\,km/h: direct gives the engine's own speed (12), the greedy chain finds the upstream speed 9 and "
          r"then repeats `\dn{दर: 6 घंटे}' (a speed in hours) until it answers 6, and the five sampled chains disagree, so the "
          r"plurality vote (6, from two chains) is wrong too. (b) A pair the judge gets wrong: the long reply never computes a number, yet the 4-bit judge "
          r"picks whichever answer it reads first and the fp32 judge picks the long reply in both orders.}", r"\label{fig:qualC}", r"\end{figure}"]
    (REPORT / "qualitative_a.tex").write_text("\n".join(a) + "\n", encoding="utf-8")
    (REPORT / "qualitative_c.tex").write_text("\n".join(c) + "\n", encoding="utf-8")


def main():
    FIG.mkdir(exist_ok=True)
    for fn in (fig_a_tasks, fig_a_outcomes, fig_a_rank, fig_a_dpo, fig_a_drift, fig_b, fig_c1_tokens, fig_c1_grid, fig_c2_judge):
        fn()
        print("wrote", fn.__name__)
    write_qualitative()
    print("wrote report/qualitative_a.tex, report/qualitative_c.tex")


if __name__ == "__main__":
    main()
