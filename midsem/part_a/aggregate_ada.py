#!/usr/bin/env python3
"""Aggregates the Ada study (midsem/results_ada/, synced from the job's to_transfer/results) into
midsem/results_ada/summary.json and summary.md, and draws its figures into midsem/figures/.

  - LR grid (seed 42): rank x LR -> best validation loss, validation / unseen-entity answer EM
  - main table: each rank at its chosen LR, mean +- sd over 3 seeds on validation and the three test sets
  - DPO sweep: validation answer EM for beta x LR x step, the chosen configuration per beta
  - DPO at the chosen configuration (3 seeds) and the original fixed protocol (LR 1e-4, 500 steps, seed 42)
  - drift: 500-document PPL and full-validation-split PPL (mean +- sd where there are seeds)
A single run's accuracy also gets a Wilson 95% interval (n = 1,000 questions per set).
"""

import json
import math
import statistics as st
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from midsem import plot_style as ps  # noqa: E402  (one palette and look for every report figure)

ps.apply()
MIDSEM = Path(__file__).resolve().parents[1]
RA = MIDSEM / "results_ada"
FIG = MIDSEM / "figures"
RANKS, SFT_LRS, SEEDS = [1, 4, 16], ["3e-4", "1e-3", "3e-3"], [42, 43, 44]
BETAS, DPO_LRS, DPO_STEPS = ["0.1", "0.5"], ["1e-5", "3e-5", "1e-4"], [100, 200, 300, 400, 500]
SETS = ["seen_entities", "unseen_entities", "unseen_wording"]


def load(name):
    p = RA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def wilson(p_pct, n=1000, z=1.96):
    """95% Wilson interval for an accuracy given in percent."""
    p = p_pct / 100
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def ms(values):
    """mean, sample sd (None for a single value)."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(st.mean(values), 4), "sd": round(st.stdev(values), 4) if len(values) > 1 else None, "n": len(values),
            "values": values}


def sft_tag(r, lr, s):
    return f"ada_r{r}_lr{lr}_s{s}"


def dpo_tag(b, lr, s, upto=None):
    return f"ada_dpo_b{b}_lr{lr}_s{s}" + (f"_upto{upto}" if upto else "")


def eval_row(name):
    ev = load(f"eval_{name}.json")
    if not ev or "sets" not in ev:
        return None
    row = {"name": name, "ppl_500": ev.get("pretrain_val_ppl_500docs")}
    for s, v in ev["sets"].items():
        row[s] = {"answer_em": v["answer_em"], "chain_em": v["chain_em"], "answered": v["answered"]}
    full = load(f"ppl_full_{name}.json")
    row["ppl_full"] = full["perplexity"] if full else None
    return row


def main():
    sel = {k: (load(f"ada_frozen_{k}.json") or {}).get("value") for k in ["lr_r1", "lr_r4", "lr_r16", "best_lora", "dpo_b0.1", "dpo_b0.5"]}
    out = {"selections": sel}

    # LR grid
    grid = []
    for r in RANKS:
        for lr in SFT_LRS:
            h, e = load(f"{sft_tag(r, lr, 42)}_history.json"), eval_row(sft_tag(r, lr, 42))
            grid.append({"rank": r, "lr": lr, "best_val_loss": h["best_val_reasoning_loss"] if h else None,
                         "best_epoch": h["best_epoch"] if h else None,
                         "val_em": e["val"]["answer_em"] if e and "val" in e else None,
                         "unseen_entities_em": e["unseen_entities"]["answer_em"] if e and "unseen_entities" in e else None})
    out["lr_grid"] = grid

    # main table: chosen LR per rank, 3 seeds
    main_rows = {}
    for r in RANKS:
        lr = sel.get(f"lr_r{r}")
        if not lr:
            continue
        rows = [eval_row(sft_tag(r, lr, s)) for s in SEEDS]
        hists = [load(f"{sft_tag(r, lr, s)}_history.json") for s in SEEDS]
        agg = {"lr": lr, "seeds": [s for s, x in zip(SEEDS, rows) if x]}
        for s in ["val"] + SETS:
            agg[s] = ms([x[s]["answer_em"] for x in rows if x and s in x])
            agg[s + "_chain"] = ms([x[s]["chain_em"] for x in rows if x and s in x])
        agg["ppl_500"] = ms([x["ppl_500"] for x in rows if x])
        agg["ppl_full"] = ms([x["ppl_full"] for x in rows if x])
        agg["best_val_loss"] = ms([h["best_val_reasoning_loss"] for h in hists if h])
        main_rows[str(r)] = agg
    out["lora"] = main_rows
    full = eval_row("full_ft")
    if full:
        full["wilson"] = {s: wilson(full[s]["answer_em"]) for s in SETS if s in full}
        full["ppl_full"] = (load("ppl_full_full_ft.json") or {}).get("perplexity")
    out["full_ft"] = full
    out["base"] = {"ppl_full": (load("ppl_full_base.json") or {}).get("perplexity")}

    # full finetune over 3 seeds: the Phase 3 checkpoint (seed 42, scored on Ada) + seeds 43, 44 (Phase 3 recipe,
    # trained and scored on the laptop: midsem/results/eval_fullft_s4{3,4}.json)
    lap = MIDSEM / "results"
    fs = [full] + [(lambda d: None if d is None else {**{s: {"answer_em": v["answer_em"], "chain_em": v["chain_em"]} for s, v in d["sets"].items()},
                                                      "ppl_500": d["pretrain_val_ppl_500docs"]})(
        json.loads((lap / f"eval_fullft_s{s}.json").read_text()) if (lap / f"eval_fullft_s{s}.json").exists() else None) for s in (43, 44)]
    fs = [x for x in fs if x]
    if len(fs) > 1:
        fagg = {"n_seeds": len(fs)}
        for s in ["val"] + SETS:
            fagg[s] = ms([x[s]["answer_em"] for x in fs if s in x])
            fagg[s + "_chain"] = ms([x[s]["chain_em"] for x in fs if s in x])
        fagg["ppl_500"] = ms([x["ppl_500"] for x in fs])
        out["full_ft_seeds"] = fagg

    # DPO sweep (seed 42)
    sweep = {}
    for b in BETAS:
        for lr in DPO_LRS:
            vals = []
            for step in DPO_STEPS:
                ev = load(f"eval_{dpo_tag(b, lr, 42)}_step{step}.json")
                vals.append(ev["sets"]["val"]["answer_em"] if ev and "val" in ev.get("sets", {}) else None)
            h = load(f"{dpo_tag(b, lr, 42)}_history.json")
            sweep[f"{b}|{lr}"] = {"val_em_by_step": vals,
                                  "final_val_margin": h["val_log"][-1]["margin"] if h else None,
                                  "final_val_pref_acc": h["val_log"][-1]["preference_accuracy"] if h else None,
                                  "step0_loss": h["step0"]["first_batch_loss"] if h else None}
    out["dpo_sweep"] = sweep
    best = sel.get("best_lora")
    best_row = eval_row(best) if best else None
    out["best_lora_row"] = best_row

    # DPO chosen configuration, 3 seeds; fixed protocol
    chosen = {}
    for b in BETAS:
        c = sel.get(f"dpo_b{b}")
        if not c:
            continue
        names = [f"{dpo_tag(b, c['lr'], 42)}_step{c['step']}"] + [dpo_tag(b, c["lr"], s, upto=c["step"]) for s in SEEDS[1:]]
        rows = [eval_row(n) for n in names]
        agg = {"lr": c["lr"], "step": c["step"], "val_em_seed42": c["val_answer_em"], "models": names}
        for s in ["val"] + SETS:
            agg[s] = ms([x[s]["answer_em"] for x in rows if x and s in x])
            agg[s + "_answered"] = ms([x[s]["answered"] for x in rows if x and s in x])
        agg["ppl_500"] = ms([x["ppl_500"] for x in rows if x])
        agg["ppl_full"] = ms([x["ppl_full"] for x in rows if x])
        hs = [load(f"{dpo_tag(b, c['lr'], 42)}_history.json")] + [load(f"{dpo_tag(b, c['lr'], s, upto=c['step'])}_history.json") for s in SEEDS[1:]]
        chosen[b] = agg
        chosen[b]["step0_losses"] = [h["step0"]["first_batch_loss"] for h in hs if h]
    out["dpo_chosen"] = chosen
    fixed = {}
    for b in BETAS:
        row = eval_row(f"{dpo_tag(b, '1e-4', 42)}_step500")
        if row:
            fixed[b] = row
    out["dpo_fixed_protocol"] = fixed

    # step-0 loss of every DPO run (must be ln 2 regardless of beta, LR and seed)
    step0 = []
    for p in sorted(RA.glob("ada_dpo_*_history.json")):
        h = json.loads(p.read_text(encoding="utf-8"))
        step0.append({"run": p.name[:-len("_history.json")], "first_batch": h["step0"]["first_batch_loss"], "train_set": h["step0"]["train_set_loss"]})
    out["dpo_step0"] = {"runs": len(step0), "max_abs_dev_from_ln2": max((max(abs(r["first_batch"] - math.log(2)), abs(r["train_set"] - math.log(2)))
                                                                       for r in step0), default=None), "per_run": step0}

    # laptop first pass vs. Ada (same configuration: r = 16, LR 1e-3, seed 42; and the full finetune)
    lap = MIDSEM / "results"
    cross = {}
    for lap_name, ada_name in [("lora_r16", "ada_r16_lr1e-3_s42"), ("lora_r4", "ada_r4_lr1e-3_s42"), ("lora_r1", "ada_r1_lr1e-3_s42"),
                               ("full_ft", "full_ft")]:
        a, b = (json.loads((lap / f"eval_{lap_name}.json").read_text()) if (lap / f"eval_{lap_name}.json").exists() else None), eval_row(ada_name)
        if a and b:
            cross[lap_name] = {s: {"laptop": a["sets"][s]["answer_em"], "ada": b[s]["answer_em"]} for s in ["val"] + SETS if s in b}
    out["laptop_vs_ada"] = cross

    (RA / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown(out)
    figures(out)
    print(json.dumps({"selections": sel, "lora": {r: {s: v[s]["mean"] if v.get(s) else None for s in ["val"] + SETS} for r, v in main_rows.items()},
                      "dpo_chosen": {b: {s: v[s]["mean"] if v.get(s) else None for s in SETS} for b, v in chosen.items()}}, indent=1))


def write_markdown(out):
    """results_ada/summary.md: the report's Part A tables as Markdown."""
    f = lambda v, d=1: "–" if not v else (f"{v['mean']:.{d}f} ± {v['sd']:.{d}f}" if v.get("sd") is not None else f"{v['mean']:.{d}f}")
    L = ["# Ada study summary (job 84119; mean ± sd over 3 seeds where marked)", "",
         f"Selections: LR per rank {dict((r, out['selections'].get(f'lr_r{r}')) for r in RANKS)}; best LoRA {out['selections'].get('best_lora')}; "
         f"DPO β=0.1 → {out['selections'].get('dpo_b0.1', {}) and (out['selections']['dpo_b0.1']['lr'], out['selections']['dpo_b0.1']['step'])}, "
         f"β=0.5 → {out['selections'].get('dpo_b0.5', {}) and (out['selections']['dpo_b0.5']['lr'], out['selections']['dpo_b0.5']['step'])}", "",
         "| model | validation | seen entities | unseen entities | unseen wording | PPL (500 docs) | PPL (full split) |", "|---|---|---|---|---|---|---|"]
    ff, fs = out.get("full_ft"), out.get("full_ft_seeds")
    if ff:
        L.append(f"| Full FT (Phase 3 checkpoint) | {ff['val']['answer_em']} | {ff['seen_entities']['answer_em']} | {ff['unseen_entities']['answer_em']} | "
                 f"{ff['unseen_wording']['answer_em']} | {ff['ppl_500']} | {ff.get('ppl_full')} |")
    if fs:
        L.append(f"| Full FT, {fs['n_seeds']} seeds | {f(fs['val'])} | {f(fs['seen_entities'])} | {f(fs['unseen_entities'])} | {f(fs['unseen_wording'])} | {f(fs['ppl_500'], 2)} | – |")
    for r in RANKS:
        v = out["lora"].get(str(r))
        if v:
            L.append(f"| LoRA r={r} (LR {v['lr']}) | {f(v['val'])} | {f(v['seen_entities'])} | {f(v['unseen_entities'])} | {f(v['unseen_wording'])} | "
                     f"{f(v['ppl_500'], 2)} | {f(v['ppl_full'], 2)} |")
    br = out.get("best_lora_row")
    if br:
        L.append(f"| Best LoRA ({out['selections']['best_lora']}) = DPO reference | {br['val']['answer_em']} | {br['seen_entities']['answer_em']} | "
                 f"{br['unseen_entities']['answer_em']} | {br['unseen_wording']['answer_em']} | {br['ppl_500']} | {br.get('ppl_full')} |")
    for b in BETAS:
        c = out["dpo_chosen"].get(b)
        if c:
            L.append(f"| DPO β={b} chosen (LR {c['lr']}, {c['step']} steps) | {f(c['val'])} | {f(c['seen_entities'])} | {f(c['unseen_entities'])} | "
                     f"{f(c['unseen_wording'])} | {f(c['ppl_500'], 2)} | {f(c['ppl_full'], 2)} |")
    for b in BETAS:
        x = out["dpo_fixed_protocol"].get(b)
        if x:
            L.append(f"| DPO β={b} fixed (LR 1e-4, 500 steps) | {x['val']['answer_em']} | {x['seen_entities']['answer_em']} | {x['unseen_entities']['answer_em']} | "
                     f"{x['unseen_wording']['answer_em']} | {x['ppl_500']} | {x.get('ppl_full')} |")
    if out["base"].get("ppl_full"):
        L.append(f"| Pretrained base | – | – | 0 (Phase 3) | – | 57.95 | {out['base']['ppl_full']} |")
    s0 = out.get("dpo_step0") or {}
    L += ["", f"DPO step-0 loss: ln 2 in all {s0.get('runs')} Ada runs (max deviation {s0.get('max_abs_dev_from_ln2')})."]
    (RA / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def smooth(y, k=10):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.concatenate([np.full(k - 1, np.nan), np.convolve(y, np.ones(k) / k, mode="valid")])


def margin_figure(out):
    """A4 'margin over training for both runs': one panel per DPO LR, both betas, seed 42 (Ada),
    training batches (10-step mean) and the 400 held-out validation pairs; chosen step marked."""
    hs = {(b, lr): load(f"{dpo_tag(b, lr, 42)}_history.json") for b in BETAS for lr in DPO_LRS}
    if not any(hs.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, 2.5))
    for ax, lr in zip(axes, DPO_LRS):
        for b in BETAS:
            color, h = ps.BETA[b], hs[(b, lr)]
            if not h:
                continue
            steps = [r["step"] for r in h["train_log"]]
            ax.plot(steps, smooth([r["margin"] for r in h["train_log"]]), color=color, linewidth=1.2, alpha=0.9)
            ax.plot([r["step"] for r in h["val_log"]], [r["margin"] for r in h["val_log"]], "o--", color=color, markersize=2.8,
                    linewidth=0.9, markeredgecolor=ps.SURFACE, markeredgewidth=0.5)
            c = out["selections"].get(f"dpo_b{b}")
            if c and c["lr"] == lr:
                ax.axvline(c["step"], color=color, linestyle=":", linewidth=1.3)
        ax.axhline(0, color=ps.AXIS, linewidth=0.8)
        ax.set(title=f"DPO LR {lr}", xlabel="Training step (8 pairs each)")
    axes[0].set_ylabel("Mean log-ratio margin (nats)")
    handles = [Line2D([], [], color=ps.BETA[b], linewidth=1.4, label=f"β={b} training (10-step mean)") for b in BETAS] + \
              [Line2D([], [], color=ps.BETA[b], marker="o", markersize=3, linestyle="--", linewidth=0.9, label=f"β={b} held-out pairs") for b in BETAS] + \
              [Line2D([], [], color=ps.INK2, linestyle=":", linewidth=1.3, label="step chosen on validation EM")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=5, fontsize=6.5, columnspacing=1.0)
    fig.tight_layout()
    ps.save(fig, FIG / "fig_ada_dpo_margins.png")


def curves_figure(out):
    """A2: validation loss and drift PPL per epoch, mean +- sd over the 3 seeds at each rank's chosen LR (Ada),
    against the Phase 3 full finetune."""
    full_hist = MIDSEM / "project" / "hindi" / "checkpoints" / "finetune_history.json"
    if not out["lora"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 2.6))
    if full_hist.exists():
        eps = json.loads(full_hist.read_text())["epochs"]
        for ax, key in zip(axes, ["val_reasoning_loss", "pretrain_ppl"]):
            ax.plot([e["epoch"] for e in eps], [e[key] for e in eps], "o-", color=ps.FULL, markersize=4,
                    markeredgecolor=ps.SURFACE, markeredgewidth=0.8, label="Full FT (Phase 3)")
    for r in RANKS:
        color, row = ps.RANK[r], out["lora"].get(str(r))
        if not row:
            continue
        hs = [load(f"{sft_tag(r, row['lr'], s)}_history.json") for s in SEEDS]
        hs = [h for h in hs if h]
        for ax, key in zip(axes, ["val_reasoning_loss", "pretrain_ppl"]):
            m = np.array([[e[key] for e in h["history"]] for h in hs])
            x = np.arange(m.shape[1])
            mu, sd = m.mean(0), m.std(0, ddof=1) if len(hs) > 1 else np.zeros(m.shape[1])
            ax.plot(x, mu, "o-", color=color, markersize=4, markeredgecolor=ps.SURFACE, markeredgewidth=0.8,
                    label=f"LoRA r={r} (LR {row['lr']}, {len(hs)} seeds)")
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
    axes[0].set_yscale("log")
    axes[0].set(title="Reasoning validation loss", xlabel="Epoch", ylabel="Cross-entropy (log scale)")
    axes[1].set(title="Drift: PPL on 500 pretraining-validation documents", xlabel="Epoch", ylabel="PPL")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, FIG / "fig_ada_curves.png")


def logratio_figure(out):
    """Chosen and rejected log-ratios over DPO training (seed 42) for each beta's chosen LR and for LR 1e-4."""
    panels = []
    for b in BETAS:
        c = out["selections"].get(f"dpo_b{b}")
        for lr, label in ([(c["lr"], f"β={b}, chosen LR {c['lr']}")] if c else []) + [("1e-4", f"β={b}, LR 1e-4 (fixed protocol)")]:
            h = load(f"{dpo_tag(b, lr, 42)}_history.json")
            if h and (b, lr) not in [(p[0], p[1]) for p in panels]:
                panels.append((b, lr, label, h, c["step"] if c and c["lr"] == lr else None))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(ps.TEXT_WIDTH_IN, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (b, lr, label, h, chosen_step) in zip(axes, panels):
        steps = [r["step"] for r in h["train_log"]]
        ax.plot(steps, smooth([r["chosen_logratio"] for r in h["train_log"]]), color=ps.CORRECT_CELL, linewidth=1.4,
                label="chosen (correct reply)")
        ax.plot(steps, smooth([r["rejected_logratio"] for r in h["train_log"]]), color=ps.WRONG, linestyle="--", linewidth=1.4,
                label="rejected (wrong reply)")
        if chosen_step:
            ax.axvline(chosen_step, color=ps.INK2, linestyle=":", linewidth=1.2, label="chosen step")
        ax.axhline(0, color=ps.AXIS, linewidth=0.8)
        ax.set_title(label.replace(", ", "\n", 1), fontsize=7.5)
        ax.set_xlabel("Training step")
    axes[0].set_ylabel("log πθ/πref (nats, 10-step mean)")
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    ps.save(fig, FIG / "fig_ada_dpo_logratios.png")


def drift_figure(out):
    """A5: drift PPL (500 documents and whole validation split) for base, full FT, best LoRA and DPO models."""
    rows = []
    base_eval = MIDSEM / "results" / "eval_base.json"          # 500-document PPL of the base model (deterministic)
    base500 = json.loads(base_eval.read_text())["pretrain_val_ppl_500docs"] if base_eval.exists() else None
    if out["base"].get("ppl_full"):
        rows.append(("Pretrained\nbase", base500, out["base"]["ppl_full"], (0, 0)))
    if out.get("full_ft"):
        rows.append(("Full FT\n(Phase 3)", out["full_ft"]["ppl_500"], out["full_ft"].get("ppl_full"), (0, 0)))
    br = out.get("best_lora_row")
    if br:
        rows.append((f"Best LoRA\n({out['selections']['best_lora'].split('_')[1]})", br["ppl_500"], br.get("ppl_full"), (0, 0)))
    for b in BETAS:
        c = out["dpo_chosen"].get(b)
        if c and c.get("ppl_500"):
            rows.append((f"DPO β={b}\n(chosen)", c["ppl_500"]["mean"], (c["ppl_full"] or {}).get("mean"),
                         (c["ppl_500"].get("sd") or 0, (c["ppl_full"] or {}).get("sd") or 0)))
    for b in BETAS:
        f = out["dpo_fixed_protocol"].get(b)
        if f:
            rows.append((f"DPO β={b}\n(fixed)", f["ppl_500"], f.get("ppl_full"), (0, 0)))
    if len(rows) < 2:
        return
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH_IN, 2.5))
    x = np.arange(len(rows))
    w = 0.36
    kw = dict(capsize=2, edgecolor=ps.SURFACE, linewidth=0.6, error_kw={"elinewidth": 0.8})
    b1 = ax.bar(x - w / 2, [r[1] or 0 for r in rows], w, yerr=[r[3][0] or np.nan for r in rows], color=ps.AXIS,
                label="first 500 documents (Phase 3 measure)", **kw)
    b2 = ax.bar(x + w / 2, [r[2] or 0 for r in rows], w, yerr=[r[3][1] or np.nan for r in rows], color=ps.INK2,
                label="whole validation split (Phase 2 packing, 23.8M tokens)", **kw)
    ax.bar_label(b1, fmt="%.1f", fontsize=6, padding=1.5, color=ps.INK2)
    ax.bar_label(b2, fmt="%.1f", fontsize=6, padding=1.5, color=ps.INK2)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Perplexity", ylim=(0, max(max(r[1] or 0, r[2] or 0) for r in rows) * 1.18))
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, FIG / "fig_ada_drift.png")


def figures(out):
    FIG.mkdir(parents=True, exist_ok=True)
    margin_figure(out)
    curves_figure(out)
    logratio_figure(out)
    drift_figure(out)
    # DPO sweep: validation EM by step, one panel per LR, both betas
    if any(v["val_em_by_step"][0] is not None for v in out["dpo_sweep"].values()):
        ref = out["best_lora_row"]["val"]["answer_em"] if out.get("best_lora_row") else None
        fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, 2.4), sharey=True)
        for ax, lr in zip(axes, DPO_LRS):
            if ref is not None:
                ax.axhline(ref, color=ps.INK2, linestyle=":", linewidth=1, label="best LoRA (reference, step 0)")
            for b in BETAS:
                v = out["dpo_sweep"][f"{b}|{lr}"]["val_em_by_step"]
                ax.plot([0] + DPO_STEPS, [ref] + v, marker="o", color=ps.BETA[b], markersize=4.5,
                        markeredgecolor=ps.SURFACE, markeredgewidth=0.8, label=f"β={b}")
                c = out["selections"].get(f"dpo_b{b}")
                if c and c["lr"] == lr:
                    ax.plot([c["step"]], [c["val_answer_em"]], marker="o", markersize=9, markerfacecolor="none",
                            markeredgecolor=ps.BETA[b], markeredgewidth=1.2, linestyle="none")
            ax.set(title=f"DPO LR {lr}", xlabel="DPO step")
        axes[0].set_ylabel("Validation answer EM (%)")
        axes[0].legend(fontsize=6.5, loc="lower left")
        fig.tight_layout()
        ps.save(fig, FIG / "fig_a4_dpo_sweep_val.png")
    # main table as bars with seed error bars
    if out["lora"]:
        labels = ["Full FT (Phase 3)"] + [f"LoRA r={r}" for r in RANKS if str(r) in out["lora"]] + \
                 [f"DPO β={b}" for b in BETAS if b in out["dpo_chosen"]]
        fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH_IN, 2.8))
        width = 0.8 / len(labels)
        colors = [ps.FULL, ps.RANK[1], ps.RANK[4], ps.RANK[16], ps.BETA["0.1"], ps.BETA["0.5"]]
        for i, lab in enumerate(labels):
            means, sds = [], []
            for s in SETS:
                if lab.startswith("Full"):
                    means.append(out["full_ft"][s]["answer_em"] if out["full_ft"] else 0)
                    sds.append(0)
                elif lab.startswith("LoRA"):
                    v = out["lora"][lab.split("=")[1]][s]
                    means.append(v["mean"] if v else 0)
                    sds.append((v["sd"] or 0) if v else 0)
                else:
                    v = out["dpo_chosen"][lab.split("=")[1]][s]
                    means.append(v["mean"] if v else 0)
                    sds.append((v["sd"] or 0) if v else 0)
            x = np.arange(len(SETS)) + (i - (len(labels) - 1) / 2) * width
            bars = ax.bar(x, means, width, yerr=[s or np.nan for s in sds], capsize=2, color=colors[i % len(colors)], label=lab,
                          edgecolor=ps.SURFACE, linewidth=0.6, error_kw={"elinewidth": 0.8})
            ax.bar_label(bars, fmt="%.1f", fontsize=5.5, padding=1, color=ps.INK2, zorder=4,
                         bbox=dict(boxstyle="square,pad=0.05", facecolor=ps.SURFACE, edgecolor="none"))
        ax.set_xticks(np.arange(len(SETS)), ["Seen entities", "Unseen entities", "Unseen wording"])
        ax.grid(axis="x", visible=False)
        ax.set(ylabel="Answer EM (%)", ylim=(0, 100))
        ax.axhline(40.3, color=ps.INK2, linestyle=":", linewidth=1, label="chance (≈40%)")
        ax.legend(ncol=7, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
        fig.tight_layout()
        ps.save(fig, FIG / "fig_ada_main.png")


if __name__ == "__main__":
    main()
