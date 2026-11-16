#!/usr/bin/env python3
"""Writes midsem/report/appendix_tables.tex (LR probe, per-task accuracy, preference-pair examples,
unseen-wording examples, C1 per-problem table, C2 per-pair verdicts) from midsem/results and midsem/data."""

import json
from pathlib import Path

MIDSEM = Path(__file__).resolve().parents[1]
RES, DATA = MIDSEM / "results", MIDSEM / "data"
OUT = Path(__file__).resolve().parent / "appendix_tables.tex"


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def tex(s: str) -> str:
    """Escapes LaTeX specials and wraps Devanagari runs in \\dn{}."""
    s = s.replace("{", "").replace("}", "")      # strip braces first, so the macros below keep their {}
    s = s.replace("\\", "/").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#").replace("_", "\\_")
    s = s.replace("<", "\\textless{}").replace(">", "\\textgreater{}").replace("$", "\\$")
    out, buf, dev = [], [], None
    for ch in s:
        is_dev = "\u0900" <= ch <= "\u097F"
        if dev is None or ch == " " or is_dev == dev:
            buf.append(ch)
            if ch != " ":
                dev = is_dev
            continue
        out.append(("\\dn{" + "".join(buf) + "}") if dev else "".join(buf))
        buf, dev = [ch], is_dev
    if buf:
        out.append(("\\dn{" + "".join(buf) + "}") if dev else "".join(buf))
    return "".join(out).replace("\\dn{ ", " \\dn{")


def main():
    L = []
    lr = load(RES / "lr_choice.json")
    if lr:
        L += ["\\subsection*{First pass (laptop RTX 4050, seed 42): LoRA learning-rate probe at $r=4$ ($\\alpha=16$, 5 epochs)}",
              "\\begin{tabular}{lcccc}\\toprule LR & best epoch & best val.\\ reasoning loss & val.\\ loss by epoch 1--5 & PPL after epoch 5\\\\\\midrule"]
        for r in lr["probe"]:
            L.append(f"{r['lr']:g} & {r['best_epoch']} & {r['best_val_reasoning_loss']:.4f} & {', '.join(f'{v:.3f}' for v in r['val_loss_by_epoch'][1:])} & {r['ppl_by_epoch'][-1]:.2f}\\\\")
        L += ["\\bottomrule\\end{tabular}", f"\n\nChosen: {lr['chosen_lr']:g} (lowest best-epoch validation loss).\n"]
        ah, ae = load(RES / "ablation_r1_lr2.5e-4_history.json"), load(RES / "eval_ablation_r1_lr2.5e-4.json")
        if ah and ae:
            L.append(f"\n\\textbf{{Control for $r=1$.}} With $\\alpha=16$ fixed, $r=1$ scales its update 4$\\times$ more than $r=4$ at the same LR. "
                     f"Re-running $r=1$ at LR $2.5\\times10^{{-4}}$ (so $\\frac{{\\alpha}}{{r}}\\cdot$LR equals $r=4$'s) gives a best validation loss of "
                     f"{ah['best_val_reasoning_loss']:.3f} (epoch {ah['best_epoch']}), validation EM {ae['sets']['val']['answer_em']:.1f}\\% and unseen-entity EM "
                     f"{ae['sets']['unseen_entities']['answer_em']:.1f}\\% (chain EM {ae['sets']['unseen_entities']['chain_em']:.1f}\\%). This is still chance level, "
                     f"so the failure of $r=1$ is a capacity limit, not an LR artefact.\n")

    rows = [("Full FT (Phase 3)", "full_ft"), ("LoRA r=1", "lora_r1"), ("LoRA r=4", "lora_r4"), ("LoRA r=16", "lora_r16"),
            ("DPO beta=0.1 (LR 1e-4, 500 steps)", "dpo_beta0.1"), ("DPO beta=0.5 (LR 1e-4, 500 steps)", "dpo_beta0.5")]
    evs = [(lab, load(RES / f"eval_{n}.json")) for lab, n in rows]
    evs = [(lab, e) for lab, e in evs if e and "sets" in e]
    if evs:
        L += ["\\subsection*{First pass (laptop, seed 42, LR $10^{-3}$ for every rank): answer EM \\% (chain EM)}",
              "\\begin{tabular}{lcccc}\\toprule model & val & seen ent. & unseen ent. & unseen word.\\\\\\midrule"]
        for lab, e in evs:
            cells = [f"{e['sets'][s]['answer_em']:.1f} ({e['sets'][s]['chain_em']:.1f})" for s in ("val", "seen_entities", "unseen_entities", "unseen_wording")]
            L.append(lab.replace("beta", "$\\beta$") + " & " + " & ".join(cells) + "\\\\")
        L += ["\\bottomrule\\end{tabular}", ""]

    # Ada per-task accuracy and chain EM (means over the 3 seeds)
    RA = MIDSEM / "results_ada"
    ada = load(RA / "summary.json")
    if ada and ada.get("lora"):
        import statistics as st
        groups = [("Full FT (Phase 3)", ["full_ft"])]
        for r in ("1", "4", "16"):
            if r in ada["lora"]:
                lr = ada["lora"][r]["lr"]
                groups.append((f"LoRA $r={r}$", [f"ada_r{r}_lr{lr}_s{s}" for s in (42, 43, 44)]))
        for b in ("0.1", "0.5"):
            c = ada.get("dpo_chosen", {}).get(b)
            if c:
                groups.append((f"DPO $\\beta={b}$ (chosen)", c["models"]))
        tasks = [("transitive", "trans."), ("numeric_extremum", "extr."), ("numeric_pair", "pair"), ("verbal_comparison", "verbal")]
        L += ["\\subsection*{Ada study: answer EM (\\%) per task, mean over seeds}",
              "\\begin{tabular}{l" + "cccc" * 3 + "}\\toprule",
              " & \\multicolumn{4}{c}{seen entities} & \\multicolumn{4}{c}{unseen entities} & \\multicolumn{4}{c}{unseen wording}\\\\",
              "model" + "".join(f" & {t[1]}" for t in tasks) * 3 + "\\\\\\midrule"]
        chain_rows = []
        for lab, names in groups:
            evs2 = [load(RA / f"eval_{n}.json") for n in names]
            evs2 = [e for e in evs2 if e and "unseen_wording" in e.get("sets", {})]
            if not evs2:
                continue
            cells = []
            for sname in ("seen_entities", "unseen_entities", "unseen_wording"):
                for t, _ in tasks:
                    cells.append(f"{st.mean(e['sets'][sname]['task_answer_em'][t] for e in evs2):.1f}")
            L.append(lab + " & " + " & ".join(cells) + "\\\\")
            ch = []
            for sname in ("val", "seen_entities", "unseen_entities", "unseen_wording"):
                v = [e["sets"][sname]["chain_em"] for e in evs2]
                ch.append(f"{st.mean(v):.1f}" + (f"$\\pm${st.stdev(v):.1f}" if len(v) > 1 else ""))
            chain_rows.append(lab + " & " + " & ".join(ch) + "\\\\")
        L += ["\\bottomrule\\end{tabular}", ""]
        L += ["\\subsection*{Ada study: chain EM (\\%, the whole generated reasoning equals the gold chain), mean$\\pm$sd over seeds}",
              "\\begin{tabular}{lcccc}\\toprule model & val & seen ent. & unseen ent. & unseen word.\\\\\\midrule"] + chain_rows + ["\\bottomrule\\end{tabular}", ""]

    pairs = jsonl(DATA / "part_a_preference_pairs.jsonl")
    if pairs:
        L += ["\\subsection*{A3: one preference pair per question type (prompt abridged to the question)}", "\\begin{footnotesize}"]
        seen = set()
        for p in pairs:
            if p["question_type"] in seen:
                continue
            seen.add(p["question_type"])
            q = p["prompt"].split("\n")[0]
            L += [f"\\textbf{{{tex(p['question_type'])}}} ({tex(p['rejected_error'])}).\\\\",
                  f"{tex(q)}\\\\", f"\\textit{{chosen:}} {tex(p['chosen'].strip())}\\\\",
                  f"\\textit{{rejected:}} {tex(p['rejected'].strip())}\\\\[3pt]"]
        L += ["\\end{footnotesize}", ""]

    uw = jsonl(DATA / "test_unseen_wording.jsonl")
    if uw:
        L += ["\\subsection*{Unseen-wording test set: examples}", "\\begin{footnotesize}"]
        seen = set()
        for r in uw:
            if r["task"] in seen:
                continue
            seen.add(r["task"])
            L.append(f"{tex(r['task'])}: {tex(r['question'])} $\\rightarrow$ {tex(r['answer'])}\\\\")
        L += ["\\end{footnotesize}", ""]

    c1 = jsonl(DATA / "c1_outputs.jsonl")
    if c1:
        L += ["\\subsection*{C1: per-problem results (extracted answer; \\checkmark\\ = correct by the checker)}",
              "\\begin{tabular}{llllrlrl}\\toprule id & gold & direct & CoT (greedy) & CoT tok. & SC votes $\\rightarrow$ majority (seed 1234) & SC tok. & seeds 2234 / 3234\\\\\\midrule"]
        ok = lambda e: "\\checkmark" if e["correct"] else ""
        short = lambda v: "--" if v is None else v.split(" (")[0]
        for r in c1:
            votes = ", ".join(short(v) for v in r["sc"]["votes"])
            L.append(f"{r['id']} & {r['gold']} & {short(r['direct']['extracted'])} {ok(r['direct'])} & {short(r['cot']['extracted'])} {ok(r['cot'])} & "
                     f"{r['cot']['tokens']}{'*' if r['cot']['truncated'] else ''} & {votes} $\\rightarrow$ {short(r['sc']['majority'])} {ok(r['sc'])} & {r['sc']['tokens']} & "
                     + " / ".join(f"{short(b['majority'])} {ok(b)}" for b in r.get("sc_other_seeds", {}).values()) + "\\\\")
        L += ["\\bottomrule\\end{tabular}", "\n\n{\\footnotesize * truncated at 1,024 tokens (repetition loop).}\n"]

    c2 = jsonl(DATA / "c2_pairs_with_verdicts.jsonl")
    fp = {x["pair_id"]: x for x in jsonl(MIDSEM / "data_ada" / "c2_pairs_with_verdicts_fp32.jsonl")}
    if c2:
        L += ["\\subsection*{C2: the 30 pairs and every verdict (preferred reply; $p_A$ = judge's probability of ``A'')}",
              "\\begin{footnotesize}\\begin{tabular}{lllrrlllll}\\toprule pair & problem & CoT reply & dir.\\ chars & CoT chars & better & 4-bit: dir.\\ first & 4-bit: CoT first"
              + (" & fp32: dir.\\ first & fp32: CoT first" if fp else " & & ") + "\\\\\\midrule"]
        for x in c2:
            v1, v2 = x["verdict_direct_first"], x["verdict_cot_first"]
            row = (f"{x['pair_id']} & {x['problem_id']} & {x['cot_source'].replace('_', ' ')} & {x['direct_chars']} & {x['cot_chars']} & {x['checker_winner']} & "
                   f"{v1['prefers']} ({v1['p_A']:.2f}) & {v2['prefers']} ({v2['p_A']:.2f})")
            f = fp.get(x["pair_id"])
            row += (f" & {f['verdict_direct_first']['prefers']} ({f['verdict_direct_first']['p_A']:.2f}) & {f['verdict_cot_first']['prefers']} ({f['verdict_cot_first']['p_A']:.2f})"
                    if f else " & & ")
            L.append(row + "\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{footnotesize}", ""]

    ada = load(MIDSEM / "results_ada" / "summary.json")
    if ada and ada.get("lr_grid") and any(r["best_val_loss"] is not None for r in ada["lr_grid"]):
        f = lambda v, d=1: "--" if v is None else f"{v:.{d}f}"
        L += ["\\subsection*{Ada study: LoRA learning-rate grid (seed 42, 5 epochs; choice = lowest validation loss per rank)}",
              "\\begin{tabular}{llcccc}\\toprule rank & LR & best val.\\ loss (epoch) & val.\\ EM & unseen-entity EM & chosen\\\\\\midrule"]
        for r in ada["lr_grid"]:
            chosen = ada["selections"].get(f"lr_r{r['rank']}") == r["lr"]
            L.append(f"$r={r['rank']}$ & {r['lr']} & {f(r['best_val_loss'], 4)} ({r['best_epoch'] or '--'}) & {f(r['val_em'])} & "
                     f"{f(r['unseen_entities_em'])} & {'\\checkmark' if chosen else ''}\\\\")
        L += ["\\bottomrule\\end{tabular}", ""]
    if ada and ada.get("dpo_sweep") and any(v["val_em_by_step"][0] is not None for v in ada["dpo_sweep"].values()):
        ref = ada["best_lora_row"]["val"]["answer_em"] if ada.get("best_lora_row") else None
        L += [f"\\subsection*{{Ada study: DPO sweep, validation answer EM (\\%) of the policy saved every 100 steps (seed 42; step 0 = best LoRA, {ref})}}",
              "\\begin{tabular}{ll" + "c" * 5 + "cc}\\toprule $\\beta$ & LR & 100 & 200 & 300 & 400 & 500 & val.\\ margin (500) & pref.\\ acc.\\\\\\midrule"]
        for key, v in ada["dpo_sweep"].items():
            b, lr = key.split("|")
            c = ada["selections"].get(f"dpo_b{b}") or {}
            cells = []
            for step, em in zip([100, 200, 300, 400, 500], v["val_em_by_step"]):
                s = "--" if em is None else f"{em:.1f}"
                cells.append(f"\\textbf{{{s}}}" if c.get("lr") == lr and c.get("step") == step else s)
            margin = "--" if v["final_val_margin"] is None else f"{v['final_val_margin']:.1f}"
            acc = "--" if v["final_val_pref_acc"] is None else f"{100 * v['final_val_pref_acc']:.1f}\\%"
            L.append(f"{b} & {lr} & " + " & ".join(cells) + f" & {margin} & {acc}\\\\")
        L += ["\\bottomrule\\end{tabular}", "", "Bold: the configuration chosen for each $\\beta$ (highest validation EM; ties go to fewer steps).", ""]
    if ada and ada.get("laptop_vs_ada"):
        L += ["\\subsection*{Hardware cross-check: the first pass (laptop RTX 4050) vs.\\ the same configuration on Ada}",
              "\\begin{tabular}{lcccc}\\toprule model & val & seen ent. & unseen ent. & unseen word.\\\\\\midrule"]
        names = {"lora_r16": "LoRA $r=16$, LR $10^{-3}$, seed 42", "lora_r4": "LoRA $r=4$, LR $10^{-3}$, seed 42",
                 "lora_r1": "LoRA $r=1$, LR $10^{-3}$, seed 42", "full_ft": "Full FT (Phase 3 checkpoint)"}
        for k, v in ada["laptop_vs_ada"].items():
            cells = [f"{v[s]['laptop']:.1f} / {v[s]['ada']:.1f}" if s in v else "--" for s in ["val", "seen_entities", "unseen_entities", "unseen_wording"]]
            L.append(f"{names.get(k, k)} & " + " & ".join(cells) + "\\\\")
        L += ["\\bottomrule\\end{tabular}", "", "Each cell: laptop / Ada answer EM (\\%). Same code, seed and data; only the GPU differs.", ""]

    # The report keeps only the tables the main text cites; the rest stay in the repository (results/, data/).
    keep_prefixes = ("\\subsection*{First pass (laptop, seed 42, LR", "\\subsection*{A3: one preference pair",
                     "\\subsection*{Ada study: LoRA learning-rate grid", "\\subsection*{Ada study: DPO sweep")
    blocks, cur = [], []
    for line in L:
        if line.startswith("\\subsection*") and cur:
            blocks.append(cur); cur = []
        cur.append(line)
    if cur:
        blocks.append(cur)
    L = [line for blk in blocks if blk[0].startswith(keep_prefixes) for line in blk]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
