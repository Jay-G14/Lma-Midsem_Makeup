#!/usr/bin/env python3
"""Writes midsem/report/numbers.tex: one LaTeX macro per reported number, read from midsem/results/*.json,
so the report never contains a hand-copied number. Missing results become a red "??"."""

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parent / "numbers.tex"

MODELS = {"full_ft": "Full", "lora_r1": "LoraOne", "lora_r4": "LoraFour", "lora_r16": "LoraSixteen",
          "dpo_beta0.1": "DpoLow", "dpo_beta0.5": "DpoHigh", "base": "Base"}
SETS = {"seen_entities": "Seen", "unseen_entities": "UnEnt", "unseen_wording": "UnWord", "val": "Val"}
RANK_WORD = {1: "One", 4: "Four", 16: "Sixteen"}


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    m = {}

    def put(key, value, fmt="{}"):
        text = fmt.format(value) if value is not None else r"\textcolor{red}{??}"
        if text.startswith("-"):
            text = r"\ensuremath{-}" + text[1:]   # a real minus sign, valid in text and in math mode
        m[key] = text

    for model, tag in MODELS.items():
        ev = load(f"eval_{model}.json")
        put(f"Ppl{tag}", ev["pretrain_val_ppl_500docs"] if ev else None, "{:.2f}")
        for s, stag in SETS.items():
            sets = ev.get("sets", {}) if ev else {}
            put(f"Acc{tag}{stag}", sets[s]["answer_em"] if s in sets else None, "{:.1f}")
            put(f"Chain{tag}{stag}", sets[s]["chain_em"] if s in sets else None, "{:.1f}")
            put(f"Ans{tag}{stag}", sets[s]["answered"] if s in sets else None, "{:.0f}")
        if ev and "sets" in ev:
            for task, ttag in [("transitive", "Tr"), ("numeric_extremum", "Ex"), ("numeric_pair", "Pa"), ("verbal_comparison", "Ve")]:
                for s, stag in [("unseen_entities", "UnEnt"), ("unseen_wording", "UnWord"), ("seen_entities", "Seen")]:
                    put(f"Task{tag}{stag}{ttag}", ev["sets"][s]["task_answer_em"][task], "{:.1f}")

    sanity = load("lora_sanity.json")
    for r, word in RANK_WORD.items():
        v = sanity["ranks"][str(r)] if sanity else None
        put(f"Par{word}", v["trainable"] if v else None, "{:,}")
        put(f"Tot{word}", v["total"] if v else None, "{:,}")
        put(f"Share{word}", v["trainable_share_pct"] if v else None, "{:.3f}")
        put(f"Grad{word}", v["step0_grad_norm_B"] if v else None, "{:.2f}")
        put(f"After{word}", v["after_one_step_max_abs_logit_diff"] if v else None, "{:.2f}")
        put(f"Ident{word}", ("yes" if v["step0_logits_bit_identical"] else "NO") if v else None)
        h = load(f"lora_r{r}_history.json")
        put(f"BestEp{word}", h["best_epoch"] if h else None)
        put(f"ValLoss{word}", h["best_val_reasoning_loss"] if h else None, "{:.3f}")
    put("ProbeShape", "{}$\\times${}".format(*sanity["probe_batch_shape"]) if sanity else None)

    lr = load("lr_choice.json")
    put("LoraLR", lr["chosen_lr"] if lr else None, "{:g}")
    if lr:
        for row in lr["probe"]:
            key = {1e-4: "A", 3e-4: "B", 1e-3: "C", 3e-3: "D"}.get(row["lr"])
            if key:
                put(f"Probe{key}Loss", row["best_val_reasoning_loss"], "{:.3f}")
                put(f"Probe{key}Ppl", row["ppl_by_epoch"][-1], "{:.2f}")
    full_hist = json.loads((RES.parent / "project/hindi/checkpoints/finetune_history.json").read_text())
    put("ValLossFull", min(e["val_reasoning_loss"] for e in full_hist["epochs"] if e["epoch"] > 0), "{:.3f}")
    best = load("best_lora.json")
    put("BestRank", best["rank"] if best else None)
    best_ev = load(f"eval_lora_r{best['rank']}.json") if best else None
    for s, stag in SETS.items():
        put(f"AccBest{stag}", best_ev["sets"][s]["answer_em"] if best_ev else None, "{:.1f}")
        put(f"ChainBest{stag}", best_ev["sets"][s]["chain_em"] if best_ev else None, "{:.1f}")
    put("PplBest", best_ev["pretrain_val_ppl_500docs"] if best_ev else None, "{:.2f}")

    mem = load("memory_profile.json")
    for tag in ("Full", "LoraOne", "LoraFour", "LoraSixteen"):
        for kind in ("MemPeak", "MemOpt", "MemGrad", "MemW", "MemSpeed", "MemTrain"):
            put(f"{kind}{tag}", None)
    put("MemSteps", None)
    put("MemGpu", None)
    if mem:
        for row in mem["rows"]:
            tag = "Full" if row["rank"] is None else "Lora" + RANK_WORD[row["rank"]]
            put(f"MemPeak{tag}", row["peak_mb"], "{:,.0f}")
            put(f"MemOpt{tag}", row["optimizer_mb"], "{:,.2f}")
            put(f"MemGrad{tag}", row["grad_mb"], "{:,.2f}")
            put(f"MemW{tag}", row["weights_mb"], "{:,.0f}")
            put(f"MemSpeed{tag}", row["steps_per_second"], "{:.1f}")
            put(f"MemTrain{tag}", row["trainable_params"], "{:,}")
        put("MemSteps", mem["steps"])
        put("MemGpu", mem["gpu"].replace("NVIDIA GeForce ", ""))

    for b, tag in (("0.1", "DpoLow"), ("0.5", "DpoHigh")):
        h = load(f"dpo_beta{b}_history.json")
        put(f"{tag}StepZero", h["step0"]["first_batch_loss"] if h else None, "{:.6f}")
        put(f"{tag}StepZeroAll", h["step0"]["train_set_loss"] if h else None, "{:.6f}")
        put(f"{tag}Steps", h["steps"] if h else None)
        put(f"{tag}MarginTrain", h["final_train_set"]["margin"] if h else None, "{:.1f}")
        put(f"{tag}ChosenTrain", h["final_train_set"]["chosen_logratio"] if h else None, "{:.1f}")
        put(f"{tag}RejectedTrain", h["final_train_set"]["rejected_logratio"] if h else None, "{:.1f}")
        put(f"{tag}LossTrain", h["final_train_set"]["loss"] if h else None, "{:.3f}")
        put(f"{tag}AccTrain", 100 * h["final_train_set"]["preference_accuracy"] if h else None, "{:.1f}")
        put(f"{tag}MarginVal", h["val_log"][-1]["margin"] if h else None, "{:.1f}")
        put(f"{tag}ChosenVal", h["val_log"][-1]["chosen_logratio"] if h else None, "{:.1f}")
        put(f"{tag}RejectedVal", h["val_log"][-1]["rejected_logratio"] if h else None, "{:.1f}")
        put(f"{tag}AccVal", 100 * h["val_log"][-1]["preference_accuracy"] if h else None, "{:.1f}")
        put(f"{tag}LossVal", h["val_log"][-1]["loss"] if h else None, "{:.3f}")
        put(f"{tag}LR", h["hparams"]["lr"] if h else None, "{:g}")
        put(f"{tag}Epochs", h["hparams"]["epochs"] if h else None)

    abl_h, abl_e = load("ablation_r1_lr2.5e-4_history.json"), load("eval_ablation_r1_lr2.5e-4.json")
    put("AblValLoss", abl_h["best_val_reasoning_loss"] if abl_h else None, "{:.3f}")
    put("AblValEM", abl_e["sets"]["val"]["answer_em"] if abl_e else None, "{:.1f}")
    put("AblUnEnt", abl_e["sets"]["unseen_entities"]["answer_em"] if abl_e else None, "{:.1f}")

    ts = load("preference_pairs_token_stats.json")
    put("PairReplyTok", ts["mean_reply_tokens_chosen"] if ts else None, "{:.1f}")
    put("PairSharedTok", ts["mean_shared_prefix_tokens"] if ts else None, "{:.1f}")
    put("PairSharedPct", 100 * ts["mean_shared_fraction"] if ts else None, "{:.0f}")

    c1 = load("c1_results.json")
    for cond, tag in (("direct", "Direct"), ("cot", "Cot"), ("sc", "Sc")):
        r = c1["results"][cond] if c1 else None
        put(f"C{tag}Std", r["standard_accuracy"] if r else None, "{:.1f}")
        put(f"C{tag}StdN", f"{r['standard_correct']}/{r['standard_n']}" if r else None)
        put(f"C{tag}Mod", r["modified_accuracy"] if r else None, "{:.1f}")
        put(f"C{tag}ModN", f"{r['modified_correct']}/{r['modified_n']}" if r else None)
        put(f"C{tag}All", r["overall_accuracy"] if r else None, "{:.1f}")
        put(f"C{tag}Tok", r["mean_generated_tokens"] if r else None, "{:,.1f}")
        put(f"C{tag}Tmpl", r["modified_template_answer_rate"] if r else None, "{:.0f}")
        put(f"C{tag}Trunc", r["truncated_replies"] if r else None)
    put("CScTokSample", c1["results"]["sc"]["mean_generated_tokens_per_sample"] if c1 else None, "{:.1f}")
    if c1:   # tokens spent relative to the direct answer
        d_tok = c1["results"]["direct"]["mean_generated_tokens"]
        put("CCotTokRatio", c1["results"]["cot"]["mean_generated_tokens"] / d_tok, "{:.0f}")
        put("CScTokRatio", c1["results"]["sc"]["mean_generated_tokens"] / d_tok, "{:.0f}")
    put("CDeva", 100 * c1["results"]["cot_devanagari_share"] if c1 else None, "{:.1f}")

    c2 = load("c2_results.json")
    if c2:
        put("CTwoPairs", c2["pairs"])
        put("CTwoShortRight", c2["pair_mix"].get("direct", 0))
        c2_pairs = [json.loads(l) for l in (RES.parent / "data" / "c2_pairs_with_verdicts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        put("CTwoShortRightProblems", ", ".join(sorted({x["problem_id"] for x in c2_pairs if x["checker_winner"] == "direct"})))
        put("CTwoLongRight", c2["pair_mix"].get("cot", 0))
        put("CTwoPosBias", 100 * c2["position_bias_rate"], "{:.1f}")
        put("CTwoPosBiasN", round(c2["position_bias_rate"] * c2["pairs"]))
        put("CTwoConsistent", c2["order_consistent_pairs"])
        put("CTwoAgree", 100 * c2["agreement_with_checker_consistent"] if c2["agreement_with_checker_consistent"] is not None else None, "{:.1f}")
        put("CTwoKappa", c2["cohen_kappa_consistent"], "{:.2f}")
        put("CTwoAgreeAll", 100 * c2["agreement_with_checker_all_verdicts"], "{:.1f}")
        put("CTwoLonger", 100 * c2["longer_answer_win_rate"], "{:.1f}")
        put("CTwoLongerN", round(c2["longer_answer_win_rate"] * c2["verdicts"]))
        put("CTwoFirst", 100 * c2["first_position_win_rate"], "{:.1f}")
        put("CTwoFirstN", round(c2["first_position_win_rate"] * c2["verdicts"]))
        put("CTwoConsCot", c2["consistent_preferences"].get("cot", 0))
        put("CTwoConsDirect", c2["consistent_preferences"].get("direct", 0))
        put("CTwoTopAB", c2["top_token_is_A_or_B"])
        put("CTwoProblems", c2.get("problems_covered"))
        avg = c2.get("order_averaged")
        put("CTwoAvgAgree", 100 * avg["agreement_with_checker"] if avg else None, "{:.1f}")
        put("CTwoAvgKappa", avg["cohen_kappa"] if avg else None, "{:.2f}")
        put("CTwoAvgCot", avg["prefers_cot"] if avg else None)

    # ---- Ada study (multi-seed), from midsem/results_ada/summary.json ----
    ada = None
    ada_path = RES.parent / "results_ada" / "summary.json"
    if ada_path.exists():
        ada = json.loads(ada_path.read_text(encoding="utf-8"))
    SETTAG = {"val": "Val", "seen_entities": "Seen", "unseen_entities": "UnEnt", "unseen_wording": "UnWord"}

    def pm(v, digits=1):
        """'mean ± sd' (or just the value for a single run)."""
        if not v:
            return None
        s = f"{v['mean']:.{digits}f}"
        return s + (f"$\\pm${v['sd']:.{digits}f}" if v.get("sd") is not None else "")

    for r, word in RANK_WORD.items():
        row = (ada or {}).get("lora", {}).get(str(r))
        put(f"AdaLr{word}", row["lr"] if row else None)
        for s, stag in SETTAG.items():
            put(f"AdaAcc{word}{stag}", pm(row.get(s)) if row else None)
            put(f"AdaChain{word}{stag}", pm(row.get(s + "_chain")) if row else None)
        put(f"AdaPpl{word}", pm(row.get("ppl_500"), 2) if row else None)
        put(f"AdaPplFull{word}", pm(row.get("ppl_full"), 2) if row else None)
        put(f"AdaLoss{word}", pm(row.get("best_val_loss"), 3) if row else None)
    ff = (ada or {}).get("full_ft")
    for s, stag in SETTAG.items():
        put(f"AdaAccFull{stag}", ff[s]["answer_em"] if ff and s in ff else None, "{:.1f}")
        put(f"AdaChainFull{stag}", ff[s]["chain_em"] if ff and s in ff else None, "{:.1f}")
    fseeds = (ada or {}).get("full_ft_seeds")
    for s, stag in SETTAG.items():
        put(f"FullSeeds{stag}", pm(fseeds.get(s)) if fseeds else None)
        put(f"FullSeedsChain{stag}", pm(fseeds.get(s + "_chain")) if fseeds else None)
    put("FullSeedsPpl", pm(fseeds.get("ppl_500"), 2) if fseeds else None)
    # seed ranges on unseen entities and the drop from unseen entities to unseen wording (means)
    rng = lambda v: f"{min(v['values']):.1f}--{max(v['values']):.1f}" if v else None
    put("FullSeedsRangeUnEnt", rng(fseeds.get("unseen_entities")) if fseeds else None)
    l16 = (ada or {}).get("lora", {}).get("16")
    put("SixteenRangeUnEnt", rng(l16.get("unseen_entities")) if l16 else None)
    put("FullSeedsWordDrop", (fseeds["unseen_entities"]["mean"] - fseeds["unseen_wording"]["mean"]) if fseeds else None, "{:.0f}")
    put("SixteenWordDrop", (l16["unseen_entities"]["mean"] - l16["unseen_wording"]["mean"]) if l16 else None, "{:.0f}")
    put("FullSeedsN", fseeds["n_seeds"] if fseeds else None)
    put("AdaPplFull", ff.get("ppl_500") if ff else None, "{:.2f}")
    put("AdaPplFullFull", ff.get("ppl_full") if ff else None, "{:.2f}")
    put("AdaPplFullBase", (ada or {}).get("base", {}).get("ppl_full"), "{:.2f}")
    best = (ada or {}).get("selections", {}).get("best_lora")
    put("AdaBest", f"LoRA $r={best.split('_')[1][1:]}$, seed {best.split('_s')[-1]}" if best else None)
    put("AdaBestRank", best.split("_")[1][1:] if best else None)
    put("AdaBestSeed", best.split("_s")[-1] if best else None)
    br = (ada or {}).get("best_lora_row")
    for s, stag in SETTAG.items():
        put(f"AdaAccBest{stag}", br[s]["answer_em"] if br and s in br else None, "{:.1f}")
    put("AdaPplBest", br.get("ppl_500") if br else None, "{:.2f}")
    put("AdaPplFullBest", br.get("ppl_full") if br else None, "{:.2f}")
    for b, tag in (("0.1", "Low"), ("0.5", "High")):
        c = (ada or {}).get("dpo_chosen", {}).get(b)
        put(f"AdaDpo{tag}Lr", c["lr"] if c else None)
        put(f"AdaDpo{tag}Step", c["step"] if c else None)
        for s, stag in SETTAG.items():
            put(f"AdaDpo{tag}{stag}", pm(c.get(s)) if c else None)
            put(f"AdaDpo{tag}Ans{stag}", pm(c.get(s + "_answered"), 0) if c else None)
        put(f"AdaDpo{tag}Ppl", pm(c.get("ppl_500"), 2) if c else None)
        put(f"AdaDpo{tag}PplFull", pm(c.get("ppl_full"), 2) if c else None)
        put(f"AdaDpo{tag}StepZero", "/".join(f"{x:.6f}" for x in c["step0_losses"]) if c and c.get("step0_losses") else None)
        fx = (ada or {}).get("dpo_fixed_protocol", {}).get(b)
        for s, stag in SETTAG.items():
            put(f"AdaFix{tag}{stag}", fx[s]["answer_em"] if fx and s in fx else None, "{:.1f}")
            put(f"AdaFix{tag}Ans{stag}", fx[s]["answered"] if fx and s in fx else None, "{:.0f}")
        put(f"AdaFix{tag}Ppl", fx.get("ppl_500") if fx else None, "{:.2f}")
        put(f"AdaFix{tag}PplFull", fx.get("ppl_full") if fx else None, "{:.2f}")
    s0 = (ada or {}).get("dpo_step0") or {}
    put("AdaStepZeroRuns", s0.get("runs") or None)
    put("AdaStepZeroDev", None if s0.get("max_abs_dev_from_ln2") is None else f"{s0['max_abs_dev_from_ln2']:.1e}".replace("e-0", "e-"))
    put("LocalPplFullBase", (load("ppl_full_base.json") or {}).get("perplexity"), "{:.2f}")
    put("LocalPplFullFull", (load("ppl_full_full_ft.json") or {}).get("perplexity"), "{:.2f}")
    c1 = load("c1_results.json")
    scs = (c1 or {}).get("results", {}).get("sc_seeds")
    put("CScStdMean", scs["standard_accuracy_mean"] if scs else None, "{:.1f}")
    put("CScStdSd", scs["standard_accuracy_sd"] if scs else None, "{:.1f}")
    put("CScAllMean", scs["overall_accuracy_mean"] if scs else None, "{:.1f}")
    put("CScAllSd", scs["overall_accuracy_sd"] if scs else None, "{:.1f}")
    put("CScModMean", scs["modified_accuracy_mean"] if scs else None, "{:.1f}")
    put("CScTokMean", scs["mean_generated_tokens_mean"] if scs else None, "{:,.0f}")
    put("CScSeedsStd", " / ".join(f"{v['standard_accuracy']:.1f}" for v in scs["per_seed"].values()) if scs else None)
    c2f = load("c2_results_fp32.json") or (json.loads((RES.parent / "results_ada" / "c2_results_fp32.json").read_text())
                                         if (RES.parent / "results_ada" / "c2_results_fp32.json").exists() else None)
    if c2f:
        put("FpPosBias", 100 * c2f["position_bias_rate"], "{:.1f}")
        put("FpPosBiasN", round(c2f["position_bias_rate"] * c2f["pairs"]))
        put("FpFirst", 100 * c2f["first_position_win_rate"], "{:.1f}")
        put("FpConsistent", c2f["order_consistent_pairs"])
        put("FpAgree", 100 * c2f["agreement_with_checker_consistent"] if c2f["agreement_with_checker_consistent"] is not None else None, "{:.1f}")
        put("FpKappa", c2f["cohen_kappa_consistent"], "{:.2f}")
        put("FpLonger", 100 * c2f["longer_answer_win_rate"], "{:.1f}")
        put("FpAgreeAll", 100 * c2f["agreement_with_checker_all_verdicts"], "{:.1f}")
        put("FpAvgAgree", 100 * c2f["order_averaged"]["agreement_with_checker"], "{:.1f}")
        put("FpAvgKappa", c2f["order_averaged"]["cohen_kappa"], "{:.2f}")
        put("FpConsCot", c2f["consistent_preferences"].get("cot", 0))
        put("FpConsDirect", c2f["consistent_preferences"].get("direct", 0))
    else:
        for k in ("FpPosBias", "FpFirst", "FpConsistent", "FpAgree", "FpKappa", "FpLonger", "FpAgreeAll", "FpAvgAgree", "FpAvgKappa",
                  "FpConsCot", "FpConsDirect"):
            put(k, None)

    # C2 per-pair breakdowns, for the 4-bit (laptop) and fp32 (Ada) verdict files
    for tag, path in (("Q", RES.parent / "data" / "c2_pairs_with_verdicts.jsonl"),
                      ("Fp", RES.parent / "data_ada" / "c2_pairs_with_verdicts_fp32.jsonl")):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
        if rows:
            put(f"{tag}PaDirectFirst", sum(r["verdict_direct_first"]["p_A"] for r in rows) / len(rows), "{:.2f}")
            put(f"{tag}PaCotFirst", sum(r["verdict_cot_first"]["p_A"] for r in rows) / len(rows), "{:.2f}")
            for w, wtag in (("direct", "Short"), ("cot", "Long")):
                sub = [r for r in rows if r["checker_winner"] == w]
                put(f"{tag}{wtag}Right", sum(v["prefers"] == w for r in sub for v in (r["verdict_direct_first"], r["verdict_cot_first"])))
                put(f"{tag}{wtag}N", 2 * len(sub))
        else:
            for k in ("PaDirectFirst", "PaCotFirst", "ShortRight", "ShortN", "LongRight", "LongN"):
                put(f"{tag}{k}", None)

    lines = ["% generated by make_numbers.py - do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [k for k, v in m.items() if "??" in v]
    print(f"wrote {len(m)} macros to {OUT}; missing: {len(missing)}")


if __name__ == "__main__":
    main()
