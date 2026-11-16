#!/usr/bin/env python3
"""C2: Can you trust an LLM judge?

Pairs (30): each pairs Qwen's DIRECT reply to a problem with one of its CHAIN-OF-THOUGHT replies to
the same problem (the greedy CoT reply or one of the 5 self-consistency samples). Only pairs in which
exactly one reply is correct by the checker are used, so the checker names a unique better answer.
They are balanced between "shorter (direct) reply correct, longer (CoT) wrong" and the reverse, and
spread over as many problems as possible (round-robin, greedy CoT reply first, then SC samples).

Judge: Meta Llama-3.2-3B-Instruct (a different family from Qwen), 4-bit NF4 weights with bf16 compute.
It sees the Hindi problem and the two replies as "Answer A" / "Answer B" and is asked which is
better. Its verdict is the more probable of the two next tokens "A" and "B" (a forced choice, so every
verdict is parseable); the unrestricted most likely next token is logged too. Every pair is judged in
both orders, giving 60 verdicts.

Metrics:
  position-bias rate   share of the 30 pairs whose preferred reply changes when the order is swapped
                       (i.e. the judge picked the same position both times)
  agreement, kappa     over order-consistent pairs only: judge's preferred reply vs. the checker's,
                       raw agreement and Cohen's kappa (two raters, labels {direct, cot})
  longer-answer wins   share of all 60 verdicts that pick the longer reply
"""

import argparse
import json
from collections import Counter
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from midsem.part_a.common import DATA_DIR, OUT_DATA_DIR, RESULTS_DIR, write_json, write_jsonl

JUDGE_ID = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
SYSTEM = "You are an impartial judge who checks solutions to math word problems."
TEMPLATE = (
    "Below is a math word problem written in Hindi, followed by two candidate answers.\n\n"
    "Problem:\n{problem}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\n"
    "Which answer is better, that is, more likely to reach the correct final result? "
    "Reply with only one letter: A or B."
)


def build_pairs(rows: List[Dict], n_pairs: int) -> List[Dict]:
    """Up to n_pairs (direct, CoT) pairs with exactly one correct reply, balanced by which one is correct."""
    pools = {"direct_correct": [], "cot_correct": []}
    for r in rows:
        cots = [("cot_greedy", r["cot"])] + [(f"sc_sample_{k + 1}", s) for k, s in enumerate(r["sc"]["samples"])]
        for seed, block in r.get("sc_other_seeds", {}).items():
            cots += [(f"sc_seed{seed}_sample_{k + 1}", s) for k, s in enumerate(block["samples"])]
        for source, c in cots:
            if r["direct"]["correct"] == c["correct"]:
                continue
            kind = "direct_correct" if r["direct"]["correct"] else "cot_correct"
            pools[kind].append({"row": r, "source": source, "cot": c})

    def round_robin(cands: List[Dict]) -> List[Dict]:
        """Orders candidates so consecutive picks come from different problems."""
        by_problem: Dict[str, List[Dict]] = {}
        for c in cands:
            by_problem.setdefault(c["row"]["id"], []).append(c)
        out, depth = [], 0
        while any(len(v) > depth for v in by_problem.values()):
            out += [v[depth] for v in by_problem.values() if len(v) > depth]
            depth += 1
        return out

    ordered = {k: round_robin(v) for k, v in pools.items()}
    half = n_pairs // 2
    take_direct = min(half, len(ordered["direct_correct"]))
    take_cot = min(n_pairs - take_direct, len(ordered["cot_correct"]))
    take_direct = min(n_pairs - take_cot, len(ordered["direct_correct"]))
    chosen = ordered["direct_correct"][:take_direct] + ordered["cot_correct"][:take_cot]
    pairs = []
    for i, c in enumerate(sorted(chosen, key=lambda c: (c["row"]["id"], c["source"]))):
        r, cot = c["row"], c["cot"]
        d_len, c_len = len(r["direct"]["reply"]), len(cot["reply"])
        pairs.append({
            "pair_id": f"P{i + 1:02d}", "problem_id": r["id"], "problem_type": r["type"], "problem": r["problem"], "gold": r["gold"],
            "direct_reply": r["direct"]["reply"], "direct_extracted": r["direct"]["extracted"], "direct_correct": r["direct"]["correct"],
            "cot_source": c["source"], "cot_reply": cot["reply"], "cot_extracted": cot["extracted"], "cot_correct": cot["correct"],
            "checker_winner": "direct" if r["direct"]["correct"] else "cot",
            "direct_chars": d_len, "cot_chars": c_len, "longer": "cot" if c_len > d_len else "direct",
        })
    return pairs


@torch.no_grad()
def judge(model, tok, problem: str, a: str, b: str) -> Dict:
    """Forced choice between next tokens "A" and "B" after the chat prompt."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TEMPLATE.format(problem=problem, a=a, b=b)}]
    # The Llama 3.2 template writes "Today Date: <today>" into the system header; pin it to the
    # template's own default so the prompt does not depend on the day the script is run.
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="26 Jul 2024")
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    hidden = model.model(**enc).last_hidden_state[0, -1]            # final-norm hidden state of the last position
    ids = [tok.encode(k, add_special_tokens=False) for k in ("A", "B")]
    assert all(len(v) == 1 for v in ids)
    # The two letter logits in fp32 (bf16 logits tie on ~1 in 30 prompts, which would bias ties to one letter).
    head = model.lm_head.weight
    w = head[[ids[0][0], ids[1][0]]].float().to(hidden.device)
    two = torch.softmax(w @ hidden.float(), dim=0)
    top = tok.decode([int((hidden.to(head.device).to(head.dtype) @ head.T).argmax())])
    return {"letter": "A" if two[0] > two[1] else "B", "p_A": round(two[0].item(), 6), "p_B": round(two[1].item(), 6),
            "top_token": top, "prompt_tokens": enc["input_ids"].shape[1]}


def cohen_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main():
    p = argparse.ArgumentParser(description="C2: LLM-as-judge reliability")
    p.add_argument("--judge", default=JUDGE_ID)
    p.add_argument("--pairs", type=int, default=30)
    p.add_argument("--pairs-in", default=None, help="judge these pairs (JSONL) instead of rebuilding them from c1_outputs.jsonl")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="compute dtype (the default 4-bit checkpoint stores weights in NF4 and computes in this dtype)")
    p.add_argument("--device-map", default="cuda", help='"cuda" (one GPU) or "auto" (spread over all visible GPUs)')
    p.add_argument("--tag", default="", help="suffix for the output files, e.g. _fp32")
    args = p.parse_args()
    if args.pairs_in:
        keep = ("pair_id", "problem_id", "problem_type", "problem", "gold", "direct_reply", "direct_extracted", "direct_correct",
                "cot_source", "cot_reply", "cot_extracted", "cot_correct", "checker_winner", "direct_chars", "cot_chars", "longer")
        pairs = [{k: v for k, v in json.loads(l).items() if k in keep} for l in open(args.pairs_in, encoding="utf-8")]
    else:
        rows = [json.loads(l) for l in open(DATA_DIR / "c1_outputs.jsonl", encoding="utf-8")]
        pairs = build_pairs(rows, args.pairs)
    print(f"{len(pairs)} pairs: {Counter(x['checker_winner'] for x in pairs)} | problems {len({x['problem_id'] for x in pairs})}")

    tok = AutoTokenizer.from_pretrained(args.judge)
    model = AutoModelForCausalLM.from_pretrained(args.judge, device_map=args.device_map, dtype=getattr(torch, args.dtype)).eval()
    for x in pairs:
        v1 = judge(model, tok, x["problem"], x["direct_reply"], x["cot_reply"])       # direct shown first
        v2 = judge(model, tok, x["problem"], x["cot_reply"], x["direct_reply"])       # CoT shown first
        v1["prefers"] = "direct" if v1["letter"] == "A" else "cot"
        v2["prefers"] = "cot" if v2["letter"] == "A" else "direct"
        x["verdict_direct_first"], x["verdict_cot_first"] = v1, v2
        x["order_consistent"] = v1["prefers"] == v2["prefers"]
        x["judge_preference"] = v1["prefers"] if x["order_consistent"] else None
        print(x["pair_id"], x["problem_id"], "checker:", x["checker_winner"], "| direct-first ->", v1["prefers"], v1["p_A"],
              "| cot-first ->", v2["prefers"], v2["p_A"])
    write_jsonl(OUT_DATA_DIR / f"c2_pairs_with_verdicts{args.tag}.jsonl", pairs)

    verdicts = [(x, v) for x in pairs for v in (x["verdict_direct_first"], x["verdict_cot_first"])]
    consistent = [x for x in pairs if x["order_consistent"]]
    # Mitigation check: average the judge's probability for each reply over both orders, so a pure
    # position preference cancels; the order-averaged verdict exists for every pair.
    for x in pairs:
        p_direct = (x["verdict_direct_first"]["p_A"] + x["verdict_cot_first"]["p_B"]) / 2
        x["order_averaged_p_direct"] = round(p_direct, 6)
        x["order_averaged_preference"] = "direct" if p_direct > 0.5 else "cot"
    averaged = [x["order_averaged_preference"] for x in pairs]
    results = {
        "judge": args.judge, "dtype": args.dtype, "device_map": args.device_map, "pairs": len(pairs), "verdicts": len(verdicts),
        "pair_mix": dict(Counter(x["checker_winner"] for x in pairs)),
        "problems_covered": len({x["problem_id"] for x in pairs}),
        "pairs_per_problem": dict(Counter(x["problem_id"] for x in pairs)),
        "position_bias_rate": round(sum(not x["order_consistent"] for x in pairs) / len(pairs), 4),
        "order_consistent_pairs": len(consistent),
        "first_position_win_rate": round(sum(v["letter"] == "A" for _, v in verdicts) / len(verdicts), 4),
        "agreement_with_checker_consistent": round(sum(x["judge_preference"] == x["checker_winner"] for x in consistent) / len(consistent), 4) if consistent else None,
        "cohen_kappa_consistent": round(cohen_kappa([x["judge_preference"] for x in consistent], [x["checker_winner"] for x in consistent]), 4) if consistent else None,
        "agreement_with_checker_all_verdicts": round(sum(v["prefers"] == x["checker_winner"] for x, v in verdicts) / len(verdicts), 4),
        "longer_answer_win_rate": round(sum(v["prefers"] == x["longer"] for x, v in verdicts) / len(verdicts), 4),
        "cot_preferred_rate": round(sum(v["prefers"] == "cot" for _, v in verdicts) / len(verdicts), 4),
        "top_token_is_A_or_B": sum(v["top_token"].strip() in ("A", "B") for _, v in verdicts),
        "consistent_preferences": dict(Counter(x["judge_preference"] for x in consistent)),
        "order_averaged": {
            "agreement_with_checker": round(sum(a == x["checker_winner"] for a, x in zip(averaged, pairs)) / len(pairs), 4),
            "cohen_kappa": round(cohen_kappa(averaged, [x["checker_winner"] for x in pairs]), 4),
            "prefers_cot": sum(a == "cot" for a in averaged),
        },
        "by_checker_winner": {w: {"pairs": sum(x["checker_winner"] == w for x in pairs),
                                  "consistent": sum(x["checker_winner"] == w and x["order_consistent"] for x in pairs),
                                  "judge_correct_verdicts": sum(v["prefers"] == w for x, v in verdicts if x["checker_winner"] == w)}
                              for w in ("direct", "cot")},
    }
    write_json(RESULTS_DIR / f"c2_results{args.tag}.json", results)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
