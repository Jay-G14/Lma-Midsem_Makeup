#!/usr/bin/env python3
"""C1: Does thinking longer help? Qwen2.5-1.5B-Instruct on the 20 Hindi problems, three conditions.

  direct   greedy decoding, the prompt asks for the final number only          (max 48 new tokens)
  cot      zero-shot chain of thought: "आइए चरण-दर-चरण सोचें।", greedy          (max 1024 new tokens)
  sc       self-consistency: the CoT prompt, 5 samples at T = 0.7, top-p 0.95,
           majority vote over the checker-normalised answers                   (max 1024 new tokens)

Prompts and problems are in Hindi (Model H's language); nothing is translated to English.
Every decoding setting is passed explicitly in a fresh GenerationConfig, so the model's default
sampling config (top-k 20, repetition penalty 1.1) is not silently applied; greedy is plain argmax.
Generated tokens are counted up to, not including, the end-of-turn token. Outputs (every reply, its
extracted answer, correctness and token count) go to midsem/data/c1_outputs.jsonl.
"""

import argparse
import json
import re
import time
from collections import Counter
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from midsem.part_a.common import DATA_DIR, RESULTS_DIR, write_json, write_jsonl
from midsem.part_c.checker import extract_answer, parse_number, same_answer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DIRECT_SUFFIX = "\n\nकेवल अंतिम उत्तर एक संख्या के रूप में लिखिए। कोई व्याख्या या इकाई मत लिखिए।"
COT_SUFFIX = "\n\nआइए चरण-दर-चरण सोचें। अपना हल लिखिए और अंत में 'उत्तर: <संख्या>' लिखिए।"
SC_SAMPLES = 5


def load(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    return tok, model


def chat(tok, text: str) -> str:
    """Qwen chat template with its default system prompt, ready for the assistant turn."""
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def gen_config(tok, sample: bool, max_new_tokens: int) -> GenerationConfig:
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    if sample:
        return GenerationConfig(do_sample=True, temperature=0.7, top_p=0.95, top_k=0, repetition_penalty=1.0,
                                max_new_tokens=max_new_tokens, eos_token_id=eos, pad_token_id=tok.pad_token_id)
    return GenerationConfig(do_sample=False, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                            eos_token_id=eos, pad_token_id=tok.pad_token_id)


@torch.no_grad()
def generate(model, tok, prompts: List[str], config: GenerationConfig, n: int = 1) -> List[List[Tuple[str, int, bool]]]:
    """For each prompt, n (reply, generated tokens excluding end-of-turn, truncated?) triples."""
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**enc, generation_config=config, num_return_sequences=n)
    gen = out[:, enc["input_ids"].shape[1]:].tolist()
    eos = set(config.eos_token_id)
    results = []
    for row in gen:
        stop = next((i for i, t in enumerate(row) if t in eos), None)
        ids = row if stop is None else row[:stop]
        results.append((tok.decode(ids, skip_special_tokens=True).strip(), len(ids), stop is None))
    return [results[i * n:(i + 1) * n] for i in range(len(prompts))]


def majority(answers: List[Optional[Fraction]]) -> Optional[Fraction]:
    """Most frequent parsed answer (ties: the earliest sample); unparseable replies do not vote."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    top = max(counts.values())
    return next(a for a in valid if counts[a] == top)


def devanagari_share(text: str) -> float:
    """Share of letters in the reply that are Devanagari (how much of the reasoning is in Hindi)."""
    letters = [c for c in text if c.isalpha()]
    return sum("ऀ" <= c <= "ॿ" for c in letters) / len(letters) if letters else 0.0


def fmt(x: Optional[Fraction]) -> Optional[str]:
    if x is None:
        return None
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator} (= {float(x):g})"


def summarise(rows: List[Dict]) -> Dict:
    out = {}
    for cond in ("direct", "cot", "sc"):
        res = {}
        for kind in ("standard", "modified"):
            sub = [r for r in rows if r["type"] == kind]
            res[f"{kind}_correct"] = sum(r[cond]["correct"] for r in sub)
            res[f"{kind}_n"] = len(sub)
            res[f"{kind}_accuracy"] = round(100.0 * res[f"{kind}_correct"] / len(sub), 1)
        res["overall_accuracy"] = round(100.0 * sum(r[cond]["correct"] for r in rows) / len(rows), 1)
        res["mean_generated_tokens"] = round(sum(r[cond]["tokens"] for r in rows) / len(rows), 1)
        if cond == "sc":
            res["mean_generated_tokens_per_sample"] = round(res["mean_generated_tokens"] / SC_SAMPLES, 1)
        mods = [r for r in rows if r["type"] == "modified"]
        res["modified_template_answer_rate"] = round(100.0 * sum(r[cond]["gave_template_answer"] for r in mods) / len(mods), 1)
        res["truncated_replies"] = sum(r[cond]["truncated"] for r in rows)
        out[cond] = res
    out["cot_devanagari_share"] = round(sum(devanagari_share(r["cot"]["reply"]) for r in rows) / len(rows), 3)
    return out


def sc_block(samples_raw, gold, template, entry):
    """Self-consistency result for one problem and one seed: the 5 samples, their votes and the majority."""
    samples = [entry(*x) for x in samples_raw]
    vote = majority([extract_answer(x["reply"]) for x in samples])
    return {"samples": samples, "votes": [x["extracted"] for x in samples], "majority": fmt(vote),
            "correct": same_answer(vote, gold), "tokens": sum(x["tokens"] for x in samples),
            "truncated": sum(x["truncated"] for x in samples),
            "gave_template_answer": template is not None and same_answer(vote, template)}


def main():
    p = argparse.ArgumentParser(description="C1: direct vs zero-shot CoT vs self-consistency")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--sc-seeds", default="1234,2234,3234",
                   help="one self-consistency run (5 samples) per seed; the first is the primary run in the table")
    p.add_argument("--out-suffix", default="", help="write c1_outputs<suffix>.jsonl / c1_results<suffix>.json "
                                                    "(e.g. _rerun, to check that a second run reproduces the first)")
    args = p.parse_args()
    seeds = [int(x) for x in args.sc_seeds.split(",")]
    problems = [json.loads(l) for l in open(DATA_DIR / "c1_problems.jsonl", encoding="utf-8")]
    tok, model = load(args.model)
    t0 = time.time()

    # Every prompt is decoded on its own (no padding), so a problem's replies never depend on which other problems
    # share its batch: editing one problem cannot change the replies to the others.
    direct = [generate(model, tok, [chat(tok, q["problem"] + DIRECT_SUFFIX)], gen_config(tok, False, 48))[0][0] for q in problems]
    print(f"  direct done ({time.time() - t0:.0f}s)", flush=True)
    cot = [generate(model, tok, [chat(tok, q["problem"] + COT_SUFFIX)], gen_config(tok, False, 1024))[0][0] for q in problems]
    print(f"  greedy CoT done ({time.time() - t0:.0f}s)", flush=True)
    sc = {seed: [] for seed in seeds}
    for seed in seeds:
        for k, q in enumerate(problems):
            torch.manual_seed(seed + k)
            sc[seed].append(generate(model, tok, [chat(tok, q["problem"] + COT_SUFFIX)], gen_config(tok, True, 1024), n=SC_SAMPLES)[0])
        print(f"  SC seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for k, (q, d, c) in enumerate(zip(problems, direct, cot)):
        gold = parse_number(q["answer"])
        template = parse_number(q["template_answer"]) if q["type"] == "modified" else None

        def entry(reply, tokens, truncated):
            ans = extract_answer(reply)
            return {"reply": reply, "extracted": fmt(ans), "correct": same_answer(ans, gold), "tokens": tokens, "truncated": truncated,
                    "gave_template_answer": template is not None and same_answer(ans, template)}

        rows.append({
            "id": q["id"], "type": q["type"], "problem": q["problem"], "gold": q["answer"],
            "direct": entry(*d), "cot": entry(*c),
            "sc": sc_block(sc[seeds[0]][k], gold, template, entry),
            "sc_other_seeds": {str(seed): sc_block(sc[seed][k], gold, template, entry) for seed in seeds[1:]},
        })
    write_jsonl(DATA_DIR / f"c1_outputs{args.out_suffix}.jsonl", rows)

    results = summarise(rows)
    per_seed = {}
    for seed in seeds:
        key = "sc" if seed == seeds[0] else None
        pick = (lambda r: r["sc"]) if key else (lambda r, s=str(seed): r["sc_other_seeds"][s])
        std = [r for r in rows if r["type"] == "standard"]
        mod = [r for r in rows if r["type"] == "modified"]
        per_seed[str(seed)] = {"standard_accuracy": round(100.0 * sum(pick(r)["correct"] for r in std) / len(std), 1),
                               "modified_accuracy": round(100.0 * sum(pick(r)["correct"] for r in mod) / len(mod), 1),
                               "overall_accuracy": round(100.0 * sum(pick(r)["correct"] for r in rows) / len(rows), 1),
                               "mean_generated_tokens": round(sum(pick(r)["tokens"] for r in rows) / len(rows), 1)}
    vals = lambda k: [v[k] for v in per_seed.values()]
    mean = lambda xs: sum(xs) / len(xs)
    sd = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    results["sc_seeds"] = {"per_seed": per_seed,
                           **{f"{k}_mean": round(mean(vals(k)), 1) for k in ("standard_accuracy", "modified_accuracy", "overall_accuracy", "mean_generated_tokens")},
                           **{f"{k}_sd": round(sd(vals(k)), 1) for k in ("standard_accuracy", "modified_accuracy", "overall_accuracy")}}
    summary = {"model": args.model, "seconds": round(time.time() - t0, 1),
               "settings": {"batching": "one prompt per generate call (no padding)", "direct": "greedy, max 48 new tokens", "cot": "greedy, max 1024 new tokens",
                            "sc": f"{SC_SAMPLES} samples, T=0.7, top-p 0.95, top-k off, majority vote, max 1024 new tokens",
                            "sc_seeds": seeds, "repetition_penalty": 1.0, "dtype": "bfloat16",
                            "direct_suffix": DIRECT_SUFFIX, "cot_suffix": COT_SUFFIX},
               "results": results}
    write_json(RESULTS_DIR / f"c1_results{args.out_suffix}.json", summary)
    print(json.dumps(summary["results"], indent=1, ensure_ascii=False))
    for r in rows:
        print(r["id"], r["gold"], "| direct", r["direct"]["extracted"], "| cot", r["cot"]["extracted"], "| sc", r["sc"]["votes"], "->", r["sc"]["majority"],
              "| other seeds", {s: b["majority"] for s, b in r["sc_other_seeds"].items()})


if __name__ == "__main__":
    main()
