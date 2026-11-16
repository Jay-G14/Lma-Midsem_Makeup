#!/usr/bin/env python3
"""A3: builds DPO preference pairs from the Phase 3 TRAINING split (Hindi, the model's language).

Every pair shares the prompt the model sees at test time, "प्रश्न: ...\\nतर्क:", and has two complete
replies (reasoning + answer):

  chosen    the gold reply from the training sample, i.e. the correct answer with its correct chain
  rejected  a plausible wrong reply: it copies the gold chain up to the decisive step, gets that one
            step wrong, and then follows the error consistently to a wrong final answer that is one
            of the question's own candidates (the other entity in the comparison, or the other
            yes/no word)

Decisive step per question type (what the rejected reply gets wrong):
  transitive, largest / smallest  builds the order with the top (bottom) two entities swapped and
                                  names the runner-up              -> क्रम: B > A > C ... उत्तर: B
  transitive, yes/no              reads the order backwards for the asked pair  -> अतः X < Y  (flipped)
  numeric_extremum                picks the runner-up value                      -> मान 40 = B  (not 58 = A)
  numeric_pair, which / yes-no    compares the two numbers the wrong way round   -> 55 < 27 ... (flipped)
  verbal_comparison               derives the asked relation the wrong way round -> अतः X < Y  (flipped)
Questions with equal values or an equality claim are skipped: "=" has no plausible single-step flip.

Leakage control (audited into midsem/results/preference_pairs_audit.json):
  - only training-split questions are used, so every name comes from the train entity pool, which is
    disjoint from the unseen-entity test pool;
  - every prompt sentence uses a training template (so none of the held-out unseen-wording frames);
  - no pair prompt equals a question of any of the three test sets.
A separate set of validation pairs is built the same way from the validation split (validation names)
to monitor held-out preference accuracy during DPO; it is never trained on.
"""

import argparse
import random
import re
from collections import Counter
from typing import Dict, List, Optional

import midsem  # noqa: F401  (puts midsem/project/, the copied project code, first on sys.path)
from hindi.eval.eval_phase3 import REASONING_CUE, load_samples
from midsem.part_a.build_test_sets import gen, names_in, sentence_templates
from midsem.part_a.common import DATA_DIR, REASONING_DIR, RESULTS_DIR, TEST_SETS, write_json, write_jsonl

YES, NO = "हाँ", "नहीं"
FLIP = {">": "<", "<": ">"}

TRANSITIVE = re.compile(r"^(?P<pre>.* क्रम: )(?P<chain>.+?) है। (?P<concl>अतः .+)$")
T_EXTREME = re.compile(r"^अतः सबसे (?P<dir>अधिक|कम) (?P<noun>\S+) (?P<ent>\S+) (?P<poss>का|की) है।$")
T_PAIR = re.compile(r"^अतः (?P<x>\S+) (?P<op>[<>]) (?P<y>\S+) है। (?P<claim>प्रश्न का दावा (?P=x) (?P<c>[<>]) (?P=y) है।)$")
EXTREMUM = re.compile(r"^(?P<listing>.+) है। सबसे (?P<dir>अधिक|कम) मान (?P<v>\d+) = (?P<e>\S+) है।$")
PAIR = re.compile(r"^(?P<va>\d+) = (?P<a>\S+) और (?P<vb>\d+) = (?P<b>\S+) है। (?P=va) (?P<s>[<>=]) (?P=vb) है, अतः (?P<rest>.+)$")
PAIR_WHICH = re.compile(r"^(?P<l>\S+) (?P<op>[>=]) (?P<r>\S+) है।$")
PAIR_YN = re.compile(r"^(?P<x>\S+) (?P<s2>[<>=]) (?P<y>\S+) है। (?P<claim>प्रश्न का दावा (?P=x) (?P<c>[<>=]) (?P=y) है।)$")
VERBAL = re.compile(r"^(?P<pre>कथन के अनुसार \S+ [<>] \S+ है, अतः )(?P<x>\S+) (?P<op>[<>]) (?P<y>\S+) है। "
                    r"(?P<claim>प्रश्न का दावा (?P=x) (?P<c>[<>]) (?P=y) है।)$")


def yes_no(op: str, claimed: str) -> str:
    return YES if op == claimed else NO


def corrupt(sample: Dict) -> Optional[Dict]:
    """Returns {"body", "answer", "qtype", "error"} for the rejected reply, or None if not eligible."""
    body = sample["reasoning"][len("तर्क: "):]
    task, gold = sample["task"], sample["answer"]

    if task == "transitive":
        m = TRANSITIVE.match(body)
        chain = m["chain"].split(" > ")
        e = T_EXTREME.match(m["concl"])
        if e:
            most = e["dir"] == "अधिक"
            assert gold == (chain[0] if most else chain[-1])
            wrong = chain[:]
            i, j = (0, 1) if most else (-1, -2)
            wrong[i], wrong[j] = wrong[j], wrong[i]
            answer = wrong[0] if most else wrong[-1]
            new = f"{m['pre']}{' > '.join(wrong)} है। अतः सबसे {e['dir']} {e['noun']} {answer} {e['poss']} है।"
            return {"body": new, "answer": answer, "qtype": "transitive_" + ("max" if most else "min"),
                    "error": "order built with the two extreme-end entities swapped"}
        p = T_PAIR.match(m["concl"])
        assert gold == yes_no(p["op"], p["c"])
        op = FLIP[p["op"]]
        new = f"{m['pre']}{m['chain']} है। अतः {p['x']} {op} {p['y']} है। {p['claim']}"
        return {"body": new, "answer": yes_no(op, p["c"]), "qtype": "transitive_yesno",
                "error": "relation of the asked pair read backwards from the order"}

    if task == "numeric_extremum":
        m = EXTREMUM.match(body)
        values = [(int(v), name) for v, name in re.findall(r"(\d+) = (\S+)", m["listing"])]
        most = m["dir"] == "अधिक"
        ranked = sorted(values, reverse=most)
        assert ranked[0] == (int(m["v"]), m["e"]) and gold == m["e"]
        v2, e2 = ranked[1]
        new = f"{m['listing']} है। सबसे {m['dir']} मान {v2} = {e2} है।"
        return {"body": new, "answer": e2, "qtype": "numeric_extremum_" + ("max" if most else "min"),
                "error": "runner-up value picked as the extreme"}

    if task == "numeric_pair":
        m = PAIR.match(body)
        if m["s"] == "=":
            return None                                    # equal values: no plausible flip
        head = f"{m['va']} = {m['a']} और {m['vb']} = {m['b']} है। {m['va']} {FLIP[m['s']]} {m['vb']} है, अतः"
        w = PAIR_WHICH.match(m["rest"])
        if w:
            big, small = w["l"], w["r"]
            assert w["op"] == ">" and gold in (big, small)
            answer = small if gold == big else big
            return {"body": f"{head} {small} > {big} है।", "answer": answer, "qtype": "numeric_pair_which",
                    "error": "the two numbers compared the wrong way round"}
        y = PAIR_YN.match(m["rest"])
        if y["c"] == "=":
            return None                                    # equality claim: skipped
        assert gold == yes_no(y["s2"], y["c"])
        s2 = FLIP[y["s2"]]
        return {"body": f"{head} {y['x']} {s2} {y['y']} है। {y['claim']}", "answer": yes_no(s2, y["c"]),
                "qtype": "numeric_pair_yesno", "error": "the two numbers compared the wrong way round"}

    if task == "verbal_comparison":
        m = VERBAL.match(body)
        assert gold == yes_no(m["op"], m["c"])
        op = FLIP[m["op"]]
        return {"body": f"{m['pre']}{m['x']} {op} {m['y']} है। {m['claim']}", "answer": yes_no(op, m["c"]),
                "qtype": "verbal_yesno", "error": "asked relation derived the wrong way round"}
    raise ValueError(task)


def make_pair(sample: Dict, index: int, split: str) -> Optional[Dict]:
    """Chosen = the gold continuation of the training text; rejected = corrupt(sample)."""
    bad = corrupt(sample)
    if bad is None:
        return None
    prompt = sample["question"] + REASONING_CUE
    chosen = sample["text"][len(prompt):]
    rejected = f" {bad['body']}\nउत्तर: {bad['answer']}"
    assert sample["text"].startswith(prompt) and chosen.startswith(" ")
    assert bad["answer"] != sample["answer"] and bad["answer"] in sample["candidates"]
    assert rejected != chosen
    return {"id": f"{split}-{index}", "split": split, "source_index": index, "task": sample["task"],
            "question_type": bad["qtype"], "prompt": prompt, "chosen": chosen, "rejected": rejected,
            "chosen_answer": sample["answer"], "rejected_answer": bad["answer"], "rejected_error": bad["error"]}


def stratified(pairs: List[Dict], n: int, rng: random.Random) -> List[Dict]:
    """Samples n pairs with the Phase 3 task mix (40 / 20 / 20 / 20)."""
    by_task = {t: [p for p in pairs if p["task"] == t] for t in gen.TASKS}
    out = []
    for task, weight in zip(gen.TASKS, gen.TASK_WEIGHTS):
        out += rng.sample(by_task[task], round(n * weight))
    rng.shuffle(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build DPO preference pairs from the Phase 3 training split")
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    built = {}
    for split in ("train", "val"):
        samples = load_samples(str(REASONING_DIR / f"{split}.jsonl"))
        eligible = [p for i, s in enumerate(samples) if (p := make_pair(s, i, split)) is not None]
        built[split] = (samples, eligible)
        print(f"{split}: {len(eligible)} of {len(samples)} samples eligible")
    train_pairs = stratified(built["train"][1], args.num_train, rng)
    val_pairs = stratified(built["val"][1], args.num_val, rng)
    write_jsonl(DATA_DIR / "part_a_preference_pairs.jsonl", train_pairs)
    write_jsonl(DATA_DIR / "part_a_preference_pairs_val.jsonl", val_pairs)

    # Leakage audit
    pools = gen.split_entity_pools(gen.HINDI_ENTITIES, random.Random("42-hindi"))
    pool = {s: {e[0] for e in p["people"] + p["objects"]} for s, p in pools.items()}
    train_templates = {t for s in built["train"][0] for t in sentence_templates(s["question"])}
    test_questions = {s["question"] for path in TEST_SETS.values() for s in load_samples(str(path))}
    pair_names = set().union(*(names_in(p["prompt"]) for p in train_pairs))
    pair_sentences = [t for p in train_pairs for t in sentence_templates(p["prompt"][:-len(REASONING_CUE)])]
    audit = {
        "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
        "eligible_train_samples": len(built["train"][1]),
        "task_mix": dict(Counter(p["task"] for p in train_pairs)),
        "question_types": dict(Counter(p["question_type"] for p in train_pairs)),
        "chosen_answer_types": dict(Counter("yes/no" if p["chosen_answer"] in (YES, NO) else "entity" for p in train_pairs)),
        "names_in_train_pool_pct": round(100.0 * len(pair_names & pool["train"]) / len(pair_names), 2),
        "names_in_test_pool": len(pair_names & pool["test"]),
        "names_in_val_pool": len(pair_names & pool["val"]),
        "prompt_sentences_with_training_template_pct": round(100.0 * sum(t in train_templates for t in pair_sentences) / len(pair_sentences), 2),
        "prompts_equal_to_a_test_question": sum(p["prompt"][:-len(REASONING_CUE)] in test_questions for p in train_pairs),
        "example": train_pairs[0],
    }
    write_json(RESULTS_DIR / "preference_pairs_audit.json", audit)
    print({k: v for k, v in audit.items() if k != "example"})
    token_stats()


def token_stats():
    """How many reply tokens chosen and rejected share before the first difference (they cancel in the DPO margin)."""
    from midsem.part_a.common import tokenizer
    from midsem.part_a.dpo import encode_pairs
    encoded = encode_pairs(DATA_DIR / "part_a_preference_pairs.jsonl", tokenizer())
    shared = [next(i for i, (a, b) in enumerate(zip(c, r)) if a != b) for _, c, r in encoded]
    n = len(encoded)
    stats = {"pairs": n, "mean_reply_tokens_chosen": sum(len(c) for _, c, _ in encoded) / n,
             "mean_reply_tokens_rejected": sum(len(r) for _, _, r in encoded) / n,
             "mean_shared_prefix_tokens": sum(shared) / n,
             "mean_shared_fraction": sum(k / len(c) for k, (_, c, _) in zip(shared, encoded)) / n,
             "mean_tokens_from_first_difference_chosen": sum(len(c) - k for k, (_, c, _) in zip(shared, encoded)) / n}
    write_json(RESULTS_DIR / "preference_pairs_token_stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
