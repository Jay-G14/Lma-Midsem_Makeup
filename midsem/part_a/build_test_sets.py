#!/usr/bin/env python3
"""Builds the three Phase 3 test sets for Part A and audits them for leakage.

The project shipped one held-out test set, hindi/data/reasoning/test.jsonl. Its people and objects
come from a test-only entity pool, so it is the UNSEEN-ENTITIES set and is used unchanged. The two
other sets are generated with the project generator (scripts/generate_reasoning_data.py), from the
same entity pools (re-created with the same seed) and the same task mix:

  seen_entities   training templates, names drawn from the TRAIN pool. Every text is new: it is
                  deduplicated against all train / val / test texts.
  unseen_wording  names from the TRAIN pool, but every sentence of the question (introduction,
                  premises, value statements, question) uses a frame that never occurs in training.
                  The reasoning the model has to write is the usual canonical chain.

So each of the three sets changes one factor relative to training: nothing (seen entities), the
names (unseen entities) or the wording (unseen wording).

Audit (written to midsem/results/test_sets_audit.json): sentence templates are obtained by masking
names as <E> and numbers as <N>; the share of each set's sentences whose template occurs in the
training questions should be 100% for the first two sets and 0% for unseen wording.
"""

import argparse
import importlib.util
import re
import random
from typing import Dict, List, Optional, Set

from midsem.part_a.common import DATA_DIR, GENERATOR_SCRIPT, REASONING_DIR, RESULTS_DIR, write_json, write_jsonl
from hindi.eval.eval_phase3 import load_samples


def load_generator():
    """Imports scripts/generate_reasoning_data.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("generate_reasoning_data", GENERATOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_generator()


class UnseenWordingHindi(gen.HindiPhrasing):
    """Held-out Hindi wording. Every sentence frame of the question is new; the reasoning format
    (symbolic transcription, अर्थात्, क्रम, अतः ...) and the answer words are inherited unchanged.

    Training frame                              -> held-out frame (example)
    "A, B और C तीन दोस्त हैं।"                  -> "A, B और C के बारे में कुछ जानकारी दी गई है।"
    "A की लंबाई B से अधिक है।"                  -> "A की लंबाई B की लंबाई से ज़्यादा है।"
    "B की लंबाई A से कम है।"                    -> "B की लंबाई A के मुकाबले कम है।"
    "B की तुलना में A की लंबाई अधिक है।"        -> "A के मुकाबले B की लंबाई कम है।"
    "A B से लंबा है।" / "B A से छोटा है।"       -> "A कद में B से लंबा है।", "A B से तेज़ दौड़ता है।"
    "A की लंबाई 150 सेमी है।"                   -> "A 150 सेमी लंबा है।", "A का भार 60 किग्रा है।"
    "इनमें से किसकी लंबाई सबसे अधिक है?"       -> "बताइए, सबसे ज़्यादा लंबाई किसकी है?"
    "क्या A की लंबाई B से अधिक है?"             -> "क्या यह सही है कि A की लंबाई B से ज़्यादा है?"
    "A और B में से किसकी लंबाई अधिक है?"        -> "A या B, किसकी लंबाई ज़्यादा है?"
    "क्या A और B की लंबाई बराबर है?"            -> "क्या A की लंबाई और B की लंबाई समान है?"
    "क्या A B से लंबा है?"                       -> "क्या A कद में B से लंबा है?"
    """

    QUESTION_MORE = {True: "ज़्यादा", False: "कम"}
    ADJ_FRAME = {"लंबाई": "कद में", "आयु": "उम्र में", "वजन": "वज़न में"}

    @staticmethod
    def _verb(gender: Optional[str]) -> str:
        return "दौड़ती" if gender == "f" else "दौड़ता"

    def _who(self, attr, objects: bool) -> str:
        return f"किस वस्तु {attr.poss}" if objects else "किस" + attr.poss

    def introduce(self, ents, objects):
        names = [e[0] for e in ents]
        return f"{', '.join(names[:-1])} और {names[-1]} के बारे में कुछ जानकारी दी गई है।"

    def _adj_clause(self, subj, obj, attr, more: bool) -> str:
        """"subj has more / less attr than obj" with a new adjective frame (gender-agreeing)."""
        name, gender = subj
        if attr.noun == "गति":
            return f"{name} {obj[0]} से {'तेज़' if more else 'धीरे'} {self._verb(gender)} है"
        word = attr.more_adj if more else attr.less_adj
        return f"{name} {self.ADJ_FRAME[attr.noun]} {obj[0]} से {self.adj(word, gender)} है"

    def relation_templates(self, hi, lo, attr):
        (a, _), (b, _) = hi, lo
        n, p = attr.noun, attr.poss
        out = [(f"{a} {p} {n} {b} {p} {n} से ज़्यादा है।", f"{a} > {b}"),
               (f"{b} {p} {n} {a} के मुकाबले कम है।", f"{b} < {a}"),
               (f"{a} के मुकाबले {b} {p} {n} कम है।", f"{a} > {b}")]
        if attr.more_adj:  # people attributes
            out += [(self._adj_clause(hi, lo, attr, True) + "।", f"{a} > {b}"),
                    (self._adj_clause(lo, hi, attr, False) + "।", f"{b} < {a}")]
        return out

    def value(self, e, attr, v):
        name, gender = e
        if attr.noun == "लंबाई":
            return f"{name} {v} {attr.unit} {self.adj('लंबा', gender)} है।"
        if attr.noun == "आयु":
            return f"{name} {v} {attr.unit} {'की' if gender == 'f' else 'का'} है।"
        if attr.noun == "गति":
            return f"{name} {v} {attr.unit} की रफ़्तार से {self._verb(gender)} है।"
        if attr.noun == "मूल्य":
            return f"{name} की कीमत {v} {attr.unit} है।"
        if attr.noun == "वजन":
            return f"{name} का भार {v} {attr.unit} है।"
        raise ValueError(attr.noun)

    def q_extreme(self, attr, most, objects):
        return f"बताइए, सबसे {self.QUESTION_MORE[most]} {attr.noun} {self._who(attr, objects)} है?"

    def q_pair(self, x, y, attr, more):
        return f"क्या यह सही है कि {x[0]} {attr.poss} {attr.noun} {y[0]} से {self.QUESTION_MORE[more]} है?"

    def q_which(self, x, y, attr, more, objects):
        return f"{x[0]} या {y[0]}, {self._who(attr, objects)} {attr.noun} {self.QUESTION_MORE[more]} है?"

    def q_equal(self, x, y, attr):
        return f"क्या {x[0]} {attr.poss} {attr.noun} और {y[0]} {attr.poss} {attr.noun} समान है?"

    def q_adj(self, x, y, attr, more):
        return "क्या " + self._adj_clause(x, y, attr, more) + "?"


# --- audit helpers ---

ALL_NAMES = set(gen.HINDI_ENTITIES["male"] + gen.HINDI_ENTITIES["female"] + gen.HINDI_ENTITIES["objects"])


def sentence_templates(question: str) -> List[str]:
    """Splits a question into sentences and masks entity names (<E>) and numbers (<N>)."""
    body = question.split("प्रश्न:", 1)[1]
    out = []
    for sentence in re.split(r"(?<=[।?])\s+", body.strip()):
        tokens = []
        for tok in sentence.split():
            core = tok.rstrip("।?,")
            tail = tok[len(core):]
            core = "<E>" if core in ALL_NAMES else "<N>" if core.isdigit() else core
            tokens.append(core + tail)
        if tokens:
            out.append(" ".join(tokens))
    return out


def names_in(question: str) -> Set[str]:
    return {tok.rstrip("।?,") for tok in question.split()} & ALL_NAMES


def audit(name: str, samples: List[Dict], train_templates: Set[str], pools, train_questions: Set[str]) -> Dict:
    """Template overlap with training, entity pools used, and question overlap with training."""
    sentences = [t for s in samples for t in sentence_templates(s["question"])]
    names = set().union(*(names_in(s["question"]) for s in samples))
    pool_names = {split: {e[0] for e in p["people"] + p["objects"]} for split, p in pools.items()}
    return {
        "set": name,
        "size": len(samples),
        "tasks": dict(sorted({t: sum(s["task"] == t for s in samples) for t in gen.TASKS}.items())),
        "distinct_names": len(names),
        "names_in_train_pool_pct": round(100.0 * len(names & pool_names["train"]) / len(names), 2),
        "names_in_test_pool_pct": round(100.0 * len(names & pool_names["test"]) / len(names), 2),
        "sentences": len(sentences),
        "distinct_sentence_templates": len(set(sentences)),
        "sentences_with_training_template_pct": round(100.0 * sum(t in train_templates for t in sentences) / len(sentences), 2),
        "questions_also_in_train": sum(s["question"] in train_questions for s in samples),
    }


def main():
    parser = argparse.ArgumentParser(description="Build the seen-entity and unseen-wording test sets")
    parser.add_argument("--num", type=int, default=1000, help="samples per new test set (Phase 3 test set has 1000)")
    parser.add_argument("--seed", type=int, default=42, help="the Phase 3 generator seed (re-creates the entity pools)")
    args = parser.parse_args()

    # 1. Re-create the Phase 3 entity pools: the generator draws them first from Random(f"{seed}-hindi").
    pools = gen.split_entity_pools(gen.HINDI_ENTITIES, random.Random(f"{args.seed}-hindi"))
    splits = {s: load_samples(str(REASONING_DIR / f"{s}.jsonl")) for s in ("train", "val", "test")}
    for split, samples in splits.items():
        used = set().union(*(names_in(s["question"]) for s in samples))
        pool = {e[0] for e in pools[split]["people"] + pools[split]["objects"]}
        assert used <= pool, f"{split}: names outside the re-created pool, e.g. {sorted(used - pool)[:5]}"
    print("Re-created entity pools match the Phase 3 splits.")

    # 2. Generate the two new sets; `seen` starts with every existing text so nothing repeats.
    seen = {s["text"] for samples in splits.values() for s in samples}
    seen_entities = gen.build_dataset_split(gen.PHRASINGS["hindi"], "seen_entities", args.num, pools["train"],
                                            random.Random(f"{args.seed}-hindi-seen_entities"), seen)
    unseen_wording = gen.build_dataset_split(UnseenWordingHindi(), "unseen_wording", args.num, pools["train"],
                                             random.Random(f"{args.seed}-hindi-unseen_wording"), seen)
    for row in seen_entities:
        row["test_set"] = "seen_entities"
    for row in unseen_wording:
        row["test_set"] = "unseen_wording"
    write_jsonl(DATA_DIR / "test_seen_entities.jsonl", seen_entities)
    write_jsonl(DATA_DIR / "test_unseen_wording.jsonl", unseen_wording)

    # 3. Audit all three test sets against the training questions.
    train_questions = {s["question"] for s in splits["train"]}
    train_templates = {t for s in splits["train"] for t in sentence_templates(s["question"])}
    report = {
        "train_distinct_sentence_templates": len(train_templates),
        "pools": {split: {k: len(v) for k, v in p.items()} for split, p in pools.items()},
        "sets": [audit("seen_entities", seen_entities, train_templates, pools, train_questions),
                 audit("unseen_entities", splits["test"], train_templates, pools, train_questions),
                 audit("unseen_wording", unseen_wording, train_templates, pools, train_questions)],
        "examples": {"seen_entities": seen_entities[0]["text"], "unseen_wording": unseen_wording[:3]},
    }
    write_json(RESULTS_DIR / "test_sets_audit.json", report)
    for row in report["sets"]:
        print(row)


if __name__ == "__main__":
    main()
