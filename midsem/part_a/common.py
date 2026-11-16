"""Shared helpers for Part A: paths, model loading, three-test-set evaluation and drift perplexity.

Everything reuses the Phase 3 code so the numbers are comparable with the project:
  - prompts, greedy generation and answer extraction come from hindi/eval/eval_phase3.py;
  - the finetuning dataset (question masking, </s>, 15% replay) comes from hindi/train/finetune.py;
  - drift perplexity uses the same 500 pretraining-validation documents as Phase 3.
"""

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tokenizers import Tokenizer

from hindi.eval.eval_phase3 import ANSWER_MARKER, extract_answer, generate_batch, load_model, load_samples, split_sample
from hindi.model.transformer import TransformerLM
from midsem.part_a.lora import inject_lora, load_adapter_state

MIDSEM = Path(__file__).resolve().parents[1]
PROJECT = MIDSEM / "project"                    # copies of the individual project's code, data and checkpoints
BASE_CKPT = PROJECT / "hindi/checkpoints/final_model.pt"            # Phase 2 pretrained Model H
FULL_FT_CKPT = PROJECT / "hindi/checkpoints/finetuned_model.pt"     # Phase 3 full finetune
TOKENIZER_PATH = PROJECT / "hindi/tokenizer/tokenizer.json"
REASONING_DIR = PROJECT / "hindi/data/reasoning"
PRETRAIN_TRAIN = PROJECT / "hindi/data/processed/train.jsonl"       # first 3,000 documents (replay reads 2,117)
PRETRAIN_VAL = PROJECT / "hindi/data/processed/val.jsonl"
GENERATOR_SCRIPT = PROJECT / "scripts/generate_reasoning_data.py"

# MIDSEM_STUDY=1 (set by ada/orchestrate.py for itself and every task it starts): the multi-seed study writes to the
# *_ada folders, so it never mixes with the single-seed pass in results/, checkpoints/, logs/ and data/.
STUDY = os.environ.get("MIDSEM_STUDY") == "1"
DATA_DIR = MIDSEM / "data"                                         # inputs (test sets, preference pairs, C1/C2 data)
OUT_DATA_DIR = MIDSEM / ("data_ada" if STUDY else "data")          # data written by a run (the fp32 C2 verdicts)
RESULTS_DIR = MIDSEM / ("results_ada" if STUDY else "results")
FIG_DIR = MIDSEM / "figures"
LOG_DIR = MIDSEM / ("logs_ada" if STUDY else "logs")
CKPT_DIR = MIDSEM / ("checkpoints_ada" if STUDY else "checkpoints")   # LoRA / DPO adapters, extra full-finetune seeds

TEST_SETS = {
    "seen_entities": DATA_DIR / "test_seen_entities.jsonl",
    "unseen_entities": REASONING_DIR / "test.jsonl",          # the Phase 3 test set (test-only names)
    "unseen_wording": DATA_DIR / "test_unseen_wording.jsonl",
}
TEST_SET_LABELS = {"seen_entities": "Seen entities", "unseen_entities": "Unseen entities", "unseen_wording": "Unseen wording"}
LORA_ALPHA = 16.0


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Seeds Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def write_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: List[Dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_base(dev: torch.device) -> TransformerLM:
    """The Phase 2 pretrained Model H in eval mode."""
    model, _ = load_model(str(BASE_CKPT), dev)
    return model


def load_lora_model(dev: torch.device, r: int, alpha: float = LORA_ALPHA, adapter_path: Optional[Path] = None) -> TransformerLM:
    """Pretrained Model H with LoRA on W_Q, W_K, W_V, W_O of every block (optionally loading trained A / B)."""
    model = load_base(dev)
    inject_lora(model, r, alpha)
    model.to(dev)
    if adapter_path is not None:
        state = torch.load(adapter_path, map_location="cpu", weights_only=False)
        load_adapter_state(model, state["adapter"])
    model.eval()
    return model


@torch.no_grad()
def pretrain_ppl(model: TransformerLM, tok: Tokenizer, dev: torch.device, max_docs: int = 500) -> float:
    """Pretraining-validation perplexity on the first 500 documents, computed exactly as Phase 3
    (finetune.evaluate_lm_ppl): each document truncated to 512 tokens, token-weighted mean loss.
    Only the first max_docs lines are read instead of the whole 272 MB file."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with open(PRETRAIN_VAL, "r", encoding="utf-8") as f:
        docs = []
        for line in f:
            if line.strip():
                docs.append(json.loads(line)["text"])
                if len(docs) >= max_docs:
                    break
    for text in docs:
        ids = tok.encode(text).ids[:512]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=dev)
        _, loss, _ = model(x, targets=y)
        total_loss += loss.item() * (len(ids) - 1)
        total_tokens += len(ids) - 1
    return float(np.exp(total_loss / max(total_tokens, 1)))


def evaluate_reasoning(model: TransformerLM, tok: Tokenizer, samples: List[Dict], dev: torch.device,
                       keep_generations: int = 0) -> Dict:
    """Phase 3 protocol: prompt = question + "\\nतर्क:", greedy decoding until </s> (max 128 tokens),
    answer = first word after the model's own "उत्तर:". Returns answer EM, chain EM and per-task accuracy."""
    split = [split_sample(s) for s in samples]
    generations = generate_batch(model, tok, [p for p, _, _ in split], dev, max_new_tokens=128)
    correct, total, chain = Counter(), Counter(), Counter()
    answered, kept = 0, []
    for s, (prompt, gold_cont, gold_ans), gen in zip(samples, split, generations):
        pred = extract_answer(gen)
        ok = pred == gold_ans
        correct[s["task"]] += ok
        total[s["task"]] += 1
        chain[s["task"]] += " ".join(gen.split()) == " ".join(gold_cont.split())
        answered += ANSWER_MARKER in gen
        if len(kept) < keep_generations:
            kept.append({"question": s["question"], "gold": gold_ans, "prediction": pred, "generation": gen, "correct": ok})
    n = len(samples)
    out = {
        "n": n,
        "answer_em": round(100.0 * sum(correct.values()) / n, 2),
        "chain_em": round(100.0 * sum(chain.values()) / n, 2),
        "answered": round(100.0 * answered / n, 2),
        "task_answer_em": {t: round(100.0 * correct[t] / total[t], 2) for t in sorted(total)},
    }
    if kept:
        out["examples"] = kept
    return out


def evaluate_all_sets(model: TransformerLM, tok: Tokenizer, dev: torch.device, include_val: bool = False,
                      keep_generations: int = 0, only: Optional[List[str]] = None) -> Dict:
    """Answer / chain EM on the three test sets (and optionally the validation split); `only` keeps a subset."""
    sets = dict(TEST_SETS)
    if include_val:
        sets = {"val": REASONING_DIR / "val.jsonl", **sets}
    if only:
        sets = {k: v for k, v in sets.items() if k in only}
    return {name: evaluate_reasoning(model, tok, load_samples(str(path)), dev, keep_generations) for name, path in sets.items()}
