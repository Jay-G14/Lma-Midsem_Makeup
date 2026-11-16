"""Comprehensive Phase 3 Evaluation Suite for Hindi Transformer LM

Evaluates:
  1. Reasoning Exact-Match Accuracy (Overall & Task-Specific). The model is given only the
     question ("प्रश्न: ...") followed by "तर्क:", writes its own reasoning and "उत्तर: <answer>",
     and the first word after its own "उत्तर:" must equal the gold answer exactly. Chain exact match
     additionally requires the whole generated reasoning to equal the gold reasoning.
  2. Baselines: chance level (1 / number of candidate answers) and a zero-shot likelihood baseline
     for the pretrained model (rank the candidate answers by log-probability after "उत्तर:"),
     plus the answer-only control model finetuned without reasoning chains.
  3. Reasoning-Chain Overlap: BLEU-1, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L of the generated
     reasoning + answer against the gold reasoning + answer.
  4. Pretraining Language Model Sanity: Validation PPL on original corpus.
  5. Generation Diversity & Repetition: Distinct-1, Distinct-2, Repetition Rate.
  6. Attention Analysis with the Phase 2 toolkit (hindi/eval/evaluate.py): per-layer / per-head
     entropy and mean attention distance on reasoning prompts, Phase 2 early/late-layer heatmaps for
     both checkpoints, and pretrained vs. finetuned comparison figures.
  7. Error Analysis: per-step accuracy of the reasoning chain for each task, and how often a wrong
     entity answer names someone who does not appear in the question at all.

Includes tqdm progress bars and clean emoji-free logging.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer
from tqdm import tqdm

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM
from hindi.eval import evaluate as phase2_eval

REASONING_CUE = "\nतर्क:"
ANSWER_CUE = "\nउत्तर:"
ANSWER_MARKER = "उत्तर:"
PUNCTUATION = "।!?,.:;\"'()[]{}"
NON_ENTITY_ANSWERS = {"हाँ", "नहीं", "हो", "होइन", "बराबर"}
LOCAL_HEAD_MAX_DIST = 2.0       # heads whose mean attention distance is <= 2 tokens count as local
LONG_RANGE_HEAD_MIN_DIST = 10.0  # heads whose mean attention distance is >= 10 tokens count as long-range


def load_model(model_path: str, device: torch.device) -> Tuple[TransformerLM, TransformerConfig]:
    """Loads model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        config = TransformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
    else:
        raise ValueError(f"Config missing in checkpoint: {model_path}")

    model = TransformerLM(config).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, config


def load_samples(jsonl_path: str) -> List[Dict]:
    """Reads the reasoning test set."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_sample(sample: Dict, answer_only: bool = False) -> Tuple[str, str, str]:
    """Returns (prompt, gold continuation, gold answer) for one test sample.

    The prompt is only the question plus the reasoning cue; the gold continuation is the
    reasoning and answer the model is expected to write itself. For the answer-only control model
    the prompt ends with the answer cue and the continuation is just the answer.
    """
    if answer_only:
        return sample["question"] + ANSWER_CUE, sample["answer"], sample["answer"]
    prompt = sample["question"] + REASONING_CUE
    gold_continuation = sample["text"][len(prompt):].strip()
    return prompt, gold_continuation, sample["answer"]


def extract_answer(generation: str, answer_only: bool = False) -> str:
    """Final answer = first word after the model's own "उत्तर:", punctuation stripped ("" if it never answers).

    For the answer-only control model the generation starts right after "उत्तर:", so its first word is used.
    """
    if not answer_only:
        if ANSWER_MARKER not in generation:
            return ""
        generation = generation.split(ANSWER_MARKER, 1)[1]
    words = generation.strip().split("\n")[0].split()
    return words[0].strip(PUNCTUATION) if words else ""


def reasoning_clauses(text: str) -> List[str]:
    """Splits a reasoning chain (or its generated counterpart) into clauses on danda, comma and newline."""
    return [c.strip() for c in re.split(r"[।,\n]", text) if c.strip()]


def generate_batch(model: TransformerLM, tokenizer: Tokenizer, prompts: List[str], device: torch.device,
                   max_new_tokens: int = 128, batch_size: int = 16) -> List[str]:
    """Greedy-decodes every prompt until </s>.

    Prompts with the same token length are batched together, so no padding (and no attention
    mask) is needed; each output is cut at its first </s>.
    """
    eos_id = tokenizer.token_to_id("</s>")
    encoded = [tokenizer.encode(p).ids for p in prompts]
    by_length = defaultdict(list)
    for i, ids in enumerate(encoded):
        by_length[len(ids)].append(i)

    outputs = [""] * len(prompts)
    batches = [(L, idx[s:s + batch_size]) for L, idx in by_length.items() for s in range(0, len(idx), batch_size)]
    for length, chunk in tqdm(batches, desc="Generating", leave=False):
        x = torch.tensor([encoded[i] for i in chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_id=eos_id)
        for row, i in zip(out.tolist(), chunk):
            gen = row[length:]
            if eos_id in gen:
                gen = gen[:gen.index(eos_id)]
            outputs[i] = tokenizer.decode(gen).strip()
    return outputs


# --- Baselines ---

def candidate_log_likelihoods(model: TransformerLM, tokenizer: Tokenizer, prompt: str, candidates: List[str],
                              device: torch.device) -> List[float]:
    """Sum of token log-probabilities of " <candidate>" after the prompt, for every candidate answer.

    All candidates are scored in one right-padded batch; causal attention means padding after a
    candidate never affects its own scores.
    """
    prompt_len = len(tokenizer.encode(prompt).ids)
    seqs = [tokenizer.encode(f"{prompt} {c}").ids for c in candidates]
    max_len = max(len(s) for s in seqs)
    x = torch.full((len(seqs), max_len), model.config.pad_token_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, device=device)
    with torch.no_grad():
        logits, _, _ = model(x)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    scores = []
    for i, s in enumerate(seqs):
        positions = range(prompt_len, len(s))
        scores.append(sum(log_probs[i, p - 1, s[p]].item() for p in positions))
    return scores


def zero_shot_candidate_accuracy(model: TransformerLM, tokenizer: Tokenizer, samples: List[Dict], device: torch.device) -> Dict[str, Any]:
    """Zero-shot baseline: pick the candidate answer with the highest likelihood after "प्रश्न: ...\\nउत्तर:"."""
    correct, total = Counter(), Counter()
    for s in tqdm(samples, desc="Zero-shot candidate ranking", leave=False):
        scores = candidate_log_likelihoods(model, tokenizer, s["question"] + ANSWER_CUE, s["candidates"], device)
        pred = s["candidates"][int(np.argmax(scores))]
        correct[s["task"]] += pred == s["answer"]
        total[s["task"]] += 1
    return {
        "accuracy": round(100.0 * sum(correct.values()) / sum(total.values()), 2),
        "task_accuracy": {t: round(100.0 * correct[t] / total[t], 2) for t in sorted(total)},
    }


def chance_accuracy(samples: List[Dict]) -> Dict[str, Any]:
    """Expected accuracy of guessing uniformly among each question's candidate answers."""
    by_task = defaultdict(list)
    for s in samples:
        by_task[s["task"]].append(1.0 / len(s["candidates"]))
    all_vals = [v for vals in by_task.values() for v in vals]
    return {
        "accuracy": round(100.0 * float(np.mean(all_vals)), 2),
        "task_accuracy": {t: round(100.0 * float(np.mean(v)), 2) for t, v in sorted(by_task.items())},
    }


# --- Metric Calculation Helpers ---

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Returns n-gram frequencies."""
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def compute_bleu(ref_tokens: List[str], hyp_tokens: List[str], max_n: int = 4) -> Dict[str, float]:
    """Computes BLEU-1 and BLEU-4 scores."""
    if not hyp_tokens:
        return {"bleu_1": 0.0, "bleu_4": 0.0}

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        clipped_count = sum((hyp_ngrams & ref_ngrams).values())
        total_count = max(sum(hyp_ngrams.values()), 1)
        precisions.append(clipped_count / total_count)

    bp = math.exp(min(0, 1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    bleu_1 = bp * precisions[0]

    p_prod = math.prod(max(p, 1e-9) for p in precisions)
    bleu_4 = bp * (p_prod ** (1.0 / max_n))
    return {"bleu_1": round(bleu_1 * 100, 2), "bleu_4": round(bleu_4 * 100, 2)}


def lcs_length(x: List[str], y: List[str]) -> int:
    """Computes Length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_rouge(ref_tokens: List[str], hyp_tokens: List[str]) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    if not hyp_tokens or not ref_tokens:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    def rouge_n(n: int) -> float:
        r_ngrams = get_ngrams(ref_tokens, n)
        h_ngrams = get_ngrams(hyp_tokens, n)
        overlap = sum((r_ngrams & h_ngrams).values())
        rec = overlap / max(sum(r_ngrams.values()), 1)
        prec = overlap / max(sum(h_ngrams.values()), 1)
        return (2 * prec * rec / (prec + rec)) * 100 if (prec + rec) > 0 else 0.0

    r1 = rouge_n(1)
    r2 = rouge_n(2) if len(ref_tokens) >= 2 and len(hyp_tokens) >= 2 else 0.0

    lcs = lcs_length(ref_tokens, hyp_tokens)
    rec_l = lcs / len(ref_tokens)
    prec_l = lcs / len(hyp_tokens)
    rl = (2 * prec_l * rec_l / (prec_l + rec_l)) * 100 if (prec_l + rec_l) > 0 else 0.0

    return {"rouge_1": round(r1, 2), "rouge_2": round(r2, 2), "rouge_l": round(rl, 2)}


def compute_diversity(texts: List[str]) -> Dict[str, float]:
    """Computes Distinct-1, Distinct-2, and Repetition Rate."""
    unigrams, bigrams = set(), set()
    total_unigrams, total_bigrams = 0, 0

    for text in texts:
        words = text.split()
        total_unigrams += len(words)
        unigrams.update(words)
        b_list = list(zip(words[:-1], words[1:]))
        total_bigrams += len(b_list)
        bigrams.update(b_list)

    dist1 = len(unigrams) / max(total_unigrams, 1)
    dist2 = len(bigrams) / max(total_bigrams, 1)
    return {"distinct_1": round(dist1, 4), "distinct_2": round(dist2, 4), "repetition_rate": round(1.0 - dist1, 4)}


def compute_lm_ppl(model: TransformerLM, val_jsonl_path: str, tokenizer: Tokenizer, device: torch.device, max_samples: int = 500) -> float:
    """Evaluates validation perplexity on the pretraining corpus (same 500 documents finetuning monitors)."""
    model.eval()
    if not os.path.exists(val_jsonl_path):
        return 0.0

    total_loss, total_tokens = 0.0, 0
    with open(val_jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()][:max_samples]

    with torch.no_grad():
        for line in lines:
            doc = json.loads(line)
            tokens = tokenizer.encode(doc["text"]).ids
            if len(tokens) < 2:
                continue
            tokens = tokens[:512]
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
            _, loss, _ = model(x, targets=y)
            total_loss += loss.item() * len(tokens[1:])
            total_tokens += len(tokens[1:])

    avg_loss = total_loss / max(1, total_tokens)
    return round(math.exp(avg_loss), 2)


# --- Attention Analysis (Phase 2 toolkit) ---

def attention_summary(model: TransformerLM, tokenizer: Tokenizer, prompts: List[str], device: torch.device) -> Dict[str, Any]:
    """Per-layer / per-head entropy (bits) and mean attention distance (tokens) from the Phase 2 toolkit,
    averaged over reasoning prompts, plus layer means and counts of local / long-range heads."""
    analysis = phase2_eval.compute_attention_analysis(model, tokenizer, prompts, device)
    entropy = np.array(analysis["layer_head_entropy"])
    distance = np.array(analysis["layer_head_mean_distance"])
    return {
        "layer_head_entropy": analysis["layer_head_entropy"],
        "layer_head_mean_distance": analysis["layer_head_mean_distance"],
        "layer_mean_entropy": [round(float(v), 4) for v in entropy.mean(axis=1)],
        "layer_mean_distance": [round(float(v), 4) for v in distance.mean(axis=1)],
        "mean_entropy": round(float(entropy.mean()), 4),
        "mean_distance": round(float(distance.mean()), 4),
        "head_entropy_spread": round(float(entropy.std(axis=1).mean()), 4),
        "local_heads": int((distance <= LOCAL_HEAD_MAX_DIST).sum()),
        "long_range_heads": int((distance >= LONG_RANGE_HEAD_MIN_DIST).sum()),
        "num_prompts": len(prompts),
    }


def layer_attention(model: TransformerLM, tokenizer: Tokenizer, prompt: str, device: torch.device) -> Dict[str, np.ndarray]:
    """Head-averaged attention of the first and last layer for one prompt."""
    x = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, attns = model(x, return_attention=True)
    return {"early": attns[0][0].mean(dim=0).cpu().numpy(), "late": attns[-1][0].mean(dim=0).cpu().numpy()}


def plot_comparative_heatmaps(pre_attn: Dict[str, np.ndarray], fine_attn: Dict[str, np.ndarray], output_path: str):
    """Plots a 2x2 grid: early / late layer (rows) x pretrained / finetuned (columns), head-averaged."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    panels = [("early", "Early Layer (1)"), ("late", "Late Layer (last)")]

    for row, (key, layer_name) in enumerate(panels):
        for col, (attn, model_name) in enumerate([(pre_attn, "Pretrained"), (fine_attn, "Finetuned")]):
            ax = axes[row, col]
            sns.heatmap(attn[key], ax=ax, cmap="viridis", cbar=True, square=True,
                        cbar_kws={"label": "Attention weight (mean over heads)"})
            ax.set_title(f"{model_name} Model - {layer_name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")

    plt.suptitle("Hindi Model: Pretrained vs. Finetuned Attention on a Reasoning Prompt", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Comparative Attention Heatmap to: {output_path}")


def plot_head_shift(pre_summary: Dict[str, Any], fine_summary: Dict[str, Any], output_path: str):
    """Plots the per-head change (finetuned - pretrained) in attention entropy and mean attention distance."""
    d_entropy = np.array(fine_summary["layer_head_entropy"]) - np.array(pre_summary["layer_head_entropy"])
    d_distance = np.array(fine_summary["layer_head_mean_distance"]) - np.array(pre_summary["layer_head_mean_distance"])
    n_layer, n_head = d_entropy.shape

    fig, axes = plt.subplots(1, 2, figsize=(16, 0.7 * n_layer + 2.5), dpi=300)
    for ax, values, title, label in [
        (axes[0], d_entropy, "Change in Attention Entropy (bits)", "Finetuned - Pretrained entropy (bits)"),
        (axes[1], d_distance, "Change in Mean Attention Distance (tokens)", "Finetuned - Pretrained distance (tokens)"),
    ]:
        lim = float(np.abs(values).max()) or 1.0
        sns.heatmap(values, ax=ax, cmap="RdBu_r", vmin=-lim, vmax=lim, annot=True, fmt=".2f",
                    xticklabels=[f"H{h + 1}" for h in range(n_head)], yticklabels=[f"L{l + 1}" for l in range(n_layer)],
                    cbar_kws={"label": label})
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Layer")

    plt.suptitle(f"Hindi Model: Per-Head Attention Shift After Finetuning ({pre_summary['num_prompts']} reasoning prompts)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Per-Head Attention Shift to: {output_path}")


# --- Reasoning Evaluation ---

def evaluate_model(model: TransformerLM, tokenizer: Tokenizer, samples: List[Dict], device: torch.device,
                   answer_only: bool = False, num_qualitative: int = 5) -> Dict[str, Any]:
    """Runs the reasoning evaluation: generation from the question alone, then exact match on the answer."""
    split = [split_sample(s, answer_only) for s in samples]
    generations = generate_batch(model, tokenizer, [p for p, _, _ in split], device,
                                 max_new_tokens=8 if answer_only else 128)

    task_correct, task_total, task_chain = Counter(), Counter(), Counter()
    bleu_1_list, bleu_4_list = [], []
    rouge_1_list, rouge_2_list, rouge_l_list = [], [], []
    answered = 0
    successes, failures = [], []
    step_hits = defaultdict(lambda: defaultdict(list))  # task -> clause index -> [correct?]
    entity_errors, entity_errors_not_in_question = 0, 0

    for sample, (prompt, gold_continuation, gold_answer), generation in zip(samples, split, generations):
        task_name = sample.get("task", "general")
        pred_answer = extract_answer(generation, answer_only)
        is_correct = pred_answer == gold_answer
        answered += answer_only or ANSWER_MARKER in generation
        task_correct[task_name] += is_correct
        task_total[task_name] += 1

        if not is_correct and pred_answer and gold_answer not in NON_ENTITY_ANSWERS and pred_answer not in NON_ENTITY_ANSWERS:
            entity_errors += 1
            entity_errors_not_in_question += pred_answer not in sample["question"]

        if not answer_only:
            task_chain[task_name] += " ".join(generation.split()) == " ".join(gold_continuation.split())

            # Error analysis: which reasoning step first goes wrong
            gold_clauses, gen_clauses = reasoning_clauses(gold_continuation), reasoning_clauses(generation)
            for i, clause in enumerate(gold_clauses):
                step_hits[task_name][i].append(i < len(gen_clauses) and gen_clauses[i] == clause)

            # BLEU & ROUGE of the generated reasoning chain vs the gold one
            ref_tokens = gold_continuation.split()
            hyp_tokens = generation.split()
            b = compute_bleu(ref_tokens, hyp_tokens)
            r = compute_rouge(ref_tokens, hyp_tokens)
            bleu_1_list.append(b["bleu_1"])
            bleu_4_list.append(b["bleu_4"])
            rouge_1_list.append(r["rouge_1"])
            rouge_2_list.append(r["rouge_2"])
            rouge_l_list.append(r["rouge_l"])

        bucket = successes if is_correct else failures
        if len(bucket) < num_qualitative:
            bucket.append({
                "task": task_name,
                "prompt": prompt,
                "target": gold_answer,
                "generation": generation,
                "prediction": pred_answer,
                "status": "PASS" if is_correct else "FAIL"
            })

    n = len(samples)
    results = {
        "accuracy": round(100.0 * sum(task_correct.values()) / n, 2) if n else 0.0,
        "task_accuracy": {t: round(100.0 * task_correct[t] / task_total[t], 2) for t in sorted(task_total)},
        "answered_rate": round(100.0 * answered / n, 2) if n else 0.0,
        "wrong_entity_not_in_question": round(100.0 * entity_errors_not_in_question / entity_errors, 2) if entity_errors else 0.0,
        "qualitative_samples": successes + failures
    }
    if not answer_only:
        results.update({
            "chain_exact_match": round(100.0 * sum(task_chain.values()) / n, 2) if n else 0.0,
            "task_chain_exact_match": {t: round(100.0 * task_chain[t] / task_total[t], 2) for t in sorted(task_total)},
            "bleu_1": round(float(np.mean(bleu_1_list)), 2),
            "bleu_4": round(float(np.mean(bleu_4_list)), 2),
            "rouge_1": round(float(np.mean(rouge_1_list)), 2),
            "rouge_2": round(float(np.mean(rouge_2_list)), 2),
            "rouge_l": round(float(np.mean(rouge_l_list)), 2),
            "diversity": compute_diversity(generations),
            "step_accuracy": {t: [round(100.0 * sum(v) / len(v), 2) for _, v in sorted(steps.items())]
                              for t, steps in sorted(step_hits.items())},
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Phase 3 Evaluation Suite for Hindi Model")
    parser.add_argument("--pretrain-path", type=str, default="hindi/checkpoints/final_model.pt", help="Pretrained model path")
    parser.add_argument("--finetune-path", type=str, default="hindi/checkpoints/finetuned_model.pt", help="Finetuned model path")
    parser.add_argument("--answer-only-path", type=str, default="hindi/checkpoints/answer_only/finetuned_model.pt", help="Answer-only control model path")
    parser.add_argument("--tokenizer-path", type=str, default="hindi/tokenizer/tokenizer.json", help="Tokenizer path")
    parser.add_argument("--test-data", type=str, default="hindi/data/reasoning/test.jsonl", help="Test dataset path")
    parser.add_argument("--pretrain-val-data", type=str, default="hindi/data/processed/val.jsonl", help="Pretraining validation data path")
    parser.add_argument("--output-json", type=str, default="hindi/eval/phase3_results.json", help="Output JSON path")
    parser.add_argument("--output-heatmap", type=str, default="report/figures/pretrained_vs_finetuned_attention_hindi.png", help="Output heatmap path")
    parser.add_argument("--output-head-shift", type=str, default="report/figures/attention_head_shift_hindi.png", help="Per-head attention shift plot")
    parser.add_argument("--viz-dir", type=str, default="report/figures/phase3_attention/hindi", help="Phase 2 toolkit heatmaps for both checkpoints")
    parser.add_argument("--num-attention-prompts", type=int, default=200, help="Test prompts averaged in the attention analysis")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    samples = load_samples(args.test_data)

    # Attention statistics are averaged over reasoning prompts (as the model sees them); heatmaps use
    # the shortest of the first 50 test questions so every token label stays readable.
    attention_prompts = [split_sample(s)[0] for s in samples[:args.num_attention_prompts]]
    heatmap_prompt = min(attention_prompts[:50], key=lambda p: len(tokenizer.encode(p).ids))

    results = {"chance": chance_accuracy(samples)}
    print(f"Chance level -> {results['chance']['accuracy']}%")

    # 1. Pretrained Model: generation from the question, zero-shot likelihood ranking, PPL, attention
    print("Loading Pretrained Model...")
    pretrained_model, config = load_model(args.pretrain_path, device)
    pre_attn_summary = attention_summary(pretrained_model, tokenizer, attention_prompts, device)
    pre_attn = layer_attention(pretrained_model, tokenizer, heatmap_prompt, device)
    phase2_eval.generate_attention_visualizations(pretrained_model, tokenizer, heatmap_prompt, Path(args.viz_dir) / "pretrained", device)
    results["pretrained"] = {
        "metrics": evaluate_model(pretrained_model, tokenizer, samples, device),
        "zero_shot_candidate": zero_shot_candidate_accuracy(pretrained_model, tokenizer, samples, device),
        "pretrain_corpus_ppl": compute_lm_ppl(pretrained_model, args.pretrain_val_data, tokenizer, device),
        "attention": pre_attn_summary,
    }
    del pretrained_model
    print(f"Pretrained Model -> Accuracy: {results['pretrained']['metrics']['accuracy']}% | Zero-shot ranking: {results['pretrained']['zero_shot_candidate']['accuracy']}% | Pretrain PPL: {results['pretrained']['pretrain_corpus_ppl']}")

    # 2. Finetuned Model
    if os.path.exists(args.finetune_path):
        print("Loading Finetuned Model...")
        finetuned_model, _ = load_model(args.finetune_path, device)
        fine_attn_summary = attention_summary(finetuned_model, tokenizer, attention_prompts, device)
        fine_attn = layer_attention(finetuned_model, tokenizer, heatmap_prompt, device)
        phase2_eval.generate_attention_visualizations(finetuned_model, tokenizer, heatmap_prompt, Path(args.viz_dir) / "finetuned", device)
        results["finetuned"] = {
            "metrics": evaluate_model(finetuned_model, tokenizer, samples, device),
            "pretrain_corpus_ppl": compute_lm_ppl(finetuned_model, args.pretrain_val_data, tokenizer, device),
            "attention": fine_attn_summary,
        }
        del finetuned_model
        m = results["finetuned"]["metrics"]
        print(f"Finetuned Model  -> Accuracy: {m['accuracy']}% | Chain EM: {m['chain_exact_match']}% | Pretrain PPL: {results['finetuned']['pretrain_corpus_ppl']} | BLEU-4: {m['bleu_4']} | ROUGE-L: {m['rouge_l']}")
        print(f"Finetuned Task Accuracy -> {m['task_accuracy']}")

        plot_comparative_heatmaps(pre_attn, fine_attn, args.output_heatmap)
        plot_head_shift(pre_attn_summary, fine_attn_summary, args.output_head_shift)
    else:
        print(f"Finetuned model checkpoint not found at: {args.finetune_path}. Skipping finetuned eval.")

    # 3. Answer-only control model (finetuned without reasoning chains)
    if os.path.exists(args.answer_only_path):
        print("Loading Answer-Only Control Model...")
        control_model, _ = load_model(args.answer_only_path, device)
        results["finetuned_answer_only"] = {
            "metrics": evaluate_model(control_model, tokenizer, samples, device, answer_only=True),
            "pretrain_corpus_ppl": compute_lm_ppl(control_model, args.pretrain_val_data, tokenizer, device),
        }
        del control_model
        print(f"Answer-Only Model -> Accuracy: {results['finetuned_answer_only']['metrics']['accuracy']}% | Task Accuracy: {results['finetuned_answer_only']['metrics']['task_accuracy']}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved Comprehensive Phase 3 evaluation results to: {args.output_json}")


if __name__ == "__main__":
    main()
