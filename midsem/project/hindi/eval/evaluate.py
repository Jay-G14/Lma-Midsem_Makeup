"""Comprehensive Phase 2 PDF-Compliant Evaluation Suite for Hindi Transformer Language Model.

Evaluates:
1. Intrinsic Metrics: Cross-Entropy Loss, Perplexity (PPL), Bits-Per-Byte (BPB).
2. Empirical Sanity Check: Causal Mask Integrity (future token invariance).
3. Qualitative Text Generation: Native prompts with Greedy, Temp=0.5, Temp=1.0, Temp=1.5 sampling.
4. Fluency & Diversity Diagnostics: Distinct-1, Distinct-2, Repetition Rate across all 4 sampling regimes.
5. Reference-Based Quality Metrics: Zero-dependency BLEU-4, chrF++, and ROUGE-L evaluated per sampling regime.
6. Attention Analysis & Visualizations:
   - Layer-wise and Head-wise Attention Entropy H(A).
   - Layer-wise and Head-wise Mean Attention Distance.
   - Matplotlib/Seaborn Attention Heatmaps (Early vs. Late Layers, Heads, Token Annotations).
   - Layer x Head Attention Entropy Heatmap Matrix.
"""
import argparse
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM
from hindi.train.dataset import PretrainDataset

# Font setup for native script rendering
devanagari_font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
dev_font_prop = fm.FontProperties(fname=devanagari_font_path) if Path(devanagari_font_path).exists() else None


def test_causal_mask_integrity(model: TransformerLM, device: torch.device) -> bool:
    """Empirical sanity check verifying that changing future tokens produces ZERO change in past logits."""
    model.eval()
    config = model.config
    seq_len = min(64, config.max_seq_len)

    with torch.no_grad():
        x1 = torch.randint(4, config.vocab_size, (1, seq_len), device=device)
        x2 = x1.clone()
        
        mid = seq_len // 2
        x2[0, mid:] = (x2[0, mid:] + 13) % config.vocab_size

        logits1, _, _ = model(x1)
        logits2, _, _ = model(x2)

        diff = torch.abs(logits1[:, :mid, :] - logits2[:, :mid, :]).max().item()
        passed = diff < 1e-6

    print(f"  [Causal Mask Test] Max Logit Diff in past positions (0..{mid-1}): {diff:.8e} -> {'PASSED' if passed else 'FAILED'}")
    return passed


def generate_sample(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    greedy: bool = False,
    device: torch.device = torch.device("cpu")
) -> str:
    """Autoregressive text generation supporting Greedy and Temperature (0.5, 1.0, 1.5) sampling."""
    model.eval()
    config = model.config

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) >= config.max_seq_len:
                break

            logits, _, _ = model(input_ids)
            next_token_logits = logits[0, -1, :]

            if greedy:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / max(temperature, 1e-5)

                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


def compute_diversity(texts: List[str]) -> Dict[str, float]:
    """Compute Distinct-1, Distinct-2, and Repetition Rate."""
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


def compute_bleu4(ref_list: List[str], hyp_list: List[str]) -> float:
    """Compute sentence-level BLEU-4 with brevity penalty."""
    c_len, r_len = 0, 0
    total_log_prec = 0.0
    
    for r, h in zip(ref_list, hyp_list):
        r_toks, h_toks = r.split(), h.split()
        c_len += len(h_toks)
        r_len += len(r_toks)

    bp = math.exp(min(0, 1 - r_len / max(c_len, 1))) if c_len > 0 else 0.0

    for n in range(1, 5):
        matched, total = 0, 0
        for r, h in zip(ref_list, hyp_list):
            r_toks, h_toks = r.split(), h.split()
            h_ngrams = Counter([tuple(h_toks[i:i+n]) for i in range(len(h_toks)-n+1)])
            r_ngrams = Counter([tuple(r_toks[i:i+n]) for i in range(len(r_toks)-n+1)])
            for ng, count in h_ngrams.items():
                matched += min(count, r_ngrams.get(ng, 0))
            total += max(len(h_toks) - n + 1, 0)

        prec = (matched + 1e-6) / max(total + 1e-6, 1)
        total_log_prec += 0.25 * math.log(prec)

    return round(bp * math.exp(total_log_prec) * 100.0, 2)


def compute_chrf(ref_list: List[str], hyp_list: List[str], n_char: int = 6, beta: float = 2.0) -> float:
    """Compute character n-gram F-score (chrF++)."""
    total_f1 = 0.0
    for r, h in zip(ref_list, hyp_list):
        r_chars, h_chars = list(r.replace(" ", "")), list(h.replace(" ", ""))
        matched, total_h, total_r = 0, 0, 0
        for n in range(1, n_char + 1):
            h_ngrams = Counter([tuple(h_chars[i:i+n]) for i in range(len(h_chars)-n+1)])
            r_ngrams = Counter([tuple(r_chars[i:i+n]) for i in range(len(r_chars)-n+1)])
            for ng, count in h_ngrams.items():
                matched += min(count, r_ngrams.get(ng, 0))
            total_h += max(len(h_chars) - n + 1, 0)
            total_r += max(len(r_chars) - n + 1, 0)

        prec = matched / max(total_h, 1)
        rec = matched / max(total_r, 1)
        if prec + rec > 0:
            f1 = ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec)
        else:
            f1 = 0.0
        total_f1 += f1

    return round((total_f1 / max(len(ref_list), 1)) * 100.0, 2)


def compute_rouge_l(ref_list: List[str], hyp_list: List[str]) -> float:
    """Compute ROUGE-L based on Longest Common Subsequence."""
    def lcs(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    total_f1 = 0.0
    for r, h in zip(ref_list, hyp_list):
        r_toks, h_toks = r.split(), h.split()
        if not r_toks or not h_toks:
            continue
        lcs_len = lcs(r_toks, h_toks)
        rec = lcs_len / len(r_toks)
        prec = lcs_len / len(h_toks)
        if rec + prec > 0:
            f1 = (2 * prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        total_f1 += f1

    return round((total_f1 / max(len(ref_list), 1)) * 100.0, 2)


def compute_attention_analysis(model: TransformerLM, tokenizer: Tokenizer, prompts: List[str], device: torch.device) -> Dict[str, list]:
    """Compute per-layer and per-head attention entropy H(A) and Mean Attention Distance."""
    model.eval()
    layer_entropies = [[] for _ in range(model.config.n_layer)]
    layer_distances = [[] for _ in range(model.config.n_layer)]

    with torch.no_grad():
        for p in prompts:
            ids = tokenizer.encode(p).ids
            if len(ids) == 0:
                continue
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            _, _, attentions = model(input_ids, return_attention=True)
            
            if attentions is not None:
                T = input_ids.size(1)
                pos = torch.arange(T, device=device).float()
                dist_matrix = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))

                for l_idx, layer_attn in enumerate(attentions):
                    attn = layer_attn[0]  # (n_head, T, T)
                    entropy = -torch.sum(attn * torch.log2(attn + 1e-12), dim=-1).mean(dim=-1)
                    layer_entropies[l_idx].append(entropy.cpu().tolist())

                    mean_dist = torch.sum(attn * dist_matrix, dim=(-1, -2)) / float(T)
                    layer_distances[l_idx].append(mean_dist.cpu().tolist())

    head_entropies = []
    head_distances = []
    for l_idx in range(model.config.n_layer):
        if layer_entropies[l_idx]:
            e_mean = torch.tensor(layer_entropies[l_idx]).mean(dim=0).tolist()
            d_mean = torch.tensor(layer_distances[l_idx]).mean(dim=0).tolist()
            head_entropies.append([round(e, 4) for e in e_mean])
            head_distances.append([round(d, 4) for d in d_mean])
        else:
            head_entropies.append([0.0] * model.config.n_head)
            head_distances.append([0.0] * model.config.n_head)

    return {
        "layer_head_entropy": head_entropies,
        "layer_head_mean_distance": head_distances
    }


def generate_attention_visualizations(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    output_dir: Path,
    device: torch.device
):
    """Generate Matplotlib/Seaborn Attention Heatmaps (Early vs Late Layer across heads) & Attention Entropy Matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    tokens = [tokenizer.decode([tid]) for tid in encoded.ids]

    with torch.no_grad():
        _, _, attentions = model(input_ids, return_attention=True)

    if attentions is None or len(attentions) == 0:
        return

    n_layer = len(attentions)
    n_head = model.config.n_head
    early_layer_idx = 0
    late_layer_idx = n_layer - 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. Early vs. Late Layer Attention Heatmaps
        fig, axes = plt.subplots(2, n_head, figsize=(n_head * 3.5, 7), squeeze=False)
        
        for h in range(n_head):
            # Early Layer
            early_attn = attentions[early_layer_idx][0, h].cpu().numpy()
            sns.heatmap(early_attn, ax=axes[0, h], cmap="viridis", cbar=(h == n_head-1),
                        xticklabels=tokens, yticklabels=tokens, square=True)
            axes[0, h].set_title(f"L{early_layer_idx+1} Head {h+1}", fontsize=10)
            axes[0, h].tick_params(axis='both', which='major', labelsize=8)

            if dev_font_prop:
                for t in axes[0, h].get_xticklabels() + axes[0, h].get_yticklabels():
                    t.set_fontproperties(dev_font_prop)

            # Late Layer
            late_attn = attentions[late_layer_idx][0, h].cpu().numpy()
            sns.heatmap(late_attn, ax=axes[1, h], cmap="rocket", cbar=(h == n_head-1),
                        xticklabels=tokens, yticklabels=tokens, square=True)
            axes[1, h].set_title(f"L{late_layer_idx+1} Head {h+1}", fontsize=10)
            axes[1, h].tick_params(axis='both', which='major', labelsize=8)

            if dev_font_prop:
                for t in axes[1, h].get_xticklabels() + axes[1, h].get_yticklabels():
                    t.set_fontproperties(dev_font_prop)

        fig.suptitle(f"Attention Heatmaps (Early L1 vs Late L{n_layer})\nPrompt: '{prompt}'", fontsize=14, y=1.02)
        plt.tight_layout()
        heatmap_path = output_dir / "attention_heatmaps_early_late.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Visualization] Saved Attention Heatmaps -> {heatmap_path}")

        # 2. Attention Entropy Heatmap (Layers vs Heads Matrix)
        entropies = []
        for l_idx, layer_attn in enumerate(attentions):
            attn = layer_attn[0]
            entropy = -torch.sum(attn * torch.log2(attn + 1e-12), dim=-1).mean(dim=-1).cpu().numpy()
            entropies.append(entropy)

        entropy_matrix = np.array(entropies)

        fig, ax = plt.subplots(figsize=(n_head * 1.1, n_layer * 0.8 + 1))
        sns.heatmap(entropy_matrix, ax=ax, cmap="YlGnBu", annot=True, fmt=".2f",
                    xticklabels=[f"Head {h+1}" for h in range(n_head)],
                    yticklabels=[f"Layer {l+1}" for l in range(n_layer)])
        ax.set_title("Layer-wise & Head-wise Attention Entropy H(A)", fontsize=12)
        plt.tight_layout()
        entropy_path = output_dir / "attention_entropy_matrix.png"
        plt.savefig(entropy_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [Visualization] Saved Attention Entropy Matrix -> {entropy_path}")


def evaluate_perplexity_and_bpb(
    model: TransformerLM,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Compute Cross Entropy Loss, Perplexity (PPL), and Bits-Per-Byte (BPB)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, loss, _ = model(x, targets=y)

            batch_tokens = (y != model.config.pad_token_id).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    bpb = (avg_loss / math.log(2))

    return {"loss": round(avg_loss, 4), "perplexity": round(ppl, 2), "bits_per_byte": round(bpb, 4), "eval_tokens": total_tokens}


def evaluate_decoding_regimes(
    model: TransformerLM,
    tokenizer: Tokenizer,
    test_data_path: str,
    device: torch.device,
    num_prefixes: int = 50
) -> Dict[str, Dict[str, float]]:
    """Evaluates BLEU-4, chrF++, ROUGE-L, Distinct-1, Distinct-2, and Repetition Rate across all 4 decoding regimes (Greedy, Temp=0.5, 1.0, 1.5)."""
    print(f"\n--- EVALUATING GENERATION & REFERENCE METRICS ACROSS DECODING REGIMES ---")
    
    prefix_data = []
    with Path(test_data_path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(prefix_data) >= num_prefixes:
                break
            try:
                txt = json.loads(line).get("text", "").strip()
                words = txt.split()
                if len(words) > 10:
                    prompt_len = max(3, len(words) // 4)
                    prompt_str = " ".join(words[:prompt_len])
                    gen_len = max(10, len(words) - prompt_len)
                    prefix_data.append((prompt_str, txt, gen_len))
            except Exception:
                pass

    regimes = [
        ("greedy", True, 1.0),
        ("temp_0.5", False, 0.5),
        ("temp_1.0", False, 1.0),
        ("temp_1.5", False, 1.5),
    ]

    results_by_regime = {}

    for regime_name, is_greedy, temp in regimes:
        ref_samples = []
        hyp_samples = []

        for prompt_str, ref_str, gen_len in prefix_data:
            hyp_str = generate_sample(
                model, tokenizer, prompt_str,
                max_new_tokens=gen_len, greedy=is_greedy, temperature=temp, device=device
            )
            ref_samples.append(ref_str)
            hyp_samples.append(hyp_str)

        bleu = compute_bleu4(ref_samples, hyp_samples)
        chrf = compute_chrf(ref_samples, hyp_samples)
        rouge = compute_rouge_l(ref_samples, hyp_samples)
        div = compute_diversity(hyp_samples)

        regime_metrics = {
            "bleu_4": bleu,
            "chrf_pp": chrf,
            "rouge_l": rouge,
            "distinct_1": div["distinct_1"],
            "distinct_2": div["distinct_2"],
            "repetition_rate": div["repetition_rate"]
        }
        results_by_regime[regime_name] = regime_metrics

        print(f"  [{regime_name.upper()}] BLEU-4: {bleu} | chrF++: {chrf} | ROUGE-L: {rouge} | Dist-1: {div['distinct_1']} | Dist-2: {div['distinct_2']} | Repetition: {div['repetition_rate']}")

    return results_by_regime


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hindi Transformer LM on Held-Out Test Split")
    parser.add_argument("--model-path", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer.json")
    parser.add_argument("--test-data", required=True, help="Path to test.jsonl")
    parser.add_argument("--output-json", default="eval_results.json", help="Path to save evaluation JSON")
    parser.add_argument("--viz-dir", default="hindi/eval/visualizations", help="Directory to save attention heatmap plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device)
    config = TransformerConfig(**checkpoint["config"])
    
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"[Eval] Loaded model weights from {args.model_path}")

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # 1. Causal Mask Integrity
    causal_pass = test_causal_mask_integrity(model, device)

    # 2. Intrinsic PPL & BPB on Held-Out Test Set
    test_dataset = PretrainDataset(args.test_data, args.tokenizer_path, max_seq_len=config.max_seq_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    metrics = evaluate_perplexity_and_bpb(model, test_loader, device)

    # 3. Qualitative Text Generation Samples (Greedy, Temp=0.5, Temp=1.0, Temp=1.5)
    hindi_prompts = [
        "भारत एक",
        "विज्ञान और तकनीक",
        "भाषा का महत्व"
    ]
    print(f"\n--- QUALITATIVE GENERATION SAMPLES (HINDI) ---")
    qualitative_results = {}
    
    for prompt in hindi_prompts:
        greedy_out = generate_sample(model, tokenizer, prompt, max_new_tokens=40, greedy=True, device=device)
        t05_out = generate_sample(model, tokenizer, prompt, max_new_tokens=40, temperature=0.5, device=device)
        t10_out = generate_sample(model, tokenizer, prompt, max_new_tokens=40, temperature=1.0, device=device)
        t15_out = generate_sample(model, tokenizer, prompt, max_new_tokens=40, temperature=1.5, device=device)

        qualitative_results[prompt] = {
            "greedy": greedy_out,
            "temp_0.5": t05_out,
            "temp_1.0": t10_out,
            "temp_1.5": t15_out
        }

        print(f"\n[Prompt]: {prompt}")
        print(f"  Greedy:   {greedy_out}")
        print(f"  Temp=0.5: {t05_out}")
        print(f"  Temp=1.0: {t10_out}")
        print(f"  Temp=1.5: {t15_out}")

    # 4. Multi-Regime Reference Metrics (BLEU-4, chrF++, ROUGE-L, Distinct-1/2, Repetition per regime)
    regime_metrics = evaluate_decoding_regimes(model, tokenizer, args.test_data, device, num_prefixes=50)
    metrics["generation_metrics_by_regime"] = regime_metrics

    # Set top-level metrics to greedy by default for backward compatibility
    metrics["bleu_4"] = regime_metrics["greedy"]["bleu_4"]
    metrics["chrf_pp"] = regime_metrics["greedy"]["chrf_pp"]
    metrics["rouge_l"] = regime_metrics["greedy"]["rouge_l"]
    metrics["distinct_1"] = regime_metrics["greedy"]["distinct_1"]
    metrics["distinct_2"] = regime_metrics["greedy"]["distinct_2"]
    metrics["repetition_rate"] = regime_metrics["greedy"]["repetition_rate"]

    # 5. Attention Entropy and Mean Distance Analysis
    attn_analysis = compute_attention_analysis(model, tokenizer, hindi_prompts, device)
    metrics["attention_analysis"] = attn_analysis

    # 6. Matplotlib / Seaborn Visualizations
    print(f"\n--- GENERATING ATTENTION VISUALIZATIONS ---")
    generate_attention_visualizations(model, tokenizer, "भारत एक महान देश है", Path(args.viz_dir), device)

    print(f"\n=== FINAL HINDI EVALUATION RESULTS ===")
    print(f"  Loss: {metrics['loss']}")
    print(f"  Perplexity (PPL): {metrics['perplexity']}")
    print(f"  Bits-Per-Byte (BPB): {metrics['bits_per_byte']}")
    print(f"  Causal Mask Pass: {causal_pass}")
    print(f"  Greedy BLEU-4: {metrics['bleu_4']} | chrF++: {metrics['chrf_pp']} | ROUGE-L: {metrics['rouge_l']}")

    # 7. Save JSON
    metrics["causal_mask_integrity"] = causal_pass
    metrics["qualitative_samples"] = qualitative_results
    Path(args.output_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved full evaluation report to {args.output_json}")


if __name__ == "__main__":
    main()
