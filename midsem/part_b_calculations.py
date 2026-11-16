#!/usr/bin/env python3
"""Part B (variant 1): every number in the written answers, computed step by step.

Variant 1 values: LSH b = 10, r = 10 | C = 3e21 FLOPs | LoRA 4096 x 4096, r = 4 | 8B model | alpha = 0.5.
Phase 1 training-split token counts (report/report.md, Table "Train Token Count (90%)"):
Hindi (Model H) 611,672,304; Nepali (Model L) 450,106,294.
"""

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "part_b.json"


def b1_lsh(b: int = 10, r: int = 10):
    """P(candidate) = 1 - (1 - s^r)^b for two documents with Jaccard similarity s."""
    p = lambda s: 1 - (1 - s ** r) ** b
    # similarity at which P = 0.5: (1 - s^r)^b = 0.5  =>  s = (1 - 0.5^(1/b))^(1/r)
    s_half = (1 - 0.5 ** (1 / b)) ** (1 / r)
    return {
        "b": b, "r": r, "hashes": b * r,
        "s0.8": {"s^r": 0.8 ** r, "1-s^r": 1 - 0.8 ** r, "(1-s^r)^b": (1 - 0.8 ** r) ** b, "P": p(0.8)},
        "s0.3": {"s^r": 0.3 ** r, "1-s^r": 1 - 0.3 ** r, "(1-s^r)^b": (1 - 0.3 ** r) ** b, "P": p(0.3)},
        "threshold_(1/b)^(1/r)": (1 / b) ** (1 / r),
        "P_at_threshold": p((1 / b) ** (1 / r)),
        "s_where_P_is_0.5": s_half,
        "curve": {f"{s:.2f}": p(s) for s in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)},
        "alternative_b20_r5": {"threshold": (1 / 20) ** (1 / 5), "P(0.8)": 1 - (1 - 0.8 ** 5) ** 20,
                                "P(0.5)": 1 - (1 - 0.5 ** 5) ** 20, "P(0.3)": 1 - (1 - 0.3 ** 5) ** 20},
    }


def b2_mixing(alpha: float = 0.5, n_h: int = 611_672_304, n_l: int = 450_106_294, budget: float = 500e6):
    """Temperature sampling q_i = p_i^alpha / sum_j p_j^alpha with p_i the share by size."""
    total = n_h + n_l
    p_h, p_l = n_h / total, n_l / total
    w_h, w_l = p_h ** alpha, p_l ** alpha
    q_h, q_l = w_h / (w_h + w_l), w_l / (w_h + w_l)
    return {
        "tokens": {"hindi": n_h, "nepali": n_l, "total": total},
        "by_size": {"hindi": p_h, "nepali": p_l},
        "p^alpha": {"hindi": w_h, "nepali": w_l, "sum": w_h + w_l},
        "alpha_sampling": {"hindi": q_h, "nepali": q_l},
        "budget_tokens": budget,
        "tokens_drawn_alpha": {"hindi": q_h * budget, "nepali": q_l * budget},
        "tokens_drawn_by_size": {"hindi": p_h * budget, "nepali": p_l * budget},
        "epochs_alpha": {"hindi": q_h * budget / n_h, "nepali": q_l * budget / n_l},
        "epochs_by_size": {"hindi": p_h * budget / n_h, "nepali": p_l * budget / n_l},
        "uniform_alpha0_nepali_epochs": 0.5 * budget / n_l,
        "nepali_upsampling_factor": q_l / p_l,
    }


def b3_chinchilla(C: float = 3e21, scale: float = 4.0):
    """C = 6 N D with D = 20 N  =>  N = sqrt(C / 120). Loss terms with the Besiroglu et al. (2024)
    refit of Chinchilla's parametric law (E = 1.82, A = 482.01, alpha = 0.3478, B = 2085.43,
    beta = 0.3658), whose optimum is ~20 tokens per parameter."""
    N = math.sqrt(C / 120)
    D = 20 * N
    N2 = scale * N
    D2 = C / (6 * N2)
    E, A, a, B, b = 1.82, 482.01, 0.3478, 2085.43, 0.3658
    L = lambda n, d: (E + A / n ** a + B / d ** b, A / n ** a, B / d ** b)
    l1, p1, d1 = L(N, D)
    l2, p2, d2 = L(N2, D2)
    # optimum of L under C = 6ND: N* = G (C/6)^(b/(a+b)), G = (aA / (bB))^(1/(a+b))
    G = (a * A / (b * B)) ** (1 / (a + b))
    n_opt = G * (C / 6) ** (b / (a + b))
    return {
        "C": C, "N": N, "D": D, "check_6ND": 6 * N * D,
        "colleague": {"N": N2, "D": D2, "tokens_per_param": D2 / N2, "check_6ND": 6 * N2 * D2},
        "param_term_factor_4^-alpha": scale ** -a, "data_term_factor_4^beta": scale ** b,
        "loss_compute_optimal": {"L": l1, "A/N^a": p1, "B/D^b": d1},
        "loss_colleague": {"L": l2, "A/N^a": p2, "B/D^b": d2},
        "delta_L": l2 - l1,
        "balance_at_optimum": {"alpha*A/N^a": a * p1, "beta*B/D^b": b * d1},
        "fit_optimum": {"N*": n_opt, "D*": C / (6 * n_opt), "tokens_per_param": C / (6 * n_opt) / n_opt},
        "chinchilla_original_fit": {
            name: {"L": 1.69 + 406.4 / n ** 0.34 + 410.7 / d ** 0.28}
            for name, (n, d) in {"N=5e9": (N, D), "N=2e10": (N2, D2)}.items()},
    }


def b4_lora_memory(d: int = 4096, k: int = 4096, r: int = 4, n_params: float = 8e9):
    lora = r * (d + k)
    full = d * k
    fp32 = 4
    return {
        "lora_params": lora, "full_update_params": full, "share_pct": 100 * lora / full, "ratio": full / lora,
        "inference_fp32_bytes": n_params * fp32, "inference_fp32_GB": n_params * fp32 / 1e9, "inference_fp32_GiB": n_params * fp32 / 2 ** 30,
        "full_ft_adam_bytes_per_param": {"weights": 4, "gradients": 4, "adam_m": 4, "adam_v": 4, "total": 16},
        "full_ft_adam_GB": n_params * 16 / 1e9, "full_ft_adam_GiB": n_params * 16 / 2 ** 30,
        "mixed_precision_bytes_per_param": {"bf16 weights": 2, "bf16 grads": 2, "fp32 master": 4, "adam_m": 4, "adam_v": 4, "total": 16},
    }


def main():
    out = {"B1": b1_lsh(), "B2": b2_mixing(), "B3": b3_chinchilla(), "B4": b4_lora_memory()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
