#!/usr/bin/env python3
"""A1: trainable-parameter counts and the step-0 identity check for LoRA on W_Q, W_K, W_V, W_O.

For each rank r in {1, 4, 16}:
  - counts trainable vs. total parameters and checks them against the closed form
    24 matrices x r x (512 + 512);
  - runs the pretrained model and the freshly wrapped model on the same inputs (eval mode, so no
    dropout) and compares the logits element by element: they must be bit-identical, because B = 0
    makes the LoRA branch output exactly 0.0 and W0 x + b + 0.0 == W0 x + b in IEEE arithmetic;
  - takes one backward pass and records the gradient norms of A and B: at step 0, dL/dA = s B^T g x^T
    is exactly zero (B = 0) while dL/dB = s g (A x)^T is not, so the first optimiser step moves B only;
  - takes one AdamW step and shows the logits then differ, so the identity is not vacuous.
"""

import json

import torch

import midsem  # noqa: F401  (puts midsem/project/, the copied project code, first on sys.path)
from hindi.eval.eval_phase3 import load_samples, split_sample
from hindi.train.finetune import pad_collate_fn
from midsem.part_a.common import (LORA_ALPHA, PRETRAIN_VAL, REASONING_DIR, RESULTS_DIR, device, load_base, set_seed,
                                  tokenizer, write_json)
from midsem.part_a.lora import (LoRALinear, expected_lora_count, inject_lora, lora_parameters, max_delta_norm,
                                parameter_counts)


def probe_batch(tok, dev, n_reasoning: int = 16, n_prose: int = 8):
    """Reasoning texts from the validation split plus pretraining-validation prose, right-padded."""
    eos = tok.token_to_id("</s>")
    items = []
    for s in load_samples(str(REASONING_DIR / "val.jsonl"))[:n_reasoning]:
        prompt, _, _ = split_sample(s)
        items.append((torch.tensor(tok.encode(s["text"]).ids + [eos]), len(tok.encode(prompt).ids)))
    with open(PRETRAIN_VAL, encoding="utf-8") as f:
        for line in f:
            if len(items) >= n_reasoning + n_prose:
                break
            ids = tok.encode(json.loads(line)["text"]).ids[:256]
            if len(ids) > 1:
                items.append((torch.tensor(ids), 0))
    x, y = pad_collate_fn(items, pad_id=1)
    return x.to(dev), y.to(dev)


def main():
    dev = device()
    tok = tokenizer()
    x, y = probe_batch(tok, dev)
    base = load_base(dev)
    with torch.no_grad():
        base_logits, base_loss, _ = base(x, targets=y)
    results = {"probe_batch_shape": list(x.shape), "alpha": LORA_ALPHA, "ranks": {}}

    for r in (1, 4, 16):
        set_seed(42)
        model = load_base(dev)
        wrapped = inject_lora(model, r, LORA_ALPHA)
        model.to(dev).eval()
        counts = parameter_counts(model)
        assert counts["trainable"] == expected_lora_count(model, r)
        assert all(p.requires_grad == (".lora_" in n) for n, p in model.named_parameters())

        with torch.no_grad():
            logits, loss, _ = model(x, targets=y)
        diff = (logits - base_logits).abs().max().item()
        identical = torch.equal(logits, base_logits)

        # One backward pass at step 0: which LoRA matrices receive gradient?
        model.zero_grad(set_to_none=True)
        _, loss_train, _ = model(x, targets=y)
        loss_train.backward()
        grad_a = torch.sqrt(sum(m.lora_A.grad.pow(2).sum() for m in model.modules() if isinstance(m, LoRALinear))).item()
        grad_b = torch.sqrt(sum(m.lora_B.grad.pow(2).sum() for m in model.modules() if isinstance(m, LoRALinear))).item()

        # One AdamW step, then compare again.
        opt = torch.optim.AdamW(lora_parameters(model), lr=1e-3, weight_decay=0.01)
        opt.step()
        with torch.no_grad():
            logits_after, _, _ = model(x, targets=y)
        results["ranks"][str(r)] = {
            "wrapped_matrices": len(wrapped),
            "trainable": counts["trainable"],
            "total": counts["total"],
            "base_params": counts["frozen"],
            "trainable_share_pct": round(counts["trainable_share_pct"], 4),
            "trainable_share_of_base_pct": round(100.0 * counts["trainable"] / counts["frozen"], 4),
            "step0_logits_bit_identical": identical,
            "step0_max_abs_logit_diff": diff,
            "step0_loss_base": round(base_loss.item(), 6),
            "step0_loss_lora": round(loss.item(), 6),
            "step0_grad_norm_A": grad_a,
            "step0_grad_norm_B": grad_b,
            "after_one_step_max_abs_logit_diff": (logits_after - base_logits).abs().max().item(),
            "after_one_step_max_delta_W_norm": max_delta_norm(model),
        }
        print(f"r={r:2d} trainable={counts['trainable']:,} / {counts['total']:,} ({counts['trainable_share_pct']:.4f}%) | "
              f"identical={identical} max|dlogit|={diff:.1e} | grad A={grad_a:.2e} grad B={grad_b:.2e} | "
              f"after 1 step max|dlogit|={results['ranks'][str(r)]['after_one_step_max_abs_logit_diff']:.2e}")
        del model, opt
    write_json(RESULTS_DIR / "lora_sanity.json", results)


if __name__ == "__main__":
    main()
