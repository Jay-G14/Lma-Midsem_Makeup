#!/usr/bin/env python3
"""A4: Direct Preference Optimisation (Rafailov et al., 2023) written from scratch (no trl).

    L(theta) = - E_(x, y_w, y_l) log sigmoid( beta * [ (log pi_theta(y_w|x) - log pi_ref(y_w|x))
                                                     - (log pi_theta(y_l|x) - log pi_ref(y_l|x)) ] )

  log pi(y|x)  sum of the log-probabilities of the reply tokens (reasoning, answer and </s>); the
               prompt tokens are not scored
  policy       the best LoRA model; its LoRA A / B keep training, W0 stays frozen
  reference    a second copy of the same best LoRA model, fully frozen
  margin       (chosen log-ratio) - (rejected log-ratio), in nats; beta * margin is the reward margin

At step 0 the policy and the reference are the same network, so both log-ratios are exactly 0 and
the loss is -log sigmoid(0) = log 2 = 0.6931 for every pair, whatever beta is. Dropout is disabled in
both networks (as TRL's disable_dropout default does) so that this holds exactly and so the
log-ratios measure the parameter change, not dropout noise.

Tokens the chosen and rejected replies share before the first difference have identical
log-probabilities under each model, so they cancel in the margin: only the tokens from the
decisive step onwards carry the preference signal.
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

from pathlib import Path

import torch
import torch.nn.functional as F

from midsem.part_a.common import (CKPT_DIR, MIDSEM, DATA_DIR, LORA_ALPHA, RESULTS_DIR, device, load_lora_model, read_json, set_seed,
                                  tokenizer, write_json)
from midsem.part_a.lora import adapter_state_dict, lora_parameters

PAD = 1


def encode_pairs(path, tok) -> List[Tuple[List[int], List[int], List[int]]]:
    """(prompt ids, chosen reply ids, rejected reply ids); each reply ends with </s>.

    Replies are taken as the suffix of encode(prompt + reply) after the prompt tokens, which is how the
    finetuning data is tokenised; the prompt tokens must be an exact prefix in both sequences."""
    eos = tok.token_to_id("</s>")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            p = tok.encode(row["prompt"]).ids
            c = tok.encode(row["prompt"] + row["chosen"]).ids + [eos]
            r = tok.encode(row["prompt"] + row["rejected"]).ids + [eos]
            assert c[:len(p)] == p and r[:len(p)] == p, row["id"]
            out.append((p, c[len(p):], r[len(p):]))
    return out


def collate(batch, dev) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stacks the B chosen sequences, then the B rejected ones: inputs (2B, L-1), targets, reply mask.

    For a sequence s with prompt length P, the input is s[:-1] and the target s[1:]; target position j
    predicts s[j+1], so the reply tokens s[P:] are the targets at positions P-1 ... len(s)-2."""
    seqs = [p + c for p, c, _ in batch] + [p + r for p, _, r in batch]
    plens = [len(p) for p, _, _ in batch] * 2
    width = max(len(s) for s in seqs) - 1
    x = torch.full((len(seqs), width), PAD, dtype=torch.long)
    y = torch.full((len(seqs), width), PAD, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.float32)
    for i, (s, plen) in enumerate(zip(seqs, plens)):
        n = len(s) - 1
        x[i, :n] = torch.tensor(s[:-1])
        y[i, :n] = torch.tensor(s[1:])
        mask[i, plen - 1:n] = 1.0
    return x.to(dev), y.to(dev), mask.to(dev)


def sequence_logps(model, x, y, mask) -> torch.Tensor:
    """Sum over reply tokens of log p(token | prefix), one value per sequence."""
    logits, _, _ = model(x)
    logp = torch.gather(F.log_softmax(logits.float(), dim=-1), 2, y.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(dim=-1)


def dpo_loss(policy_c, policy_r, ref_c, ref_r, beta: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """The DPO loss and its diagnostics. Inputs are per-pair sequence log-probabilities."""
    chosen_lr = policy_c - ref_c
    rejected_lr = policy_r - ref_r
    margin = chosen_lr - rejected_lr
    loss = -F.logsigmoid(beta * margin).mean()
    return loss, {"margin": margin, "chosen_logratio": chosen_lr, "rejected_logratio": rejected_lr,
                  "policy_chosen_logp": policy_c, "policy_rejected_logp": policy_r}


def forward_pair_batch(policy, reference, batch, dev, beta):
    x, y, mask = collate(batch, dev)
    n = len(batch)
    logps = sequence_logps(policy, x, y, mask)
    with torch.no_grad():
        ref = sequence_logps(reference, x, y, mask)
    return dpo_loss(logps[:n], logps[n:], ref[:n], ref[n:], beta)


@torch.no_grad()
def evaluate_pairs(policy, reference, pairs, dev, beta, batch_size=32) -> Dict[str, float]:
    """Mean loss / margin / log-ratios and preference accuracy (margin > 0) on a set of pairs."""
    sums, count, wins = {"loss": 0.0, "margin": 0.0, "chosen_logratio": 0.0, "rejected_logratio": 0.0}, 0, 0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        loss, d = forward_pair_batch(policy, reference, batch, dev, beta)
        sums["loss"] += loss.item() * len(batch)
        for k in ("margin", "chosen_logratio", "rejected_logratio"):
            sums[k] += d[k].sum().item()
        wins += (d["margin"] > 0).sum().item()
        count += len(batch)
    out = {k: v / count for k, v in sums.items()}
    out["preference_accuracy"] = wins / count
    return out


def save_adapter(path, policy, rank, alpha, adapter, args, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"adapter": adapter_state_dict(policy), "rank": rank, "alpha": alpha, "init_adapter": str(adapter),
                "beta": args.beta, "steps": step, "hparams": vars(args)}, path)


def resolve_adapter(path) -> Path:
    """Adapter paths are stored relative to midsem/; an absolute path from before the adapters moved from
    hindi/checkpoints/midsem/ to midsem/checkpoints/ is mapped to the same run folder there."""
    p = Path(path)
    if not p.is_absolute():
        return MIDSEM / p
    return p if p.exists() else CKPT_DIR / p.parent.name / p.name


def train(args):
    dev = device()
    tok = tokenizer()
    if args.init_adapter:                       # explicit starting point (e.g. the best LoRA run on Ada)
        adapter = args.init_adapter
    else:                                       # default: the best LoRA model chosen on validation
        adapter = read_json(RESULTS_DIR / args.best_lora_file)["adapter_path"]
    adapter = str(resolve_adapter(adapter))
    meta = torch.load(adapter, map_location="cpu", weights_only=False)
    rank, alpha = meta["rank"], meta.get("alpha", LORA_ALPHA)
    tag = args.tag or f"dpo_beta{args.beta}"
    save_steps = {int(s) for s in args.save_steps.split(",") if s.strip()} if args.save_steps else set()
    set_seed(args.seed)

    policy = load_lora_model(dev, rank, alpha, adapter)
    reference = load_lora_model(dev, rank, alpha, adapter)
    for p in reference.parameters():
        p.requires_grad_(False)
    policy.eval()      # dropout off; gradients still flow
    reference.eval()
    params = lora_parameters(policy)
    assert all(p.requires_grad for p in params) and not any(p.requires_grad for p in reference.parameters())

    train_pairs = encode_pairs(DATA_DIR / "part_a_preference_pairs.jsonl", tok)
    val_pairs = encode_pairs(DATA_DIR / "part_a_preference_pairs_val.jsonl", tok)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(args.seed)

    # Step 0, before any update.
    step0_batch = forward_pair_batch(policy, reference, train_pairs[:args.batch_size], dev, args.beta)
    step0_train = evaluate_pairs(policy, reference, train_pairs, dev, args.beta)
    log = [{"step": 0, "loss": step0_batch[0].item(), **{k: v.mean().item() for k, v in step0_batch[1].items()},
            "preference_accuracy": (step0_batch[1]["margin"] > 0).float().mean().item()}]
    val_log = [{"step": 0, **evaluate_pairs(policy, reference, val_pairs, dev, args.beta)}]
    print(f"[beta={args.beta}] step 0 loss {log[0]['loss']:.6f} (log 2 = {math.log(2):.6f}); "
          f"full-train-set loss {step0_train['loss']:.6f}; max |margin| {step0_batch[1]['margin'].abs().max().item():.2e}")

    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_pairs), generator=gen).tolist()
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            batch = [train_pairs[j] for j in order[i:i + args.batch_size]]
            loss, d = forward_pair_batch(policy, reference, batch, dev, args.beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            step += 1
            log.append({"step": step, "epoch": epoch, "loss": loss.item(), **{k: v.mean().item() for k, v in d.items()},
                        "preference_accuracy": (d["margin"] > 0).float().mean().item()})
            if step in save_steps:              # intermediate policies, evaluated later on the validation split
                save_adapter(CKPT_DIR / f"{tag}_step{step}" / "adapter.pt", policy, rank, alpha, adapter, args, step)
            if args.max_steps and step >= args.max_steps:
                break
            if step % args.eval_every == 0:
                val_log.append({"step": step, **evaluate_pairs(policy, reference, val_pairs, dev, args.beta)})
                v = val_log[-1]
                print(f"  step {step:4d} | train loss {loss.item():.4f} margin {d['margin'].mean().item():7.2f} | "
                      f"val loss {v['loss']:.4f} margin {v['margin']:7.2f} acc {v['preference_accuracy']:.3f}")
        if args.max_steps and step >= args.max_steps:
            break
    if val_log[-1]["step"] != step:
        val_log.append({"step": step, **evaluate_pairs(policy, reference, val_pairs, dev, args.beta)})

    adapter_path = CKPT_DIR / tag / "adapter.pt"
    save_adapter(adapter_path, policy, rank, alpha, adapter, args, step)
    summary = {"tag": tag, "beta": args.beta, "lr": args.lr, "seed": args.seed, "rank": rank, "init_adapter": str(adapter),
               "adapter_path": str(adapter_path), "saved_steps": sorted(save_steps),
               "steps": step, "seconds": round(time.time() - t0, 1), "hparams": vars(args),
               "step0": {"first_batch_loss": log[0]["loss"], "train_set_loss": step0_train["loss"], "log2": math.log(2)},
               "final_train_set": evaluate_pairs(policy, reference, train_pairs, dev, args.beta),
               "train_log": log, "val_log": val_log}
    write_json(RESULTS_DIR / f"{tag}_history.json", summary)
    print(f"Saved {adapter_path}; final val {val_log[-1]}")


def main():
    p = argparse.ArgumentParser(description="DPO from scratch on the best LoRA model")
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = all epochs)")
    p.add_argument("--tag", type=str, default=None, help="run name (default dpo_beta<beta>)")
    p.add_argument("--init-adapter", type=str, default=None, help="LoRA adapter to start from (default: results/<best-lora-file>)")
    p.add_argument("--best-lora-file", type=str, default="best_lora.json")
    p.add_argument("--save-steps", type=str, default="", help="comma-separated steps at which to save the policy")
    train(p.parse_args())


if __name__ == "__main__":
    main()
