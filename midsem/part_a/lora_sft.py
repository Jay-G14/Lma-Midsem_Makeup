#!/usr/bin/env python3
"""A2: LoRA finetuning of Model H on the Phase 3 reasoning data (rank sweep r = 1, 4, 16).

Identical to the Phase 3 full finetune (hindi/train/finetune.py) except for what is trained:
  same data      12,000 reasoning samples + 2,117 pretraining replay documents (15%), built by the
                 Phase 3 ReasoningDataset; question tokens masked out of the loss; </s> appended
  same recipe    AdamW (weight decay 0.01), constant LR, gradient clipping 1.0, batch size 8,
                 5 epochs, seed 42 with the same per-epoch data order, best epoch chosen by
                 reasoning-validation loss
  different      only the LoRA A / B matrices on W_Q, W_K, W_V, W_O of all 6 blocks train
                 (alpha = 16, scaling alpha / r); the LR is higher than full finetuning's 2e-5
                 because only a low-rank update starting from zero is learned

Checkpoints hold only the adapter (A, B), so they are a few hundred KB; latest_adapter.pt also
holds optimiser / scheduler state, epoch, step and history, so --resume continues a killed run.
"""

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import midsem  # noqa: F401  (puts midsem/project/, the copied project code, first on sys.path)
from hindi.train.finetune import ReasoningDataset, evaluate_reasoning_loss, pad_collate_fn
from midsem.part_a.common import (CKPT_DIR, LORA_ALPHA, PRETRAIN_TRAIN, REASONING_DIR, RESULTS_DIR, device, load_base,
                                  pretrain_ppl, set_seed, tokenizer, write_json)
from midsem.part_a.lora import ATTENTION_TARGETS, adapter_state_dict, inject_lora, load_adapter_state, lora_parameters, parameter_counts


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Bytes held by AdamW's exp_avg / exp_avg_sq (and step) tensors."""
    return sum(t.numel() * t.element_size() for st in optimizer.state.values() for t in st.values() if torch.is_tensor(t))


def save(path, model, optimizer, scheduler, args, epoch, step, best_val, history, with_optimizer: bool):
    """Adapter checkpoint; the resume checkpoint additionally holds optimiser and scheduler state."""
    state = {"adapter": adapter_state_dict(model), "rank": args.rank, "alpha": args.alpha, "targets": list(ATTENTION_TARGETS),
             "epoch": epoch, "step": step, "best_val_reasoning_loss": best_val, "history": history, "hparams": vars(args)}
    if with_optimizer:
        state.update({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
    torch.save(state, path)


def train(args):
    dev = device()
    tok = tokenizer()
    out_dir = args.output_dir or str(CKPT_DIR / f"lora_r{args.rank}")
    os.makedirs(out_dir, exist_ok=True)
    pad_id = tok.token_to_id("<pad>")

    # Model: pretrained Model H, every weight frozen, LoRA on W_Q / W_K / W_V / W_O of every block.
    set_seed(args.seed)
    model = load_base(dev)
    inject_lora(model, args.rank, args.alpha)
    model.to(dev)
    counts = parameter_counts(model)
    print(f"[LoRA r={args.rank}] trainable {counts['trainable']:,} of {counts['total']:,} ({counts['trainable_share_pct']:.4f}%)")

    # Data: exactly the Phase 3 finetuning datasets.
    train_ds = ReasoningDataset(str(REASONING_DIR / "train.jsonl"), tok, max_seq_len=512,
                                pretrain_jsonl_path=str(PRETRAIN_TRAIN), replay_ratio=args.replay_ratio)
    val_ds = ReasoningDataset(str(REASONING_DIR / "val.jsonl"), tok, max_seq_len=512)
    collate = lambda batch: pad_collate_fn(batch, pad_id=pad_id)
    data_rng = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=data_rng)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    params = lora_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    latest = os.path.join(out_dir, "latest_adapter.pt")
    start_epoch, step, best_val, history = 1, 0, float("inf"), []
    if args.resume and os.path.exists(latest):
        state = torch.load(latest, map_location="cpu", weights_only=False)
        load_adapter_state(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, step, best_val, history = state["epoch"] + 1, state["step"], state["best_val_reasoning_loss"], state["history"]
        print(f"[Resume] from epoch {state['epoch']} (step {step})")
    else:
        val0 = evaluate_reasoning_loss(model, val_loader, dev)
        history.append({"epoch": 0, "train_loss": None, "val_reasoning_loss": val0, "pretrain_ppl": pretrain_ppl(model, tok, dev)})
        print(f"[Epoch 0] val reasoning loss {val0:.4f} | pretrain PPL {history[0]['pretrain_ppl']:.2f}")

    step_losses = []
    for epoch in range(start_epoch, args.epochs + 1):
        torch.manual_seed(args.seed + epoch)       # same per-epoch data order and dropout as Phase 3
        data_rng.manual_seed(args.seed + epoch)
        model.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0, running = time.time(), 0.0
        for x, y in tqdm(train_loader, desc=f"r={args.rank} epoch {epoch}/{args.epochs}", mininterval=30):
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item()
            if step % 50 == 0:
                step_losses.append({"step": step, "loss": loss.item()})
        peak_mb = torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None
        val = evaluate_reasoning_loss(model, val_loader, dev)
        ppl = pretrain_ppl(model, tok, dev)
        history.append({"epoch": epoch, "train_loss": running / len(train_loader), "val_reasoning_loss": val, "pretrain_ppl": ppl,
                        "epoch_seconds": round(time.time() - t0, 1), "peak_train_memory_mb": peak_mb,
                        "optimizer_state_bytes": optimizer_state_bytes(optimizer)})
        print(f"[Epoch {epoch}] train {history[-1]['train_loss']:.4f} | val reasoning {val:.4f} | PPL {ppl:.2f} | "
              f"{history[-1]['epoch_seconds']}s | peak {peak_mb:.0f} MB")
        if val < best_val:
            best_val = val
            save(os.path.join(out_dir, "best_adapter.pt"), model, optimizer, scheduler, args, epoch, step, best_val, history, False)
        save(latest, model, optimizer, scheduler, args, epoch, step, best_val, history, True)

    best_epoch = min((h for h in history if h["epoch"] > 0), key=lambda h: h["val_reasoning_loss"])["epoch"]
    summary = {"rank": args.rank, "alpha": args.alpha, "lr": args.lr, "params": counts, "best_epoch": best_epoch,
               "best_val_reasoning_loss": best_val, "history": history, "step_losses": step_losses, "hparams": vars(args)}
    name = args.tag or f"lora_r{args.rank}"
    write_json(os.path.join(out_dir, "history.json"), summary)
    write_json(RESULTS_DIR / f"{name}_history.json", summary)
    print(f"Done. Best epoch {best_epoch} (val reasoning loss {best_val:.4f}).")


def main():
    p = argparse.ArgumentParser(description="LoRA finetuning of Model H on the Phase 3 reasoning data")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=float, default=LORA_ALPHA)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--replay-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None, help="name of the results/<tag>_history.json file")
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
