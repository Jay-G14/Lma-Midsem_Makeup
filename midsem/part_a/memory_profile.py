#!/usr/bin/env python3
"""A2: peak GPU memory and optimiser-state size, LoRA vs. full finetuning.

Every configuration trains on the SAME batches: the first --steps batches of epoch 1 of the Phase 3
finetuning data (seed 42 + 1, batch size 8, 15% replay), in train mode with the Phase 3 optimiser
(AdamW, weight decay 0.01, clipping 1.0). Recorded per configuration:
  weights_mb        memory allocated once the model is on the GPU (fp32 weights)
  grad_mb           size of the .grad tensors of the trainable parameters
  optimizer_mb      AdamW exp_avg + exp_avg_sq (+ step counters)
  peak_mb           torch.cuda.max_memory_allocated over the steps (weights + grads + optimiser +
                    activations + temporary buffers)
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

import midsem  # noqa: F401  (puts midsem/project/, the copied project code, first on sys.path)
from hindi.train.finetune import ReasoningDataset, pad_collate_fn
from midsem.part_a.common import LORA_ALPHA, PRETRAIN_TRAIN, REASONING_DIR, RESULTS_DIR, device, load_base, set_seed, tokenizer, write_json
from midsem.part_a.lora import inject_lora, lora_parameters, parameter_counts

MB = 2 ** 20


def profile(name, rank, lr, dataset, dev, steps, seed):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    before = torch.cuda.memory_allocated(dev)
    set_seed(seed)
    model = load_base(dev)
    if rank is None:
        for p in model.parameters():
            p.requires_grad_(True)
        params = [p for p in model.parameters()]
    else:
        inject_lora(model, rank, LORA_ALPHA)
        model.to(dev)
        params = lora_parameters(model)
    weights = torch.cuda.memory_allocated(dev) - before
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen, collate_fn=lambda b: pad_collate_fn(b, pad_id=1))
    model.train()
    torch.manual_seed(seed + 1)
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    t0, tokens = time.time(), 0
    for i, (x, y) in enumerate(loader):
        if i >= steps:
            break
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)   # as in finetune.py / lora_sft.py
        _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        tokens += x.numel()
    torch.cuda.synchronize(dev)
    seconds = time.time() - t0
    grads = sum(p.grad.numel() * p.grad.element_size() for p in params if p.grad is not None)
    opt_bytes = sum(t.numel() * t.element_size() for st in opt.state.values() for t in st.values() if torch.is_tensor(t))
    counts = parameter_counts(model)
    out = {"config": name, "rank": rank, "trainable_params": counts["trainable"], "weights_mb": round(weights / MB, 1),
           "grad_mb": round(grads / MB, 2), "optimizer_mb": round(opt_bytes / MB, 2),
           "peak_mb": round(torch.cuda.max_memory_allocated(dev) / MB, 1),
           "steps": steps, "steps_per_second": round(steps / seconds, 2), "tokens_per_second": round(tokens / seconds)}
    print(out)
    del model, opt, params
    return out


def main():
    p = argparse.ArgumentParser(description="Peak memory: full finetuning vs. LoRA on identical batches")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-lr", type=float, default=1e-3)
    args = p.parse_args()
    dev = device()
    assert dev.type == "cuda", "memory profiling needs a GPU"
    tok = tokenizer()
    dataset = ReasoningDataset(str(REASONING_DIR / "train.jsonl"), tok, max_seq_len=512,
                               pretrain_jsonl_path=str(PRETRAIN_TRAIN), replay_ratio=0.15)
    rows = [profile("full finetune (Phase 3)", None, 2e-5, dataset, dev, args.steps, args.seed)]
    rows += [profile(f"LoRA r={r}", r, args.lora_lr, dataset, dev, args.steps, args.seed) for r in (1, 4, 16)]
    write_json(RESULTS_DIR / "memory_profile.json", {"gpu": torch.cuda.get_device_name(dev), "batch_size": 8,
                                                     "steps": args.steps, "rows": rows})


if __name__ == "__main__":
    main()
