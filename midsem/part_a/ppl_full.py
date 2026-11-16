#!/usr/bin/env python3
"""A5 (extended): drift perplexity on the FULL pretraining validation split, computed with the
project's Phase 2 method (hindi/train/dataset.PretrainDataset + hindi/eval/evaluate.evaluate_perplexity_and_bpb):
every document is wrapped in <s> ... </s>, all documents are concatenated, and the stream is cut into
non-overlapping 513-token chunks (512 predictions each). This complements the 500-document measure
used by Phase 3 (pretrain_ppl in common.py), which truncates each document to 512 tokens.

The packed tensor is cached (--cache) so several worker processes do not re-tokenize 272 MB of text.
Writes results/ppl_full_<name>.json for every model name (same names as evaluate_models.py).
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

import midsem  # noqa: F401  (puts midsem/project/, the copied project code, first on sys.path)
from hindi.eval.evaluate import evaluate_perplexity_and_bpb
from hindi.train.dataset import PretrainDataset
from midsem.part_a.common import PRETRAIN_VAL, RESULTS_DIR, TOKENIZER_PATH, device, write_json
from midsem.part_a.evaluate_models import build


def packed_chunks(cache: str) -> torch.Tensor:
    """(num_chunks, 513) int32 tensor of the packed validation split, built once and cached."""
    if cache and os.path.exists(cache):
        return torch.load(cache)
    ds = PretrainDataset(str(PRETRAIN_VAL), str(TOKENIZER_PATH), max_seq_len=512)
    if cache:
        tmp = cache + f".tmp{os.getpid()}"
        torch.save(ds.samples, tmp)
        os.replace(tmp, cache)                 # atomic, so a concurrent reader never sees a partial file
    return ds.samples


def main():
    p = argparse.ArgumentParser(description="Full-validation-split perplexity (Phase 2 method)")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--cache", type=str, default=str(RESULTS_DIR / ".." / "val_packed_513.pt"))
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    dev = device()
    chunks = packed_chunks(args.cache)
    x, y = chunks[:, :-1].long(), chunks[:, 1:].long()
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=False)
    for name in args.models:
        out = RESULTS_DIR / f"ppl_full_{name}.json"
        if out.exists():
            print(f"skip {name} (exists)")
            continue
        t0 = time.time()
        model, desc = build(name, dev)
        metrics = evaluate_perplexity_and_bpb(model, loader, dev)
        metrics.update({"model": name, "description": desc, "chunks": int(chunks.shape[0]), "seconds": round(time.time() - t0, 1),
                        "method": "Phase 2 PretrainDataset packing: <s> doc </s> concatenated, 513-token chunks"})
        write_json(out, metrics)
        print(f"{name}: PPL {metrics['perplexity']} over {metrics['eval_tokens']:,} tokens ({metrics['seconds']}s)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
