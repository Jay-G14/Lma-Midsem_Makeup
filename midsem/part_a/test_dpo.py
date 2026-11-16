#!/usr/bin/env python3
"""Checks for the from-scratch DPO code (runs on CPU in a few seconds).

1. sequence_logps on a padded batch equals a per-sequence teacher-forced sum over reply tokens only.
2. With policy == reference the DPO loss is exactly log 2 and every margin is exactly 0.
3. Tokens shared by the chosen and rejected replies before their first difference cancel: the
   margin computed from the full replies equals the margin from the suffixes after the shared prefix.
"""

import math

import torch
import torch.nn.functional as F

from midsem.part_a.common import DATA_DIR, load_base, set_seed, tokenizer
from midsem.part_a.dpo import collate, dpo_loss, encode_pairs, sequence_logps
from midsem.part_a.lora import inject_lora


@torch.no_grad()
def single_logp(model, seq, plen):
    x = torch.tensor([seq[:-1]])
    logp = F.log_softmax(model(x)[0][0].float(), dim=-1)
    return sum(logp[t, seq[t + 1]].item() for t in range(plen - 1, len(seq) - 1))


def main():
    dev = torch.device("cpu")
    tok = tokenizer()
    pairs = encode_pairs(DATA_DIR / "part_a_preference_pairs.jsonl", tok)[:4]
    set_seed(0)
    model = load_base(dev)
    inject_lora(model, 4, 16.0)
    for m in model.modules():                       # make B non-zero so the test is not trivial
        if hasattr(m, "lora_B"):
            torch.nn.init.normal_(m.lora_B, std=0.02)
    model.eval()

    x, y, mask = collate(pairs, dev)
    with torch.no_grad():
        batched = sequence_logps(model, x, y, mask)
    expected = [single_logp(model, p + c, len(p)) for p, c, _ in pairs] + [single_logp(model, p + r, len(p)) for p, _, r in pairs]
    err = max(abs(a - b) for a, b in zip(batched.tolist(), expected))
    print(f"1. batched vs per-sequence reply log-prob: max |diff| = {err:.2e}")
    assert err < 1e-3

    n = len(pairs)
    loss, d = dpo_loss(batched[:n], batched[n:], batched[:n].clone(), batched[n:].clone(), beta=0.5)
    print(f"2. policy == reference: loss = {loss.item():.6f} (log 2 = {math.log(2):.6f}), max |margin| = {d['margin'].abs().max().item()}")
    assert loss.item() == math.log(2) or abs(loss.item() - math.log(2)) < 1e-7
    assert d["margin"].abs().max().item() == 0.0

    # 3. shared prefix cancels: margin(full) == margin(suffix after first difference), for any two models
    ref = load_base(dev)
    inject_lora(ref, 4, 16.0)
    ref.eval()
    for p, c, r in pairs:
        k = next(i for i, (a, b) in enumerate(zip(c, r)) if a != b)
        full = (single_logp(model, p + c, len(p)) - single_logp(ref, p + c, len(p))) - \
               (single_logp(model, p + r, len(p)) - single_logp(ref, p + r, len(p)))
        pk = p + c[:k]
        suffix = (single_logp(model, pk + c[k:], len(pk)) - single_logp(ref, pk + c[k:], len(pk))) - \
                 (single_logp(model, pk + r[k:], len(pk)) - single_logp(ref, pk + r[k:], len(pk)))
        print(f"3. shared prefix {k:3d} of {len(c)} reply tokens: margin full {full:.5f} vs suffix-only {suffix:.5f}")
        assert abs(full - suffix) < 1e-3
    print("All DPO checks passed.")


if __name__ == "__main__":
    main()
