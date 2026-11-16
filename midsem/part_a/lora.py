"""LoRA (Hu et al., 2021) written from scratch for the Phase 2 TransformerLM (no peft).

A wrapped projection computes

    h = W0 x + b0 + (alpha / r) * B (A x)

    W0, b0 : the pretrained weight (d_out x d_in) and bias of the original nn.Linear, frozen
    A      : trainable, shape (r x d_in), Kaiming-uniform init (the same init nn.Linear uses)
    B      : trainable, shape (d_out x r), initialised to zero

Because B = 0 at initialisation, B (A x) is exactly the zero vector, so a freshly wrapped model
produces bit-identical outputs to the pretrained model. A is random so that the gradient of B,
dL/dB = (alpha / r) * g (A x)^T, is non-zero from the first step.

Variant 1 adapts W_Q, W_K, W_V and W_O in every block: in hindi/model/transformer.py these are
CausalSelfAttention.q_proj, k_proj, v_proj and out_proj.
"""

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

# W_Q, W_K, W_V, W_O of CausalSelfAttention (variant 1 target layers)
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    """Frozen nn.Linear plus a trainable rank-r update (alpha / r) * B A."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        if r < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        weight = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=weight.device, dtype=weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, T, d_out). The low-rank path goes through the r-dim bottleneck
        (x A^T is (B, T, r)), so B A is never materialised during training."""
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The dense update (alpha / r) * B A, shape (d_out x d_in)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def extra_repr(self) -> str:
        return f"in={self.base.in_features}, out={self.base.out_features}, r={self.r}, alpha={self.alpha}"


def inject_lora(model: nn.Module, r: int, alpha: float, targets: Sequence[str] = ATTENTION_TARGETS) -> List[str]:
    """Freezes every parameter of the model, then wraps the target projections of every block.

    Returns the names of the wrapped modules. After this call the only trainable tensors are the
    A and B matrices of the new LoRALinear modules.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for i, block in enumerate(model.blocks):
        for name in targets:
            base = getattr(block.attn, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"blocks.{i}.attn.{name} is {type(base).__name__}, expected nn.Linear")
            setattr(block.attn, name, LoRALinear(base, r, alpha))
            wrapped.append(f"blocks.{i}.attn.{name}")
    return wrapped


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """All LoRA A / B matrices of the model (the tensors the optimiser updates)."""
    return [p for n, p in model.named_parameters() if ".lora_A" in n or ".lora_B" in n]


def parameter_counts(model: nn.Module) -> Dict[str, float]:
    """Trainable vs. total parameter count. model.parameters() yields each tensor once, so the tied
    token-embedding / LM-head matrix is counted once, as in the project's count_parameters()."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable,
            "trainable_share_pct": 100.0 * trainable / total}


def expected_lora_count(model: nn.Module, r: int, targets: Sequence[str] = ATTENTION_TARGETS) -> int:
    """Closed form r * (d_in + d_out) summed over the wrapped matrices (checks parameter_counts)."""
    total = 0
    for block in model.blocks:
        for name in targets:
            layer = getattr(block.attn, name)
            layer = layer.base if isinstance(layer, LoRALinear) else layer
            total += r * (layer.in_features + layer.out_features)
    return total


def adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Only the LoRA tensors (a few hundred KB) - the frozen base comes from the project checkpoint."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


def load_adapter_state(model: nn.Module, state: Dict[str, torch.Tensor]):
    """Loads LoRA tensors into an injected model; every LoRA tensor must be present."""
    own = {k for k in model.state_dict() if ".lora_A" in k or ".lora_B" in k}
    missing, unexpected = own - set(state), set(state) - own
    if missing or unexpected:
        raise KeyError(f"adapter mismatch: missing={sorted(missing)[:4]} unexpected={sorted(unexpected)[:4]}")
    model.load_state_dict(state, strict=False)


def max_delta_norm(model: nn.Module) -> float:
    """Largest Frobenius norm of (alpha / r) B A over the wrapped layers (0 at initialisation)."""
    norms = [m.delta_weight().norm().item() for m in model.modules() if isinstance(m, LoRALinear)]
    return max(norms) if norms else 0.0
