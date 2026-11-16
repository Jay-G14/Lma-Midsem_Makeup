"""From-scratch PyTorch Decoder-only Transformer implementation.

Fulfills all Phase 2 architectural constraints:
- Built strictly with primitive PyTorch layers (nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout).
- Zero dependencies on nn.Transformer* or HuggingFace model classes.
- Explicit tensor reshapes for Multi-Head Causal Attention and scaled dot-product.
- Pre-LayerNorm architecture for deep stability.
- Weight-tying option between input embedding and output projection head.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hindi.model.config import TransformerConfig


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention block from first principles."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head
        self.d_model = config.d_model

        # Projection matrices for Query, Key, Value
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Upper triangular causal mask: 0 on allowed positions, -inf on future positions
        mask = torch.triu(torch.full((config.max_seq_len, config.max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, T, d_model)
        Returns:
            output: Tensor of shape (B, T, d_model)
            attn_weights: Tensor of shape (B, h, T, T) for visualization
        """
        B, T, C = x.size()

        # 1. Project inputs into Queries, Keys, Values: (B, T, C) -> (B, T, C)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into h heads: (B, T, C) -> (B, h, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention: Q @ K^T / sqrt(d_k)
        # (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. Apply additive Causal Masking M
        mask = self.causal_mask[:T, :T]
        scores = scores + mask

        # 5. Softmax & Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_weights)

        # 6. Weighted sum of Values: (B, h, T, T) @ (B, h, T, d_k) -> (B, h, T, d_k)
        out = attn_probs @ v

        # 7. Concatenate heads back: (B, h, T, d_k) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output projection
        out = self.resid_dropout(self.out_proj(out))
        return out, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Decoder Transformer Block with residual connections."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = PositionwiseFeedForward(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-Norm Causal Attention sublayer + Residual
        normed_x = self.ln1(x)
        attn_out, attn_weights = self.attn(normed_x)
        x = x + attn_out

        # Pre-Norm Feedforward sublayer + Residual
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class TransformerLM(nn.Module):
    """Decoder-only GPT Transformer Language Model (~25M parameters)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Input representation
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output LM projection head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optionally tie weights between embedding and output head
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights cleanly
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """Forward pass.
        
        Args:
            input_ids: (B, T) token indices
            targets: (B, T) target token indices for causal LM loss calculation
            return_attention: Whether to collect attention weight matrices per block
            
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar CrossEntropy loss (if targets provided)
            attentions: list of (B, h, T, T) attention tensors per layer
        """
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        x = self.token_embedding(input_ids)
        if self.config.use_pos_emb:
            positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            x = x + self.pos_embedding(positions)
        x = self.emb_dropout(x)

        # Pass through Transformer blocks
        attentions = [] if return_attention else None
        for block in self.blocks:
            x, attn_weights = block(x)
            if return_attention:
                attentions.append(attn_weights)

        # Final LayerNorm
        x = self.ln_f(x)

        # Next-token logits
        logits = self.lm_head(x)

        # Calculate causal cross-entropy loss if targets supplied
        loss = None
        if targets is not None:
            # Flatten B and T to compute cross entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_token_id
            )

        return logits, loss, attentions

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Autoregressively generates new token IDs using greedy or sampling decoding.
        
        Args:
            input_ids: (B, T) initial prompt token indices
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_k: Top-k filtering threshold
            top_p: Nucleus top-p filtering threshold
            eos_token_id: Optional EOS token ID for early stopping
            
        Returns:
            input_ids: (B, T + max_new_tokens) complete generated sequence indices
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if temperature == 0.0 or temperature < 1e-5:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return input_ids


    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self.config.tie_weights:
            # Weight tying shares token_embedding and lm_head, so don't double count
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

