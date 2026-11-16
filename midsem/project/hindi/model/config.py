"""Model configuration for ~25M parameter Hindi Decoder-only Transformer."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class TransformerConfig:
    vocab_size: int = 20000
    max_seq_len: int = 512
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True
    pad_token_id: int = 1
    use_pos_emb: bool = True
    
    @property
    def d_k(self) -> int:
        return self.d_model // self.n_head

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TransformerConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
