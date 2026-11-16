"""PyTorch Dataset for sequence packing JSONL corpora into fixed max_seq_len samples.

Optimized to prevent Host RAM OOM during DDP launch:
- Uses array.array('i') / int32 for compact memory representation during document ingestion.
- Converts to torch.int32 tensor and pin memory cleanly without duplicating Python list overhead.
"""
import array
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


from tqdm import tqdm


class PretrainDataset(Dataset):
    """Memory-efficient Dataset that tokenizes JSONL documents and packs tokens into max_seq_len chunks."""

    def __init__(self, jsonl_path: str, tokenizer_path: str, max_seq_len: int = 512, max_docs: int = 0):
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        path = Path(jsonl_path)
        assert path.exists(), f"Dataset file not found: {path}"

        print(f"[Dataset] Loading & tokenizing documents from {path}...")
        # Use compact C-style integer array ('i' = 4-byte signed int) instead of Python int objects
        token_stream = array.array("i")
        
        bos_id = self.tokenizer.token_to_id("<s>") or 2
        eos_id = self.tokenizer.token_to_id("</s>") or 3

        doc_count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc="[Dataset] Tokenizing JSONL"):

                if max_docs > 0 and doc_count >= max_docs:
                    break
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "").strip()
                    if text:
                        ids = self.tokenizer.encode(text).ids
                        token_stream.append(bos_id)
                        token_stream.extend(ids)
                        token_stream.append(eos_id)
                        doc_count += 1
                except Exception:
                    continue

        print(f"[Dataset] Tokenized {doc_count:,} documents into {len(token_stream):,} total tokens.")

        # Chunk token stream into non-overlapping sequences of length max_seq_len + 1
        seq_len_plus_1 = max_seq_len + 1
        num_samples = len(token_stream) // seq_len_plus_1
        usable_tokens = num_samples * seq_len_plus_1
        
        if num_samples == 0:
            raise ValueError(f"Corpus too small ({len(token_stream)} tokens) for max_seq_len={max_seq_len}")

        # Store as 2D int32 tensor of shape (num_samples, max_seq_len + 1)
        # Using int32 halves memory footprint compared to int64
        self.samples = torch.frombuffer(
            memoryview(token_stream[:usable_tokens]), dtype=torch.int32
        ).view(num_samples, seq_len_plus_1)
        print(f"[Dataset] Formed {len(self.samples):,} sequence samples of length {max_seq_len}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.samples[idx]
        x = chunk[:-1].long()  # Input sequence (0 to T-1)
        y = chunk[1:].long()   # Target sequence (1 to T)
        return x, y
