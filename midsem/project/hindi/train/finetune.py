"""Reasoning Finetuning Engine for Hindi Transformer LM (Phase 3)

Finetunes the pretrained ~25M Transformer on the synthetic comparative-reasoning dataset.

- Training targets are the reasoning chain + answer + </s>; question tokens are masked out of the loss.
- A replay buffer of pretraining text is mixed in to limit catastrophic forgetting, and pretraining
  validation perplexity is tracked every epoch.
- Checkpoints use the same resume-capable format as pretraining (model, optimizer, scheduler,
  training step, configuration) plus the finetuning hyperparameters; --resume continues from the
  last completed epoch.
- --answer-only trains the control model that sees question -> answer with no reasoning chain.

Includes tqdm progress bars.
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from hindi.model.config import TransformerConfig
from hindi.model.transformer import TransformerLM


# The model is prompted with the question followed by one of these cues and must write the rest.
REASONING_CUE = "\nतर्क:"
ANSWER_CUE = "\nउत्तर:"

LATEST_CHECKPOINT = "latest_finetune_checkpoint.pt"
BEST_CHECKPOINT = "finetuned_model.pt"


class ReasoningDataset(Dataset):
    """Dataset for JSONL synthetic reasoning prompts and pretraining replay text.

    Each item is (token_ids, prompt_len). Reasoning samples end with </s> so the model learns to
    stop after its answer, and prompt_len covers the question + cue so the loss is only taken on
    what the model has to produce. Replay samples use prompt_len 0 (plain LM loss).
    With answer_only=True the reasoning chain is dropped: question + "\\nउत्तर: <answer>".
    """
    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, max_seq_len: int = 512, pretrain_jsonl_path: str = None,
                 replay_ratio: float = 0.15, answer_only: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        eos_id = tokenizer.token_to_id("</s>")

        # Load synthetic reasoning samples
        reasoning_samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if answer_only:
                        text, prompt_text = f"{item['question']}{ANSWER_CUE} {item['answer']}", item["question"] + ANSWER_CUE
                    else:
                        text, prompt_text = item["text"], item["question"] + REASONING_CUE
                    tokens = tokenizer.encode(text).ids + [eos_id]
                    prompt = tokenizer.encode(prompt_text).ids
                    prompt_len = len(prompt) if tokens[:len(prompt)] == prompt else 0
                    if len(tokens) > max_seq_len:
                        tokens = tokens[:max_seq_len]
                    if len(tokens) > 1:
                        reasoning_samples.append((torch.tensor(tokens, dtype=torch.long), prompt_len))

        # Optionally load pretraining replay text to prevent catastrophic forgetting
        if pretrain_jsonl_path and os.path.exists(pretrain_jsonl_path) and replay_ratio > 0:
            num_replay = int(len(reasoning_samples) * replay_ratio / (1.0 - replay_ratio))
            replay_samples = []
            with open(pretrain_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        text = item.get("text", "")
                        if text:
                            tokens = tokenizer.encode(text).ids
                            if len(tokens) > max_seq_len:
                                tokens = tokens[:max_seq_len]
                            if len(tokens) > 1:
                                replay_samples.append((torch.tensor(tokens, dtype=torch.long), 0))
                            if len(replay_samples) >= num_replay:
                                break
            self.samples = reasoning_samples + replay_samples
            print(f"Replay-Buffer Dataset Loaded: {len(reasoning_samples)} reasoning samples + {len(replay_samples)} pretraining replay samples (Total: {len(self.samples)})")
        else:
            self.samples = reasoning_samples
            print(f"Standard Dataset Loaded: {len(self.samples)} reasoning samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pad_collate_fn(batch: List[Tuple[torch.Tensor, int]], pad_id: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collates (tokens, prompt_len) items into padded tensors for Causal LM (inputs=s[:-1], targets=s[1:]).

    Targets that would predict a prompt token are set to pad_id, which the loss ignores.
    """
    max_len = max(len(s) for s, _ in batch)
    seq_len = max(max_len - 1, 1)
    padded_batch = torch.full((len(batch), seq_len), fill_value=pad_id, dtype=torch.long)
    targets = torch.full((len(batch), seq_len), fill_value=pad_id, dtype=torch.long)

    for i, (s, prompt_len) in enumerate(batch):
        if len(s) > 1:
            l = len(s) - 1
            padded_batch[i, :l] = s[:-1]
            targets[i, :l] = s[1:]
            targets[i, :max(prompt_len - 1, 0)] = pad_id
        elif len(s) == 1:
            padded_batch[i, 0] = s[0]
            targets[i, 0] = s[0]

    return padded_batch, targets


def evaluate_reasoning_loss(model: TransformerLM, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluates cross-entropy loss on reasoning validation set."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Eval Reasoning Loss", leave=False):
            x, y = x.to(device), y.to(device)
            logits, loss, _ = model(x, targets=y)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(1, total_batches)


def evaluate_lm_ppl(model: TransformerLM, val_jsonl_path: str, tokenizer: Tokenizer, device: torch.device, max_samples: int = 500) -> float:
    """Evaluates Perplexity (PPL) on original pretraining validation set to monitor catastrophic forgetting."""
    model.eval()
    if not os.path.exists(val_jsonl_path):
        return 0.0

    total_loss = 0.0
    total_tokens = 0

    with open(val_jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()][:max_samples]

    with torch.no_grad():
        for line in tqdm(lines, desc="Eval Pretrain PPL", leave=False):
            doc = json.loads(line)
            tokens = tokenizer.encode(doc["text"]).ids
            if len(tokens) < 2:
                continue
            if len(tokens) > 512:
                tokens = tokens[:512]

            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            logits, loss, _ = model(x, targets=y)
            total_loss += loss.item() * len(tokens[1:])
            total_tokens += len(tokens[1:])

    avg_loss = total_loss / max(1, total_tokens)
    return math.exp(avg_loss)


def set_seed(seed: int):
    """Seeds Python and PyTorch RNGs (data order, dropout) for a reproducible run."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, model: TransformerLM, optimizer, scheduler, config: TransformerConfig,
                    hparams: Dict, step: int, epoch: int, best_val_loss: float, history: List[Dict]):
    """Saves a resume-capable checkpoint (same keys as pretraining: model / optimizer / scheduler / step / config)."""
    last = history[-1]
    torch.save({
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.__dict__,
        "hparams": hparams,
        "val_reasoning_loss": last["val_reasoning_loss"],
        "val_lm_ppl": last["pretrain_ppl"],
        "best_val_loss": best_val_loss,
        "history": history,
    }, path)


def write_json(path: str, obj: Dict):
    """Writes a JSON file (hyperparameters / history) next to the checkpoints."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def finetune(args):
    """Runs (or resumes) reasoning finetuning, saving the best epoch by reasoning validation loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hindi Reasoning Finetuning] Device: {device}")
    hparams = {k: v for k, v in vars(args).items() if k != "resume"}
    hparams.update({"optimizer": "AdamW", "weight_decay": 0.01, "grad_clip": 1.0, "lr_schedule": "constant"})
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "finetune_config.json"), hparams)
    set_seed(args.seed)

    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>") if tokenizer.token_to_id("<pad>") is not None else 1

    # 2. Load Pretrained Model Checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        config = TransformerConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
    else:
        config = TransformerConfig.load(args.config_path)

    model = TransformerLM(config).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded pretrained checkpoint from: {args.model_path}")

    # 3. Datasets & DataLoaders
    train_dataset = ReasoningDataset(
        args.train_data,
        tokenizer,
        max_seq_len=config.max_seq_len,
        pretrain_jsonl_path=args.pretrain_train_data,
        replay_ratio=args.replay_ratio,
        answer_only=args.answer_only
    )
    val_dataset = ReasoningDataset(args.val_data, tokenizer, max_seq_len=config.max_seq_len, answer_only=args.answer_only)

    collate = lambda batch: pad_collate_fn(batch, pad_id=pad_id)
    data_rng = torch.Generator()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=data_rng)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 4. Optimizer & (constant) learning-rate schedule, kept as a scheduler so its state is checkpointed
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    # 5. Resume or evaluate the zero-shot baseline
    start_epoch, step, best_val_loss, history = 1, 0, float("inf"), []
    latest_path = os.path.join(args.output_dir, LATEST_CHECKPOINT)
    if args.resume and os.path.exists(latest_path):
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, step = state["epoch"] + 1, state["step"]
        best_val_loss, history = state["best_val_loss"], state["history"]
        print(f"[Resume] Resumed from epoch {state['epoch']} (step {step}) in {latest_path}")
    else:
        initial_reasoning_loss = evaluate_reasoning_loss(model, val_loader, device)
        initial_lm_ppl = evaluate_lm_ppl(model, args.pretrain_val_data, tokenizer, device)
        history.append({"epoch": 0, "train_loss": None, "val_reasoning_loss": initial_reasoning_loss, "pretrain_ppl": initial_lm_ppl})
        print(f"Initial Baseline -> Reasoning Loss: {initial_reasoning_loss:.4f} | Pretrain LM PPL: {initial_lm_ppl:.2f}")

    # 6. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Data order and dropout depend only on (seed, epoch), so a resumed run replays the same epochs.
        torch.manual_seed(args.seed + epoch)
        data_rng.manual_seed(args.seed + epoch)
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss, _ = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        val_reasoning_loss = evaluate_reasoning_loss(model, val_loader, device)
        val_lm_ppl = evaluate_lm_ppl(model, args.pretrain_val_data, tokenizer, device)
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_reasoning_loss": val_reasoning_loss, "pretrain_ppl": val_lm_ppl})

        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {avg_train_loss:.4f} | Reasoning Val Loss: {val_reasoning_loss:.4f} | Pretrain LM PPL: {val_lm_ppl:.2f}")

        # Save best model
        if val_reasoning_loss < best_val_loss:
            best_val_loss = val_reasoning_loss
            save_path = os.path.join(args.output_dir, BEST_CHECKPOINT)
            save_checkpoint(save_path, model, optimizer, scheduler, config, hparams, step, epoch, best_val_loss, history)
            print(f"Saved best finetuned model to: {save_path}")

        # Save the resume point and the training history
        save_checkpoint(latest_path, model, optimizer, scheduler, config, hparams, step, epoch, best_val_loss, history)
        best_epoch = min((h for h in history if h["epoch"] > 0), key=lambda h: h["val_reasoning_loss"])["epoch"]
        write_json(os.path.join(args.output_dir, "finetune_history.json"),
                   {"hparams": hparams, "best_epoch": best_epoch, "epochs": history})

    print("Reasoning Finetuning Complete!")


def main():
    parser = argparse.ArgumentParser(description="Hindi Transformer LM Reasoning Finetune Script")
    parser.add_argument("--model-path", type=str, default="hindi/checkpoints/final_model.pt", help="Pretrained model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, default="hindi/tokenizer/tokenizer.json", help="Tokenizer path")
    parser.add_argument("--config-path", type=str, default="hindi/configs/model_config.json", help="Model config path")
    parser.add_argument("--train-data", type=str, default="hindi/data/reasoning/train.jsonl", help="Synthetic reasoning train data")
    parser.add_argument("--val-data", type=str, default="hindi/data/reasoning/val.jsonl", help="Synthetic reasoning val data")
    parser.add_argument("--pretrain-train-data", type=str, default="hindi/data/processed/train.jsonl", help="Pretraining train data for replay buffer")
    parser.add_argument("--pretrain-val-data", type=str, default="hindi/data/processed/val.jsonl", help="Original pretraining val data")
    parser.add_argument("--replay-ratio", type=float, default=0.15, help="Ratio of pretraining replay data in dataset (default 0.15)")
    parser.add_argument("--output-dir", type=str, default="hindi/checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of finetuning epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (constant)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (data order, dropout)")
    parser.add_argument("--answer-only", action="store_true", help="Control run: train question -> answer without the reasoning chain")
    parser.add_argument("--resume", action="store_true", help="Resume from <output-dir>/latest_finetune_checkpoint.pt")
    args = parser.parse_args()

    finetune(args)


if __name__ == "__main__":
    main()
