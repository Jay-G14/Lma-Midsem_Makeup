#!/usr/bin/env python3
"""Uploads the Part A inputs to a PRIVATE Hugging Face dataset repo. The Ada job downloads them to the
node-local /scratch (data never lives in the cluster home), checks every SHA-256, and links them into
the experiment folder for the duration of the job.

Usage (from the repository root; the token is read from the environment, never printed):
  HF_TOKEN=... python3 midsem/ada/upload_assets.py --repo <user>/lma-midsem-assets [--only midsem/data/c2_pairs_with_verdicts.jsonl]
"""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

MIDSEM = Path(__file__).resolve().parents[1]
ROOT = MIDSEM.parent
# path in the HF repo -> how the local file is made (None = as is). Project inputs (hindi/...) are read from the
# copies in midsem/project/, everything else from the same path under the repository root.
FILES = {
    "hindi/checkpoints/final_model.pt": None,
    "hindi/checkpoints/finetuned_model.pt": None,
    "hindi/tokenizer/tokenizer.json": None,
    "hindi/data/reasoning/train.jsonl": None,
    "hindi/data/reasoning/val.jsonl": None,
    "hindi/data/reasoning/test.jsonl": None,
    "hindi/data/processed/val.jsonl": None,
    "hindi/data/processed/train.jsonl": "head3000",   # the replay buffer only reads the first 2,117 documents
    "midsem/data/test_seen_entities.jsonl": None,
    "midsem/data/test_unseen_wording.jsonl": None,
    "midsem/data/part_a_preference_pairs.jsonl": None,
    "midsem/data/part_a_preference_pairs_val.jsonl": None,
}


def local_path(repo_path: str) -> Path:
    return MIDSEM / "project" / repo_path if repo_path.startswith("hindi/") else ROOT / repo_path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description="Upload Part A inputs to a private HF dataset repo")
    p.add_argument("--repo", required=True)
    p.add_argument("--only", nargs="*", default=None, help="upload just these repo paths (local path = same path)")
    args = p.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    files = {k: v for k, v in FILES.items()} if not args.only else {k: None for k in args.only}
    tmpdir = Path(tempfile.mkdtemp())
    sums = {}
    for repo_path, src in files.items():
        local = local_path(repo_path)
        if src == "head3000":
            local = tmpdir / "train_head3000.jsonl"
            with open(local_path(repo_path), encoding="utf-8") as fin, open(local, "w", encoding="utf-8") as fout:
                for i, line in enumerate(fin):
                    if i >= 3000:
                        break
                    fout.write(line)
        sums[repo_path] = sha256(local)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=repo_path, repo_id=args.repo, repo_type="dataset")
        print(f"uploaded {repo_path} ({local.stat().st_size / 2**20:.1f} MB)", flush=True)
    # merge with the existing checksum file so partial uploads keep one complete SHA256SUMS
    existing = {}
    try:
        existing_path = api.hf_hub_download(args.repo, "SHA256SUMS", repo_type="dataset")
        existing = dict(reversed(line.split("  ", 1)) for line in Path(existing_path).read_text().splitlines() if line.strip())
        existing = {k.strip(): v for k, v in existing.items()}
    except Exception:
        pass
    existing.update(sums)
    text = "".join(f"{h}  {path}\n" for path, h in sorted(existing.items()))
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo="SHA256SUMS", repo_id=args.repo, repo_type="dataset")
    print("SHA256SUMS updated:", len(existing), "files")


if __name__ == "__main__":
    main()
