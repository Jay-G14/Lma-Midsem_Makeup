#!/bin/bash
# Builds midsem/<rollno>_alternativemidsem.zip for Moodle. The zip holds the report PDF, the README and the whole
# midsem/ folder (code, project code and Phase 3 data copied into midsem/project/, deliverable JSONL files, results,
# figures, logs, the chosen adapters). Left out because of size: the Phase 2/3 checkpoints and the pretraining text in
# midsem/project/hindi/{checkpoints,data/processed}/ (README: where to get them) and the bulk of midsem/checkpoints*/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=midsem/2023113001_alternativemidsem.zip
rm -f "$OUT"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/2023113001_alternativemidsem"
mkdir -p "$DEST"

# Report (PDF at the top level, LaTeX source with the code) and README
cp midsem/report/report.pdf "$DEST/2023113001_alternativemidsem_report.pdf"
cp midsem/README.md "$DEST/README.md"

# Code, copied project files, deliverable JSONL files, results, figures, logs
rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '/README.md' --exclude '*.zip' --exclude 'LMA_MissedMidSem_Assgn.pdf' \
  --exclude 'report/*.aux' --exclude 'report/*.log' --exclude 'report/*.out' --exclude 'report/report.pdf' \
  --exclude '/checkpoints' --exclude '/checkpoints_ada' --exclude '/drive_upload' \
  --exclude '/project/hindi/checkpoints/*.pt' --exclude '/project/hindi/data/processed' \
  midsem/ "$DEST/midsem/"

# First-pass LoRA / DPO adapters (A and B matrices only; latest_adapter.pt is a resume checkpoint with optimiser state)
for d in lora_r1 lora_r4 lora_r16 dpo_beta0.1 dpo_beta0.5; do
  mkdir -p "$DEST/midsem/checkpoints/$d"
  for f in best_adapter.pt adapter.pt history.json; do
    if [ -f "midsem/checkpoints/$d/$f" ]; then cp "midsem/checkpoints/$d/$f" "$DEST/midsem/checkpoints/$d/"; fi
  done
done

# Ada study: the best LoRA model and, per beta, the seed-42 DPO policy at the chosen (LR, step)
if [ -f midsem/results_ada/summary.json ]; then
  python3 - "$DEST" << 'PY'
import json, shutil, sys
from pathlib import Path
dest = Path(sys.argv[1]) / "midsem/checkpoints_ada"
src = Path("midsem/checkpoints_ada")
sel = json.loads(Path("midsem/results_ada/summary.json").read_text())["selections"]
runs = [(sel["best_lora"], "best_adapter.pt")] if sel.get("best_lora") else []
for b in ("0.1", "0.5"):
    c = sel.get(f"dpo_b{b}")
    if c:
        runs.append((f"ada_dpo_b{b}_lr{c['lr']}_s42_step{c['step']}", "adapter.pt"))
for name, fname in runs:
    if (src / name / fname).exists():
        (dest / name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name / fname, dest / name / fname)
        print("included", name)
PY
fi

(cd "$STAGE" && zip -qr9 "$OLDPWD/$OUT" 2023113001_alternativemidsem)
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
unzip -l "$OUT" | tail -1
