#!/bin/bash
# CL3410 make-up mid-sem, extended Part A study (multi-seed LoRA, DPO LR sweep, full-split drift PPL)
# and the fp32 re-judging of the C2 pairs, on one Ada node with 4 GPUs.
# Based on ~/.claude/templates/slurm_ada_template.sh (user jayant.g). Submit from the experiment folder:
#     cd ~/lma-midsem && sbatch scripts/run.sh
# Data policy: inputs are downloaded from the private HF dataset repo GJay098/lma-midsem-assets to the
# node-local /scratch (checked against SHA256SUMS) and removed on exit; only code, the venv, logs and
# to_transfer/ live in the experiment folder in home. Results of value go to to_transfer/.
# Experiment folder = ~/lma-midsem/midsem (code only) + scripts/run.sh, synced without data or checkpoints:
#     rsync -a --exclude .venv --exclude '__pycache__' --exclude 'project/hindi/checkpoints' --exclude 'project/hindi/data' \
#           --exclude 'project/hindi/tokenizer' --exclude 'checkpoints*' --exclude '*.zip' midsem/ <ada>:lma-midsem/midsem/
# The staged inputs are linked in below: project files (hindi/...) into midsem/project/, the rest into midsem/.
#SBATCH --job-name=lma_midsem
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 36
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --prefer=2080ti
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=jayant.g@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=lma_midsem.log
#SBATCH --error=lma_midsem.err
# Fresh .log/.err on every submission; only an automatic GPU-count resubmission appends.
#SBATCH --open-mode=truncate

set -e  # stop the whole job if any stage fails

NUM_WORKERS=8  # dataloader workers per GPU rank (36 CPUs / 4 ranks); the tasks below load data in-process

echo "Job started on $(hostname) at $(date)"

# $SLURM_SUBMIT_DIR, not ${BASH_SOURCE[0]}: SLURM runs a copy of this script from /var/spool/slurmd/.
BASE_DIR="$SLURM_SUBMIT_DIR"
if [ -z "$BASE_DIR" ]; then BASE_DIR=$(pwd); fi
if [ "$(basename "$BASE_DIR")" = "scripts" ]; then BASE_DIR="$(dirname "$BASE_DIR")"; fi
cd "$BASE_DIR" || { echo "Error: Could not cd into $BASE_DIR"; exit 1; }
echo ">>> Running from $BASE_DIR"

# GPUs on this node, for the log. Any model is accepted (GTX 1080, RTX 2080 / 2080 Ti).
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
echo "GPUs on $(hostname): $(echo "$GPU_NAMES" | sort | uniq -c | tr -s ' \n' ' ')"

# Use the NODE's system Python. sbatch exports the submitting shell's PATH, which puts ~/miniconda3 first;
# its Python 3.14 has no torch cu121 wheels (first attempt, job 84118, failed on exactly that).
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/miniconda3/' | paste -sd: -)
hash -r
echo ">>> node python: $(command -v python3) ($(python3 --version 2>&1))"

# venv in the experiment folder in home, built here inside the job (login nodes: 1 CPU / 2 GB, no compilers).
echo ">>> Setting up environment..."
NODE_PY=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ -x .venv/bin/python ] && [ "$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" != "$NODE_PY" ]; then
    echo ">>> .venv was built for a different Python; rebuilding it for Python $NODE_PY"
    rm -rf .venv
fi
if [ -d ".venv" ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi   # leftover of a failed build
if [ ! -d ".venv" ]; then
    python3 -m venv .venv || { rm -rf .venv; python3 -m venv --without-pip .venv && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python; }
fi
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # cu121 builds still support GTX 1080
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
mkdir -p to_transfer
python3 -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

# One source of truth for the GPU count: the --gres line above. Trust what torch sees.
NUM_GPUS=$(grep -oP '(?<=--gres=gpu:)[0-9]+' "${BASH_SOURCE[0]}" | head -1)
ACTUAL_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "NUM_GPUS wanted = $NUM_GPUS, torch.cuda.device_count() actually sees = $ACTUAL_GPUS"
if [ "$ACTUAL_GPUS" -lt "$NUM_GPUS" ]; then
    echo "ERROR: wanted $NUM_GPUS GPUs but $(hostname -s) only has $ACTUAL_GPUS visible to this job (Slurm may still"
    echo "list it as a 4-GPU node: a GPU that fails at runtime is not visible to Slurm)."
    ATTEMPT=${RESUBMIT_ATTEMPT:-0}
    MAX_RESUBMITS=3
    PREV_EXC=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -oP 'ExcNodeList=\K\S+' | grep -v '(null)' || true)
    EXC="${PREV_EXC:+$PREV_EXC,}$(hostname -s)"
    if [ -n "$SLURM_JOB_ID" ] && [ "$ATTEMPT" -lt "$MAX_RESUBMITS" ]; then
        # RESUBMIT_EXTRA: extra sbatch options the first submission used (e.g. a --time / --deadline), kept on resubmission
        NEW_JOB=$(RESUBMIT_ATTEMPT=$((ATTEMPT + 1)) sbatch --parsable --open-mode=append --exclude="$EXC" ${RESUBMIT_EXTRA:-} scripts/run.sh) || NEW_JOB=""
        if [ -n "$NEW_JOB" ]; then
            echo ">>> Resubmitted as job $NEW_JOB (attempt $((ATTEMPT + 1)) of $MAX_RESUBMITS), excluding: $EXC"
        else
            echo ">>> Automatic resubmission failed. Resubmit by hand: sbatch --exclude=$EXC scripts/run.sh"
        fi
    else
        echo ">>> Not resubmitting (attempt $ATTEMPT of $MAX_RESUBMITS). Resubmit by hand: sbatch --exclude=$EXC scripts/run.sh"
    fi
    exit 1
fi

# Stage data on the node's /scratch: downloaded from the private HF repo, removed on exit (success or failure).
# The experiment folder only gets symlinks to it (listed in .scratch_links, removed by the trap).
SCRATCH_DATA="/scratch/$USER/lma_midsem_data_${SLURM_JOB_ID:-manual}"
mkdir -p "$SCRATCH_DATA" || { echo "ERROR: /scratch/$USER not writable on $(hostname)"; exit 1; }
LINKS="$BASE_DIR/.scratch_links"
cleanup() {
    if [ -f "$LINKS" ]; then while read -r l; do [ -L "$l" ] && rm -f "$l"; done < "$LINKS"; rm -f "$LINKS"; fi
    rm -rf "$SCRATCH_DATA"
}
trap cleanup EXIT
export HF_HOME="$SCRATCH_DATA/hf_home"
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=$(cat "$BASE_DIR/.hf_token")   # chmod 600, read-only use: download from the private assets repo
ASSETS="$SCRATCH_DATA/assets"
python3 - <<PYDL
from huggingface_hub import snapshot_download
snapshot_download("GJay098/lma-midsem-assets", repo_type="dataset", local_dir="$ASSETS")
PYDL

# Integrity check: fail in seconds if anything is missing or corrupt, not hours into training.
( cd "$ASSETS" && sha256sum -c --quiet SHA256SUMS ) || { echo "ERROR: SHA256 check of the staged assets failed"; exit 1; }
echo "assets staged to $ASSETS ($(du -sh "$ASSETS" | cut -f1)), SHA256 verified for $(wc -l < "$ASSETS/SHA256SUMS") files"

# Outputs live in to_transfer/; the code reads/writes them through these links.
mkdir -p to_transfer/results to_transfer/logs to_transfer/data to_transfer/data_ada to_transfer/checkpoints_midsem
mkdir -p midsem
for pair in "midsem/results_ada:to_transfer/results" "midsem/logs_ada:to_transfer/logs" "midsem/data:to_transfer/data" \
            "midsem/data_ada:to_transfer/data_ada" "midsem/checkpoints_ada:to_transfer/checkpoints_midsem"; do
    link="${pair%%:*}"; target="$BASE_DIR/${pair##*:}"
    if [ ! -L "$link" ]; then rm -rf "$link"; ln -s "$target" "$link"; fi
done
# Inputs: symlink every staged file into place.
( cd "$ASSETS" && find . -type f ! -name SHA256SUMS ! -name .gitattributes ! -path './.cache/*' ) | while read -r f; do
    f="${f#./}"; dest="$BASE_DIR/$f"
    case "$f" in hindi/*) dest="$BASE_DIR/midsem/project/$f" ;; esac   # the project's inputs live in midsem/project/
    mkdir -p "$(dirname "$dest")"
    ln -sfn "$ASSETS/$f" "$dest"; echo "$dest" >> "$LINKS"
done

# Download the judge's pretrained weights ONCE, before any worker starts (to the /scratch HF cache).
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Llama-3.2-3B-Instruct', allow_patterns=['*.json', '*.safetensors', '*.jinja', 'tokenizer*'])"

# JUDGE_ONLY=1 (sbatch --export=ALL,JUDGE_ONLY=1 scripts/run.sh): skip the Part A study and (re-)judge the
# current C2 pairs in fp32, e.g. after the C1 replies changed.
JUDGE_ONLY="${JUDGE_ONLY:-0}"

# Pipeline: the DAG scheduler keeps all GPUs busy (one task per GPU, each with CUDA_VISIBLE_DEVICES set).
if [ "$JUDGE_ONLY" != "1" ]; then
    PYTHONPATH="$BASE_DIR" python3 midsem/ada/orchestrate.py --gpus "$NUM_GPUS" --ppl-cache "$SCRATCH_DATA/val_packed_513.pt" \
        || echo "WARNING: orchestrator reported failed tasks (see to_transfer/logs/ada/)"
fi

# fp32 re-judging of the C2 pairs (the model is spread over all GPUs).
if [ -f midsem/data/c2_pairs_with_verdicts.jsonl ] && { [ "$JUDGE_ONLY" = "1" ] || [ ! -f to_transfer/results/c2_results_fp32.json ]; }; then
    MIDSEM_STUDY=1 PYTHONPATH="$BASE_DIR" python3 midsem/part_c/c2_judge.py --judge unsloth/Llama-3.2-3B-Instruct \
        --pairs-in midsem/data/c2_pairs_with_verdicts.jsonl --dtype float32 --device-map auto --tag _fp32 \
        > to_transfer/logs/c2_judge_fp32.log 2>&1 || echo "WARNING: fp32 judge failed (see to_transfer/logs/c2_judge_fp32.log)"
fi

echo ">>> Pipeline complete at $(date). Results in to_transfer/"
