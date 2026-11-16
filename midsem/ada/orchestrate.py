#!/usr/bin/env python3
"""Extended Part A study on one Ada node: a small DAG scheduler that keeps every GPU busy.

Protocol (all selections use the VALIDATION split only; test sets are only scored):
  1. SFT LR grid      r in {1, 4, 16} x LR in {3e-4, 1e-3, 3e-3}, seed 42, 5 epochs (Phase 3 recipe).
  2. LR per rank      lowest best-epoch reasoning-validation loss over the grid.
  3. SFT seeds        seeds 43 and 44 at each rank's chosen LR  -> 3 seeds per rank.
  4. Evaluation       every SFT run: validation + the three test sets (+ 500-document drift PPL);
                      the Phase 3 full finetune is re-scored on the same hardware.
  5. Best LoRA        highest validation answer EM over the 9 main runs (3 ranks x 3 seeds).
  6. DPO sweep        beta in {0.1, 0.5} x LR in {1e-5, 3e-5, 1e-4}, seed 42, 2 epochs (500 steps),
                      policy saved every 100 steps; every saved policy is scored on the validation split.
  7. DPO choice       per beta: the (LR, step) with the highest validation answer EM (early stopping).
  8. DPO seeds        seeds 43 and 44 trained to the chosen (LR, step) for each beta.
  9. DPO test scores  chosen configuration x 3 seeds, plus the original fixed protocol (LR 1e-4, 500 steps).
 10. Drift            full-validation-split PPL (Phase 2 packing) for base, full FT, the 9 main LoRA runs
                      and every DPO model scored on the test sets.
 (11. The fp32 re-judging of the C2 pairs runs after this scheduler, from scripts/run.sh.)

Every task is idempotent (it is skipped when its output file exists), so a resubmitted job resumes.
Run from the experiment root:  PYTHONPATH=. python3 midsem/ada/orchestrate.py --gpus 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
os.environ["MIDSEM_STUDY"] = "1"          # every task writes to results_ada/, checkpoints_ada/, logs_ada/ (common.py)
RES = ROOT / "midsem" / "results_ada"
CKPT = ROOT / "midsem" / "checkpoints_ada"
LOGS = ROOT / "midsem" / "logs_ada" / "ada"

RANKS = [1, 4, 16]
SFT_LRS = ["3e-4", "1e-3", "3e-3"]
SEEDS = [42, 43, 44]
BETAS = ["0.1", "0.5"]
DPO_LRS = ["1e-5", "3e-5", "1e-4"]
DPO_STEPS = [100, 200, 300, 400, 500]
PY = sys.executable


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sft_tag(r, lr, seed):
    return f"ada_r{r}_lr{lr}_s{seed}"


def dpo_tag(beta, lr, seed, upto=None):
    return f"ada_dpo_b{beta}_lr{lr}_s{seed}" + (f"_upto{upto}" if upto else "")


TEST_SETS = {"seen_entities", "unseen_entities", "unseen_wording"}


@dataclass
class Task:
    tid: str
    cmd: List[str]
    output: Path
    deps: List[str] = field(default_factory=list)
    priority: int = 50          # lower runs first
    gpu: bool = True
    needs_test_sets: bool = False   # an eval file only counts as done if it holds the three test sets

    def done(self) -> bool:
        if not self.output.exists():
            return False
        if self.needs_test_sets:
            d = read(self.output)
            return d is not None and set(d.get("sets", {})) >= TEST_SETS
        return True


# ---------------------------------------------------------------------------------------------
# Selections (pure functions of the result files; None while their inputs are incomplete)
# ---------------------------------------------------------------------------------------------

def frozen(name: str, compute: Callable):
    """A selection is made once and then frozen in results/ada_frozen_<name>.json, so a later re-evaluation
    (e.g. the full test scoring of a chosen checkpoint, which also re-scores validation) can never change it."""
    path = RES / f"ada_frozen_{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["value"]
    value = compute()
    if value is not None:
        path.write_text(json.dumps({"value": value}, indent=2), encoding="utf-8")
    return value


def lr_choice(r) -> Optional[str]:
    def compute():
        hist = {lr: read(RES / f"{sft_tag(r, lr, 42)}_history.json") for lr in SFT_LRS}
        if any(h is None for h in hist.values()):
            return None
        return min(SFT_LRS, key=lambda lr: hist[lr]["best_val_reasoning_loss"])
    return frozen(f"lr_r{r}", compute)


def main_runs() -> Optional[List[str]]:
    choice = {r: lr_choice(r) for r in RANKS}
    if any(v is None for v in choice.values()):
        return None
    return [sft_tag(r, choice[r], s) for r in RANKS for s in SEEDS]


def best_lora() -> Optional[str]:
    def compute():
        runs = main_runs()
        if runs is None:
            return None
        evals = {t: read(RES / f"eval_{t}.json") for t in runs}
        hists = {t: read(RES / f"{t}_history.json") for t in runs}
        if any(v is None or set(v.get("sets", {})) < TEST_SETS for v in evals.values()) or any(v is None for v in hists.values()):
            return None
        return max(runs, key=lambda t: (evals[t]["sets"]["val"]["answer_em"], -hists[t]["best_val_reasoning_loss"]))
    return frozen("best_lora", compute)


def dpo_choice(beta) -> Optional[Dict]:
    def compute():
        cands = []
        for lr in DPO_LRS:
            for step in DPO_STEPS:
                ev = read(RES / f"eval_{dpo_tag(beta, lr, 42)}_step{step}.json")
                if ev is None:
                    return None
                cands.append({"lr": lr, "step": step, "val_answer_em": ev["sets"]["val"]["answer_em"]})
        best = max(cands, key=lambda c: (c["val_answer_em"], -c["step"]))      # ties: fewer steps (closer to the reference)
        return {**best, "candidates": cands}
    return frozen(f"dpo_b{beta}", compute)


def write_selections():
    out = {"lr_per_rank": {str(r): lr_choice(r) for r in RANKS}, "main_runs": main_runs(), "best_lora": best_lora(),
           "dpo_choice": {b: dpo_choice(b) for b in BETAS}}
    (RES / "ada_selections.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Task plan (recomputed every scheduler tick, so tasks appear as soon as their selection is known)
# ---------------------------------------------------------------------------------------------

def plan(args) -> List[Task]:
    tasks: List[Task] = []
    ev = lambda name, sets=None, prio=40, deps=(): Task(
        f"eval:{name}" + (":val" if sets == ["val"] else ""),
        [PY, "midsem/part_a/evaluate_models.py", "--models", name] + (["--sets", *sets] if sets else []),
        RES / f"eval_{name}.json", list(deps), prio, needs_test_sets=sets is None)
    ppl = lambda name, deps=(): Task(f"ppl:{name}", [PY, "midsem/part_a/ppl_full.py", "--models", name, "--cache", args.ppl_cache],
                                     RES / f"ppl_full_{name}.json", ["pack", *deps], 70)

    # packing the validation split for the full-split PPL (CPU)
    tasks.append(Task("pack", [PY, "-c", f"from midsem.part_a.ppl_full import packed_chunks; packed_chunks({args.ppl_cache!r})"],
                      Path(args.ppl_cache), [], 0, gpu=False))

    # 1-3. SFT grid and seeds
    for r in RANKS:
        for lr in SFT_LRS:
            t = sft_tag(r, lr, 42)
            tasks.append(Task(f"sft:{t}", [PY, "midsem/part_a/lora_sft.py", "--rank", str(r), "--lr", lr, "--epochs", "5",
                                           "--seed", "42", "--resume", "--tag", t, "--output-dir", str(CKPT / t)],
                              RES / f"{t}_history.json", [], 10 + RANKS.index(r)))
        lr = lr_choice(r)
        if lr:
            for s in SEEDS[1:]:
                t = sft_tag(r, lr, s)
                tasks.append(Task(f"sft:{t}", [PY, "midsem/part_a/lora_sft.py", "--rank", str(r), "--lr", lr, "--epochs", "5",
                                               "--seed", str(s), "--resume", "--tag", t, "--output-dir", str(CKPT / t)],
                                  RES / f"{t}_history.json", [], 15))

    # 4. evaluation of every SFT run (main runs first), and the full finetune on the same hardware
    main = set(main_runs() or [])
    for r in RANKS:
        for lr in SFT_LRS:
            for s in SEEDS:
                t = sft_tag(r, lr, s)
                if (RES / f"{t}_history.json").exists():
                    tasks.append(ev(t, prio=20 if t in main else 60, deps=[f"sft:{t}"]))
    tasks.append(ev("full_ft", prio=65))

    # 6. DPO sweep from the best LoRA model
    best = best_lora()
    if best:
        init = str(CKPT / best / "best_adapter.pt")
        for beta in BETAS:
            for lr in DPO_LRS:
                t = dpo_tag(beta, lr, 42)
                tasks.append(Task(f"dpo:{t}", [PY, "midsem/part_a/dpo.py", "--beta", beta, "--lr", lr, "--epochs", "2", "--seed", "42",
                                               "--tag", t, "--init-adapter", init, "--save-steps", ",".join(map(str, DPO_STEPS))],
                                  RES / f"{t}_history.json", [], 25))
                if (RES / f"{t}_history.json").exists():
                    for step in DPO_STEPS:
                        tasks.append(ev(f"{t}_step{step}", sets=["val"], prio=30, deps=[f"dpo:{t}"]))
            # 7-9. seeds at the chosen configuration, test scores, fixed protocol
            choice = dpo_choice(beta)
            if choice:
                lr, step = choice["lr"], choice["step"]
                tasks.append(ev(f"{dpo_tag(beta, lr, 42)}_step{step}", prio=35))
                for s in SEEDS[1:]:
                    t = dpo_tag(beta, lr, s, upto=step)
                    tasks.append(Task(f"dpo:{t}", [PY, "midsem/part_a/dpo.py", "--beta", beta, "--lr", lr, "--epochs", "2", "--seed", str(s),
                                                   "--tag", t, "--init-adapter", init, "--max-steps", str(step)],
                                      RES / f"{t}_history.json", [], 26))
                    if (RES / f"{t}_history.json").exists():
                        tasks.append(ev(t, prio=35, deps=[f"dpo:{t}"]))
            fixed = f"{dpo_tag(beta, '1e-4', 42)}_step500"
            if (RES / f"{dpo_tag(beta, '1e-4', 42)}_history.json").exists():
                tasks.append(ev(fixed, prio=45))

    # 10. drift on the full validation split
    tasks += [ppl("base"), ppl("full_ft")]
    for t in main:
        if (RES / f"{t}_history.json").exists():
            tasks.append(ppl(t))
    for path in sorted(RES.glob("eval_ada_dpo_*.json")):
        d = read(path)
        if d and set(d.get("sets", {})) >= {"seen_entities", "unseen_entities", "unseen_wording"}:
            tasks.append(ppl(path.stem[len("eval_"):]))
    return tasks


# ---------------------------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------------------------

def run(args):
    LOGS.mkdir(parents=True, exist_ok=True)
    running: Dict[str, tuple] = {}          # tid -> (Popen, slot, log handle, start)
    failed: Dict[str, int] = {}
    free = list(range(args.gpus))
    cpu_busy = False
    t0 = time.time()
    while True:
        # reap finished processes
        for tid, (proc, slot, fh, start) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                del running[tid]
                if slot == "cpu":
                    cpu_busy = False
                else:
                    free.append(slot)
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode != 0:
                    failed[tid] = failed.get(tid, 0) + 1
                print(f"[{time.time() - t0:7.0f}s] done   {tid} ({time.time() - start:.0f}s) {status}", flush=True)
        write_selections()
        tasks = {t.tid: t for t in plan(args)}
        done = {tid for tid, t in tasks.items() if t.done()}
        ready = sorted((t for tid, t in tasks.items() if tid not in done and tid not in running
                        and failed.get(tid, 0) < 2 and all(d in done for d in t.deps)), key=lambda t: (t.priority, t.tid))
        for t in ready:
            if not t.gpu:
                if cpu_busy:
                    continue
                slot, env = "cpu", {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
                cpu_busy = True
            else:
                if not free:
                    continue
                slot = free.pop(0)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot), "PYTHONPATH": str(ROOT)}
            fh = open(LOGS / (t.tid.replace(":", "__").replace("/", "_") + ".log"), "a")
            proc = subprocess.Popen(t.cmd, cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)}, stdout=fh, stderr=subprocess.STDOUT)
            running[t.tid] = (proc, slot, fh, time.time())
            print(f"[{time.time() - t0:7.0f}s] start  {t.tid} on {'cpu' if slot == 'cpu' else 'gpu ' + str(slot)}", flush=True)
        if not running and not ready:
            pending = [tid for tid in tasks if tid not in done]
            print(f"[{time.time() - t0:7.0f}s] scheduler finished; {len(done)} tasks done, pending/blocked: {pending}, failed: {failed}", flush=True)
            break
        time.sleep(args.poll)
    write_selections()

    return 0 if not failed else 1


def main():
    p = argparse.ArgumentParser(description="Extended Part A study on one multi-GPU node")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--ppl-cache", default=str(ROOT / "midsem" / "val_packed_513.pt"))
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
