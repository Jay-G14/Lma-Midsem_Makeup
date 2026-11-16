#!/usr/bin/env python3
"""Picks the LoRA learning rate from the r = 4 probe: the LR whose best epoch (of 5) has the lowest
reasoning-validation loss (validation split only, never a test set). Prints the LR, writes
results/lr_choice.json, and installs that probe run as the r = 4 run of the rank sweep (identical
configuration: r = 4, same LR, 5 epochs, seed 42)."""

import shutil

from midsem.part_a.common import CKPT_DIR, RESULTS_DIR, read_json, write_json


def main():
    rows = []
    for path in sorted(RESULTS_DIR.glob("lrprobe_r4_lr*_history.json")):
        h = read_json(path)
        rows.append({"lr": h["lr"], "tag": path.name[:-len("_history.json")], "epochs": h["history"][-1]["epoch"],
                     "best_epoch": h["best_epoch"], "best_val_reasoning_loss": h["best_val_reasoning_loss"],
                     "val_loss_by_epoch": [round(e["val_reasoning_loss"], 4) for e in h["history"]],
                     "ppl_by_epoch": [round(e["pretrain_ppl"], 2) for e in h["history"]]})
    rows.sort(key=lambda r: r["lr"])
    best = min(rows, key=lambda r: r["best_val_reasoning_loss"])
    write_json(RESULTS_DIR / "lr_choice.json", {"chosen_lr": best["lr"], "criterion": "lowest best-epoch reasoning-validation loss, r=4, 5 epochs",
                                                "probe": rows})
    src = CKPT_DIR / "lrprobe" / best["tag"].replace("lrprobe_", "")
    shutil.copytree(src, CKPT_DIR / "lora_r4", dirs_exist_ok=True)
    hist = read_json(RESULTS_DIR / f"{best['tag']}_history.json")
    hist["note"] = f"identical configuration to the LR-probe run {best['tag']} (copied from it)"
    write_json(RESULTS_DIR / "lora_r4_history.json", hist)
    print(f"{best['lr']:g}")


if __name__ == "__main__":
    main()
