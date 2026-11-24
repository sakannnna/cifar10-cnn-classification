"""Compare the four CIFAR-10 augmentation and learning-rate settings."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = [
    ("baseline_seed42", "Baseline", "#454f63", "-"),
    ("cosine_seed42", "Cosine LR only", "#7056a5", "-."),
    ("augment_seed42", "Crop + flip", "#007d79", "--"),
    ("augment_cosine_seed42", "Crop + flip + cosine LR", "#bd4c19", "-"),
]


def read_run(path):
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    with (path / "history.csv").open(encoding="utf-8") as source:
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(source)]
    if [r["epoch"] for r in history] != list(range(1, cfg["epochs"] + 1)):
        raise ValueError(f"Incomplete training history in {path}")
    with np.load(path / "split.npz", allow_pickle=False) as saved:
        split = (saved["train"], saved["val"])
    return cfg, history, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("results/seed42"))
    args = parser.parse_args()
    baseline_cfg, _, baseline_split = read_run(args.runs / RUNS[0][0])
    summary = []
    fig, (acc_ax, loss_ax) = plt.subplots(1, 2, figsize=(10, 4))
    for dirname, label, color, style in RUNS:
        cfg, history, split = read_run(args.runs / dirname)
        if not all(np.array_equal(a, b) for a, b in zip(split, baseline_split)):
            raise ValueError(f"Data split differs in {dirname}")
        for key in ("model", "epochs", "batch_size", "seed", "optimizer", "lr", "normalization"):
            if cfg[key] != baseline_cfg[key]:
                raise ValueError(f"{dirname} has a different {key}")
        best = max(history, key=lambda row: row["val_acc"])
        lowest = min(history, key=lambda row: row["val_loss"])
        last = history[-1]
        summary.append({
            "run": dirname, "epochs": len(history), "best_epoch": int(best["epoch"]),
            "best_val_acc_pct": round(100 * best["val_acc"], 2),
            "final_val_acc_pct": round(100 * last["val_acc"], 2),
            "final_train_acc_pct": round(100 * last["train_acc"], 2),
            "min_val_loss": round(lowest["val_loss"], 5),
            "min_loss_epoch": int(lowest["epoch"]),
            "final_val_loss": round(last["val_loss"], 5),
        })
        epochs = [r["epoch"] for r in history]
        options = dict(color=color, lw=1.8, ls=style, label=label)
        acc_ax.plot(epochs, [r["val_acc"] * 100 for r in history], **options)
        acc_ax.scatter(best["epoch"], best["val_acc"] * 100, c=color, s=38,
                       edgecolor="white", linewidth=0.8, zorder=3)
        loss_ax.plot(epochs, [r["val_loss"] for r in history], **options)
    for ax, title in [(acc_ax, "Validation accuracy (%)"), (loss_ax, "Validation loss (cross-entropy)")]:
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*acc_ax.get_legend_handles_labels(), loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "validation_curves.png", dpi=180)
    plt.close(fig)
    with (args.out / "comparison.csv").open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
