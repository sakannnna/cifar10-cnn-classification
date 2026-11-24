import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot a baseline training run")
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    with (args.run / "history.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        parser.error("No completed epochs in history.csv")
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
    for ax, metric, label, factor in zip(axes, ["loss", "acc"], ["Cross-entropy loss", "Accuracy (%)"], [1, 100]):
        for split, name in [("train", "Training"), ("val", "Validation")]:
            values = [float(row[f"{split}_{metric}"]) * factor for row in rows]
            ax.plot(epochs, values, label=name, marker="." if len(epochs) == 1 else None)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.savefig(args.run / "curves.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
