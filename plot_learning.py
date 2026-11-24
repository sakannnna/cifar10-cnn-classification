"""Compare the baseline and selected model on both training and validation data."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("results/seed42/learning_curves.png"))
    args = parser.parse_args()
    fonts = font_manager.fontManager.get_font_names()
    font = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if f in fonts), None)
    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.45))
    for dirname, name, color in [
        ("baseline_seed42", "基础组" if font else "Baseline", "#485c85"),
        ("augment_cosine_seed42", "增强＋调度" if font else "Aug. + cosine", "#b74c24"),
    ]:
        with (args.runs / dirname / "history.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        for subset, text, style in [("train", "训练" if font else "train", "--"),
                                    ("val", "验证" if font else "val.", "-")]:
            for ax, key, factor in zip(axes, ["loss", "acc"], [1, 100]):
                ax.plot(epochs, [float(r[f"{subset}_{key}"]) * factor for r in rows],
                        lw=1.85, color=color, ls=style, label=f"{name} · {text}")
    titles = ["交叉熵损失", "准确率（%）"] if font else ["Cross-entropy loss", "Accuracy (%)"]
    for ax, title in zip(axes, titles):
        ax.set(title=title, xlabel="训练轮次" if font else "Epoch", xticks=[1, 5, 10, 15, 20, 25])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="lower center",
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
