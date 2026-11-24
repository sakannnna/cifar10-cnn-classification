"""Render the official test confusion matrix and the six selected original images."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import PowerNorm


CLASS_ZH = dict(zip(
    ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, nargs="?", default=Path("results/seed42/test"))
    root = parser.parse_args().result
    available = font_manager.fontManager.get_font_names()
    font = next((f for f in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"] if f in available), None)
    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    classes = metrics["classes"]
    names = [CLASS_ZH[c] for c in classes] if font else classes
    with (root / "confusion.csv").open(encoding="utf-8") as f:
        matrix = np.array([[int(row[c]) for c in classes] for row in csv.DictReader(f)])
    fig, ax = plt.subplots(figsize=(7.4, 6.4), layout="constrained")
    norm = PowerNorm(gamma=0.55, vmin=0, vmax=1000)
    im = ax.imshow(matrix, cmap="Blues", norm=norm, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.82, pad=0.035, ticks=[0, 50, 100, 250, 500, 750, 1000],
                 label="图像数" if font else "Image count")
    ax.set(xticks=range(10), yticks=range(10), yticklabels=names,
           xlabel="预测类别" if font else "Predicted class", ylabel="真实类别" if font else "True class")
    ax.set_xticklabels(names, rotation=0 if font else 30, ha="center" if font else "right")
    accuracy = metrics["accuracy"]
    ax.set_title((f"CIFAR-10 测试集混淆矩阵   ·   准确率 {accuracy:.2%}" if font else
                  f"CIFAR-10 test confusion matrix   ·   accuracy {accuracy:.2%}"), pad=14)
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center", fontsize=9,
                color="white" if norm(count) > 0.57 else "#17334b")
    ax.tick_params(length=0)
    fig.savefig(root / "confusion_matrix.png", dpi=220, facecolor="white")
    plt.close(fig)

    with (root / "examples.csv").open(encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.5), layout="constrained")
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, case in zip(axes.T.flat, cases):
        with Image.open(root / case["image"]) as img:
            ax.imshow(np.asarray(img), interpolation="nearest")
        t, p = case["true"], case["predicted"]
        heading = f"{CLASS_ZH[t]} → {CLASS_ZH[p]}" if font else f"{t} → {p}"
        ax.set_title(heading, fontsize=12, pad=4)
        ax.text(0.5, -0.04, f"#{case['index']}  ·  {float(case['confidence']):.1%}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#36475c")
    fig.suptitle("典型错例：真实类别 → 预测类别" if font else "Test errors: true class → prediction", fontsize=15)
    fig.supxlabel("图下标注测试集索引和预测类别的 softmax 分数" if font else
                  "Test index and predicted-class softmax score are shown below each image", fontsize=10)
    fig.savefig(root / "typical_errors.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
