"""Compute and plot signed, pre-bias class activation maps for the tested CNN."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def extract(root, data):
    import torch
    from torchvision.datasets import CIFAR10
    from model import BaselineCNN

    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = Path(metrics["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metrics["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the one used for the official test")
    records = read_csv(root / "predictions.csv")
    selected = [r for r in read_csv(root / "examples.csv") if r["group"] == "animal"
                and {r["true"], r["predicted"]} == {"cat", "dog"}]
    if len(selected) != 2:
        raise ValueError("Expected both directions of the original cat/dog error pair")
    for cls in ["cat", "dog"]:
        correct = [r for r in records if r["true"] == r["predicted"] == cls]
        middle = np.median([float(r["confidence"]) for r in correct])
        selected.append(min(correct, key=lambda r: abs(float(r["confidence"]) - middle)))

    ds = CIFAR10(root=data, train=False, download=False)
    ids = np.array([int(r["index"]) for r in selected])
    images = ds.data[ids]
    weight = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    device, batch_size = config["device"], config["batch_size"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    model = BaselineCNN().to(device=device, memory_format=torch.channels_last).eval()
    model.load_state_dict(weight["model"])
    with torch.inference_mode():
        # Match each example's original inference batch, including convolution precision.
        activations, scores = [], []
        for idx in ids:
            start = idx // batch_size * batch_size
            batch = torch.as_tensor(ds.data[start:start + batch_size], device=device)
            pixels = batch.contiguous().permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
            feature_batch = model.features(pixels)
            activations.append(feature_batch[idx - start].cpu())
            scores.append(model.classifier(feature_batch)[idx - start].cpu())
        features = torch.stack(activations)
        head = model.classifier[-1]
        maps = torch.einsum("kc,nchw->nkhw", head.weight.cpu(), features)
        logits = torch.stack(scores)
        bias = head.bias.cpu()
        torch.testing.assert_close(maps.mean(dim=(2, 3)) + bias, logits, atol=1e-5, rtol=1e-5)
        prob = logits.softmax(1)
        if (ds.classes != metrics["classes"]
                or [ds.classes[ds.targets[i]] for i in ids] != [r["true"] for r in selected]
                or [ds.classes[j] for j in prob.argmax(1)] != [r["predicted"] for r in selected]):
            raise ValueError("Sample labels or model predictions differ from the test records")
        np.testing.assert_allclose(prob.amax(1).numpy(),
                                   [float(r["confidence"]) for r in selected], atol=1e-5)
        columns = [ds.classes.index(c) for c in ("cat", "dog")]
        np.savez(root / "cat_dog_cam.npz", ids=ids, images=images, maps=maps[:, columns].numpy(),
                 probabilities=prob[:, columns].numpy(), bias=bias[columns].numpy(),
                 truth=[r["true"] for r in selected], predicted=[r["predicted"] for r in selected])


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import Normalize

    with np.load(root / "cat_dog_cam.npz") as data:
        ids, images, maps, truth, pred = (data[k] for k in ("ids", "images", "maps", "truth", "predicted"))
        probs = data["probabilities"]
    font = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei")
                 if f in font_manager.fontManager.get_font_names()), None)
    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    names = {"cat": "猫", "dog": "狗"} if font else {"cat": "Cat", "dog": "Dog"}
    limit = float(np.max(np.abs(maps)))
    norm = Normalize(-limit, limit)
    fig, axes = plt.subplots(len(ids), 3, figsize=(6.6, 7.6))
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.14, top=0.89, hspace=0.13, wspace=0.08)
    fig.suptitle("猫与狗：同一张图对两类预测的贡献" if font else "Cat and dog class activation maps", y=.985)
    headings = ["测试原图", "猫 CAM", "狗 CAM"] if font else ["Test image", "Cat CAM", "Dog CAM"]
    for ax, name in zip(axes[0], headings):
        ax.set_title(name)
    for row, (idx, image, pair) in enumerate(zip(ids, images, maps)):
        first = axes[row, 0]
        first.imshow(image, interpolation="nearest")
        score = probs[row, int(pred[row] == "dog")]
        label = f"#{idx}   {names[truth[row]]} → {names[pred[row]]}"
        caption = (f"预测分数 {score:.1%}" if font else f"Pred. score {score:.1%}")
        first.text(-.14, .5, label + "\n" + caption, transform=first.transAxes,
                   va="center", ha="right", linespacing=1.7, fontsize=10)
        for ax, values in zip(axes[row, 1:], pair):
            heat = ax.imshow(values, cmap="RdBu_r", norm=norm, interpolation="nearest")
        for ax in axes[row]:
            ax.set_axis_off()
    cax = fig.add_axes([.34, .082, .55, .017])
    bar = fig.colorbar(heat, cax=cax, orientation="horizontal")
    bar.set_label("对类别分数的贡献（蓝色为负，红色为正）" if font else "Class score contribution (blue: negative, red: positive)")
    fig.savefig(root / "cat_dog_cam.png", dpi=210, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("results/seed42/test"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--step", choices=["all", "extract", "plot"], default="all")
    args = parser.parse_args()
    if args.step in ("all", "extract"):
        extract(args.result, args.data)
    if args.step in ("all", "plot"):
        plot(args.result)
