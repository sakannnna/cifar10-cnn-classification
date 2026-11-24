"""Evaluate the validation-selected CNN on the official CIFAR-10 test split."""
import argparse
import csv
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.datasets import CIFAR10

from model import BaselineCNN


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(rows)


def select_examples(matrix, truth, pred, confidence, classes):
    """Pick each group's most confused pair and a typical error per direction."""
    vehicles = {"airplane", "automobile", "ship", "truck"}
    pairs = {"animal": [], "vehicle": [], "animal_vehicle": []}
    for a, b in combinations(range(len(classes)), 2):
        vehicle_count = int(classes[a] in vehicles) + int(classes[b] in vehicles)
        group = ("animal", "animal_vehicle", "vehicle")[vehicle_count]
        pairs[group].append((a, b))
    for group, candidates in pairs.items():
        a, b = max(candidates, key=lambda p: int(matrix[p]) + int(matrix[p[::-1]]))
        for src, dst in [(a, b), (b, a)]:
            errors = np.flatnonzero((truth == src) & (pred == dst))
            if errors.size:
                dist = np.abs(confidence[errors] - np.median(confidence[errors]))
                yield group, int(errors[np.argmin(dist)]), src, dst, len(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/augment_cosine_seed42"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="Download and verify CIFAR-10 if it is missing")
    parser.add_argument("--out", type=Path, default=Path("results/seed42/test"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    weight_path = args.run / "best.pt"
    weight = torch.load(weight_path, map_location="cpu", weights_only=True)
    ds = CIFAR10(args.data, train=False, download=args.download)
    truth = np.array(ds.targets)
    if ds.classes != config["classes"] or not np.array_equal(np.bincount(truth), [1000] * 10):
        raise ValueError("Unexpected test split or class order")
    if config["normalization"] != "x / 127.5 - 1":
        raise ValueError("This evaluator expects pixels normalized to [-1, 1]")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    model = BaselineCNN().to(device=args.device, memory_format=torch.channels_last)
    model.load_state_dict(weight["model"])
    model.eval()
    outputs = []
    batch_size = config["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(ds), batch_size):
            batch = torch.as_tensor(ds.data[start:start + batch_size], device=args.device)
            pixels = batch.contiguous().permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
            outputs.append(model(pixels).softmax(dim=1).cpu().numpy())
    probs = np.concatenate(outputs)
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite model outputs")
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    matrix = np.bincount(10 * truth + pred, minlength=100).reshape(10, 10)
    n_correct = int(np.count_nonzero(pred == truth))
    summary = {"dataset": "CIFAR-10 official test", "checkpoint": weight_path.as_posix(),
               "checkpoint_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
               "epoch": weight["epoch"], "validation_accuracy": weight["val_acc"],
               "total": len(ds), "correct": n_correct, "errors": len(ds) - n_correct,
               "accuracy": n_correct / len(ds), "classes": ds.classes}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.out / "predictions.csv", ["index", "true", "predicted", "confidence"],
              ((i, ds.classes[t], ds.classes[p], float(confidence[i]))
               for i, (t, p) in enumerate(zip(truth, pred))))
    write_csv(args.out / "confusion.csv", ["true/predicted", *ds.classes],
              ([ds.classes[i], *row] for i, row in enumerate(matrix)))
    write_csv(args.out / "per_class.csv", ["class", "total", "correct", "accuracy"],
              ((name, row.sum(), row[i], row[i] / row.sum())
               for i, (name, row) in enumerate(zip(ds.classes, matrix))))
    sample_dir = args.out / "examples"
    sample_dir.mkdir(exist_ok=True)
    cases = []
    for group, i, t, p, count in select_examples(matrix, truth, pred, confidence, ds.classes):
        image_path = Path("examples") / f"{i:04d}.png"
        Image.fromarray(ds.data[i]).save(args.out / image_path)
        cases.append((group, i, ds.classes[t], ds.classes[p], count,
                      float(confidence[i]), image_path.as_posix()))
    write_csv(args.out / "examples.csv",
              ["group", "index", "true", "predicted", "direction_count", "confidence", "image"], cases)
    print(f"Epoch {weight['epoch']} | correct {n_correct}/{len(ds)} | accuracy {summary['accuracy']:.2%}")


if __name__ == "__main__":
    main()
