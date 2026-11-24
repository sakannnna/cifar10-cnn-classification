import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10

from model import BaselineCNN


def stratified_split(labels, seed):
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) != 5000:
            raise ValueError(f"Expected 5000 training examples for class {label}")
        validation.append(indices[:500])
        training.append(indices[500:])
    return np.concatenate(training), np.concatenate(validation)


def read_split(path, labels):
    with np.load(path, allow_pickle=False) as split:
        train, val = split["train"], split["val"]
    for indices, per_class in [(train, 4500), (val, 500)]:
        if (indices.shape != (10 * per_class,) or indices.dtype.kind not in "iu"
                or indices.min() < 0 or indices.max() >= len(labels)):
            raise ValueError(f"Invalid CIFAR-10 indices in {path}")
        if not np.all(np.bincount(labels[indices], minlength=10) == per_class):
            raise ValueError(f"Unexpected class counts in {path}")
    if np.unique(np.concatenate([train, val])).size != len(labels):
        raise ValueError(f"Duplicate or missing examples in {path}")
    return train.astype(np.int64), val.astype(np.int64)


def augment(images, generator):
    """Independently crop and flip each NHWC uint8 image in the batch."""
    size, h, w, _ = images.shape
    padded = nn.functional.pad(images, (0, 0, 4, 4, 4, 4))
    device = images.device
    starts = torch.randint(9, (2, size, 1, 1), device=device, generator=generator)
    rows = starts[0] + torch.arange(h, device=device).view(1, h, 1)
    cols = torch.arange(w, device=device).view(1, 1, w)
    flip = torch.rand((size, 1, 1), device=device, generator=generator) < 0.5
    cols = starts[1] + torch.where(flip, w - 1 - cols, cols)
    batch = torch.arange(size, device=device).view(-1, 1, 1)
    return padded[batch, rows, cols]


def run_epoch(model, images, labels, indices, batch_size, optimizer=None, aug_rng=None):
    training = optimizer is not None
    model.train(training)
    if training:
        indices = indices[torch.randperm(len(indices), device=images.device)]
    loss_sum = torch.zeros((), device=images.device, dtype=torch.float64)
    correct = torch.zeros((), device=images.device, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for batch in indices.split(batch_size):
            # Keep the dataset as compact uint8 images. Normalize only this batch.
            pixels = images[batch]
            if training and aug_rng is not None:
                pixels = augment(pixels, aug_rng)
            inputs = pixels.permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
            targets = labels[batch]
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.detach() * len(batch)
            correct += (logits.detach().argmax(1) == targets).sum()
    return loss_sum.item() / len(indices), correct.item() / len(indices)


def main():
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 baseline CNN")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True, help="New run directory")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--split", type=Path, help="Reuse a saved split.npz")
    parser.add_argument("--augment", action="store_true", help="Training-only random crop and flip")
    parser.add_argument("--scheduler", choices=["constant", "cosine"], default="constant")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.lr) <= 0 or args.seed < 0:
        parser.error("epochs, batch-size and lr must be positive, seed nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    dataset = CIFAR10(root=args.data, train=True, download=True)
    raw_labels = np.array(dataset.targets)
    if args.split:
        train_ids, val_ids = read_split(args.split, raw_labels)
    else:
        train_ids, val_ids = stratified_split(raw_labels, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / "split.npz", train=train_ids, val=val_ids)
    device = torch.device(args.device)
    # All 50k images take about 147 MiB as uint8 and fit on this GPU.
    images = torch.as_tensor(dataset.data, device=device).contiguous()
    labels = torch.as_tensor(raw_labels, dtype=torch.long, device=device)
    train_ids = torch.as_tensor(train_ids, device=device)
    val_ids = torch.as_tensor(val_ids, device=device)
    model = BaselineCNN().to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - 1), eta_min=args.lr * 0.01
        )
    # Keep data shuffling on its original RNG stream, independent of augmentation.
    aug_rng = torch.Generator(device=device).manual_seed(args.seed + 1) if args.augment else None
    config = {
        "model": "BaselineCNN", "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "optimizer": "Adam", "lr": args.lr, "augmentation": args.augment,
        "augmentation_policy": "zero-pad 4, random 32x32 crop, horizontal flip p=0.5" if args.augment else None,
        "scheduler": args.scheduler, "min_lr": scheduler.eta_min if scheduler else args.lr,
        "split_source": str(args.split) if args.split else None,
        "normalization": "x / 127.5 - 1", "train_count": len(train_ids),
        "val_count": len(val_ids), "classes": dataset.classes, "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)
    best_acc = -1
    with (args.out / "history.csv").open("x", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds", "lr"])
        log.flush()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(model, images, labels, train_ids, args.batch_size, optimizer, aug_rng)
            val_loss, val_acc = run_epoch(model, images, labels, val_ids, args.batch_size)
            if val_acc > best_acc:
                best_acc = val_acc
                temp = args.out / "best.tmp"
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_acc}, temp)
                temp.replace(args.out / "best.pt")
            seconds = time.perf_counter() - start
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, round(seconds, 2), lr])
            log.flush()
            print(f"Epoch {epoch:02}/{args.epochs} train {train_loss:.4f} {train_acc:.2%} "
                  f"val {val_loss:.4f} {val_acc:.2%} ({seconds:.1f}s)", flush=True)
            if scheduler is not None and epoch < args.epochs:
                scheduler.step()


if __name__ == "__main__":
    main()
