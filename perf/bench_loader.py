# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "datasets", "numpy", "pillow"]
# ///
#
# PyTorch counterpart of perf/bench_loader.jl: data-loading-only timing of `torch.utils.data`'s
# multiprocess DataLoader on the same CIFAR-10 on-the-fly decode+augment pipeline, for a direct
# comparison with MLUtils' `num_workers` (+ shared-memory `shm`) path. Run with:
#   uv run perf/bench_loader.py            # EPOCHS=8 by default
#
# Iterates the training loader with NO model, isolating decode + augment + collate + worker IPC.
# Epoch 1 (cold: worker spawn + per-worker dataset re-open) is reported separately from the WARM
# steady-state epochs, matching the Julia benchmark's methodology.

import os
import time

import numpy as np
import torch
from datasets import load_dataset, disable_progress_bars
from datasets.utils.logging import set_verbosity_error
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

disable_progress_bars()
set_verbosity_error()

BATCHSIZE = 128
EPOCHS = int(os.environ.get("EPOCHS", "8"))

# Standard CIFAR-10 per-channel mean/std (same values as bench_loader.jl).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Normalize first, then crop+flip — matching the Julia pipeline (which augments already-normalized
# tensors, so the pad-by-4 border is zero in normalized space).
TRAIN_TF = transforms.Compose([
    transforms.ToTensor(),                    # PIL HWC uint8 -> CHW float in [0, 1]
    transforms.Normalize(MEAN, STD),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])


class HFImageDataset(Dataset):
    """On-the-fly: index the Arrow-backed HF dataset and decode+augment per item. Memory-mapped and
    picklable, so with num_workers>0 each worker re-opens it (data shared, not copied) — the same
    mechanism as the Julia num_workers path."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return self.transform(row["img"]), int(row["label"])


def epoch_times(train_hf, num_workers):
    ds = HFImageDataset(train_hf, TRAIN_TF)
    # persistent_workers keeps the pool alive across epochs (like Julia's leased warm pool).
    loader = DataLoader(ds, batch_size=BATCHSIZE, shuffle=True, num_workers=num_workers,
                        persistent_workers=num_workers > 0)
    times = []
    for _ in range(EPOCHS):
        t0 = time.perf_counter()
        seen = 0
        for x, y in loader:
            seen += y.shape[0]
        times.append(time.perf_counter() - t0)
    del loader
    return times


def main():
    train_hf = load_dataset("uoft-cs/cifar10", split="train")

    print("### WARMUP")
    epoch_times(train_hf, 0)
    epoch_times(train_hf, 2)

    results = []  # (label, setup=epoch1, warm=mean(epoch 2:))
    print(f"\n#### CIFAR-10 loader-only, on-the-fly decode+augment, {EPOCHS} epochs ###########")
    for nw in (0, 2, 4, 8):
        ts = epoch_times(train_hf, nw)
        warm = sum(ts[1:]) / (len(ts) - 1) if len(ts) > 1 else ts[0]
        label = "serial" if nw == 0 else f"num_workers={nw}"
        print(f"### {label}:  epoch1(cold)={ts[0]:.2f}s   warm/epoch={warm:.3f}s")
        results.append((label, ts[0], warm))

    print(f"\n#### SUMMARY — WARM steady-state per-epoch (the real loading cost) ###########")
    serial_warm = results[0][2]
    print(f"{'config':<16}{'setup (s)':<12}{'warm/epoch (s)':<16}speedup")
    for label, setup, warm in results:
        print(f"{label:<16}{setup:<12.2f}{warm:<16.3f}{serial_warm / warm:.2f}x")


if __name__ == "__main__":
    main()
