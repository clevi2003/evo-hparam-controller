"""Small smoke runner for Trainer.fit() that uses a tiny synthetic dataset to exercise run metadata writing.

This script creates a minimal config dict, a tiny random dataset (few samples), constructs model/optimizer/scheduler,
instantiates src.training.engine.Trainer and runs fit() for 1 epoch with mixed_precision False. After run, it prints run_dir and
prints the contents of run_meta.json, config.json, and env.json to stdout.
"""

import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from src.training.engine import Trainer


class TinyModel(nn.Module):
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # expect input [N,3,8,8]
        return self.net(x)


def make_tiny_dataloaders(batch_size=4):
    # create 20 random samples of 3x8x8
    xs = torch.randn(20, 3, 8, 8)
    ys = torch.randint(0, 10, (20,))
    ds = TensorDataset(xs, ys)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    # use same for val
    return loader, loader


def main():
    # minimal config dict similar to expected structure
    cfg = {
        "data": {"root": "./data", "batch_size": 4, "num_workers": 0, "augment": False},
        "model": {"num_classes": 10},
        "optim": {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "train": {"epochs": 1, "label_smoothing": 0.0},
        "log": {"dir_tb": "runs/tb", "csv_path": "runs/log.csv", "log_interval": 10},
        "seed": 123,
    }

    device = torch.device("cpu")
    train_loader, val_loader = make_tiny_dataloaders(batch_size=cfg["data"]["batch_size"])

    model = TinyModel()
    optimizer = optim.SGD(model.parameters(), lr=cfg["optim"]["lr"], momentum=cfg["optim"]["momentum"]) 
    scheduler = None
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
                      train_loader=train_loader, val_loader=val_loader, device=device,
                      hooks=None, epochs=1, mixed_precision=False, cfg=cfg)

    summary = trainer.fit()

    run_dir = Path(summary.get("run_dir")) if summary.get("run_dir") else None
    # The Trainer writes run_dir path into state; but also writes files to run_dir returned via state; we can search runs dir

    # Find most recent run folder created under ./runs
    runs_root = Path(os.getcwd()) / "runs"
    candidates = sorted(runs_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        recent = candidates[0]
        print("Run dir:", recent)
        for fname in ["run_meta.json", "config.json", "env.json"]:
            p = recent / fname
            if p.exists():
                print(f"--- {fname} ---")
                print(p.read_text())
            else:
                print(f"{fname} not found in {recent}")
    else:
        print("No runs found in ./runs")


if __name__ == '__main__':
    main()
