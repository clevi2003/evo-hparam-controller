import argparse
import os
import time
import yaml
from rich import print
from tqdm.auto import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR

from .data import get_dataloaders
from .models import resnet20
from .utils import CSVLogger, TBLogger, accuracy, seed_everything, save_checkpoint, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--exp-name", type=str, default=None)
    return p.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    model = resnet20(num_classes=cfg["model"]["num_classes"])
    return model


def build_optimizer(cfg, model):
    name = cfg["optim"]["name"].lower()
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg["optim"]["lr"],
            momentum=cfg["optim"]["momentum"],
            weight_decay=cfg["optim"]["weight_decay"],
            nesterov=True,
        )
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer {name}")


def build_scheduler(cfg, optimizer, steps_per_epoch):
    sc = cfg["scheduler"]["name"].lower()
    if sc == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg["scheduler"]["t_max"])
    elif sc == "step":
        return StepLR(optimizer, step_size=cfg["scheduler"]["step_size"], gamma=cfg["scheduler"]["gamma"])
    elif sc == "onecycle":
        # max_lr == initial lr by default
        return OneCycleLR(
            optimizer,
            max_lr=cfg["optim"]["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=cfg["train"]["epochs"],
            pct_start=cfg["scheduler"]["pct_start"],
        )
    elif sc == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler {sc}")


def train_one_epoch(model, loader, criterion, optimizer, device, logger_tb, csv_logger, epoch, log_interval):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(loader, total=len(loader), desc=f"train epoch {epoch}", leave=False)
    for step, (x, y) in enumerate(pbar, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = accuracy(logits, y)
        running_loss += loss.item()
        running_acc += acc

        if step % log_interval == 0:
            logger_tb.log_scalar("train/loss", loss.item(), (epoch-1)*len(loader) + step)
            logger_tb.log_scalar("train/acc", acc, (epoch-1)*len(loader) + step)

        # Update progress bar postfix with smoothed stats
        pbar.set_postfix({"loss": f"{running_loss/step:.4f}", "acc": f"{running_acc/step:.4f}"})

    return running_loss / len(loader), running_acc / len(loader)



def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    pbar = tqdm(loader, total=len(loader), desc=f"eval epoch {epoch}", leave=False)
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item()
            acc_sum += accuracy(logits, y)
            steps = max(1, pbar.n)
            pbar.set_postfix({"loss": f"{loss_sum/steps:.4f}", "acc": f"{acc_sum/steps:.4f}"})
    return loss_sum / len(loader), acc_sum / len(loader)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    exp_name = args.exp_name or cfg.get("exp_name", "run")

    seed_everything(cfg.get("seed", 1337), cfg["train"].get("deterministic", False))

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        augment=cfg["data"]["augment"],
    )

    model = build_model(cfg).to(device)
    print(f"[bold green]Model params:[/bold green] {count_params(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = build_optimizer(cfg, model)

    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch)

    # Logging
    tb = TBLogger(cfg["log"]["dir_tb"], exp_name)
    csv_logger = CSVLogger(cfg["log"]["csv_path"])

    best_acc = 0.0
    ckpt_dir = os.path.join(cfg["ckpt"]["dir"], exp_name)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, tb, csv_logger, epoch, cfg["log"]["log_interval"])
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, epoch)
        epoch_time = time.time() - t0

        # Scheduler
        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            scheduler.step()

        # Log scalars
        step = epoch * steps_per_epoch
        tb.log_scalar("val/loss", val_loss, step)
        tb.log_scalar("val/acc", val_acc, step)
        for i, pg in enumerate(optimizer.param_groups):
            tb.log_scalar(f"lr/group{i}", pg["lr"], step)

        # CSV
        csv_logger.log(epoch=epoch, step=step, split="train", loss=tr_loss, acc=tr_acc, lr=optimizer.param_groups[0]["lr"])
        csv_logger.log(epoch=epoch, step=step, split="val", loss=val_loss, acc=val_acc, lr=optimizer.param_groups[0]["lr"])

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": cfg,
            }, ckpt_dir, "best.pt")

        print(f"[epoch {epoch:03d}] train_acc={tr_acc:.4f} val_acc={val_acc:.4f} time={epoch_time:.1f}s")

    # Last checkpoint
    save_checkpoint({
        "epoch": cfg["train"]["epochs"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc": best_acc,
        "config": cfg,
    }, ckpt_dir, "last.pt")

    tb.close(); csv_logger.close()

if __name__ == "__main__":
    main()
