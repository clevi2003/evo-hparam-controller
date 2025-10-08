from __future__ import annotations
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = 0,
    global_step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint. extra can include run metadata, config dicts, metrics, etc.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, p)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    strict: bool = True,
) -> Tuple[int, int]:
    """
    Load a checkpoint into model/optimizer/scheduler.
    Returns (epoch, global_step). If a component is missing, it is skipped.
    """
    ckpt = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))

