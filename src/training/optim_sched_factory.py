from __future__ import annotations
from typing import Dict, Any, Optional, Literal, NamedTuple
from torch.nn import Module
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim.lr_scheduler as sched

from src.core.config.config_train import SchedCfg, OptimCfg

def _lower(s: str | None) -> str:
    return (s or "").strip().lower()

def build_optimizer(model: Module, cfg: OptimCfg) -> Optimizer:
    name = _lower(cfg.name)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found in model.")

    if name in ("sgd",):
        kwargs: Dict[str, Any] = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "momentum": cfg.momentum,
            "nesterov": cfg.extra.get("nesterov", False),
        }
        return optim.SGD(params, **kwargs)

    if name in ("adam",):
        betas = cfg.betas if cfg.betas is not None else (0.9, 0.999)
        kwargs = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "betas": betas,
            "eps": cfg.extra.get("eps", 1e-8),
            "amsgrad": cfg.extra.get("amsgrad", False),
        }
        return optim.Adam(params, **kwargs)

    if name in ("adamw",):
        betas = cfg.betas if cfg.betas is not None else (0.9, 0.999)
        kwargs = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "betas": betas,
            "eps": cfg.extra.get("eps", 1e-8),
            "amsgrad": cfg.extra.get("amsgrad", False),
        }
        return optim.AdamW(params, **kwargs)

    if name in ("rmsprop",):
        kwargs = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "momentum": cfg.momentum,
            "alpha": cfg.extra.get("alpha", 0.99),
            "eps": cfg.extra.get("eps", 1e-8),
            "centered": cfg.extra.get("centered", False),
        }
        return optim.RMSprop(params, **kwargs)

    raise ValueError(
        f"Unknown optimizer '{cfg.name}'. Supported: SGD, Adam, AdamW, RMSprop."
    )


StepWhen = Literal["epoch", "batch", "disabled"]

class SchedulerHandle(NamedTuple):
    scheduler: Optional[_LRScheduler]
    step_when: StepWhen
    warmup_scheduler: Optional[_LRScheduler] = None # TODO add warmup support

def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def build_baseline_scheduler(
    optimizer: Optimizer,
    sched_cfg: SchedCfg,
    optim_cfg: OptimCfg,
    *,
    steps_per_epoch: int,
    epochs: int,
) -> SchedulerHandle:
    name = _lower(sched_cfg.name)

    if name in ("", "none", "off", "disabled", "null", "nil"):
        return SchedulerHandle(None, "disabled", None)

    if name == "cosine":
        # trainer will step this each epoch
        t_max = sched_cfg.t_max if sched_cfg.t_max is not None else epochs
        scheduler = sched.CosineAnnealingLR(optimizer, T_max=t_max)
        return SchedulerHandle(scheduler, "epoch")

    if name == "step":
        scheduler = sched.StepLR(optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma)
        return SchedulerHandle(scheduler, "epoch")

    if name == "multistep":
        milestones = sched_cfg.milestones or []
        scheduler = sched.MultiStepLR(optimizer, milestones=milestones, gamma=sched_cfg.gamma)
        return SchedulerHandle(scheduler, "epoch")

    if name == "onecycle":
        if steps_per_epoch <= 0 or epochs <= 0:
            raise ValueError("OneCycle requires positive steps_per_epoch and epochs.")
        total_steps = steps_per_epoch * epochs
        max_lr = optim_cfg.lr
        # pct_start validated in config; OneCycleLR handles anneal strategy
        scheduler = sched.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=sched_cfg.pct_start or 0.1,
            anneal_strategy="cos",
            div_factor=optim_cfg.extra.get("onecycle_div_factor", 25.0),
            final_div_factor=optim_cfg.extra.get("onecycle_final_div_factor", 1e4),
        )
        return SchedulerHandle(scheduler, "batch")

    raise ValueError(
        f"Unknown scheduler '{sched_cfg.name}'. "
        "Supported: cosine, step, multistep, onecycle, none."
    )