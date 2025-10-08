from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split

# hooks and training engine
from src.core.hooks.hook_base import Hook
from src.core.hooks.hook_composition import HookList
from src.core.hooks.common_hooks import LambdaHook
from src.utils.seed_device import get_device, seed_everything
from src.training.engine import Trainer

# controller
from src.controller.features import FeatureExtractor
from src.controller.lr_scheduler import LrControllerScheduler
from src.controller.runtime import ControllerRuntime, RuntimeConfig
from src.controller.serialization import unflatten_params
from src.controller.controller import ControllerMLP

# logging
from src.core.logging.loggers import ControllerTickLogger, make_train_parquet_logger, make_val_parquet_logger

# Fitness summary
from src.evolution.fitness import RunSummary

from src.models.resnet_cifar10 import resnet20


class _BaseSchedulerStepHook(Hook):
    """Step the base scheduler each batch before controller applies its action"""
    def __init__(self, ctrl_scheduler: LrControllerScheduler) -> None:
        self.ctrl_scheduler = ctrl_scheduler

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        # step base schedule first. ControllerRuntime's on_batch_end will run after
        self.ctrl_scheduler.step_base()


class _CollectEvalPointsHook(Hook):
    """Collect (global_step, val_acc) at the end of each evaluation"""
    def __init__(self, collector: List[Tuple[int, float]]) -> None:
        self.collector = collector

    def on_eval_end(self, state: Dict[str, Any]) -> None:
        global_step = int(state.get("global_step", 0))
        val_acc = state.get("val_acc", None)
        if val_acc is not None:
            self.collector.append((global_step, float(val_acc)))


class _CollectActionDeltasHook(Hook):
    """Collect applied delta log lr each time the controller acts"""
    def __init__(self, collector: List[float]) -> None:
        self.collector = collector

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        if state.get("controller_applied", False):
            applied_delta = state.get("controller_delta_applied", None)
            if applied_delta is not None:
                self.collector.append(float(applied_delta))


class _DivergenceGuardHook(Hook):
    """
    Detect nans or explosions. Sets state['diverged']=True when detected
    TODO: extend this to add gradient/weight explosion thresholds if needed
    """
    def __init__(self, loss_is_nan_penalty: float = 0.0) -> None:
        self.loss_is_nan_penalty = loss_is_nan_penalty

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        loss = state.get("loss", None)
        if loss is None:
            return
        # check for inf/nan
        try:
            f = float(loss)
        except Exception:
            state["diverged"] = True
            return
        if math.isnan(f) or math.isinf(f):
            state["diverged"] = True


def _default_build_cifar10_dataloaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    subset_pct: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Minimal CIFAR-10 loaders using torchvision. If you have a custom dataset module,
    pass a different builder to TruncatedTrainingEvaluator.
    """
    try:
        import torchvision
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError(
            "Default CIFAR-10 data builder requires torchvision. "
            "Please install torchvision or supply a custom dataloader builder."
        ) from e

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tf_train)
    test_set  = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=tf_test)

    if subset_pct < 1.0:
        # Deterministic subset of the train set
        total = len(train_set)
        keep = max(1, int(total * subset_pct))
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(total, generator=g)[:keep]
        train_set = Subset(train_set, idx.tolist())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def _default_build_cifar_model() -> torch.nn.Module:
    """
    Very small adapter: try to import a project-specific CIFAR ResNet;
    fallback to torchvision ResNet18 (adjusted classifier) if unavailable.
    """
    try:
        return resnet20()
    except Exception as e:
        raise RuntimeError("Could not build a model.") from e


@dataclass
class EvaluatorConfig:
    """
    pluggable configuration for the evaluator
    if it's passed custom builders, the defaults are ignored
    """
    # builders
    model_builder: Optional[Callable[[str, int], torch.nn.Module]] = None
    dataloaders_builder: Optional[
        Callable[[str | Path, int, int, float, int], Tuple[DataLoader, DataLoader]]
    ] = None

    # logging and artifacts
    out_dir: Optional[str | Path] = None # if set, parquet logs will be written to this dir
    write_controller_ticks: bool = True
    write_train_val_logs: bool = True


class TruncatedTrainingEvaluator:
    """
    Evaluate a candidate controller parameter vector by running a short,
    reproducible training budget and computing a RunSummary for fitness
    """

    def __init__(
        self,
        train_cfg,
        ctrl_cfg,
        budget_cfg,
        device: Optional[torch.device] = None,
        evaluator_cfg: Optional[EvaluatorConfig] = None,
    ) -> None:
        self.train_cfg = train_cfg
        self.ctrl_cfg = ctrl_cfg
        self.budget_cfg = budget_cfg
        self.device = device or get_device(getattr(train_cfg, "device", None))
        self.cfg = evaluator_cfg or EvaluatorConfig()

        # builders
        self._model_builder = self.cfg.model_builder or (lambda arch, nc: _default_build_cifar_model())
        self._data_builder = self.cfg.dataloaders_builder or (
            lambda root, bs, nw, pct, seed: _default_build_cifar10_dataloaders(root, bs, nw, pct, seed)
        )

        # output directory for optional logs
        self.out_dir = Path(self.cfg.out_dir) if self.cfg.out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, controller_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Load the candidate weights into a fresh controller, run the budgeted
        training, and return a dict with RunSummary and convenience scalars
        """
        # set seed and device
        seed_everything(self.budget_cfg.fixed_seed)

        # build data
        train_loader, val_loader = self._data_builder(
            self.train_cfg.data.data_root,
            self.train_cfg.data.batch_size,
            self.train_cfg.data.num_workers,
            self.budget_cfg.fixed_subset_pct,
            self.budget_cfg.fixed_seed,
        )

        # build model
        model = self._model_builder(self.train_cfg.model.arch, self.train_cfg.model.num_classes)
        model.to(self.device)

        # optimizer
        optimizer = self._build_optimizer(model)

        # base scheduler setup with a hook
        base_scheduler = self._build_scheduler(optimizer)

        # controller, build & load candidate vector
        in_dim = len(self.ctrl_cfg.features)
        controller = ControllerMLP(in_dim=in_dim,
                                   hidden=self.ctrl_cfg.controller_arch.hidden,
                                   max_step=self.ctrl_cfg.action.max_step).to(self.device)

        # load candidate params into controller
        unflatten_params(controller, controller_vector, strict=True)

        # feature extractor and controller scheduler and runtime
        extractor = FeatureExtractor(
            spec=self.ctrl_cfg.features,
            ema_alpha=self.ctrl_cfg.feature_ema_alpha,
            standardize=True,
            device=self.device,
        )

        ctrl_scheduler = LrControllerScheduler(
            optimizer=optimizer,
            base_scheduler=base_scheduler,
            lr_min=self.ctrl_cfg.action.lr_min,
            lr_max=self.ctrl_cfg.action.lr_max,
        )

        # optional Parquet logs
        ticks_logger = None
        train_logger = None
        val_logger = None
        if self.out_dir and self.cfg.write_controller_ticks:
            ticks_logger = self._make_tick_logger(self.out_dir / "controller_tick.parquet")
        if self.out_dir and self.cfg.write_train_val_logs:
            train_logger = make_train_parquet_logger(self.out_dir / "logs_train.parquet")
            val_logger = make_val_parquet_logger(self.out_dir / "logs_val.parquet")

        runtime = ControllerRuntime(
            controller=controller,
            extractor=extractor,
            scheduler=ctrl_scheduler,
            tick_logger=ticks_logger,
            device=self.device,
            cfg=RuntimeConfig(
                update_interval=self.ctrl_cfg.update_interval,
                cooldown=self.ctrl_cfg.cooldown,
                action_ema=self.ctrl_cfg.action.ema,
                nan_guard=True,
                max_abs_delta=self.ctrl_cfg.controller_arch.hidden * 0.0 or None,
            ),
            run_id="evolve_eval",
            seed=self.budget_cfg.fixed_seed,
            arch=self.train_cfg.model.arch,
            dataset=self.train_cfg.data.dataset if hasattr(self.train_cfg.data, "dataset") else "CIFAR10",
        )

        # collectors for fitness
        val_curve: List[Tuple[int, float]] = []
        action_stream: List[float] = []
        diverged_flag = {"val": False}  # small mutable holder

        collectors = HookList([
            _BaseSchedulerStepHook(ctrl_scheduler), # step base scheduler at each batch
            extractor, # keep extractor updated
            runtime, # apply controller decisions
            _CollectEvalPointsHook(val_curve), # gather (step, val_acc)
            _CollectActionDeltasHook(action_stream), # gather delta log lr
            _DivergenceGuardHook(), # detect nans/explosions
        ])

        # optional scalar parquet logging (train/val)
        if train_logger is not None or val_logger is not None:
            collectors.add(self._make_scalar_logging_hook(train_logger, val_logger))

        # build loss & trainer (no scheduler here; base stepping is handled by the hook)
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=None, # base scheduler stepped via hook
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            hooks=collectors,
            epochs=self.budget_cfg.epochs,
            max_steps=self.budget_cfg.max_steps,
            mixed_precision=False,
            grad_clip_norm=None,
            metric_fn=None, # compute val acc in validation path, not here
            cfg=self.train_cfg,
        )

        summary = trainer.fit()
        final_val = float(summary.get("final_val_acc", 0.0))

        # TODO: Refactor this bc it's icky, it should come from the guard
        # divergence detection: engine already ran so check state from guard if needed
        # the guard sets state['diverged'] mid-run. since we don't have that state here,
        # decide divergence from the collected curve being empty or last val being NaN/Inf
        diverged = (len(val_curve) == 0) or (math.isnan(final_val) or math.isinf(final_val))

        run_summary = RunSummary(
            val_acc_curve=val_curve,
            action_deltas=action_stream,
            final_val_acc=0.0 if (math.isnan(final_val) or math.isinf(final_val)) else final_val,
            diverged=diverged,
            total_steps=int(summary.get("final_step", 0)),
            total_epochs=int(summary.get("final_epoch", 0)),
        )

        return {
            "summary": run_summary,
            "final_val_acc": run_summary.final_val_acc,
            "val_curve_points": len(run_summary.val_acc_curve),
            "num_actions": len(run_summary.action_deltas),
            "diverged": run_summary.diverged,
        }

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        name = self.train_cfg.optim.name.lower()
        lr = float(self.train_cfg.optim.lr)
        wd = float(self.train_cfg.optim.weight_decay)
        if name == "sgd":
            momentum = float(getattr(self.train_cfg.optim, "momentum", 0.0) or 0.0)
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=False)
        elif name == "adam":
            betas = getattr(self.train_cfg.optim, "betas", (0.9, 0.999)) or (0.9, 0.999)
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        elif name == "adamw":
            betas = getattr(self.train_cfg.optim, "betas", (0.9, 0.999)) or (0.9, 0.999)
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        else:
            raise ValueError(f"Unsupported optimizer '{self.train_cfg.optim.name}' in evaluator.")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        name = (self.train_cfg.sched.name or "none").lower()
        if name in ("none", "off", "disabled"):
            return None
        if name == "cosine":
            t_max = self.train_cfg.sched.t_max or max(1, int(self.budget_cfg.max_steps or self.budget_cfg.epochs * len(self._peek_loader_len())))
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        if name == "step":
            step_size = int(self.train_cfg.sched.step_size or 30)
            gamma = float(self.train_cfg.sched.gamma or 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if name == "multistep":
            milestones = [int(m) for m in (self.train_cfg.sched.milestones or [])]
            gamma = float(self.train_cfg.sched.gamma or 0.1)
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return None

    def _peek_loader_len(self) -> Tuple[int,]:
        # helper when deriving T_max, returns (approx_len,)
        try:
            # a rough proxy can be batch_count in a fresh data build
            # TODO: this could be cached if needed and also could be a real calculation rather than a proxy
            train_loader, _ = self._data_builder(
                self.train_cfg.data.data_root,
                self.train_cfg.data.batch_size,
                self.train_cfg.data.num_workers,
                self.budget_cfg.fixed_subset_pct,
                self.budget_cfg.fixed_seed,
            )
            return (len(train_loader),)
        except Exception:
            return (1000,)

    def _make_tick_logger(self, path: Path) -> ControllerTickLogger:
        return ControllerTickLogger.to_parquet(path)

    def _make_scalar_logging_hook(self, train_logger, val_logger) -> Hook:
        """
        Emit small Parquet scalar logs during evolution runs, this can be optional
        """
        def log_train(s: Dict[str, Any]) -> None:
            if train_logger is None:
                return
            row = {
                "global_step": int(s.get("global_step", 0)),
                "epoch": int(s.get("epoch", 0)),
                "loss": float(s.get("loss", 0.0) or 0.0),
                "acc": float(s.get("acc", 0.0) or 0.0),
                "lr": float(s.get("lr", 0.0) or 0.0),
                "grad_norm": float(s.get("grad_norm", 0.0) or 0.0),
            }
            train_logger.log(row)

        def log_val(state: Dict[str, Any]) -> None:
            if val_logger is None:
                return
            row = {
                "global_step": int(state.get("global_step", 0)),
                "epoch": int(state.get("epoch", 0)),
                "val_loss": float(state.get("val_loss", 0.0) or 0.0),
                "val_acc": float(state.get("val_acc", 0.0) or 0.0),
            }
            val_logger.log(row)

        return HookList([
            LambdaHook(on_batch_end=log_train),
            LambdaHook(on_eval_end=log_val),
        ])
