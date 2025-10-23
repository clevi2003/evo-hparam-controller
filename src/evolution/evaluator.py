from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

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

# logging and fitness
from src.core.logging.loggers import (
    ControllerTickLogger,
    make_train_parquet_logger,
    make_val_parquet_logger,
)
from src.evolution.fitness import RunSummary
from src.training.checkpoints import CheckpointIO
from src.data_.cifar10 import get_dataloaders
from src.models.resnet_cifar10 import resnet20
from src.training.optim_sched_factory import build_optimizer, build_baseline_scheduler



class _BaseSchedulerStepHook(Hook):
    """step the base LR schedule each batch before the controller applies its action"""
    def __init__(self, ctrl_scheduler: LrControllerScheduler) -> None:
        self.ctrl_scheduler = ctrl_scheduler
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        self.ctrl_scheduler.step_base()


class _CollectEvalPointsHook(Hook):
    """collect (global_step, val_acc) at the end of each evaluation"""
    def __init__(self, collector: List[Tuple[int, float]]) -> None:
        self.collector = collector
    def on_eval_end(self, state: Dict[str, Any]) -> None:
        val_acc = state.get("val_acc", None)
        if val_acc is not None:
            self.collector.append((int(state.get("global_step", 0)), float(val_acc)))


class _CollectActionDeltasHook(Hook):
    """collect applied delta(log lr) each time the controller acts"""
    def __init__(self, collector: List[float]) -> None:
        self.collector = collector
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        if state.get("controller_applied", False):
            d = state.get("controller_delta_applied", None)
            if d is not None:
                self.collector.append(float(d))


class _DivergenceGuardHook(Hook):
    """detect nans/infs in loss; sets state['diverged']=True"""
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        loss = state.get("loss", None)
        if loss is None:
            return
        try:
            v = float(loss)
        except Exception:
            state["diverged"] = True
            return
        if math.isnan(v) or math.isinf(v):
            state["diverged"] = True


def _default_build_cifar10_dataloaders(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    subset_pct: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    try:
        import torchvision
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError(
            "Default CIFAR-10 data builder requires torchvision. "
            "Install torchvision or provide a custom builder."
        ) from e

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(root=str(data_root), train=True,  download=True, transform=tf_train)
    test_set  = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=tf_test)

    if subset_pct < 1.0:
        total = len(train_set)
        keep = max(1, int(total * subset_pct))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(total, generator=g)[:keep]
        train_set = Subset(train_set, idx.tolist())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def _default_build_cifar_model() -> torch.nn.Module:
    try:
        return resnet20()
    except Exception as e:
        raise RuntimeError("Could not build CIFAR-10 model (resnet20).") from e


@dataclass
class EvaluatorConfig:
    model_builder: Optional[Callable[[str, int], torch.nn.Module]] = None
    dataloaders_builder: Optional[
        Callable[[str | Path, int, int, float, int], Tuple[DataLoader, DataLoader]]
    ] = None
    out_dir: Optional[str | Path] = None
    write_controller_ticks: bool = True
    write_train_val_logs: bool = True
    checkpoint_io: Optional[CheckpointIO] = None


@dataclass
class EvalResult:
    fitness_primary: float
    primary_metric: str = "fitness"
    fitness_vector: Optional[List[float]] = None
    penalties: Dict[str, float] = None
    metrics_snapshot: Dict[str, float] = None
    budget_used: Dict[str, float] = None
    truncation_reason: str = "complete"
    artifacts: Optional[Dict[str, str]] = None
    def __post_init__(self) -> None:
        if self.penalties is None:
            self.penalties = {}
        if self.metrics_snapshot is None:
            self.metrics_snapshot = {}
        if self.budget_used is None:
            self.budget_used = {"epochs": 0, "steps": 0, "wall_time_s": 0.0}
        if self.fitness_vector is None:
            self.fitness_vector = []
        if self.artifacts is None:
            self.artifacts = {}


class TruncatedTrainingEvaluator:
    """
    run a short, deterministic training budget for a controller candidate vector
    produces a RunSummary used as fitness
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
        self.device = get_device(getattr(train_cfg, "device", None))
        print("DEVICE:", self.device)
        self.evaluator_cfg = evaluator_cfg
        self.cfg = evaluator_cfg or EvaluatorConfig()

        # builders
        self._model_builder = self.cfg.model_builder or (lambda arch, nc: _default_build_cifar_model())
        self._data_builder = self.cfg.dataloaders_builder or (
            lambda root, bs, nw, pct, seed: _default_build_cifar10_dataloaders(root, bs, nw, pct, seed)
        )

        # optional output dir for Parquet logs
        self.out_dir = Path(self.cfg.out_dir) if self.cfg.out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_result(self, candidate: Any) -> EvalResult:
        raw = self.evaluate(candidate)
        if isinstance(raw, EvalResult):
            res = raw
        elif isinstance(raw, dict):
            res = EvalResult(
                fitness_primary=float(raw.get("fitness", raw.get("fitness_primary", 0.0))),
                primary_metric=str(raw.get("primary_metric", "fitness")),
                fitness_vector=list(raw.get("fitness_vector", [])) if raw.get("fitness_vector") is not None else [],
                penalties=dict(raw.get("penalties", {})),
                metrics_snapshot=dict(raw.get("metrics_snapshot", {})),
                budget_used=dict(raw.get("budget_used", {"epochs": 0, "steps": 0, "wall_time_s": 0.0})),
                truncation_reason=str(raw.get("truncation_reason", "complete")),
                artifacts=dict(raw.get("artifacts", {})) if raw.get("artifacts") else {},
            )
        else:
            # scalar fallback
            try:
                fitness_val = float(raw)
            except Exception:
                fitness_val = 0.0
            res = EvalResult(fitness_primary=fitness_val, primary_metric="fitness")

        # optional truncated checkpoint artifact
        ckpt = self.cfg.checkpoint_io
        if ckpt is not None:
            try:
                vec = getattr(candidate, "vec", None)
                if vec is not None:
                    ckpt.save_warmup({"controller_vec": vec})
            except Exception:
                pass
            rel = ckpt.save_final(model_state=None, optimizer_state=None, scheduler_state=None, extra={"mode": "truncated"})
            res.artifacts = dict(res.artifacts or {})
            res.artifacts["final"] = rel

        return res

    def evaluate(self, controller_vector: Any) -> Dict[str, Any]:
        # deterministic budget for fair comparison across candidates
        seed_everything(self.budget_cfg.fixed_seed)

        # # Data
        # train_loader, val_loader = self._data_builder(
        #     self.train_cfg.data.data_root,
        #     self.train_cfg.data.batch_size,
        #     self.train_cfg.data.num_workers,
        #     self.budget_cfg.fixed_subset_pct,
        #     self.budget_cfg.fixed_seed,
        # )
        #
        # # Model
        # model = self._model_builder(self.train_cfg.model.arch, self.train_cfg.model.num_classes).to(self.device)
        train_loader, val_loader = get_dataloaders(
            root=self.train_cfg.data.data_root,
            batch_size=self.train_cfg.data.batch_size,
            num_workers=self.train_cfg.data.num_workers,
            # fall back to True if your DataCfg doesn’t define `augment`
            augment=getattr(self.train_cfg.data, "augment", True),
        )
        model = resnet20().to(self.device)


        # optimizer & base scheduler
        optimizer = build_optimizer(model, self.train_cfg.optim)

        # derive steps_per_epoch for schedulers that need it
        steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else 0
        epochs_for_sched = int(self.budget_cfg.epochs or 1)

        sched_handle = build_baseline_scheduler(
            optimizer=optimizer,
            sched_cfg=self.train_cfg.sched,
            optim_cfg=self.train_cfg.optim,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs_for_sched,
        )

        base_scheduler = sched_handle.scheduler  # may be None
        step_when = sched_handle.step_when

        # Controller vector coerce to 1d torch tensor on device
        if isinstance(controller_vector, np.ndarray):
            controller_vector = torch.as_tensor(controller_vector, dtype=torch.float32, device=self.device).view(-1)
        elif isinstance(controller_vector, torch.Tensor):
            controller_vector = controller_vector.detach().to(self.device, dtype=torch.float32).view(-1)
        else:
            raise TypeError(f"Unsupported candidate type: {type(controller_vector)}")

        # controller
        in_dim = len(self.ctrl_cfg.features)
        controller = ControllerMLP(
            in_dim=in_dim,
            hidden=self.ctrl_cfg.controller_arch.hidden,
            max_step=self.ctrl_cfg.action.max_step,
        ).to(self.device)
        unflatten_params(controller, controller_vector, strict=True)

        # features, controller scheduler, runtime
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

        # optional Parquet writers
        ticks_logger = None
        train_logger = None
        val_logger = None
        if self.out_dir and self.cfg.write_controller_ticks:
            ticks_logger = self._make_tick_logger(self.out_dir / "controller_calls.parquet")
        if self.out_dir and self.cfg.write_train_val_logs:
            train_logger = make_train_parquet_logger(self.out_dir / "logs_train.parquet")
            val_logger   = make_val_parquet_logger(self.out_dir / "logs_val.parquet")

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
                max_abs_delta=None,
            ),
            run_id="evolve_eval",
            seed=self.budget_cfg.fixed_seed,
            arch=self.train_cfg.model.arch,
            dataset=getattr(self.train_cfg.data, "dataset", "CIFAR10"),
        )

        # fitness collectors
        val_curve: List[Tuple[int, float]] = []
        action_stream: List[float] = []

        collectors = HookList([
            # _BaseSchedulerStepHook(ctrl_scheduler), # step base on every batch
            extractor, # keep features updated
            runtime, # controller decisions
            _CollectEvalPointsHook(val_curve), # (step, val_acc)
            _CollectActionDeltasHook(action_stream), # delta log lr
            _DivergenceGuardHook(), # nan/inf guard
        ])

        # train/val scalar parquet logging via hooks
        if (train_logger is not None) or (val_logger is not None):
            collectors.add(self._make_scalar_logging_hook(train_logger, val_logger))

        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=base_scheduler,
            scheduler_step_when=step_when,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            hooks=collectors,
            epochs=self.budget_cfg.epochs,
            max_steps=self.budget_cfg.max_steps,
            mixed_precision=getattr(self.train_cfg, "amp", False),
            grad_clip=float(getattr(getattr(self.train_cfg, "optim", {}), "grad_clip_norm", 0.0) or 0.0) or None,
            cfg=self.train_cfg,
        )

        # make sure parquet writers flush/close even if training crashes
        try:
            summary = trainer.fit()
        finally:
            for w in (train_logger, val_logger, ticks_logger):
                try:
                    if w is not None and hasattr(w, "close"):
                        w.close()
                except Exception:
                    pass

        final_val = float(summary.get("final_val_acc", 0.0))
        diverged = (len(val_curve) == 0) or math.isnan(final_val) or math.isinf(final_val)

        run_summary = RunSummary(
            val_acc_curve=val_curve,
            action_deltas=action_stream,
            final_val_acc=0.0 if diverged else final_val,
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


    def _peek_loader_len(self) -> int:
        try:
            train_loader, _ = self._data_builder(
                self.train_cfg.data.data_root,
                self.train_cfg.data.batch_size,
                self.train_cfg.data.num_workers,
                self.budget_cfg.fixed_subset_pct,
                self.budget_cfg.fixed_seed,
            )
            return len(train_loader)
        except Exception:
            return 1000

    def _make_tick_logger(self, path: Path) -> ControllerTickLogger:
        return ControllerTickLogger.to_parquet(path)

    def _make_scalar_logging_hook(self, train_logger, val_logger) -> Hook:
        """
        emit small Parquet scalar logs during evolution runs
        """
        def log_train(s: Dict[str, Any]) -> None:
            if train_logger is None:
                return
            try:
                train_logger.log({
                    "global_step": int(s.get("global_step", 0)),
                    "epoch":       int(s.get("epoch", 0)),
                    "loss":        float(s.get("loss", 0.0) or 0.0),
                    "acc":         float(s.get("acc", 0.0) or 0.0),
                    "lr":          float(s.get("lr", 0.0) or 0.0),
                    "grad_norm":   float(s.get("grad_norm", 0.0) or 0.0),
                })
            except Exception:
                pass

        def log_val(s: Dict[str, Any]) -> None:
            if val_logger is None:
                return
            try:
                val_logger.log({
                    "global_step": int(s.get("global_step", 0)),
                    "epoch":       int(s.get("epoch", 0)),
                    "val_loss":    float(s.get("val_loss", 0.0) or 0.0),
                    "val_acc":     float(s.get("val_acc", 0.0) or 0.0),
                })
            except Exception:
                pass

        return HookList([
            LambdaHook(on_batch_end=log_train),
            LambdaHook(on_eval_end=log_val),
        ])
