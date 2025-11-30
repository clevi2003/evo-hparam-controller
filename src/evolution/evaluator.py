from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.controller.controller import ControllerMLP
from src.controller.features import FeatureExtractor
from src.controller.lr_scheduler import LrControllerScheduler
from src.controller.runtime import ControllerRuntime, RuntimeConfig
from src.controller.serialization import controller_version_hash_from_vector
from src.controller.serialization import unflatten_params
from src.evolution.fitness import RunSummary, score_run, FitnessWeights, auc
from src.models.resnet_cifar10 import resnet20
from src.training.checkpoints import CheckpointIO
from src.training.engine import Trainer
from src.utils.seed_device import get_device, seed_everything
from src.training.optim_sched_factory import build_optimizer, build_baseline_scheduler
from src.core.hooks.hook_base import Hook
from src.core.hooks.hook_composition import HookList


class _CollectEvalPointsHook(Hook):
    """Collect (global_step, val_acc) at the end of each evaluation."""
    def __init__(self, collector: List[Tuple[int, float]]) -> None:
        self.collector = collector

    def on_eval_end(self, state: Dict[str, Any]) -> None:
        val_acc = state.get("val_acc", None)
        if val_acc is not None:
            self.collector.append((int(state.get("global_step", 0)), float(val_acc)))


class _CollectActionDeltasHook(Hook):
    """Collect applied delta(log lr) each time the controller acts."""
    def __init__(self, collector: List[float]) -> None:
        self.collector = collector

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        if state.get("controller_applied", False):
            d = state.get("controller_delta_applied", None)
            if d is not None:
                self.collector.append(float(d))


class _DivergenceGuardHook(Hook):
    """Detect NaNs/Infs in loss; sets state['diverged']=True."""
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


@dataclass
class EvaluatorConfig:
    model_builder: Optional[Callable[[str, int], torch.nn.Module]] = None
    dataloaders_builder: Optional[
        Callable[[str | Path, int, int, float, int], Tuple[DataLoader, DataLoader]]
    ] = None
    out_dir: Optional[str | Path] = None
    write_controller_ticks: bool = False  # disable per-batch logging
    write_train_val_logs: bool = False    # disable per-epoch logging
    checkpoint_io: Optional[CheckpointIO] = None
    save_candidate_models: bool = False
    fitness_weights: Optional[FitnessWeights] = None
    static_ids: Optional[Dict[str, Any]] = None


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
    """Run short, deterministic training for a candidate vector to produce fitness."""

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
        self.cfg = evaluator_cfg or EvaluatorConfig()

        # Builders
        self._model_builder = self.cfg.model_builder or (lambda arch, nc: resnet20())
        self._data_builder = self.cfg.dataloaders_builder or (lambda root, bs, nw, pct, seed: 
            __import__('src.data_.cifar10').get_dataloaders(
                root=root, batch_size=bs, num_workers=nw, augment=True, subset_fraction=pct, subset_seed=seed
            )
        )

        self.fitness_weights: FitnessWeights = self.cfg.fitness_weights or FitnessWeights(
            primary="auc_val_acc",
            lr_volatility_weight=0.1,
            nan_penalty=100.0,
        )


    def evaluate_result(self, candidate: Any, static_ids: Optional[Dict[str, Any]] = None) -> EvalResult:
        raw = self.evaluate(candidate, static_ids=static_ids)
        fitness_score = float(raw.get("fitness", 0.0))
        auc_val = float(raw["metrics_snapshot"].get("auc_val_acc", 0.0))
        lr_vol = float(raw["metrics_snapshot"].get("lr_delta_std", 0.0))
        return EvalResult(
            fitness_primary=fitness_score,
            primary_metric=self.fitness_weights.primary,
            metrics_snapshot={
                "auc_val_acc": auc_val,
                "lr_delta_std": lr_vol,
            },
        )


    def evaluate(self, controller_vector: Any, static_ids: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Deterministic seed
        seed_everything(self.budget_cfg.fixed_seed)

        subset_fraction = float(getattr(self.budget_cfg, "fixed_subset_pct", 1.0) or 1.0)
        subset_seed = int(getattr(self.budget_cfg, "fixed_seed", 2025))

        train_loader, val_loader = self._data_builder(
            root=self.train_cfg.data.data_root,
            bs=self.train_cfg.data.batch_size,
            nw=self.train_cfg.data.num_workers,
            pct=subset_fraction,
            seed=subset_seed,
        )

        model = self._model_builder(self.train_cfg.model.arch, 10).to(self.device)

        # Optimizer & Scheduler
        optimizer = build_optimizer(model, self.train_cfg.optim)
        steps_per_epoch = len(train_loader)
        epochs_for_sched = int(self.budget_cfg.epochs or 1)
        sched_handle = build_baseline_scheduler(
            optimizer=optimizer,
            sched_cfg=self.train_cfg.sched,
            optim_cfg=self.train_cfg.optim,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs_for_sched,
        )
        base_scheduler = sched_handle.scheduler
        step_when = sched_handle.step_when

        # Controller
        if isinstance(controller_vector, np.ndarray):
            controller_vector = torch.as_tensor(controller_vector, dtype=torch.float32, device=self.device).view(-1)
        elif isinstance(controller_vector, torch.Tensor):
            controller_vector = controller_vector.detach().to(self.device, dtype=torch.float32).view(-1)
        else:
            raise TypeError(f"Unsupported candidate type: {type(controller_vector)}")

        in_dim = len(self.ctrl_cfg.features)
        controller = ControllerMLP(
            in_dim=in_dim,
            hidden=self.ctrl_cfg.controller_arch.hidden,
            max_step=self.ctrl_cfg.action.max_step,
        ).to(self.device)
        unflatten_params(controller, controller_vector, strict=True)

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

        val_curve: List[Tuple[int, float]] = []
        action_stream: List[float] = []

        collectors = HookList([
            extractor,
            _CollectEvalPointsHook(val_curve),
            _CollectActionDeltasHook(action_stream),
            _DivergenceGuardHook(),
        ])

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

        summary = trainer.fit()

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

        fitness_score = score_run(run_summary, self.fitness_weights)
        auc_val = float(auc(val_curve)) if len(val_curve) >= 2 else 0.0
        lr_vol = float(np.std(action_stream)) if len(action_stream) >= 2 else 0.0

        return {
            "fitness": fitness_score,
            "summary": run_summary,
            "metrics_snapshot": {
                "auc_val_acc": auc_val,
                "lr_delta_std": lr_vol,
            },
        }
