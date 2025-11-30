from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.controller.controller import ControllerMLP
# controller
from src.controller.features import FeatureExtractor
from src.controller.lr_scheduler import LrControllerScheduler
from src.controller.runtime import ControllerRuntime, RuntimeConfig
from src.controller.serialization import controller_version_hash_from_vector
from src.controller.serialization import unflatten_params
from src.core.hooks.common_hooks import LambdaHook
# hooks and training engine
from src.core.hooks.hook_base import Hook
from src.core.hooks.hook_composition import HookList
# logging and fitness
from src.core.logging.loggers import (
    ControllerTickLogger,
    make_train_parquet_logger,
    make_val_parquet_logger,
    ContextLogger,
)
from src.data_.cifar10 import get_dataloaders
from src.evolution.fitness import RunSummary, score_run, FitnessWeights, auc
from src.models.resnet_cifar10 import resnet20
from src.training.checkpoints import CheckpointIO
from src.training.engine import Trainer
from src.utils.seed_device import get_device, seed_everything
from src.training.optim_sched_factory import build_optimizer, build_baseline_scheduler


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

    save_candidate_models: bool = False

    # inject fitness weights & static IDs to be stamped on parquet rows
    fitness_weights: Optional[FitnessWeights] = None
    static_ids: Optional[Dict[str, Any]] = None  # set per-candidate from runner


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
        self.cfg = evaluator_cfg or EvaluatorConfig()

        # builders
        self._model_builder = self.cfg.model_builder or (lambda arch, nc: resnet20())
        self._data_builder = self.cfg.dataloaders_builder or (
            lambda root, bs, nw, pct, seed: get_dataloaders(
                root=root,
                batch_size=bs,
                num_workers=nw,
                augment=getattr(self.train_cfg.data, "augment", True),
                subset_fraction=pct,
                subset_seed=seed,
            )
        )

        # optional output dir for Parquet logs
        self.out_dir = Path(self.cfg.out_dir) if self.cfg.out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        # fitness weights defaults
        self.fitness_weights: FitnessWeights = self.cfg.fitness_weights or FitnessWeights(
            primary="auc_val_acc",
            lr_volatility_weight=0.1,
            nan_penalty=100.0,
        )

    def evaluate_result(self, candidate: Any, static_ids: Optional[Dict[str, Any]] = None) -> EvalResult:
        raw = self.evaluate(candidate, static_ids=static_ids)
        if isinstance(raw, EvalResult):
            return raw
        elif isinstance(raw, dict):
            return EvalResult(
                fitness_primary=float(raw.get("fitness", raw.get("fitness_primary", 0.0))),
                primary_metric=str(raw.get("primary_metric", self.fitness_weights.primary)),
                fitness_vector=list(raw.get("fitness_vector", [])) if raw.get("fitness_vector") is not None else [],
                penalties=dict(raw.get("penalties", {})),
                metrics_snapshot=dict(raw.get("metrics_snapshot", {})),
                budget_used=dict(raw.get("budget_used", {"epochs": 0, "steps": 0, "wall_time_s": 0.0})),
                truncation_reason=str(raw.get("truncation_reason", "complete")),
                artifacts=dict(raw.get("artifacts", {})) if raw.get("artifacts") else {},
            )
        else:
            try:
                fitness_val = float(raw)
            except Exception:
                fitness_val = 0.0
            return EvalResult(fitness_primary=fitness_val, primary_metric=self.fitness_weights.primary)

    def evaluate(self, controller_vector: Any, static_ids: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # deterministic budget for fair comparison across candidates
        seed_everything(self.budget_cfg.fixed_seed)

        # Data / Model
        subset_fraction = float(getattr(self.budget_cfg, "fixed_subset_pct", 1.0) or 1.0)
        subset_seed = int(getattr(self.budget_cfg, "fixed_seed", 2025))

        train_loader, val_loader = self._data_builder(
            root=self.train_cfg.data.data_root,
            bs=self.train_cfg.data.batch_size,
            nw=self.train_cfg.data.num_workers,
            pct=subset_fraction,
            seed=subset_seed,
        )

        model = resnet20().to(self.device)

        # optional warm start from checkpoint, controlled by evolve.budget
        start_from_ckpt = bool(getattr(self.budget_cfg, "start_from_checkpoint", False))
        ckpt_path = getattr(self.budget_cfg, "checkpoint_path", None)

        if start_from_ckpt and ckpt_path:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.is_file():
                print(
                    f"[evolve] WARNING: start_from_checkpoint=True but "
                    f"checkpoint_path='{ckpt_path}' does not exist; using random init."
                )
            else:
                try:
                    bundle = torch.load(ckpt_path, map_location="cpu")
                    # handle both {"model_state": state_dict} and raw state_dict
                    state_dict = bundle.get("model_state", bundle)
                    model.load_state_dict(state_dict, strict=True)
                    print(f"[evolve] Loaded model checkpoint from '{ckpt_path}'.")
                except Exception as e:
                    print(
                        f"[evolve] WARNING: failed to load checkpoint from '{ckpt_path}': {e}. "
                        "Using random init."
                    )

        # optimizer & base scheduler
        optimizer = build_optimizer(model, self.train_cfg.optim)
        steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else 0
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

        # Controller vector -> tensor on device
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

        # parquet writers (wrapped to inject IDs)
        # merge external static_ids with a computed controller_version from the vector.
        version_tag = controller_version_hash_from_vector(controller_vector)
        merged_ids: Dict[str, Any] = dict(self.cfg.static_ids or {})
        merged_ids.update(static_ids or {})
        merged_ids.setdefault("controller_version", version_tag)

        ticks_logger = None
        train_logger = None
        val_logger = None
        if self.out_dir and self.cfg.write_controller_ticks:
            # use ContextTickLogger so runtime can call .log_tick(...)
            from src.core.logging.loggers import ControllerTickLogger, ContextTickLogger
            base_tick = ControllerTickLogger.to_parquet(self.out_dir / "controller_calls.parquet")
            ticks_logger = ContextTickLogger(base_tick, static_fields=merged_ids)
        if self.out_dir and self.cfg.write_train_val_logs:
            from src.core.logging.loggers import make_train_parquet_logger, make_val_parquet_logger, ContextLogger
            train_logger = ContextLogger(make_train_parquet_logger(self.out_dir / "logs_train.parquet"), merged_ids)
            val_logger = ContextLogger(make_val_parquet_logger(self.out_dir / "logs_val.parquet"), merged_ids)

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
            extractor,
            runtime,
            _CollectEvalPointsHook(val_curve),
            _CollectActionDeltasHook(action_stream),
            _DivergenceGuardHook(),
        ])

        # scalar parquet logging via hooks
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

        # ensure parquet writers close even if training crashes
        try:
            summary = trainer.fit()
        finally:
            for w in (train_logger, val_logger, ticks_logger):
                try:
                    if w is not None and hasattr(w, "close"):
                        try:
                            w.flush()
                        except Exception as e:
                            print("WARNING: failed to flush logger:\n", e)
                            pass
                        w.close()
                except Exception as e:
                    print("WARNING: failed to close logger:\n", e)
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

        # optionally save the ResNet model for this candidate
        if getattr(self.cfg, "save_candidate_models", False):
            cid = None
            if static_ids is not None:
                cid = static_ids.get("candidate_id", None)

            if cid is not None and self.out_dir is not None:
                try:
                    models_dir = self.out_dir / "models"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    model_path = models_dir / f"model_{cid}.pt"
                    torch.save({"model_state": model.state_dict()}, model_path)
                except Exception as e:
                    print(f"WARNING: failed to save model for candidate {cid}: {e}")

        # fitness
        fitness_score = score_run(run_summary, self.fitness_weights)
        auc_val = float(auc(val_curve)) if len(val_curve) >= 2 else 0.0
        lr_vol = float(np.std(action_stream)) if len(action_stream) >= 2 else 0.0

        return {
            "fitness": fitness_score,
            "fitness_primary": fitness_score,
            "primary_metric": self.fitness_weights.primary,
            "summary": run_summary,
            "final_val_acc": run_summary.final_val_acc,
            "val_curve_points": len(run_summary.val_acc_curve),
            "num_actions": len(run_summary.action_deltas),
            "diverged": run_summary.diverged,
            "metrics_snapshot": {
                "auc_val_acc": auc_val,
                "final_val_acc": run_summary.final_val_acc,
                "lr_delta_std": lr_vol,
                "diverged": float(run_summary.diverged),
                "total_steps": float(run_summary.total_steps),
                "total_epochs": float(run_summary.total_epochs),
            },
        }

    def _make_scalar_logging_hook(self, train_logger, val_logger) -> Hook:
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
            except Exception as e:
                print("WARNING: failed to log train:\n", e)
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
            except Exception as e:
                print("WARNING: failed to log val:\n", e)
                pass

        return HookList([
            LambdaHook(on_batch_end=log_train),
            LambdaHook(on_eval_end=log_val),
        ])
