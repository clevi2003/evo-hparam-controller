from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.core.hooks.hook_base import Hook, NullHook
from src.utils import compute_grad_norm, detach_scalar, get_current_lr
from src.core.logging.loggers import (
    make_train_parquet_logger,
    make_val_parquet_logger,
    ControllerTickLogger,
)
import uuid
import json
import subprocess
import sys
from datetime import datetime
import yaml

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


def _default_accuracy_fn(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy for classification. If logits are not 2D, returns 0.0.
    """
    if logits.dim() < 2:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct / max(1, total))


class Trainer:
    """
    Hook-driven training engine.

    Responsibilities:
      Run train & validation loops for a given number of epochs (or max_steps cap)
      Maintain a shared `state` dict and call Hook events at well-defined points
      Provide optional AMP and gradient clipping
      Expose a fit() API that returns summary metrics

    The engine is intentionally unopinionated about logging—use hooks for that
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        hooks: Optional[Hook],
        epochs: int,
        max_steps: Optional[int] = None,
        mixed_precision: bool = False,
        grad_clip_norm: Optional[float] = None,
        metric_fn: Optional[MetricFn] = _default_accuracy_fn,
        cfg: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = int(epochs)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.grad_clip_norm = grad_clip_norm
        self.metric_fn = metric_fn
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.hooks = hooks or NullHook()
        self.cfg = cfg

        # runtime counters
        self.global_step = 0

        # move model
        self.model.to(self.device)

        # logging placeholders (created lazily when fit() is called)
        self._train_logger = None
        self._val_logger = None
        self._controller_logger = None


    def fit(self) -> Dict[str, float]:
        """
        Run the training process. Returns summary dict with best/final metrics.
        Keys: best_val_acc, final_val_acc, final_train_acc, final_epoch, final_step
        """
        state: Dict[str, Any] = {
            "epoch": 0,
            "global_step": 0,
            "batch_idx": -1,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "device": self.device,
            "cfg": self.cfg,
            # scalars updated during run
            "loss": None,
            "acc": None,
            "val_loss": None,
            "val_acc": None,
            "lr": get_current_lr(self.optimizer),
            "grad_norm": None,
            # logging hooks (populated with Logger / ControllerTickLogger instances)
            "train_logger": None,
            "val_logger": None,
            "controller_logger": None,
        }

        best_val_acc = float("-inf")

        self.hooks.on_train_start(state)

        # --- initialize run metadata and loggers ---
        cfg = self.cfg or {}

        # create a timestamped run directory unless caller provided a final run_dir
        run_id = uuid.uuid4().hex[:8]
        start_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if isinstance(cfg, dict) and cfg.get("run_dir"):
            # treat provided run_dir as a runs root and always create a dated child folder
            run_root = Path(cfg.get("run_dir"))
            run_dir = run_root / f"{start_ts}_{run_id}"
        else:
            run_dir = Path(os.getcwd()) / "runs" / f"{start_ts}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # ensure subfolders follow the canonical layout
        (run_dir / "logs_val").mkdir(parents=True, exist_ok=True)
        (run_dir / "Checkpoints").mkdir(parents=True, exist_ok=True)

        # collect run-level metadata
        def _git_commit():
            try:
                return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            except Exception:
                return ""

        def _git_branch():
            try:
                return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            except Exception:
                return ""

        env = {
            "python_version": sys.version.replace("\n", " "),
            "torch_version": getattr(torch, "__version__", ""),
            "cuda_version": getattr(torch.version, "cuda", None) if hasattr(torch, "version") else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends and hasattr(torch.backends, "cudnn") else None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None,
            "git_commit": _git_commit(),
            "git_branch": _git_branch(),
            "start_time": start_ts,
            "mixed_precision": bool(self.mixed_precision),
        }

        # run-level meta: configuration summary, model/opt/scheduler descriptors, dataset and seeds
        def _summarize_optimizer(opt):
            try:
                return {"class": opt.__class__.__name__, "param_groups": [{k: v for k, v in pg.items() if k != "params"} for pg in opt.param_groups]}
            except Exception:
                return {"class": str(type(opt))}

        def _summarize_scheduler(sched):
            try:
                return {"class": sched.__class__.__name__}
            except Exception:
                return {"class": str(type(sched))}

        batch_size = getattr(self.train_loader, "batch_size", None)
        dataset_name = getattr(getattr(self.train_loader, 'dataset', None), '__class__', None)
        dataset_name = dataset_name.__name__ if dataset_name is not None else None

        run_meta = {
            "run_id": run_id,
            "start_time": start_ts,
            "git_commit": env["git_commit"],
            "git_branch": env["git_branch"],
            "config": cfg if isinstance(cfg, dict) else str(cfg),
            "dataset": dataset_name,
            "model": self.model.__class__.__name__ if hasattr(self.model, "__class__") else str(type(self.model)),
            "optimizer": _summarize_optimizer(self.optimizer) if self.optimizer is not None else None,
            "scheduler": _summarize_scheduler(self.scheduler) if self.scheduler is not None else None,
            "batch_size": int(batch_size) if batch_size is not None else None,
            "epochs": int(self.epochs),
            "augment": (cfg.get("data", {}).get("augment") if isinstance(cfg, dict) else None),
            "seed": (cfg.get("seed") if isinstance(cfg, dict) else None),
            "env": env,
        }

        # persist config and run_meta to run_dir
        try:
            cfg_path = run_dir / "config.yaml"
            with cfg_path.open("w") as fh:
                if isinstance(cfg, dict):
                    yaml.safe_dump(cfg, fh)
                else:
                    fh.write(str(cfg))
        except Exception:
            pass

        try:
            env_path = run_dir / "env.json"
            with env_path.open("w") as fh:
                json.dump(env, fh, indent=2)
        except Exception:
            pass

        try:
            meta_path = run_dir / "run_meta.json"
            with meta_path.open("w") as fh:
                json.dump(run_meta, fh, indent=2)
        except Exception:
            pass

        train_log_path = run_dir / "Logs_train.parquet"
        val_log_path = run_dir / "logs_val" / "test.parquet"
        controller_log_path = run_dir / "Controller_calls.parquet"

        # create parquet loggers
        self._train_logger = make_train_parquet_logger(train_log_path)
        self._val_logger = make_val_parquet_logger(val_log_path)
        self._controller_logger = ControllerTickLogger.to_parquet(controller_log_path)

        # expose loggers and run_meta to hooks and controllers via state
        state["train_logger"] = self._train_logger
        state["val_logger"] = self._val_logger
        state["controller_logger"] = self._controller_logger
        state["run_meta"] = run_meta
        state["run_dir"] = str(run_dir)

        for epoch in range(self.epochs):
            state["epoch"] = epoch
            self.hooks.on_epoch_start(state)

            # Training epoch
            self.model.train(True)
            for batch_idx, batch in enumerate(self.train_loader):
                state["batch_idx"] = batch_idx

                # Early-stop on max_steps budget
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    # run validation once at budget boundary if available
                    if self.val_loader is not None:
                        self._run_validation(state)
                    self.hooks.on_train_end(state)
                    return {
                        "best_val_acc": float(max(best_val_acc, state.get("val_acc") or float("-inf"))),
                        "final_val_acc": float(state.get("val_acc") or 0.0),
                        "final_train_acc": float(state.get("acc") or 0.0),
                        "final_epoch": epoch,
                        "final_step": self.global_step,
                    }

                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.hooks.on_batch_start(state)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    logits = self.model(inputs)
                    loss = self.loss_fn(logits, targets)

                # backward
                self.optimizer.zero_grad(set_to_none=True)
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                # scheduler step
                if self.scheduler is not None and hasattr(self.scheduler, "step"):
                    # Many schedulers are designed to step per-iteration; this is the common case for cosine/warmup
                    self.scheduler.step()

                # update counters
                self.global_step += 1
                state["global_step"] = self.global_step

                # compute metrics
                acc = None
                if self.metric_fn is not None:
                    try:
                        acc = self.metric_fn(logits.detach(), targets.detach())
                    except Exception:
                        acc = None

                state["loss"] = detach_scalar(loss)
                state["acc"] = acc
                state["lr"] = get_current_lr(self.optimizer)
                state["grad_norm"] = compute_grad_norm(self.model)

                # write per-batch train row if logger present
                if self._train_logger is not None:
                    try:
                        row = {
                            "global_step": int(self.global_step),
                            "epoch": int(epoch),
                            "loss": float(state.get("loss") or 0.0),
                            "acc": float(state.get("acc") or 0.0),
                            "lr": float(state.get("lr") or 0.0),
                            "grad_norm": float(state.get("grad_norm") or 0.0),
                        }
                        self._train_logger.log(row)
                    except Exception:
                        # non-fatal: logging failure shouldn't stop training
                        pass

                self.hooks.on_batch_end(state)

            # Validation at epoch end
            if self.val_loader is not None:
                self._run_validation(state)
                if state["val_acc"] is not None:
                    best_val_acc = max(best_val_acc, float(state["val_acc"]))

                # after validation, write a val row
                if self._val_logger is not None:
                    try:
                        vrow = {
                            "global_step": int(self.global_step),
                            "epoch": int(state.get("epoch") or epoch),
                            "val_loss": float(state.get("val_loss") or 0.0),
                            "val_acc": float(state.get("val_acc") or 0.0),
                        }
                        self._val_logger.log(vrow)
                    except Exception:
                        pass

        # Train end
        self.hooks.on_train_end(state)

        # flush & close loggers
        try:
            if self._train_logger is not None:
                self._train_logger.flush()
                self._train_logger.close()
        except Exception:
            pass
        try:
            if self._val_logger is not None:
                self._val_logger.flush()
                self._val_logger.close()
        except Exception:
            pass
        try:
            if self._controller_logger is not None:
                self._controller_logger.flush()
                self._controller_logger.close()
        except Exception:
            pass

        return {
            "best_val_acc": float(best_val_acc if best_val_acc != float("-inf") else (state.get("val_acc") or 0.0)),
            "final_val_acc": float(state.get("val_acc") or 0.0),
            "final_train_acc": float(state.get("acc") or 0.0),
            "final_epoch": int(state.get("epoch", self.epochs - 1)),
            "final_step": int(state.get("global_step", self.global_step)),
        }

    @torch.no_grad()
    def _run_validation(self, state: Dict[str, Any]) -> None:
        self.model.train(False)
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in self.val_loader:
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)

            total_loss += float(loss.item()) * targets.size(0)

            if self.metric_fn is not None and logits.dim() >= 2:
                preds = logits.argmax(dim=1)
                total_correct += int((preds == targets).sum().item())
            total_count += int(targets.size(0))

        mean_loss = total_loss / max(1, total_count)
        acc = (total_correct / max(1, total_count)) if self.metric_fn is not None else None

        state["val_loss"] = mean_loss
        state["val_acc"] = acc

        # Inform hooks around validation lifecycle
        self.hooks.on_eval_start(state)
        self.hooks.on_eval_end(state)

        # switch back to train mode
        self.model.train(True)


    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support common dataset return patterns: (inputs, targets) or dicts.
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            # try common keys
            x = batch.get("inputs", None)
            y = batch.get("targets", None)
            if x is None:
                # fall back to first tensor-like item
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        if x is None:
                            x = v
                        else:
                            y = v
                            break
            if x is None or y is None:
                raise ValueError("Could not unpack dict batch into (inputs, targets).")
            return x, y
        raise TypeError("Unsupported batch type; expected (inputs, targets) tuple or a dict.")

