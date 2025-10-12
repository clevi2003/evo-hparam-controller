from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from src.core.hooks.hook_base import Hook, NullHook
from src.utils import compute_grad_norm, detach_scalar, get_current_lr

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
        }

        best_val_acc = float("-inf")

        self.hooks.on_train_start(state)

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
                    state["logits"] = logits.detach()
                    state["targets"] = targets.detach()
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

                self.hooks.on_batch_end(state)
                state.pop("logits", None)
                state.pop("targets", None)

            # Validation at epoch end
            if self.val_loader is not None:
                self._run_validation(state)
                if state["val_acc"] is not None:
                    best_val_acc = max(best_val_acc, float(state["val_acc"]))

        # Train end
        self.hooks.on_train_end(state)

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
            state["logits"] = logits.detach()
            state["targets"] = targets.detach()
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
        state.pop("logits", None)
        state.pop("targets", None)

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

