from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple, Literal
import sys
import torch
from torch.utils.data import DataLoader
from src.core.hooks.hook_base import Hook, NullHook
import time
# from torch.profiler import profiler, ProfilerActivity

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]
StepWhen = Literal["batch", "epoch", "disabled"]

# ---- tqdm (safe) -------------------------------------------------------------
try:
    from tqdm.auto import tqdm as _tqdm_auto
except Exception:  # pragma: no cover
    _tqdm_auto = None  # fallback to no progress if not available


def _tqdm(iterable=None, *, total=None, desc=None, leave=False, disable=False):
    """
    Safe tqdm wrapper. Falls back to a plain iterable if tqdm is unavailable.
    """
    if disable or _tqdm_auto is None:
        # mimic tqdm's minimal API (context manager and iterator)
        class _NoTQDM:
            def __init__(self, it, total=None, desc=None, leave=False, disable=False):
                self.it = it
            def __iter__(self): return iter(self.it)
            def update(self, *a, **k): pass
            def set_postfix(self, *a, **k): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): pass
        return _NoTQDM(iterable if iterable is not None else range(total or 0))
    return _tqdm_auto(iterable, total=total, desc=desc, leave=leave, disable=disable)


# ---- helpers ----------------------------------------------------------------
def _default_accuracy_fn(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for classification; returns 0.0 if logits aren't 2D."""
    if logits.dim() < 2:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct / max(1, total))


def _detach_scalar(x: torch.Tensor | float | int | None) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        try:
            return float(x.detach().item())
        except Exception:
            return float(x.detach().mean().item())
    return float(x)


def _compute_grad_norm(model: torch.nn.Module) -> Optional[float]:
    total = 0.0
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            param_norm = p.grad.data.norm(2)
            total += float(param_norm.item() ** 2)
    if not has_grad:
        return None
    return float(total ** 0.5)


def _get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    # assume single param group for reporting; fine for CIFAR baselines
    return float(optimizer.param_groups[0].get("lr", 0.0))


class Trainer:
    """
    hook driven training engine with tqdm progress bars

    trainer does not manage RunContext/IO. It just uses hook events so the
    driver and hooks can handle logging and checkpoints

    lifecycle hooks called with state:
      on_train_start, on_train_end
      on_epoch_start, on_epoch_end
      on_batch_start, on_batch_end
      on_eval_start, on_eval_end

    state keys:
      epoch, global_step, batch_idx, optimizer, scheduler, device, cfg
      logits (detached), targets (detached)
      loss, acc, val_loss, val_acc, lr, grad_norm
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: DataLoader,
        device: torch.device,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_step_when: StepWhen = "disabled",
        hooks: Optional[Hook] = None,
        max_steps: Optional[int] = None,
        mixed_precision: bool = False,
        grad_clip: Optional[float] = None,
        metric_fn: Optional[MetricFn] = _default_accuracy_fn,
        cfg: Optional[Any] = None,
        show_progress: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_when = scheduler_step_when
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = int(epochs)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.grad_clip = grad_clip
        self.metric_fn = metric_fn
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self.hooks = hooks or NullHook()
        self.cfg = cfg
        self.global_step = 0
        self.best_val_acc = float("-inf")
        # tqdm control
        # disable if no tty (non-interactive) or user asked to hide
        self._progress_enabled = bool(show_progress) and sys.stderr.isatty()

    def fit(self, *, epochs: Optional[int] = None, max_steps: Optional[int] = None) -> Dict[str, float]:
        """
        run training. optionally override epochs/max_steps at call time
        returns a summary dict: {best_val_acc, final_val_acc, final_train_acc, final_epoch, final_step}
        """
        total_epochs = int(epochs) if epochs is not None else self.epochs
        budget_steps = int(max_steps) if max_steps is not None else self.max_steps

        state: Dict[str, Any] = {
            "epoch": 0,
            "global_step": 0,
            "batch_idx": -1,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "device": self.device,
            "cfg": self.cfg,
            "model": self.model,
            # live scalars
            "loss": None,
            "acc": None,
            "val_loss": None,
            "val_acc": None,
            "lr": _get_current_lr(self.optimizer),
            "grad_norm": None,
        }

        self.hooks.on_train_start(state)

        t_train_start = time.perf_counter()
        epoch_data = []

        # epoch progress bar
        with _tqdm(
            total=total_epochs,
            desc="Epochs",
            leave=True,
            disable=not self._progress_enabled,
        ) as pbar_epochs:

            for epoch in range(total_epochs):
                state["epoch"] = epoch
                self.hooks.on_epoch_start(state)

                # sync for cuda timings
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t_epoch_start = time.perf_counter()
                epoch_samples = 0

                # training epoch
                self.model.train(True)

                # batch progress bar
                total_batches = None
                try:
                    total_batches = len(self.train_loader)
                except Exception:
                    total_batches = None

                with _tqdm(
                    total=total_batches,
                    desc=f"Train [{epoch+1}/{total_epochs}]",
                    leave=False,
                    disable=not self._progress_enabled,
                ) as pbar_train:

                    for batch_idx, batch in enumerate(self.train_loader):
                        state["batch_idx"] = batch_idx

                        # budgeted training: exit early after finishing this batch
                        if budget_steps is not None and self.global_step >= budget_steps:
                            break

                        inputs, targets = self._unpack_batch(batch)
                        inputs = inputs.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)

                        # keeping a running total of samples for the throughput
                        epoch_samples += int(targets.size(0))

                        self.hooks.on_batch_start(state)

                        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                            logits = self.model(inputs)
                            loss = self.loss_fn(logits, targets)

                        # backward
                        self.optimizer.zero_grad(set_to_none=True)

                        if self.mixed_precision:
                            self.scaler.scale(loss).backward()
                            self.scaler.unscale_(self.optimizer)
                            self.hooks.on_after_backward(state)
                            self.hooks.on_before_optimizer_step(state)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.hooks.on_after_optimizer_step(state)

                        else:
                            loss.backward()
                            self.hooks.on_after_backward(state)
                            self.hooks.on_before_optimizer_step(state)
                            self.optimizer.step()
                            self.hooks.on_after_optimizer_step(state)

                        # update counters and collect metrics
                        self.global_step += 1
                        state["global_step"] = self.global_step

                        acc = None
                        if self.metric_fn is not None:
                            try:
                                acc = self.metric_fn(logits.detach(), targets.detach())
                            except Exception:
                                acc = None

                        state["logits"] = logits.detach()
                        state["targets"] = targets.detach()
                        state["loss"] = _detach_scalar(loss)
                        state["acc"] = acc
                        state["lr"] = _get_current_lr(self.optimizer)
                        state["grad_norm"] = _compute_grad_norm(self.model)

                        # progress bar feedback
                        if self._progress_enabled:
                            pbar_train.set_postfix(
                                loss=f"{state['loss']:.4f}" if state["loss"] is not None else "—",
                                acc=f"{(state['acc']*100):.2f}%" if state["acc"] is not None else "—",
                                lr=f"{state['lr']:.3e}",
                            )
                            pbar_train.update(1)

                        self.hooks.on_batch_end(state)

                        # cleanup heavy tensors in state
                        state.pop("logits", None)
                        state.pop("targets", None)

                        # budget check after finishing the batch
                        if budget_steps is not None and self.global_step >= budget_steps:
                            break
                
                # sync for accurate cuda timings
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t_epoch_end = time.perf_counter()
                t_epoch = t_epoch_end - t_epoch_start

                data = (epoch_samples, t_epoch)
                epoch_data.append(data)

                # validation once per epoch
                if self.val_loader is not None:
                    self._run_validation(state, total_epochs, epoch)

                # scheduler step per-epoch if requested
                if self.scheduler is not None and self.scheduler_step_when == "epoch":
                    self.scheduler.step()

                self.hooks.on_epoch_end(state)

                # track best val acc
                if state["val_acc"] is not None:
                    self.best_val_acc = max(self.best_val_acc, float(state["val_acc"]))

                # update epoch bar
                if self._progress_enabled:
                    pbar_epochs.set_postfix(
                        val_loss=f"{state['val_loss']:.4f}" if state["val_loss"] is not None else "—",
                        val_acc=f"{(state['val_acc']*100):.2f}%" if state["val_acc"] is not None else "—",
                    )
                    pbar_epochs.update(1)

                # early exit if budget maxed out
                if budget_steps is not None and self.global_step >= budget_steps:
                    break
    
        # train end
        t_train_end = time.perf_counter()

        t_train       = t_train_end - t_train_start
        total_samples = sum(s for s, t in data)
        total_t_epoch = sum(t for s, t in data)

        state['t_epoch']       = total_t_epoch / total_epochs
        state['samples_per_s'] = total_samples / max(total_time, 1e-9)
        state['t_train']       = t_train
        
        self.hooks.on_train_end(state)

        return {
            "best_val_acc": float(self.best_val_acc if self.best_val_acc != float("-inf") else (state.get("val_acc") or 0.0)),
            "final_val_acc": float(state.get("val_acc") or 0.0),
            "final_train_acc": float(state.get("acc") or 0.0),
            "final_epoch": int(state.get("epoch", total_epochs - 1)),
            "final_step": int(state.get("global_step", self.global_step)),
        }

    @torch.no_grad()
    def _run_validation(self, state: Dict[str, Any], total_epochs: int, epoch_idx: int) -> None:
        self.hooks.on_eval_start(state)
        self.model.train(False)
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        # val progress
        v_total = None
        try:
            v_total = len(self.val_loader)
        except Exception:
            v_total = None

        with _tqdm(
            total=v_total,
            desc=f" Val   [{epoch_idx+1}/{total_epochs}]",
            leave=False,
            disable=not self._progress_enabled,
        ) as pbar_val:

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

                if self._progress_enabled:
                    # lightweight running estimates
                    denom = max(1, total_count)
                    est_loss = total_loss / denom
                    est_acc = (total_correct / denom) if self.metric_fn is not None else None
                    pbar_val.set_postfix(
                        loss=f"{est_loss:.4f}",
                        acc=f"{(est_acc*100):.2f}%" if est_acc is not None else "—",
                    )
                    pbar_val.update(1)

        mean_loss = total_loss / max(1, total_count)
        acc = (total_correct / max(1, total_count)) if self.metric_fn is not None else None

        state["val_loss"] = mean_loss
        state["val_acc"] = acc

        self.hooks.on_eval_end(state)
        self.model.train(True)

    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Support (inputs, targets) tuples or dicts with common keys."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            x = batch.get("inputs", None)
            y = batch.get("targets", None)
            if x is None:
                # fall back to first two tensor-like entries
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        if x is None:
                            x = v
                        elif y is None:
                            y = v
                        if x is not None and y is not None:
                            break
            if x is None or y is None:
                raise ValueError("Could not unpack dict batch into (inputs, targets).")
            return x, y
        raise TypeError("Unsupported batch type; expected (inputs, targets) or dict.")
