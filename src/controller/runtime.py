from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from features import _safe_float
from src.core.hooks.hook_base import Hook, State
from src.controller.features import FeatureExtractor
from src.controller.lr_scheduler import LrControllerScheduler
from src.core.logging.loggers import ControllerTickLogger
from src.utils import detach_scalar


@dataclass
class RuntimeConfig:
    """
    Lightweight config for runtime behavior.
    """
    update_interval: int = 50     # steps between decisions
    cooldown: int = 25            # steps to wait after an action
    action_ema: float = 0.5       # smoothing on delta log lr (0..1), higher is slower
    # guards
    nan_guard: bool = True        # skip actions producing nans/infs
    max_abs_delta: Optional[float] = None  # optional hard clip on |delta log lr| before EMA


class ControllerRuntime(Hook):
    """
    Hook that:
      builds the feature vector,
      runs the controller to get delta log lr,
      smooths and applies the LR change via LrControllerScheduler,
      logs a schema-stable decision tick.

    integration points:
      - this hook goes after FeatureExtractor in a HookList.
      - make sure training engine populates the usual state keys

    state side-effects (for other hooks/loggers):
      - controller_delta_raw: float
      - controller_delta_applied: float
      - lr_before: float
      - lr_after: float
      - controller_applied: bool
      - controller_cooldown_left: int
    """

    def __init__(
        self,
        controller: nn.Module,
        extractor: FeatureExtractor,
        scheduler: LrControllerScheduler,
        tick_logger: Optional[ControllerTickLogger],
        device: torch.device,
        cfg: Optional[RuntimeConfig] = None,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
        arch: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> None:
        self.controller = controller.eval()   # runtime uses eval mode (no dropout/bn updates)
        self.extractor = extractor
        self.scheduler = scheduler
        self.tick_logger = tick_logger

        # identity/meta for logs
        self.run_id = run_id or ""
        self.seed = int(seed) if seed is not None else None
        self.arch = arch or ""
        self.dataset = dataset or ""

        # runtime config
        self.cfg = cfg or RuntimeConfig()
        if self.cfg.update_interval <= 0:
            raise ValueError("update_interval must be > 0")
        if self.cfg.cooldown < 0:
            raise ValueError("cooldown must be >= 0")
        if not (0.0 <= self.cfg.action_ema <= 1.0):
            raise ValueError("action_ema must be in [0, 1]")

        self.device = device
        self.controller.to(self.device)

        # internal state
        self._cooldown_left = 0
        self._last_action = 0.0          # last applied delta log lr (after EMA)
        self._last_tick_step: Optional[int] = None  # track last decision step

    def on_train_start(self, state: State) -> None:
        self._cooldown_left = 0
        self._last_action = 0.0
        self._last_tick_step = None

    def on_batch_end(self, state: State) -> None:
        gs = int(state.get("global_step", 0))
        step_in_epoch = int(state.get("batch_idx", 0))
        epoch = int(state.get("epoch", 0))

        # default state flags for other hooks
        state["controller_applied"] = False
        state["controller_cooldown_left"] = int(self._cooldown_left)

        # deal with cooldown
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            state["controller_cooldown_left"] = int(self._cooldown_left)
            return

        # only act on configured interval
        if gs % self.cfg.update_interval != 0:
            return

        # build features
        with torch.no_grad():
            feat = self.extractor.get_vector(state).to(self.device)  # shape [F]
            feat = feat.unsqueeze(0)  # [1, F] for batch-1 forward

            # controller forward → delta log lr (raw)
            delta_raw_t = self.controller(feat)  # expect shape [1] or [1, 1]
            if isinstance(delta_raw_t, (tuple, list)):
                delta_raw_t = delta_raw_t[0]
            delta_raw = float(delta_raw_t.squeeze().detach().cpu().item())

        if self.cfg.max_abs_delta is not None:
            delta_raw = float(max(-self.cfg.max_abs_delta, min(self.cfg.max_abs_delta, delta_raw)))

        # nan/inf guard
        if self.cfg.nan_guard and (math.isnan(delta_raw) or math.isinf(delta_raw)):
            self._log_tick(state, applied=False, delta_raw=delta_raw, delta_applied=0.0)
            return

        # EMA smoothing in log-space
        alpha = float(self.cfg.action_ema)
        delta_applied = alpha * self._last_action + (1.0 - alpha) * delta_raw

        # apply via scheduler
        lr_before = self.scheduler.current_lr()
        new_lrs = self.scheduler.apply_delta(delta_applied)  # returns per-group effective LRs (clamped)
        lr_after = self.scheduler.current_lr()

        # update internal and external state
        self._last_action = float(delta_applied)
        self._cooldown_left = int(self.cfg.cooldown)
        self._last_tick_step = gs

        state["controller_delta_raw"] = float(delta_raw)
        state["controller_delta_applied"] = float(delta_applied)
        state["lr_before"] = float(lr_before)
        state["lr_after"] = float(lr_after)
        state["controller_applied"] = True
        state["controller_cooldown_left"] = int(self._cooldown_left)

        # log decision tick
        self._log_tick(
            state,
            applied=True,
            delta_raw=delta_raw,
            delta_applied=delta_applied,
        )

    def on_eval_end(self, state: State) -> None:
        """
        Optionally emit a lightweight 'post' row carrying validation outcomes
        at the same global_step as the last decision, so downstream analysis
        can correlate outcomes with the most recent action.
        """
        if self.tick_logger is None:
            return
        if self._last_tick_step is None:
            return

        gs = int(state.get("global_step", 0))
        # Only attach outcomes if this eval corresponds to the current progress.
        # (It usually does at epoch end; if not, we still write a row that ties to the last tick.)
        row = {
            "run_id": self.run_id,
            "seed": self.seed,
            "arch": self.arch,
            "dataset": self.dataset,
            "epoch": int(state.get("epoch", 0)),
            "global_step": int(self._last_tick_step),
            "step_in_epoch": int(state.get("batch_idx", 0)),
            # outcomes
            "val_acc_post": detach_scalar(state.get("val_acc", 0.0) or 0.0),
            "val_loss_post": detach_scalar(state.get("val_loss", 0.0) or 0.0),
            # mark as non-action row; leave other fields null in parquet
            "applied": False,
        }
        self.tick_logger.log_tick(row)

    def _log_tick(self, state: State, *, applied: bool, delta_raw: float, delta_applied: float) -> None:
        if self.tick_logger is None:
            return

        row: Dict[str, Any] = {
            # identity
            "run_id": self.run_id,
            "seed": self.seed,
            "arch": self.arch,
            "dataset": self.dataset,
            # time
            "epoch": int(state.get("epoch", 0)),
            "global_step": int(state.get("global_step", 0)),
            "step_in_epoch": int(state.get("batch_idx", 0)),
            # inputs (best-effort; OK if some are None—Parquet NULL)
            "train_loss": _safe_float(state.get("loss")),
            "val_loss": _safe_float(state.get("val_loss")),
            "train_acc": _safe_float(state.get("acc")),
            "val_acc": _safe_float(state.get("val_acc")),
            "ema_loss": _safe_float(state.get("ema_loss")),
            "ema_acc": _safe_float(state.get("ema_acc")),
            "d_loss": _safe_float(state.get("d_loss")),
            "d_acc": _safe_float(state.get("d_acc")),
            "lr_before": _safe_float(state.get("lr_before", state.get("lr"))),
            "grad_norm": _safe_float(state.get("grad_norm")),
            "update_ratio": _safe_float(state.get("update_ratio")),
            "clip_events": int(state.get("clip_events", 0) or 0),
            # action
            "delta_log_lr_raw": float(delta_raw) if (delta_raw is not None and not math.isnan(delta_raw) and not math.isinf(delta_raw)) else None,
            "delta_log_lr_applied": float(delta_applied),
            "lr_after": _safe_float(state.get("lr_after", self.scheduler.current_lr())),
            "applied": bool(applied),
            "cooldown_left": int(self._cooldown_left),
            # outcomes will be logged later in on_eval_end (as val_*_post)
            "nan_flag": False if (delta_raw is not None and not math.isnan(delta_raw) and not math.isinf(delta_raw)) else True,
            "overflow_flag": False,  # reserved for future gradient overflow detection
        }

        self.tick_logger.log_tick(row)
