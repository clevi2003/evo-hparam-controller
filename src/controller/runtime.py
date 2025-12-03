from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.controller.features import _safe_float
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
    update_interval: int = 50 # steps between decisions
    cooldown: int = 25 # steps to wait after an action
    warmup_len: int = 0 # initial steps to skip (for feature warmup)
    action_ema: float = 0.5 # smoothing on delta log lr (0..1), higher is slower
    # guards
    nan_guard: bool = True # skip actions producing nans/infs
    max_abs_delta: Optional[float] = None # optional hard clip on |delta log lr| before EMA
    epoch_offset: int = 0
    log_features_json: bool = True


class ControllerRuntime(Hook):
    """
    Hook that:
      builds the feature vector,
      runs the controller to get delta log lr,
      smooths and applies the LR change via LrControllerScheduler,
      logs a schema-stable decision tick

    integration points:
      this hook goes after FeatureExtractor in a HookList

    state side-effects (for other hooks/loggers):
      controller_delta_raw: float
      controller_delta_applied: float
      lr_before: float
      lr_after: float
      controller_applied: bool
      controller_cooldown_left: int
      gen_gap: float or None
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
        self.controller = controller.eval() # runtime uses eval mode (no dropout/bn updates)
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
        self._last_action = 0.0 # last applied delta log lr (after EMA)
        self._last_tick_step: Optional[int] = None # track last decision step
        self._cooldown_skips_cum = 0
        self._last_decision_step: Optional[int] = None

    def on_train_start(self, state: State) -> None:
        self._cooldown_left = 0
        self._last_action = 0.0
        self._last_tick_step = None
        self._cooldown_skips_cum = 0
        self._last_decision_step = None

    def on_batch_end(self, state: State) -> None:
        gs = int(state.get("global_step", 0))

        # apply epoch offset so both features and logs see effective epoch
        raw_epoch = int(state.get("epoch", 0))
        if self.cfg.epoch_offset:
            state["epoch"] = raw_epoch + self.cfg.epoch_offset

        # default state flags for other hooks
        state["controller_applied"] = False
        state["controller_cooldown_left"] = int(self._cooldown_left)

        if self.cfg.warmup_len > 0 and gs < self.cfg.warmup_len:
            return

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
            feat = self.extractor.get_vector(state).to(self.device) # shape [F]
            feat = feat.unsqueeze(0) # [1, F] for batch-1 forward

            # controller forward delta log lr (raw)
            # print(f"ControllerRuntime: computing delta log lr at step {gs} with features {feat.cpu().numpy()}")
            delta_raw_t = self.controller(feat) # expect shape [1] or [1, 1]
            if isinstance(delta_raw_t, (tuple, list)):
                delta_raw_t = delta_raw_t[0]
            delta_raw = float(delta_raw_t.squeeze().detach().cpu().item())

        delta_clamped = delta_raw
        clamp_flag = False
        if (self.cfg.max_abs_delta is not None) and math.isfinite(delta_raw):
            bound = float(self.cfg.max_abs_delta)
            if abs(delta_raw) > bound:
                delta_clamped = float(max(-bound, min(bound, delta_raw)))
                clamp_flag = True

        # nan/inf guard
        if self.cfg.nan_guard and (not math.isfinite(delta_raw)):
            self._log_tick(
                state,
                applied=False,
                delta_raw=delta_raw,
                delta_applied=0.0,
                delta_target=delta_raw,
                delta_clamped=None,
                safety_event="nan_guard",
                safety_detail="controller produced NaN/Inf",
            )
            return

        # EMA smoothing in log-space
        alpha = float(self.cfg.action_ema)
        delta_applied = alpha * self._last_action + (1.0 - alpha) * delta_clamped
        lr_multiplier = math.exp(delta_applied) if math.isfinite(delta_applied) else None

        mom_before = _get_pg_avg(self.scheduler.optimizer, "momentum")
        wd_before = _get_pg_avg(self.scheduler.optimizer, "weight_decay")

        # apply via scheduler
        lr_before = self.scheduler.current_lr()
        new_lrs = self.scheduler.apply_delta(delta_applied) # returns per-group effective LRs (clamped)
        lr_after = self.scheduler.current_lr()

        mom_after = _get_pg_avg(self.scheduler.optimizer, "momentum")
        wd_after = _get_pg_avg(self.scheduler.optimizer, "weight_decay")
        mom_mult = (mom_after / mom_before) if (mom_before and mom_after) else None
        wd_mult = (wd_after / wd_before) if (wd_before and wd_after) else None

        # update internal and external state
        steps_since_last_call = 0 if (self._last_decision_step is None) else (gs - self._last_decision_step)
        self._last_decision_step = gs
        self._last_action = float(delta_applied)
        self._cooldown_left = int(self.cfg.cooldown)
        self._last_tick_step = gs

        state["controller_delta_raw"] = float(delta_raw)
        state["controller_delta_applied"] = float(delta_applied)
        state["lr_before"] = float(lr_before)
        state["lr_after"] = float(lr_after)
        state["controller_applied"] = True
        state["controller_cooldown_left"] = int(self._cooldown_left)

        train_acc = state.get("acc", None)
        val_acc = state.get("val_acc", None)
        gen_gap = None
        if (train_acc is not None) and (val_acc is not None):
            try:
                gen_gap = float(val_acc) - float(train_acc)
            except Exception:
                gen_gap = None
        state["gen_gap"] = gen_gap

        features_json = None
        if self.cfg.log_features_json:
            try:
                features_json = self.extractor.features_json(state)
            except Exception:
                # annoying json issues should not block training, just throw exception
                features_json = None

        safety_event = "delta_clamped" if clamp_flag else None
        safety_detail = None
        if clamp_flag and self.cfg.max_abs_delta is not None:
            safety_detail = f"abs(delta_target)>{self.cfg.max_abs_delta:g}"

        # log decision tick
        self._log_tick(
            state,
            applied=True,
            delta_raw=delta_raw,
            delta_applied=delta_applied,
            delta_target=delta_raw,
            delta_clamped=delta_clamped,
            lr_multiplier=lr_multiplier,
            steps_since_last_call=steps_since_last_call,
            momentum_before=mom_before,
            momentum_after=mom_after,
            momentum_multiplier=mom_mult,
            wd_before=wd_before,
            wd_after=wd_after,
            wd_multiplier=wd_mult,
            safety_event=safety_event,
            safety_detail=safety_detail,
            features_json=features_json,
        )

    def on_eval_end(self, state: State) -> None:
        """
        Optionally emit a lightweight post row carrying validation outcomes
        at the same global_step as the last decision, so downstream analysis
        can correlate outcomes with the most recent action.
        """
        if self.tick_logger is None:
            return
        if self._last_tick_step is None:
            return

        # gs = int(state.get("global_step", 0))
        # Only attach outcomes if this eval corresponds to the current progress
        # (It usually does at epoch end, if not still write a row that ties to the last tick)
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
            "applied": False,
        }
        self.tick_logger.log_tick(row)

    def _log_tick(
        self,
        state: State,
        *,
        applied: bool,
        delta_raw: float,
        delta_applied: float,
        delta_target: Optional[float] = None,
        delta_clamped: Optional[float] = None,
        lr_multiplier: Optional[float] = None,
        steps_since_last_call: Optional[int] = None,
        momentum_before: Optional[float] = None,
        momentum_after: Optional[float] = None,
        momentum_multiplier: Optional[float] = None,
        wd_before: Optional[float] = None,
        wd_after: Optional[float] = None,
        wd_multiplier: Optional[float] = None,
        safety_event: Optional[str] = None,
        safety_detail: Optional[str] = None,
        features_json: Optional[str] = None,
    ) -> None:
        if self.tick_logger is None:
            return

        entropy = state.get("entropy", None)
        margin = state.get("margin", None)

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
            # inputs (best-effort)
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
            # action (three stages + result)
            "delta_log_lr_raw": _safe_float(delta_raw),
            "delta_log_lr_applied": _safe_float(delta_applied),
            "delta_log_lr_target": _safe_float(delta_target),
            "delta_log_lr_clamped": _safe_float(delta_clamped),
            "lr_after": _safe_float(state.get("lr_after", self.scheduler.current_lr())),
            "lr_multiplier": _safe_float(lr_multiplier),
            "applied": bool(applied),
            "cooldown_left": int(self._cooldown_left),
            # cadence/meta
            "call_interval": int(self.cfg.update_interval),
            "warmup_len": int(self.cfg.warmup_len),
            "action_ema_alpha": float(self.cfg.action_ema),
            "steps_since_last_call": int(steps_since_last_call or 0),
            "cooldown_skips_cum": int(self._cooldown_skips_cum),
            # derived
            "gen_gap": _safe_float(state.get("gen_gap")),
            "ema_grad_norm": _safe_float(state.get("ema_grad_norm")),
            "entropy": _safe_float(entropy),
            "margin": _safe_float(margin),
            # optimizer placeholders
            "momentum_before": _safe_float(momentum_before),
            "momentum_after": _safe_float(momentum_after),
            "momentum_multiplier": _safe_float(momentum_multiplier),
            "wd_before": _safe_float(wd_before),
            "wd_after": _safe_float(wd_after),
            "wd_multiplier": _safe_float(wd_multiplier),
            # safety
            "nan_flag": False if (delta_raw is not None and math.isfinite(float(delta_raw))) else True,
            "overflow_flag": False,
            "safety_event": safety_event,
            "safety_detail": safety_detail,
            # snapshot
            "features_json": features_json if self.cfg.log_features_json else None,
        }

        self.tick_logger.log_tick(row)


def _get_pg_avg(optimizer: torch.optim.Optimizer, key: str) -> Optional[float]:
    """
    Return the average value of a parameter-group hyperparameter (could be momentum, weight_decay, etc)
    If the key is not present in any group, returns None
    """
    vals = []
    for g in optimizer.param_groups:
        if key in g and g[key] is not None:
            try:
                vals.append(float(g[key]))
            except Exception:
                pass
    if not vals:
        return None
    return float(sum(vals) / len(vals))
