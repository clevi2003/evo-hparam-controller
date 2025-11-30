from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
Number = Union[int, float]


@dataclass
class LrBounds:
    lr_min: float = 1e-10
    lr_max: float = 1.0

    def validate(self) -> None:
        if self.lr_min <= 0 or self.lr_max <= 0:
            raise ValueError("lr_min and lr_max must be > 0.")
        if self.lr_min >= self.lr_max:
            raise ValueError("lr_min must be < lr_max.")

    def as_tuple(self) -> Tuple[float, float]:
        return self.lr_min, self.lr_max


class LrControllerScheduler:
    """
    Wraps an optimizer (and optional base scheduler) to compose a multiplicative
    controller scale on top of a base learning rate.

    Important:
    The underlying (base) scheduler always steps on unscaled base LRs.
    The controller holds per-param-group multiplicative scales (starts at 1.0).
    The optimizer's public LR is always the effective LR: base * scale, clamped.
    LR clamps are enforced to [lr_min, lr_max] per param group.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr_min: float = 1e-5,
        lr_max: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.bounds = LrBounds(lr_min=lr_min, lr_max=lr_max)
        self.bounds.validate()

        # Per-group state
        self._n_groups = len(self.optimizer.param_groups)
        self._base_lrs: List[float] = [float(g.get("lr", 0.0)) for g in self.optimizer.param_groups]
        self._scales: List[float] = [1.0 for _ in range(self._n_groups)]
        self._effective_lrs: List[float] = self._compute_effective_lrs()

        # Apply immediately so optimizer.param_groups reflect effective values
        self._write_effective_lrs()

        # Bookkeeping for "before/after" reporting (averaged across groups)
        self._last_before: float = self._avg(self._effective_lrs)
        self._last_after: float = self._last_before

    def step_base(self) -> None:
        """
        Advance the base scheduler by one step (if present) in a way that the
        controller scale does NOT leak into the base schedule:
        """
        # Temporarily set optimizer LRs to base LRs (remove controller scale)
        self._write_base_lrs()

        # Advance base scheduler (if any)
        if self.base_scheduler is not None:
            self.base_scheduler.step()

        # Resync base from optimizer's current LRs
        self._base_lrs = [float(g.get("lr", 0.0)) for g in self.optimizer.param_groups]

        # Reapply controller scales and clamp
        self._effective_lrs = self._compute_effective_lrs()
        self._write_effective_lrs()

    def apply_delta(self, delta_log_lr: Union[Number, Sequence[Number]]) -> List[float]:
        """
        Update controller scales multiplicatively in log-space:
            scale[i] <- scale[i] * exp(delta_log_lr[i])

        Accepts either a scalar (applied to all param groups) or a sequence per group.
        Returns the new effective LRs (per group) after clamping & application.
        """
        deltas = self._expand_to_groups(delta_log_lr)
        # Report average before for logging convenience
        self._last_before = self.current_lr()

        for idx, delta in enumerate(deltas):
            # update scale
            self._scales[idx] *= math.exp(float(delta))

            # clamp scale so that base * scale stays within [lr_min, lr_max]
            base = self._base_lrs[idx]
            # If base is zero, effective will be zero. leave scale as is.
            if base > 0.0:
                min_scale = self.bounds.lr_min / base
                max_scale = self.bounds.lr_max / base
                self._scales[idx] = float(max(min_scale, min(max_scale, self._scales[idx])))

        # recompute & write effective LRs
        self._effective_lrs = self._compute_effective_lrs()
        self._write_effective_lrs()

        self._last_after = self.current_lr()
        return list(self._effective_lrs)

    def set_scales(self, scales: Sequence[Number]) -> List[float]:
        """
        Overwrite controller scales directly.
        Scales are clamped so base*scale stays within bounds.
        """
        if len(scales) != self._n_groups:
            raise ValueError(f"set_scales expects {self._n_groups} values, got {len(scales)}.")
        for i, s in enumerate(scales):
            s = float(s)
            if self._base_lrs[i] > 0.0:
                min_scale = self.bounds.lr_min / self._base_lrs[i]
                max_scale = self.bounds.lr_max / self._base_lrs[i]
                s = max(min_scale, min(max_scale, s))
            self._scales[i] = s
        self._effective_lrs = self._compute_effective_lrs()
        self._write_effective_lrs()
        return list(self._effective_lrs)

    def reset_scales(self, value: float = 1.0) -> None:
        """Reset all controller scales to value (default 1) and re-apply"""
        self._scales = [float(value) for _ in range(self._n_groups)]
        self._effective_lrs = self._compute_effective_lrs()
        self._write_effective_lrs()

    def current_lr(self) -> float:
        """Average effective LR across param groups for quick logging"""
        return self._avg(self._effective_lrs)
        
    #Add the current base learning rate method (Kevin)
    def current_base_lr(self) -> float:
        """Average base LR across param groups (before controller scaling)."""
        return self._avg(self._base_lrs)

    def lrs(self) -> List[float]:
        """Effective LRs per param group"""
        return list(self._effective_lrs)

    def lr_before(self) -> float:
        """Average LR before the last apply_delta() call"""
        return float(self._last_before)

    def lr_after(self) -> float:
        """Average LR after the last apply_delta() call"""
        return float(self._last_after)

    def lr_bounds_for_group(self, group_index: int) -> Tuple[float, float]:
        """
        Bounds on effective LR for a given group, given current base LR.
        """
        base = self._base_lrs[group_index]
        if base <= 0.0:
            # If base is 0, effective is 0 regardless; still return global bounds.
            return self.bounds.as_tuple()
        return max(self.bounds.lr_min, 0.0), self.bounds.lr_max

    def resync_base_from_optimizer(self) -> None:
        """
        If external code changed optimizer.param_groups[i]['lr'] to redefine the base,
        call this to capture the new base LRs and reapply controller scales.
        This is not commonly needed but can be useful for custom LR schedules or if we've
        got wonky code that directly messes with optimizer LRs.
        """
        self._base_lrs = [float(g.get("lr", 0.0)) for g in self.optimizer.param_groups]
        self._effective_lrs = self._compute_effective_lrs()
        self._write_effective_lrs()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "bounds": asdict(self.bounds),
            "base_lrs": list(self._base_lrs),
            "scales": list(self._scales),
            "effective_lrs": list(self._effective_lrs),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        bounds = state.get("bounds", None)
        if bounds is not None:
            self.bounds = LrBounds(**bounds)
            self.bounds.validate()
        base_lrs = state.get("base_lrs", None)
        scales = state.get("scales", None)
        if base_lrs is not None:
            if len(base_lrs) != self._n_groups:
                raise ValueError("Mismatched base_lrs length.")
            self._base_lrs = [float(x) for x in base_lrs]
        if scales is not None:
            if len(scales) != self._n_groups:
                raise ValueError("Mismatched scales length.")
            self._scales = [float(x) for x in scales]
        self._effective_lrs = self._compute_effective_lrs()
        self._write_effective_lrs()

    def _compute_effective_lrs(self) -> List[float]:
        eff: List[float] = []
        for base, scale in zip(self._base_lrs, self._scales):
            if base <= 0.0:
                eff.append(0.0)
                continue
            lr = base * scale
            # clamp effective LR
            lr = max(self.bounds.lr_min, min(self.bounds.lr_max, lr))
            eff.append(float(lr))
        return eff

    def _write_effective_lrs(self) -> None:
        for group, lr in zip(self.optimizer.param_groups, self._effective_lrs):
            group["lr"] = float(lr)

    def _write_base_lrs(self) -> None:
        for group, base in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = float(base)

    @staticmethod
    def _avg(xs: Sequence[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    def _expand_to_groups(self, x: Union[Number, Sequence[Number]]) -> List[float]:
        if isinstance(x, (int, float)):
            return [float(x) for _ in range(self._n_groups)]
        seq = list(x)
        if len(seq) != self._n_groups:
            raise ValueError(f"Expected {self._n_groups} deltas, got {len(seq)}.")
        return [float(v) for v in seq]

### bahhhh this was way more work than I thought. If you mess with the param groups,
# increment this wasted time counter:
# Wasted Time: 3 hours

