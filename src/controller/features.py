from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import math
import torch
from ..core.hooks.hook_base import Hook, State


class _EmaStat:
    """
    Exponential moving averages for mean and approx. variance.
    Keep EMA of x and EMA of x^2, std_ema = sqrt(max(eps, ema_x2 - ema_x^2))
    """
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.mean = 0.0
        self.mean_sq = 0.0
        self.initialized = False

    def update(self, x: float) -> None:
        if not self.initialized:
            self.mean = float(x)
            self.mean_sq = float(x * x)
            self.initialized = True
            return
        a = self.alpha
        self.mean = (1.0 - a) * self.mean + a * float(x)
        self.mean_sq = (1.0 - a) * self.mean_sq + a * float(x * x)

    @property
    def std(self) -> float:
        var = max(0.0, self.mean_sq - self.mean * self.mean)
        return math.sqrt(var)

    def zscore(self, x: float, eps: float = 1e-6) -> float:
        s = self.std
        if s < eps:
            return 0.0
        return (float(x) - self.mean) / (s + eps)

_VALID_FEATURES = {
    # time/state
    "epoch",
    "step_in_epoch",
    "global_step",
    # train/val metrics
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    # LR & optimizer signals
    "lr_current",
    "grad_norm",
    "update_ratio",  # ||deltaθ|| / ||θ||
    "clip_events",  # like number of clips this step
    # EMAs/deltas computed here
    "ema_loss",
    "ema_acc",
    "d_loss",
    "d_acc",
}

_NUMERIC_DEFAULTS = {
    "epoch": 0.0,
    "step_in_epoch": 0.0,
    "global_step": 0.0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "lr_current": 0.0,
    "grad_norm": 0.0,
    "update_ratio": 0.0,
    "clip_events": 0.0,
    "ema_loss": 0.0,
    "ema_acc": 0.0,
    "d_loss": 0.0,
    "d_acc": 0.0,
}


@dataclass
class FeatureExtractorConfig:
    spec: List[str]
    ema_alpha: float = 0.1
    standardize: bool = True
    # Could include transformations per feature (applied to raw value before standardization) idk yet
    use_log1p_for_steps: bool = True  # log1p for large counters to stabilize scale


class FeatureExtractor(Hook):
    """
    Hook that maintains short-term statistics (EMA mean/std, deltas) and
    produces a fixed-order feature vector for the controller.

    Expected state keys (populated by the training engine / other hooks):
      epoch:int, global_step:int, batch_idx:int, loss:float, acc:float,
      val_loss:float, val_acc:float, lr:float, grad_norm:float,
      update_ratio:float, clip_events:int

    Public:
      get_vector(state) -> torch.Tensor[feature_dim]
      feature_names -> List[str]  (exact order matching the tensor)

    Important:
      All outputs are float32
      Missing values default to 0
      ema_loss/ema_acc are computed here from train metrics
      d_loss/d_acc are stepwise differences of train metrics
    """

    def __init__(
            self,
            spec: Sequence[str],
            ema_alpha: float = 0.1,
            standardize: bool = True,
            device: Optional[torch.device] = None,
            use_log1p_for_steps: bool = True,
    ) -> None:
        unknown = [f for f in spec if f not in _VALID_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features in spec: {unknown}. Valid: {sorted(_VALID_FEATURES)}")

        self._spec = list(spec)
        self._device = device or torch.device("cpu")
        self._standardize = bool(standardize)
        self._ema_alpha = float(ema_alpha)
        self._use_log1p_for_steps = bool(use_log1p_for_steps)

        # EMA trackers for select features (where standardization isn't dumb)
        self._ema_stats: Dict[str, _EmaStat] = {}
        for name in ("train_loss", "train_acc", "lr_current", "grad_norm", "update_ratio"):
            self._ema_stats[name] = _EmaStat(alpha=self._ema_alpha)

        # Derived EMAs get maintained explicitly
        self._ema_stats["ema_loss"] = _EmaStat(alpha=self._ema_alpha)
        self._ema_stats["ema_acc"] = _EmaStat(alpha=self._ema_alpha)

        # Last values to compute deltas
        self._last_vals: Dict[str, float] = {"train_loss": 0.0, "train_acc": 0.0}

        # Cached last computed feature vector (good for logging tick before/after)
        self._last_vector: Optional[torch.Tensor] = None

    def on_batch_end(self, state: State) -> None:
        # Read current signals from state
        train_loss = _safe_float(state.get("loss", None))
        train_acc = _safe_float(state.get("acc", None))
        lr = _safe_float(state.get("lr", None))
        grad_norm = _safe_float(state.get("grad_norm", None))
        upd_ratio = _safe_float(state.get("update_ratio", None))

        # Update EMAs (if value present)
        if train_loss is not None:
            self._ema_stats["train_loss"].update(train_loss)
            self._ema_stats["ema_loss"].update(train_loss)  
        if train_acc is not None:
            self._ema_stats["train_acc"].update(train_acc)
            self._ema_stats["ema_acc"].update(train_acc)
        if lr is not None:
            self._ema_stats["lr_current"].update(lr)
        if grad_norm is not None:
            self._ema_stats["grad_norm"].update(grad_norm)
        if upd_ratio is not None:
            self._ema_stats["update_ratio"].update(upd_ratio)

        # Deltas for train metrics
        if train_loss is not None:
            prev = self._last_vals.get("train_loss", train_loss)
            self._last_vals["train_loss"] = train_loss
            state["d_loss"] = train_loss - prev
        else:
            state["d_loss"] = 0.0

        if train_acc is not None:
            prev = self._last_vals.get("train_acc", train_acc)
            self._last_vals["train_acc"] = train_acc
            state["d_acc"] = train_acc - prev
        else:
            state["d_acc"] = 0.0

        # Also mirror EMAs into state
        state["ema_loss"] = self._ema_stats["ema_loss"].mean if self._ema_stats["ema_loss"].initialized else 0.0
        state["ema_acc"] = self._ema_stats["ema_acc"].mean if self._ema_stats["ema_acc"].initialized else 0.0

        # Build/cache the latest vector (so controller can call immediately)
        self._last_vector = self.get_vector(state)

    def on_eval_end(self, state: State) -> None:
        # Could update EMA trackers with val metrics if needed later.
        # TODO idk, could be expanded later
        self._last_vector = self.get_vector(state)

    @property
    def feature_names(self) -> List[str]:
        return list(self._spec)

    def get_vector(self, state: State) -> torch.Tensor:
        """
        Produce a 1D float32 tensor of shape [len(spec)] on the configured device.
        Order matches self._spec.
        """
        vals: List[float] = []
        for name in self._spec:
            v = self._read_feature_value(name, state)
            v = self._pre_transform(name, v)
            if self._standardize and self._standardizable(name):
                v = self._standardize_value(name, v)
            vals.append(float(v))
        return torch.tensor(vals, dtype=torch.float32, device=self._device)


    @staticmethod
    def _read_feature_value(name: str, state: State) -> float:
        # Pull raw values from state or our internal EMAs/deltas
        if name == "epoch":
            return float(state.get("epoch", 0))
        if name == "step_in_epoch":
            return float(state.get("batch_idx", 0))
        if name == "global_step":
            return float(state.get("global_step", 0))

        if name == "train_loss":
            return _fallback_zero(state.get("loss"))
        if name == "val_loss":
            return _fallback_zero(state.get("val_loss"))
        if name == "train_acc":
            return _fallback_zero(state.get("acc"))
        if name == "val_acc":
            return _fallback_zero(state.get("val_acc"))

        if name == "lr_current":
            return _fallback_zero(state.get("lr"))
        if name == "grad_norm":
            return _fallback_zero(state.get("grad_norm"))
        if name == "update_ratio":
            return _fallback_zero(state.get("update_ratio"))
        if name == "clip_events":
            return float(state.get("clip_events", 0) or 0)

        if name == "ema_loss":
            # Already mirrored into state in on_batch_end; also available from tracker
            return float(state.get("ema_loss", 0.0))
        if name == "ema_acc":
            return float(state.get("ema_acc", 0.0))
        if name == "d_loss":
            return float(state.get("d_loss", 0.0))
        if name == "d_acc":
            return float(state.get("d_acc", 0.0))

        # Should never hit due to validation in __init__ but good to have all control paths lead to return
        return 0.0

    def _pre_transform(self, name: str, v: float) -> float:
        # Apply stabilizing transforms before standardization
        if self._use_log1p_for_steps and name in ("epoch", "step_in_epoch", "global_step"):
            return math.log1p(max(0.0, v))
        return v

    def _standardizable(self, name: str) -> bool:
        # Only standardize numeric signals where EMA is useful
        return name in self._ema_stats or name in ("train_loss", "train_acc", "lr_current", "grad_norm", "update_ratio")

    def _standardize_value(self, name: str, v: float) -> float:
        stat = self._ema_stats.get(name)
        if stat is None:
            # Some features share EMA trackers
            if name in ("train_loss", "ema_loss", "d_loss"):
                stat = self._ema_stats.get("train_loss")
            elif name in ("train_acc", "ema_acc", "d_acc"):
                stat = self._ema_stats.get("train_acc")
            elif name == "lr_current":
                stat = self._ema_stats.get("lr_current")
            elif name == "grad_norm":
                stat = self._ema_stats.get("grad_norm")
            elif name == "update_ratio":
                stat = self._ema_stats.get("update_ratio")
        if stat is None or not stat.initialized:
            return 0.0  # until stats are warm, return neutral standardized value
        return stat.zscore(v, eps=1e-6)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fallback_zero(x: Any) -> float:
    v = _safe_float(x)
    return 0.0 if v is None else v
