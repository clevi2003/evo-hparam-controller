from .config_utils import _ensure_between, _ensure_choice, _ensure_positive, _dict_to_dataclass, _read_yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass
class ControllerActionCfg:
    lr_min: float = 1e-5
    lr_max: float = 1.0
    ema: float = 0.5           # smoothing of actions
    max_step: float = 0.2      # bounds for deltalog lr via tanh * max_step

    def validate(self) -> None:
        _ensure_between("controller.action.ema", self.ema, 0.0, 1.0)
        _ensure_positive("controller.action.max_step", self.max_step)
        if self.lr_min <= 0 or self.lr_max <= 0:
            raise ValueError("controller.action.lr_min and lr_max must be > 0.")
        if self.lr_min >= self.lr_max:
            raise ValueError("controller.action.lr_min must be < lr_max.")


@dataclass
class ControllerArchCfg:
    hidden: int = 32
    # Leave room for future architectures:
    depth: int = 2
    activation: str = "tanh"  # "tanh"|"relu" etc.

    def validate(self) -> None:
        _ensure_positive("controller.arch.hidden", self.hidden)
        _ensure_positive("controller.arch.depth", self.depth)
        _ensure_choice("controller.arch.activation", self.activation, ["tanh", "relu", "gelu", "silu"])


@dataclass
class ControllerCfg:
    enabled: bool = True
    update_interval: int = 50
    cooldown: int = 25
    feature_ema_alpha: float = 0.1  # for FeatureExtractor EMAs
    features: List[str] = field(default_factory=lambda: [
        "epoch", "step_in_epoch", "global_step",
        "train_loss", "val_loss", "train_acc", "val_acc",
        "lr_current", "ema_loss", "ema_acc", "d_loss", "d_acc"
    ])

    action: ControllerActionCfg = field(default_factory=ControllerActionCfg)
    controller_arch: ControllerArchCfg = field(default_factory=ControllerArchCfg)

    def validate(self) -> None:
        _ensure_positive("controller.update_interval", self.update_interval)
        _ensure_positive("controller.cooldown", self.cooldown, allow_zero=True)
        _ensure_between("controller.feature_ema_alpha", self.feature_ema_alpha, 0.0, 1.0)
        if not self.features:
            raise ValueError("controller.features must be a non-empty list.")
        self.action.validate()
        self.controller_arch.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _build_controller_cfg(d: Dict[str, Any]) -> ControllerCfg:
    action = _dict_to_dataclass(ControllerActionCfg, d.get("action", {}))
    arch = _dict_to_dataclass(ControllerArchCfg, d.get("controller_arch", {}))

    top = {k: v for k, v in d.items() if k not in ("action", "controller_arch")}
    cfg = ControllerCfg(
        action=action,
        controller_arch=arch,
        **top
    )
    cfg.validate()
    return cfg

def load_controller_cfg(path: Union[str, Path]) -> ControllerCfg:
    """Load & validate controller config (e.g., configs/controller.yaml)."""
    return _build_controller_cfg(_read_yaml(path))