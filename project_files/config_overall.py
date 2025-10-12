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
    logging_features_json: bool = True

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
    # action = _dict_to_dataclass(ControllerActionCfg, d.get("action", {}))
    # arch = _dict_to_dataclass(ControllerArchCfg, d.get("controller_arch", {}))

    # top = {k: v for k, v in d.items() if k not in ("action", "controller_arch")}
    cfg = ControllerCfg(
        update_interval=int(d.get("update_interval", 50)),
        cooldown=int(d.get("cooldown", 25)),
        feature_ema_alpha=float(d.get("feature_ema_alpha", 0.1)),
        features=list(d.get("features", [])) or ControllerCfg().features,
        action=_dict_to_dataclass(ControllerActionCfg, d.get("action", {})),
        controller_arch=_dict_to_dataclass(ControllerArchCfg, d.get("controller_arch", {})),
        logging_features_json = bool(d.get("logging", {}).get("features_json", True)),
    )
    cfg.validate()
    return cfg

def load_controller_cfg(path: Union[str, Path]) -> ControllerCfg:
    """Load & validate controller config (e.g., configs/controller.yaml)."""
    return _build_controller_cfg(_read_yaml(path))

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
from .config_utils import _as_path, _ensure_between, _ensure_choice, _ensure_positive, _dict_to_dataclass, _read_yaml

@dataclass
class SearchCfg:
    algo: str = "ga"  # "ga" | "es" | "cmaes" (TODO later)
    pop_size: int = 32
    parents: int = 8
    elite: int = 2
    mutate_sigma: float = 0.05
    p_mutation: float = 0.2
    tournament_k: int = 3
    sigma_decay: float = 1.0  # 1.0 = no decay

    def validate(self) -> None:
        _ensure_choice("evolve.search.algo", self.algo, ["ga", "es", "cmaes"])
        _ensure_positive("evolve.search.pop_size", self.pop_size)
        _ensure_positive("evolve.search.parents", self.parents)
        _ensure_positive("evolve.search.elite", self.elite, allow_zero=True)
        _ensure_between("evolve.search.p_mutation", self.p_mutation, 0.0, 1.0)
        _ensure_positive("evolve.search.mutate_sigma", self.mutate_sigma, allow_zero=False)
        _ensure_positive("evolve.search.tournament_k", self.tournament_k)
        _ensure_positive("evolve.search.sigma_decay", self.sigma_decay, allow_zero=False)
        if self.parents > self.pop_size:
            raise ValueError("evolve.search.parents must be <= pop_size.")
        if self.elite > self.pop_size:
            raise ValueError("evolve.search.elite must be <= pop_size.")


@dataclass
class BudgetCfg:
    epochs: int = 25
    fixed_seed: int = 2025
    fixed_subset_pct: float = 0.3  # for CIFAR-10 etc.; evaluator should honor this
    max_steps: Optional[int] = None

    def validate(self) -> None:
        _ensure_positive("evolve.budget.epochs", self.epochs)
        _ensure_positive("evolve.budget.fixed_seed", self.fixed_seed, allow_zero=True)
        _ensure_between("evolve.budget.fixed_subset_pct", self.fixed_subset_pct, 0.0, 1.0)
        if self.max_steps is not None:
            _ensure_positive("evolve.budget.max_steps", self.max_steps)


@dataclass
class FitnessWeightsCfg:
    primary: str = "auc_val_acc"  # main objective
    # Optional penalties:
    lr_volatility_weight: float = 0.0
    nan_penalty: float = 100.0

    def validate(self) -> None:
        _ensure_choice("evolve.fitness.primary", self.primary, ["auc_val_acc", "final_val_acc"])
        _ensure_positive("evolve.fitness.nan_penalty", self.nan_penalty, allow_zero=False)
        _ensure_positive("evolve.fitness.lr_volatility_weight", self.lr_volatility_weight, allow_zero=True)

@dataclass
class EvolveLoggingCfg:
    artifacts: bool = True # write per-candidate parquet artifacts (can be disabled for speed)


@dataclass
class EvolveCfg:
    search: SearchCfg = field(default_factory=SearchCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    fitness: FitnessWeightsCfg = field(default_factory=FitnessWeightsCfg)
    out_dir: Path = Path("./runs")
    logging: EvolveLoggingCfg = field(default_factory=EvolveLoggingCfg)

    def validate(self) -> None:
        self.search.validate()
        self.budget.validate()
        self.fitness.validate()
        self.out_dir = _as_path(self.out_dir) or Path("./runs").resolve()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d

def _build_evolve_cfg(d: Dict[str, Any]) -> EvolveCfg:
    search = _dict_to_dataclass(SearchCfg, d.get("search", {}))
    budget = _dict_to_dataclass(BudgetCfg, d.get("budget", {}))
    fitness = _dict_to_dataclass(FitnessWeightsCfg, d.get("fitness", {}))
    logging = _dict_to_dataclass(EvolveLoggingCfg, d.get("logging", {}))

    top = {k: v for k, v in d.items() if k not in ("search", "budget", "fitness", "logging")}
    cfg = EvolveCfg(
        search=search,
        budget=budget,
        fitness=fitness,
        logging=logging,
        **top
    )
    cfg.validate()
    return cfg

def load_evolve_cfg(path: Union[str, Path]) -> EvolveCfg:
    """Load & validate evolution config (e.g., configs/evolve.yaml)."""
    return _build_evolve_cfg(_read_yaml(path))

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dataclasses import asdict, dataclass, field
from .config_utils import _as_path, _ensure_between, _ensure_positive, _dict_to_dataclass, _read_yaml

@dataclass
class DataCfg:
    dataset: str = "CIFAR10"
    data_root: Path = Path("./data")
    batch_size: int = 128
    num_workers: int = 4
    subset_fraction: float = 1.0  # used by evaluator when you want a smaller budget

    def validate(self) -> None:
        _ensure_positive("data.batch_size", self.batch_size)
        _ensure_positive("data.num_workers", self.num_workers, allow_zero=True)
        _ensure_between("data.subset_fraction", self.subset_fraction, 0.0, 1.0)
        self.data_root = _as_path(self.data_root) or Path("./data").resolve()


@dataclass
class ModelCfg:
    arch: str = "resnet20"  # matches typical CIFAR-10 ResNet-20
    num_classes: int = 10
    # Optional arch args:
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive("model.num_classes", self.num_classes)


@dataclass
class OptimCfg:
    name: str = "SGD"  # or "Adam", etc.
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9  # used if optimizer supports it
    betas: Optional[Tuple[float, float]] = None  # for Adam-like
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive("optim.lr", self.lr)
        _ensure_positive("optim.weight_decay", self.weight_decay, allow_zero=True)
        _ensure_positive("optim.momentum", self.momentum, allow_zero=True)
        if self.betas is not None:
            if len(self.betas) != 2:
                raise ValueError("optim.betas must be a tuple of length 2.")
            for i, b in enumerate(self.betas):
                _ensure_between(f"optim.betas[{i}]", b, 0.0, 1.0)


@dataclass
class SchedCfg:
    name: Optional[str] = "cosine"  # or None/"step"/"multistep"/"none"
    # Common scheduler params:
    warmup_epochs: int = 0
    step_size: int = 30
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=list)
    t_max: Optional[int] = None  # cosine
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self, total_epochs: int) -> None:
        if self.name is None or self.name.lower() in ("none", "off", "disabled"):
            return
        _ensure_positive("sched.warmup_epochs", self.warmup_epochs, allow_zero=True)
        _ensure_positive("sched.step_size", self.step_size, allow_zero=True)
        _ensure_between("sched.gamma", self.gamma, 0.0, 1.0)
        if self.t_max is not None:
            _ensure_positive("sched.t_max", self.t_max)
        # milestones can be empty or ascending ints
        if any(m < 0 for m in self.milestones):
            raise ValueError("sched.milestones must be non-negative integers.")
        if any(self.milestones[i] > self.milestones[i + 1] for i in range(len(self.milestones) - 1)):
            raise ValueError("sched.milestones must be in non-decreasing order.")
        if total_epochs <= 0:
            raise ValueError("Total epochs must be > 0 for scheduler validation.")

@dataclass
class TrainLoggingCfg:
    # Parquet-first outputs in the run root
    out_dir: str = "Runs" # base Runs/ root; final run folder decided at runtime
    controller_ticks: bool = True # write controller_calls.parquet
    train_val_scalars: bool = True # write logs_train.parquet & logs_val.parquet
    features_json: bool = True # include a JSON feature snapshot in ticks (compact & flexible)

    dir_tb: Optional[str] = None # "./runs" for TensorBoard if used
    csv_path: Optional[str] = None # back compatibility with CSV summary path
    log_interval: int = 100

@dataclass
class TrainCfg:
    # Top-level training options likely to match baseline.yaml
    epochs: int = 200
    max_steps: Optional[int] = None  # if set, can cap total steps regardless of epochs
    device: Optional[str] = None     # "cuda", "mps", "cpu", or None to auto-detect
    seed: int = 42
    log_dir: Path = Path("./runs")
    log: TrainLoggingCfg = field(default_factory=TrainLoggingCfg)

    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    sched: SchedCfg = field(default_factory=SchedCfg)

    # Anything extra to carry through without breaking:
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _ensure_positive("train.epochs", self.epochs)
        if self.max_steps is not None:
            _ensure_positive("train.max_steps", self.max_steps)
        _ensure_positive("train.seed", self.seed, allow_zero=True)
        self.log_dir = _as_path(self.log_dir) or Path("./runs").resolve()

        self.data.validate()
        self.model.validate()
        self.optim.validate()
        self.sched.validate(total_epochs=self.epochs)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["log_dir"] = str(self.log_dir)
        d["data"]["data_root"] = str(self.data.data_root)
        return d

def _build_train_cfg(d: Dict[str, Any]) -> TrainCfg:
    data = _dict_to_dataclass(DataCfg, d.get("data", {}))
    model = _dict_to_dataclass(ModelCfg, d.get("model", {}))
    optim = _dict_to_dataclass(OptimCfg, d.get("optim", {}))
    sched = _dict_to_dataclass(SchedCfg, d.get("sched", {}))

    top = {k: v for k, v in d.items() if k not in ("data", "model", "optim", "sched", "log")}
    log_src = d.get("log", {}) or {}
    train_log = TrainLoggingCfg(
        out_dir="Runs",  # can be overridden by runner at runtime
        controller_ticks=True,
        train_val_scalars=True,
        features_json=True,
        dir_tb=log_src.get("dir_tb"),
        csv_path=log_src.get("csv_path"),
        log_interval=int(log_src.get("log_interval", 100)),
    )
    cfg = TrainCfg(
        data=data,
        model=model,
        optim=optim,
        sched=sched,
        log=train_log,
        **top
    )
    cfg.validate()
    return cfg

def load_train_cfg(path: Union[str, Path]) -> TrainCfg:
    """Load & validate training config (e.g., configs/baseline.yaml)."""
    return _build_train_cfg(_read_yaml(path))


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

def _read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping (dict). Got: {type(data)} in {p}")
    return data


def _ensure_positive(name: str, value: Optional[Union[int, float]], allow_zero: bool = False) -> None:
    if value is None:
        raise ValueError(f"Missing required positive value for '{name}'.")
    if allow_zero:
        if not (value >= 0):
            raise ValueError(f"'{name}' must be >= 0. Got {value}.")
    else:
        if not (value > 0):
            raise ValueError(f"'{name}' must be > 0. Got {value}.")


def _ensure_between(name: str, value: float, lo: float, hi: float, inclusive: bool = True) -> None:
    if inclusive:
        ok = (lo <= value <= hi)
    else:
        ok = (lo < value < hi)
    if not ok:
        raise ValueError(f"'{name}' must be between {lo} and {hi}{' inclusive' if inclusive else ''}. Got {value}.")


def _ensure_choice(name: str, value: str, choices: List[str]) -> None:
    if value not in choices:
        raise ValueError(f"'{name}' must be one of {choices}. Got '{value}'.")


def _as_path(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()

def _dict_to_dataclass(cls, data: Dict[str, Any]):
    # Shallow mapping for known nested types; keeps future-proofing by ignoring extras
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)