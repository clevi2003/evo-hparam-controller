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