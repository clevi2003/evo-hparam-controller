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
    generations: int = 10
    warm_start_candidates_dir: Optional[str] = None
    warm_start_num_candidates: int = 0

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
        if self.warm_start_num_candidates < 0:
            raise ValueError("evolve.search.warm_start_num_candidates must be >= 0.")
        if self.warm_start_num_candidates > 0 and not self.warm_start_candidates_dir:
            raise ValueError(
                "evolve.search.warm_start_candidates_dir must be set if "
                "warm_start_num_candidates > 0."
            )
        if self.warm_start_num_candidates > self.pop_size:
            raise ValueError(
                "evolve.search.warm_start_num_candidates must be <= pop_size."
            )


@dataclass
class BudgetCfg:
    epochs: int = 25
    fixed_seed: int = 2025
    fixed_subset_pct: float = 0.3  # for CIFAR-10 etc.; evaluator should honor this
    max_steps: Optional[int] = None
    start_from_checkpoint: bool = False
    checkpoint_path: Optional[str] = None

    def validate(self) -> None:
        _ensure_positive("evolve.budget.epochs", self.epochs)
        _ensure_positive("evolve.budget.fixed_seed", self.fixed_seed, allow_zero=True)
        _ensure_between("evolve.budget.fixed_subset_pct", self.fixed_subset_pct, 0.0, 1.0)
        if self.max_steps is not None:
            _ensure_positive("evolve.budget.max_steps", self.max_steps)
        if self.start_from_checkpoint and not self.checkpoint_path:
            raise ValueError(
                "evolve.budget.checkpoint_path must be set if start_from_checkpoint is True."
            )


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
    save_top_k_controllers: int = 1
    save_best_model: bool = False


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