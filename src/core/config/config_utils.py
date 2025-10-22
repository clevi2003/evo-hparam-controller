from __future__ import annotations

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

def _normalize_yaml_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept both old and new shapes and coerce into the dataclass schema
    """
    d = dict(d)

    # pull train section up into TrainCfg fields
    train_sec = d.get("train", {}) or {}
    if "epochs" in train_sec and "epochs" not in d:
        d["epochs"] = train_sec["epochs"]
    if "label_smoothing" in train_sec and "label_smoothing" not in d:
        d["label_smoothing"] = train_sec["label_smoothing"]
    if "grad_clip_norm" in train_sec and "grad_clip_norm" not in d:
        d["grad_clip_norm"] = train_sec["grad_clip_norm"]
    if "deterministic" in train_sec and "deterministic" not in d:
        d["deterministic"] = train_sec["deterministic"]

    # standardize scheduler key name
    if "scheduler" in d and "sched" not in d:
        d["sched"] = d.pop("scheduler")

    # standardize checkpoint section name
    if "checkpoint" in d and "ckpt" not in d:
        d["ckpt"] = d.pop("checkpoint")
    if "checkpoints" in d and "ckpt" not in d:
        d["ckpt"] = d.pop("checkpoints")

    # data aliases
    data_sec = d.get("data", {}) or {}
    if "root" in data_sec and "data_root" not in data_sec:
        data_sec["data_root"] = data_sec.pop("root")
    d["data"] = data_sec

    # model aliases
    model_sec = d.get("model", {}) or {}
    if "name" in model_sec and "arch" not in model_sec:
        model_sec["arch"] = model_sec.pop("name")
    d["model"] = model_sec

    # scheduler aliases
    sched_sec = d.get("sched", {}) or {}
    if "name" in sched_sec and isinstance(sched_sec["name"], str):
        sched_sec["name"] = sched_sec["name"].strip()
    d["sched"] = sched_sec

    # ckpt aliases
    ckpt_sec = d.get("ckpt", {}) or {}
    d["ckpt"] = ckpt_sec

    # ensure log_interval is int
    log_sec = d.get("log", {}) or {}
    if "log_interval" in log_sec:
        try:
            log_sec["log_interval"] = int(log_sec["log_interval"])
        except Exception:
            raise ValueError("log.log_interval must be an integer.")
    d["log"] = log_sec

    return d