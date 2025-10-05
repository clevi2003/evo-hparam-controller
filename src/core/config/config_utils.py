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