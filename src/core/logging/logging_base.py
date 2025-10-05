from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from datetime import datetime

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise ImportError(
        "Parquet-first logging requires 'pyarrow'. "
    ) from e

def _ensure_parent_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_row(row: Dict[str, Any], fieldnames: Sequence[str], fill: Any = None) -> Dict[str, Any]:
    return {k: row.get(k, fill) for k in fieldnames}

class BaseAppender:
    """
    Minimal append-only interface. Implementations are responsible for buffering,
    schema consistency, and safe resource closing.
    """
    def append(self, row: Dict[str, Any]) -> None:
        raise NotImplementedError

    def append_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        for r in rows:
            self.append(r)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()