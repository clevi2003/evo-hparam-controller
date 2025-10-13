from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.logging.loggers import (
    make_evo_candidates_logger,
    make_gen_summary_logger,
)


def paths_for_run(run_dir: Path) -> Dict[str, Path]:
    """
    calc agrreed upon artifact paths under the run directory
    """
    run_dir = Path(run_dir)
    return {
        "train_log": run_dir / "logs_train.parquet",
        "val_log": run_dir / "logs_val.parquet",
        "controller_calls": run_dir / "controller_calls.parquet",
        "checkpoints": run_dir / "checkpoints",
        "graphs": run_dir / "graphs",
        "evolution": run_dir / "evolution",
        "evo_candidates": run_dir / "evolution" / "candidates.parquet",
        "evo_gen_summary": run_dir / "evolution" / "gen_summary.parquet",
    }


def ensure_run_tree(run_dir: Path) -> Dict[str, Path]:
    """
    create the standardized folder layout and return all resolved paths
    """
    p = paths_for_run(run_dir)
    p["checkpoints"].mkdir(parents=True, exist_ok=True)
    p["graphs"].mkdir(parents=True, exist_ok=True)
    p["evolution"].mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class RunWriters:
    train: Any
    val: Any
    controller_calls: Any
    candidates: Any
    gen_summary: Any


def open_evolution_writers(paths: Dict[str, Path]) -> RunWriters:
    """
    Initialize parquet writers for all artifacts
    """
    candidates = make_evo_candidates_logger(paths["evo_candidates"])
    gen_summary = make_gen_summary_logger(paths["evo_gen_summary"])
    # return handles so the caller can close explicitly
    return RunWriters(train=None, val=None, controller_calls=None,
                      candidates=candidates, gen_summary=gen_summary)


def close_writers(w: Optional[RunWriters]) -> None:
    if w is None:
        return
    for h in (w.train, w.val, w.controller_calls, w.candidates, w.gen_summary):
        try:
            if h is not None and hasattr(h, "close"):
                h.close()
        except Exception:
            pass


@dataclass
class IOContext:
    paths: Dict[str, Path]
    writers: RunWriters


def bootstrap_io(run_context, symlink_legacy_tick: bool = True) -> IOContext:
    """
    Create Phase-4 folder layout and all parquet writers under RunContext.run_dir.
    """
    run_dir: Path = Path(run_context.run_dir)
    paths = ensure_run_tree(run_dir)
    writers = open_writers(paths)
    return IOContext(paths=paths, writers=writers)
