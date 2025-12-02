from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math


Point = Tuple[int, float]


def auc(points: Sequence[Point]) -> float:
    """
    Compute trapezoidal AUC over (x, y) pairs sorted by x.
    If fewer than 2 points, returns 0.0.
    """
    if not points or len(points) < 2:
        return 0.0
    # ensure sorted by x
    pts = sorted(points, key=lambda point: point[0])
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx = float(x1 - x0)
        if dx <= 0:
            continue
        area += 0.5 * dx * (float(y0) + float(y1))
    return area


def std_dev(xs: Iterable[float]) -> float:
    xs_list = [float(x) for x in xs]
    n = len(xs_list)
    if n <= 1:
        return 0.0
    mean = sum(xs_list) / n
    var = sum((x - mean) ** 2 for x in xs_list) / (n - 1)
    return math.sqrt(max(0.0, var))


def lr_volatility_penalty(deltas: Sequence[float]) -> float:
    """
    A small fitness penalty
    based on the standard deviation of delta log lr.
    """
    return std_dev(deltas)


@dataclass
class FitnessWeights:
    """
    Tunable weights for combining metrics into a single scalar fitness.
    """
    primary: str = "auc_val_acc"  # or "final_val_acc"
    lr_volatility_weight: float = 0.1
    nan_penalty: float = 100.0  # applied when divergence/NaN detected


@dataclass
class RunSummary:
    """
    Minimal data required to compute fitness for a truncated run.
    """
    # key curves
    val_acc_curve: List[Point] # (global_step, val_acc)
    action_deltas: List[float] # delta log lr applied over time

    # convenience scalars
    final_val_acc: float
    diverged: bool # True if nan/inf/explosions detected

    # optional metadata not used for scoring but might be good for logs
    total_steps: int = 0
    total_epochs: int = 0


def compute_primary(summary: RunSummary, primary: str) -> float:
    if primary == "auc_val_acc":
        return auc(summary.val_acc_curve)
    elif primary == "final_val_acc":
        return float(summary.final_val_acc)
    else:
        raise ValueError(f"Unknown primary metric '{primary}'")


def score_run(summary: RunSummary, w: FitnessWeights) -> float:
    """
    Combine metrics into a fitness value
    Fitness = primary - lr_volatility_weight * std(delta log lr) - nan_penalty(if diverged)
    """
    score = compute_primary(summary, w.primary)
    if w.lr_volatility_weight > 0.0 and summary.action_deltas:
        score -= float(w.lr_volatility_weight) * lr_volatility_penalty(summary.action_deltas)
    if summary.diverged:
        score -= float(w.nan_penalty)
    return float(score)
