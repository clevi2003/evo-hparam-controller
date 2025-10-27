from __future__ import annotations
import json
import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import numpy as np

from src.evolution.evaluator import EvalResult


def _round_list(xs: Iterable[float], digits: int) -> list[float]:
    return [float(round(float(v), digits)) for v in xs]


def serialize_params(vec: Union[np.ndarray, list[float], tuple[float, ...]],
                     round_digits: int = 7) -> str:
    """
    compact JSON for controller parameters, keeps parquet cells small & diffable
    """
    if isinstance(vec, np.ndarray):
        data = vec.astype(float).tolist()
    else:
        data = list(vec)
    return json.dumps(_round_list(data, round_digits), separators=(",", ":"))


def describe_candidate(genome) -> Tuple[str, int]:
    """
    Return a display string and a 'hidden' size for quick filtering.
    If you do not encode arch in the genome, fall back to simple labels.
    """
    hidden = getattr(getattr(genome, "arch", None), "hidden", None)
    if hidden is None:
        hidden = int(getattr(genome, "hidden", 0) or 0)
    arch_str = getattr(getattr(genome, "arch", None), "name", None)
    if arch_str is None:
        arch_str = "ControllerMLP"
    return str(arch_str), int(hidden)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _norms(vec: Any) -> Tuple[int, float, float]:
    try:
        arr = np.asarray(vec, dtype=np.float32)
        dim = int(arr.size)
        l2 = float(np.linalg.norm(arr)) if dim > 0 else 0.0
        max_abs = float(np.max(np.abs(arr))) if dim > 0 else 0.0
        return dim, l2, max_abs
    except Exception:
        return 0, 0.0, 0.0

def compose_candidate_row(
    *,
    run_id: str,
    generation: int,
    candidate_idx: int,
    candidate_seed: int,
    controller_params: Any,
    eval_result: Any,
    budget_cfg: Any,
    arch_string: str,
    arch_hidden: int,
    float_round_digits: int = 7,
) -> Dict[str, Any]:
    """
    Produce a row dict that matches EVO_CANDIDATES_SCHEMA.
    Works with EvalResult returned from evaluator.evaluate_result.
    """
    # unpack eval_result
    fitness = _safe_float(getattr(eval_result, "fitness_primary", None), 0.0)
    primary_metric = getattr(eval_result, "primary_metric", "fitness")
    artifacts = getattr(eval_result, "artifacts", {}) or {}

    # metrics_snapshot is optional in your evaluator's return path
    metrics_snapshot = getattr(eval_result, "metrics_snapshot", None)
    if metrics_snapshot is None and isinstance(eval_result, dict):
        metrics_snapshot = eval_result.get("metrics_snapshot")

    auc_val = 0.0
    final_val_acc = 0.0
    lr_delta_std = 0.0
    if isinstance(metrics_snapshot, dict):
        auc_val = _safe_float(metrics_snapshot.get("auc_val_acc"), 0.0)
        final_val_acc = _safe_float(metrics_snapshot.get("final_val_acc"), 0.0)
        lr_delta_std = _safe_float(metrics_snapshot.get("lr_delta_std"), 0.0)

    # diverged and step/epoch counts
    diverged = False
    total_steps = 0
    total_epochs = 0
    # eval_result may have "summary" with these numbers
    summary = getattr(eval_result, "summary", None)
    if summary is None and isinstance(eval_result, dict):
        summary = eval_result.get("summary")
    if summary is not None:
        try:
            diverged = bool(getattr(summary, "diverged", False))
            total_steps = int(getattr(summary, "total_steps", 0))
            total_epochs = int(getattr(summary, "total_epochs", 0))
        except Exception:
            pass
    else:
        # try dict fields on eval_result
        diverged = bool(getattr(eval_result, "diverged", False) or False)

    # budget
    budget_epochs = int(getattr(getattr(budget_cfg, "__dict__", {}), "epochs", getattr(budget_cfg, "epochs", 0)) or 0)
    budget_max_steps = int(getattr(getattr(budget_cfg, "__dict__", {}), "max_steps", getattr(budget_cfg, "max_steps", 0)) or 0)

    # params
    param_dim, param_l2, param_max_abs = _norms(controller_params)

    # version tag if present on eval_result.artifacts or genome
    controller_version = ""
    try:
        controller_version = str(getattr(eval_result, "controller_version", "")) or ""
    except Exception:
        controller_version = ""

    # trial_id not yet used, placeholder 0
    trial_id = 0
    # artifacts json
    try:
        artifacts_json = json.dumps(artifacts)
    except Exception:
        artifacts_json = "{}"

    return {
        "run_id": str(run_id),
        "generation": int(generation),
        "candidate_idx": int(candidate_idx),
        "candidate_seed": int(candidate_seed),
        "trial_id": int(trial_id),
        "fitness": float(round(fitness, float_round_digits)),
        "primary_metric": str(primary_metric),
        "auc_val_acc": float(round(auc_val, float_round_digits)),
        "final_val_acc": float(round(final_val_acc, float_round_digits)),
        "lr_delta_std": float(round(lr_delta_std, float_round_digits)),
        "diverged": bool(diverged),
        "total_steps": int(total_steps),
        "total_epochs": int(total_epochs),
        "arch_string": str(arch_string),
        "arch_hidden": int(arch_hidden),
        "controller_version": str(controller_version),
        "param_dim": int(param_dim),
        "param_l2": float(round(param_l2, float_round_digits)),
        "param_max_abs": float(round(param_max_abs, float_round_digits)),
        "budget_epochs": int(budget_epochs),
        "budget_max_steps": int(budget_max_steps),
        "artifacts_json": artifacts_json,
    }


def compose_gen_summary_row(
    *,
    generation: int,
    population_size: int,
    results: list[Optional[EvalResult]],
    best_idx: int,
    best_seed: int,
    sigma: Optional[Union[float, list[float]]] = None,
    ga_mutation_rate: Optional[float] = None,
    promotions: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """
    build one row for evolution/gen_summary.parquet.
    """
    fitnesses = [r.fitness_primary for r in results if r is not None]
    if len(fitnesses) == 0:
        # avoid nans in empty gens
        mean_fit = 0.0
        std_fit = 0.0
        best_fit = 0.0
    else:
        mean_fit = float(np.mean(fitnesses))
        std_fit = float(np.std(fitnesses))
        best_fit = float(results[best_idx].fitness_primary)

    if sigma is None:
        mutation_sigma = None
        sigma_json = "[]"
    elif isinstance(sigma, (float, int)):
        mutation_sigma = float(sigma)
        sigma_json = "[]"
    else:
        mutation_sigma = None
        sigma_json = json.dumps(list(map(float, sigma)), separators=(",", ":"))

    row: Dict[str, Any] = {
        "generation": int(generation),
        "pop_size": int(population_size),
        "eval_count": int(len(fitnesses)),
        "best_fitness": best_fit,
        "mean_fitness": mean_fit,
        "std_fitness": std_fit,
        "best_idx": int(best_idx),
        "best_seed": int(best_seed),
        "mutation_sigma": mutation_sigma,
        "sigma_json": sigma_json,
        "ga_mutation_rate": float(ga_mutation_rate) if ga_mutation_rate is not None else None,
        "promotions_json": json.dumps(promotions or [], separators=(",", ":")),
        "notes": "",
    }
    return row
