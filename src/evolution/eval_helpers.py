from __future__ import annotations
import json
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


def describe_candidate(candidate: Any) -> Tuple[str, Dict[str, Any]]:
    """
    return a stable (arch_string, hidden_dict) description of the candidate
    tries common attributes but falls back to type name if those fail
    """
    # try common patterns
    if hasattr(candidate, "arch_string"):
        arch = str(candidate.arch_string)
    elif hasattr(candidate, "controller") and hasattr(candidate.controller, "arch_string"):
        arch = str(candidate.controller.arch_string)
    elif hasattr(candidate, "arch"):
        arch = str(candidate.arch)
    else:
        arch = f"{type(candidate).__name__}"

    hidden: Dict[str, Any] = {}
    for key in ("hidden", "arch_kwargs", "extra", "cfg"):
        if hasattr(candidate, key):
            try:
                val = getattr(candidate, key)
                # Avoid dumping giant objects
                if isinstance(val, dict):
                    hidden = val
                break
            except Exception:
                pass
    return arch, hidden


def compose_candidate_row(
    *,
    run_id: str,
    generation: int,
    candidate_idx: int,
    candidate_seed: int,
    controller_params: Union[np.ndarray, list[float], tuple[float, ...]],
    eval_result: EvalResult,
    budget_cfg: Dict[str, Any],
    arch_string: str,
    arch_hidden: Dict[str, Any],
    float_round_digits: int = 7,
) -> Dict[str, Any]:
    """
    build one row for evolution/candidates.parquet
    """
    row: Dict[str, Any] = {
        # descriptor
        "run_id": run_id,
        "generation": int(generation),
        "candidate_idx": int(candidate_idx),
        "seed": int(candidate_seed),
        "controller_params": serialize_params(controller_params, round_digits=float_round_digits),
        "arch": arch_string,
        "hidden": json.dumps(arch_hidden or {}, separators=(",", ":")),
        # fitness
        "fitness": float(eval_result.fitness_primary),
        "primary_metric": str(eval_result.primary_metric),
        "penalties_json": json.dumps(eval_result.penalties or {}, separators=(",", ":")),
        "fitness_vector_json": json.dumps(eval_result.fitness_vector or [], separators=(",", ":")),
        # budget
        "budget_epochs": int(budget_cfg.get("epochs", 0)) if budget_cfg else None,
        "budget_steps": int(budget_cfg.get("steps", 0)) if budget_cfg else None,
        "used_epochs": int(eval_result.budget_used.get("epochs", 0)),
        "used_steps": int(eval_result.budget_used.get("steps", 0)),
        "wall_time_s": float(eval_result.budget_used.get("wall_time_s", 0.0)),
        "truncation_reason": str(eval_result.truncation_reason or "complete"),
        # lifecycle
        "promoted": False,
        "artifacts_json": json.dumps(eval_result.artifacts or {}, separators=(",", ":")),
    }
    return row


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
