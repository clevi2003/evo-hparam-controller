from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import time
import numpy as np
import torch

from src.core.config import load_train_cfg, load_controller_cfg, load_evolve_cfg
from src.utils.seed_device import seed_everything
from src.controller.serialization import flatten_params, unflatten_params, save_vector, to_numpy, from_numpy
from src.core.logging.loggers import Logger
from src.core.logging.logging_json_csv import CSVAppender
from src.evolution.evaluator import TruncatedTrainingEvaluator, EvaluatorConfig
from src.evolution.fitness import FitnessWeights, score_run
from src.evolution.algorithms import Strategy, GA, GAConfig, ES, ESConfig, Genome
from src.controller.controller import ControllerMLP


def _build_strategy(evo_cfg) -> Strategy:
    algo = evo_cfg.search.algo.lower()
    if algo == "ga":
        cfg = GAConfig(
            pop_size=evo_cfg.search.pop_size,
            elite=evo_cfg.search.elite,
            tournament_k=evo_cfg.search.tournament_k,
            sigma=evo_cfg.search.mutate_sigma,
            p_mutation=evo_cfg.search.p_mutation,
            sigma_decay=evo_cfg.search.sigma_decay,
        )
        return GA(cfg)
    elif algo == "es":
        cfg = ESConfig(
            mu=evo_cfg.search.parents,
            lam=evo_cfg.search.pop_size,
            sigma=evo_cfg.search.mutate_sigma,
            sigma_decay=evo_cfg.search.sigma_decay,
        )
        return ES(cfg)
    else:
        raise ValueError(f"Unsupported evolution algo '{evo_cfg.search.algo}'. Use 'ga' or 'es'.")


def _prepare_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "gen_logs").mkdir(parents=True, exist_ok=True)


def _save_configs(out_dir: Path, train_cfg, ctrl_cfg, evo_cfg) -> None:
    with (out_dir / "train_cfg.json").open("w") as f:
        json.dump(train_cfg.to_dict(), f, indent=2)
    with (out_dir / "controller_cfg.json").open("w") as f:
        json.dump(ctrl_cfg.to_dict(), f, indent=2)
    with (out_dir / "evolve_cfg.json").open("w") as f:
        json.dump(evo_cfg.to_dict(), f, indent=2)


def _init_controller_vector(ctrl_cfg, device: torch.device) -> torch.Tensor:
    """Instantiate a controller to determine parameter dimensionality and return a flat vector (torch)."""
    in_dim = len(ctrl_cfg.features)
    controller = ControllerMLP(
        in_dim=in_dim,
        hidden=ctrl_cfg.controller_arch.hidden,
        max_step=ctrl_cfg.action.max_step,
    ).to(device)
    flattened = flatten_params(controller, device=device, dtype=torch.float32)
    return flattened  # shape [D]


def run(args: argparse.Namespace) -> None:
    # load configs
    train_cfg = load_train_cfg(args.train_config)
    ctrl_cfg = load_controller_cfg(args.controller_config)
    evo_cfg = load_evolve_cfg(args.evolve_config)

    # prep output dir
    out_dir = Path(args.outdir)
    _prepare_outdir(out_dir)
    _save_configs(out_dir, train_cfg, ctrl_cfg, evo_cfg)

    # ensure determinism
    seed_everything(evo_cfg.budget.fixed_seed)

    # move to fast device if available
    device = torch.device(train_cfg.device) if train_cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init controller vector and strategy
    v0_t = _init_controller_vector(ctrl_cfg, device=device)
    dim = int(v0_t.numel())
    v0_np = to_numpy(v0_t)

    strategy = _build_strategy(evo_cfg)
    strategy.initialize(dim=dim, init_vec=v0_np)

    # init evaluator
    evaluator = TruncatedTrainingEvaluator(
        train_cfg=train_cfg,
        ctrl_cfg=ctrl_cfg,
        budget_cfg=evo_cfg.budget,
        evaluator_cfg=EvaluatorConfig(out_dir=None if args.no_artifacts else out_dir / "artifacts"),
    )

    # init fitness weights
    fw = FitnessWeights(
        primary=evo_cfg.fitness.primary,
        lr_volatility_weight=evo_cfg.fitness.lr_volatility_weight,
        nan_penalty=evo_cfg.fitness.nan_penalty,
    )

    # per generation logger (CSV for quick glance, artifacts contain Parquet if enabled)
    gen_log = Logger(CSVAppender(out_dir / "gen_logs" / "summary.csv",
                                 fieldnames=["generation", "candidate_idx", "fitness", "is_best", "elapsed_sec"]))

    best_score = -1e18
    best_vec_np: np.ndarray | None = None
    t_start = time.time()

    generations = int(getattr(evo_cfg.search, "generations", getattr(args, "generations", 10)))

    for gen in range(generations):
        cand_count = evo_cfg.search.pop_size if hasattr(evo_cfg.search, "pop_size") else 32
        candidates = strategy.ask(cand_count)

        evaluated: List[Genome] = []
        for idx, cand in enumerate(candidates):
            # convert vector to torch for evaluator
            vec_t = from_numpy(cand.vec, device=device, dtype=torch.float32)

            result = evaluator.evaluate(vec_t)
            summary = result["summary"]
            fitness = float(score_run(summary, fw))
            cand.score = fitness
            evaluated.append(cand)

            is_best = False
            if fitness > best_score:
                best_score = fitness
                best_vec_np = cand.vec.copy()
                is_best = True

                # save best vector and a controller state_dict for replay
                best_dir = out_dir / "checkpoints"
                best_dir.mkdir(parents=True, exist_ok=True)

                # save vector
                save_vector(best_dir / "best_controller_vec.pt", vec_t.detach().cpu(), meta={
                    "fitness": best_score,
                    "generation": gen,
                })

                # also save a controller .pt
                # build a fresh controller and load the vector
                controller = ControllerMLP(
                    in_dim=len(ctrl_cfg.features),
                    hidden=ctrl_cfg.controller_arch.hidden,
                    max_step=ctrl_cfg.action.max_step,
                ).to(device)
                unflatten_params(controller, vec_t, strict=True)
                torch.save({"state_dict": controller.state_dict(),
                            "meta": {"fitness": best_score, "generation": gen}},
                           best_dir / "best_controller.pt")

            gen_log.log({
                "generation": gen,
                "candidate_idx": idx,
                "fitness": fitness,
                "is_best": int(is_best),
                "elapsed_sec": time.time() - t_start,
            })

        # update strategy with evaluated genomes
        strategy.tell(evaluated)

        # per generation info
        snap_path = out_dir / "checkpoints" / f"strategy_gen{gen:04d}.pt"
        torch.save(strategy.state_dict(), snap_path)

    gen_log.close()

    # final best summary
    if best_vec_np is not None:
        print(f"[evolve] Best fitness: {best_score:.6f}")
        print(f"[evolve] Saved to: {out_dir / 'checkpoints'}")
    else:
        print("[evolve] No valid candidates evaluated.")

def main():
    parser = argparse.ArgumentParser(description="Evolutionary search runner for LR controller.")
    parser.add_argument("--train-config", required=True, help="Path to training YAML (e.g., configs/baseline.yaml)")
    parser.add_argument("--controller-config", required=True, help="Path to controller YAML (e.g., configs/controller.yaml)")
    parser.add_argument("--evolve-config", required=True, help="Path to evolve YAML (e.g., configs/evolve.yaml)")
    parser.add_argument("--outdir", required=True, help="Output directory for run artifacts/checkpoints")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations (fallback if not in YAML)")
    parser.add_argument("--no-artifacts", action="store_true", help="Disable per-candidate Parquet artifacts to speed up search")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
