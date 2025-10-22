from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import time
import numpy as np
import torch

from src.core.config import load_train_cfg, load_controller_cfg, load_evolve_cfg
from src.utils.seed_device import seed_everything
from src.controller.serialization import flatten_params, to_numpy
from src.evolution.evaluator import TruncatedTrainingEvaluator, EvaluatorConfig
#from src.evolution.fitness import FitnessWeights, score_run
from src.evolution.algorithms import Strategy, GA, GAConfig, ES, ESConfig, Genome
from src.controller.controller import ControllerMLP
from src.training.run_io import ensure_run_tree
from src.training.run_context import RunContext
from src.core.logging.loggers import make_evo_candidates_logger, make_gen_summary_logger
from src.evolution.eval_helpers import compose_candidate_row, compose_gen_summary_row, describe_candidate
from src.training.checkpoints import CheckpointIO

from tqdm import tqdm


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

    # seed everything for determinism
    fixed_seed = getattr(evo_cfg.budget, "fixed_seed", None)
    base_seed = getattr(evo_cfg, "random_seed", 12345)
    seed_everything(fixed_seed if fixed_seed is not None else base_seed)

    device = (
        torch.device(train_cfg.device)
        if getattr(train_cfg, "device", None)
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # setup run context (creates run_dir)
    run_context = RunContext([args.train_config, args.controller_config, args.evolve_config], cfg={
        "train_config_path": str(args.train_config),
        "controller_config_path": str(args.controller_config),
        "evolve_config_path": str(args.evolve_config),
        "args": vars(args),
        "schema_version": "v1",
    })
    run_context.set_env({
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    })
    run_context.set_config({
        "train": train_cfg,
        "controller": ctrl_cfg,
        "evolve": evo_cfg,
    })

    out_dir = Path(run_context.run_dir)
    paths = ensure_run_tree(out_dir)

    ckpt_io = CheckpointIO(root=out_dir / "checkpoints")

    # init evaluator
    artifacts_dir = None if args.no_artifacts else (out_dir / "artifacts")
    eval_cfg = EvaluatorConfig(
        out_dir=artifacts_dir,
        write_train_val_logs=True,
        write_controller_ticks=True,
        checkpoint_io=ckpt_io
    )
    evaluator = TruncatedTrainingEvaluator(
        train_cfg=train_cfg,
        ctrl_cfg=ctrl_cfg,
        budget_cfg=evo_cfg.budget,
        device=device,
        evaluator_cfg=eval_cfg
    )

    # evo log writers
    evo_cand_logger = make_evo_candidates_logger(paths["evo_candidates"])
    evo_gen_logger = make_gen_summary_logger(paths["evo_gen_summary"])

    # strategy and parameter dimension
    v0_t = _init_controller_vector(ctrl_cfg, device=device)
    dim = int(v0_t.numel())
    v0_np = to_numpy(v0_t)

    strategy = _build_strategy(evo_cfg)
    strategy.initialize(dim=dim, init_vec=v0_np)

    # overall evolution loop
    generations = args.generations or getattr(evo_cfg, "generations", 1)
    cand_count = getattr(evo_cfg.search, "pop_size", 32)

    best_fitness = -np.inf
    best_genome: Optional[Genome] = None
    t_start = time.time()
    # tqdm setup
    disable_tqdm = getattr(args, "no_tqdm", False) or not sys.stdout.isatty()

    try:
        gen_pbar = tqdm(
            iterable=range(generations),
            total=generations,
            desc="Generations",
            dynamic_ncols=True,
            disable=disable_tqdm,
            position=0
        )
        for gen in gen_pbar:
            # Ask strategy for candidates
            genomes = strategy.ask(cand_count)

            # Evaluate candidates
            results = []
            fitnesses = []

            cand_pbar = tqdm(
                iterable=enumerate(genomes),
                total=len(genomes),
                desc=f"Gen {gen} candidates",
                dynamic_ncols=True,
                leave=False,
                disable=disable_tqdm,
                position=1
            )

            for idx, genome in cand_pbar:
                # each genome should have a parameter vector and a seed
                vec = getattr(genome, "vec", None)
                seed = int(getattr(genome, "seed", base_seed + gen * 1000 + idx))

                # evaluate each candidate
                res = evaluator.evaluate_result(genome)
                results.append(res)
                fitness_value = float(res.fitness_primary)
                fitnesses.append(fitness_value)
                genome.fitness = fitness_value # communicate to strategy

                # update inner bar postfix with last fitness
                cand_pbar.set_postfix(last_fit=f"{fitness_value:.4f}")

                # per-candidate logging
                arch_str, hidden = describe_candidate(genome)
                try:
                    row = compose_candidate_row(
                        run_id=run_context.run_id,
                        generation=gen,
                        candidate_idx=idx,
                        candidate_seed=seed,
                        controller_params=vec if vec is not None else [],
                        eval_result=res,
                        budget_cfg=getattr(evo_cfg, "budget", {}),
                        arch_string=arch_str,
                        arch_hidden=hidden,
                        float_round_digits=getattr(evo_cfg, "float_round_digits", 7),
                    )
                    evo_cand_logger.log(row)
                except Exception:
                    # never crash evo bc of logging issues
                    pass

            # close inner bar before proceeding
            cand_pbar.close()
            # track best of generation
            if len(fitnesses) > 0:
                gen_best_idx = int(np.nanargmax(np.asarray(fitnesses)))
                gen_best_fit = float(fitnesses[gen_best_idx])
                if gen_best_fit > best_fitness:
                    best_fitness = gen_best_fit
                    best_genome = genomes[gen_best_idx]
                    try:
                        if hasattr(best_genome, "vec") and best_genome.vec is not None:
                            ckpt_io.save_best_vector(best_genome.vec)
                        # update outer bar immediately when a new global-best is found
                        gen_pbar.set_postfix(best=f"{best_fitness:.4f}")
                    except Exception:
                        pass

            # tell strategy about outcomes
            strategy.tell(genomes)

            # persist strategy snapshot for reproducibility
            try:
                snap_path = out_dir / "checkpoints" / f"strategy_gen{gen:04d}.pt"
                torch.save(strategy.state_dict(), snap_path)
            except Exception:
                pass

            # per-generation summary logging
            # try to extract sigma / mutation rate if available from strategy
            sigma = None
            ga_mut_rate = None
            try:
                # ES
                if hasattr(strategy, "sigma"):
                    sigma = getattr(strategy, "sigma")
                elif hasattr(strategy, "state") and hasattr(strategy.state, "sigma"):
                    sigma = getattr(strategy.state, "sigma")
            except Exception:
                sigma = None
            try:
                # GA
                if hasattr(strategy, "mutation_rate"):
                    ga_mut_rate = getattr(strategy, "mutation_rate")
                elif hasattr(strategy, "state") and hasattr(strategy.state, "mutation_rate"):
                    ga_mut_rate = getattr(strategy.state, "mutation_rate")
            except Exception:
                ga_mut_rate = None

            best_seed = -1
            if best_genome is not None:
                try:
                    best_seed = int(getattr(best_genome, "seed", -1))
                except Exception:
                    best_seed = -1

            try:
                gen_row = compose_gen_summary_row(
                    generation=gen,
                    population_size=len(genomes),
                    results=results,
                    best_idx=int(np.nanargmax(np.asarray(fitnesses))) if len(fitnesses) else 0,
                    best_seed=best_seed,
                    sigma=sigma,
                    ga_mutation_rate=ga_mut_rate,
                    promotions=getattr(evo_cfg, "promotions", []),
                )
                evo_gen_logger.log(gen_row)
            except Exception:
                pass

            # refresh outer bar postfix each generation
            gen_pbar.set_postfix(best=f"{best_fitness:.4f}" if np.isfinite(best_fitness) else "N/A",
                                 sigma=f"{sigma:.4f}" if sigma is not None else "—",
                                 mut=f"{ga_mut_rate:.4f}" if ga_mut_rate is not None else "—")
        gen_pbar.close()

        # final print summary
        elapsed = time.time() - t_start
        if best_genome is not None:
            tqdm.write(f"[evolve] Best fitness: {best_fitness:.6f} in {elapsed:.1f}s")
            try:
                best_path = out_dir / "checkpoints" / "best_controller_vec.pt"
                # if Genome has vec, persist it for convenience
                if hasattr(best_genome, "vec") and best_genome.vec is not None:
                    torch.save({"vec": np.asarray(best_genome.vec, dtype=np.float32)}, best_path)
                    tqdm.write(f"[evolve] Saved best vector to: {best_path}")
            except Exception:
                pass
        else:
            tqdm.write("[evolve] No valid candidates evaluated.")

    finally:
        try:
            evo_cand_logger.close()
        except Exception:
            pass
        try:
            evo_gen_logger.close()
        except Exception:
            pass
        run_context.finalize()

def main():
    parser = argparse.ArgumentParser(description="Evolutionary search runner for LR controller.")
    parser.add_argument("--train-config", required=True, help="Path to training YAML")
    parser.add_argument("--controller-config", required=True, help="Path to controller YAML")
    parser.add_argument("--evolve-config", required=True, help="Path to evolve YAML")
    parser.add_argument("--outdir", required=True, help="Output directory for run artifacts/checkpoints")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations (fallback if not in YAML)")
    parser.add_argument("--no-artifacts", action="store_true", help="Disable per-candidate Parquet artifacts to speed up search")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
