from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np
import random

@dataclass
class Genome:
    vec: np.ndarray
    score: Optional[float] = None
    # optional metadata for debugging
    meta: dict | None = None


class Strategy:
    """
    minimal strategy interface used by the runner:
      initialize(dim, init_vec) to set up internal state
      ask(n) to propose n candidate genomes (vec only)
      tell(evaluated) to receive fitness and update state
      state_dict()/load_state_dict() for checkpointing
    """
    def initialize(self, dim: int, init_vec: np.ndarray | None = None) -> None:
        raise NotImplementedError

    def ask(self, n: int) -> List[Genome]:
        raise NotImplementedError

    def tell(self, genomes: List[Genome]) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


# generic genetic algorithm: (μ + λ) with tournament selection, gaussian mutation, and elitism
@dataclass
class GAConfig:
    pop_size: int = 32
    elite: int = 2
    tournament_k: int = 3
    sigma: float = 0.05 # std for Gaussian mutation in parameter space
    p_mutation: float = 0.2 # per coordinate mutation probability
    sigma_decay: float = 1.0 # multiply sigma each generation (1.0 = none)
    clip: Optional[float] = None # clip magnitude per coordinate (optional)


class GA(Strategy):
    def __init__(self, cfg: GAConfig) -> None:
        self.cfg = cfg
        self.dim: int = 0
        self._pop: List[Genome] = []
        self._gen: int = 0
        self._sigma: float = float(cfg.sigma)

    def initialize(self, dim: int, init_vec: np.ndarray | None = None) -> None:
        self.dim = int(dim)
        self._sigma = float(self.cfg.sigma)
        self._gen = 0

        if init_vec is None:
            init_vec = np.zeros(self.dim, dtype=np.float64)

        self._pop = []
        for _ in range(self.cfg.pop_size):
            vec = init_vec.copy()
            # small random jitter to break symmetry
            vec += np.random.normal(0.0, self._sigma, size=self.dim)
            self._pop.append(Genome(vec=vec))

    def _tournament(self, k: int) -> Genome:
        # pick k at random and return best by score, score should be set
        cand = random.sample(self._pop, k=min(k, len(self._pop)))
        cand = [g for g in cand if g.score is not None]
        if not cand:
            # fall back to a random member
            return random.choice(self._pop)
        return max(cand, key=lambda g: g.score)

    def _mutate(self, parent_vec: np.ndarray) -> np.ndarray:
        # gaussian mutation with per coordinate probability
        child = parent_vec.copy()
        mask = np.random.rand(self.dim) < self.cfg.p_mutation
        noise = np.random.normal(0.0, self._sigma, size=self.dim)
        child[mask] += noise[mask]
        if self.cfg.clip is not None:
            np.clip(child, -self.cfg.clip, self.cfg.clip, out=child)
        return child

    def ask(self, n: int) -> List[Genome]:
        # use tournament selection to pick parents
        out: List[Genome] = []
        for _ in range(n):
            p = self._tournament(self.cfg.tournament_k)
            child_vec = self._mutate(p.vec)
            out.append(Genome(vec=child_vec))
        return out

    def tell(self, genomes: List[Genome]) -> None:
        # merge current population and new evaluated offspring, keep elites and best rest
        # make sure all scores are present
        evaluated = [g for g in genomes if g.score is not None]
        if not evaluated:
            return

        # add elites from current pop (by score)
        elites = sorted([g for g in self._pop if g.score is not None],
                        key=lambda g: g.score, reverse=True)[: self.cfg.elite]

        combined = elites + evaluated
        combined = sorted(combined, key=lambda g: g.score if g.score is not None else -1e9, reverse=True)

        # truncate to population size
        self._pop = combined[: self.cfg.pop_size]
        self._gen += 1
        self._sigma *= float(self.cfg.sigma_decay)

    def state_dict(self) -> dict:
        return {
            "gen": self._gen,
            "sigma": self._sigma,
            "pop": [(g.vec, g.score) for g in self._pop],
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, state: dict) -> None:
        self._gen = int(state.get("gen", 0))
        self._sigma = float(state.get("sigma", self.cfg.sigma))
        pop_serial = state.get("pop", [])
        self._pop = [Genome(vec=np.array(v, dtype=np.float64), score=s) for (v, s) in pop_serial]


# simple evolutionary strategy (μ, λ)-ES with isotropic gaussian mutation around a moving mean
@dataclass
class ESConfig:
    mu: int = 8
    lam: int = 32
    sigma: float = 0.05
    sigma_decay: float = 1.0
    clip: Optional[float] = None


class ES(Strategy):
    def __init__(self, cfg: ESConfig) -> None:
        self.cfg = cfg
        self.dim: int = 0
        self.mean: np.ndarray | None = None
        self._sigma: float = float(cfg.sigma)
        self._gen: int = 0

    def initialize(self, dim: int, init_vec: np.ndarray | None = None) -> None:
        self.dim = int(dim)
        self._sigma = float(self.cfg.sigma)
        self._gen = 0
        if init_vec is None:
            init_vec = np.zeros(self.dim, dtype=np.float64)
        self.mean = init_vec.copy()

    def ask(self, n: int) -> List[Genome]:
        assert self.mean is not None, "ES must be initialized first"
        out: List[Genome] = []
        for _ in range(n):
            noise = np.random.normal(0.0, self._sigma, size=self.dim)
            vec = self.mean + noise
            if self.cfg.clip is not None:
                np.clip(vec, -self.cfg.clip, self.cfg.clip, out=vec)
            out.append(Genome(vec=vec))
        return out

    def tell(self, genomes: List[Genome]) -> None:
        assert self.mean is not None, "ES must be initialized first"
        # select top mu and update mean
        evaluated = [g for g in genomes if g.score is not None]
        if not evaluated:
            return
        evaluated.sort(key=lambda g: g.score, reverse=True)
        top = evaluated[: self.cfg.mu]
        self.mean = np.stack([g.vec for g in top], axis=0).mean(axis=0)
        self._gen += 1
        self._sigma *= float(self.cfg.sigma_decay)

    def state_dict(self) -> dict:
        return {
            "gen": self._gen,
            "sigma": self._sigma,
            "mean": None if self.mean is None else self.mean.copy(),
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, state: dict) -> None:
        self._gen = int(state.get("gen", 0))
        self._sigma = float(state.get("sigma", self.cfg.sigma))
        mean = state.get("mean", None)
        self.mean = None if mean is None else np.array(mean, dtype=np.float64)
