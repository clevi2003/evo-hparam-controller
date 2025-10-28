from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from src.core.hooks.hook_base import Hook, State


@dataclass
class UncertaintyConfig:
    """
    config for UncertaintyHook

    attributes:
        logits_key: State key that holds the last batch logits tensor (N, C, ...)
        targets_key: Optional state key for integer class labels (N,)
        temperature: Softmax temperature. >1 smooths, <1 sharpens. 1.0 = identity
        require_targets: If True, skip computation when labels are missing
        reduce: "mean" (default) or "none" for per-sample (kept internal; hook writes only scalars)
    """
    logits_key: str = "logits"
    targets_key: Optional[str] = "targets"
    temperature: float = 1.0
    require_targets: bool = False
    reduce: str = "mean"


class UncertaintyHook(Hook):
    """
    Calcs classification uncertainty signals from logits:
      entropy: mean categorical entropy over the batch
      margin: mean (p_top1 - p_top2) over the batch

    writes to state:
      state["entropy"] is float
      state["margin"]  is float

    passes when the expected tensors are missing in state
    """

    def __init__(self, cfg: Optional[UncertaintyConfig] = None) -> None:
        self.cfg = cfg or UncertaintyConfig()
        if self.cfg.reduce not in ("mean", "none"):
            raise ValueError("UncertaintyConfig.reduce must be 'mean' or 'none'.")

    @torch.no_grad()
    def on_batch_end(self, state: State) -> None:
        logits = state.get(self.cfg.logits_key, None)
        if logits is None or not isinstance(logits, torch.Tensor):
            return

        # optional labels
        targets = None
        if self.cfg.targets_key:
            tt = state.get(self.cfg.targets_key, None)
            if isinstance(tt, torch.Tensor):
                targets = tt

        if self.cfg.require_targets and targets is None:
            return

        entropy, margin = _entropy_and_margin(
            logits=logits,
            temperature=self.cfg.temperature,
            reduce=self.cfg.reduce,
        )

        # store scalars for downstream logging/analysis
        state["entropy"] = float(entropy)
        state["margin"] = float(margin)

@torch.no_grad()
def _entropy_and_margin(
    logits: torch.Tensor,
    temperature: float = 1.0,
    reduce: str = "mean",
) -> Tuple[float, float]:
    """
    calc mean categorical entropy and mean margin from logits

    args:
        logits: Tensor of shape (N, C, ...) where classes are along dim=1
        temperature: Softmax temperature
        reduce: "mean" or "none" (returns scalars; none is kept for future extension)

    returns:
        (entropy_mean, margin_mean) as floats
    """
    # collapse extra dims after class dim to (N, C)
    if logits.dim() < 2:
        # can't calc so return zeros
        return 0.0, 0.0

    N = logits.shape[0]
    C = logits.shape[1]
    x = logits

    if x.dim() > 2:
        x = x.view(N, C, -1).mean(dim=-1)  # spatial/global average for CNN outputs

    if temperature != 1.0:
        x = x / float(temperature)

    # stable softmax via logsumexp
    x = x.float()
    x = x - x.max(dim=1, keepdim=True).values
    expx = torch.exp(x)
    Z = expx.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p = expx / Z  # (N, C)

    # entropy: -sum p log p
    plogp = p.clamp_min(1e-12).log().mul(p) # p * log p
    entropy = (-plogp.sum(dim=1))  # (N,)

    # margin: p_top1 - p_top2
    top2 = torch.topk(p, k=2, dim=1).values # (N, 2)
    margin = top2[:, 0] - top2[:, 1] # (N,)

    if reduce == "mean":
        e_mean = float(entropy.mean().item())
        m_mean = float(margin.mean().item())
        return e_mean, m_mean

    # none branch for potential future per-sample logging but still return means to match the hook setup
    e_mean = float(entropy.mean().item())
    m_mean = float(margin.mean().item())
    return e_mean, m_mean
