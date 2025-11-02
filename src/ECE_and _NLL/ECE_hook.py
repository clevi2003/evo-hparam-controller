# Kevin Liang
from __future__ import annotations
import torch
import torch.nn.functional as F
from src.core.hooks.hook_base import Hook

class _ECEAccumulator:
    """
    Online ECE: track per-bin counts/sums during validation.
    No logits are stored; the engine feeds (logits, targets) each val batch.
    """
    def __init__(self, n_bins: int = 15, device: torch.device | str = "cpu"):
        self.n_bins = int(n_bins)
        self.device = torch.device(device)
        self.edges = torch.linspace(0, 1, steps=self.n_bins + 1, device=self.device)
        self.counts   = torch.zeros(self.n_bins, dtype=torch.long,    device=self.device)
        self.sum_conf = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)
        self.sum_corr = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        corr = (preds == targets).to(torch.float32)

        # Bin by confidence: (0,e1], (e1,e2], ..., (e_{B-1},1]
        idx = torch.bucketize(conf.clamp_(0, 1 - 1e-8), self.edges, right=True) - 1  # 0..B-1
        for b in range(self.n_bins):
            m = (idx == b)
            cnt = int(m.sum().item())
            if cnt:
                self.counts[b]   += cnt
                self.sum_conf[b] += conf[m].sum()
                self.sum_corr[b] += corr[m].sum()

    @torch.no_grad()
    def result(self) -> float:
        N = int(self.counts.sum().item())
        if N == 0:
            return 0.0
        ece = 0.0
        for b in range(self.n_bins):
            cnt = int(self.counts[b].item())
            if not cnt:
                continue
            acc_b  = (self.sum_corr[b] / cnt).item()
            conf_b = (self.sum_conf[b] / cnt).item()
            ece += (cnt / N) * abs(acc_b - conf_b)
        return float(ece)


class ECEHook(Hook):
    """
    Compute ECE at validation time without a second pass.
    Writes 'ece' (and optional 'nll') into state['metrics'] on on_eval_end.
    """
    def __init__(self, device, n_bins: int = 15, write_nll: bool = True):
        self.device = device
        self.n_bins = n_bins
        self.write_nll = write_nll

    def on_eval_start(self, state):
        # Install the accumulator so the engine can feed it logits/targets each val batch.
        state["_ece_acc"] = _ECEAccumulator(n_bins=self.n_bins, device=self.device)

    def on_eval_end(self, state):
        acc = state.get("_ece_acc", None)
        ece = float(acc.result()) if acc is not None else 0.0

        m = state.get("metrics", {}) or {}
        if self.write_nll:
            # Cross-entropy val_loss equals NLL for hard labels.
            m["nll"] = float(state.get("val_loss") or 0.0)
        m["ece"] = ece

        # Keep val_acc/val_loss together in the same dict for downstream logging
        m.setdefault("val_acc",  float(state.get("val_acc")  or 0.0))
        m.setdefault("val_loss", float(state.get("val_loss") or 0.0))
        state["metrics"] = m

        # cleanup
        state.pop("_ece_acc", None)
