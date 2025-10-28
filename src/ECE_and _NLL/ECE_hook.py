# Kevin Liang
from __future__ import annotations
from typing import Optional
from src.core.hooks.hook_base import Hook
from src.ECE_and_NLL.ECE_calculation import compute_ece

class ECEHook(Hook):
    """
    Compute ECE at validation time and place it (and NLL) into state['metrics'].
    Runs on on_eval_end so it does one extra forward pass over val_loader.
    """
    def __init__(self, val_loader, device, n_bins: int = 15, write_nll: bool = True):
        self.val_loader = val_loader
        self.device = device
        self.n_bins = n_bins
        self.write_nll = write_nll

    def on_eval_end(self, state):
        # compute ECE
        ece = compute_ece(
            model=state["model"],
            device=self.device,
            val_loader=self.val_loader,
            n_bins=self.n_bins,
        )

        #metrics dictionary
        m = state.get("metrics", {}) or {}
        # val_loss is CE -> equals NLL
        if self.write_nll:
            m["nll"] = float(state.get("val_loss") or 0.0)
        m["ece"] = float(ece)

        # also mirror val_acc/val_loss for completeness (useful to have together)
        if "val_acc" not in m:
            m["val_acc"] = float(state.get("val_acc") or 0.0)
        if "val_loss" not in m:
            m["val_loss"] = float(state.get("val_loss") or 0.0)

        state["metrics"] = m
