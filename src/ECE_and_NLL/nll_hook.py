from __future__ import annotations
from src.core.hooks.hook_base import Hook

class NLLHook(Hook):
    """
    Mirror val_loss into state['metrics'][key] at validation end.
    Assumes validation criterion is cross-entropy (so val_loss == average NLL).
    """
    def __init__(self, key: str = "nll") -> None:
        self.key = key

    def on_eval_end(self, state):
        val_loss = state.get("val_loss", None)
        if val_loss is None:
            return  # nothing to mirror

        m = state.get("metrics", {}) or {}
        # don't clobber if something else already set it (e.g., another hook)
        m.setdefault(self.key, float(val_loss))
        # keep core vals together for convenience (only set if missing)
        m.setdefault("val_loss", float(val_loss))
        if state.get("val_acc") is not None:
            m.setdefault("val_acc", float(state["val_acc"]))
        state["metrics"] = m
