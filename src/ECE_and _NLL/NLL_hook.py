from __future__ import annotations
from src.core.hooks.hook_base import Hook

class NLLHook(Hook):
    """
    Mirror val_loss into state['metrics']['nll'] at validation end.
    Use this when val_loss is cross-entropy loss.
    """
    def on_eval_end(self, state):
        m = state.get("metrics", {}) or {}
        m["nll"] = float(state.get("val_loss") or 0.0)
        # keep core vals together for convenience
        m.setdefault("val_loss", float(state.get("val_loss") or 0.0))
        m.setdefault("val_acc",  float(state.get("val_acc")  or 0.0))
        state["metrics"] = m