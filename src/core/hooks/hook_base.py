from __future__ import annotations

from typing import Any, Dict

State = Dict[str, Any]

def get_global_step(state: State) -> int:
    """Convenience accessor with a stable default."""
    return int(state.get("global_step", 0))


def get_epoch(state: State) -> int:
    """Convenience accessor with a stable default."""
    return int(state.get("epoch", 0))

class Hook:
    """
    Base training/eval hook with optional callbacks.
    Subclass and override only what you need.

    The `state` dictionary is the single source of truth for all runtime signals.
    Common keys we expect the trainer to populate (not enforced here):
      - "epoch": int
      - "global_step": int
      - "batch_idx": int
      - "optimizer": torch.optim.Optimizer
      - "scheduler": torch.optim.lr_scheduler._LRScheduler | None
      - "loss": float
      - "train_acc": float | None
      - "val_acc": float | None
      - "device": torch.device
      - "cfg": dict or a dataclass config
    """

    # Lifecycle
    def on_train_start(self, state: State) -> None:
        pass

    def on_epoch_start(self, state: State) -> None:
        pass

    def on_epoch_end(self, state: State) -> None:
        pass

    def on_batch_start(self, state: State) -> None:
        pass

    def on_batch_end(self, state: State) -> None:
        pass

    def on_eval_start(self, state: State) -> None:
        pass

    def on_eval_end(self, state: State) -> None:
        pass

    def on_train_end(self, state: State) -> None:
        pass

    def on_after_backward(self, state: State) -> None:
        pass

    def on_before_optimizer_step(self, state: State) -> None:
        pass

    def on_after_optimizer_step(self, state: State) -> None:
        pass

    # optional cleanup for resources (files, writers, etc.) might be needed
    def close(self) -> None:
        pass

class NullHook(Hook):
    """A do-nothing hook (for placeholding)."""
    pass