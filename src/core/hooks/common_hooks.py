from typing import Callable, Optional
from .hook_base import Hook, State

class EveryNSteps(Hook):
    """
    Wrap another Hook and only forward on_batch_* when global_step % n == 0.

    This is useful for expensive logic (controller decisions, heavy logging).
    Other lifecycle events are always forwarded.
    """

    def __init__(self, wrapped: Hook, n: int) -> None:
        if n <= 0:
            raise ValueError("EveryNSteps requires n > 0.")
        self.wrapped = wrapped
        self.n = n

    def on_train_start(self, state: State) -> None:
        self.wrapped.on_train_start(state)

    def on_epoch_start(self, state: State) -> None:
        self.wrapped.on_epoch_start(state)

    def on_batch_start(self, state: State) -> None:
        gs = int(state.get("global_step", 0))
        if gs % self.n == 0:
            self.wrapped.on_batch_start(state)

    def on_batch_end(self, state: State) -> None:
        gs = int(state.get("global_step", 0))
        if gs % self.n == 0:
            self.wrapped.on_batch_end(state)

    def on_eval_start(self, state: State) -> None:
        self.wrapped.on_eval_start(state)

    def on_eval_end(self, state: State) -> None:
        self.wrapped.on_eval_end(state)

    def on_train_end(self, state: State) -> None:
        try:
            self.wrapped.on_train_end(state)
        finally:
            self.close()

    def close(self) -> None:
        self.wrapped.close()


class WithCooldown(Hook):
    """
    Wrap another Hook and only allow on_batch_end forwarding after a cooldown.

    Good for throttling controller decisions. The cooldown counter decreases
    on every batch and resets when a forwarded call occurs.
    """

    def __init__(self, wrapped: Hook, cooldown_steps: int) -> None:
        if cooldown_steps < 0:
            raise ValueError("WithCooldown requires cooldown_steps >= 0.")
        self.wrapped = wrapped
        self.cooldown_steps = cooldown_steps
        self._cooldown_left = 0

    @property
    def cooldown_left(self) -> int:
        return self._cooldown_left

    def on_train_start(self, state: State) -> None:
        self._cooldown_left = 0
        self.wrapped.on_train_start(state)

    def on_epoch_start(self, state: State) -> None:
        self.wrapped.on_epoch_start(state)

    def on_batch_start(self, state: State) -> None:
        # Always forward, gating is on_batch_end to align with most decision points
        self.wrapped.on_batch_start(state)

    def on_batch_end(self, state: State) -> None:
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return
        self.wrapped.on_batch_end(state)
        self._cooldown_left = self.cooldown_steps

    def on_eval_start(self, state: State) -> None:
        self.wrapped.on_eval_start(state)

    def on_eval_end(self, state: State) -> None:
        self.wrapped.on_eval_end(state)

    def on_train_end(self, state: State) -> None:
        try:
            self.wrapped.on_train_end(state)
        finally:
            self.close()

    def close(self) -> None:
        self.wrapped.close()


class LambdaHook(Hook):
    """
    Quick in-line hook from callables. Provide any subset of callbacks
    """

    def __init__(
        self,
        on_train_start: Optional[Callable[[State], None]] = None,
        on_epoch_start: Optional[Callable[[State], None]] = None,
        on_batch_start: Optional[Callable[[State], None]] = None,
        on_batch_end: Optional[Callable[[State], None]] = None,
        on_eval_start: Optional[Callable[[State], None]] = None,
        on_eval_end: Optional[Callable[[State], None]] = None,
        on_train_end: Optional[Callable[[State], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
        on_after_backward: Optional[Callable[[], None]] = None,
        on_before_optimizer_step: Optional[Callable[[], None]] = None,
        on_after_optimizer_step: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_train_start = on_train_start
        self._on_epoch_start = on_epoch_start
        self._on_batch_start = on_batch_start
        self._on_batch_end = on_batch_end
        self._on_eval_start = on_eval_start
        self._on_eval_end = on_eval_end
        self._on_train_end = on_train_end
        self._on_close = on_close
        self._on_after_backward = on_after_backward
        self._on_before_optimizer_step = on_before_optimizer_step
        self._on_after_optimizer_step = on_after_optimizer_step

    def on_train_start(self, state: State) -> None:
        if self._on_train_start:
            self._on_train_start(state)

    def on_epoch_start(self, state: State) -> None:
        if self._on_epoch_start:
            self._on_epoch_start(state)

    def on_batch_start(self, state: State) -> None:
        if self._on_batch_start:
            self._on_batch_start(state)

    def on_batch_end(self, state: State) -> None:
        if self._on_batch_end:
            self._on_batch_end(state)

    def on_eval_start(self, state: State) -> None:
        if self._on_eval_start:
            self._on_eval_start(state)

    def on_eval_end(self, state: State) -> None:
        if self._on_eval_end:
            self._on_eval_end(state)

    def on_train_end(self, state: State) -> None:
        if self._on_train_end:
            self._on_train_end(state)

    def on_after_backward(self, state: State) -> None:
        if self._on_after_backward:
            self._on_after_backward(state)

    def on_before_optimizer_step(self, state: State) -> None:
        if self._on_before_optimizer_step:
            self._on_before_optimizer_step(state)

    def on_after_optimizer_step(self, state: State) -> None:
        if self._on_after_optimizer_step:
            self._on_after_optimizer_step(state)

    def close(self) -> None:
        if self._on_close:
            self._on_close()
