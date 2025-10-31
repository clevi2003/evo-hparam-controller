from typing import Iterable, List, Optional
from .hook_base import Hook, State

class HookList(Hook):
    """
    Composite that forwards every event to each child hook in order.
    If swallow_exceptions, each hook's exception is caught and stored,
    allowing the rest to keep running. Exceptions are raised again at the end of
    the event call if any occurred, with all messages concatenated.
    """

    def __init__(self, hooks: Optional[Iterable[Hook]] = None, swallow_exceptions: bool = False) -> None:
        self._hooks: List[Hook] = list(hooks) if hooks is not None else []
        self._swallow = swallow_exceptions
        self._errors: List[str] = []

    # Management of the hook list
    def add(self, hook: Hook) -> None:
        self._hooks.append(hook)

    def extend(self, hooks: Iterable[Hook]) -> None:
        self._hooks.extend(hooks)

    def _call_each(self, method_name: str, state: State) -> None:
        had_error = False
        messages: List[str] = []
        for h in self._hooks:
            try:
                getattr(h, method_name)(state)
            except Exception as e:
                if self._swallow:
                    had_error = True
                    messages.append(f"{h.__class__.__name__}.{method_name} error: {e}")
                else:
                    # let it propagate immediately
                    raise
        if had_error:
            msg = " | ".join(messages)
            self._errors.append(msg)
            # Raise again at the end to signal failure to the caller, but after all hooks ran
            raise RuntimeError(f"HookList encountered errors during '{method_name}': {msg}")

    # Lifecycle forwarding
    def on_train_start(self, state: State) -> None:
        self._call_each("on_train_start", state)

    def on_epoch_start(self, state: State) -> None:
        self._call_each("on_epoch_start", state)

    def on_epoch_end(self, state: State) -> None:
        self._call_each("on_epoch_end", state)

    def on_batch_start(self, state: State) -> None:
        self._call_each("on_batch_start", state)

    def on_batch_end(self, state: State) -> None:
        self._call_each("on_batch_end", state)

    def on_after_backward(self, state: State) -> None:
        self._call_each("on_after_backward", state)

    def on_before_optimizer_step(self, state: State) -> None:
        self._call_each("on_before_optimizer_step", state)

    def on_after_optimizer_step(self, state: State) -> None:
        self._call_each("on_after_optimizer_step", state)

    def on_eval_start(self, state: State) -> None:
        self._call_each("on_eval_start", state)

    def on_eval_end(self, state: State) -> None:
        self._call_each("on_eval_end", state)

    def on_train_end(self, state: State) -> None:
        try:
            self._call_each("on_train_end", state)
        finally:
            # Ensure close is attempted even if on_train_end fails
            self.close()

    def close(self) -> None:
        errors: List[str] = []
        for h in self._hooks:
            try:
                h.close()
            except Exception as e:
                errors.append(f"{h.__class__.__name__}.close error: {e}")
        if errors:
            joined = " | ".join(errors)
            self._errors.append(joined)
            raise RuntimeError(f"HookList encountered errors during 'close': {joined}")

    @property
    def errors(self) -> List[str]:
        """Collected error messages (when swallow_exceptions=True)."""
        return list(self._errors)