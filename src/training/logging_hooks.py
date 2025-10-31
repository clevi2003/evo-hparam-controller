from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import sys

from pathlib import Path
from src.core.hooks.hook_base import Hook

def _as_row_writer(writer_like: Any) -> Callable[[Dict[str, Any]], None]:
    """
    normalize IO writers so the hook just calls callable(row)
    supports:
      writer.write_row(dict)
      writer.write(dict)
      writer.append(dict)
      writer(row)
    """
    if callable(writer_like):
        return writer_like

    for name in ("log", "write_row", "write", "append"):
        fn = getattr(writer_like, name, None)
        if callable(fn):
            return fn

    raise TypeError(
        "Unsupported writer interface. Expected a callable(row) or an object with "
        "write_row(dict) / write(dict) / append(dict)."
    )


def _flush_if_possible(writer_like: Any) -> None:
    for name in ("flush", "close"):
        fn = getattr(writer_like, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

def _to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

class _TrainMetricsHook(Hook):
    """
    logs per batch training metrics every interval steps into the given writer
    expected keys in state on on_batch_end:
        epoch, global_step, batch_idx, loss, acc, lr, grad_norm, clip
    """
    def __init__(self, writer_like: Any, log_interval: int = 100) -> None:
        super().__init__() if hasattr(super(), "__init__") else None
        self._write_row = _as_row_writer(writer_like)
        self.log_interval = max(1, int(log_interval))

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        try:
            gs = int(state.get("global_step", 0))
            if gs % self.log_interval != 0:
                return
            row = {
                "event": "train_batch",
                "epoch": int(state.get("epoch", 0)),
                "global_step": gs,
                "batch_idx": int(state.get("batch_idx", -1)),
                "loss": _to_float(state.get("loss")),
                "acc": _to_float(state.get("acc")),
                "lr": _to_float(state.get("lr")),
                "grad_norm": _to_float(state.get("grad_norm")),
                "clip": int(bool(state.get("clip", False))),
            }
            self._write_row(row)
        except Exception:
            # never interrupt training for logging errors
            pass

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        # flush at epoch boundaries for better durability
        try:
            _flush_if_possible(self._write_row)
            row = {
                "event": "epoch_end",
                "nan_inf_flag": int(state.get('nan_inf_flag')),
                "lr": _to_float(state.get("lr")),
                "grad_norm": _to_float(state.get("grad_norm")),
                "acc": _to_float(state.get("acc")),
                "loss": _to_float(state.get("loss")),
                "t_epoch": _to_float(state.get("t_epoch")),
                "samples_per_s": _to_float(state.get("samples_per_s")),
                "train_loss_raw": _to_float(state.get("train_loss_raw")),
                "train_loss_ema": _to_float(state.get("train_loss_ema")),
            }

            self._write_row(row)
            print(f"End of epoch {row.get('epoch', '?')}: {row}")
            
        except Exception:
            pass

    def on_train_end(self, state: Dict[str, Any]) -> None:
        try:
            _flush_if_possible(self._write_row)
            row = {
                "event": "train_end",
                
                # end results
                "best_val_acc": _to_float(state.get("best_val_acc")),
                "final_train_loss": _to_float(state.get("loss")),
                "final_train_acc": _to_float(state.get("acc")),
                "lr": _to_float(state.get("lr")),
                "grad_norm": _to_float(state.get("grad_norm")),
                "train_loss_raw": _to_float(state.get("train_loss_raw")),
                "train_loss_ema": _to_float(state.get("train_loss_ema")),

                # throughput
                "t_train": _to_float(state.get("t_train")),
                "t_per_epoch": _to_float(state.get("epoch_avg_t")),
                "samples_per_s": _to_float(state.get("samples_per_s")),
                # <TODO: FLOPS>

                # stability
                "divergence_count": int(state.get("divergence_count", 0)),
                "early_stop_reasons": state.get("early_stop_reasons", None),
            }
            self._write_row(row)
            print(f"Final training stats: {row}")
        except Exception as e:
            try:
                print("TrainMetricsHook on_train_end error:", e, file=sys.stderr)
            except Exception:
                pass

    def close(self) -> None:
        try:
            _flush_if_possible(self._write_row)
        except Exception:
            pass


class _ValMetricsHook(Hook):
    """
    logs validation summary once per eval cycle (on_eval_end)
    expected keys in state at on_eval_end:
      epoch, global_step, val_loss, val_acc
    """
    def __init__(self, writer_like: Any) -> None:
        super().__init__() if hasattr(super(), "__init__") else None
        self._write_row = _as_row_writer(writer_like)

    def on_eval_end(self, state: Dict[str, Any]) -> None:
        try:
            row = {
                "event": "val_epoch",
                "epoch": int(state.get("epoch", 0)),
                "global_step": int(state.get("global_step", 0)),
                "val_loss": _to_float(state.get("val_loss")),
                "val_acc": _to_float(state.get("val_acc")),
                "best_val_acc": _to_float(state.get("best_val_acc")),
            }
            self._write_row(row)
        except Exception:
            pass

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        try:
            _flush_if_possible(self._write_row)
        except Exception:
            pass

    def close(self) -> None:
        try:
            _flush_if_possible(self._write_row)
        except Exception:
            pass


def make_train_metrics_hook(io_ctx, log_interval: int = 100) -> Hook:
    """
    io_ctx.writers should expose either: TODO make sure this is enforced
      .train with write_row(row)/write(row)/append(row)/callable(row), or
      be directly a writer-like itself
    """
    writer_like = getattr(getattr(io_ctx, "writers", io_ctx), "train", None)
    if writer_like is None:
        raise ValueError("io_ctx.writers.train is required for train metrics logging.")
    return _TrainMetricsHook(writer_like, log_interval=log_interval)


def make_val_metrics_hook(io_ctx) -> Hook:
    """
    io_ctx.writers must expose either:
      - .val with write_row(row)/write(row)/append(row)/callable(row), or
      - be directly a writer-like itself.
    """
    writer_like = getattr(getattr(io_ctx, "writers", io_ctx), "val", None)
    if writer_like is None:
        raise ValueError("io_ctx.writers.val is required for validation metrics logging.")
    return _ValMetricsHook(writer_like)

