from typing import Any, Dict, Iterable, List, Sequence, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from .logging_json_csv import CSVAppender, JSONLAppender
from .logging_parquet import ParquetAppender, PartitionedParquetAppender
from .logging_base import BaseAppender, _now_iso
import pyarrow as pa


class Logger:
    """Thin facade around BaseAppender."""
    def __init__(self, appender: BaseAppender) -> None:
        self.appender = appender

    def log(self, row: Dict[str, Any]) -> None:
        self.appender.append(row)

    def log_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.appender.append_many(rows)

    def flush(self) -> None:
        self.appender.flush()

    def close(self) -> None:
        self.appender.close()

    def __enter__(self):
        self.appender.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.appender.__exit__(exc_type, exc, tb)


class MultiLogger(Logger):
    """Fan-out logger to multiple sinks (Parquet and JSONL)."""
    def __init__(self, loggers: Sequence[Logger]) -> None:
        self._loggers = list(loggers)

    def log(self, row: Dict[str, Any]) -> None:
        for lg in self._loggers:
            lg.log(row)

    def log_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        for lg in self._loggers:
            lg.log_many(rows)

    def flush(self) -> None:
        for lg in self._loggers:
            lg.flush()

    def close(self) -> None:
        errs: List[str] = []
        for lg in self._loggers:
            try:
                lg.close()
            except Exception as e:  # pragma: no cover
                errs.append(f"{lg.__class__.__name__}.close error: {e}")
        if errs:
            raise RuntimeError(" | ".join(errs))


DEFAULT_CONTROLLER_TICK_FIELDS: List[str] = [
    # identity
    "run_id", "seed", "arch", "dataset",
    # time
    "epoch", "global_step", "step_in_epoch", "timestamp",
    # inputs as seen by controller
    "train_loss", "val_loss", "train_acc", "val_acc",
    "ema_loss", "ema_acc", "d_loss", "d_acc",
    "lr_before", "grad_norm", "update_ratio", "clip_events",
    # action
    "delta_log_lr_raw", "delta_log_lr_applied",
    "lr_after", "applied", "cooldown_left",
    # outcomes (still need to decide on this)
    "val_acc_post", "val_loss_post",
    # stability flags
    "nan_flag", "overflow_flag",
]

# Arrow schema for strong typing & efficient compression
CONTROLLER_TICK_SCHEMA = pa.schema([
    pa.field("run_id", pa.string()),
    pa.field("seed", pa.int64()),
    pa.field("arch", pa.string()),
    pa.field("dataset", pa.string()),

    pa.field("epoch", pa.int64()),
    pa.field("global_step", pa.int64()),
    pa.field("step_in_epoch", pa.int64()),
    pa.field("timestamp", pa.string()),

    pa.field("train_loss", pa.float32()),
    pa.field("val_loss", pa.float32()),
    pa.field("train_acc", pa.float32()),
    pa.field("val_acc", pa.float32()),
    pa.field("ema_loss", pa.float32()),
    pa.field("ema_acc", pa.float32()),
    pa.field("d_loss", pa.float32()),
    pa.field("d_acc", pa.float32()),
    pa.field("lr_before", pa.float32()),
    pa.field("grad_norm", pa.float32()),
    pa.field("update_ratio", pa.float32()),
    pa.field("clip_events", pa.int32()),

    pa.field("delta_log_lr_raw", pa.float32()),
    pa.field("delta_log_lr_applied", pa.float32()),
    pa.field("lr_after", pa.float32()),
    pa.field("applied", pa.bool_()),
    pa.field("cooldown_left", pa.int32()),

    pa.field("val_acc_post", pa.float32()),
    pa.field("val_loss_post", pa.float32()),

    pa.field("nan_flag", pa.bool_()),
    pa.field("overflow_flag", pa.bool_()),
])


@dataclass
class ControllerTickLogger:
    """
    Schema-aware logger for controller decision ticks

    Default behavior:
      Writes a single Parquet file with a strong schema
      Adds ISO timestamp automatically if missing
      Can switch to partitioned mode for very long runs
    """
    appender: BaseAppender
    fields: List[str] = None

    def __post_init__(self):
        if self.fields is None:
            self.fields = list(DEFAULT_CONTROLLER_TICK_FIELDS)

    @classmethod
    def to_parquet(cls, path: Union[str, Path], schema: pa.Schema = CONTROLLER_TICK_SCHEMA,
                   buffer_rows: int = 2048) -> "ControllerTickLogger":
        app = ParquetAppender(path, schema=schema, buffer_rows=buffer_rows)
        return cls(appender=app, fields=list(schema.names))

    @classmethod
    def to_parquet_dir(cls, directory: Union[str, Path], schema: pa.Schema = CONTROLLER_TICK_SCHEMA,
                       rows_per_file: int = 100_000, buffer_rows: int = 4096) -> "ControllerTickLogger":
        app = PartitionedParquetAppender(directory, schema=schema, rows_per_file=rows_per_file, buffer_rows=buffer_rows)
        return cls(appender=app, fields=list(schema.names))

    @classmethod
    def to_csv(cls, path: Union[str, Path], fields: Optional[Sequence[str]] = None) -> "ControllerTickLogger":
        f = list(fields) if fields is not None else list(DEFAULT_CONTROLLER_TICK_FIELDS)
        return cls(appender=CSVAppender(path, f), fields=f)

    @classmethod
    def to_jsonl(cls, path: Union[str, Path], fields: Optional[Sequence[str]] = None) -> "ControllerTickLogger":
        f = list(fields) if fields is not None else list(DEFAULT_CONTROLLER_TICK_FIELDS)
        return cls(appender=JSONLAppender(path, keep_fields=f), fields=f)

    def log_tick(self, row: Dict[str, Any]) -> None:
        if "timestamp" not in row:
            row = {**row, "timestamp": _now_iso()}
        self.appender.append(row)

    def log_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        def ensure_ts(r: Dict[str, Any]) -> Dict[str, Any]:
            return r if "timestamp" in r else {**r, "timestamp": _now_iso()}
        self.appender.append_many(ensure_ts(r) for r in rows)

    def flush(self) -> None:
        self.appender.flush()

    def close(self) -> None:
        self.appender.close()

TRAIN_SCHEMA = pa.schema([
    pa.field("global_step", pa.int64()),
    pa.field("epoch", pa.int64()),
    pa.field("loss", pa.float32()),
    pa.field("acc", pa.float32()),
    pa.field("lr", pa.float32()),
    pa.field("grad_norm", pa.float32()),
])

VAL_SCHEMA = pa.schema([
    pa.field("global_step", pa.int64()),
    pa.field("epoch", pa.int64()),
    pa.field("val_loss", pa.float32()),
    pa.field("val_acc", pa.float32()),
])


def make_train_parquet_logger(path: Union[str, Path], buffer_rows: int = 4096) -> Logger:
    return Logger(ParquetAppender(path, schema=TRAIN_SCHEMA, buffer_rows=buffer_rows))


def make_val_parquet_logger(path: Union[str, Path], buffer_rows: int = 2048) -> Logger:
    return Logger(ParquetAppender(path, schema=VAL_SCHEMA, buffer_rows=buffer_rows))


def make_train_csv_logger(path: Union[str, Path]) -> Logger:
    return Logger(CSVAppender(path, TRAIN_SCHEMA.names))


def make_val_csv_logger(path: Union[str, Path]) -> Logger:
    return Logger(CSVAppender(path, VAL_SCHEMA.names))