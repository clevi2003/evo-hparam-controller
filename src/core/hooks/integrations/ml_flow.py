from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow  # I assume this is pip install mlflow

from ..hook_base import State, Hook

class MLflowHook(Hook):
    """
    Mirrors key metrics to MLflow and logs artifacts on train_end.
    """
    def __init__(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None,
                 log_every_n_steps: int = 50, artifacts_dir: Optional[str | Path] = None) -> None:
        self.run_name = run_name
        self.tags = tags or {}
        self.n = max(1, int(log_every_n_steps))
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None

    def on_train_start(self, state: State) -> None:
        mlflow.start_run(run_name=self.run_name)
        if self.tags:
            mlflow.set_tags(self.tags)
        train_cfg = state.get("cfg")
        if train_cfg is not None:
            mlflow.log_params({
                "epochs": getattr(train_cfg, "epochs", None),
                "optimizer": getattr(getattr(train_cfg, "optim", None), "name", None),
                "lr": getattr(getattr(train_cfg, "optim", None), "lr", None),
            })

    def on_batch_end(self, state: State) -> None:
        gs = int(state.get("global_step", 0))
        if gs % self.n != 0:
            return
        metrics = {}
        for k in ("loss", "acc", "lr", "grad_norm"):
            if k in state and state[k] is not None:
                metrics[k] = float(state[k])
        if metrics:
            mlflow.log_metrics(metrics, step=gs)

    def on_eval_end(self, state: State) -> None:
        gs = int(state.get("global_step", 0))
        for k in ("val_loss", "val_acc"):
            if k in state and state[k] is not None:
                mlflow.log_metric(k, float(state[k]), step=gs)

    def on_train_end(self, state: State) -> None:
        # Upload artifacts (parquet logs, checkpoints)
        if self.artifacts_dir and self.artifacts_dir.exists():
            mlflow.log_artifacts(str(self.artifacts_dir))
        mlflow.end_run()
