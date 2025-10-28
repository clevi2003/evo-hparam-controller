from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import wandb  # I assume this is pip install wandb
from ..hook_base import Hook, State


class WandbHook(Hook):
    def __init__(self, project: str, run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 log_every_n_steps: int = 50,
                 artifacts_dir: Optional[str | Path] = None,
                 mode: Optional[str] = None  # "online"|"offline"
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.n = max(1, int(log_every_n_steps))
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.mode = mode

    def on_train_start(self, state: State) -> None:
        wandb.init(project=self.project, name=self.run_name, config=self.config, mode=self.mode)

    def on_batch_end(self, state: State) -> None:
        gs = int(state.get("global_step", 0))
        if gs % self.n != 0:
            return
        log = {}
        for k in ("loss", "acc", "lr", "grad_norm"):
            if k in state and state[k] is not None:
                log[k] = float(state[k])
        if log:
            wandb.log(log, step=gs, commit=False)

    def on_eval_end(self, state: State) -> None:
        gs = int(state.get("global_step", 0))
        log = {}
        for k in ("val_loss", "val_acc"):
            if k in state and state[k] is not None:
                log[k] = float(state[k])
        if log:
            wandb.log(log, step=gs, commit=True)

    def on_train_end(self, state: State) -> None:
        # Upload parquet logs and checkpoints as an artifact
        if self.artifacts_dir and self.artifacts_dir.exists():
            art = wandb.Artifact(name="run_artifacts", type="training-run")
            art.add_dir(str(self.artifacts_dir))
            wandb.log_artifact(art)
        wandb.finish()
