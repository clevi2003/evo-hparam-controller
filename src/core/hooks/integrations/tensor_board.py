from __future__ import annotations
from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from ..hook_base import Hook, State

class TensorBoardHook(Hook):
    def __init__(self, log_dir: str | Path, log_every_n_steps: int = 50) -> None:
        self.log_dir = Path(log_dir)
        self.n = max(1, int(log_every_n_steps))
        self.writer: Optional[SummaryWriter] = None

    def on_train_start(self, state: State) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def on_batch_end(self, state: State) -> None:
        if self.writer is None:
            return
        gs = int(state.get("global_step", 0))
        if gs % self.n != 0:
            return
        if (v := state.get("loss")) is not None: self.writer.add_scalar("train/loss", v, gs)
        if (v := state.get("acc")) is not None:  self.writer.add_scalar("train/acc", v, gs)
        if (v := state.get("lr")) is not None:   self.writer.add_scalar("train/lr", v, gs)
        if (v := state.get("grad_norm")) is not None: self.writer.add_scalar("train/grad_norm", v, gs)

    def on_eval_end(self, state: State) -> None:
        if self.writer is None:
            return
        gs = int(state.get("global_step", 0))
        if (v := state.get("val_loss")) is not None: self.writer.add_scalar("val/loss", v, gs)
        if (v := state.get("val_acc")) is not None:  self.writer.add_scalar("val/acc", v, gs)

    def on_train_end(self, state: State) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
