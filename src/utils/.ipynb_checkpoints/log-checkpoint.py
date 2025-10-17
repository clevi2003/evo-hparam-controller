import csv
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class CSVLogger:
    def __init__(self, csv_path: str):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self.fieldnames = [
            "epoch", "step", "split", "loss", "acc", "lr", "wall_time"
        ]
        write_header = not os.path.exists(csv_path)
        self.f = open(csv_path, "a", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        if write_header:
            self.writer.writeheader()
        self.start_time = datetime.now().timestamp()

    def log(self, **kwargs):
        now = datetime.now().timestamp()
        row = {"wall_time": now - self.start_time}
        row.update(kwargs)
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()

class TBLogger:
    def __init__(self, log_dir: str, exp_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, exp_name))

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
