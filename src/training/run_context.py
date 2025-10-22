from __future__ import annotations
from typing import Any, Dict, Optional
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import uuid
from src.core.logging.logging_json_csv import JSONLAppender
from src.core.logging.loggers import Logger


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return ""

def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    except Exception:
        return ""

class RunContext:
    """
    Encapsulate run directory and metadata writing for training/eval runs.

    Responsibilities:
      - Create timestamped run dir (optionally under a provided run_root)
      - Persist env.json, config.json and run_meta.json
      - Provide finalize(end_status) to write end_time and status
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        run_id = os.environ.get("RUN_ID") or uuid.uuid4().hex[:8]
        self.run_id = run_id
        # store start as a timezone-aware datetime (UTC) for accurate duration calculation
        self._start_dt = datetime.now(timezone.utc)
        start_ts = self.get_time(self._start_dt)

        if isinstance(self.cfg, dict) and self.cfg.get("run_dir"):
            run_root = Path(self.cfg.get("run_dir"))
            self.run_dir = run_root / f"{start_ts}_{run_id}"
        else:
            self.run_dir = Path(os.getcwd()) / "runs" / f"{start_ts}_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_meta = {
            "run_id": run_id,
            "start_time": start_ts,
            "end_time": None,
            "duration_seconds": None,
            "git_commit": _git_commit(),
            "git_branch": _git_branch(),
            "status": "running",
        }

        self.config = None
        self.env = None

    def get_time(self, dt: Optional[datetime] = None) -> str:
        """Return an ISO-like UTC timestamp with microsecond precision.

        If dt is provided, it should be a timezone-aware datetime. If not,
        the current UTC time will be used.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        # normalize to UTC and format with Z suffix
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y%m%dT%H%M%S.%fZ")

    def set_env(self, env):
        try:
            self.env = dict(env) if isinstance(env, dict) else env
        except Exception:
            self.env = env

    def set_config(self,config):
        try:
            self.config = self._to_jsonable_config(config)
        except Exception:
            self.config = str(config)

    def build_env(self, torch_mod, mixed_precision: bool) -> Dict[str, Any]:
        """
        Build a standard env payload from the torch module and mixed_precision flag.

        Stores the payload via set_env and returns it.
        """
        try:
            env = {
                "pytorch_version": getattr(torch_mod, "__version__", ""),
                "cuda_version": getattr(getattr(torch_mod, "version", None), "cuda", None) if hasattr(torch_mod, "version") else None,
                "cudnn_version": torch_mod.backends.cudnn.version() if hasattr(torch_mod, "backends") and hasattr(torch_mod.backends, "cudnn") else None,
                "gpu_model": torch_mod.cuda.get_device_name(0) if getattr(torch_mod, "cuda", None) is not None and torch_mod.cuda.is_available() and torch_mod.cuda.device_count() > 0 else None,
                "gpu_count": torch_mod.cuda.device_count() if getattr(torch_mod, "cuda", None) is not None and torch_mod.cuda.is_available() else 0,
                "mixed_precision": bool(mixed_precision),
            }
        except Exception:
            env = {"mixed_precision": bool(mixed_precision)}
        self.set_env(env)
        return env



    def build_config(self, model, optimizer, scheduler, train_loader, cfg, epochs: int):
        """Build a standard config payload and store it via set_config.

        Returns the payload.
        """
        try:
            batch_size = getattr(train_loader, "batch_size", None)
            dataset_name = getattr(getattr(train_loader, 'dataset', None), '__class__', None)
            dataset_name = dataset_name.__name__ if dataset_name is not None else None

            payload = {
                "dataset": dataset_name,
                "model": model.__class__.__name__ if hasattr(model, "__class__") else str(type(model)),
                "optimizer": self._summarize_optimizer(optimizer) if optimizer is not None else None,
                "scheduler": self._summarize_scheduler(scheduler) if scheduler is not None else None,
                "controller": (cfg.get("controller") if isinstance(cfg, dict) else None),
                "batch_size": int(batch_size) if batch_size is not None else None,
                "epochs": int(epochs),
                "augmentation": (cfg.get("data", {}).get("augment") if isinstance(cfg, dict) else None),
                "seeds": (cfg.get("seeds") if isinstance(cfg, dict) and "seeds" in cfg else cfg.get("seed") if isinstance(cfg, dict) else None),
                "full_config": self._to_jsonable_config(cfg),
            }
        except Exception:
            payload = {"full_config": str(cfg)}
        self.set_config(payload)
        return payload

    def log_exception(self, e):
        try:
            path = self.run_dir / "exceptions.txt"
            with open(path, "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()}: {str(e)}\n")
        except Exception:
            pass

    def write_meta(self) -> None:
        try:
            p = self.run_dir / "run_meta.jsonl"
            lg = Logger(JSONLAppender(p, keep_fields=None))
            lg.log(self.run_meta)
            lg.flush()
            lg.close()
        except Exception:
            pass

    def write_config(self) -> None:
        try:
            p = self.run_dir / "config.jsonl"
            lg = Logger(JSONLAppender(p, keep_fields=None))
            lg.log(self.config)
            lg.flush()
            lg.close()
        except Exception:
            pass

    def write_env(self) -> None:
        try:
            p = self.run_dir / "env.jsonl"
            lg = Logger(JSONLAppender(p, keep_fields=None))
            lg.log(self.env)
            lg.flush()
            lg.close()
        except Exception:
            pass

    def finalize(self, status: str = "finished") -> None:
        try:
            end_dt = datetime.now(timezone.utc)
            end_ts = self.get_time(end_dt)
            self.run_meta["end_time"] = end_ts
            # compute duration_seconds using timezone-aware datetimes
            try:
                if isinstance(self._start_dt, datetime):
                    # ensure both are timezone-aware
                    start = self._start_dt if self._start_dt.tzinfo is not None else self._start_dt.replace(tzinfo=timezone.utc)
                    duration = (end_dt - start).total_seconds()
                else:
                    duration = None
            except Exception:
                duration = None
            self.run_meta["duration_seconds"] = duration
            self.run_meta["status"] = status
            self.write_meta()
            self.write_config()
            self.write_env()
        except Exception:
            pass

    @staticmethod
    def _summarize_optimizer(opt):
        try:
            return {"class": opt.__class__.__name__, "param_groups": [{k: v for k, v in pg.items() if k != "params"} for pg in opt.param_groups]}
        except Exception:
            return {"class": str(type(opt))}

    @staticmethod
    def _summarize_scheduler(sched):
        try:
            return {"class": sched.__class__.__name__}
        except Exception:
            return {"class": str(type(sched))}

    @staticmethod
    def _to_jsonable_config(c):
        try:
            json.dumps(c)
            return c
        except Exception:
            pass
        try:
            import importlib
            oc = importlib.import_module("omegaconf")
            OmegaConf = getattr(oc, "OmegaConf", None)
            if OmegaConf is not None and OmegaConf.is_config(c):
                return OmegaConf.to_container(c, resolve=True)
        except Exception:
            pass
        return str(c)