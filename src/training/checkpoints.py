from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch

from src.core.hooks.hook_base import Hook as HookBase

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_cpu_state(state: Any) -> Any:
    """Recursively move tensors to CPU to avoid device coupling in saved files."""
    if isinstance(state, torch.Tensor):
        return state.detach().to("cpu")
    if isinstance(state, dict):
        return {k: _to_cpu_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        t = [_to_cpu_state(v) for v in state]
        return type(state)(t) if isinstance(state, tuple) else t
    return state


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _safe_copy(target: Path, link_path: Path) -> None:
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink(missing_ok=True)
        link_path.symlink_to(target.name) # relative symlink inside the same folder
    except Exception:
        # fallback to copy
        shutil.copyfile(target, link_path)

def _safe_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    try:
        if obj is None:
            return None
        return obj.state_dict()
    except Exception:
        return None

def _is_better(curr: Optional[float], best: Optional[float], mode: str) -> bool:
    if curr is None:
        return False
    if best is None:
        return True
    return (curr > best) if mode == "max" else (curr < best)

def _extract_metric(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    val = metrics.get(key, None)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

def _normalize_mode(mode: str) -> str:
    m = (mode or "max").strip().lower()
    return "min" if m in ("min", "lower", "smaller", "less") else "max"

def _normalize_monitor(monitor: str) -> str:
    return (monitor or "val_acc").strip()

def _should_save_every(epoch: Optional[int], every: Optional[int]) -> bool:
    if epoch is None or not every:
        return False
    return (epoch >= 0) and (epoch % int(every) == 0)

def _grab(ctx: Dict[str, Any], key: str, default=None):
    return ctx.get(key, default) if isinstance(ctx, dict) else default

def _best_metric_default(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    # fallback heuristic if requested monitor is missing: use val acc, else -val loss
    if not isinstance(metrics, dict):
        return None
    if "val_acc" in metrics:
        try:
            return float(metrics["val_acc"])
        except Exception:
            return None
    if "val_loss" in metrics:
        try:
            return -float(metrics["val_loss"]) # negate so max still means better
        except Exception:
            return None
    return None

def _final_metric_value(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    if "val_acc" in metrics:
        try:
            return float(metrics["val_acc"])
        except Exception:
            return None
    if "val_loss" in metrics:
        try:
            return float(metrics["val_loss"])
        except Exception:
            return None
    return None

def _bundle_extra(ctx: Dict[str, Any]) -> Dict[str, Any]:
    extra = {}
    for k in ("epoch", "global_step", "run_id"):
        if k in ctx:
            extra[k] = ctx[k]
    metrics = _grab(ctx, "metrics", None)
    if isinstance(metrics, dict):
        compact = {}
        for mk, mv in metrics.items():
            try:
                compact[mk] = float(mv) if isinstance(mv, (int, float)) else mv
            except Exception:
                compact[mk] = str(mv)
        extra["metrics"] = compact
    return extra

def _ensure_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _maybe_epoch(ctx: Dict[str, Any]) -> Optional[int]:
    return _ensure_int(_grab(ctx, "epoch", None), None)

@dataclass
class CheckpointIO:
    """
    centralized, cpu-compatible checkpointing for all training/evo

    Directory layout:
      run_dir/
        checkpoints/
          final.pt
          best_val.pt
          epoch_020.pt
          best_controller_vec.pt
          best_controller.pt
          warmup.pt
          latest.pt   (symlink/copy to most recent)
          manifest.json
    """
    root: Path
    register_artifact: Optional[Callable[[str, str], None]] = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        if not self._manifest_path.exists():
            _atomic_write_json(self._manifest_path, {
                "schema": "v1",
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "last_epoch": None,
                "best_val": None,
                "final_path": None,
                "latest_path": None,
            })

    def save_epoch(
        self,
        model_state: Optional[Dict[str, Any]],
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        epoch: int = -1,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        name = f"epoch_{epoch:03d}.pt" if epoch >= 0 else "epoch_unknown.pt"
        return self._save_bundle(name, model_state, optimizer_state, scheduler_state, None, extra, epoch=epoch)

    def save_best_val(
        self,
        model_state: Optional[Dict[str, Any]],
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        metric_value: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        rel = self._save_bundle("best_val.pt", model_state, optimizer_state, scheduler_state, None, extra)
        self._update_manifest(best_val={
            "metric": "val_primary",
            "value": float(metric_value) if metric_value is not None else None,
            "path": rel
        })
        return rel

    def save_final(
        self,
        model_state: Optional[Dict[str, Any]],
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        rel = self._save_bundle("final.pt", model_state, optimizer_state, scheduler_state, None, extra)
        latest = self.root / "latest.pt"
        _safe_copy(self.root / "final.pt", latest)
        self._update_manifest(final_path=rel, latest_path="checkpoints/latest.pt")
        return rel

    def save_best_vector(self, vec: Union[np.ndarray, torch.Tensor]) -> str:
        if isinstance(vec, torch.Tensor):
            arr = vec.detach().to("cpu").float().numpy()
        else:
            arr = np.asarray(vec, dtype=np.float32)
        payload = {"schema": "v1", "controller_vec": arr.tolist(), "saved_at": _utc_now_iso()}
        rel = "checkpoints/best_controller_vec.pt"
        path = self.root / "best_controller_vec.pt"
        torch.save(payload, path, _use_new_zipfile_serialization=True)
        self._register("best_controller_vec", rel)
        self._touch_manifest()
        return rel

    def save_controller_state(self, state_dict: Dict[str, Any]) -> str:
        payload = {"schema": "v1", "model_state": _to_cpu_state(state_dict), "saved_at": _utc_now_iso()}
        rel = "checkpoints/best_controller.pt"
        path = self.root / "best_controller.pt"
        torch.save(payload, path, _use_new_zipfile_serialization=True)
        self._register("best_controller", rel)
        self._touch_manifest()
        return rel

    def save_warmup(self, state: Dict[str, Any]) -> str:
        payload = {"schema": "v1", "warmup_state": _to_cpu_state(state), "saved_at": _utc_now_iso()}
        rel = "checkpoints/warmup.pt"
        path = self.root / "warmup.pt"
        torch.save(payload, path, _use_new_zipfile_serialization=True)
        self._register("warmup", rel)
        self._touch_manifest()
        return rel

    def _save_bundle(
        self,
        filename: str,
        model_state: Optional[Dict[str, Any]],
        optimizer_state: Optional[Dict[str, Any]],
        scheduler_state: Optional[Dict[str, Any]],
        controller_vec: Optional[Union[np.ndarray, torch.Tensor]],
        extra: Optional[Dict[str, Any]],
        *,
        epoch: Optional[int] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "schema": "v1",
            "epoch": int(epoch) if epoch is not None else None,
            "model_state": _to_cpu_state(model_state) if model_state is not None else None,
            "optimizer_state": _to_cpu_state(optimizer_state) if optimizer_state is not None else None,
            "scheduler_state": _to_cpu_state(scheduler_state) if scheduler_state is not None else None,
            "controller_vec": None,
            "extra": extra or {},
            "saved_at": _utc_now_iso(),
        }
        if controller_vec is not None:
            if isinstance(controller_vec, torch.Tensor):
                payload["controller_vec"] = controller_vec.detach().to("cpu").float().numpy().tolist()
            else:
                payload["controller_vec"] = np.asarray(controller_vec, dtype=np.float32).tolist()

        path = self.root / filename
        torch.save(payload, path, _use_new_zipfile_serialization=True)

        updates: Dict[str, Any] = {"updated_at": _utc_now_iso()}
        if epoch is not None:
            updates["last_epoch"] = int(epoch)
        self._update_manifest(**updates)

        rel = f"checkpoints/{filename}"
        self._register(filename, rel)
        return rel

    def _update_manifest(self, **fields: Any) -> None:
        try:
            if self._manifest_path.exists():
                with self._manifest_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"schema": "v1"}
            data.update(fields)
            data["updated_at"] = _utc_now_iso()
            _atomic_write_json(self._manifest_path, data)
        except Exception:
            # do not crash training/evolution
            pass

    def _touch_manifest(self) -> None:
        self._update_manifest()

    def _register(self, name: str, rel_path: str) -> None:
        if self.register_artifact is not None:
            try:
                self.register_artifact(name, rel_path)
            except Exception:
                pass
    def make_checkpoint_hook(
        self,
        *,
        monitor: str = "val_acc",
        mode: str = "max", # max for accuracy, min for loss
        save_every: Optional[int] = None,
        save_best: bool = True,
        save_final: bool = True,
        with_optimizer: bool = True,
        with_scheduler: bool = True,
    ) -> "CheckpointHook":
        """
        return a Hook that saves periodic, best, and final checkpoints

        expected ctx keys at events:
          on_eval_end(ctx):  epoch:int, model, optimizer, scheduler, metrics:dict
          on_epoch_end(ctx): epoch:int, model, optimizer, scheduler
          on_train_end(ctx): epoch:int, model, optimizer, scheduler, metrics:dict (optional)
        """
        return CheckpointHook(
            ckpt=self,
            monitor=monitor,
            mode=mode,
            save_every=save_every,
            save_best=save_best,
            save_final=save_final,
            with_optimizer=with_optimizer,
            with_scheduler=with_scheduler,
        )

class CheckpointHook(HookBase):
    """
    a Hook that manages checkpointing policy:
      periodic epoch snapshots (save_every)
      best-by-metric on eval end (monitor/mode)
      final snapshot on train end
    """

    def __init__(
        self,
        *,
        ckpt: CheckpointIO,
        monitor: str = "val_acc",
        mode: str = "max",
        save_every: Optional[int] = None,
        save_best: bool = True,
        save_final: bool = True,
        with_optimizer: bool = True,
        with_scheduler: bool = True,
    ) -> None:
        super().__init__() if hasattr(super(), "__init__") else None
        self.ckpt = ckpt
        self.monitor = _normalize_monitor(monitor)
        self.mode = _normalize_mode(mode)          # "max" or "min"
        self.save_every = int(save_every) if save_every else None
        self.save_best = bool(save_best)
        self.save_final = bool(save_final)
        self.with_optimizer = bool(with_optimizer)
        self.with_scheduler = bool(with_scheduler)

        # internal best tracking
        self._best_comp: Optional[float] = None # value compared with max
        self._best_raw: Optional[float] = None # raw monitored value (non-negated bc sometimes we negate to still use max)

    def on_eval_end(self, ctx: Dict[str, Any]) -> None:
        metrics = _grab(ctx, "metrics", None)
        epoch = _maybe_epoch(ctx)

        # periodic epoch save
        if _should_save_every(epoch, self.save_every):
            self._save_epoch_ctx(ctx)

        if not self.save_best:
            return

        comp_curr = self._resolve_monitored_value(metrics)
        if _is_better(comp_curr, self._best_comp, mode="max"): # compare on normalized max scale
            self._best_comp = comp_curr
            raw_val = _extract_metric(metrics, self.monitor)
            self._best_raw = raw_val if raw_val is not None else _final_metric_value(metrics)
            self._save_best_ctx(ctx, self._best_raw)

    def on_epoch_end(self, ctx: Dict[str, Any]) -> None:
        epoch = _maybe_epoch(ctx)
        if _should_save_every(epoch, self.save_every):
            self._save_epoch_ctx(ctx)

    def on_train_end(self, ctx: Dict[str, Any]) -> None:
        if self.save_final:
            self._save_final_ctx(ctx)

    def _resolve_monitored_value(self, metrics: Optional[Dict[str, Any]]) -> Optional[float]:
        v = _extract_metric(metrics, self.monitor)
        if v is not None:
            return v if self.mode == "max" else -v
        return _best_metric_default(metrics)

    def _extract_states(self, ctx: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        model = _grab(ctx, "model", None)
        optim = _grab(ctx, "optimizer", None) if self.with_optimizer else None
        sched = _grab(ctx, "scheduler", None) if self.with_scheduler else None
        return _safe_state_dict(model), _safe_state_dict(optim), _safe_state_dict(sched)

    def _save_epoch_ctx(self, ctx: Dict[str, Any]) -> Optional[str]:
        try:
            m, o, s = self._extract_states(ctx)
            return self.ckpt.save_epoch(m, o, s, epoch=_maybe_epoch(ctx), extra=_bundle_extra(ctx))
        except Exception:
            return None

    def _save_best_ctx(self, ctx: Dict[str, Any], value: Optional[float]) -> Optional[str]:
        try:
            m, o, s = self._extract_states(ctx)
            return self.ckpt.save_best_val(m, o, s, metric_value=value, extra=_bundle_extra(ctx))
        except Exception:
            return None

    def _save_final_ctx(self, ctx: Dict[str, Any]) -> Optional[str]:
        try:
            m, o, s = self._extract_states(ctx)
            return self.ckpt.save_final(m, o, s, extra=_bundle_extra(ctx))
        except Exception:
            return None
