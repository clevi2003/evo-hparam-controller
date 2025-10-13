from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_cpu_state(state: Any) -> Any:
    """
    recursively move tensors to CPU to avoid device coupling in saved files
    """
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
            if link_path.is_symlink():
                link_path.unlink(missing_ok=True)
            else:
                link_path.unlink()
        link_path.symlink_to(target.name)  # relative symlink inside the same folder
    except Exception:
        # fallback to copy
        shutil.copyfile(target, link_path)


@dataclass
class CheckpointIO:
    """
    centralized, cpu-safe checkpointing for all training/evolution flows

    dir layout:
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
        # latest alias
        latest = self.root / "latest.pt"
        _safe_copy(self.root / "final.pt", latest)
        self._update_manifest(final_path=rel, latest_path="checkpoints/latest.pt")
        return rel

    def save_best_vector(self, vec: Union[np.ndarray, torch.Tensor]) -> str:
        if isinstance(vec, torch.Tensor):
            arr = vec.detach().to("cpu").float().numpy()
        else:
            arr = np.asarray(vec, dtype=np.float32)
        payload = {
            "schema": "v1",
            "controller_vec": arr.tolist(),
            "saved_at": _utc_now_iso(),
        }
        rel = "checkpoints/best_controller_vec.pt"
        path = self.root / "best_controller_vec.pt"
        torch.save(payload, path, _use_new_zipfile_serialization=True)
        self._register("best_controller_vec", rel)
        self._touch_manifest()
        return rel

    def save_controller_state(self, state_dict: Dict[str, Any]) -> str:
        payload = {
            "schema": "v1",
            "model_state": _to_cpu_state(state_dict),
            "saved_at": _utc_now_iso(),
        }
        rel = "checkpoints/best_controller.pt"
        path = self.root / "best_controller.pt"
        torch.save(payload, path, _use_new_zipfile_serialization=True)
        self._register("best_controller", rel)
        self._touch_manifest()
        return rel

    def save_warmup(self, state: Dict[str, Any]) -> str:
        payload = {
            "schema": "v1",
            "warmup_state": _to_cpu_state(state),
            "saved_at": _utc_now_iso(),
        }
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

        # update manifest
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
