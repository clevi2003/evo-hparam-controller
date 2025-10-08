from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

import torch
import numpy as np

@dataclass(frozen=True)
class FlatSpec:
    """Specification describing how a flat vector maps to model tensors."""
    names: Tuple[str, ...]
    shapes: Tuple[torch.Size, ...]
    dtypes: Tuple[torch.dtype, ...]
    devices: Tuple[torch.device, ...]
    sizes: Tuple[int, ...]  # number of elements per tensor

    @property
    def total(self) -> int:
        return int(sum(self.sizes))


def _iter_named_tensors(
    model: torch.nn.Module,
    trainable_only: bool = True,
    include_buffers: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Deterministic iterator over tensors to serialize
    Order: all named_parameters (registration order), then optionally named_buffers
    This keeps registration order for stability. PyTorch preserves it
    """
    for name, p in model.named_parameters(recurse=True):
        if trainable_only and not p.requires_grad:
            continue
        yield name, p.data
    if include_buffers:
        for name, b in model.named_buffers(recurse=True):
            # buffers are typically running stats. Skip if not floating.
            yield f"[buffer]{name}", b.data


def _make_spec(
    tensors: Iterable[Tuple[str, torch.Tensor]]
) -> FlatSpec:
    names: List[str] = []
    shapes: List[torch.Size] = []
    dtypes: List[torch.dtype] = []
    devices: List[torch.device] = []
    sizes: List[int] = []
    for n, t in tensors:
        names.append(n)
        shapes.append(t.shape)
        dtypes.append(t.dtype)
        devices.append(t.device)
        sizes.append(int(t.numel()))
    return FlatSpec(tuple(names), tuple(shapes), tuple(dtypes), tuple(devices), tuple(sizes))


def flatten_params(
    model: torch.nn.Module,
    *,
    trainable_only: bool = True,
    include_buffers: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    spec_out: Optional[List[FlatSpec]] = None,
) -> torch.Tensor:
    """
    Flatten selected tensors from model into a single 1-D tensor

    parameters:
        trainable_only: if True, include only params with requires_grad
        include_buffers: if True, append buffers after parameters
        device: device for the returned vector default: model device of first tensor
        dtype: dtype for the returned vector, default is float32
        spec_out: if provided, this list will receive the FlatSpec used for the flattening

    returns:
        1-D tensor with concatenated values in deterministic order
    """
    items = list(_iter_named_tensors(model, trainable_only=trainable_only, include_buffers=include_buffers))
    if not items:
        return torch.empty(0, dtype=dtype or torch.float32, device=device or torch.device("cpu"))

    if spec_out is not None:
        spec_out.append(_make_spec(items))

    # choose target device if not provided
    if device is None:
        device = items[0][1].device

    vecs: List[torch.Tensor] = []
    for _, t in items:
        tt = t.detach().reshape(-1)
        if dtype is not None and tt.dtype != dtype:
            tt = tt.to(dtype)
        if tt.device != device:
            tt = tt.to(device)
        vecs.append(tt)

    return torch.cat(vecs, dim=0)


def unflatten_params(
    model: torch.nn.Module,
    vector: torch.Tensor,
    *,
    trainable_only: bool = True,
    include_buffers: bool = False,
    strict: bool = True,
    expected_spec: Optional[FlatSpec] = None,
) -> None:
    """
    Load a flat vector back into the model and optionally buffers in the same order
    used by flatten_params

    parameters:
        vector: 1-D tensor with the serialized values
        strict: if True, validates shape/length and names/dtypes/devices when spec is given
        expected_spec: optional FlatSpec to validate against, can be useful for replay

    notes:
        respects original tensor dtypes/devices, casts from the vector
        does not change requires_grad flags
    """
    items = list(_iter_named_tensors(model, trainable_only=trainable_only, include_buffers=include_buffers))
    spec = _make_spec(items)

    if strict:
        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1-D, got shape {tuple(vector.shape)}")
        if int(vector.numel()) != spec.total:
            raise ValueError(f"Vector length {int(vector.numel())} != expected {spec.total}")
        if expected_spec is not None:
            if expected_spec.names != spec.names or expected_spec.shapes != spec.shapes:
                raise ValueError("Model structure does not match expected FlatSpec (names/shapes differ).")

    # copy slices into param/buffer tensors
    offset = 0
    for (name, t), size in zip(items, spec.sizes):
        chunk = vector[offset: offset + size]
        offset += size
        # reshape and cast to original tensor dtype/device
        chunk = chunk.to(dtype=t.dtype, device=t.device).view(t.shape)
        t.copy_(chunk)  # in-place without changing requires_grad
    assert offset == vector.numel(), "Serialization offset mismatch"

def flat_spec(
    model: torch.nn.Module,
    *,
    trainable_only: bool = True,
    include_buffers: bool = False,
) -> FlatSpec:
    """Return the FlatSpec that would be used for flatten/unflatten"""
    return _make_spec(_iter_named_tensors(model, trainable_only=trainable_only, include_buffers=include_buffers))


def num_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters that would be serialized (excluding buffers by default)"""
    return int(sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)))


def set_requires_grad(model: torch.nn.Module, requires_grad: bool) -> None:
    """Enable/disable gradients for all parameters (useful for evolution evaluation)"""
    for p in model.parameters():
        p.requires_grad = requires_grad


def to_numpy(vector: torch.Tensor) -> "np.ndarray":
    """Return a detached CPU float64 np array """
    return vector.detach().to("cpu", dtype=torch.float64).numpy()


def from_numpy(array: "np.ndarray", device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:  # type: ignore[name-defined]
    """Create a 1-D torch tensor from a NumPy array with target device/dtype"""
    if not isinstance(array, np.ndarray):
        raise TypeError("from_numpy expects a NumPy ndarray.")
    t = torch.from_numpy(array.copy())  # copy to avoid write-through weird behaviors
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t.view(-1)

def save_vector(path: str | Path, vector: torch.Tensor, meta: Optional[dict] = None) -> None:
    """
    Save a flat vector plus optional metadata. Uses torch.save for simplicity
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vector": vector.detach().cpu(), "meta": meta or {}}
    torch.save(payload, p)


def load_vector(path: str | Path, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Load a flat vector saved with save_vector
    """
    payload = torch.load(Path(path), map_location="cpu")
    vec: torch.Tensor = payload["vector"]
    if dtype is not None:
        vec = vec.to(dtype)
    if device is not None:
        vec = vec.to(device)
    return vec.view(-1)
