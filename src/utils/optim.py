from __future__ import annotations
import torch

def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters. trainable_only=True counts requires_grad params only.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Return the first param-group LR (common when all groups share one LR).
    If groups differ, return the average LR.
    """
    lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
    return float(sum(lrs) / max(1, len(lrs)))


@torch.no_grad()
def compute_grad_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute total gradient norm across parameters. Returns 0.0 if no grads.
    """
    params = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    device = params[0].device
    if norm_type == float("inf"):
        norms = [g.abs().max().to(device) for g in params]
        return float(torch.stack(norms).max().item())
    total = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in params]), norm_type)
    return float(total.item())


def detach_scalar(x: torch.Tensor | float | int) -> float:
    """
    Turn a tensor scalar or python number into a python float.
    """
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    return float(x)