#Kevin Liang
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

@torch.no_grad()
def compute_ece(model: torch.nn.Module,
                device: torch.device,
                val_loader: DataLoader,
                n_bins: int = 15) -> float:
    """
    Expected Calibration Error using max-probability binning.
    Returns a single float in [0, 1] (lower is better).
    """
    model.eval()
    confidences, correctness = [], []

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        confidences.append(conf)
        correctness.append((preds == y).float())

    if not confidences:
        return 0.0

    conf = torch.cat(confidences)
    corr = torch.cat(correctness)

    bins = torch.linspace(0, 1, steps=n_bins + 1, device=conf.device)
    ece = torch.tensor(0.0, device=conf.device)
    N = conf.numel()

    for i in range(n_bins):
        in_bin = (conf > bins[i]) & (conf <= bins[i + 1])
        count = in_bin.sum()
        if count.item() > 0:
            acc_bin = corr[in_bin].mean()
            conf_bin = conf[in_bin].mean()
            ece += (count.float() / N) * (acc_bin - conf_bin).abs()

    return float(ece.item())