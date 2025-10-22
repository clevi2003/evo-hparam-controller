import os
import random
import numpy as np
import torch
from typing import Optional

def seed_everything(seed: int = 1337, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    else:
        torch.backends.cudnn.benchmark = True

def mps_ok() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available()

def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_ok():
        return torch.device("mps")
    return torch.device("cpu")

def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    pick a device. If preference is given ("cuda"|"mps"|"cpu"|"auto"), try that;
    otherwise auto detect with priority: CUDA > MPS > CPU
    """
    pref = (preferred or "auto").strip().lower()
    if pref in ("auto", "any", ""):
        return auto_device()
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else auto_device()
    if pref == "mps":
        return torch.device("mps") if mps_ok() else auto_device()
    if pref == "cpu":
        return torch.device("cpu")
    return auto_device()