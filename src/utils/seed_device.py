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

def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Choose a device. If `preferred` is given ("cuda"|"mps"|"cpu"), try that, else auto-detect.
    """
    if preferred:
        pref = preferred.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): 
            return torch.device("mps")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
