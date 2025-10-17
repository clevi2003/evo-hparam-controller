import os
import torch

def save_checkpoint(state: dict, ckpt_dir: str, name: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, name)
    torch.save(state, path)
    return path


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
