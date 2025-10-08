import torch, torch.nn as nn, torch.nn.functional as F

class ControllerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, max_step: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.max_step = max_step

    def forward(self, x):
        raw = self.net(x)        # shape [B, 1]
        delta = torch.tanh(raw) * self.max_step
        return delta.squeeze(-1) # delta log lr
