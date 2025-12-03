import torch
import torch.nn as nn


class ControllerMLP(nn.Module):
    """
    Lightweight MLP used as a learned LR controller.

    Now includes LayerNorm to stabilize inputs whose scale and distribution
    drift during training (loss/acc/log-step/grad-norm/etc).
    LayerNorm is batch-size agnostic and works perfectly with B=1.
    """

    def __init__(self, in_dim: int, hidden: int = 32, max_step: float = 0.2):
        super().__init__()
        # Normalizes across features for each sample (batch-independent).
        self.ln = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # max magnitude of delta(log lr)
        self.max_step = max_step
        # initialize weights more stably (optional but beneficial)
        self._init_weights()

    def _init_weights(self):
        # small weight initialization to avoid exploding deltas early in training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [B, F]
        returns: tensor of shape [B] representing delta(log lr)
        """
        # normalize input features
        x = self.ln(x)
        # forward pass through MLP
        raw = self.net(x)  # shape [B, 1]
        # bound output using tanh to prevent wild LR changes
        delta = torch.tanh(raw) * self.max_step

        return delta.squeeze(-1)  # [B]
