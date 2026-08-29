"""The learned channel/spatial axis gate -- the one new component in DATNet."""
import math

import torch
import torch.nn as nn


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class AxisGate(nn.Module):
    """Per-channel blend weight g in [0,1]: out = g * channel_branch + (1-g) * spatial_branch.

    Two variants share this class:

    * fixed (cond_dim=None)  -- g = sigmoid(w), w a learnable per-channel vector.
      Learned during training, constant at inference. This is what produces the
      per-task axis-preference measurement.
    * conditioned (cond_dim=k) -- g = sigmoid(MLP(d)) for a degradation embedding
      d. The final linear layer is zero-initialised with bias `_logit(init)`, so
      an untrained conditioned gate is numerically identical to the fixed gate at
      its prior. That makes Phase 5 a strict extension of Phases 1-4 rather than
      a different initialisation.
    """

    def __init__(self, dim, init=0.5, cond_dim=None, hidden=64):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        b0 = _logit(init)
        if cond_dim is None:
            self.w = nn.Parameter(torch.full((dim,), b0))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].bias, b0)

    def forward(self, cond=None):
        """Returns g broadcastable over (B, C, H, W)."""
        if self.cond_dim is None:
            return torch.sigmoid(self.w).view(1, -1, 1, 1)
        if cond is None:
            raise ValueError("conditioned AxisGate called without a degradation embedding")
        return torch.sigmoid(self.mlp(cond)).view(cond.shape[0], -1, 1, 1)

    @torch.no_grad()
    def value(self, cond=None):
        """Mean gate value, for logging the axis-preference figure."""
        return self.forward(cond).mean().item()
