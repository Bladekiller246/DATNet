"""Gated depth-wise feed-forward network (GDFN, Restormer)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GDFN(nn.Module):
    """Conv1x1 -> DWConv3x3 -> split -> GELU(a) * b -> Conv1x1.

    The gating branch lets the FFN suppress channels that carry no useful signal
    at this location, which is why it outperforms a plain MLP on restoration. It
    is also the single most memory-hungry part of the block: the hidden width is
    2 * expansion * dim.
    """

    def __init__(self, dim, expansion=2.66, bias=False):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        a, b = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)
