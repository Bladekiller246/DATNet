"""Primitive layers shared by every DATNet variant."""
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W).

    `bias_free=True` reproduces Restormer's BiasFree_LayerNorm: it divides by the
    channel std without subtracting the mean. Subtracting the mean removes a DC
    component that restoration networks need to carry through, which is why
    Restormer/NAFNet default to the bias-free form.
    """

    def __init__(self, dim, bias_free=True, eps=1e-5):
        super().__init__()
        self.bias_free = bias_free
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = None if bias_free else nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        if self.bias_free:
            sigma = x.var(dim=1, keepdim=True, unbiased=False)
            out = x / torch.sqrt(sigma + self.eps)
            return out * self.weight.view(1, -1, 1, 1)
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        out = (x - mu) / torch.sqrt(sigma + self.eps)
        return out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class Downsample(nn.Module):
    """C -> 2C at half resolution, via conv + pixel-unshuffle (Restormer style).

    Pixel-unshuffle moves spatial detail into channels instead of discarding it,
    so the downsample is information-preserving up to the 3x3 conv.
    """

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Downsample needs an even channel count"
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """C -> C/2 at double resolution, via conv + pixel-shuffle."""

    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class SkipFusion(nn.Module):
    """Concatenate a decoder feature with its encoder skip, then project back.

    `reduce=False` keeps the doubled width (Restormer does this at its widest
    decoder level); `reduce=True` projects 2C -> C with a 1x1.
    """

    def __init__(self, dim, reduce=True):
        super().__init__()
        self.proj = nn.Conv2d(dim * 2, dim, 1, bias=False) if reduce else nn.Identity()

    def forward(self, x, skip):
        return self.proj(torch.cat([x, skip], dim=1))
