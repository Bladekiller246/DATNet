"""Dual-Axis Transformer Block (DATB)."""
import torch
import torch.nn as nn

from .attention import ChannelAttention, WindowAttention
from .ffn import GDFN
from .gate import AxisGate
from .layers import LayerNorm2d

MODES = ("channel_only", "window_only", "dual")


class DATB(nn.Module):
    """x + Fuse(MDTA(x), SW-MSA(x))  then  x + GDFN(x).

    `mode` selects the ablation arm:
      channel_only -- Branch B removed  (Restormer-lite)
      window_only  -- Branch A removed  (SwinIR / Uformer-lite)
      dual         -- both branches, blended by a learned per-channel gate

    Parameter matching between arms is done by widening `dim`, never by changing
    the block count -- see utils/matching.py.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mode="dual",
        window_size=8,
        shift=False,
        ffn_expansion=2.66,
        gate_init=0.5,
        cond_dim=None,
        bias=False,
        bias_free_norm=True,
        normalise_branches=True,
    ):
        super().__init__()
        assert mode in MODES, f"unknown mode {mode!r}; expected one of {MODES}"
        self.mode = mode
        self.dim = dim
        self.normalise_branches = normalise_branches

        self.norm1 = LayerNorm2d(dim, bias_free=bias_free_norm)
        self.norm2 = LayerNorm2d(dim, bias_free=bias_free_norm)

        self.channel_attn = (
            ChannelAttention(dim, num_heads, bias=bias)
            if mode in ("channel_only", "dual") else None
        )
        self.window_attn = (
            WindowAttention(dim, num_heads, window_size=window_size, shift=shift)
            if mode in ("window_only", "dual") else None
        )
        self.gate = AxisGate(dim, init=gate_init, cond_dim=cond_dim) if mode == "dual" else None

        self.ffn = GDFN(dim, expansion=ffn_expansion, bias=bias)

    @staticmethod
    def _rms(x, eps=1e-6):
        """Per-sample RMS over (C, H, W) -- one scalar per image."""
        return torch.sqrt(x.float().pow(2).mean(dim=(1, 2, 3), keepdim=True) + eps)

    def _balance(self, a, b):
        """Rescale A and B to a COMMON magnitude so only `g` sets the balance.

        Why not simply normalise each branch to unit RMS: the block output feeds
        a residual add, so it must be able to emit a SMALL correction when
        little is needed. Forcing unit RMS removes that freedom and measurably
        breaks fitting -- the single-image overfit check fell from 51.9 dB to
        34.1 dB.

        Instead both branches are scaled to the mean of their two RMS values.
        The ratio |A|/|B| becomes exactly 1, so the network can no longer shift
        the blend by rescaling a branch, while the shared magnitude stays
        data-dependent and free to shrink.

        Note this is a no-op for a single branch (m == rms(a) => a unchanged),
        which is why the single-axis arms need no special case and the ablation
        stays clean: `dual` differs from `channel_only` by the spatial branch
        and nothing else. It also adds NO parameters, so the existing
        width-matching is unaffected.
        """
        ra, rb = self._rms(a), self._rms(b)
        m = 0.5 * (ra + rb)
        return a * (m / ra).to(a.dtype), b * (m / rb).to(b.dtype)

    def forward(self, x, cond=None):
        y = self.norm1(x)

        if self.mode == "channel_only":
            attn = self.channel_attn(y)
        elif self.mode == "window_only":
            attn = self.window_attn(y)
        else:
            a = self.channel_attn(y)
            b = self.window_attn(y)
            if self.normalise_branches:
                # Without this the network rescales A and B internally (each
                # ends in its own 1x1 projection), reproduces any effective
                # blend it likes, and leaves g frozen at initialisation --
                # measured on a real 30k run: g moved 0.005 while the true
                # |A|/|B| ratio sat at 0.28 and the effective balance was 0.26
                # against a nominal 0.54.
                a, b = self._balance(a, b)
            g = self.gate(cond)
            attn = g * a + (1.0 - g) * b

        x = x + attn
        return x + self.ffn(self.norm2(x))

    @torch.no_grad()
    def effective_ratio(self, x, cond=None):
        """|gA| / (|gA| + |(1-g)B|) -- what the block ACTUALLY does.

        Report this, not `g`. With `normalise_branches=True` the two agree by
        construction; without it they can disagree wildly, which is exactly the
        failure this method exists to expose.
        """
        if self.mode != "dual":
            return 1.0 if self.mode == "channel_only" else 0.0
        y = self.norm1(x)
        a, b = self.channel_attn(y), self.window_attn(y)
        if self.normalise_branches:
            a, b = self._balance(a, b)
        g = self.gate(cond)
        ca = (g * a).abs().mean().item()
        cb = ((1.0 - g) * b).abs().mean().item()
        return ca / max(ca + cb, 1e-12)
