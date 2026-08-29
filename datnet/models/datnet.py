"""DATNet -- Dual-Axis Transformer Network for image restoration."""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .block import DATB
from .layers import Downsample, SkipFusion, Upsample


class _Stage(nn.Module):
    """A run of DATBs at one resolution, with alternating window shift."""

    def __init__(self, dim, depth, num_heads, mode, window_size, ffn_expansion,
                 gate_init, cond_dim, bias, grad_ckpt=False,
                 normalise_branches=True):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.blocks = nn.ModuleList([
            DATB(
                dim=dim,
                num_heads=num_heads,
                mode=mode,
                window_size=window_size,
                shift=(i % 2 == 1),      # alternate offset, as in Swin/SwinIR
                ffn_expansion=ffn_expansion,
                gate_init=gate_init,
                cond_dim=cond_dim,
                bias=bias,
                normalise_branches=normalise_branches,
            )
            for i in range(depth)
        ])

    def forward(self, x, cond=None):
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = cp.checkpoint(blk, x, cond, use_reentrant=False)
            else:
                x = blk(x, cond)
        return x


class RestorationTail(nn.Module):
    """Same-resolution tasks: predict the residual and add the input back."""

    def __init__(self, dim, out_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(dim, out_ch, 3, 1, 1, bias=False)

    def forward(self, feat, inp):
        return self.conv(feat) + inp


class SRTail(nn.Module):
    """Super-resolution: sub-pixel upsample by `scale`, no global residual.

    The LR input is not bicubic pre-upsampled, so there is no same-size tensor to
    add back; a global residual would have to be an explicit bicubic term, which
    costs accuracy at x4 (SwinIR-classical drops it too).
    """

    def __init__(self, dim, scale, out_ch=3):
        super().__init__()
        self.scale = scale
        self.conv_up = nn.Conv2d(dim, out_ch * scale * scale, 3, 1, 1, bias=True)
        self.shuffle = nn.PixelShuffle(scale)
        self.conv_last = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True)

    def forward(self, feat, inp):
        return self.conv_last(self.shuffle(self.conv_up(feat)))


class DATNet(nn.Module):
    """Dual-axis restoration backbone.

    topology='unet'  -- 3-level encoder/decoder. Used for denoise / deblur / derain.
    topology='flat'  -- single-resolution trunk at LR size, SwinIR style. Used for SR,
                        where an 4x downsample would reduce a 64x64 LR patch to 16x16.

    Every geometry knob is a constructor argument so that the three ablation arms
    (channel_only / window_only / dual) can be width-matched without touching depth.
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        width=32,
        topology="unet",
        enc_depths=(2, 3),
        bottleneck_depth=4,
        dec_depths=(3, 2),
        refine_depth=2,
        flat_depth=12,
        heads=(1, 2, 4),
        mode="dual",
        window_size=8,
        ffn_expansion=2.66,
        gate_init=(0.8, 0.5, 0.2),
        cond_dim=None,
        bias=False,
        task="restoration",
        scale=1,
        grad_ckpt=False,
        normalise_branches=True,
    ):
        super().__init__()
        assert topology in ("unet", "flat")
        assert task in ("restoration", "sr")
        self.topology = topology
        self.task = task
        self.scale = scale
        self.window_size = window_size
        self.mode = mode
        self.cond_dim = cond_dim

        self.patch_embed = nn.Conv2d(in_ch, width, 3, 1, 1, bias=bias)

        def stage(dim, depth, nh, g):
            return _Stage(dim, depth, nh, mode, window_size, ffn_expansion,
                          g, cond_dim, bias, grad_ckpt, normalise_branches)

        if topology == "unet":
            c1, c2, c3 = width, width * 2, width * 4
            g1, g2, g3 = gate_init
            self.enc1 = stage(c1, enc_depths[0], heads[0], g1)
            self.down1 = Downsample(c1)
            self.enc2 = stage(c2, enc_depths[1], heads[1], g2)
            self.down2 = Downsample(c2)
            self.bottleneck = stage(c3, bottleneck_depth, heads[2], g3)
            self.up2 = Upsample(c3)
            self.fuse2 = SkipFusion(c2, reduce=True)
            self.dec2 = stage(c2, dec_depths[0], heads[1], g2)
            self.up1 = Upsample(c2)
            self.fuse1 = SkipFusion(c1, reduce=True)
            self.dec1 = stage(c1, dec_depths[1], heads[0], g1)
            self.refine = stage(c1, refine_depth, heads[0], g1)
            feat_dim = c1
            # feature maps must survive two /2 downsamples and still tile windows
            self.size_multiple = 4 * window_size
        else:
            self.trunk = stage(width, flat_depth, heads[0], gate_init[0])
            self.conv_after_trunk = nn.Conv2d(width, width, 3, 1, 1, bias=bias)
            feat_dim = width
            self.size_multiple = window_size

        self.tail = (
            RestorationTail(feat_dim, out_ch) if task == "restoration"
            else SRTail(feat_dim, scale, out_ch)
        )

    # ------------------------------------------------------------------ forward
    def forward_features(self, x, cond=None):
        """Shared trunk. Returns (features, padded_input).

        Split out from forward() so the Phase-5 all-in-one model can hang several
        task-specific tails off one backbone without duplicating the trunk.
        """
        x = self._pad(x)
        if self.topology == "unet":
            f = self.patch_embed(x)
            e1 = self.enc1(f, cond)
            e2 = self.enc2(self.down1(e1), cond)
            b = self.bottleneck(self.down2(e2), cond)
            d2 = self.dec2(self.fuse2(self.up2(b), e2), cond)
            d1 = self.dec1(self.fuse1(self.up1(d2), e1), cond)
            feat = self.refine(d1, cond)
        else:
            shallow = self.patch_embed(x)
            feat = shallow + self.conv_after_trunk(self.trunk(shallow, cond))
        return feat, x

    def forward(self, x, cond=None):
        h, w = x.shape[-2:]
        feat, padded = self.forward_features(x, cond)
        out = self.tail(feat, padded)
        s = self.scale if self.task == "sr" else 1
        return out[..., : h * s, : w * s]

    def _pad(self, x):
        """Pad to a multiple of size_multiple so every stage tiles cleanly."""
        m = self.size_multiple
        h, w = x.shape[-2:]
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        return x

    # ------------------------------------------------------------------ logging
    @torch.no_grad()
    def gate_report(self, cond=None):
        """{block_name: mean gate value} -- the raw data for the axis-preference figure.

        Values near 1.0 mean the block leans on channel attention, near 0.0 on
        spatial attention.
        """
        report = OrderedDict()
        for name, module in self.named_modules():
            if isinstance(module, DATB) and module.gate is not None:
                report[name] = module.gate.value(cond)
        return report
