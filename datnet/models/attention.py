"""The two attention axes: channel (MDTA) and spatial (shifted-window MSA)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """MDTA (Restormer). Attention over the CHANNEL axis.

    The attention map is (C/h, C/h) rather than (HW, HW), so cost is linear in
    the number of pixels. q and k are L2-normalised along the pixel axis and the
    logits are scaled by a learned per-head temperature, which is what keeps the
    softmax stable when HW changes between training patches and full test images.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)

        def heads(t):
            return t.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q, k, v = heads(q), heads(k), heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)


class WindowAttention(nn.Module):
    """Shifted-window MSA (SwinIR/Swin). Attention over the SPATIAL axis.

    Self-attention inside non-overlapping MxM windows, with the window grid
    offset by M/2 on alternate blocks so information crosses window boundaries.
    Cost is linear in pixels for fixed M.

    Inputs and outputs are NCHW so this drops into the same slot as MDTA. Feature
    maps are reflect-padded up to a multiple of M and cropped afterwards, so the
    block accepts arbitrary test-image sizes.
    """

    def __init__(self, dim, num_heads, window_size=8, shift=False, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

        m = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * m - 1) * (2 * m - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(torch.arange(m), torch.arange(m), indexing="ij")
        ).flatten(1)                                   # (2, m*m)
        rel = coords[:, :, None] - coords[:, None, :]  # (2, m*m, m*m)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += m - 1
        rel[:, :, 1] += m - 1
        rel[:, :, 0] *= 2 * m - 1
        self.register_buffer("rp_index", rel.sum(-1), persistent=False)
        self._mask_cache = {}

    # -- window partition / reverse, operating on (B, heads, head_dim, H, W) ----
    def _partition(self, t, b, hp, wp):
        m = self.window_size
        t = t.view(b, self.num_heads, self.head_dim, hp // m, m, wp // m, m)
        t = t.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
        return t.view(-1, self.num_heads, m * m, self.head_dim)

    def _reverse(self, t, b, hp, wp):
        m = self.window_size
        t = t.view(b, hp // m, wp // m, self.num_heads, m, m, self.head_dim)
        t = t.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        return t.view(b, self.dim, hp, wp)

    def _attn_mask(self, hp, wp, device, dtype):
        """Additive mask that stops shifted windows from mixing wrapped-around regions."""
        if self.shift_size == 0:
            return None
        key = (hp, wp, device, dtype)
        if key in self._mask_cache:
            return self._mask_cache[key]
        m, s = self.window_size, self.shift_size
        img = torch.zeros(1, 1, hp, wp, device=device)
        cnt = 0
        for hsl in (slice(0, -m), slice(-m, -s), slice(-s, None)):
            for wsl in (slice(0, -m), slice(-m, -s), slice(-s, None)):
                img[:, :, hsl, wsl] = cnt
                cnt += 1
        img = img.view(1, 1, hp // m, m, wp // m, m)
        img = img.permute(0, 2, 4, 1, 3, 5).reshape(-1, m * m)
        mask = img.unsqueeze(1) - img.unsqueeze(2)               # (nW, m*m, m*m)
        mask = mask.masked_fill(mask != 0, float("-inf")).masked_fill(mask == 0, 0.0)
        mask = mask.unsqueeze(1).to(dtype)                        # (nW, 1, m*m, m*m)
        self._mask_cache[key] = mask
        return mask

    def forward(self, x):
        b, c, h, w = x.shape
        m, s = self.window_size, self.shift_size
        pad_b, pad_r = (m - h % m) % m, (m - w % m) % m
        if pad_b or pad_r:
            # reflect padding needs pad < dim; fall back to replicate on tiny maps
            mode = "reflect" if (pad_r < w and pad_b < h) else "replicate"
            x = F.pad(x, (0, pad_r, 0, pad_b), mode=mode)
        hp, wp = h + pad_b, w + pad_r

        qkv = self.qkv(x)
        if s > 0:
            qkv = torch.roll(qkv, shifts=(-s, -s), dims=(2, 3))
        q, k, v = qkv.chunk(3, dim=1)
        q = self._partition(q, b, hp, wp) * self.scale
        k = self._partition(k, b, hp, wp)
        v = self._partition(v, b, hp, wp)

        bias = self.relative_position_bias_table[self.rp_index.view(-1)]
        bias = bias.view(m * m, m * m, self.num_heads).permute(2, 0, 1)
        bias = bias.unsqueeze(0).to(q.dtype)                      # (1, heads, m*m, m*m)
        mask = self._attn_mask(hp, wp, x.device, q.dtype)
        if mask is not None:
            # (nW,1,mm,mm) + (1,heads,mm,mm) -> (nW,heads,mm,mm), tiled b-major
            # to match the (b, nWin) flattening order used by _partition.
            bias = (bias + mask).repeat(b, 1, 1, 1)

        # scale already folded into q, so pass scale=1.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)

        out = self._reverse(out, b, hp, wp)
        if s > 0:
            out = torch.roll(out, shifts=(s, s), dims=(2, 3))
        out = self.proj(out)
        return out[:, :, :h, :w] if (pad_b or pad_r) else out
