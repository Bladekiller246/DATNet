"""MATLAB-compatible bicubic resize.

Every SR benchmark number in the literature is produced against LR images made
by MATLAB imresize with antialiasing. PIL BICUBIC and cv2.INTER_CUBIC are both
different kernels and produce LR images that are systematically easier, which
inflates PSNR by roughly 0.2-0.5 dB and makes the comparison meaningless. If the
official DIV2K bicubic LR set is available, use it; this module exists for the
cases where LR has to be generated (validation crops, extra scales).
"""
import math

import torch


def _cubic(x):
    """The MATLAB cubic kernel (a = -0.5)."""
    absx = x.abs()
    absx2, absx3 = absx ** 2, absx ** 3
    inner = (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1).to(x.dtype)
    outer = (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * \
            ((absx > 1) & (absx <= 2)).to(x.dtype)
    return inner + outer


def _contributions(in_len, out_len, scale, kernel_width, antialias):
    if antialias and scale < 1:
        kernel_width = kernel_width / scale
    x = torch.linspace(1, out_len, out_len)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)
    p = math.ceil(kernel_width) + 2
    indices = left.view(out_len, 1).expand(out_len, p) + \
        torch.linspace(0, p - 1, p).view(1, p).expand(out_len, p)
    distance = u.view(out_len, 1).expand(out_len, p) - indices
    weights = _cubic(distance * scale) * scale if (antialias and scale < 1) \
        else _cubic(distance)
    weights = weights / weights.sum(1, keepdim=True)

    # mirror out-of-range indices back inside, as MATLAB does
    indices = indices.clamp(min=1, max=in_len).long() - 1
    # drop columns whose weight is zero everywhere
    keep = weights.abs().sum(0) > 1e-8
    return indices[:, keep].contiguous(), weights[:, keep].contiguous()


def imresize(img, scale=None, out_shape=None, antialias=True):
    """Resize a (C, H, W) float tensor in [0,1]. Matches MATLAB imresize bicubic."""
    squeeze = img.dim() == 2
    if squeeze:
        img = img.unsqueeze(0)
    c, h, w = img.shape
    if out_shape is None:
        assert scale is not None, "pass scale or out_shape"
        out_h, out_w = int(math.ceil(h * scale)), int(math.ceil(w * scale))
        sh = sw = scale
    else:
        out_h, out_w = out_shape
        sh, sw = out_h / h, out_w / w

    idx_h, wt_h = _contributions(h, out_h, sh, 4.0, antialias)
    idx_w, wt_w = _contributions(w, out_w, sw, 4.0, antialias)
    wt_h, wt_w = wt_h.to(img.dtype), wt_w.to(img.dtype)

    # vertical pass, then horizontal
    out = img[:, idx_h.view(-1), :].view(c, out_h, -1, w)
    out = (out * wt_h.view(1, out_h, -1, 1)).sum(2)
    out = out[:, :, idx_w.view(-1)].view(c, out_h, out_w, -1)
    out = (out * wt_w.view(1, 1, out_w, -1)).sum(3)

    out = out.clamp(0, 1)
    return out.squeeze(0) if squeeze else out
