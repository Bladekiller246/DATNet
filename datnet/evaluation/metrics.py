"""PSNR / SSIM matching the MATLAB + BasicSR conventions the literature uses.

Every function here takes float tensors in [0, 1], shape (B, 3, H, W), and
converts internally to the [0, 255] scale the reference implementations assume.
Getting the colour space or border shave wrong silently shifts numbers by
several tenths of a dB, which is larger than most of the effects this project is
trying to measure -- hence the unit tests in tests/test_metrics.py.
"""
import math

import torch
import torch.nn.functional as F

# ITU-R BT.601 luma coefficients on the [0, 255] scale, as used by MATLAB's
# rgb2ycbcr and therefore by every SR and deraining paper.
_Y_COEFF = (65.481, 128.553, 24.966)
_Y_OFFSET = 16.0


def to_y_channel(img):
    """(B,3,H,W) in [0,1] -> (B,1,H,W) luma on the [0,255] scale."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = (_Y_COEFF[0] * r + _Y_COEFF[1] * g + _Y_COEFF[2] * b) + _Y_OFFSET
    return y


def _prepare(pred, target, crop_border=0, y_channel=False):
    pred = pred.clamp(0, 1).to(torch.float64)
    target = target.clamp(0, 1).to(torch.float64)
    if y_channel:
        pred, target = to_y_channel(pred), to_y_channel(target)
    else:
        pred, target = pred * 255.0, target * 255.0
    if crop_border:
        c = crop_border
        pred = pred[..., c:-c, c:-c]
        target = target[..., c:-c, c:-c]
    return pred, target


def psnr(pred, target, crop_border=0, y_channel=False):
    pred, target = _prepare(pred, target, crop_border, y_channel)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    return (10.0 * torch.log10(255.0 ** 2 / mse)).mean().item()


def _gaussian_window(size=11, sigma=1.5, device="cpu", dtype=torch.float64):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])


def ssim(pred, target, crop_border=0, y_channel=False):
    """MATLAB-equivalent SSIM: 11x11 Gaussian (sigma 1.5), 'valid' convolution."""
    pred, target = _prepare(pred, target, crop_border, y_channel)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ch = pred.shape[1]
    win = _gaussian_window(device=pred.device, dtype=pred.dtype)
    win = win.expand(ch, 1, 11, 11).contiguous()

    def filt(x):
        return F.conv2d(x, win, groups=ch)  # 'valid', matches MATLAB's default

    mu1, mu2 = filt(pred), filt(target)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1 = filt(pred ** 2) - mu1_sq
    sigma2 = filt(target ** 2) - mu2_sq
    sigma12 = filt(pred * target) - mu1_mu2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    return (num / den).mean(dim=(1, 2, 3)).mean().item()
