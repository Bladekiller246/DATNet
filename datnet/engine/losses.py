"""Reconstruction losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant used by most restoration papers (eps = 1e-3)."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class FFTLoss(nn.Module):
    """L1 in the frequency domain.

    Helps on motion deblurring, where the degradation is a spatial convolution
    and the missing information is concentrated at specific frequencies. Use a
    small weight -- a large one oversmooths, trading visible detail for PSNR.
    """

    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred.float(), norm="backward")
        tf = torch.fft.rfft2(target.float(), norm="backward")
        return F.l1_loss(torch.stack([pf.real, pf.imag], -1),
                         torch.stack([tf.real, tf.imag], -1))


class RestorationLoss(nn.Module):
    """w_pixel * pixel + w_fft * fft, with an optional auxiliary classifier term."""

    def __init__(self, pixel="l1", w_pixel=1.0, w_fft=0.0, w_cls=0.0):
        super().__init__()
        self.pixel = {"l1": nn.L1Loss(), "charbonnier": CharbonnierLoss()}[pixel]
        self.fft = FFTLoss() if w_fft > 0 else None
        self.w_pixel, self.w_fft, self.w_cls = w_pixel, w_fft, w_cls
        self.ce = nn.CrossEntropyLoss() if w_cls > 0 else None

    def forward(self, pred, target, logits=None, task_id=None):
        parts = {"pixel": self.pixel(pred, target)}
        total = self.w_pixel * parts["pixel"]
        if self.fft is not None:
            parts["fft"] = self.fft(pred, target)
            total = total + self.w_fft * parts["fft"]
        if self.ce is not None and logits is not None and task_id is not None:
            parts["cls"] = self.ce(logits, task_id)
            total = total + self.w_cls * parts["cls"]
        parts["total"] = total
        # detach before the float() cast: parts is for logging only
        return total, {k: float(v.detach()) for k, v in parts.items()}
