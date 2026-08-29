"""Protocol unit tests.

The plan flags getting the evaluation convention wrong as the failure mode that
silently invalidates every comparison. These tests pin the conventions down.

Run:  python -m pytest tests/ -q     (or  python tests/test_metrics.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datnet.evaluation.metrics import psnr, ssim, to_y_channel
from datnet.evaluation.protocols import get_protocol


def test_identical_images():
    x = torch.rand(1, 3, 64, 64)
    assert psnr(x, x) > 100, "identical images must give a very large PSNR"
    assert abs(ssim(x, x) - 1.0) < 1e-6, "identical images must give SSIM 1.0"


def test_known_psnr():
    """A constant offset of 1/255 over the whole image is exactly 48.13 dB."""
    x = torch.zeros(1, 3, 32, 32)
    y = torch.full_like(x, 1.0 / 255.0)
    expected = 10 * torch.log10(torch.tensor(255.0 ** 2 / 1.0)).item()
    assert abs(psnr(x, y) - expected) < 1e-4


def test_y_channel_matches_matlab():
    """Pure white must map to Y = 235, pure black to Y = 16 (BT.601, studio swing)."""
    white = torch.ones(1, 3, 4, 4)
    black = torch.zeros(1, 3, 4, 4)
    assert abs(to_y_channel(white).mean().item() - 235.0) < 1e-3
    assert abs(to_y_channel(black).mean().item() - 16.0) < 1e-6


def test_y_channel_changes_the_number():
    """RGB and Y PSNR must differ -- if they do not, the conversion is a no-op."""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    y = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    assert abs(psnr(x, y, y_channel=False) - psnr(x, y, y_channel=True)) > 0.1


def test_crop_border_changes_the_number():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    y = x.clone()
    y[..., :4, :] = 0  # corrupt only the border that a shave would remove
    assert psnr(x, y, crop_border=4) > psnr(x, y, crop_border=0) + 5


def test_protocol_table():
    assert get_protocol("sr", scale=4)["crop_border"] == 4
    assert get_protocol("sr", scale=2)["crop_border"] == 2
    assert get_protocol("sr")["y_channel"] is True
    assert get_protocol("deblur")["y_channel"] is False
    assert get_protocol("derain")["y_channel"] is True
    assert get_protocol("denoise_gaussian")["y_channel"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
