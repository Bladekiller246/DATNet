"""Test-set evaluation under the correct per-task protocol."""
import torch

from .inference import restore
from .metrics import psnr, ssim
from .protocols import get_protocol


@torch.no_grad()
def evaluate(model, dataset, task, scale=1, device="cuda", tile=None,
             amp_dtype=torch.bfloat16, cond=None, save_dir=None):
    """Returns {'psnr':..., 'ssim':..., 'n':...} under the protocol for `task`.

    The protocol (colour space and border shave) comes from protocols.py, never
    from the caller -- that is the whole point of having the table.
    """
    proto = get_protocol(task, scale)
    model = model.to(device).eval()
    tot_psnr = tot_ssim = 0.0

    for i in range(len(dataset)):
        item = dataset[i]
        x = item["input"].unsqueeze(0).to(device)
        y = item["target"].unsqueeze(0).to(device)
        with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = restore(model, x, scale=scale, tile=tile, cond=cond)
        pred = pred.float().clamp(0, 1)
        # a tiled/padded forward can overshoot by a few pixels; trim to GT
        pred = pred[..., : y.shape[-2], : y.shape[-1]]

        tot_psnr += psnr(pred, y, proto["crop_border"], proto["y_channel"])
        tot_ssim += ssim(pred, y, proto["crop_border"], proto["y_channel"])

        if save_dir:
            from ..data.io import imwrite
            import os
            imwrite(os.path.join(save_dir, item.get("name", f"{i:04d}.png")),
                    pred[0].permute(1, 2, 0).cpu().numpy())

    n = len(dataset)
    return {"psnr": tot_psnr / n, "ssim": tot_ssim / n, "n": n}


def make_val_fn(dataset, task, scale=1, device="cuda", max_images=None, tile=None):
    """A light validation callable for the trainer.

    Capped at `max_images` because validation runs inside the training loop and
    a full 100-image Urban100 pass at full resolution costs minutes, not seconds.
    """
    class _Subset(torch.utils.data.Dataset):
        def __len__(self):
            return min(len(dataset), max_images or len(dataset))

        def __getitem__(self, i):
            return dataset[i]

    subset = _Subset()

    def val_fn(model):
        return evaluate(model, subset, task, scale=scale, device=device, tile=tile)

    return val_fn
