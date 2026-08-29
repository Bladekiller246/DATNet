"""Full-image inference, with tiling for images that do not fit in 8 GB.

Urban100 and Manga109 images reach 1024x1024 and beyond. At width 32 a full-image
forward pass stores activations for the whole map at every level, which on an
8 GB card OOMs well before the largest test images. Tiled inference with a
generous overlap and a raised-cosine blend removes the seams; a pure hard-edge
tiling leaves visible grid artefacts that cost real PSNR.
"""
import torch


@torch.no_grad()
def forward_whole(model, x, cond=None):
    return model(x, cond) if cond is not None else model(x)


@torch.no_grad()
def forward_tiled(model, x, tile=256, overlap=32, scale=1, cond=None):
    """Sliding-window inference with cosine-feathered blending.

    `scale` is the output/input size ratio (1 for restoration, s for SR).
    """
    b, c, h, w = x.shape
    stride = tile - overlap
    out_h, out_w = h * scale, w * scale
    acc = torch.zeros(b, c, out_h, out_w, device=x.device, dtype=torch.float32)
    wgt = torch.zeros(1, 1, out_h, out_w, device=x.device, dtype=torch.float32)

    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)

    ramp = _feather(tile * scale, overlap * scale, x.device)
    mask = (ramp.view(-1, 1) * ramp.view(1, -1)).view(1, 1, tile * scale, tile * scale)

    for y in ys:
        for xx in xs:
            patch = x[:, :, y:y + tile, xx:xx + tile]
            pred = forward_whole(model, patch, cond).float()
            oy, ox = y * scale, xx * scale
            ph, pw = pred.shape[-2:]
            acc[:, :, oy:oy + ph, ox:ox + pw] += pred * mask[..., :ph, :pw]
            wgt[:, :, oy:oy + ph, ox:ox + pw] += mask[..., :ph, :pw]
    return acc / wgt.clamp(min=1e-8)


def _feather(size, overlap, device):
    """1-D raised-cosine ramp, strictly positive at the tile edge.

    The ramp must never reach exactly zero. Interior pixels are covered by
    several tiles, so the `acc / wgt` normalisation recovers the right value for
    any positive weights. But pixels on the image boundary are covered by
    exactly ONE tile: if that tile's weight there is 0, then acc = 0 and wgt = 0,
    the clamp turns it into 0 / 1e-8, and the output is a black ring one pixel
    wide.

    That ring is only ~1% of a 481x321 image and still cost ~6 dB -- tiled
    inference read 26.77 dB where whole-image inference read 32.81 dB on the
    same checkpoint. Sampling the cosine on the open interval (0, 1) keeps every
    weight positive and makes the two paths agree.
    """
    r = torch.ones(size, device=device)
    if overlap > 0:
        t = (torch.arange(overlap, device=device, dtype=torch.float32) + 1.0) \
            / (overlap + 1.0)
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * t)
        r[:overlap] = ramp
        r[-overlap:] = ramp.flip(0)
    return r


@torch.no_grad()
def restore(model, x, scale=1, tile=None, overlap=32, cond=None):
    """Whole-image if it fits, tiled otherwise. Falls back to tiling on OOM."""
    if tile is None:
        try:
            return forward_whole(model, x, cond)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tile = 256
    return forward_tiled(model, x, tile=tile, overlap=overlap, scale=scale, cond=cond)
