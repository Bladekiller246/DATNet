"""Paired crop / augmentation, shared by every dataset."""
import random

import numpy as np
import torch


def paired_random_crop(lq, gt, patch_size, scale=1):
    """Crop matching patches from a low-quality / ground-truth pair.

    `scale` handles super-resolution, where the GT crop is `scale` times the size
    of the LR crop and must be aligned to the same origin.
    """
    h, w = lq.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"image {h}x{w} smaller than patch {patch_size}")
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    lq = lq[top:top + patch_size, left:left + patch_size]
    gt = gt[top * scale:(top + patch_size) * scale,
            left * scale:(left + patch_size) * scale]
    return lq, gt


def augment(imgs, hflip=True, rot=True):
    """Flip + 90-degree rotation, applied identically to every image in the list.

    This is the D4 group. It is the only augmentation restoration papers use --
    anything that changes pixel statistics (colour jitter, blur, noise) would
    corrupt the degradation being learned.
    """
    do_h = hflip and random.random() < 0.5
    do_v = rot and random.random() < 0.5
    do_t = rot and random.random() < 0.5

    def _one(img):
        if do_h:
            img = img[:, ::-1, :]
        if do_v:
            img = img[::-1, :, :]
        if do_t:
            img = img.transpose(1, 0, 2)
        return np.ascontiguousarray(img)

    return [_one(i) for i in imgs]


def to_tensor(img):
    """HWC uint8 or float [0,1] -> CHW float32 tensor in [0,1]."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
