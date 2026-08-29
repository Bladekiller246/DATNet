"""Training and test datasets, one class per degradation family.

All of them return {'input', 'target', 'task_id'} so the trainer never branches
on task. `task_id` is ignored by the single-task phases and consumed by the
Phase-5 conditioned gate.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .bicubic import imresize
from .io import imread, list_images
from .transforms import augment, paired_random_crop, to_tensor

TASK_IDS = {"denoise": 0, "deblur": 1, "sr": 2, "derain": 3}


class _Base(Dataset):
    def __init__(self, patch_size, task, augment_data=True, repeat=1):
        self.patch_size = patch_size
        self.task = task
        self.task_id = TASK_IDS[task]
        self.augment_data = augment_data
        self.repeat = repeat

    def _pack(self, lq, gt):
        return {
            "input": to_tensor(lq),
            "target": to_tensor(gt),
            "task_id": torch.tensor(self.task_id, dtype=torch.long),
        }


class GaussianDenoiseDataset(_Base):
    """Clean images + synthetic Gaussian noise generated on the fly.

    No paired download, instant iteration, and it shares its source images with
    the SR phase -- one DIV2K download covers Phase 1 and Phase 3. Noise is added
    after cropping and augmentation so that each epoch sees a fresh realisation,
    which is the standard protocol and also acts as regularisation.
    """

    def __init__(self, root, patch_size=128, sigmas=(15, 25, 50),
                 augment_data=True, repeat=1):
        super().__init__(patch_size, "denoise", augment_data, repeat)
        self.files = list_images(root)
        self.sigmas = list(sigmas)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        gt = imread(self.files[idx % len(self.files)])
        gt, _ = paired_random_crop(gt, gt, self.patch_size, scale=1)
        if self.augment_data:
            gt, = augment([gt])
        gt = gt.astype(np.float32) / 255.0
        sigma = random.choice(self.sigmas) / 255.0
        lq = gt + np.random.randn(*gt.shape).astype(np.float32) * sigma
        return self._pack(np.clip(lq, 0, 1), gt)


class PairedDataset(_Base):
    """Pre-paired degraded/clean folders: GoPro, Rain13K, SIDD crops.

    Filenames are matched by sorted order, which is how every one of these
    datasets ships. The length check catches a half-extracted download early
    rather than after six hours of training on misaligned pairs.
    """

    def __init__(self, lq_root, gt_root, task, patch_size=128,
                 augment_data=True, repeat=1):
        super().__init__(patch_size, task, augment_data, repeat)
        self.lq_files = list_images(lq_root)
        self.gt_files = list_images(gt_root)
        if len(self.lq_files) != len(self.gt_files):
            raise ValueError(
                f"pair count mismatch: {len(self.lq_files)} in {lq_root} vs "
                f"{len(self.gt_files)} in {gt_root}"
            )

    def __len__(self):
        return len(self.lq_files) * self.repeat

    def __getitem__(self, idx):
        i = idx % len(self.lq_files)
        lq, gt = imread(self.lq_files[i]), imread(self.gt_files[i])
        lq, gt = paired_random_crop(lq, gt, self.patch_size, scale=1)
        if self.augment_data:
            lq, gt = augment([lq, gt])
        return self._pack(lq, gt)


class SRDataset(_Base):
    """HR images with matching bicubic LR.

    `lr_root=None` generates LR on the fly with the MATLAB-compatible kernel.
    That is correct but costs CPU per sample; prefer the official DIV2K_train_LR_bicubic
    folders when they are on disk. `patch_size` is the LR patch size, so the GT
    patch is patch_size * scale.
    """

    def __init__(self, hr_root, lr_root=None, scale=4, patch_size=64,
                 augment_data=True, repeat=1):
        super().__init__(patch_size, "sr", augment_data, repeat)
        self.hr_files = list_images(hr_root)
        self.lr_files = list_images(lr_root) if lr_root else None
        if self.lr_files is not None and len(self.lr_files) != len(self.hr_files):
            raise ValueError("HR/LR file count mismatch")
        self.scale = scale

    def __len__(self):
        return len(self.hr_files) * self.repeat

    def __getitem__(self, idx):
        i = idx % len(self.hr_files)
        gt = imread(self.hr_files[i])
        if self.lr_files is not None:
            lq = imread(self.lr_files[i])
        else:
            t = torch.from_numpy(gt.transpose(2, 0, 1)).float() / 255.0
            lq = (imresize(t, scale=1.0 / self.scale).numpy()
                  .transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        # crop in LR coordinates; GT crop is scale times larger, same origin
        h, w = lq.shape[:2]
        gt = gt[:h * self.scale, :w * self.scale]
        lq, gt = paired_random_crop(lq, gt, self.patch_size, scale=self.scale)
        if self.augment_data:
            lq, gt = augment([lq, gt])
        return self._pack(lq, gt)


class TestSet(Dataset):
    """Full-image evaluation set. No cropping, no augmentation.

    For denoising the noise is seeded per index so the test set is byte-identical
    across runs and checkpoints -- otherwise validation PSNR wobbles by ~0.05 dB
    for reasons that have nothing to do with the model.
    """

    def __init__(self, gt_root, lq_root=None, task="denoise", sigma=25, scale=1):
        self.gt_files = list_images(gt_root)
        self.lq_files = list_images(lq_root) if lq_root else None
        self.task, self.sigma, self.scale = task, sigma, scale

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt = imread(self.gt_files[idx])
        if self.lq_files is not None:
            lq = imread(self.lq_files[idx])
        elif self.task == "denoise":
            rng = np.random.default_rng(seed=idx)
            g = gt.astype(np.float32) / 255.0
            lq = np.clip(g + rng.standard_normal(g.shape).astype(np.float32)
                         * (self.sigma / 255.0), 0, 1)
            return {"input": to_tensor(lq), "target": to_tensor(g),
                    "name": os.path.basename(self.gt_files[idx]),
                    "task_id": torch.tensor(TASK_IDS[self.task], dtype=torch.long)}
        elif self.task == "sr":
            t = torch.from_numpy(gt.transpose(2, 0, 1)).float() / 255.0
            # crop GT to a multiple of scale so LR and HR stay commensurate
            h = (t.shape[1] // self.scale) * self.scale
            w = (t.shape[2] // self.scale) * self.scale
            t = t[:, :h, :w]
            lq_t = imresize(t, scale=1.0 / self.scale)
            return {"input": lq_t, "target": t,
                    "name": os.path.basename(self.gt_files[idx]),
                    "task_id": torch.tensor(TASK_IDS[self.task], dtype=torch.long)}
        else:
            raise ValueError(f"task {self.task} needs an lq_root")
        return {"input": to_tensor(lq), "target": to_tensor(gt),
                "name": os.path.basename(self.gt_files[idx]),
                "task_id": torch.tensor(TASK_IDS[self.task], dtype=torch.long)}


class SyntheticDataset(_Base):
    """Clean images + a degradation sampled per patch, generated on the fly.

    Generalises GaussianDenoiseDataset to any mix of noise or blur operators.
    Used for the widened Phase 1 (all noise types) and the synthetic arm of
    Phase 2 (general blur, not just motion).

    `degradations` is a list of specs, one of which is sampled per patch:

        [{"kind": "gaussian", "sigma": [5, 50], "weight": 2},
         {"kind": "speckle",  "sigma": [10, 60]},
         {"kind": "poisson",  "peak":  [5, 60]}]

        [{"kind": "gaussian", "sigma":  [0.5, 3.0]},
         {"kind": "defocus",  "radius": [1.0, 5.0]},
         {"kind": "motion",   "length": [5, 25], "angle": [0, 180]}]

    Ranges are sampled uniformly, so severity is a continuous variable rather
    than a fixed ladder -- which is what lets Phase 2 vary blur strength as a
    controlled input to the gate measurement.

    `family` selects the operator table: "noise" or "blur". They are kept
    separate rather than auto-detected because `{"kind": "gaussian"}` is a valid
    spec in both and means completely different things.
    """

    def __init__(self, root, degradations, family="noise", task="denoise",
                 patch_size=128, augment_data=True, repeat=1):
        super().__init__(patch_size, task, augment_data, repeat)
        if family not in ("noise", "blur"):
            raise ValueError(f"family must be 'noise' or 'blur', got {family!r}")
        if not degradations:
            raise ValueError("degradations list is empty")
        self.files = list_images(root)
        self.degradations = list(degradations)
        self.family = family
        self.weights = [d.get("weight", 1.0) for d in self.degradations]

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        from .degradations import apply_noise, apply_kernel, sample_blur_kernel

        gt = imread(self.files[idx % len(self.files)])
        gt, _ = paired_random_crop(gt, gt, self.patch_size, scale=1)
        if self.augment_data:
            gt, = augment([gt])
        gt = gt.astype(np.float32) / 255.0

        spec = random.choices(self.degradations, weights=self.weights, k=1)[0]
        rng = np.random.default_rng()
        if self.family == "noise":
            lq = apply_noise(gt, spec, rng)
        else:
            lq = apply_kernel(gt, sample_blur_kernel(spec, rng))

        return self._pack(np.clip(lq, 0, 1), gt)
