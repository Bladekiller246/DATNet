"""Config -> DataLoader."""
import os

import torch
from torch.utils.data import DataLoader

from .datasets import (GaussianDenoiseDataset, PairedDataset, SRDataset,
                       SyntheticDataset, TestSet)


def build_train_dataset(cfg):
    kind = cfg["kind"]
    if kind == "gaussian_denoise":
        return GaussianDenoiseDataset(
            cfg["root"], patch_size=cfg.get("patch_size", 128),
            sigmas=cfg.get("sigmas", (15, 25, 50)), repeat=cfg.get("repeat", 1))
    if kind == "synthetic":
        return SyntheticDataset(
            cfg["root"], cfg["degradations"], family=cfg.get("family", "noise"),
            task=cfg.get("task", "denoise"), patch_size=cfg.get("patch_size", 128),
            repeat=cfg.get("repeat", 1))
    if kind == "paired":
        return PairedDataset(
            cfg["lq_root"], cfg["gt_root"], cfg["task"],
            patch_size=cfg.get("patch_size", 128), repeat=cfg.get("repeat", 1))
    if kind == "sr":
        return SRDataset(
            cfg["hr_root"], cfg.get("lr_root"), scale=cfg.get("scale", 4),
            patch_size=cfg.get("patch_size", 64), repeat=cfg.get("repeat", 1))
    raise ValueError(f"unknown dataset kind {kind!r}")


def build_test_dataset(cfg):
    return TestSet(cfg["gt_root"], cfg.get("lq_root"), task=cfg.get("task", "denoise"),
                   sigma=cfg.get("sigma", 25), scale=cfg.get("scale", 1))


def make_loader_factory(cfg):
    """Returns a callable(micro_batch) -> DataLoader.

    A factory rather than a loader because the trainer rebuilds the loader when
    it backs off after an OOM. `num_workers` defaults to 4: Windows uses spawn,
    so each worker is a full process holding its own torch import (~300-400 MB
    RSS), and this machine has 15.4 GB of system RAM shared with everything else.
    """
    dataset = build_train_dataset(cfg)
    workers = cfg.get("num_workers", 4)

    def factory(micro_batch):
        return DataLoader(
            dataset,
            batch_size=micro_batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=workers > 0,
            prefetch_factor=cfg.get("prefetch_factor", 2) if workers > 0 else None,
        )

    return factory, dataset
