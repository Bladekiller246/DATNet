"""Degradation conditioning for the Phase-5 all-in-one model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OracleEncoder(nn.Module):
    """d = one-hot(task_id). Upper bound on what routing can buy."""

    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

    def forward(self, x, task_id=None):
        if task_id is None:
            raise ValueError("OracleEncoder needs task_id")
        return F.one_hot(task_id, self.num_tasks).float(), None


class BlindEncoder(nn.Module):
    """Small conv head that predicts the degradation from the input image.

    Returns (soft task posterior, logits). The logits feed an auxiliary
    cross-entropy loss so the head is supervised even though the gate could in
    principle learn to ignore it. This is the realistic setting -- the one
    PromptIR and AirNet operate in.
    """

    def __init__(self, num_tasks, width=32, in_ch=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 2, 1), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, 2, 1), nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 4, num_tasks)

    def forward(self, x, task_id=None):
        logits = self.head(self.body(x).flatten(1))
        return logits.softmax(dim=-1), logits


def build_encoder(kind, num_tasks, **kw):
    if kind == "oracle":
        return OracleEncoder(num_tasks)
    if kind == "blind":
        return BlindEncoder(num_tasks, **kw)
    raise ValueError(f"unknown conditioning kind {kind!r}")


class ConditionedDATNet(nn.Module):
    """DATNet whose axis gates are driven by a predicted degradation embedding."""

    def __init__(self, backbone, encoder):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder

    def forward(self, x, task_id=None):
        cond, logits = self.encoder(x, task_id)
        return self.backbone(x, cond), logits
