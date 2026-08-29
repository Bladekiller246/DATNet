"""Phase-5 all-in-one model: one backbone, one checkpoint, four tasks.

The plan leaves one architectural question open, and it has to be answered here:
Phases 1-4 use a 3-level U-Net for the same-size tasks and a flat trunk for SR,
which are two different backbones. A single all-in-one checkpoint cannot have
two backbones, so the joint model uses the **U-Net backbone for everything** and
switches only the tail.

That is viable because the SR input is the LR image: at a 96x96 LR patch the L3
feature map is 24x24, comfortably above the 8x8 window minimum. The concern that
motivated the flat topology in Phase 3 -- a 48x48 patch collapsing to 6x6 -- is
avoided by training SR at a larger LR patch in the joint phase. That is a real
cost (SR sees fewer, larger patches per step) and it should be reported as a
deliberate design choice, not hidden.

Being explicit about this matters for the comparison: the Phase-5 SR column is
NOT the Phase-3 architecture, so any gap between them mixes "all-in-one is hard"
with "the backbone changed". Run a flat-vs-unet single-task SR control before
attributing the gap to joint training.
"""
import torch
import torch.nn as nn

from .conditioning import build_encoder
from .datnet import DATNet, RestorationTail, SRTail


class AllInOneDATNet(nn.Module):
    """Shared dual-axis backbone + per-task tail + degradation-conditioned gates."""

    def __init__(self, backbone_cfg, tasks=("denoise", "deblur", "derain", "sr"),
                 sr_scale=4, conditioning="oracle", encoder_kwargs=None):
        super().__init__()
        self.tasks = list(tasks)
        self.task_index = {t: i for i, t in enumerate(self.tasks)}
        num_tasks = len(self.tasks)

        cfg = dict(backbone_cfg)
        cfg.update(topology="unet", task="restoration", scale=1,
                   cond_dim=num_tasks)
        self.backbone = DATNet(**cfg)
        width = cfg.get("width", 32)

        self.tails = nn.ModuleDict()
        for t in self.tasks:
            self.tails[t] = (SRTail(width, sr_scale) if t == "sr"
                             else RestorationTail(width))
        self.sr_scale = sr_scale
        self.encoder = build_encoder(conditioning, num_tasks,
                                     **(encoder_kwargs or {}))
        self.cond_dim = num_tasks

    def forward(self, x, task_id=None, task_name=None):
        """Batches are task-homogeneous (see data/multitask.py), so one tail per call."""
        cond, logits = self.encoder(x, task_id)
        h, w = x.shape[-2:]
        feat, padded = self.backbone.forward_features(x, cond)

        if task_name is None:
            if task_id is None:
                raise ValueError("need task_name or task_id to select a tail")
            ids = task_id.unique()
            if ids.numel() != 1:
                raise ValueError(
                    f"mixed-task batch {ids.tolist()}; batches must be "
                    "task-homogeneous so a single tail applies"
                )
            task_name = self.tasks[int(ids.item())]

        out = self.tails[task_name](feat, padded)
        s = self.sr_scale if task_name == "sr" else 1
        return out[..., : h * s, : w * s], logits

    @torch.no_grad()
    def gate_report(self, cond=None):
        return self.backbone.gate_report(cond)

    @torch.no_grad()
    def per_task_gates(self, device="cuda"):
        """{task: {block: gate}} -- the Phase-5 version of the axis-preference figure."""
        eye = torch.eye(len(self.tasks), device=device)
        return {t: self.backbone.gate_report(eye[i:i + 1])
                for i, t in enumerate(self.tasks)}
