"""Exponential moving average of weights.

Restoration models are evaluated from EMA weights almost universally; on small
datasets like GoPro (2,103 pairs) the EMA is worth a few tenths of a dB on its
own and materially reduces run-to-run variance, which matters when the effect
being measured is itself small.
"""
import copy

import torch


class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.module = copy.deepcopy(self._unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") and hasattr(model, "_ddp_params_and_buffers_to_ignore") else model

    @torch.no_grad()
    def update(self, model, step=None):
        d = self.decay
        if step is not None:
            # warm the EMA up so early checkpoints are not dominated by the
            # random initialisation still sitting in the shadow weights
            d = min(d, (1 + step) / (10 + step))
        msd = self._unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.device), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
