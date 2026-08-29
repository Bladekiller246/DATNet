"""Phase-5 multi-task sampling.

Two problems have to be solved together:

1. **Shape.** SR batches are (B,3,64,64) -> (B,3,256,256) while the other tasks
   are (B,3,128,128) -> (B,3,128,128). Those cannot be collated into one tensor,
   so batches are task-homogeneous: each micro-batch is drawn from a single
   task, and gradient accumulation mixes tasks within an optimiser step. With
   accum=4 every step already averages four task draws.

2. **Scale.** Rain13K has 13.7k pairs and GoPro has 2.1k, so uniform sampling
   drowns deblurring; sqrt-proportional sampling is the usual compromise between
   uniform (ignores data volume) and proportional (ignores small tasks).
   Separately, denoising sits near 40 dB while SR sits near 27 dB, so raw L1
   magnitudes differ by roughly an order of magnitude. Each task loss is divided
   by a running mean of its own magnitude before summation.
"""
import math
import random

import torch
from torch.utils.data import DataLoader


def sqrt_proportional_weights(sizes):
    roots = {k: math.sqrt(v) for k, v in sizes.items()}
    total = sum(roots.values())
    return {k: v / total for k, v in roots.items()}


class MultiTaskLoader:
    """Yields task-homogeneous micro-batches from several DataLoaders."""

    def __init__(self, loaders, weights=None, seed=0):
        self.loaders = loaders
        self.names = list(loaders)
        sizes = {k: len(v.dataset) for k, v in loaders.items()}
        self.weights = weights or sqrt_proportional_weights(sizes)
        self.probs = [self.weights[n] for n in self.names]
        self._iters = {}
        self._rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        name = self._rng.choices(self.names, weights=self.probs, k=1)[0]
        if name not in self._iters:
            self._iters[name] = iter(self.loaders[name])
        try:
            batch = next(self._iters[name])
        except StopIteration:
            self._iters[name] = iter(self.loaders[name])
            batch = next(self._iters[name])
        batch["task_name"] = name
        return batch

    def state_dict(self):
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, sd):
        self._rng.setstate(sd["rng"])


class LossBalancer:
    """Divides each task loss by a running mean of its own magnitude.

    Without this the joint gradient is dominated by whichever task happens to
    have the largest raw L1, which is SR, and denoising quietly stops improving.
    The running mean is detached, so this rescales gradients without adding a
    term the optimiser can game.
    """

    def __init__(self, tasks, momentum=0.99, eps=1e-8, warmup=100):
        self.mean = {t: None for t in tasks}
        self.count = {t: 0 for t in tasks}
        self.momentum, self.eps, self.warmup = momentum, eps, warmup

    def scale(self, task, loss):
        v = float(loss.detach())
        m = self.mean[task]
        self.mean[task] = v if m is None else self.momentum * m + (1 - self.momentum) * v
        self.count[task] += 1
        if self.count[task] < self.warmup:
            return loss  # let the estimate settle before rescaling anything
        return loss / (self.mean[task] + self.eps)

    def state_dict(self):
        return {"mean": self.mean, "count": self.count}

    def load_state_dict(self, sd):
        self.mean, self.count = sd["mean"], sd["count"]
