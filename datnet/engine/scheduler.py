"""Per-iteration cosine schedule with linear warmup."""
import math


class CosineWarmup:
    def __init__(self, optimizer, total_iters, warmup_iters=5000, base_lr=3e-4, min_lr=1e-6):
        self.opt = optimizer
        self.total, self.warmup = total_iters, warmup_iters
        self.base_lr, self.min_lr = base_lr, min_lr
        self.last_iter = -1

    def lr_at(self, it):
        if it < self.warmup:
            return self.base_lr * (it + 1) / max(1, self.warmup)
        prog = (it - self.warmup) / max(1, self.total - self.warmup)
        prog = min(max(prog, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * prog))

    def step(self, it):
        lr = self.lr_at(it)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.last_iter = it
        return lr

    def state_dict(self):
        return {"last_iter": self.last_iter}

    def load_state_dict(self, sd):
        self.last_iter = sd["last_iter"]
