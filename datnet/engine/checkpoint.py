"""Atomic, fully-resumable checkpointing.

A checkpoint captures everything needed to make segment N+1 indistinguishable
from having never stopped: weights, EMA shadow, optimiser moments, schedule
position, AMP scaler state, and all four RNG streams. Without the RNG state a
resume re-draws the same augmentations it already saw, which quietly biases long
runs that are stopped and started many times -- exactly the regime this project
runs in on a laptop.
"""
import os
import random
import tempfile

import torch


def _rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _load_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, ema, iteration,
                    config, best=None, extra=None):
    payload = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "rng": _rng_state(),
        "config": config,
        "best": best,
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # write-then-rename: a power cut mid-save must not destroy the last good ckpt
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    os.close(fd)
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path, *, model, optimizer=None, scheduler=None, scaler=None,
                    ema=None, map_location="cpu", restore_rng=True, strict=True):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"], strict=strict)
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if ema is not None and ck.get("ema"):
        ema.load_state_dict(ck["ema"])
    if restore_rng:
        _load_rng(ck.get("rng"))
    return ck
