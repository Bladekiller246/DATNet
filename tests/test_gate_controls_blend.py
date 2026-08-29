"""The gate must be the ONLY thing that sets the channel/spatial balance.

Phase 1 failed this without anyone checking. `out = g*A + (1-g)*B` with a
learnable 1x1 projection at the end of each branch lets the network rescale A
and B to produce any effective blend it likes, leaving `g` parked at its
initialisation. Measured on the real 30k run: mean g moved 0.005 in 30,000
iterations while the true |A|/|B| ratio sat at 0.28 -- so a block reporting
"80% channel" was actually running ~47/53, and one was 8x spatial.

That makes the axis-preference figure -- the entire contribution of the project
-- a plot of a number that means nothing. These tests exist so it cannot happen
again silently.

Run:  python tests/test_gate_controls_blend.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datnet.models.block import DATB

BASE = dict(dim=32, num_heads=2, mode="dual", window_size=8)


def ok(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return passed


def test_effective_ratio_tracks_gate():
    """With normalisation on, the measured blend must follow g."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, 64)
    passed = True
    for g_target in (0.1, 0.5, 0.9):
        blk = DATB(**BASE, gate_init=g_target, normalise_branches=True).eval()
        with torch.no_grad():
            r = blk.effective_ratio(x)
        close = abs(r - g_target) < 0.12
        passed &= ok(f"effective ratio follows g={g_target}", close, f"measured {r:.3f}")
    return passed


def test_branch_rescaling_cannot_fool_the_gate():
    """Blow up one branch 10x; the effective ratio must barely move.

    This is the exact escape route the old architecture left open.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, 64)

    blk = DATB(**BASE, gate_init=0.5, normalise_branches=True).eval()
    with torch.no_grad():
        before = blk.effective_ratio(x)
        blk.channel_attn.proj.weight.mul_(10.0)   # make branch A 10x louder
        after = blk.effective_ratio(x)
    drift = abs(after - before)
    passed = ok("10x branch rescale does not shift the blend (normalised)",
                drift < 0.05, f"{before:.3f} -> {after:.3f}  drift {drift:.4f}")

    # and show the OLD behaviour fails, so the test is proving something
    blk2 = DATB(**BASE, gate_init=0.5, normalise_branches=False).eval()
    with torch.no_grad():
        b0 = blk2.effective_ratio(x)
        blk2.channel_attn.proj.weight.mul_(10.0)
        b1 = blk2.effective_ratio(x)
    old_drift = abs(b1 - b0)
    passed &= ok("un-normalised control DOES drift (proves the test bites)",
                 old_drift > 0.2, f"{b0:.3f} -> {b1:.3f}  drift {old_drift:.4f}")
    return passed


def test_gate_still_identity_at_extremes():
    """g=1 must use only the channel branch, g=0 only the spatial branch."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, 64)
    blk = DATB(**BASE, gate_init=0.5, normalise_branches=True).eval()
    with torch.no_grad():
        blk.gate.w.fill_(20.0)          # sigmoid(20) == 1
        r1 = blk.effective_ratio(x)
        blk.gate.w.fill_(-20.0)         # sigmoid(-20) == 0
        r0 = blk.effective_ratio(x)
    return ok("g=1 -> ratio 1, g=0 -> ratio 0", r1 > 0.999 and r0 < 0.001,
              f"{r1:.4f} / {r0:.4f}")


def test_gradient_reaches_gate():
    torch.manual_seed(0)
    blk = DATB(**BASE, gate_init=0.5, normalise_branches=True)
    x = torch.randn(2, 32, 64, 64)
    blk(x).pow(2).mean().backward()
    g = blk.gate.w.grad
    return ok("gate receives gradient", g is not None and g.abs().mean() > 0,
              f"mean |grad| = {g.abs().mean():.3e}")


if __name__ == "__main__":
    results = [
        test_effective_ratio_tracks_gate(),
        test_branch_rescaling_cannot_fool_the_gate(),
        test_gate_still_identity_at_extremes(),
        test_gradient_reaches_gate(),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
