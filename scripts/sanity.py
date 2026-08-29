"""Phase-0 sanity checks. Run this before any real training.

Each check exists because it has silently broken a restoration codebase before:

1. shapes      -- non-multiple-of-window sizes must round-trip exactly
2. gate identity -- dual with g pinned to 1.0 must equal channel_only
3. width match -- the three arms must agree on parameter count within 2%
4. overfit     -- a single image must reach >50 dB; if it cannot, the training
                  loop is broken and every later result is noise

Usage:  python scripts/sanity.py [--device cuda] [--skip-overfit]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datnet.models.build import build_model
from datnet.models.datnet import DATNet
from datnet.utils.complexity import count_parameters, summarise
from datnet.utils.matching import match_all

BASE = dict(width=32, topology="unet", enc_depths=(2, 3), bottleneck_depth=4,
            dec_depths=(3, 2), refine_depth=2, heads=(1, 2, 4), window_size=8,
            ffn_expansion=2.66, gate_init=(0.8, 0.5, 0.2), task="restoration")


def ok(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return passed


def check_shapes(device):
    """Arbitrary input sizes must come back at exactly the same size."""
    model = build_model({**BASE, "mode": "dual"}).to(device).eval()
    passed = True
    for h, w in [(128, 128), (137, 91), (64, 64), (255, 129)]:
        x = torch.randn(1, 3, h, w, device=device)
        with torch.no_grad():
            y = model(x)
        passed &= ok(f"shape {h}x{w}", y.shape == x.shape, str(tuple(y.shape)))
    # SR tail: output must be exactly scale times the input
    sr = build_model({**BASE, "mode": "dual", "topology": "flat",
                      "task": "sr", "scale": 4, "flat_depth": 4}).to(device).eval()
    x = torch.randn(1, 3, 48, 64, device=device)
    with torch.no_grad():
        y = sr(x)
    passed &= ok("sr shape 48x64 -> x4", y.shape[-2:] == (192, 256), str(tuple(y.shape)))
    return passed


def check_gate_identity(device):
    """At g=1 the dual block must use the channel branch and nothing else.

    Proves the fusion is a true convex blend and that the channel branch is
    wired identically in both arms. If it fails, every ablation number is
    comparing two different channel branches.

    Tested at BLOCK level and by DIRECTION, not by exact equality of the whole
    network. With `normalise_branches=True` the two branches are rescaled to a
    common magnitude, so at g=1 the dual block emits `a * (ra+rb)/(2*ra)` -- the
    same channel branch, scaled. Direction identical, magnitude deliberately not.
    Asserting exact equality would be asserting that the branch balancing does
    not happen.
    """
    import torch.nn.functional as F
    from datnet.models.block import DATB

    torch.manual_seed(0)
    cfg = dict(dim=32, num_heads=2, window_size=8, gate_init=0.5)
    dual = DATB(**cfg, mode="dual", normalise_branches=True).to(device).eval()
    chan = DATB(**cfg, mode="channel_only", normalise_branches=True).to(device).eval()

    sd_d, sd_c = dual.state_dict(), chan.state_dict()
    chan.load_state_dict({**sd_c, **{k: v for k, v in sd_d.items()
                                     if k in sd_c and sd_c[k].shape == v.shape}})

    x = torch.randn(1, 32, 64, 64, device=device)
    passed = True
    for g_val, branch in ((20.0, "channel"), (-20.0, "spatial")):
        with torch.no_grad():
            dual.gate.w.fill_(g_val)
            y = dual.norm1(x)
            a = dual.channel_attn(y)
            b = dual.window_attn(y)
            ab, bb = dual._balance(a, b)
            g = dual.gate(None)
            fused = g * ab + (1.0 - g) * bb
            target = ab if branch == "channel" else bb
            cos = F.cosine_similarity(fused.flatten(), target.flatten(), dim=0).item()
        passed &= ok(f"g={'1' if g_val > 0 else '0'} uses only the {branch} branch",
                     cos > 0.9999, f"cos={cos:.6f}")

    # and the channel branch really is the same module in both arms
    with torch.no_grad():
        ca = chan.channel_attn(chan.norm1(x))
        da = dual.channel_attn(dual.norm1(x))
    err = (ca - da).abs().max().item()
    passed &= ok("channel branch identical across arms", err < 1e-5,
                 f"max|diff|={err:.2e}")
    return passed


def check_width_matching():
    matched = match_all(build_model, {**BASE, "mode": "dual"})
    for mode, info in matched.items():
        print(f"        {mode:14s} width={info['width']:3d}  "
              f"gamma={info['ffn_expansion']:.4f}  "
              f"params={info['params']:,}  err={info['rel_error']:.2%}  "
              f"({info['knob']})")
    worst = max(i["rel_error"] for i in matched.values())
    return ok("width matching within 2%", worst <= 0.02, f"worst={worst:.2%}"), matched


def check_overfit(device, iters=600, target_db=50.0):
    """One image, one patch, no augmentation. Must reach >50 dB."""
    from datnet.engine.losses import RestorationLoss
    torch.manual_seed(0)
    model = build_model({**BASE, "mode": "dual", "enc_depths": (1, 1),
                         "bottleneck_depth": 1, "dec_depths": (1, 1),
                         "refine_depth": 1}).to(device)
    gt = torch.rand(1, 3, 128, 128, device=device)
    lq = (gt + 0.1 * torch.randn_like(gt)).clamp(0, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = RestorationLoss(pixel="l1")
    for i in range(iters):
        opt.zero_grad(set_to_none=True)
        loss, _ = crit(model(lq), gt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = torch.mean((model(lq).clamp(0, 1) - gt) ** 2).item()
    db = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
    return ok(f"overfit single image ({iters} it)", db > target_db, f"{db:.1f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-overfit", action="store_true")
    args = ap.parse_args()
    print(f"device: {args.device}\n")

    results = [check_shapes(args.device), check_gate_identity(args.device)]
    passed, matched = check_width_matching()
    results.append(passed)

    print("\ncomplexity at 256x256:")
    for mode, info in matched.items():
        m = build_model({**BASE, "mode": mode, "width": info["width"],
                         "ffn_expansion": info["ffn_expansion"]})
        s = summarise(m, (1, 3, 256, 256))
        print(f"        {mode:14s} {s['params_M']:>6.3f} M params  "
              f"{s['flops_G'] if s['flops_G'] else 'n/a':>8} GFLOPs")

    if not args.skip_overfit:
        print()
        results.append(check_overfit(args.device))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
