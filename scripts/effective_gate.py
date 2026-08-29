"""Report the EFFECTIVE channel/spatial balance, not the nominal gate value.

`g` is what the gate parameter says. The effective ratio

    |g*A| / (|g*A| + |(1-g)*B|)

is what the block actually does. With `normalise_branches=True` these agree by
construction. Without it they can disagree wildly -- on the 30k Phase-1 run a
block reporting g=0.80 was running an effective 0.87 ratio (~47/53), and one
reporting g=0.80 was 8x spatial.

Always report the effective ratio. It is the honest quantity, it is comparable
across runs with different architectures, and it is what the axis-preference
figure should plot.

Usage:
    python scripts/effective_gate.py --run-dir runs/phase1_denoise__dual__seed0
    python scripts/effective_gate.py --run-dir ... --out results/x/effective.json
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rp", os.path.join(_HERE, "report_phase.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

from datnet.data.datasets import TestSet   # noqa: E402
from datnet.models.block import DATB       # noqa: E402


def measure(model, x, cond=None):
    """Per-block nominal g and effective ratio, in signal order."""
    out = {}

    def make_hook(name):
        def f(mod, inp, _out):
            if mod.mode != "dual":
                return
            g = mod.gate(cond).mean().item()
            out[name] = {"g": g, "effective": mod.effective_ratio(inp[0], cond)}
        return f

    hooks = [m.register_forward_hook(make_hook(n))
             for n, m in model.named_modules() if isinstance(m, DATB)]
    with torch.no_grad():
        model(x, cond) if cond is not None else model(x)
    for h in hooks:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--test-root", default="data/test/CBSD68")
    ap.add_argument("--sigma", type=int, default=25)
    ap.add_argument("--images", type=int, default=4,
                    help="average over this many test images")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, meta, ck, used = rp.load_model(args.run_dir, args.checkpoint, args.device)
    if meta["model_cfg"].get("mode") != "dual":
        sys.exit(f"{meta['model_cfg'].get('mode')} has no gate; nothing to report")

    ds = TestSet(args.test_root, task="denoise", sigma=args.sigma)
    n = min(args.images, len(ds))
    acc = {}
    for i in range(n):
        x = ds[i]["input"].unsqueeze(0).to(args.device)
        for k, v in measure(model, x).items():
            a = acc.setdefault(k, {"g": 0.0, "effective": 0.0})
            a["g"] += v["g"] / n
            a["effective"] += v["effective"] / n

    normalised = meta["model_cfg"].get("normalise_branches", False)
    print(f"{os.path.basename(args.run_dir)}   {used} @ iteration {ck['iteration']:,}")
    print(f"normalise_branches={normalised}   averaged over {n} images "
          f"({os.path.basename(args.test_root)}, sigma={args.sigma})\n")
    print(f"{'block':24s} {'nominal g':>10} {'effective':>10} {'delta':>8}")
    print("-" * 56)
    worst = 0.0
    for k, v in acc.items():
        d = v["effective"] - v["g"]
        worst = max(worst, abs(d))
        print(f"{k:24s} {v['g']:>10.4f} {v['effective']:>10.4f} {d:>+8.4f}")

    mean_eff = sum(v["effective"] for v in acc.values()) / max(len(acc), 1)
    print(f"\n  mean effective ratio = {mean_eff:.4f}  "
          f"(1.0 = pure channel, 0.0 = pure spatial)")
    print(f"  largest |effective - g| = {worst:.4f}")
    if worst > 0.05:
        print("\n  WARNING: the gate does not describe the real balance. Do not")
        print("  plot g. Either the branches are un-normalised, or something else")
        print("  is letting the network shift the blend behind the gate.")

    out = args.out or os.path.join("results", os.path.basename(args.run_dir.rstrip("/\\")),
                                   "effective_gate.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"run": args.run_dir, "checkpoint": used,
                   "iteration": ck["iteration"], "normalise_branches": normalised,
                   "sigma": args.sigma, "images": n,
                   "mean_effective": mean_eff, "max_abs_delta": worst,
                   "blocks": acc}, f, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
