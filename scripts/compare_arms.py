"""The Phase-1 verdict: does `dual` beat both single-axis arms at matched params?

This is the kill switch. If `dual` does not beat `max(channel_only, window_only)`
at matched parameter count, the core hypothesis is in trouble and the project
pivots to the all-in-one-including-SR contribution instead.

Evaluates every arm identically -- same test sets, same sigmas, same protocol,
EMA weights throughout -- and prints the margin. Also checks the thing that makes
the comparison legitimate at all: that the arms really are parameter-matched.

Usage:
    python scripts/compare_arms.py --phase phase1_denoise
    python scripts/compare_arms.py --run-dirs runs/a runs/b runs/c
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datnet.data.datasets import TestSet
from datnet.evaluation.evaluate import evaluate

ARMS = ["dual", "channel_only", "window_only"]


def load_arm(run_dir, device, sigmas, test_sets):
    """Evaluate one arm; returns None with a reason if it is not usable."""
    from scripts.report_phase import load_model  # reuse the exact loading path

    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {"error": "no config.json -- run never started"}

    try:
        model, meta, ck, used = load_model(run_dir, "best", device)
    except Exception as e:
        return {"error": f"could not load checkpoint: {type(e).__name__}: {e}"}

    finite = all(torch.isfinite(p).all() for p in model.parameters())
    if not finite:
        return {"error": "checkpoint weights are non-finite (diverged)"}

    out = {
        "iteration": ck["iteration"],
        "checkpoint": used,
        "params": sum(p.numel() for p in model.parameters()),
        "width": meta["model_cfg"].get("width"),
        "ffn_expansion": meta["model_cfg"].get("ffn_expansion"),
        "test": {},
    }
    for name, root in test_sets:
        if not os.path.isdir(root):
            continue
        out["test"][name] = {}
        for sigma in sigmas:
            ds = TestSet(root, task="denoise", sigma=sigma)
            m = evaluate(model, ds, "denoise", device=device, tile=256)
            out["test"][name][sigma] = m
            print(f"    {name:9s} sigma={sigma:<3d} {m['psnr']:.3f} dB  "
                  f"SSIM {m['ssim']:.4f}")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="phase1_denoise")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dirs", nargs="*", default=None)
    ap.add_argument("--sigmas", type=int, nargs="*", default=[15, 25, 50])
    ap.add_argument("--headline-sigma", type=int, default=25)
    ap.add_argument("--headline-set", default="CBSD68")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    test_sets = [("CBSD68", "data/test/CBSD68"), ("McMaster", "data/test/McMaster")]

    if args.run_dirs:
        dirs = {os.path.basename(d): d for d in args.run_dirs}
    else:
        dirs = {a: os.path.join(args.run_root, f"{args.phase}__{a}__seed{args.seed}")
                for a in ARMS}

    results = {}
    for arm, d in dirs.items():
        print(f"\n{arm}  ({d})")
        if not os.path.isdir(d):
            results[arm] = {"error": "run directory does not exist"}
            print("    MISSING")
            continue
        results[arm] = load_arm(d, args.device, args.sigmas, test_sets)
        if "error" in results[arm]:
            print(f"    {results[arm]['error']}")

    ok = {a: r for a, r in results.items() if "error" not in r}

    # ---- parameter matching, the precondition for any of this meaning anything
    print("\n" + "=" * 62)
    print("parameter matching")
    match_ok = True
    if len(ok) > 1:
        counts = {a: r["params"] for a, r in ok.items()}
        ref = counts.get("dual", list(counts.values())[0])
        for a, n in counts.items():
            err = abs(n - ref) / ref
            flag = "" if err <= 0.02 else "   <-- OUT OF TOLERANCE"
            match_ok &= err <= 0.02
            print(f"  {a:14s} width={ok[a]['width']:<4} "
                  f"gamma={ok[a]['ffn_expansion']:<7} {n:>10,}  "
                  f"({err:+.2%}){flag}")
        if not match_ok:
            print("  WARNING: arms are not parameter-matched; the comparison is "
                  "confounded and the verdict below is not trustworthy.")

    # ---- the verdict
    print("\n" + "=" * 62)
    hs, hset = args.headline_sigma, args.headline_set
    print(f"VERDICT  ({hset}, sigma={hs}, EMA weights, RGB PSNR)")
    scores = {}
    for a, r in ok.items():
        v = r["test"].get(hset, {}).get(hs)
        if v:
            scores[a] = v["psnr"]
            print(f"  {a:14s} {v['psnr']:.3f} dB   (iteration {r['iteration']:,})")

    verdict = None
    if "dual" in scores and len(scores) >= 2:
        singles = {a: s for a, s in scores.items() if a != "dual"}
        best_single = max(singles, key=singles.get)
        margin = scores["dual"] - singles[best_single]
        print(f"\n  dual - best single ({best_single}) = {margin:+.3f} dB")
        if margin > 0:
            verdict = "PASS"
            print("  PASS: dual beats both single-axis arms at matched parameters.")
            print("  The core hypothesis survives Phase 1. Proceed to Phase 2,")
            print("  which is where the gate-flip figure is actually decided.")
        else:
            verdict = "FAIL"
            print("  FAIL: dual does NOT beat the best single-axis arm.")
            print("  Before concluding, rerun the best config at 2x iterations --")
            print("  dual has two attention paths and may simply converge slower.")
            print("  If it still fails, pivot to the all-in-one-including-SR story,")
            print("  which does not depend on dual winning any single task.")
        if abs(margin) < 0.05:
            print("\n  NOTE: margin is under 0.05 dB, which is within run-to-run")
            print("  noise for restoration. Treat this as inconclusive, not a win,")
            print("  and repeat with a second seed before believing it.")

    payload = {"phase": args.phase, "headline": {"set": hset, "sigma": hs},
               "verdict": verdict, "parameter_match_ok": match_ok,
               "scores": scores, "arms": results}
    out = args.out or os.path.join("results", f"{args.phase}__comparison.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
