"""Produce the finished-run report: real numbers, the gate figure, sample images.

Runs after a training run completes. Does NOT start any training.

Outputs into `results/<run_name>/`:
  summary.md          the numbers, in a form you can read at a glance
  gates.png           mean gate value per block -- the axis-preference figure
  samples/*.png       noisy | restored | ground-truth triptychs
  metrics.json        machine-readable

Evaluation uses the **EMA weights** and the per-task protocol from
`datnet/evaluation/protocols.py` (RGB, no border shave, for denoising).

Usage:
    python scripts/report_phase.py --run-dir runs/phase1_denoise__dual__seed0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from datnet.data.datasets import TestSet
from datnet.data.io import imwrite
from datnet.evaluation.evaluate import evaluate
from datnet.evaluation.inference import restore
from datnet.models.build import build_model


def load_model(run_dir, which="best", device="cuda"):
    """Rebuild the model from the run's own config and load its EMA weights.

    The checkpoint carries its own model_cfg, so this never depends on the YAML
    still saying what it said when the run started.
    """
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        meta = json.load(f)
    ckpt_path = os.path.join(run_dir, f"{which}.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "last.pth")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("ema") or ck["model"]

    # Honour the architecture the checkpoint was actually TRAINED with, not the
    # current default. normalise_branches is parameter-free -- it leaves no
    # `attn_scale` tensor or any other trace in the state dict, so it CANNOT be
    # detected from weights (an earlier version of this function tried exactly
    # that and was wrong for every checkpoint trained after the option was
    # added: it silently forced normalise_branches=False on all of them,
    # dropping a real 60k dual run from 31.2 dB to 26.7 dB in evaluation only --
    # the trained weights were fine). train.py now always records the resolved
    # value explicitly, so trust it. Only checkpoints from before that recording
    # started have no key at all; for those, False matches how they actually
    # trained.
    cfg = dict(meta["model_cfg"])
    if "normalise_branches" not in cfg:
        cfg["normalise_branches"] = False
    meta = {**meta, "model_cfg": cfg}

    model = build_model(cfg)
    model.load_state_dict(state)
    return model.to(device).eval(), meta, ck, os.path.basename(ckpt_path)


def save_samples(model, dataset, out_dir, n=4, device="cuda", scale=1):
    """Write noisy | restored | ground-truth strips so the result is visible."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        x = item["input"].unsqueeze(0).to(device)
        y = item["target"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            pred = restore(model, x, scale=scale, tile=256)
        pred = pred.float().clamp(0, 1)[..., : y.shape[-2], : y.shape[-1]]

        trio = [t[0].permute(1, 2, 0).cpu().numpy()
                for t in (x.float(), pred, y.float())]
        gap = np.ones((trio[0].shape[0], 8, 3), dtype=np.float32)
        strip = np.concatenate(
            [trio[0], gap, trio[1], gap, trio[2]], axis=1)
        name = item.get("name", f"{i:03d}.png")
        path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_triptych.png")
        imwrite(path, strip)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sigmas", type=int, nargs="*", default=[15, 25, 50])
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    run_name = os.path.basename(args.run_dir.rstrip("/\\"))
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    model, meta, ck, ckpt_used = load_model(args.run_dir, "best", args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{run_name}\n  checkpoint {ckpt_used}  iteration {ck['iteration']:,}"
          f"  params {n_params/1e6:.3f} M")

    results = {}
    for name, root in [("CBSD68", "data/test/CBSD68"),
                       ("McMaster", "data/test/McMaster")]:
        if not os.path.isdir(root):
            print(f"  {name}: not on disk, skipped")
            continue
        results[name] = {}
        for sigma in args.sigmas:
            ds = TestSet(root, task="denoise", sigma=sigma)
            m = evaluate(model, ds, "denoise", device=args.device, tile=256)
            results[name][f"sigma{sigma}"] = m
            print(f"  {name:9s} sigma={sigma:<3d} "
                  f"PSNR {m['psnr']:.3f} dB   SSIM {m['ssim']:.4f}  (n={m['n']})")

    # sample images at the middle sigma, so the visual matches the headline number
    sample_sigma = args.sigmas[len(args.sigmas) // 2]
    sample_root = "data/test/CBSD68" if os.path.isdir("data/test/CBSD68") else "data/test/McMaster"
    samples = []
    if os.path.isdir(sample_root):
        samples = save_samples(
            model, TestSet(sample_root, task="denoise", sigma=sample_sigma),
            os.path.join(out_dir, "samples"), n=args.samples, device=args.device)
        print(f"  wrote {len(samples)} sample triptychs (sigma={sample_sigma})")

    # gate figure -- subprocess with an argv list, not os.system: the interpreter
    # path contains spaces and cmd.exe mangles nested quoting
    gate_png = os.path.join(out_dir, "gates.png")
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, os.path.join("scripts", "plot_gates.py"), args.run_dir,
             "--labels", str(meta.get("variant", "dual")), "--out", gate_png],
            capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  gate figure -> {gate_png}")
        else:
            print(f"  gate plot failed (exit {r.returncode}): {r.stderr.strip()[-300:]}")
    except Exception as e:
        print(f"  gate plot failed: {type(e).__name__}: {e}")

    # training curve summary from the ledger
    ledger = os.path.join(args.run_dir, "ledger.jsonl")
    vals = []
    if os.path.exists(ledger):
        with open(ledger, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "val":
                    vals.append((r["iteration"], r["psnr"], r["ssim"]))

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"run": run_name, "iteration": ck["iteration"],
                   "params": n_params, "checkpoint": ckpt_used,
                   "test": results, "val_curve": vals}, f, indent=2)

    lines = [
        f"# {run_name}", "",
        f"- checkpoint: `{ckpt_used}` at iteration **{ck['iteration']:,}**",
        f"- parameters: **{n_params/1e6:.3f} M**",
        f"- variant: **{meta.get('variant')}**  seed {meta.get('seed')}",
        "", "## Test results (EMA weights, RGB PSNR, no border shave)", "",
        "| test set | sigma | PSNR (dB) | SSIM |", "|---|---|---|---|",
    ]
    for name, per_sigma in results.items():
        for k, m in per_sigma.items():
            lines.append(f"| {name} | {k.replace('sigma','')} | "
                         f"{m['psnr']:.3f} | {m['ssim']:.4f} |")
    if vals:
        lines += ["", "## Validation during training (McMaster, sigma 25)", "",
                  "| iteration | PSNR | SSIM |", "|---|---|---|"]
        lines += [f"| {i:,} | {p:.3f} | {s:.4f} |" for i, p, s in vals]
    lines += ["", "## Files", "",
              f"- `gates.png` -- mean gate value per block",
              f"- `samples/` -- {len(samples)} noisy | restored | ground-truth strips",
              f"- `metrics.json` -- machine readable", "",
              "**This is one arm.** `channel_only` and `window_only` are needed "
              "before any of it is interpretable as a result.", ""]
    summary = os.path.join(out_dir, "summary.md")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nreport written to {out_dir}/summary.md")


if __name__ == "__main__":
    main()
