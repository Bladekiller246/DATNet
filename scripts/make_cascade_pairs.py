"""Generate stage-2 training pairs for a cascaded restoration pipeline.

The problem this solves
-----------------------
Chaining separately-trained models leaks quality at every hand-off, because each
stage was trained assuming the *other* stages' degradations do not exist. Measured
on this project (docs/FINDINGS.md §10): feeding denoiser output to the GoPro-
trained deblurrer COSTS 0.53 dB, because that model only ever saw camera motion
blur -- a different inverse problem from denoiser over-smoothing.

The fix is to train stage 2 on the real output distribution of stage 1: run the
trained denoiser over clean images with synthetic noise, and pair its output
(LQ) with the original clean image (GT). Stage 2 then learns to repair exactly
what stage 1 actually gets wrong, rather than a generic blur that merely
resembles it.

Output layout
-------------
Only the LQ side is written. `gt_root` points at the existing clean sub-images,
so nothing is duplicated:

    data/DIV2K/DIV2K_train_HR_sub/        <- GT (existing, untouched)
    data/DIV2K/DIV2K_train_denoised_sub/  <- LQ (written here)

Resumable: files that already exist are skipped, so an interrupted run continues
where it stopped.

Usage
-----
    python scripts/make_cascade_pairs.py \
        --denoiser runs/phase1_denoise_60k__dual__seed0 \
        --limit 8000
"""
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from datnet.data.io import list_images, imread, imwrite
from datnet.evaluation.inference import restore

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("rp", os.path.join(_HERE, "report_phase.py"))
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoiser", default="runs/phase1_denoise_60k__dual__seed0",
                    help="run dir of the trained stage-1 model")
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--src", default="data/DIV2K/DIV2K_train_HR_sub")
    ap.add_argument("--out", default="data/DIV2K/DIV2K_train_denoised_sub")
    ap.add_argument("--sigmas", type=int, nargs="*", default=[15, 25, 50],
                    help="noise levels to sample from; match stage-1 training")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of images (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, meta, ck, used = rp.load_model(args.denoiser, args.checkpoint, args.device)
    print(f"stage-1 model: {args.denoiser}")
    print(f"  {used} @ iteration {ck['iteration']:,}  params {meta['params']:,}")

    files = list_images(args.src)
    if args.limit:
        files = files[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"source: {len(files):,} images from {args.src}")
    print(f"output: {args.out}")
    print(f"sigmas: {args.sigmas}\n")

    # Seeded per index, so a resumed run reproduces the same noise for the same
    # file and the dataset stays deterministic across interruptions.
    written = skipped = 0
    for i, path in enumerate(files):
        dst = os.path.join(args.out, os.path.basename(path))
        if os.path.exists(dst):
            skipped += 1
            continue

        rng = np.random.default_rng(args.seed * 1_000_003 + i)
        gt = imread(path).astype(np.float32) / 255.0
        sigma = float(rng.choice(args.sigmas)) / 255.0
        lq = np.clip(gt + rng.standard_normal(gt.shape).astype(np.float32) * sigma, 0, 1)

        x = torch.from_numpy(lq.transpose(2, 0, 1)).unsqueeze(0).to(args.device)
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16,
                                             enabled=args.device != "cpu"):
            out = restore(model, x, scale=1, tile=None)
        out = out.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

        imwrite(dst, out)
        written += 1
        if written % 250 == 0:
            print(f"  {written:,} written ({i + 1:,}/{len(files):,})", flush=True)

    print(f"\ndone. written {written:,}, skipped {skipped:,} (already present)")
    print(f"\nNext: train stage 2 with lq_root={args.out}, gt_root={args.src}")


if __name__ == "__main__":
    main()
