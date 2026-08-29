"""Pre-crop DIV2K into overlapping sub-images.

Random-cropping a 2040x1356 PNG decodes 2.8 M pixels to use 16 k of them. At
128x128 patches that is roughly 170x more decode work than the sample needs, and
on this machine the data pipeline is CPU- and RAM-bound long before the GPU is.
Pre-cropping to 480x480 tiles cuts decode cost by about an order of magnitude and
is the single largest throughput win available outside the model itself.

Disk cost: DIV2K HR goes from ~3.3 GB to roughly 8-10 GB. There is 442 GB free.

Usage:
    python scripts/prepare_subimages.py --input data/DIV2K/DIV2K_train_HR \
        --output data/DIV2K/DIV2K_train_HR_sub --size 480 --step 240

For SR, run it on the HR folder and again on the LR folder with size and step
divided by the scale factor, so the tiles stay aligned:
    --size 120 --step 60   (for x4)
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datnet.data.io import imread, imwrite, list_images


def crop_one(job):
    path, out_dir, size, step, thresh = job
    img = imread(path)
    h, w = img.shape[:2]
    stem = os.path.splitext(os.path.basename(path))[0]

    ys = list(range(0, max(h - size, 0) + 1, step))
    xs = list(range(0, max(w - size, 0) + 1, step))
    # keep the last partial strip if it would otherwise drop a lot of pixels
    if ys and h - (ys[-1] + size) > thresh:
        ys.append(h - size)
    if xs and w - (xs[-1] + size) > thresh:
        xs.append(w - size)

    n = 0
    for y in ys:
        for x in xs:
            tile = img[y:y + size, x:x + size]
            if tile.shape[0] != size or tile.shape[1] != size:
                continue
            imwrite(os.path.join(out_dir, f"{stem}_s{n:03d}.png"), tile)
            n += 1
    return path, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--step", type=int, default=240)
    ap.add_argument("--threshold", type=int, default=None,
                    help="keep a trailing strip if the remainder exceeds this (default size/2)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    thresh = args.threshold if args.threshold is not None else args.size // 2
    os.makedirs(args.output, exist_ok=True)
    files = list_images(args.input)
    jobs = [(f, args.output, args.size, args.step, thresh) for f in files]

    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (path, n) in enumerate(ex.map(crop_one, jobs), 1):
            total += n
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} images -> {total} tiles")

    print(f"\n{len(files)} images -> {total} tiles of {args.size}x{args.size} "
          f"in {args.output}")
    print("point the config at this folder, and drop `repeat` to 1-2")


if __name__ == "__main__":
    main()
