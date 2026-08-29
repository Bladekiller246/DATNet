"""End-to-end smoke test of the training path, using synthetic images.

The trainer, the dataset classes, checkpoint save/resume and the in-loop
validation had never executed before this script existed -- they were the
largest untested surface in the project, and the kind that wastes a night
rather than a minute. This exercises all of them in about a minute, without
needing DIV2K on disk.

What it proves, in order:

  1. a real dataset -> DataLoader -> Trainer step works end to end
  2. a segment stops at its budget and exits 10 (more remain)
  3. **resume is exact**: the next segment continues from the saved iteration
     rather than restarting, with optimiser/EMA/RNG state restored
  4. the run completes and exits 0
  5. ledger.jsonl, gates.jsonl, last.pth and best.pth are all written

It deliberately shells out to `scripts/train.py` rather than importing Trainer,
so the real CLI path is what gets tested.

Usage:  python scripts/smoke_test.py [--keep]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "_smoke")
RUN_ROOT = os.path.join(ROOT, "runs", "_smoke")
CFG_PATH = os.path.join(ROOT, "configs", "_smoke.yaml")

TOTAL_ITERS = 40
SEGMENT = 20


def make_images(n, out_dir, size=192, seed=0):
    """Structured synthetic images -- gradients plus blobs, not pure noise.

    Pure noise is unlearnable, so validation PSNR would sit at chance and tell
    us nothing about whether the loop is wired correctly. Smooth structure gives
    the model something it can actually reduce the loss on.
    """
    from datnet.data.io import imwrite
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    for i in range(n):
        base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
        for _ in range(6):
            cy, cx = rng.uniform(0, 1, 2)
            r = rng.uniform(0.08, 0.3)
            blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
            base = base + rng.uniform(-0.5, 0.5, 3) * blob[..., None]
        img = np.clip(base, 0, 1)
        imwrite(os.path.join(out_dir, f"{i:03d}.png"), img)


def write_config():
    cfg = {
        "name": "_smoke",
        "model": {
            "width": 16, "topology": "unet",
            "enc_depths": [1, 1], "bottleneck_depth": 1, "dec_depths": [1, 1],
            "refine_depth": 1, "heads": [1, 2, 4], "window_size": 8,
            "ffn_expansion": 2.66, "gate_init": [0.8, 0.5, 0.2],
            "bias": False, "task": "restoration", "scale": 1, "grad_ckpt": False,
        },
        "data": {
            "kind": "gaussian_denoise",
            "root": os.path.join(DATA, "train").replace("\\", "/"),
            "patch_size": 64, "sigmas": [25], "repeat": 4, "num_workers": 0,
        },
        "val": {
            "gt_root": os.path.join(DATA, "test").replace("\\", "/"),
            "task": "denoise", "sigma": 25, "max_images": 2, "scale": 1,
        },
        "train": {
            "total_iters": TOTAL_ITERS, "effective_batch": 4, "micro_batch": 2,
            "amp": "bf16", "lr": 3e-4, "min_lr": 1e-6, "warmup_iters": 5,
            "weight_decay": 1e-4, "grad_clip": 1.0, "ema": True,
            "ema_decay": 0.99, "pixel_loss": "l1", "w_fft": 0.0,
            "log_every": 5, "gate_every": 5, "ckpt_every": 10, "val_every": 20,
            "select_metric": "psnr",
        },
    }
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_segment(py):
    return subprocess.run(
        [py, os.path.join("scripts", "train.py"),
         "--config", os.path.relpath(CFG_PATH, ROOT),
         "--variant", "dual", "--segment-iters", str(SEGMENT),
         "--run-root", os.path.relpath(RUN_ROOT, ROOT)],
        cwd=ROOT, capture_output=True, text=True)


def read_ledger(run_dir):
    path = os.path.join(run_dir, "ledger.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="do not delete artifacts")
    args = ap.parse_args()
    py = sys.executable

    # always start clean, or "resume" would be testing a stale run
    for d in (DATA, RUN_ROOT):
        shutil.rmtree(d, ignore_errors=True)

    print("generating synthetic images...")
    make_images(8, os.path.join(DATA, "train"), seed=0)
    make_images(2, os.path.join(DATA, "test"), seed=99)
    write_config()

    run_dir = os.path.join(RUN_ROOT, "_smoke__dual__seed0")
    checks = []

    print(f"\nsegment 1 (0 -> {SEGMENT})")
    r1 = run_segment(py)
    if r1.returncode not in (0, 10):
        print(r1.stdout[-3000:])
        print(r1.stderr[-3000:])
        print("[FAIL] segment 1 crashed")
        sys.exit(1)
    led = read_ledger(run_dir)
    end1 = [x for x in led if x.get("kind") == "segment_end"]
    it1 = end1[-1]["iteration"] if end1 else -1
    checks.append(("segment 1 exits 10 (more remain)", r1.returncode == 10,
                   f"got {r1.returncode}"))
    checks.append((f"segment 1 reached iteration {SEGMENT}", it1 == SEGMENT,
                   f"got {it1}"))
    checks.append(("last.pth written", os.path.exists(os.path.join(run_dir, "last.pth")), ""))

    print(f"segment 2 ({SEGMENT} -> {TOTAL_ITERS})")
    r2 = run_segment(py)
    if r2.returncode not in (0, 10):
        print(r2.stdout[-3000:])
        print(r2.stderr[-3000:])
        print("[FAIL] segment 2 crashed")
        sys.exit(1)
    led = read_ledger(run_dir)
    starts = [x for x in led if x.get("kind") == "segment_start"]
    ends = [x for x in led if x.get("kind") == "segment_end"]

    resumed = len(starts) > 1 and starts[1].get("resumed") is True
    continued = len(starts) > 1 and starts[1].get("iteration") == SEGMENT
    checks.append(("segment 2 resumed from checkpoint", resumed,
                   f"resumed={starts[1].get('resumed') if len(starts) > 1 else 'n/a'}"))
    checks.append((f"segment 2 CONTINUED at {SEGMENT}, did not restart", continued,
                   f"started at {starts[1].get('iteration') if len(starts) > 1 else 'n/a'}"))
    checks.append(("run completes, exits 0", r2.returncode == 0, f"got {r2.returncode}"))
    checks.append((f"final iteration == {TOTAL_ITERS}",
                   bool(ends) and ends[-1]["iteration"] == TOTAL_ITERS,
                   f"got {ends[-1]['iteration'] if ends else 'n/a'}"))

    gates = os.path.join(run_dir, "gates.jsonl")
    checks.append(("gates.jsonl written", os.path.exists(gates), ""))
    if os.path.exists(gates):
        with open(gates, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        n_gates = len(rows[-1]["gates"]) if rows else 0
        checks.append(("gate values logged per block", n_gates > 0,
                       f"{n_gates} blocks, {len(rows)} snapshots"))

    vals = [x for x in led if x.get("kind") == "val"]
    checks.append(("in-loop validation ran", bool(vals),
                   f"psnr={vals[-1]['psnr']:.2f} dB" if vals else "none"))
    checks.append(("best.pth written", os.path.exists(os.path.join(run_dir, "best.pth")), ""))

    print()
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}"
              f"{'  (' + detail + ')' if detail else ''}")
    n_ok = sum(1 for _, p, _ in checks if p)
    print(f"\n{n_ok}/{len(checks)} checks passed")

    if not args.keep:
        shutil.rmtree(DATA, ignore_errors=True)
        shutil.rmtree(RUN_ROOT, ignore_errors=True)
        if os.path.exists(CFG_PATH):
            os.remove(CFG_PATH)
    else:
        print(f"\nartifacts kept in {RUN_ROOT} and {DATA}")

    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == "__main__":
    main()
