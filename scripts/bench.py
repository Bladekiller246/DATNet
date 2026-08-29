"""Measure real throughput and VRAM on THIS card, then size the runs from it.

Feeds `torch.randn`, not images: VRAM and step time depend only on tensor shapes
and the op graph, so no dataset is needed. That also makes the result an UPPER
BOUND on real training throughput -- it excludes disk, decode, augmentation and
DataLoader workers.

**Resumable.** Every (arm, micro-batch) measurement is written to the results
JSON as soon as it is taken, and an existing file is loaded on startup so
already-measured points are skipped. Safe to interrupt with Ctrl-C or by losing
mains power; re-run the same command to continue. `--force` re-measures.

Usage:
  python scripts/bench.py --patch 128 --vram-ceiling 7.0
  python scripts/bench.py --width 56 --out runs/bench_w56.json
  python scripts/bench.py --patch 64 --task sr --scale 4
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datnet.engine.losses import RestorationLoss
from datnet.models.build import build_model
from datnet.utils.matching import match_all

BASE = dict(width=32, topology="unet", enc_depths=(2, 3), bottleneck_depth=4,
            dec_depths=(3, 2), refine_depth=2, heads=(1, 2, 4), window_size=8,
            ffn_expansion=2.66, gate_init=(0.8, 0.5, 0.2), task="restoration")


def time_steps(model, patch, micro_batch, scale, amp, steps=12, warmup=4):
    dev = "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = RestorationLoss(pixel="l1")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[amp]
    scaler = torch.amp.GradScaler(dev, enabled=(dtype is torch.float16))

    x = torch.randn(micro_batch, 3, patch, patch, device=dev)
    y = torch.randn(micro_batch, 3, patch * scale, patch * scale, device=dev)

    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev, dtype=dtype, enabled=dtype is not None):
            loss, _ = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return steps / dt, peak


class Results:
    """Incremental, resumable results file."""

    def __init__(self, path, meta):
        self.path = path
        self.data = {"meta": meta, "measurements": {}, "summary": {}}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    old = json.load(f)
                if old.get("meta", {}).get("key") == meta["key"]:
                    self.data = old
                    n = len(old.get("measurements", {}))
                    print(f"resuming from {path} ({n} measurements already taken)\n")
                else:
                    print(f"{path} was measured for a different configuration; "
                          f"starting fresh\n")
            except (json.JSONDecodeError, OSError):
                print(f"{path} unreadable; starting fresh\n")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def has(self, key):
        return key in self.data["measurements"]

    def get(self, key):
        return self.data["measurements"][key]

    def put(self, key, value):
        self.data["measurements"][key] = value
        self.flush()

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--width", type=int, default=32,
                    help="base width of the dual arm; the other arms are matched to it")
    ap.add_argument("--task", default="restoration", choices=["restoration", "sr"])
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--effective-batch", type=int, default=16)
    ap.add_argument("--vram-ceiling", type=float, default=7.0,
                    help="GB; leave headroom below 8 for the display and fragmentation")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--iters", type=int, default=100000, help="planned run length")
    ap.add_argument("--micro-batches", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--out", default="runs/bench.json")
    ap.add_argument("--force", action="store_true", help="re-measure everything")
    ap.add_argument("--vram-fraction", type=float, default=0.90,
                    help="cap the torch allocator so oversized configs OOM "
                         "instead of silently spilling to system RAM")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench.py needs a CUDA device")
    # Without this, Windows lets the driver back oversized allocations with
    # system RAM: the run does not OOM, it just becomes ~20x slower and reports
    # a peak larger than the card. Cap it so OOM is honest.
    if args.vram_fraction:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    props = torch.cuda.get_device_properties(0)
    print(f"gpu: {props.name}  {props.total_memory / 1024**3:.1f} GB  "
          f"sm_{props.major}{props.minor}  bf16={torch.cuda.is_bf16_supported()}")
    print(f"patch={args.patch} width={args.width} amp={args.amp} "
          f"grad_ckpt={args.grad_ckpt} effective_batch={args.effective_batch}\n")

    key = (f"w{args.width}_p{args.patch}_{args.task}x{args.scale}_"
           f"{args.amp}_gc{int(args.grad_ckpt)}_eb{args.effective_batch}")
    meta = {"key": key, "gpu": props.name, **vars(args)}
    store = Results(args.out, meta)
    if args.force:
        store.data["measurements"] = {}

    cfg = {**BASE, "width": args.width, "task": args.task, "scale": args.scale,
           "grad_ckpt": args.grad_ckpt}
    if args.task == "sr":
        cfg.update(topology="flat", flat_depth=12)
    matched = match_all(build_model, {**cfg, "mode": "dual"})

    results = {}
    for mode, info in matched.items():
        row = {"width": info["width"], "ffn_expansion": info["ffn_expansion"],
               "params_M": round(info["params"] / 1e6, 3), "knob": info["knob"]}
        best = None
        for mb in args.micro_batches:
            if mb > args.effective_batch:
                break
            mkey = f"{mode}_mb{mb}"
            accum = args.effective_batch // mb

            if store.has(mkey):
                m = store.get(mkey)
                if m.get("oom"):
                    print(f"  {mode:14s} micro_batch={mb:2d}  OOM (cached)")
                    break
                peak, opt_ips = m["peak_vram_gb"], m["opt_it_per_s"]
                print(f"  {mode:14s} micro_batch={mb:2d}  {peak:4.2f} GB  "
                      f"{opt_ips:5.2f} opt-it/s  (cached)")
            else:
                model = build_model({**cfg, "mode": mode,
                                     "width": info["width"],
                                     "ffn_expansion": info["ffn_expansion"]}).cuda()
                try:
                    ips, peak = time_steps(model, args.patch, mb, args.scale,
                                           args.amp)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {mode:14s} micro_batch={mb:2d}  OOM")
                    store.put(mkey, {"oom": True})
                    del model
                    torch.cuda.empty_cache()
                    break
                # it/s here is micro-batch steps; an optimiser step costs `accum`
                opt_ips = ips / accum
                status = "ok" if peak <= args.vram_ceiling else "over ceiling"
                print(f"  {mode:14s} micro_batch={mb:2d}  {peak:4.2f} GB  "
                      f"{opt_ips:5.2f} opt-it/s  ({status})")
                store.put(mkey, {"micro_batch": mb, "accum": accum,
                                 "peak_vram_gb": round(peak, 2),
                                 "opt_it_per_s": round(opt_ips, 3)})
                del model
                torch.cuda.empty_cache()

            # pick the FASTEST that fits, not the largest. Throughput peaks at
            # mb=8 on this card and falls off at 16 under allocator pressure.
            if peak <= args.vram_ceiling and (
                    best is None or opt_ips > best["opt_it_per_s"]):
                best = {"micro_batch": mb, "accum": accum,
                        "peak_vram_gb": round(peak, 2),
                        "opt_it_per_s": round(opt_ips, 3)}

        if best is None:
            row["error"] = "nothing fits under the VRAM ceiling"
        else:
            row.update(best)
            hours = args.iters / best["opt_it_per_s"] / 3600
            row["projected_hours"] = round(hours, 1)
            print(f"  {mode:14s} -> best micro_batch={best['micro_batch']} "
                  f"accum={best['accum']}  {args.iters:,} it = {hours:.1f} h\n")
        results[mode] = row

    total = sum(r.get("projected_hours", 0) for r in results.values())
    print(f"all three arms, {args.iters:,} iterations each: {total:.1f} GPU-hours")

    store.data["summary"] = {"results": results, "total_hours": round(total, 1)}
    store.flush()
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
