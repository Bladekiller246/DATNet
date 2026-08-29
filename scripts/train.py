"""Train one segment of one run, then exit.

This is deliberately not a long-lived process. Each invocation resumes the run
from its last checkpoint, trains at most `--segment-iters` iterations, writes a
checkpoint and a ledger entry, and exits:

    exit 0   run complete
    exit 10  segment done, more remain  (the runner loop keeps going)
    exit 3   DIVERGED -- non-finite gradients; runner must stop, not retry
    exit 1   error

Usage:
    python scripts/train.py --config configs/phase1_denoise.yaml \
        --variant dual --segment-iters 10000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from datnet.data.build import build_test_dataset, make_loader_factory
from datnet.engine.losses import RestorationLoss
from datnet.engine.trainer import DivergedError, Trainer
from datnet.evaluation.evaluate import make_val_fn
from datnet.models.build import build_model
from datnet.utils.complexity import count_parameters
from datnet.utils.matching import match_arm

EXIT_COMPLETE, EXIT_MORE, EXIT_DIVERGED = 0, 10, 3


def resolve_width(model_cfg, variant, match_to_dual=True):
    """Grow a single-axis arm until it matches the dual arm on parameters.

    Width first, then `ffn_expansion` as a fine knob when width alone cannot
    land inside 2% -- see datnet/utils/matching.py. Depth is never touched.
    """
    cfg = {**model_cfg, "mode": variant}
    # Record the resolved value explicitly, not just the DATNet default, so a
    # future eval never has to guess it back from weights -- normalise_branches
    # is parameter-free and leaves no trace in the state dict to detect it from.
    cfg.setdefault("normalise_branches", True)
    if variant == "dual" or not match_to_dual:
        return cfg, count_parameters(build_model(cfg))
    target = count_parameters(build_model({**model_cfg, "mode": "dual"}))
    info = match_arm(build_model, model_cfg, target, variant)
    cfg["width"] = info["width"]
    cfg["ffn_expansion"] = info["ffn_expansion"]
    print(f"parameter matching [{info['knob']}]: {variant} "
          f"width {model_cfg['width']} -> {info['width']}, "
          f"gamma {model_cfg.get('ffn_expansion', 2.66)} -> {info['ffn_expansion']} "
          f"({info['params']:,} params, {info['rel_error']:.2%} off dual)")
    return cfg, info["params"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["channel_only", "window_only", "dual"])
    ap.add_argument("--segment-iters", type=int, default=None,
                    help="max iterations this invocation; omit to run to completion")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # Ada supports TF32; restoration is not precision-critical at fp32 matmul level
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_name = f"{cfg['name']}__{args.variant}__seed{args.seed}"
    run_dir = os.path.join(args.run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model_cfg, n_params = resolve_width(cfg["model"], args.variant)
    model = build_model(model_cfg)
    print(f"run: {run_name}   params: {n_params/1e6:.3f} M")

    factory, dataset = make_loader_factory(cfg["data"])
    train_cfg = dict(cfg["train"])
    loader = factory(train_cfg["micro_batch"])
    train_cfg["rebuild_loader"] = factory

    loss_fn = RestorationLoss(
        pixel=train_cfg.get("pixel_loss", "l1"),
        w_pixel=train_cfg.get("w_pixel", 1.0),
        w_fft=train_cfg.get("w_fft", 0.0),
        w_cls=train_cfg.get("w_cls", 0.0),
    )

    val_fn = None
    if cfg.get("val"):
        val_ds = build_test_dataset(cfg["val"])
        val_fn = make_val_fn(val_ds, cfg["val"]["task"],
                             scale=cfg["val"].get("scale", 1),
                             device=args.device,
                             max_images=cfg["val"].get("max_images", 8),
                             tile=cfg["val"].get("tile"))

    trainer = Trainer(model, loader, loss_fn, train_cfg, run_dir,
                      device=args.device, val_fn=val_fn)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "model_cfg": model_cfg, "variant": args.variant,
                   "params": n_params, "seed": args.seed}, f, indent=2)

    try:
        record = trainer.fit(segment_iters=args.segment_iters)
    except DivergedError as e:
        # Exit 3, not 1: the runner must NOT retry this. Resuming would reload
        # the same diverged state and diverge again, and retrying is how hours
        # get burned on a model that is already dead.
        print(f"\nDIVERGED: {e}", file=sys.stderr)
        sys.exit(EXIT_DIVERGED)
    summary = trainer.ledger.summary()
    print(f"\nsegment done: {record['iters_done']:,} it  "
          f"-> {record['iteration']:,}/{trainer.total_iters:,}")
    if summary:
        print(f"  {summary['percent']}% complete  "
              f"{summary['mean_it_per_s']} it/s  "
              f"peak {summary['peak_vram_gb']} GB  "
              f"ETA {summary['eta_hours']} h")
    sys.exit(EXIT_COMPLETE if record.get("complete") else EXIT_MORE)


if __name__ == "__main__":
    main()
