"""Phase 5: joint all-in-one training, segmented exactly like the single-task runs.

Differences from scripts/train.py:
  * one shared backbone with per-task tails (models/allinone.py)
  * task-homogeneous micro-batches drawn sqrt-proportionally (data/multitask.py)
  * each task loss divided by a running mean of its own magnitude, because
    denoising sits near 40 dB and SR near 27 dB and the raw gradients are not
    comparable
  * gates are logged per task label, which is the Phase-5 form of the key figure

Usage:
    python scripts/train_multitask.py --config configs/phase5_allinone.yaml \
        --conditioning oracle --segment-iters 10000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader

from datnet.data.build import build_train_dataset
from datnet.data.multitask import LossBalancer, MultiTaskLoader, sqrt_proportional_weights
from datnet.engine.losses import RestorationLoss
from datnet.engine.trainer import Trainer
from datnet.models.allinone import AllInOneDATNet
from datnet.utils.complexity import count_parameters

EXIT_COMPLETE, EXIT_MORE = 0, 10


class MultiTaskTrainer(Trainer):
    """Trainer with per-task loss balancing. Everything else is inherited."""

    def __init__(self, *a, balancer=None, **kw):
        super().__init__(*a, **kw)
        self.balancer = balancer

    def _forward(self, batch):
        pred, logits = self.model(batch["input"], batch.get("task_id"),
                                  task_name=batch.get("task_name"))
        loss, parts = self.loss_fn(pred, batch["target"], logits,
                                   batch.get("task_id"))
        if self.balancer is not None:
            name = batch["task_name"]
            loss = self.balancer.scale(name, loss)
            parts[f"raw_{name}"] = parts["pixel"]
        return loss, parts


def build_loaders(cfg, micro_batch):
    loaders, sizes = {}, {}
    for name, dcfg in cfg["data"].items():
        ds = build_train_dataset(dcfg)
        sizes[name] = len(ds)
        loaders[name] = DataLoader(
            ds, batch_size=micro_batch, shuffle=True,
            num_workers=dcfg.get("num_workers", 2), pin_memory=True,
            drop_last=True, persistent_workers=dcfg.get("num_workers", 2) > 0)
    return loaders, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditioning", default="oracle", choices=["oracle", "blind"])
    ap.add_argument("--segment-iters", type=int, default=None)
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    tasks = list(cfg["data"].keys())
    run_name = f"{cfg['name']}__{args.conditioning}__seed{args.seed}"
    run_dir = os.path.join(args.run_root, run_name)

    model = AllInOneDATNet(
        cfg["model"], tasks=tasks, sr_scale=cfg.get("sr_scale", 4),
        conditioning=args.conditioning,
    )
    print(f"run: {run_name}   params: {count_parameters(model)/1e6:.3f} M   "
          f"tasks: {tasks}")

    train_cfg = dict(cfg["train"])
    mb = train_cfg["micro_batch"]

    def rebuild(new_mb):
        loaders, sizes = build_loaders(cfg, new_mb)
        return MultiTaskLoader(loaders, weights=sqrt_proportional_weights(sizes),
                               seed=args.seed)

    loader = rebuild(mb)
    train_cfg["rebuild_loader"] = rebuild
    print("sampling weights:", {k: round(v, 3) for k, v in loader.weights.items()})

    loss_fn = RestorationLoss(
        pixel=train_cfg.get("pixel_loss", "l1"),
        w_fft=train_cfg.get("w_fft", 0.0),
        # the classifier head is only supervised in the blind setting
        w_cls=train_cfg.get("w_cls", 0.1) if args.conditioning == "blind" else 0.0,
    )
    balancer = LossBalancer(tasks, momentum=train_cfg.get("balance_momentum", 0.99))

    trainer = MultiTaskTrainer(model, loader, loss_fn, train_cfg, run_dir,
                               device=args.device, balancer=balancer)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "conditioning": args.conditioning,
                   "tasks": tasks, "seed": args.seed}, f, indent=2)

    record = trainer.fit(segment_iters=args.segment_iters)
    summary = trainer.ledger.summary()
    print(f"\nsegment done: {record['iters_done']:,} it -> "
          f"{record['iteration']:,}/{trainer.total_iters:,}")
    if summary:
        print(f"  {summary['percent']}%  {summary['mean_it_per_s']} it/s  "
              f"peak {summary['peak_vram_gb']} GB  ETA {summary['eta_hours']} h")
    sys.exit(EXIT_COMPLETE if record.get("complete") else EXIT_MORE)


if __name__ == "__main__":
    main()
