"""Segmented training loop for a single 8 GB consumer GPU.

Two ideas carry this file:

1. **Micro-batch vs effective batch.** The three ablation arms do not use the
   same VRAM per image -- a widened `window_only` stores larger window attention
   maps than `dual`. Comparability requires a constant *effective* batch, so the
   micro-batch is a free per-arm knob and gradient accumulation makes up the
   difference. Never shrink the effective batch to fit memory; shrink the
   micro-batch instead.

2. **Segments.** A 200k-iteration run is executed as a sequence of bounded
   segments, each a separate process that resumes exactly where the last one
   stopped. On a thermally-limited laptop that is the difference between a run
   you can babysit and a run you cannot.
"""
import json
import os
import time

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .ema import ModelEMA
from .ledger import Ledger
from .scheduler import CosineWarmup
from .watchdog import GateStallWatchdog, LossSpikeWatchdog


class DivergedError(RuntimeError):
    """Training produced non-finite gradients for too many consecutive steps.

    Raised so the failure is LOUD. Divergence otherwise looks like success: the
    process keeps running, exits 0, and the crash-watchdog sees nothing wrong.
    """


def infinite(loader):
    """Cycle a finite DataLoader forever, yielding batches."""
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, model, loader, loss_fn, cfg, run_dir, device="cuda",
                 val_fn=None):
        # `rebuild_loader` is a closure and cannot be pickled. self.cfg is
        # written verbatim into every checkpoint, so it has to stay
        # serialisable -- keep the callable out of it.
        cfg = dict(cfg)
        self._rebuild_loader = cfg.pop("rebuild_loader", None)
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        # Cap the allocator BEFORE moving anything to the device.
        #
        # On Windows the NVIDIA driver silently backs over-large allocations with
        # system RAM instead of failing. Measured on this card: a config wanting
        # 12.7 GB on an 8 GB GPU did not OOM -- it ran at 0.06 it/s instead of
        # 1.17, a 20x slowdown, and would have burned a whole night looking like
        # "training, just slow". Capping the fraction makes torch raise a real
        # OutOfMemoryError, which _backoff() can then catch and halve the
        # micro-batch. Silent spilling is strictly worse than a clean crash.
        frac = cfg.get("vram_fraction", 0.90)
        if frac and torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(frac)

        self.model = model.to(device)
        if cfg.get("channels_last", False):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.loader = loader
        self.loss_fn = loss_fn
        self.val_fn = val_fn

        self.total_iters = cfg["total_iters"]
        self.effective_batch = cfg["effective_batch"]
        self.micro_batch = cfg["micro_batch"]
        assert self.effective_batch % self.micro_batch == 0, \
            "effective_batch must be a whole number of micro-batches"
        self.accum = self.effective_batch // self.micro_batch

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                          "off": None}[cfg.get("amp", "bf16")]
        # bf16 has the same exponent range as fp32, so it needs no loss scaling.
        # fp16 does. Ada (RTX 4060) supports bf16 natively, so bf16 is default.
        self.scaler = torch.amp.GradScaler(
            device, enabled=(self.amp_dtype is torch.float16)
        )

        # The axis gate is a routing logit, not a weight matrix -- decaying it
        # toward zero every step actively opposes any task signal pushing it
        # away from its init, on top of whatever the raw gradient is doing.
        # Exclude it from weight decay so a "gate never moves" result can't be
        # explained by this before it's explained by "no signal to move on".
        gate_params = [p for n, p in self.model.named_parameters() if ".gate." in n]
        other_params = [p for n, p in self.model.named_parameters() if ".gate." not in n]
        wd = cfg.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": wd},
                {"params": gate_params, "weight_decay": 0.0},
            ],
            lr=cfg["lr"], betas=tuple(cfg.get("betas", (0.9, 0.999))),
        )
        self.scheduler = CosineWarmup(
            self.optimizer, self.total_iters, cfg.get("warmup_iters", 5000),
            base_lr=cfg["lr"], min_lr=cfg.get("min_lr", 1e-6),
        )
        self.ema = ModelEMA(self.model, decay=cfg.get("ema_decay", 0.999),
                            device=device) if cfg.get("ema", True) else None

        self.iteration = 0
        self.best = None
        self.nonfinite_streak = 0
        self.nonfinite_total = 0
        # give up rather than burn hours training a corpse; see DivergedError
        self.max_nonfinite_streak = cfg.get("max_nonfinite_streak", 50)
        self.ledger = Ledger(os.path.join(run_dir, "ledger.jsonl"))
        self.gate_log = os.path.join(run_dir, "gates.jsonl")
        self.ckpt_path = os.path.join(run_dir, "last.pth")
        self.best_path = os.path.join(run_dir, "best.pth")

        # Soft-alert watchdogs -- log loudly, never abort. See watchdog.py.
        wd_cfg = cfg.get("watchdog", {})
        self.loss_spike_watchdog = LossSpikeWatchdog(
            window=wd_cfg.get("loss_spike_window", 50),
            spike_factor=wd_cfg.get("loss_spike_factor", 5.0),
            min_streak=wd_cfg.get("loss_spike_streak", 3),
        )
        self.gate_stall_watchdog = GateStallWatchdog(
            warmup_iters=wd_cfg.get("gate_stall_warmup", cfg.get("warmup_iters", 5000)),
            min_move=wd_cfg.get("gate_stall_min_move", 0.02),
        )

    # ------------------------------------------------------------------ resume
    def maybe_resume(self):
        if not os.path.exists(self.ckpt_path):
            return False
        ck = load_checkpoint(
            self.ckpt_path, model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=self.scaler, ema=self.ema,
            map_location=self.device,
        )
        self.iteration = ck["iteration"]
        self.best = ck.get("best")
        return True

    @torch.no_grad()
    def _weights_are_finite(self):
        for p in self.model.parameters():
            if not torch.isfinite(p).all():
                return False
        return True

    def _save(self, path, **extra):
        # Never overwrite a good checkpoint with a diverged model. The previous
        # run wrote NaN weights over last.pth 30 times; only best.pth survived,
        # and only by accident (NaN > best compares False).
        if not self._weights_are_finite():
            self.ledger.append({"kind": "save_refused", "iteration": self.iteration,
                                "path": os.path.basename(path),
                                "reason": "model weights are non-finite"})
            return
        save_checkpoint(
            path, model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=self.scaler, ema=self.ema,
            iteration=self.iteration, config=self.cfg, best=self.best,
            extra=extra,
        )

    # ------------------------------------------------------------------- steps
    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                v = v.to(self.device, non_blocking=True)
                if self.cfg.get("channels_last", False) and v.dim() == 4:
                    v = v.to(memory_format=torch.channels_last)
            out[k] = v
        return out

    def _forward(self, batch):
        model_out = self.model(batch["input"], batch.get("task_id"))
        if isinstance(model_out, tuple):
            pred, logits = model_out
        else:
            pred, logits = model_out, None
        return self.loss_fn(pred, batch["target"], logits, batch.get("task_id"))

    def train_step(self, data_iter):
        """One optimiser step = `accum` micro-batches.

        Guards the LOSS, not just the gradients. Measured, and counter-intuitive:
        a NaN loss does NOT reliably produce NaN gradients. With L1 and a NaN
        target the loss is nan while all 98 parameter gradients come back finite,
        so a gradient-only check sails straight past a diverged model -- which is
        precisely the failure that destroyed a 60k run, where the ledger showed
        `loss=nan` for 40k iterations. The loss is the signal that actually
        appears, so it is the one that must be checked.
        """
        self.optimizer.zero_grad(set_to_none=True)
        agg = {}
        nonfinite_micro = 0
        for _ in range(self.accum):
            batch = self._to_device(next(data_iter))
            with torch.autocast(self.device, dtype=self.amp_dtype,
                                enabled=self.amp_dtype is not None):
                loss, parts = self._forward(batch)
            if not torch.isfinite(loss):
                # never backward a non-finite loss: even when it happens to
                # produce finite gradients, those gradients are meaningless
                nonfinite_micro += 1
                continue
            self.scaler.scale(loss / self.accum).backward()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / self.accum

        if nonfinite_micro:
            # skip the WHOLE step, not just the bad micro-batch: a partial
            # gradient from a step that saw NaN is not a gradient worth taking
            self.nonfinite_streak += 1
            self.nonfinite_total += 1
            self.optimizer.zero_grad(set_to_none=True)
            agg["nonfinite_micro"] = float(nonfinite_micro)
            agg.setdefault("pixel", float("nan"))
            agg.setdefault("total", float("nan"))
            return agg

        # --- finite-gradient guard -----------------------------------------
        # With amp='bf16' the GradScaler is disabled, because bf16 needs no loss
        # scaling. But the scaler is ALSO what skips optimiser steps when
        # gradients are non-finite -- disabling it silently removes that safety
        # net, and one bad gradient then poisons every weight permanently.
        #
        # This is not hypothetical: it destroyed a 60k run. Loss went NaN around
        # iteration 17k and the run continued for 40k more iterations, ending
        # with all 248 tensors non-finite.
        #
        # grad_clip does NOT protect against this. clip_grad_norm_ computes
        # total_norm = NaN, then clip_coef = max_norm / NaN = NaN, and multiplies
        # every gradient by NaN. Clipping SPREADS the NaN rather than blocking it,
        # so the check has to come first.
        if self.cfg.get("grad_clip"):
            self.scaler.unscale_(self.optimizer)

        finite = self._grads_are_finite()
        if not finite:
            self.nonfinite_streak += 1
            self.nonfinite_total += 1
            self.optimizer.zero_grad(set_to_none=True)
            agg["skipped"] = 1.0
            return agg
        self.nonfinite_streak = 0

        if self.cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.cfg["grad_clip"])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.ema is not None:
            self.ema.update(self.model, step=self.iteration)
        return agg

    @torch.no_grad()
    def _grads_are_finite(self):
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
        return True

    # --------------------------------------------------------------------- fit
    def fit(self, segment_iters=None):
        """Train at most `segment_iters` more iterations, then exit cleanly.

        Returns the segment record; `complete=True` means the run has reached
        total_iters and no further segments are needed.
        """
        resumed = self.maybe_resume()
        start = self.iteration
        budget = self.total_iters - start
        if segment_iters is not None:
            budget = min(budget, segment_iters)
        if budget <= 0:
            return {"complete": True, "iteration": self.iteration, "iters_done": 0}

        self.ledger.append({
            "kind": "segment_start", "iteration": start, "budget": budget,
            "resumed": resumed, "micro_batch": self.micro_batch,
            "accum": self.accum, "amp": self.cfg.get("amp", "bf16"),
        })

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        data_iter = infinite(self.loader)
        self.model.train()
        t0 = time.time()
        log_every = self.cfg.get("log_every", 100)
        ckpt_every = self.cfg.get("ckpt_every", 2000)
        val_every = self.cfg.get("val_every", 5000)
        gate_every = self.cfg.get("gate_every", 500)

        done = 0
        while done < budget:
            lr = self.scheduler.step(self.iteration)
            try:
                parts = self.train_step(data_iter)
            except torch.cuda.OutOfMemoryError:
                if not self._backoff():
                    raise
                continue
            self.iteration += 1
            done += 1

            # Abort on sustained divergence. A crash is loud and the watchdog
            # catches it; divergence is silent and the process exits 0, which is
            # exactly how 40k iterations got burned on a NaN model. Turn the
            # quiet failure into a loud one.
            if self.nonfinite_streak >= self.max_nonfinite_streak:
                self.ledger.append({
                    "kind": "diverged", "iteration": self.iteration,
                    "nonfinite_streak": self.nonfinite_streak,
                    "nonfinite_total": self.nonfinite_total,
                })
                raise DivergedError(
                    f"non-finite gradients for {self.nonfinite_streak} consecutive "
                    f"steps at iteration {self.iteration}. Training aborted before "
                    f"it could overwrite good checkpoints. Lower the learning rate "
                    f"(peak was {self.cfg.get('lr')}) or lengthen warmup, then "
                    f"restart from a healthy checkpoint."
                )

            if self.iteration % log_every == 0:
                self._log_scalars(parts, lr, t0, start)
            if self.iteration % gate_every == 0:
                self._log_gates()
            if self.iteration % ckpt_every == 0:
                self._save(self.ckpt_path)
            if self.val_fn and self.iteration % val_every == 0:
                self._validate()

        self._save(self.ckpt_path)
        seconds = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
                if torch.cuda.is_available() else 0.0)
        record = self.ledger.append({
            "kind": "segment_end", "iteration": self.iteration,
            "total_iters": self.total_iters, "iters_done": done,
            "seconds": round(seconds, 1),
            "it_per_s": round(done / seconds, 3) if seconds else None,
            "peak_vram_gb": round(peak, 2),
            "best": self.best,
        })
        record["complete"] = self.iteration >= self.total_iters
        return record

    def _backoff(self):
        """Halve the micro-batch on OOM, keeping the effective batch constant.

        Needs `cfg['rebuild_loader']`: a callable taking the new micro-batch and
        returning a fresh DataLoader.
        """
        rebuild = self._rebuild_loader
        if self.micro_batch <= 1 or rebuild is None:
            return False
        self.micro_batch //= 2
        self.accum = self.effective_batch // self.micro_batch
        self.loader = rebuild(self.micro_batch)
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.ledger.append({"kind": "oom_backoff", "iteration": self.iteration,
                            "micro_batch": self.micro_batch, "accum": self.accum})
        return True

    # ----------------------------------------------------------------- logging
    def _log_scalars(self, parts, lr, t0, start):
        elapsed = max(time.time() - t0, 1e-9)
        ips = (self.iteration - start) / elapsed
        rec = {"kind": "train", "iteration": self.iteration, "lr": lr,
               "it_per_s": round(ips, 3),
               **{k: round(v, 5) for k, v in parts.items()}}
        if torch.cuda.is_available():
            rec["vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        self.ledger.append(rec)

        alert = self.loss_spike_watchdog.observe(self.iteration, parts.get("total"))
        if alert:
            self.ledger.append({"kind": "watchdog", **alert})

    def _log_gates(self):
        """Append per-block gate values -- the raw data for the key figure."""
        backbone = getattr(self.model, "backbone", self.model)
        if not hasattr(backbone, "gate_report"):
            return
        cond = None
        if getattr(backbone, "cond_dim", None):
            # log one curve per task label so Phase 5 curves stay comparable
            cond = torch.eye(backbone.cond_dim, device=self.device)
        report = backbone.gate_report(cond)
        if not report:
            return
        with open(self.gate_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"iteration": self.iteration, "gates": report}) + "\n")

        alert = self.gate_stall_watchdog.observe(self.iteration, report)
        if alert:
            self.ledger.append({"kind": "watchdog", **alert})

    @torch.no_grad()
    def _validate(self):
        target = self.ema.module if self.ema is not None else self.model
        target.eval()
        metrics = self.val_fn(target)
        target.train()
        self.ledger.append({"kind": "val", "iteration": self.iteration, **metrics})
        key = self.cfg.get("select_metric", "psnr")
        if key in metrics and (self.best is None or metrics[key] > self.best):
            self.best = metrics[key]
            self._save(self.best_path, metrics=metrics)
        return metrics
