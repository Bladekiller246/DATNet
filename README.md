# DATNet

Dual-Axis Transformer for image restoration. See [DATNet_Project_Plan.md](DATNet_Project_Plan.md)
for the research plan; this file covers running the code.

## State of the code

**Phase 0 green. Phase 1 has run once and its result is invalid** — the gate was
decorative, so `dual` was a fixed-blend model rather than the architecture the
hypothesis describes. Fixed and tested; the re-run has not happened.

| check | result |
|---|---|
| `scripts/sanity.py` | 4/4 — shapes, gate identity, width matching, 52.3 dB overfit |
| `tests/test_metrics.py` | 6/6 — evaluation protocol conventions |
| `tests/test_reference_equivalence.py` | 3/3 — **bit-exact** vs Restormer MDTA/GDFN, SwinIR window MSA |
| `tests/test_gate_controls_blend.py` | 4/4 — the gate, and only the gate, sets the axis balance |
| `scripts/smoke_test.py` | 11/11 — full training path incl. exact resume |

Measured on the RTX 4060 Laptop, on mains power:

- `dual` is **2.04 M params** at width 32; width **56** reaches 6.10 M. Three
  arms × 100 k: **37.1 GPU-h** at width 32, **63.1** at 56 — 3× the FLOPs but
  only 1.7× the clock.
- best micro-batch **8, accum 2**, at both widths
- real training runs ~**1.9 it/s**, against a 2.14 synthetic-benchmark ceiling
- **this card does not OOM, it silently runs 20× slower** — the Windows driver
  spills to system RAM. The allocator is now capped so OOM is honest; see
  [`docs/CHECKPOINT.md`](docs/CHECKPOINT.md) §4b before chasing any slowdown.

Still unexercised: `SRDataset`, all of Phase 5, and `train_multitask.py`.

## Documents

| file | what it is |
|---|---|
| [`DATNet_Project_Plan.md`](DATNet_Project_Plan.md) | the research plan — hypothesis, prior art, experiment design |
| [`docs/PROMPT.md`](docs/PROMPT.md) | working brief — decided rules, commands, work queue, open questions |
| [`docs/CHECKPOINT.md`](docs/CHECKPOINT.md) | rollback reference — measured numbers, verified checks, what has never run |
| [`docs/DATA.md`](docs/DATA.md) | dataset sources, required layout, pre-cropping |

## Setup

**The `datnet` env already exists** — torch 2.13.0+cu126 on Python 3.12.13. The
system Python is 3.14 and has no torch wheels, so every command must go through
the env interpreter explicitly; a bare `python` will be the wrong one.

```powershell
$py = "$env:USERPROFILE\.conda\envs\datnet\python.exe"
& $py -c "import torch; p=torch.cuda.get_device_properties(0); print(torch.__version__, p.name, round(p.total_memory/1024**3,2), torch.cuda.is_bf16_supported())"
```

Expected: `2.13.0+cu126 NVIDIA GeForce RTX 4060 Laptop GPU 8.0 True`.

To rebuild it from scratch, see [`docs/CHECKPOINT.md`](docs/CHECKPOINT.md) §1.
Note that `nvidia-smi` fails on this machine with `Failed to initialize NVML`;
torch works regardless, so read VRAM from the trainer's `ledger.jsonl` instead.

## Before training anything

```powershell
& $py tests/test_metrics.py                    # evaluation protocol conventions
& $py scripts/sanity.py                        # shapes, gate identity, width matching, overfit
& $py scripts/fetch_reference.py --extract     # official implementations, into reference/
& $py tests/test_reference_equivalence.py      # our MDTA/GDFN/window-MSA vs theirs
& $py scripts/bench.py --patch 128 --vram-ceiling 7.0
```

`test_reference_equivalence.py` is the one that proves the ablation is honest:
our MDTA and GDFN are **bit-exact** against Restormer, and our window attention
matches SwinIR to float32 noise. Re-run it after touching
`datnet/models/attention.py` or `ffn.py`.

`bench.py` writes `runs/bench.json` with the *fastest* micro-batch that fits
under the VRAM ceiling for each ablation arm, the measured optimiser-steps/second,
and the projected wall-clock. Throughput peaks at micro-batch 8 and falls off at
16 on this card, so fastest and largest are not the same thing. Put those numbers into the phase configs before
starting a real run -- everything in the plan's timeline depends on them.

## Training

Runs are executed as bounded **segments**. Each `train.py` invocation resumes
from the last checkpoint, trains at most `--segment-iters` iterations, writes a
checkpoint plus a ledger entry, and exits.

```powershell
# one segment by hand
& $py scripts/train.py --config configs/phase1_denoise.yaml --variant dual --segment-iters 10000

# or queue all three arms and walk away; Ctrl-C is safe at any point
.\scripts\run_segments.ps1 -Config configs\phase1_denoise.yaml -SegmentIters 10000 -MaxHours 6
```

Exit codes: `0` run complete, `10` segment done and more remain, **`3` diverged** (the runner stops rather than retrying a dead model), `1` error.

Phase 5 uses its own driver:

```powershell
& $py scripts/train_multitask.py --config configs/phase5_allinone.yaml --conditioning oracle --segment-iters 10000
```

## Progress and results

```powershell
& $py -c "import json;from datnet.engine.ledger import Ledger;print(json.dumps(Ledger('runs/phase1_denoise__dual__seed0/ledger.jsonl').summary(),indent=2))"
```

The key figure:

```powershell
& $py scripts/plot_gates.py runs/phase1_denoise__dual__seed0 runs/phase2_deblur__dual__seed0 `
  --labels denoise deblur --out figures/gate_flip.png
```

## Layout

| Path | Contents |
|---|---|
| `datnet/models/` | `attention.py` (MDTA + shifted-window MSA), `gate.py` (the axis gate), `block.py` (DATB), `datnet.py` (backbone + tails), `allinone.py` (Phase 5) |
| `datnet/data/` | datasets, MATLAB-compatible bicubic, D4 augmentation, multi-task sampling |
| `datnet/engine/` | segmented trainer, EMA, losses, schedule, atomic checkpoints, run ledger |
| `datnet/evaluation/` | per-task protocol table, PSNR/SSIM, tiled full-image inference |
| `datnet/utils/` | parameter/FLOP counting, width matching across ablation arms |
| `configs/` | one YAML per phase |
| `scripts/` | sanity, bench, train, train_multitask, plot_gates, run_segments.ps1 |
