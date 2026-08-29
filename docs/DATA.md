# Datasets — what to download and where it must go

The paths below are **not suggestions** — they are what the YAML configs in
`configs/` already point at. Either match this layout or edit the configs.

**Already on disk** (downloaded 2026-08-18): DIV2K train HR (800), DIV2K valid
HR (100), DIV2K train LR bicubic X4, plus McMaster (18) and Set5 copied out of
the SwinIR reference repo into `data/test/`.

**Still missing**: CBSD68, Kodak24, Urban100 (Phase 1 eval), GoPro (Phase 2),
Set14/BSD100/Manga109 (Phase 3 eval), Rain13K (Phase 4).

Total disk for everything here is well under 30 GB. There is 442 GB free.

---

## Layout the configs expect

```
data/
  DIV2K/
    DIV2K_train_HR/                 800 PNGs        Phase 1 + Phase 3
    DIV2K_train_LR_bicubic/X4/      800 PNGs        Phase 3
    DIV2K_train_HR_sub/             generated       see "Pre-cropping" below
  GoPro/
    train/blur/    train/sharp/     2,103 pairs     Phase 2
    test/blur/     test/sharp/      1,111 pairs
  Rain13K/
    train/input/   train/target/    13,712 pairs    Phase 4
    test/Rain100L/input/  test/Rain100L/target/
  test/
    CBSD68/    Kodak24/    Urban100/                denoising eval
    Set5/      Set14/      BSD100/     Manga109/    SR eval
```

`PairedDataset` matches degraded and clean images **by sorted filename order**,
which is how all of these ship. It raises immediately on a count mismatch, so a
half-extracted archive fails at startup rather than after six hours of training
on misaligned pairs.

---

## Priority order

Download only what the current phase needs. One DIV2K download covers Phases 1
and 3, which is why denoising is first.

| priority | what | size | needed for |
|---|---|---|---|
| 1 | DIV2K train HR | ~3.3 GB | Phase 1 (noise is synthesised on the fly) |
| 1 | CBSD68, Kodak24, Urban100 | tens of MB | Phase 1 eval |
| 2 | GoPro | ~10 GB | Phase 2 |
| 3 | DIV2K train LR bicubic X4 | ~0.5 GB | Phase 3 |
| 3 | Set5, Set14, BSD100, Manga109 | tens of MB | Phase 3 eval |
| 4 | Rain13K | ~1 GB | Phase 4 (optional) |
| later | SIDD | ~12 GB | only if the synthetic-noise result holds |

**DIV2K is automated and resumable.** It is 3.5 GB and this laptop may lose
mains power mid-transfer, so `fetch_data.py` uses HTTP Range requests —
interrupt it, re-run the same command, and it continues from the byte it
reached:

```powershell
$py = "$env:USERPROFILE\.conda\envs\datnet\python.exe"
& $py scripts/fetch_data.py --extract          # HR + LR bicubic X4 + valid HR
& $py scripts/fetch_data.py --only DIV2K_train_HR
```

**Everything else is manual, deliberately.** GoPro and Rain13K ship through
Google Drive links that rotate and need confirmation tokens; a scripted download
of those fails by handing you a truncated archive that looks fine, which is
worse than doing it by hand.

**Sources.** DIV2K is at `https://data.vision.ee.ethz.ch/cvl/DIV2K/`. GoPro and
Rain13K are most reliably taken from the Restormer or MPRNet release links
rather than the original authors' pages, because those ship the exact
train/test split the published numbers use. The classical SR and denoising test
sets are bundled together in most SR repos. Verify file counts against the
table above after extracting — a short archive is the single most common cause
of a run that trains on nonsense.

---

## Two things that will silently corrupt results

**1. Do not generate the SR LR images with PIL or `cv2.INTER_CUBIC`.**

Every SR benchmark number in the literature is produced against LR images made
by MATLAB `imresize` with antialiasing. PIL BICUBIC and `cv2.INTER_CUBIC` are
different kernels and produce LR images that are systematically *easier* —
inflating PSNR by roughly 0.2–0.5 dB and making the comparison meaningless.

Use the **official `DIV2K_train_LR_bicubic/X4` folder** when it is on disk.
`SRDataset` falls back to `datnet/data/bicubic.py`, which is a
MATLAB-compatible implementation, only when `lr_root` is null. That fallback is
correct but costs CPU per sample.

**2. Evaluation conventions differ per task and are not interchangeable.**

SR and deraining report **Y channel of YCbCr**; denoising and deblurring report
**RGB**; SR additionally shaves `scale` pixels from each border. This is
encoded once in `datnet/evaluation/protocols.py` and pinned by
`tests/test_metrics.py`. Never hardcode it at a call site.

---

## Pre-cropping DIV2K (do this before Phase 1)

**Measured — and the result was NOT what was expected.** Decoding one full-res
DIV2K PNG to take a single 128² patch costs **78 ms**, which looks like an
obvious bottleneck. It is not:

| | it/s (200-iter Phase-1 segment) |
|---|---|
| uncropped, full-res PNGs | 1.43 |
| pre-cropped to 480×480 | **1.50** |

Cropping cut decode work roughly 10× and bought **5%**. A worker sweep confirms
it: throughput saturates at `num_workers: 2` (1.44 / 1.64 / 1.67 for 0 / 2 / 4
workers). The data pipeline is **not** the constraint — see `CHECKPOINT.md` §4c
for where the time actually goes.

Pre-cropping is still worth doing — 8.8 GB for 5% throughput and much lower RAM
pressure, which matters at 15.4 GB with spawn workers — but it is **not** the
big win this file originally claimed.

```powershell
$py = "$env:USERPROFILE\.conda\envs\datnet\python.exe"
& $py scripts/prepare_subimages.py --input data/DIV2K/DIV2K_train_HR `
    --output data/DIV2K/DIV2K_train_HR_sub --size 480 --step 240
```

Then point `configs/phase1_denoise.yaml` at `DIV2K_train_HR_sub` and drop
`repeat` from 20 to 1–2, since the sub-image set is already ~10× larger.

Disk cost: DIV2K HR goes from ~3.3 GB to roughly 8–10 GB.

**For SR, crop HR and LR separately with size and step divided by the scale**, so
tiles stay aligned:

```powershell
& $py scripts/prepare_subimages.py --input data/DIV2K/DIV2K_train_LR_bicubic/X4 `
    --output data/DIV2K/DIV2K_train_LR_bicubic/X4_sub --size 120 --step 60
```

---

## Validating a download in one command

```powershell
$py = "$env:USERPROFILE\.conda\envs\datnet\python.exe"
& $py -c "import sys; sys.path.insert(0,'.'); from datnet.data.build import build_train_dataset; import yaml; d=build_train_dataset(yaml.safe_load(open('configs/phase1_denoise.yaml'))['data']); print(len(d), 'samples'); b=d[0]; print({k: tuple(v.shape) for k,v in b.items() if hasattr(v,'shape')})"
```

Should print a sample count and `{'input': (3,128,128), 'target': (3,128,128)}`.

Verified 2026-08-18: **16000 samples** (800 images × repeat 20), correct shapes,
78 ms per sample. A full 200-iteration training segment on this data completed
at 1.43 it/s with a 3.9 GB peak.
