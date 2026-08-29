# CHECKPOINT — 2026-08-18, Phase 1 running

**Purpose of this file:** a rollback reference for training. If a later change
breaks something, this is what "known good" looked like — exact environment,
exact commands, exact measured numbers. Every number here was **measured on
this machine today**, not taken from the plan and not extrapolated. Where a
number is still a projection, it says so.

**Training status: Phase 1 QUEUE RUNNING**, launched 2026-08-18 ~19:52.

`scripts/run_segments.ps1` is working through all three arms in 10,000-iteration
segments with a 45 s cooldown between them:

```
dual -> channel_only -> window_only,  60,000 iterations each
```

At ~1.7 it/s that is ~10 h per arm, **~29 h total** — more than one night. It is
fully resumable: kill it any time, re-run the same command, and it continues
from the last checkpoint.

```powershell
$py = "$env:USERPROFILE\.conda\envs\datnet\python.exe"

# progress of any arm
& $py -c "import json;from datnet.engine.ledger import Ledger;print(json.dumps(Ledger('runs/phase1_denoise__dual__seed0/ledger.jsonl').summary(),indent=2))"

# restart / continue the whole queue
powershell -NoProfile -ExecutionPolicy Bypass -Command "& './scripts/run_segments.ps1' -Config 'configs/phase1_denoise.yaml' -Variants @('dual','channel_only','window_only') -SegmentIters 10000 -CoolDownSeconds 45 -Python '$py'"
```

**Launching the queue from bash needs `-Command`, not `-File`.** With `-File`,
PowerShell binds a bare `channel_only` to the next positional parameter
(`MaxHours`) instead of continuing the `-Variants` array, and the run dies
immediately. Both failure modes were hit and are recorded here so they are not
rediscovered at 3 a.m.

Config for this phase: pre-cropped DIV2K (`DIV2K_train_HR_sub`, 22,988 tiles),
`repeat: 1`, `micro_batch: 8`, `accum: 2`, 60,000 iterations, McMaster for
in-loop validation (see the note in the config — do NOT switch it to CBSD68
mid-phase).

Determinism check: at seed 0 the loss at iteration 100 is `0.09415` on every
launch. If a rerun disagrees, something in the data or seeding has changed.

---

## 1. Environment — the only interpreter that works

The system Python is **3.14**, which has no torch wheels. Training must use the
dedicated env; running `python` from a normal shell will silently be the wrong
interpreter and fail on `import torch`.

```
C:\Users\Mann Kuvadiya\.conda\envs\datnet\python.exe
```

| package | version |
|---|---|
| Python | 3.12.13 |
| torch | **2.13.0+cu126** |
| numpy | 2.5.2 |
| opencv-python-headless | 5.0.0.93 |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.1 |
| pillow | 12.3.0 |
| tqdm | 4.70.0 |

Recreate from scratch if ever needed:

```powershell
conda create -n datnet python=3.12 -y
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu126
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" -m pip install numpy opencv-python-headless PyYAML matplotlib tqdm
```

Verify in 2 seconds:

```powershell
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" -c "import torch; p=torch.cuda.get_device_properties(0); print(torch.__version__, p.name, round(p.total_memory/1024**3,2), f'sm_{p.major}{p.minor}', torch.cuda.is_bf16_supported())"
```

Must print: `2.13.0+cu126 NVIDIA GeForce RTX 4060 Laptop GPU 8.0 sm_89 True`

**Known harmless oddity:** `nvidia-smi` on this machine fails with
`Failed to initialize NVML: Not Found`. Torch CUDA works fine regardless — do
not chase this. It means VRAM cannot be watched via `nvidia-smi`; use the
`peak_vram_gb` field the trainer writes to `ledger.jsonl` instead.

---

## 2. Verified working — the four sanity checks

`scripts/sanity.py` → **4/4 PASS**, `tests/test_metrics.py` → **6/6 PASS**.

```powershell
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" tests/test_metrics.py
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" scripts/sanity.py
```

What each one proves, and why it matters if it ever goes red:

| check | result | if it breaks |
|---|---|---|
| shape round-trip 128×128, 137×91, 64×64, 255×129 | PASS | window padding or the U-Net `size_multiple` is wrong; full-image eval will silently crop |
| SR shape 48×64 → 192×256 | PASS | PixelShuffle tail or scale plumbing broken |
| **gate identity**: `dual` with g pinned to 1.0 equals `channel_only`, `max|diff| = 0.00e+00` | PASS | the fusion is not a true convex blend, or the two arms have different channel branches — **every ablation number becomes meaningless** |
| width matching within 2% | PASS (worst 1.25%) | arms are not parameter-matched; see §5 open issue |
| overfit one image, 600 it | **51.9 dB** | the training loop is broken; nothing downstream is trustworthy |

The gate-identity check is the load-bearing one. It is the only thing proving
that `dual` and `channel_only` differ by the spatial branch **and nothing else**.

---

## 2b. Verified against the official implementations — bit-exact

`tests/test_reference_equivalence.py` → **3/3 PASS**. This runs our blocks
against the real source in `reference/`, with weights copied across:

| ours | reference | max abs diff |
|---|---|---|
| `ChannelAttention` | Restormer `Attention` (MDTA) | **0.000e+00** |
| `GDFN` | Restormer `FeedForward` | **0.000e+00** |
| `WindowAttention` | SwinIR `WindowAttention` | **5.96e-08** (float32 noise) |

**Why this matters more than it looks.** The entire ablation rests on `dual`
differing from `channel_only` by the spatial branch *and nothing else*. That
argument is only as strong as the claim that our MDTA really is Restormer's
MDTA. It is now a test, not a claim. Re-run it after any edit to
`datnet/models/attention.py` or `ffn.py`.

The SwinIR check compares a single unshifted 8×8 window, converting between
their token-major `(B*nW, N, C)` layout and our NCHW. The shifted-window mask
path is *not* covered by it — that is exercised only by the shape checks in
`sanity.py`.

```powershell
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" scripts/fetch_reference.py --extract
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" tests/test_reference_equivalence.py
```

**Reference-only dependencies** — needed to import the reference files, never
imported by `datnet/`: `einops` 0.8.2, `timm` 1.0.28. Installing timm pulled
**`torchvision` 0.28.0+cpu**, a CPU-only build. Harmless here because nothing in
this project imports torchvision; do not be alarmed by it, and do not "fix" it
unless something starts needing torchvision GPU ops. `torch` itself is
untouched and still `2.13.0+cu126` with CUDA working.

---

## 2c. Training path verified end to end — including exact resume

`scripts/smoke_test.py` → **11/11 PASS**, runs in about a minute, needs **no
dataset**. It generates structured synthetic images, writes a throwaway config,
and shells out to the real `scripts/train.py` twice so the actual CLI path is
what gets tested.

```powershell
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" scripts/smoke_test.py
```

What it proves:

| | |
|---|---|
| dataset → DataLoader → Trainer step | works |
| a segment stops at its budget, exits **10** | works |
| **resume is exact** — segment 2 continued at iteration 20, did not restart | works |
| run completes, exits **0**, final iteration == 40 | works |
| `last.pth`, `best.pth`, `ledger.jsonl`, `gates.jsonl` all written | works |
| gate values logged per block (6 blocks, 8 snapshots) | works |
| in-loop validation ran (20.60 dB on synthetic data — meaningless value, working path) | works |

### Two real bugs it caught, both fatal to a first Phase-1 run

**1. Checkpoints could not be saved at all.** `Trainer` stored the whole config,
including `rebuild_loader` — a closure — and `torch.save` cannot pickle a local
function. *Every* run would have crashed at the first `ckpt_every`.
Fixed: the callable is popped into `self._rebuild_loader` and kept out of
`self.cfg`, which is what gets serialised.

**2. Two incompatible task vocabularies.** The dataset layer names tasks by
degradation family (`denoise`); the protocol table split denoising into
`denoise_gaussian` and `denoise_real`. `configs/phase1_denoise.yaml` says
`denoise`, so the first in-loop validation raised `KeyError: unknown task
'denoise'`. Fixed with an explicit `ALIASES` map in `protocols.py`: `denoise`
resolves to the synthetic protocol, and real-noise evaluation must be asked for
by name so it is never a silent default.

Neither would have appeared until a real run was hours in.

---

## 3. Measured model size — the plan's estimate was 3× too high

The plan (§3.2) estimated "~5–7 M parameters" at width 32. **Measured: 2.04 M.**
Nothing is wrong with the model; the estimate was simply wrong.

At the default geometry (`enc [2,3]`, bottleneck 4, `dec [3,2]`, refine 2,
heads [1,2,4], window 8, γ=2.66):

| arm | width | params | GFLOPs @ 256² |
|---|---|---|---|
| `dual` | 32 | **2,037,432** | 46.92 |
| `channel_only` | 36 | 2,062,834 (+1.25%) | 47.97 |
| `window_only` | 36 | 2,042,298 (+0.24%) | 43.30 |

Width sweep for `dual`, if the size is ever revisited:

| width | params | GFLOPs @ 256² |
|---|---|---|
| 32 | 2,037,432 | 46.9 |
| 40 | 3,149,828 | 72.2 |
| 48 | 4,506,076 | 103.0 |
| **56** | **6,102,792** | **139.1** |
| 64 | 7,945,196 | 180.9 |

Width **56** is what the plan actually described (~6 M). It has **3× the FLOPs**
of width 32 but, measured, costs only **1.7× the wall-clock** — see §4. This is
an open decision, and a much closer call than the FLOP ratio suggests — see §5.

> FLOP counts come from `torch.utils.flop_counter`. It warns that triton is not
> installed; that is irrelevant here because `torch.compile` is off and no
> triton kernels run. The counts are valid.

---

## 4. Measured throughput — ON MAINS POWER

```powershell
& "$env:USERPROFILE\.conda\envs\datnet\python.exe" scripts/bench.py --patch 128 --vram-ceiling 7.0
```

`bench.py` is **resumable**: each (arm, micro-batch) measurement is written to
the results JSON as it is taken, and re-running skips what is already there.
Interrupt it freely.

**These are synthetic-input numbers and are an UPPER BOUND.** It feeds
`torch.randn`, not images, and reuses the same tensor every step. VRAM and step
time depend only on tensor shapes, so the measurement is valid — but it excludes
disk, decode, augmentation, DataLoader workers and even the host-to-device copy.
Real training throughput can only be lower.

### Width 32 (2.04 M params) — `runs/bench_w32_ac.json`

patch 128², bf16, `effective_batch=16`, no gradient checkpointing:

| arm | micro-batch | peak VRAM | opt-it/s |
|---|---|---|---|
| `dual` | 4 | 1.91 GB | 2.03 |
| | **8** | **3.74 GB** | **2.14** ← best |
| | 16 | 7.41 GB | 1.44 (over ceiling) |
| `channel_only` | **8** | **2.94 GB** | **2.43** ← best |
| | 16 | 5.80 GB | 2.31 |
| `window_only` | 4 | 1.55 GB | 2.19 |
| | 8 | 3.03 GB | 2.19 |
| | 16 | 5.99 GB | 2.15 |

**Three arms × 100 k iterations = 37.1 GPU-hours.**

### Width 56 (6.10 M params) — `runs/bench_w56_capped.json`

| arm | micro-batch | peak VRAM | opt-it/s |
|---|---|---|---|
| `dual` | **8** | **6.43 GB** | **1.15** |
| | 16 | — | **OOM** |
| `channel_only` | **8** | 5.06 GB | 1.45 |
| `window_only` | **8** | 5.15 GB | 1.41 |

**Three arms × 100 k iterations = 63.1 GPU-hours.**

### Mains vs battery, and why the FLOP ratio misleads

Battery measurements (the earlier version of this file) had `dual` at 1.64
opt-it/s. On mains it is **2.14** — a **30% speedup**, VRAM identical. Any
throughput number taken on battery understates the card by roughly that much.

Width 56 has **3× the FLOPs** of width 32 but costs only **1.7× the wall-clock**
(63.1 vs 37.1 GPU-h). Width 32 does not saturate the card — its kernels are too
small and it is memory-bound — so the extra arithmetic is comparatively cheap.
An earlier projection in this file said width 56 would cost ~140 GPU-h; that was
extrapolated from the FLOP ratio and was **2.2× too pessimistic**. Do not size
runs from FLOPs on this card; measure.

---

## 4b. TRAP: this card does not OOM, it silently gets 20× slower

Measured at width 56, micro-batch 16: `torch.cuda.max_memory_allocated()`
reported **12.72 GB on an 8 GB card**, and throughput collapsed from 1.17 to
**0.06 opt-it/s**.

That is the Windows NVIDIA driver backing oversized allocations with **system
RAM** instead of raising `OutOfMemoryError`. The consequences are bad in a
specific way:

- `Trainer._backoff()` never fires, because there is no OOM to catch
- the run does not crash, it just looks like "training, a bit slow"
- at 20× slowdown, a night of queued segments produces almost nothing

**Fixed** by capping the allocator so OOM becomes honest again:

- `Trainer` calls `torch.cuda.set_per_process_memory_fraction(cfg['vram_fraction'])`,
  default **0.90**, before moving the model to the device
- `bench.py` takes `--vram-fraction`, same default

Verified after the fix: width 56 / micro-batch 16 now raises a clean OOM, and
the backoff path is reachable. **If you ever see a reported peak larger than
8 GB, or `it_per_s` an order of magnitude below the table above, this is the
cause — do not go looking for a slow dataloader.**

---

## 4c. RESOLVED: why real training is slower than the benchmark

An earlier version of this section blamed the every-step EMA update. **That was
wrong.** Measured, 300-iteration segments on pre-cropped DIV2K, width 32,
`dual`, validation and checkpointing disabled inside the timed window:

| variant | it/s |
|---|---|
| EMA on, grad clip on | **1.67** |
| EMA **off**, grad clip on | 1.64 |
| EMA off, grad clip **off** | 1.61 |

EMA costs nothing measurable. Neither does gradient clipping. The hypothesis was
plausible — it does issue a lot of small kernels — and simply did not survive
contact with a measurement. `torch._foreach_*` would be a tidy optimisation but
**it is not a fix for anything**; do not spend time on it.

DataLoader workers, same conditions:

| num_workers | it/s |
|---|---|
| 0 | 1.44 |
| 2 | 1.64 |
| 4 | **1.67** |

Saturates at 2 workers. The pipeline is not starving the GPU either — which is
consistent with pre-cropping having bought only 5%.

### What the gap actually decomposes into

| | it/s |
|---|---|
| `bench.py`, synthetic inputs, 12 timed steps | 2.14 |
| real training, val + checkpoint inside the window | 1.43-1.50 |
| real training, val + checkpoint disabled | **1.67** |

In-loop validation and checkpoint writes account for ~12%. The remaining ~22% is
**not attributable to any single software component** — it survives removing
EMA, clipping and worker starvation.

The most likely explanation is thermal: `bench.py` times 12 steps on a GPU that
has just been idle, so it runs at boost clocks, whereas sustained training
settles to lower clocks. That is a laptop reality, not a bug, and it is exactly
why `bench.py` is documented as an upper bound.

**Plan with 1.67 it/s, not 2.14.** No further investigation is warranted; the
remaining gap is small, understood in kind, and not actionable.

---

## 4d. FIXED: tiled inference had a black-border bug worth 5.3 dB

Found 2026-08-18 by cross-checking `report_phase.py` against in-loop validation
on the *same checkpoint*. They disagreed by 6 dB, which is far too large to be
noise:

| McMaster sigma=25, iteration 15,000 | PSNR |
|---|---|
| in-loop validation (`tile=None`, whole image) | 32.810 dB |
| report (`tile=256`, tiled) | 26.771 dB |
| report after the fix | **32.066 dB** |

**Cause.** `_feather` built its raised-cosine ramp with `linspace(0, 1, overlap)`,
so the first weight was exactly **0**. Interior pixels are covered by several
tiles and the `acc / wgt` normalisation recovers the correct value for any
positive weights -- but the outermost ring of pixels is covered by exactly ONE
tile. There `acc = 0` and `wgt = 0`, the `clamp(min=1e-8)` turned it into
`0 / 1e-8`, and the output was a **black ring one pixel wide**.

A 1 px ring is ~1% of a 481x321 image, which adds ~0.0025 to an MSE of ~6.3e-4
and predicts ~25.1 dB against 25.67 observed. The arithmetic accounted for the
entire gap.

**Fix.** Sample the cosine on the open interval (0, 1) so every weight is
strictly positive: `t = (arange(overlap) + 1) / (overlap + 1)`.

**Why this mattered more than one report.** Tiled inference is the path *every*
final evaluation uses -- Urban100 and Manga109 cannot be done whole-image on
8 GB. Left alone this would have silently depressed every reported number in all
five phases, uniformly enough to look like "our model is just a bit worse"
rather than like a bug.

**The check that caught it is worth repeating:** evaluate one checkpoint both
whole-image and tiled and require agreement. Any tiling change should be
re-validated that way.

The residual 0.74 dB between validation (32.810) and the report (32.066) is not
a bug: validation uses `max_images: 8`, the report uses all 18 McMaster images.

---

## 4e. THE BIG ONE: Phase 1 diverged and the watchdog did not notice

The first real 60,000-iteration run **destroyed itself** and kept going for
40,000 more iterations. Recorded here in full because three separate defences
failed and each fix matters.

| iteration | McMaster PSNR |
|---|---|
| 15,000 | **32.810 dB** (healthy) |
| 20,000 | 20.437 dB (collapsed, roughly noisy-input quality) |
| 25,000 - 60,000 | **NaN** |

`last.pth` at 60,000: **248/248 tensors non-finite**. `best.pth` at 15,000:
0/248 non-finite, and it is the only survivor -- saved purely by accident,
because `NaN > best` evaluates False so checkpoint selection never overwrote it.

### Why nothing caught it

**1. The GradScaler was disabled, and it was the safety net.** With `amp: bf16`
no loss scaling is needed, so `GradScaler(enabled=False)`. But the scaler is
ALSO what skips optimiser steps on non-finite gradients. Disabling it silently
removed that protection.

**2. `grad_clip` made it worse, not better.** `clip_grad_norm_` with a NaN
gradient computes `total_norm = NaN`, then `clip_coef = max_norm / NaN = NaN`,
and multiplies every gradient by NaN. Clipping *spreads* NaN.

**3. The watchdog only watched for crashes.** `train.py` never crashed -- it
exited 10 cleanly after every segment while training a corpse. `0 retries` was
not the watchdog failing to fire; there was no crash to catch. The whole design
was aimed at the loud, rare failure and blind to the quiet, common one.

### The finding that made the first fix wrong

The obvious fix -- check gradients before stepping -- **would not have worked**.
Measured directly: with L1 loss and a NaN target, the loss is `nan` while all 98
parameter gradients come back **finite**. A NaN loss does not reliably produce
NaN gradients.

The loss is the signal that actually appeared (`loss=nan` in the ledger for 40k
iterations), so the loss is what must be checked. Verified by a deliberate
divergence test.

### What is in place now

- **`train_step` checks `torch.isfinite(loss)`** per micro-batch and skips the
  entire optimiser step if any micro-batch is non-finite -- a partial gradient
  from a step that saw NaN is not worth taking
- **gradient finiteness is checked too**, as a second line of defence
- **`_save` refuses to write non-finite weights**, so a good checkpoint can
  never be overwritten by a dead one (the old code wrote NaN over `last.pth` 30
  times)
- **`DivergedError` after `max_nonfinite_streak` (50) consecutive bad steps**,
  surfacing as **exit code 3**; `run_segments.ps1` treats 3 as fatal and does
  NOT retry, because resuming reloads the same dead state
- **peak LR lowered 3e-4 -> 2e-4.** 3e-4 is Restormer's figure at batch **64**;
  this project runs effective batch **16**

Verified: the guard aborts at the streak limit with weights still finite and no
NaN checkpoint written, and `smoke_test.py` still passes 11/11.

### Also fixed: the watcher fired 28,000 iterations early

`watch_and_report.ps1` treated "no `train.py` process alive" as proof the run had
stopped -- but `run_segments.ps1` leaves a 45 s cooldown between segments with no
trainer running. It reported at iteration 32,000 of 60,000. It now requires
several consecutive empty polls before believing the run is over.

---

## 4f. PHASE 1 RESULT: the gate is decorative -- fix before Phase 2

Phase 1 completed all three arms at 30,000 iterations, parameter-matched
(2,037,432 / 2,062,834 / 2,042,298, all within 1.25%). Verdict: **FAIL**, but the
verdict is not the finding.

### The measured numbers (CBSD68, all arms from last.pth at 30,000)

| arm | sigma 15 | sigma 25 | sigma 50 |
|---|---|---|---|
| `dual` | 33.602 | 31.081 | 27.852 |
| `channel_only` | 33.761 | 31.166 | 27.944 |
| `window_only` | **33.777** | **31.191** | **27.974** |

`dual` is worst at every noise level by 0.11-0.18 dB, and the ordering
`window_only > channel_only > dual` is consistent across all three sigmas.

**Compare arms at EQUAL duration.** `best.pth` selection picked iteration 22,500
for `dual` and 30,000 for the others, because in-loop validation uses only 8
McMaster images and swings ~0.2 dB. Using `best.pth` made the spread look like
0.025 dB; using `last.pth` for everything shows 0.11-0.18 dB. Never compare
checkpoints chosen by a noisy criterion.

### Why the verdict does not test the hypothesis

**The gate never moved.** Mean g went 0.5375 -> 0.5429 over 30,000 iterations --
a change of **0.005**. Every block sat on its initialisation (0.8 / 0.5 / 0.2).

It is not a wiring bug: the gate receives gradient of mean magnitude 4.7e-04
against 7.3e-04 for other parameters (ratio 0.64). Weight decay accounts for
~8e-4 of drift over the whole run -- negligible.

**The cause is architectural.** `out = g*A + (1-g)*B`, and A and B each end in
their own learnable 1x1 projection, so the network can produce any effective
blend by rescaling the branches and leaving g wherever it started. Measured on
the trained model:

| block | g says | raw \|A\|/\|B\| | effective \|gA\|/\|(1-g)B\| |
|---|---|---|---|
| enc1.blocks.0 | 0.800 (80% channel) | 0.22 | **0.87** (~47/53) |
| dec1.blocks.1 | 0.803 (80% channel) | 0.03 | **0.13** (spatial 8x) |
| median over 16 blocks | -- | **0.28** | 0.32 |

The gate value does **not** describe the real channel/spatial balance. Since the
axis-preference figure is the entire contribution of this project, plotting `g`
would have plotted a number that means nothing.

**So `dual` in Phase 1 was a FIXED-blend model** -- precisely the thing
X-Restormer already does and that DATNet exists to improve on. Phase 1 did not
test the hypothesis; it compared a fixed 0.8/0.5/0.2 blend against two
single-axis models, and the fixed blend lost.

### What has to change before Phase 2 is worth running

1. **Make the gate the only way to change the balance.** Normalise each branch
   output (RMS or LayerNorm) before gating so the two arrive at comparable
   scale, with a learnable per-block scalar afterwards to restore magnitude.
   Then g controls a real quantity.
2. **Report the effective ratio, not g.** `|gA| / (|gA| + |(1-g)B|)` is what the
   network actually does, and it can be computed retroactively from any existing
   checkpoint (`scripts/_branch_balance.py`).
3. **Re-run Phase 1 only after 1 is in.** The current FAIL is uninformative
   about the hypothesis.

Do not read the Phase-1 FAIL as "channel and spatial attention do not combine".
It says "a fixed blend with a decorative gate loses to either pure branch",
which is a statement about the implementation, not the idea.

---

## 4g. FIXED: branch balancing makes the gate mean something

Items 1 and 2 from §4f are implemented and tested. Item 3 (re-run Phase 1) is
NOT done -- it needs mains power.

### 1. Branch balancing (`DATB._balance`)

Both attention outputs are rescaled to a **common magnitude** (the mean of their
two per-sample RMS values) before the gate blends them. The ratio |A|/|B| is
then exactly 1, so the network can no longer shift the blend by rescaling a
branch behind the gate.

**First attempt was wrong and the tests caught it.** Normalising each branch to
*unit* RMS also works for the balance, but the block output feeds a residual
add and must be able to emit a SMALL correction when little is needed. Forcing
unit RMS removed that freedom: the single-image overfit check fell **51.9 dB ->
34.1 dB**. Scaling to the shared mean instead keeps the magnitude
data-dependent: overfit is back to **52.3 dB**.

Three properties that make this safe for the ablation:

- **No new parameters.** Counts are unchanged (2,037,432 / 2,062,834 /
  2,042,298), so the existing width matching still holds.
- **No-op for a single branch.** With one branch m == rms(a), so `_balance`
  leaves it untouched -- the single-axis arms are bit-identical to before and
  `dual` still differs from `channel_only` by the spatial branch and nothing else.
- **Gradient to the gate is ~15x stronger**: 4.7e-04 -> 7.1e-03 in the same
  measurement.

Measured effect, `tests/test_gate_controls_blend.py`:

| | 10x rescale of one branch |
|---|---|
| balanced (new) | effective blend drifts **0.0001** |
| un-balanced (old) | effective blend drifts **0.5171** (0.213 -> 0.730) |

### 2. Report the effective ratio (`scripts/effective_gate.py`)

```powershell
& $py scripts/effective_gate.py --run-dir runs/<run> --checkpoint last
```

Prints nominal `g` beside the effective ratio |gA| / (|gA| + |(1-g)B|) per block,
and **warns loudly** when they disagree by more than 0.05. Run on the old
un-balanced Phase-1 checkpoint it reports:

```
  mean effective ratio   = 0.2587   (nominal mean g was 0.5429)
  largest |effective-g|  = 0.6981   (dec1.blocks.1: g=0.803, actual 0.105)
```

**That number is itself a result.** The trained network ran at an effective
**26% channel / 74% spatial** for Gaussian denoising -- and `window_only` (pure
spatial) was the best arm. The two agree. The plan predicted noise would be
CHANNEL-leaning; the measurement says the opposite.

Treat that as a hypothesis-relevant signal, not a conclusion: it was measured on
a model whose gate could not express a preference. The re-run with balancing is
what tests it properly.

### The sanity check had to change, and this is why

`gate identity` previously asserted that `dual` with g pinned to 1.0 is
numerically identical to `channel_only`. With balancing that is false by design:
at g=1 the block emits `a * (ra+rb)/(2*ra)` -- same branch, same direction,
deliberately different magnitude.

It now tests at block level, by **direction**: cosine similarity between the
fused output and the target branch at g=1 and g=0 (both 1.000000), plus an exact
check that the channel branch module is identical across arms (max|diff| = 0).
Asserting exact output equality would have been asserting that the balancing
does not happen.

### Regression after the change

`sanity.py` 4/4 - `test_metrics.py` 6/6 - `test_reference_equivalence.py` 3/3 -
`test_gate_controls_blend.py` 4/4 - `smoke_test.py` 11/11.

**Old checkpoints still load.** `report_phase.load_model` infers
`normalise_branches` from whether the weights contain `attn_scale`/balancing
rather than trusting a config that predates the option, so pre-fix runs remain
readable for comparison.

---

## 4h. Prior art checked against the SOURCE, not the abstract

The plan described X-Restormer incorrectly, and the error was load-bearing for
the positioning. Corrected 2026-08-20 by reading
`reference/X-Restormer-master/xrestormer/archs/xrestormer_arch.py`.

**The plan claimed** it "alternately replaced half of the MDTA blocks with OCA
blocks", with a balance "fixed 50/50 by construction".

**The code shows** every block contains both attentions, applied sequentially at
full strength, and every level of the U-Net uses that same block:

```python
x = x + channel_attn(norm1(x))     # TransformerBlock.forward
x = x + channel_ffn(norm2(x))
x = x + spatial_attn(norm3(x))
x = x + spatial_ffn(norm4(x))
```

Not alternating. And critically, **there is no 50/50 to fix** -- X-Restormer has
no weight, gate or mixing coefficient anywhere. Nothing in it represents how
much each axis contributes.

### Why this matters more than a wording fix

It changes what DATNet can claim. "We combine channel and spatial attention" is
**not** novel -- X-Restormer, DAT (ICCV 2023) and HAT all do it, and DAT already
owns the name. The defensible claim is narrower and sharper:

> DATNet makes the channel/spatial balance an **explicit, measurable quantity**.
> X-Restormer does not have one to compare against.

That puts all the weight on the gate actually moving between degradations, which
is exactly what Phase 1 vs Phase 2 is designed to test -- and what the decorative
gate (§4f) meant had not been tested at all.

Also worth noting: X-Restormer spends **two FFNs per block**, one after each
attention. DATNet's DATB uses one. Any FLOP or parameter comparison should say so.

**Rule this establishes: check prior art against the source, not the abstract.**
The paper's description and its implementation differed enough to change the
project's positioning.

---

## 5. Open decisions — do not start Phase 1 without settling these

**5a. Model size: stay at 2.04 M, or go to 6.10 M?** Both now measured.

| | width 32 | width 56 |
|---|---|---|
| params | 2.04 M | **6.10 M** |
| GFLOPs @ 256² | 46.9 | 139.1 (3.0×) |
| best micro-batch | 8 | 8 |
| peak VRAM (`dual`) | 3.74 GB | 6.43 GB |
| 3 arms × 100 k | **37.1 GPU-h** | **63.1 GPU-h** (1.7×) |

The FLOP ratio said 3×; the clock says 1.7×, because width 32 leaves the card
underused. **Width 56 is a far better deal than it looked**, and it is what the
plan actually describes.

Against it: 6.43 GB peak leaves only ~0.6 GB of headroom under the 7.0 GB
ceiling, so width 56 has no room for a larger patch, gradient checkpointing off,
or the Phase-5 model with its four tails. And 63 GPU-h per phase × 3 phases is
~190 GPU-h for the core study.

**Recommendation: run Phase 1 at width 32.** It is the cheap kill-switch phase
and its job is to rank three arms, not to set a record. If the hypothesis
survives, move to width 56 for Phase 2 onward, where the numbers are the ones
that get reported. Whichever is chosen, state it — do not let 2.04 M be an
accident of a wrong estimate.

**5b. Width matching at width 56 — SOLVED.** It previously raised:

```
cannot match window_only to 6,102,792 params within 2% on the width grid
(best: width=62, params=5,956,988, error=2.39%)
```

Parameters grow quadratically, so one step of the width grid is worth ~6% of the
target at that scale — width alone is too coarse there.

**Fixed** by adding `ffn_expansion` (γ) as a fine second knob in
`datnet/utils/matching.py`. γ scales the GDFN hidden width, where most
parameters live, and moves the count in much smaller increments. `match_arm()`
tries width first and only refines γ when width alone misses `tol`; the result
records which knob was used (`"width"` or `"width+gamma"`), so tables can report
it. Depth is still never touched.

At width 32 γ is not needed at all — both arms match on width alone (1.25% and
0.24%).

---

## 6. What has and has not run

**Now exercised** (by `scripts/smoke_test.py`, on synthetic data):
`Trainer`, checkpoint save/resume with RNG state, EMA, the cosine schedule,
`GaussianDenoiseDataset`, the DataLoader path, in-loop validation, the ledger
and the gate log.

**Still never executed:**

- `PairedDataset` (GoPro, Rain13K) and `SRDataset` — only the denoise path ran
- `scripts/train_multitask.py` and the whole Phase-5 model
- `datnet/evaluation/inference.py` — tiling and the OOM fallback
- `datnet/data/bicubic.py` — the MATLAB-compatible resize
- `scripts/prepare_subimages.py`
- `scripts/plot_gates.py` — needs a real `gates.jsonl` with more than 8 points
- `scripts/run_segments.ps1`
- the OOM backoff path in `Trainer._backoff()` — reachable now that the
  allocator is capped (§4b), but not yet triggered in anger

**Caveat on the smoke test.** It used `num_workers: 0`, a 16-wide model and
64² patches. It proves the logic is wired correctly; it does not prove the
multi-worker DataLoader behaves on Windows spawn, which is the thing most likely
to bite at real scale.

---

## 7. Fastest path back to a known-good state

```powershell
$py = "$env:USERPROFILE\.conda\envs\datnet\python.exe"
& $py -c "import torch; print(torch.cuda.is_available())"   # True
& $py tests/test_metrics.py                                  # 6/6
& $py scripts/sanity.py                                      # 4/4, overfit > 50 dB
& $py tests/test_reference_equivalence.py                    # 3/3 vs Restormer/SwinIR
& $py scripts/smoke_test.py                                  # 11/11, training path
```

If those three are green, the model and the evaluation protocol are intact and
any problem is in data, training or config.

## 8. File manifest

```
datnet/models/    attention.py (MDTA + shifted-window MSA), gate.py (AxisGate),
                  block.py (DATB), datnet.py (backbone + tails), allinone.py,
                  conditioning.py, ffn.py, layers.py, build.py
datnet/data/      datasets.py, bicubic.py (MATLAB imresize), transforms.py,
                  multitask.py, build.py, io.py
datnet/engine/    trainer.py (segmented loop), checkpoint.py (atomic + RNG),
                  ledger.py, losses.py, ema.py, scheduler.py
datnet/evaluation/ protocols.py (the per-task convention table), metrics.py,
                  inference.py, evaluate.py
datnet/utils/     complexity.py, matching.py
configs/          phase1_denoise, phase2_deblur, phase3_sr_x4, phase4_derain,
                  phase5_allinone
scripts/          sanity.py, bench.py, train.py, train_multitask.py,
                  prepare_subimages.py, plot_gates.py, run_segments.ps1
tests/            test_metrics.py
```
