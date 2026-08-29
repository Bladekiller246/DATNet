# DATNet — working brief

Personal working notes. The formal write-up lives in
[`DATNet_Project_Plan.md`](../DATNet_Project_Plan.md); this file is the
operational counterpart — what the project is for, what is actually decided,
what to run, and what is still open. Measured state as of any given day is in
[`CHECKPOINT.md`](CHECKPOINT.md).

---

## 1. The claim, in one paragraph

Channel-axis attention (Restormer's MDTA) and spatial-axis attention (SwinIR's
shifted windows) capture complementary structure, and **the degradation type
determines which axis matters more**. No existing model attends over both axes
*and* adapts the balance to the degradation. DATNet puts both branches in one
block and blends them with a learned per-channel gate `g`, then conditions that
gate on the degradation.

Prediction: additive Gaussian noise is i.i.d. per pixel with no spatial
structure, so the gate should lean **channel**. Blur -- motion, defocus or plain
softness -- is a spatial convolution, so it should lean **spatial**. If the gate
flips direction between those two, that plot is the paper.

**Scope: "denoising" means all noise families, "deblurring" means all blur.**
Noise covers additive Gaussian, Poisson/shot, and multiplicative speckle (both
the Gaussian approximation and the gamma model). Blur covers Gaussian softness,
defocus, and camera motion.

That widening buys a sharper sub-hypothesis for free. The channel prediction
rests on noise being **signal-independent** -- but speckle and Poisson are
signal-*dependent*, so they inherit spatial structure from the image. If the
gate leans channel for Gaussian and not for speckle, the axis preference is
tracking **signal-dependence**, not the noise-versus-blur split. Same machinery,
same figure, one more line, better claim.

## 2. Why this is not just three papers stapled together

**X-Restormer (ECCV 2024) is the closest prior art and must be read first.** It
already alternates MDTA blocks with spatial (overlapping cross-attention) blocks
in one backbone, and already covers all five tasks including SR. It does not
kill the idea — its own motivating finding, that no single backbone generalises
across tasks, is exactly the gap here, now peer-reviewed instead of assumed.

What is left:

| | X-Restormer | DATNet |
|---|---|---|
| combination | **alternating** blocks, fixed 50/50 by construction | **parallel** branches, learned per-channel gate |
| axis balance | identical for every task | learned, measured, and conditioned on the degradation |
| training | **one checkpoint per task** | single all-in-one checkpoint |
| SR in the all-in-one setting | n/a (per-task) | included |

PromptIR/AirNet are all-in-one but **exclude SR** (confirmed from
`options.py`). X-Restormer covers SR but trains per-task, and has **no balance
quantity at all** — no gate, no weight, nothing representing how much each axis
contributes (confirmed from its source). A single checkpoint doing denoise +
deblur + derain + **SR** is the thing nobody has shipped.

**Verified at field level, 2026-08-20.** The IEEE TPAMI 2025 all-in-one survey
covers denoise/derain/dehaze/deblur/low-light/weather and does **not** include
super-resolution in its taxonomy, nor highlight any method combining SR with
other degradations in one model.

Two claims survive scrutiny; two do not:

| claim | status |
|---|---|
| "we combine channel + spatial attention" | **crowded** — X-Restormer, DAT, HAT |
| "we condition on inferred degradation" | **partly taken** — PromptIR's PromptGenBlock does this, with opaque prompts |
| "the channel/spatial balance is explicit and measurable" | **holds** — X-Restormer has no such quantity |
| "one checkpoint including SR" | **holds** — absent from the 2025 survey |

**Two cheap things must be true, and both get checked early:**

1. The axis preference is measurably task-dependent (Phase 2 tests this).
   If the gate lands in the same place for noise as for blur, X-Restormer's
   fixed alternation was already right and there is nothing to condition on.
2. All-in-one *including SR* is actually hard. Confirm with a joint-training
   smoke test rather than assuming.

If reading X-Restormer in full shows it is closer than the table above,
**pivot in week 1, not at submission.** The all-in-one-with-SR angle survives
almost any such discovery because it is a different problem setting.

## 3. Environment and commands

Everything runs through the dedicated env — the system Python is 3.14 and has
no torch. Full details and versions in [`CHECKPOINT.md`](CHECKPOINT.md) §1.

```powershell
$py = "$env:USERPROFILE\.conda\envs\datnet\python.exe"
```

| purpose | command |
|---|---|
| protocol tests | `& $py tests/test_metrics.py` |
| Phase-0 sanity | `& $py scripts/sanity.py` |
| **training path smoke test** | `& $py scripts/smoke_test.py` (1 min, no dataset) |
| download DIV2K (resumable) | `& $py scripts/fetch_data.py --extract` |
| fetch reference repos | `& $py scripts/fetch_reference.py --extract` |
| verify vs Restormer/SwinIR | `& $py tests/test_reference_equivalence.py` |
| size the runs | `& $py scripts/bench.py --patch 128 --vram-ceiling 7.0` |
| pre-crop DIV2K | `& $py scripts/prepare_subimages.py --input data/DIV2K/DIV2K_train_HR --output data/DIV2K/DIV2K_train_HR_sub --size 480 --step 240` |
| one segment | `& $py scripts/train.py --config configs/phase1_denoise.yaml --variant dual --segment-iters 10000` |
| queue segments | `.\scripts\run_segments.ps1 -Config configs\phase1_denoise.yaml -SegmentIters 10000 -MaxHours 6` |
| the key figure | `& $py scripts/plot_gates.py runs/phase1_denoise__dual__seed0 runs/phase2_deblur__dual__seed0 --labels denoise deblur` |

`train.py` exit codes: **0** run complete, **10** segment done and more remain,
**1** error. `run_segments.ps1` loops on 10.

## 4. Hardware reality

RTX 4060 **Laptop**, 8 GB — but the binding constraint is **15.4 GB system RAM**,
not VRAM.

- Windows `DataLoader` workers use **spawn**; each is a full process holding its
  own torch import. Cap at `num_workers: 4`. Datasets hold **paths, never
  decoded arrays** — the usual "cache the dataset in RAM" pattern does not fit.
- Plan against a **7.0 GB VRAM ceiling**, not 8.0. Push the desktop and browser
  onto the Intel Arc iGPU so the 4060 carries only training.
- Measured sweet spot: **micro-batch 8, accum 2**, at both width 32 and 56.
- **Mains power is worth 30%** — `dual` runs 1.64 opt-it/s on battery vs 2.14 on
  AC, identical VRAM. Never quote a battery number as a schedule.
- **This card does not OOM, it gets 20× slower.** The Windows driver backs
  oversized allocations with system RAM: measured 12.72 GB "allocated" on an
  8 GB card at 0.06 it/s. `Trainer` now caps the allocator
  (`vram_fraction`, default 0.90) so OOM is honest and the backoff can fire.
  If you see a peak above 8 GB, or `it_per_s` an order of magnitude low, that
  is the cause — do not go hunting for a slow dataloader.
- Laptop TGP throttles under sustained load. Use `-CoolDownSeconds` between
  segments and watch for `it_per_s` decaying across a night in `ledger.jsonl`.

## 5. Rules that are not negotiable

**`effective_batch = 16` for every arm of every phase.** The micro-batch is
whatever fits; gradient accumulation makes up the difference. The three arms do
not use the same VRAM per image, so if memory pressure were absorbed by batch
size, `dual` beating `window_only` could just be batch 8 vs batch 4 — a
confound that would appear nowhere in the results table. `Trainer` enforces
this and, on OOM, halves the micro-batch rather than the effective batch.

**Parameter matching by width, never by depth.** Changing block counts
introduces a second confound. Report exact parameter counts and FLOPs at 256²
in every table. A win at 2× the FLOPs is not a win.

**Per-task evaluation conventions.** SR and deraining report **Y channel**;
denoising and deblurring report **RGB**; SR additionally shaves `scale` pixels
from each border. One convention applied everywhere silently fails to match any
published table. `datnet/evaluation/protocols.py` is the single source of truth
and `tests/test_metrics.py` pins it.

**Separate checkpoints per task for Phases 1–4.** Do not fine-tune the denoising
checkpoint on deblurring — catastrophic forgetting, and it breaks comparability
with every baseline. Joint training is Phase 5, from scratch.

## 6. Work queue

Ordered so the cheapest kill switch fires first.

| # | step | kill condition |
|---|---|---|
| 1 | Read X-Restormer and PromptIR in full | differentiation does not hold → re-frame now |
| 2 | Settle width 32 vs 56 — **both measured**: 37.1 vs 63.1 GPU-h for 3 arms x 100 k. Recommendation: Phase 1 at 32, Phase 2 onward at 56 | — |
| 3 | Download DIV2K (`fetch_data.py`, resumable); pre-crop to sub-images | — |
| 4 | Phase 1 `dual` @ 60 k | — |
| 5 | ~~Test resume~~ **done** — `smoke_test.py` proves exact resume, 11/11 | — |
| 6 | Phase 1 `channel_only`, `window_only` @ 60 k | `dual` ≤ max(single) → rerun at 2× iterations, then reconsider |
| 6b | **Phase 1b, all noise types** (`configs/phase1b_multinoise.yaml`) — gaussian + speckle + speckle_gamma + poisson, no download needed | — |
| 7b | **Phase 2a, synthetic blur** (`configs/phase2a_synthblur.yaml`) — gaussian softness + defocus + motion, no download needed. Can run before GoPro arrives | — |
| 7 | **Plot Phase-1 gates.** Costs nothing | gates still at initialisation → the fusion is not receiving gradient. A **bug**, not a result |
| 8 | Phase 2, three arms @ 100 k | gate does not flip → fall back to the all-in-one contribution |
| 9 | **The central figure**: `plot_gates.py` phase 1 vs phase 2 | — |
| 10 | **Phase 3 SR ×4** — now a PREREQUISITE for Phase 5, not an optional extra | — |
| 11 | **Phase 5 oracle, then blind** — the strongest surviving contribution | — |
| 12 | Phase 4 derain — **cut this first** if time is short | — |

**Deferred, agreed: add Flickr2K (DF2K) before the FINAL runs, not before the
ablation.** 11.6 GB from `yangtao9009/Flickr2K` on HuggingFace, ~2,650 images,
roughly quadrupling DIV2K. It must NOT be introduced mid-phase: all three arms
have to see identical data or the comparison is confounded. So the sequence is
ablation on DIV2K -> pick the winner -> retrain the winner on DF2K for the
number that gets reported.

Cut in this order when the timeline slips: **Phase 4 first** → the `blind` arm
of Phase 5 → Phase 3 down to two arms. Do NOT cut Phase 3 before Phase 4: SR is
a prerequisite for the all-in-one-including-SR result, which the TPAMI 2025
survey confirms is the one genuinely unclaimed contribution. Never cut the Phase-2 three-way ablation, the
parameter matching, or the evaluation protocols.

## 7. Open questions and honest caveats

- **Nothing has been trained on real data.** The training path itself is proven
  (`smoke_test.py`, 11/11, including exact resume), but `PairedDataset`,
  `SRDataset`, the tiled inference path and all of Phase 5 have never run.
  See `CHECKPOINT.md` §6.
- **Model is 2.04 M at width 32, not the ~6 M the plan claimed.** Width 56 gives
  6.10 M for 1.7× the wall-clock, not the 3× the FLOP ratio implies. Still a
  choice to make and state.
- ~~Width matching breaks above width 48~~ **fixed** — `ffn_expansion` is now a
  fine second knob in `matching.py`, applied only when width alone misses 2%.
- **Phase 5 changes the SR backbone.** Phases 1–4 use a flat trunk for SR; the
  all-in-one model uses the U-Net for everything with per-task tails, because
  one checkpoint cannot have two backbones. Any Phase-3-vs-Phase-5 SR gap
  therefore mixes "all-in-one is hard" with "the backbone changed". **Run a
  single-task U-Net SR control** before attributing the gap to joint training.
- **Validation for Phase 1b measures Gaussian only.** `TestSet` synthesises
  additive Gaussian noise for in-loop validation; it has no speckle or Poisson
  path. So `best.pth` selection for the multi-noise run is driven by Gaussian
  performance alone. Fine as a progress signal, wrong as a headline number --
  build per-noise-type test sets before reporting anything from Phase 1b.
- **Deblurring is now two experiments.** Synthetic blur (fast, controllable,
  covers softness/defocus, no download) carries the gate measurement; GoPro
  carries the one table comparable to Restormer/NAFNet/MPRNet. Do not report
  synthetic-blur PSNR against published GoPro numbers.
- **Synthetic blur overlaps with SR.** Bicubic downsampling is itself a
  low-pass filter, so Phase 2a and Phase 3 may land in similar gate positions.
  That weakens the four-task story but not the core noise-versus-blur flip.
- **The gate could simply not move.** That is the load-bearing risk, and it is
  a publishable negative result if reported honestly — it saves the next person
  the experiment. It does not sink the project, because the
  all-in-one-including-SR contribution is independent of it.
