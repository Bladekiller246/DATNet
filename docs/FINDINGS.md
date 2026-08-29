# DATNet — Consolidated Findings

**Project:** Dual-Axis Transformer for Image Restoration
**Author:** Mann Kuvadiya
**Record compiled:** 22 August 2026
**Hardware:** RTX 4060 Laptop (8 GB), Core Ultra 7 155H, 15.4 GB RAM

This is the evidence record: every measurement taken, what it proves, what it
disproves, and what remains unestablished. Numbers are quoted from run
artifacts, not reconstructed.

---

## 0. Verdict at a glance

| Claim | Status | Evidence |
|---|---|---|
| Dual-axis fusion beats single-axis at matched params (denoising) | **Unresolved — leaning no** | §2.2, §2.3 |
| The learned gate is a real, measurable quantity | **Proven right** | §3.2 |
| The gate responds to the degradation type | **Proven right** | §3.3 |
| Noise → channel-leaning, blur → **spatial**-leaning | **Proven wrong** | §3.3 |
| Longer training rescues `dual` on denoising | **Partly — confounded** | §2.3 |
| Weight decay was suppressing gate movement | **Proven wrong** | §5.4 |
| Branch normalisation is required for the gate to mean anything | **Proven right** | §3.1 |
| All-in-one including SR is unclaimed in the literature | Holds (survey-checked) | §6 |

---

## 1. What was predicted

From `DATNet_Project_Plan.md` §1. The hypothesis was directional, not vague:

| Degradation | Physical character | Predicted dominant axis |
|---|---|---|
| Gaussian noise | i.i.d. per-pixel, signal-independent, no spatial structure | **Channel** |
| Motion blur | anisotropic spatial convolution, wide kernel (~50 px) | **Spatial** (strongest prediction) |

The plan staked the paper on the flip between these two (§4 Phase 2):

> "If the gate flips direction between Phase 1 and Phase 2 — channel-dominant
> for noise, spatial-dominant for blur — that plot is the central figure."

---

## 2. Phase 1 — Denoising

Data: DIV2K 800 imgs + synthetic AWGN σ∈{15,25,50}. Test: CBSD68 (n=68),
McMaster (n=18). RGB PSNR, no border shave. EMA weights.

### 2.1 Setup

Three arms at matched parameter count (within 1.3%), the single-axis arms
widened from 32→36 to compensate for the removed branch.

| Arm | Params | Width | Stands in for |
|---|---|---|---|
| `dual` | 2,037,432 | 32 | DATNet |
| `channel_only` | 2,062,834 | 36 | Restormer-lite |
| `window_only` | 2,042,298 | 36 | SwinIR-lite |

### 2.2 The 30k three-arm screening — **FAIL**

CBSD68 / McMaster, RGB PSNR (dB):

| Arm | Iter | CBSD68 σ15 | **σ25** | σ50 | McM σ15 | σ25 | σ50 |
|---|---|---|---|---|---|---|---|
| `dual` | 22,500 | 33.740 | **31.170** | 27.941 | 34.203 | 32.133 | 29.117 |
| `channel_only` | 30,000 | 33.761 | **31.166** | 27.944 | 34.243 | 32.084 | 29.037 |
| `window_only` | 30,000 | 33.777 | **31.191** | 27.974 | 34.234 | 32.151 | 29.141 |

`window_only` wins the headline metric. The recorded verdict in
`results/phase1_denoise__comparison.json` is `"FAIL"`.

**The spread across all three arms is 0.025 dB at σ25.** That is the single most
important number in this section: the three architectures are, for practical
purposes, indistinguishable on Gaussian denoising at this scale. Any narrative
built on which one "won" is reading noise.

### 2.3 The 60k confirmation — passes, with a caveat that matters

The plan's kill-switch (§4) requires rerunning `dual` at 2× iterations before
concluding. Done — a fresh run on a complete 60k cosine schedule:

| Test | σ15 | σ25 | σ50 |
|---|---|---|---|
| CBSD68 | 33.779 / .9253 | **31.239** / .8810 | 28.039 / .7889 |
| McMaster | 34.299 / .9135 | 32.279 / .8833 | 29.275 / .8154 |

Difference vs. the 30k arms at CBSD68 σ25:

| Comparison | Δ |
|---|---|
| 60k `dual` − 30k `dual` | **+0.069** |
| 60k `dual` − 30k `channel_only` | **+0.073** |
| 60k `dual` − 30k `window_only` | **+0.048** |
| 60k `dual` − 30k `window_only` (σ50) | **+0.065** |
| 60k `dual` − 30k `window_only` (σ15) | **+0.002** (tie) |

> **This is not a clean win, and should not be reported as one.** `dual` trained
> for 60k; the arms it is being compared against trained for 30k. The
> comparison is parameter-matched but **not budget-matched**. The honest
> statement is: *at 2× budget, `dual` recovers a ~0.05 dB lead over
> single-axis arms trained at 1× budget* — which is roughly the size of the
> effect a 2× budget alone would be expected to produce. Establishing whether
> `dual` genuinely beats the single-axis arms would require rerunning
> `channel_only` and `window_only` at 60k. That was not done.

### 2.4 Convergence

The 60k run was still improving at the end (val PSNR 32.861 → 32.908 over the
last 7.5k iterations, peaking at 57,500 then flat). Not under-trained; not
overfitting.

---

## 3. The gate — the central investigation

### 3.1 Discovery: the gate was decorative

The nominal gate value `g` is **not** what the block does. The quantity that
matters is the effective ratio `|g·A| / (|g·A| + |(1−g)·B|)`.

Measured on the original Phase-1 `dual` run (`normalise_branches=False`):

| Block | nominal `g` | effective | Δ |
|---|---|---|---|
| enc1.blocks.0 | 0.800 | 0.465 | −0.335 |
| enc2.blocks.1 | 0.504 | 0.224 | −0.280 |
| dec2.blocks.2 | 0.506 | 0.106 | −0.401 |
| **dec1.blocks.1** | **0.803** | **0.105** | **−0.698** |
| refine.blocks.1 | 0.802 | 0.238 | −0.564 |
| **mean effective** | — | **0.259** | (nominal ≈ 0.54) |

A block reporting `g = 0.80` — nominally 80% channel — was running an effective
blend of **0.105**, i.e. roughly 9:1 *spatial*. The network rescaled the branches
internally through their own 1×1 projections and reproduced whatever blend it
wanted, leaving `g` as a decorative number.

**Consequence: every gate reading taken before branch normalisation is void.**
Weeks of "the gate isn't moving" debugging was measuring a quantity that had no
causal relationship to the model's behaviour.

**Proven right:** `normalise_branches=True` is load-bearing, not a refinement.
With it enabled, nominal and effective agree within 0.036 (Phase 1) and 0.072
(Phase 2) — small enough that `g` becomes a legitimate proxy.

### 3.2 The gate is real and it converges

Phase 2 gate trajectory, movement of the fastest-moving block per 20k
iterations:

| Interval | Δ |
|---|---|
| 0 → 20k | +0.0255 |
| 20k → 40k | +0.0132 |
| 40k → 60k | +0.0069 |
| 60k → 80k | +0.0039 |
| 80k → 100k | +0.0022 |

Halving each interval — exponential convergence. Several decoder blocks reversed
direction and settled. The plan's guard against a premature negative
(§7.3: "if the per-block values are still drifting at the end, the run was too
short") is **satisfied**: they are not still drifting.

**Proven right:** the gate is not frozen, not decorative, and not
under-trained. It learns and it settles.

### 3.3 The direction is wrong — the core negative result

Effective ratios, 8 test images per task, both runs `normalise_branches=True`.
1.0 = pure channel, 0.0 = pure spatial.

| Block | Phase 1 (noise) | Phase 2 (blur) | Δ |
|---|---|---|---|
| enc1.blocks.0 | 0.794 | 0.812 | +0.018 |
| enc1.blocks.1 | 0.808 | 0.820 | +0.012 |
| enc2.blocks.0 | 0.510 | 0.513 | +0.003 |
| enc2.blocks.1 | 0.492 | 0.502 | +0.011 |
| enc2.blocks.2 | 0.494 | 0.514 | +0.020 |
| **bottleneck.blocks.0** | 0.198 | 0.273 | **+0.075** |
| **bottleneck.blocks.1** | 0.202 | 0.277 | **+0.075** |
| **bottleneck.blocks.2** | 0.211 | 0.313 | **+0.102** |
| **bottleneck.blocks.3** | 0.204 | 0.322 | **+0.118** |
| dec2.blocks.0 | 0.495 | 0.529 | +0.034 |
| dec2.blocks.1 | 0.519 | 0.496 | −0.023 |
| dec2.blocks.2 | 0.505 | 0.487 | −0.018 |
| dec1.blocks.0 | 0.802 | 0.809 | +0.007 |
| dec1.blocks.1 | 0.769 | 0.798 | +0.029 |
| refine.blocks.0 | 0.775 | 0.774 | −0.001 |
| refine.blocks.1 | 0.777 | 0.840 | +0.064 |
| **mean** | **0.535** | **0.568** | **+0.033** |

**Proven right — a task-dependent axis preference exists.** The four bottleneck
blocks move +0.075 to +0.118, an order of magnitude above the ±0.02 scatter
elsewhere, and the effect grows monotonically with depth (0→1→2→3). Plotted, the
lines separate. This is signal.

**Proven wrong — the predicted direction.** Deblurring pushed the bottleneck
**toward channel attention** (0.20 → 0.32). The prediction was that motion blur,
the most spatially-structured degradation in the taxonomy, would push toward
*spatial*. Both tasks are channel-leaning overall; blur is **more**
channel-leaning than noise, not less.

The paper's proposed central figure would show the lines separating in the
opposite direction from the thesis they were meant to support.

---

## 4. Phase 2 — Deblurring (performance record)

Data: GoPro, 2,103 train / 1,111 test pairs (extracted and verified). Loss:
L1 + 0.05·FFT. Validation: 8-image GoPro test subset, EMA weights.

### 4.1 Validation curve (`dual`, 100k iterations)

| Iter | PSNR | SSIM | | Iter | PSNR | SSIM |
|---|---|---|---|---|---|---|
| 5,000 | 26.486 | .8041 | | 55,000 | 30.239 | .8967 |
| 10,000 | 27.673 | .8350 | | 60,000 | 30.292 | .8977 |
| 15,000 | 28.863 | .8645 | | 65,000 | 30.382 | .8988 |
| 20,000 | 29.342 | .8765 | | 70,000 | 30.444 | .9000 |
| 25,000 | 29.537 | .8822 | | 75,000 | 30.528 | .9012 |
| 30,000 | 29.648 | .8852 | | 80,000 | 30.577 | .9016 |
| 35,000 | 29.772 | .8877 | | 85,000 | 30.608 | .9022 |
| 40,000 | 29.941 | .8906 | | 90,000 | 30.674 | .9033 |
| 45,000 | 30.036 | .8929 | | 95,000 | 30.695 | .9035 |
| 50,000 | 30.115 | .8942 | | **100,000** | **30.738** | **.9042** |

Total gain: **+4.25 dB / +0.100 SSIM**. Shape is textbook — steep early
(+2.86 dB in the first 15k), then steady diminishing returns (+0.04–0.06 dB per
5k at the end). **No plateau, no decline** — the GoPro overfitting risk the plan
flagged (2,103 pairs is small) did not materialise within 100k.

### 4.2 Visual confirmation

`results/_gate_compare/deblur_progress_40k_vs_100k.png` — same test image at
iteration 40,000 vs 100,000, blurry / restored / ground-truth. Visible sharpening
of background architecture and signage; fast-moving foreground pedestrians remain
partially blurred at both checkpoints.

### 4.3 Not established for Phase 2

- **`channel_only` and `window_only` were never run.** Whether dual-axis fusion
  improves deblurring PSNR is **unknown**. §3.3 concerns gate direction only.
- n = 1 seed, 8 evaluation images per task. No variance estimate.
- Phase 1 and Phase 2 differ in dataset *and* degradation simultaneously — "task"
  here is a bundle of factors, not a clean single variable.

---

## 5. Bugs found, with measured impact

### 5.1 `report_phase.py` mis-detected the architecture — 4.55 dB error

`load_model()` inferred `normalise_branches` by scanning the state dict for
`attn_scale` keys. But branch normalisation is **parameter-free** — it leaves no
trace in the weights. The heuristic silently returned `False` for every
checkpoint trained after the feature existed.

| Checkpoint | Evaluated wrong | Evaluated right | Error |
|---|---|---|---|
| Phase 1 60k `dual`, CBSD68 σ25 | 26.690 | **31.239** | **−4.55 dB** |

The trained weights were fine the whole time; only the evaluation was broken. It
happened to be *correct by accident* for pre-feature checkpoints, which is why it
went unnoticed.

**Fixed:** `train.py` now records the resolved value into `config.json`;
`report_phase.py` trusts it and falls back to `False` only for genuinely legacy
checkpoints. Verified — the old checkpoint still reproduces 31.170 exactly, the
new one gives 31.239.

### 5.2 Orphaned supervisor caused two concurrent training runs

A cutoff script killed `python.exe` but not the parent `run_segments.ps1` loop.
That loop's crash-retry logic — designed for transient OOM — interpreted the kill
as a failure and relaunched training. A second run was later started alongside
it; both raced on the same checkpoint file for ~3.5 hours.

| Symptom | Value |
|---|---|
| Apparent throughput during race | 0.38–0.41 it/s |
| Actual throughput once resolved | 1.06 it/s |
| Iterations lost | ~1,400 |

**Fixed:** the cycle wrapper now kills the supervisor **first**, then the workers,
and verifies zero remaining processes.

### 5.3 Wrong Python interpreter in the default shell

The shell's `python` resolved to `torch 2.13.0+cpu` (`cuda available: False`);
the real environment is the `datnet` conda env with `torch 2.13.0+cu126`. Any
training launched from the default shell would have run on CPU at unusable speed.

### 5.4 Gate weight-decay hypothesis — tested and rejected

Hypothesis: AdamW's weight decay was pulling the gate logits toward zero,
suppressing movement. A control run (seed 1) excluded gate params from decay.

| Iteration | With decay | Without decay |
|---|---|---|
| 3,500 | Δ 0.0012 | Δ 0.0002 |
| 5,500 | Δ 0.0021 | Δ 0.0003 |

Removing decay made the gate move **less**, not more — 6–7× smaller at both
checkpoints. **Hypothesis rejected.** (The real explanation was §3.1: the gate
was decorative, so its raw movement was meaningless either way.) The
`weight_decay=0` param group was kept regardless — it is correct practice for
routing logits and costs nothing.

---

## 6. Throughput engineering (measured, not projected)

### 6.1 Phase 2 — the micro-batch fix

`micro_batch` 4 → 8 at iteration ~46k, `effective_batch` held at 16 so training
dynamics are unchanged (accumulation absorbs the difference).

| Iter | it/s | Peak VRAM |
|---|---|---|
| 10,000 | 1.017 | 3.59 GB |
| 20,000 | 1.134 | 3.58 GB |
| 40,000 | 1.097 | 3.58 GB |
| *— micro_batch 4 → 8 —* | | |
| 49,000 | 1.477 | 4.32 GB |
| 64,000 | 1.486 | 4.32 GB |
| 87,000 | 1.616 | 4.63 GB |
| 102,000 | **1.699** | 4.63 GB |

**~55% throughput gain** for a config change, at no cost to comparability.

### 6.2 Phase 3 SR — width/batch sweep

| Width | micro_batch | Params | it/s | Peak VRAM |
|---|---|---|---|---|
| 32 (as shipped) | 4 | 0.254 M | 1.26 | 0.75 GB |
| 32 | 16 | 0.254 M | **1.71** | 2.94 GB |
| 64 | 8 | 0.917 M | 1.50 | 2.65 GB |
| 64 | 16 | 0.917 M | 1.41 | 5.25 GB |
| **96** | **8** | **1.990 M** | **1.20** | **3.79 GB** |
| 96 | 16 | 1.990 M | OOM | — |

---

## 7. Phase 3 — configuration decisions and why

### 7.1 Width 32 → 96

The flat SR trunk (12 blocks, uniform width) holds far fewer parameters than the
3-level U-Net at the same width — **0.254 M vs 2.037 M**, an 8× gap. Phase 3's
role in the revised plan is the **single-task upper bound** against which Phase
5's SR column is measured, and Phase 5 uses the U-Net (plan §11.1 decision 1).

A 0.254 M baseline would let the joint model beat its own upper bound, making the
all-in-one claim meaningless rather than impressive. Width 96 gives 1.990 M — a
**2% match** to the U-Net's 2.037 M, the same parameter-matching discipline
already applied across ablation arms.

### 7.2 `heads[0]` 1 → 3 (caught 1,700 iterations in)

Raising the width exposed a coupling that is easy to miss: the flat trunk uses
`heads[0]` for **all** its blocks (`datnet.py:144`). At width 32 that gave
head_dim 32; at width 96 it silently gave **head_dim 96** — a 96×96 MDTA
attention map instead of 32×32, 9× the entries, and a learnable temperature `τ`
calibrated against a different map size.

Plan §11 decision 5 holds head_dim at 32 at every level precisely so `τ` keeps
one meaning. `heads[0] = 3` restores it (96 / 3 = 32) at essentially no cost:

| Config | Params | head_dim |
|---|---|---|
| width 96, heads 1 | 1.990 M | 96 |
| width 96, heads 3 | **1.996 M** | **32** |

The corrected version is marginally *closer* to the U-Net's 2.037 M. This
mattered more than the usual "width matching moves head_dim" caveat because
Phase 5 uses the width-32 U-Net at head_dim 32, and the plan already flags a
flat-vs-U-Net backbone confound on the Phase 3 ↔ Phase 5 comparison — head_dim 96
would have stacked a second confound on the exact comparison the surviving
contribution rests on. `heads: 1` is also unconventional for window attention at
this width (SwinIR uses 6 heads at dim 180).

The 1,700-iteration run at heads 1 is archived at
`runs/_archive/phase3_sr_x4__dual__seed0_heads1_ABANDONED`.

### 7.3 `total_iters` 200,000 → 100,000

100k is the plan's own Phase-3 budget (§7.1: "3 arms @ 100k"). More importantly,
`total_iters` **defines the cosine schedule** — leaving it at 200k while only
ever training ~100k means the LR never anneals to `min_lr`, and the run is
permanently stuck mid-decay. This same discrepancy existed in the Phase 2 config
(200k in file vs 100k in plan).

---

### 7.4 Phase 3 progress — Set5 ×4, Y-channel, 4 px border shave

Protocol verified against `protocols.py`: SR reports **Y of YCbCr with `scale`
pixels shaved**, the published convention — these numbers are literature-
comparable, not inflated RGB.

| Iter | PSNR | SSIM | Δ |
|---|---|---|---|
| 5,000 | 29.196 | .8285 | — |
| 10,000 | 30.319 | .8604 | +1.123 |
| 15,000 | 30.885 | .8731 | +0.567 |
| 20,000 | 31.316 | .8809 | +0.431 |
| 25,000 | 31.595 | .8855 | +0.278 |
| 30,000 | 31.762 | .8883 | +0.168 |
| 35,000 | 31.872 | .8902 | +0.110 |
| 40,000 | 31.958 | .8913 | +0.086 |
| **45,000** | **32.024** | **.8923** | **+0.066** |

| 50,000 | 32.078 | .8930 | +0.049 |
| 55,000 | 32.131 | .8937 | +0.054 |
| 60,000 | 32.154 | .8940 | +0.023 |
| 65,000 | 32.180 | .8944 | +0.025 |
| 70,000 | 32.195 | .8946 | +0.016 |
| 75,000 | 32.211 | .8948 | +0.016 |
| 80,000 | 32.221 | .8948 | +0.010 |
| 85,000 | 32.231 | .8949 | +0.009 |
| **90,000** | **32.235** | **.8950** | **+0.005** |

Geometric decay in the gains at every interval, no plateau or reversal.
**Stopped at 93,900 by decision, not failure** — with LR already annealed to
3.0e-06 and gains at +0.005 per 5k, the remaining 6k iterations were not worth
the hour. `best.pth` is the iteration-90,000 checkpoint.

#### Final test-set results (Y-channel, 4 px shave, EMA weights)

| Test set | PSNR | SSIM | n |
|---|---|---|---|
| Set5 | **32.235** | .8950 | 5 |
| Set14 | **28.621** | .7824 | 14 |
| BSD100 | **27.593** | .7376 | 100 |
| Urban100 | **26.086** | .7849 | 100 |

Orientation only, **not** like-for-like (SwinIR classical-SR is 11.8 M params,
5.9× larger, fully trained): SwinIR reports ≈32.9 on Set5 and ≈27.5 on Urban100.
DATNet trails by ~0.67 dB on Set5 and ~1.41 dB on Urban100. The larger Urban100
gap is consistent with that set's repetitive fine structure rewarding capacity
and receptive field — the two things a 2 M-param model has least of.

Per-image against a bicubic baseline at the same output size:

| Image | Bicubic | DATNet | Δ |
|---|---|---|---|
| baby | 32.02 | 33.74 | +1.72 |
| bird | 30.44 | 34.60 | +4.16 |
| butterfly | 22.32 | 28.17 | **+5.84** |
| head | 31.71 | 32.96 | +1.24 |
| woman | 26.69 | 30.66 | +3.97 |
| **mean** | **28.64** | **32.02** | **+3.39** |

For orientation only — **not** a like-for-like comparison — SwinIR classical-SR
reports ≈32.9 dB on Set5 ×4 at 11.8 M parameters and full training. DATNet is at
32.02 with **1.996 M params (5.9× fewer)** and 46% of its own schedule. The plan
is explicit that DATNet does not compete on absolute SR numbers (§3.2); the
figure matters only as evidence the SR path works well enough to serve as a
credible single-task upper bound for Phase 5.

Samples: `results/phase3_sr_x4__dual__seed0/samples/` (bicubic | DATNet | GT).

## 8. Consequences adopted

The plan pre-committed to this outcome (§8, risk register):

> "Gate does not flip between tasks | **Medium-High** | This is the load-bearing
> risk… Fall back to the all-in-one-including-SR contribution, which is
> independent of the gate result. Report the negative result honestly in the
> ablation section — it saves the next person the experiment."

Adopted:

1. **Phase 2 arms 2 and 3 cut** (~37 GPU-hours) — they would have more precisely
   characterised a prediction already contradicted in direction.
2. **Phase 3 (SR) promoted to priority** as the prerequisite for Phase 5.
3. **Surviving contribution:** the single all-in-one checkpoint *including*
   super-resolution — unclaimed as of the TPAMI 2025 all-in-one survey
   (plan §2.0), and independent of gate direction.
4. **The gate remains publishable as a negative result.** §3.3 is a real
   measurement of a real mechanism; it simply does not reproduce the predicted
   physical story.

---

## 9. Open questions

| Question | Why it is open | Cost to close |
|---|---|---|
| Does `dual` beat single-axis arms at equal budget? | Never tested budget-matched (§2.3) | 2 × 60k denoise runs |
| Does dual-axis fusion help deblurring at all? | Phase 2 arms never run (§4.3) | 2 × 100k GoPro runs |
| Is the bottleneck effect reproducible? | n = 1 seed (§4.3) | 1 extra seed per phase |
| Why does blur prefer *channel* at the bottleneck? | Unexplained — contradicts the physical argument | Analysis, not compute |
| Phase 3 ↔ Phase 5 backbone confound | Flat vs U-Net trunk (plan §11.1) | 1 U-Net SR control run |

---

## 10. Future direction — composite degradation (not in scope, deliberately)

Recorded so the limitation is explicit rather than discovered later.

**What exists today.** Phases 1–3 produce **three separate checkpoints**. Each
handles exactly one degradation family. Nothing chains them.

**What Phase 5 will add.** One checkpoint covering all tasks — but the batching
is task-homogeneous by construction (`datnet/data/multitask.py`):

> "batches are task-homogeneous: each micro-batch is drawn from a single task,
> and gradient accumulation mixes tasks within an optimiser step."

So Phase 5 is *one model, many tasks* — **not** one model handling several
degradations present in the same image. Every training sample it ever sees
carries exactly one degradation.

**What is therefore NOT covered.** An image that is simultaneously noisy, blurred
*and* low-resolution — composite (mixed) degradation. This is a distinct and
harder problem than either single-task restoration or all-in-one task routing.

**Measured, not asserted.** The chain was tested directly (23 Aug) using the
three trained checkpoints: a 256×256 clean crop, bicubic-downsampled ×4 to
64×64, then AWGN σ=25 added. Y-channel PSNR, 4 px shave, against the clean GT:

| Pipeline | PSNR | vs bicubic |
|---|---|---|
| bicubic upsample (no model) | 20.97 | — |
| **SR only** | **19.74** | **−1.23** |
| **denoise → SR** | **23.05** | **+2.09** |
| denoise → deblur → SR | 22.52 | +1.56 |
| deblur → SR | 18.54 | −2.43 |
| *upper bound: same SR model on clean LR* | *28.60* | *+7.63* |

Three results worth keeping:

1. **The deblur stage costs 0.53 dB** (23.05 → 22.52). The intuition that
   denoising leaves over-smoothing which a deblurrer should fix does not hold:
   the Phase-2 model was trained on GoPro *camera motion blur*, a different
   inverse problem from denoiser smoothing, so its input is out of distribution.
2. **SR on a noisy input is worse than no model at all** (19.74 vs 20.97). The
   SR model only ever saw clean bicubic-downsampled LR, so it treats noise as
   signal and amplifies it ×4 into structured colour speckle
   (`results/_chain_test/chain_comparison.png`, panel 2). Order is not a
   preference — denoise *must* precede SR.
3. **The best chain still forfeits 5.5 dB** against what the same SR model
   achieves on a clean LR (23.05 vs 28.60). That gap is the price of composing
   models each trained as if the others' degradations do not exist.

**Scale note.** Only a ×4 tail was ever trained (plan §4: "Start with ×4 only").
A 126→256 (≈×2) request has no trained path; the ×2 tail does not exist.

**Why naive chaining is not a substitute.** Running denoise → deblur → SR in
sequence fails for three compounding reasons:

1. **Out-of-distribution inputs.** Each model was trained on images carrying one
   degradation. The denoiser has never seen a blurred input; its behaviour there
   is unconstrained.
2. **Error compounding.** Residual artifacts from stage 1 become signal for
   stage 2, which amplifies rather than removes them.
3. **Order dependence with no principled ordering.** denoise→SR sharpens residual
   noise into structure; SR→denoise upscales the noise first. Both are defensible
   and they give different answers.

**Cheapest real path, if pursued.** Train on composite degradations directly —
compose noise *and* blur *and* downsampling onto the same patch so the model
learns the joint inverse problem. `SyntheticDataset.__getitem__` currently
samples exactly one spec per patch (`random.choices(..., k=1)`); composing
several would be a modest change plus a config, not an architectural rewrite.
The degradation ops needed already exist in `datnet/data/degradations.py`
(`NOISE_OPS`: gaussian, speckle, speckle_gamma, poisson, salt_pepper;
plus gaussian/defocus/motion blur kernels).

Kept as a future option; not scheduled.

---

## Appendix — artifact locations

```
results/phase1_denoise__comparison.json          30k three-arm screening
results/phase1_denoise_60k__dual__seed0/         60k confirmation + samples
results/phase1_denoise__dual__seed0/effective_gate.json   decorative-gate evidence
results/_gate_compare/phase1_60k_effective.json  Phase 1 gate (normalised)
results/_gate_compare/phase2_effective.json      Phase 2 gate (normalised)
results/_gate_compare/deblur_progress_40k_vs_100k.png     visual comparison
runs/phase2_deblur__dual__seed0/gates.jsonl      full gate trajectory
runs/*/ledger.jsonl                              per-segment throughput, VRAM, val
docs/PHASE2_GATE_RESULT.md                       the negative result, standalone
```
