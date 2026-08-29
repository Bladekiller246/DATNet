# Phase 2 result: the axis-preference hypothesis is not supported

**Status:** negative result, recorded deliberately.
**Date:** 22 August 2026
**Evidence:** `runs/phase1_denoise_60k__dual__seed0` (57.5k iters, denoising),
`runs/phase2_deblur__dual__seed0` (100k iters, deblurring),
`results/_gate_compare/`.

---

## 1. What was predicted

The project's central claim (`DATNet_Project_Plan.md` §1) was that the
channel/spatial balance is **degradation-dependent in a specific direction**:

| Degradation | Predicted dominant axis |
|---|---|
| Gaussian noise (i.i.d., signal-independent, no spatial structure) | **Channel** |
| Motion blur (anisotropic, wide spatial kernel) | **Spatial** (strongest prediction) |

The plan called the resulting figure the paper's centrepiece (§4 Phase 2):

> "If the gate flips direction between Phase 1 and Phase 2 — channel-dominant
> for noise, spatial-dominant for blur — that plot is the central figure of the
> paper."

## 2. How it was measured

Per-block **effective ratio** `|g·A| / (|g·A| + |(1−g)·B|)` via
`scripts/effective_gate.py`, averaged over 8 test images, not the nominal gate
value `g`. Both checkpoints were trained with `normalise_branches=True`, under
which `g` and the effective ratio agree by construction (largest observed
divergence: 0.036 in Phase 1, 0.072 in Phase 2).

This distinction is load-bearing. An earlier Phase-1 run *without* branch
normalisation showed nominal `g` and effective ratio disagreeing by up to 0.70 —
a "decorative gate", where the network rebalanced the branches internally while
`g` sat frozen. Those readings are not comparable and are excluded here.

1.0 = pure channel attention, 0.0 = pure spatial attention.

## 3. What was measured

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

## 4. The finding

**The gate is task-responsive, but its direction contradicts the hypothesis.**

Three things are established:

1. **The gate is not decorative and not frozen.** The bottleneck blocks move
   +0.075 to +0.118 between tasks — an order of magnitude above the ±0.02 scatter
   elsewhere in the table, and monotonically increasing with depth
   (blocks 0→1→2→3). This is a real signal, and it rules out the wiring-bug
   explanation that motivated much of the Phase-1 debugging.

2. **A task-dependent axis preference exists.** The two tasks give measurably
   different gate configurations. Plotted, the lines would separate.

3. **The direction is wrong.** Deblurring pushed the bottleneck **toward channel
   attention** (0.20 → 0.32). The prediction was that motion blur — the most
   spatially-structured degradation in the plan's taxonomy — would push it
   toward *spatial*. Both tasks are channel-leaning overall, and blur is *more*
   channel-leaning than noise, not less.

The gates are **converged, not under-trained**, so this is not an artifact of a
short run. Movement of the fastest block per 20k iterations decayed
+0.026 → +0.013 → +0.007 → +0.004 → +0.002, halving each interval; several
decoder blocks reversed and settled. The plan's own guard against a premature
negative (§7.3: "if the per-block values are still drifting at the end, the run
was too short") is satisfied — they are not still drifting.

## 5. What this does NOT establish

Stated plainly, because these limits matter for how the result is reported:

- **Whether `dual` beats `channel_only` / `window_only` on deblurring is
  unknown.** Those arms were not run. This result concerns gate *direction*
  only, not whether dual-axis fusion improves PSNR.
- **n = 1 seed, 8 evaluation images per task.** No variance estimate. The
  bottleneck effect is large relative to within-table scatter, but it has not
  been replicated.
- **The two runs differ in dataset and degradation simultaneously** (DIV2K+AWGN
  vs GoPro motion blur). That is inherent to the comparison, but it means
  "task" here is a bundle of factors, not a clean single variable.
- Phase 2 ran to 100k of the 200k in its config file. 100k *is* the plan's
  stated Phase-2 budget (§7.1), and §4's convergence test is satisfied, so this
  is a complete result rather than a truncated one.

## 6. Consequence

The plan anticipated this outcome and pre-committed to the response
(§8, risk register):

> "Gate does not flip between tasks | **Medium-High** | This is the load-bearing
> risk. If the gate does not move between noise and blur, then applying both
> attentions unconditionally (X-Restormer) was already correct and conditioning
> buys nothing. Fall back to the all-in-one-including-SR contribution, which is
> independent of the gate result. Report the negative result honestly in the
> ablation section — it saves the next person the experiment."

Adopted. Consequences:

- **Phase 2 arms 2 and 3 (`channel_only`, `window_only`) are cut** — roughly 37
  GPU-hours that would have more precisely characterised a prediction already
  contradicted in direction.
- **Phase 3 (SR) becomes the priority**, as the prerequisite for Phase 5.
- **The surviving contribution is the single all-in-one checkpoint including
  super-resolution**, which no published method ships (confirmed against the
  TPAMI 2025 all-in-one survey, §2.0) and which does not depend on the gate
  direction.
- The gate remains defensible as an *interpretable, measurable* mechanism —
  just not one that reproduces the predicted physical story. Section 3's table
  is still worth publishing as a negative result.

## 7. Reproducing

```
python scripts/effective_gate.py --run-dir runs/phase1_denoise_60k__dual__seed0 --images 8
python scripts/effective_gate.py --run-dir runs/phase2_deblur__dual__seed0 \
    --test-root data/GoPro/test/sharp --images 8      # deblur needs lq_root; see results/_gate_compare/
```

Raw measurements: `results/_gate_compare/phase1_60k_effective.json`,
`results/_gate_compare/phase2_effective.json`.
