# DATNet — Dual-Axis Transformer for Image Restoration

**Project plan, v2**
Author: Mann Kuvadiya
Date: 18 August 2026
Target hardware: RTX 4060 Laptop, 8 GB VRAM / 15.4 GB system RAM (see §6)

---

## 1. The idea, stated precisely

The starting intuition was "take the best parts of Restormer, Uformer, and SwinIR and combine them." That intuition needs sharpening before it becomes a research project, because the task-level split in the original framing (Restormer = deraining, Uformer = deblurring, SwinIR = super-resolution) is an artifact of **what each paper chose to train on**, not of what each architecture is inherently good at. Restormer reports deblurring results. SwinIR reports denoising results. Uformer reports denoising results. There is no "deraining module" inside Restormer that can be cut out and transplanted.

What *is* genuinely separable is the mechanism each model uses to avoid the O(N²) cost of global self-attention:

| Model | Attention mechanism | Axis attended over | Structure | DW-conv FFN |
|---|---|---|---|---|
| Restormer | MDTA — transposed attention, C×C map | **Channel** | 4-level U-Net | Yes (GDFN, gated) |
| Uformer | LeWin — non-overlapping local windows | **Spatial** (local) | U-Net + modulators | Yes (LeFF) |
| SwinIR | Shifted windows, alternating offset | **Spatial** (local, cross-window) | Flat, single-resolution | No (conv at RSTB tail) |

Restated as a research claim:

> **Channel-axis attention and spatial-axis attention capture complementary structure, and the degradation type determines which axis matters more. No existing single model attends over both axes, or adapts the balance between them to the degradation.**

That is testable, falsifiable, and not a collage of three papers.

### The specific hypothesis

| Degradation | Physical character | Predicted dominant axis |
|---|---|---|
| Gaussian noise (additive) | i.i.d. per-pixel, signal-**in**dependent, no spatial structure | **Channel** (strongest prediction) |
| Poisson / shot noise | per-pixel but signal-**dependent**: variance tracks intensity | Channel, but weaker |
| Speckle (multiplicative) | signal-dependent, magnitude scales with local intensity, so it inherits spatial structure from the image | Channel, weakest -- **may not hold** |
| Gaussian blur / softness | isotropic spatial convolution, compact kernel (~3-9 px) | **Spatial** |
| Defocus blur | isotropic disk PSF (~2-11 px) | **Spatial** |
| Motion blur | anisotropic spatial convolution, wide kernel (up to ~50 px) | **Spatial** (strongest prediction) |
| Bicubic downsampling (SR) | spatial low-pass | **Spatial** |
| Rain streaks | spatially structured *and* channel-distinct | Mixed |

**Scope note.** "Denoising" here means *all* noise families, not only additive
Gaussian; "deblurring" means removing blur and softness generally -- defocus and
Gaussian blur as well as camera-motion blur. That widening sharpens the
hypothesis rather than diluting it, because it separates two things the original
framing conflated:

> The channel-leaning prediction rests on noise being **signal-independent**, not
> merely on it being "noise". Speckle is multiplicative, so its spatial structure
> is inherited from the image content. **If the gate leans channel for Gaussian
> but not for speckle, the axis preference is really tracking signal-dependence,
> not the noise/blur distinction** -- a sharper and more interesting claim than
> the one this plan started with.

That is a free extra result: the same machinery, the same figure, one more line.

Noise and blur sit at opposite ends of this axis, which is why they are the first two phases. Deraining is deliberately last — it is the least informative because it is predicted to be mixed.

---

## 2. Prior art you must read and cite

This space is crowded. Read these before writing a line of code; two of them are close enough that your framing has to explicitly differentiate.

| Work | Why it matters to you |
|---|---|
| **Restormer** (CVPR 2022) | Source of MDTA + GDFN. Your `channel_only` ablation is essentially Restormer-lite. |
| **SwinIR** (ICCVW 2021) | Source of shifted-window attention. Your `window_only` ablation is SwinIR-lite. |
| **Uformer** (CVPR 2022) | Window attention inside a U-Net, plus multi-scale modulators. |
| **X-Restormer** (ECCV 2024) | **Closest prior art — read this first, in week 1.** See the dedicated subsection below. |
| **PromptIR** (NeurIPS 2023) | Restormer + learned prompts for all-in-one restoration. Your Phase 5 competitor. Covers denoising, deraining, dehazing — **not SR**. |
| **AirNet** (CVPR 2022) | All-in-one via contrastive degradation encoding. Alternative conditioning approach. |
| **NAFNet** (ECCV 2022) | Drops attention almost entirely, still very strong on SIDD/GoPro. The baseline that embarrasses over-engineered attention. Include it as a control if time allows. |

### 2.0 Literature check — done 2026-08-20, against sources not abstracts

Two prior-art claims in this plan were checked directly. One was wrong; the
other held.

**X-Restormer: the plan described it incorrectly.** It does NOT "alternately
replace half the MDTA blocks". Reading
`reference/X-Restormer-master/xrestormer/archs/xrestormer_arch.py`, every block
contains both attentions applied sequentially at full strength, at every level.
More importantly there is **no balance quantity anywhere** — no gate, no weight,
no mixing coefficient. Nothing in that architecture represents how much each
axis contributes, so there is nothing to fix at 50/50 and nothing to compare a
learned gate against. Corrected throughout §2.1.

**PromptIR: the plan is right that it excludes SR.** `options.py` lists
`['denoise_15','denoise_25','denoise_50','derain','dehaze']`. But its
`PromptGenBlock` is closer to DATNet's Phase-5 conditioning than the plan
admits: it pools features, softmaxes a linear layer to infer the degradation
blindly, and blends five learned prompt tensors. That is the same *shape* of
idea as a degradation-conditioned gate. The honest distinction is
**interpretability** — PromptIR mixes five opaque tensors; DATNet mixes two
named, physically-meaningful attention axes, so `g` can be read as a preference.

**The field-level check.** The IEEE TPAMI 2025 survey on all-in-one image
restoration covers denoising, deraining, dehazing, deblurring, low-light and
weather degradations. **Super-resolution is not in its taxonomy**, and it
highlights no method handling SR jointly with other degradations in one model.

> So the all-in-one-including-SR gap is real as of a 2025 survey. Note *why* it
> is open: the survey frames all-in-one as targeting "multiple simultaneous
> degradations within the same category" rather than resolution enhancement
> alongside other corruptions — which is exactly the structural obstacle §2.1
> predicted (resolution mismatch, ~10 dB difference in operating range). The gap
> exists because it is hard, not because nobody noticed. That makes it a real
> contribution and a real risk at this compute budget.

**Rule: check prior art against the source.** The X-Restormer paper description
and its implementation differed enough to change this project's positioning.
Re-check before writing up — the LoViF 2026 challenge shows the area is moving.

---

### 2.1 X-Restormer, and what is left for you

X-Restormer (Chen et al., ECCV 2024, arXiv 2310.11881) is uncomfortably close to the original idea, and you need to know exactly how close before you commit.

**What it already did:**

- Started from Restormer and put **both** attention types inside **every** block,
  applied **sequentially at full strength** — verified by reading the source
  (`reference/X-Restormer-master/xrestormer/archs/xrestormer_arch.py`,
  `TransformerBlock.forward`):

  ```python
  x = x + channel_attn(norm1(x))     # channel (MDTA), full strength
  x = x + channel_ffn(norm2(x))
  x = x + spatial_attn(norm3(x))     # spatial (OCA), full strength
  x = x + spatial_ffn(norm4(x))
  ```

  Every level of the U-Net is built from this same block. So it does combine
  channel-axis and spatial-axis attention in one backbone — but note what it is
  *not* doing, because the distinction is the whole of DATNet's claim:

  > **X-Restormer contains no representation of the channel/spatial balance at
  > all.** There is no weight, gate or mixing coefficient anywhere — both
  > attentions simply run, one after the other. There is nothing to inspect,
  > nothing to condition on, and nothing that could differ between degradations.

  It also spends **two FFNs per block** (one after each attention), which is why
  its blocks are heavier than DATNet's single-FFN DATB.
- Evaluated on **all five tasks**: super-resolution, denoising, deblurring, deraining, dehazing. It does not omit SR.
- Its motivating finding is that "networks which excel in certain tasks often fail to deliver satisfactory results in others," and that a general backbone must meet the functional requirements of diverse tasks.

**Read that last point carefully — it validates your premise rather than killing it.** The observation that no single restoration backbone generalises across tasks is exactly the gap you identified independently. It is now a citable, peer-reviewed motivation for your project instead of an assumption you have to defend.

**What it did not do — your remaining white space:**

| Dimension | X-Restormer | DATNet |
|---|---|---|
| Combination mechanism | **Sequential** — both attentions in every block, each at full strength | **Parallel** branches from the same input, fused by a **learned** per-channel gate |
| Spatial attention type | Overlapping Cross-Attention (HAT) | Shifted-window MSA (SwinIR) |
| Axis balance | **No such quantity exists** — nothing in the architecture represents how much each axis contributes | **Learned, explicit and measurable** — produces the per-task axis-preference figure |
| Conditioning | None | Gate **predicted from the degradation** (Phase 5) |
| Training regime | **Separate checkpoint per task** (it is a backbone study) | Single all-in-one checkpoint across tasks |

**The defensible positioning, in one sentence:**

> X-Restormer showed that channel and spatial attention are *jointly* necessary for a general backbone, but applies both unconditionally — its architecture contains no representation of their relative contribution — and still trains one model per task. PromptIR showed that a single model can handle multiple degradations, but omits super-resolution. **DATNet makes the channel/spatial balance an explicit, learned, degradation-conditioned quantity, and targets a single all-in-one checkpoint that includes SR — which no existing all-in-one method does.**

Two things must therefore be true for this project to be worth twelve weeks, and both are cheap to check early:

1. **The axis preference is measurably task-dependent.** If the learned gate lands in the same place for noise as for blur, then applying both attentions unconditionally — as X-Restormer does — was already the right answer, and there is nothing to condition on. Phase 2 tests this directly and cheaply.
2. **All-in-one including SR is actually hard.** If jointly training SR with the other three turns out to be trivial, PromptIR's authors would have done it. It almost certainly is not trivial — the resolution mismatch and the ~10 dB gap in operating range are real obstacles (§3.3, §4 Phase 5) — but confirm this with a quick joint-training smoke test rather than assuming.

**If X-Restormer turns out to be closer than this table suggests once you read the full paper, pivot in week 1, not at submission.** The all-in-one-with-SR angle survives almost any such discovery, because it is a different problem setting rather than a different block design.

---

## 3. Architecture — DATNet

### 3.1 Dual-Axis Transformer Block (DATB)

The single new component. Everything else is standard.

```
x  (B, C, H, W)
│
├── LayerNorm (bias-free)
│   │
│   ├── Branch A: MDTA (channel attention)
│   │     qkv  = Conv1x1(C → 3C) → DWConv3x3(3C → 3C, groups=3C)
│   │     split q, k, v → reshape (B, heads, C/heads, H*W)
│   │     L2-normalise q, k along the H*W axis
│   │     attn = softmax(q @ kᵀ · τ)        # (B, heads, C/h, C/h)
│   │     out  = attn @ v → reshape → Conv1x1(C → C)
│   │
│   └── Branch B: Shifted-Window MSA (spatial attention)
│         partition into M×M windows (M = 8)
│         shift by M/2 on alternate blocks
│         standard MSA within each window + relative position bias
│         reverse shift, merge windows
│
├── Fuse:  out = g ⊙ A + (1 − g) ⊙ B          # g is per-channel, in [0,1]
├── x = x + out
│
├── LayerNorm (bias-free)
└── x = x + GDFN(·)
      GDFN: Conv1x1(C → 2γC) → DWConv3x3 → split(a, b) → GELU(a) ⊙ b → Conv1x1(γC → C)
      γ = 2.66
```

**The gate `g` has two variants:**

- **Fixed gate** (Phases 1–4): `g = sigmoid(w)`, where `w` is a learnable per-channel vector, one per block. Learned during training, frozen at inference. This is what produces the axis-preference measurement.
- **Conditioned gate** (Phase 5): `g = sigmoid(MLP(d))`, where `d` is a degradation embedding — either a one-hot task label or the output of a small degradation-classifier head. This is what makes it a genuine all-in-one model.

Build the fixed-gate version first. The conditioned version is a drop-in replacement of one line.

### 3.2 Network topology

Three levels, not Restormer's four. Rationale: four levels means an 8× downsample, which is destructive for super-resolution (a 48×48 LR patch becomes 6×6). SwinIR is flat for exactly this reason. Three levels is the compromise that keeps a large enough receptive field for motion deblurring while remaining usable for SR.

```
input (B, 3, H, W)
  → Conv3x3 → C=32
  → Enc L1: 2 × DATB   @ C=32,  H×W       heads=1
  → Downsample (pixel-unshuffle)
  → Enc L2: 3 × DATB   @ C=64,  H/2×W/2   heads=2
  → Downsample
  → Bottleneck L3: 4 × DATB @ C=128, H/4×W/4  heads=4
  → Upsample (pixel-shuffle) + skip from L2
  → Dec L2: 3 × DATB   @ C=64
  → Upsample + skip from L1
  → Dec L1: 2 × DATB   @ C=32
  → Refinement: 2 × DATB @ C=32
  → TAIL (see below)
```

Estimated parameter count at this config: **~5–7 M**. Restormer at full size is 26 M; SwinIR classical-SR is ~11.8 M; Uformer-B is ~51 M. You are not competing on absolute numbers — see §6.

### 3.3 Two tails

| Tail | Used for | Definition |
|---|---|---|
| Restoration | denoise, deblur, derain | `Conv3x3(C → 3)`, then **add the input** (global residual) |
| Super-resolution | SR ×2, ×3, ×4 | `Conv3x3(C → 3·s²)` → `PixelShuffle(s)` → `Conv3x3` |

For SR, **bypass the downsampling path entirely** — run the L1 blocks at native LR resolution and let PixelShuffle do the upsampling at the end. Do *not* bicubic pre-upsample the input: at ×4 that costs 16× the compute and still lands below SwinIR.

One tail per scale factor. `s=2` and `s=4` are separate tails, and in the single-task regime, separate checkpoints.

### 3.4 Gate initialisation (resolution-aware prior)

Initialise the gate bias so that high-resolution levels start channel-leaning and low-resolution levels start spatial-leaning. Justification: at full resolution an 8×8 window covers almost no semantic content and spatial attention is expensive; at H/4 the same window covers 4× the receptive field.

| Level | Resolution | `g` init (channel weight) |
|---|---|---|
| L1 | H×W | 0.8 |
| L2 | H/2 | 0.5 |
| L3 | H/4 | 0.2 |

This is an initialisation, not a constraint — the gate is free to move. It speeds convergence and makes the Phase 1 vs Phase 2 comparison cleaner.

**Window-size constraint:** with M = 8 and three levels, the L3 feature map must be at least 8×8, so training patches must be ≥ 32×32. At the planned 128×128 patch size, L3 is 32×32 = sixteen 8×8 windows. Fine.

---

## 4. Experiment plan

### Ground rule: separate checkpoints per task

Phases 1–4 each train a **fresh model from scratch**. Do **not** fine-tune the denoising checkpoint on deblurring — that gives catastrophic forgetting and, more importantly, breaks comparability with Restormer / SwinIR / Uformer, all of which report one checkpoint per task. Joint all-in-one training is Phase 5, trained from scratch on all tasks simultaneously.

### Ground rule: parameter matching

Every phase runs three variants at **matched parameter count**:

| Variant | Description | Stands in for |
|---|---|---|
| `channel_only` | Branch B removed | Restormer-lite |
| `window_only` | Branch A removed | SwinIR / Uformer-lite |
| `dual` | Both branches, learned gate | DATNet |

Removing a branch removes parameters, so the single-axis variants must be widened to match. Use **width** as the matching knob (a single scalar), not block count — changing depth introduces a second confound. Report exact parameter counts and FLOPs at 256×256 in every table. If you report a win at 2× the FLOPs, it is not a win.

You are retraining your own baselines rather than quoting published numbers, and this is correct: published Restormer numbers come from a 26 M model trained on 8×A100 with progressive patch sizes up to 384². Quoting those against your 6 M model would be meaningless.

---

### Phase 0 — Scaffolding *(build once, reuse everywhere)*

Task-agnostic infrastructure. Nothing here depends on which degradation you tackle first.

- [ ] `DATB` with `mode ∈ {channel_only, window_only, dual}`
- [ ] Full network with configurable width / block counts / tail type
- [ ] Parameter counter + FLOP counter (`fvcore` or `ptflops`), used to auto-match variants
- [ ] Dataloader base class: patch cropping, augmentation (flip + 90° rotation), on-the-fly degradation hooks
- [ ] Train loop: AMP, gradient accumulation, cosine LR with warmup, EMA of weights, checkpoint/resume
- [ ] Eval: PSNR + SSIM with **per-task channel and border conventions** (see §5)
- [ ] Logging: TensorBoard or W&B, plus per-block gate values logged every N steps — this is the data for your key figure

**Estimated effort:** 3–5 days.

**Sanity checks before moving on:**
1. `dual` with the gate hard-clamped to 1.0 should numerically match `channel_only`
2. Overfit a single training image to > 50 dB — if it can't, the training loop is broken
3. Parameter counts of the three variants agree within 2%

---

### Phase 1 — Denoising *(the kill switch)*

**Data:** DIV2K 800 training images + **synthetic Gaussian noise generated on the fly**, σ ∈ {15, 25, 50}. No paired-data download required, instant iteration, and DIV2K is the same dataset Phase 3 needs — one download covers two phases.

**Test:** CBSD68, Kodak24, Urban100 (colour, RGB PSNR).

Defer SIDD real-noise (~12 GB) until the hypothesis is validated. Add it later as a "real noise" column if the synthetic result holds.

**Runs:** `channel_only`, `window_only`, `dual` — 3 runs, ~100 k iterations each.

**Prediction:** `dual` beats both; the learned gate settles **channel-leaning** across most blocks.

> **GATE — do not proceed if this fails.** If `dual` does not beat `max(channel_only, window_only)` at matched parameters, stop. First check whether it is a training-budget artifact (dual has two attention paths and may need more iterations to converge — rerun the best config at 2× iterations before concluding). If it still fails, the core hypothesis is wrong and the project needs rethinking. Better to learn this in week 3 than week 12.

---

### Phase 2 — Deblurring *(the money experiment)*

**Data:** GoPro, 2,103 training pairs / 1,111 test.
**Test:** GoPro (RGB PSNR). Add HIDE and RealBlur-J/R for generalisation if time allows.

**Runs:** same three variants, ~200 k iterations (GoPro converges more slowly than synthetic denoising).

**Prediction:** `dual` beats both, and the gate now settles **spatial-leaning**.

> **This is the most informative experiment in the project.** If the gate flips direction between Phase 1 and Phase 2 — channel-dominant for noise, spatial-dominant for blur — that plot is the central figure of the paper. It converts the architecture from "we fused two things" into "we identified a degradation-dependent structural property and built a mechanism that exploits it." Everything else is supporting material.
>
> Plot it as: x-axis = block index (L1 → L3 → decoder), y-axis = mean gate value, one line per task. If the lines separate, you have a paper.

**Watch for overfitting** — 2,103 pairs is small. Track the train/test gap and use the EMA weights for evaluation.

---

### Phase 3 — Super-resolution *(the engineering lift)*

The only phase that changes the architecture rather than just the data. Budget accordingly.

**Changes required:** add the PixelShuffle tail, bypass the downsampling path, remove the global residual add, handle multiple scale factors.

**Data:** DIV2K (or DF2K = DIV2K + Flickr2K if disk allows), bicubic ×4 degradation.
**Test:** Set5, Set14, BSD100, Urban100, Manga109.

**Start with ×4 only.** Add ×2 only if the ×4 result is worth reporting. Two scales × three variants = six runs, which is too much for one consumer GPU in a reasonable window.

> **Protocol trap — read carefully.** SR papers report **PSNR/SSIM on the Y channel of YCbCr, with `scale` pixels shaved from each border**. Denoising and deblurring papers report **RGB PSNR on the full image**. Deraining papers (including Restormer's) report **Y-channel**. If you use one convention throughout, your numbers will silently fail to match any published table and the comparison is worthless. Implement the conventions per task in Phase 0 and unit-test them against a published number.

---

### Phase 4 — Deraining *(optional — cut this first)*

> **Priority note (2026-08-20).** The TPAMI 2025 survey confirms the
> all-in-one-including-SR gap is the strongest surviving contribution (§2.0).
> That makes **Phase 3 (SR) a prerequisite for Phase 5**, and demotes this
> phase. Deraining is already the least informative experiment — the hypothesis
> predicts "mixed", the least discriminating outcome — and it is now also the
> least strategically useful. **If time is short, cut Phase 4 before Phase 3.**


**Data:** Rain13K, 13,712 pairs, ~1 GB.
**Test:** Rain100L, Rain100H, Test100, Test1200 (**Y-channel** PSNR/SSIM).

Converges fast, costs little, adds a fourth column. Low risk and low information — the hypothesis predicts "mixed," which is the least discriminating outcome. Add it if the timeline allows; drop it without regret if it doesn't.

---

### Phase 5 — All-in-one *(the actual contribution — now confirmed unclaimed)*

Joint training from scratch on all completed tasks, with the **conditioned gate**.

**Two sub-variants, both worth running:**

- **Oracle conditioning:** `d` = one-hot task label. Upper bound; tells you how much the routing mechanism can possibly buy.
- **Blind conditioning:** `d` = output of a small degradation-classifier head trained jointly. The realistic setting, and the one PromptIR/AirNet operate in.

**Task sampling:** proportional to √(dataset size), not uniform. GoPro has 2 k pairs and Rain13K has 13.7 k — uniform sampling drowns deblurring. Additionally, normalise each task's loss by a running mean of its own magnitude, because SR sits around 26–32 dB while denoising sits near 40 dB and the raw gradients are not comparable.

**Compare against:** your four single-task checkpoints (the natural upper bound), X-Restormer's per-task results, and PromptIR / AirNet on the tasks they cover.

> **This phase carries the contribution.** PromptIR and AirNet are all-in-one but exclude SR; X-Restormer covers SR but trains one checkpoint per task. A single checkpoint that handles denoising, deblurring, deraining **and** super-resolution is the thing nobody has shipped. Even a result that trails the single-task checkpoints by a few tenths of a dB is publishable if it is one model doing all four — that is the standard all-in-one papers are judged by.

---

## 5. Evaluation protocol reference

Keep this table next to the eval code. Getting it wrong invalidates every comparison.

| Task | Colour space | Border shave | Test sets |
|---|---|---|---|
| Gaussian denoising (colour) | RGB | none | CBSD68, Kodak24, Urban100 |
| Real denoising (SIDD/DND) | RGB | none | SIDD, DND |
| Deblurring | RGB | none | GoPro, HIDE, RealBlur |
| Super-resolution | **Y of YCbCr** | **`scale` px** | Set5, Set14, BSD100, Urban100, Manga109 |
| Deraining | **Y of YCbCr** | none | Rain100L/H, Test100, Test1200 |

Approximate reference points from the literature, at *much* larger budgets than yours — for orientation, not as targets:

| | Restormer | SwinIR | NAFNet |
|---|---|---|---|
| SIDD (RGB PSNR) | ≈ 40.0 | — | ≈ 40.3 |
| GoPro (RGB PSNR) | ≈ 32.9 | — | ≈ 33.7 |
| Rain100L (Y PSNR) | ≈ 39.0 | — | — |
| Set5 ×4 (Y PSNR) | — | ≈ 32.9 | — |
| Urban100 ×4 (Y PSNR) | — | ≈ 27.5 | — |

---

## 6. Hardware budget — measured, not assumed

The earlier draft of this section targeted a generic 12 GB card. The actual machine is tighter in two places, one of which the draft did not mention at all.

| | Value | Consequence |
|---|---|---|
| GPU | RTX 4060 **Laptop**, 8 GB GDDR6, Ada (sm_89) | bf16 and TF32 are native — use bf16, skip `GradScaler` |
| Bandwidth | 128-bit, ~272 GB/s | the bottleneck is memory, not FLOPs; small-width convs will not saturate the card |
| Power | laptop TGP, sustained clocks drop after ~10 min under load | measured throughput on a 10-minute run overstates a 10-hour run |
| **System RAM** | **15.4 GB** | **the binding constraint on the data pipeline, not VRAM** |
| CPU | Core Ultra 7 155H, 16C/22T | enough for on-the-fly noise and bicubic; not enough RAM to also cache decoded images |
| Disk | 442 GB free | no constraint; all five datasets together are under 30 GB |

### 6.1 The RAM constraint

Windows `DataLoader` workers use **spawn**, so every worker is a full process that re-imports torch and holds its own copy of anything the dataset object references. At ~350–450 MB resident per worker, plus Windows itself, plus a browser, 15.4 GB does not leave room for the "cache the whole dataset in RAM" pattern that most restoration repos default to.

Consequences, all already reflected in the configs:

- `num_workers: 4` for single-task phases, `2` per task in Phase 5 (four loaders running at once).
- Datasets hold **file paths**, never decoded arrays.
- Pre-crop DIV2K to 480×480 sub-images on disk before Phase 1. Random-cropping a 2040×1356 PNG means decoding 2.8 M pixels to use 16 k of them; sub-images cut decode cost by roughly 10× and are the single largest throughput win available on the CPU side.
- If throughput is still worker-bound, drop to `num_workers: 2` with `prefetch_factor: 4` rather than adding workers.

### 6.2 The VRAM budget

Usable VRAM is **not** 8.0 GB. Reserve for the CUDA context, cuDNN/cuBLAS workspaces, and allocator fragmentation; plan against a **7.0 GB ceiling** and treat anything above it as an OOM waiting to happen three hours into a run.

One free 0.5–1.0 GB is available: this machine has an Intel Arc iGPU alongside the 4060. Force the desktop and the browser onto the iGPU (Windows Settings → Display → Graphics → set Chrome/Edge to *Power saving*) so the 4060 carries nothing but training.

**Approximate activation cost per training image at 128², width 32, bf16, no gradient checkpointing:**

| Level | Channels | Pixels | Blocks | ≈ per image |
|---|---|---|---|---|
| L1 (+decoder, +refine) | 32 | 16,384 | 6 | ~225 MB |
| L2 (+decoder) | 64 | 4,096 | 6 | ~113 MB |
| L3 bottleneck | 128 | 1,024 | 4 | ~38 MB |
| | | | **total** | **~375 MB** |

GDFN dominates that — its hidden width is `2 × 2.66 × C`, so it stores roughly 16·C·HW against the attention branches' ~8·C·HW each. Parameters, gradients, Adam moments and the EMA shadow together are only ~120 MB, which is noise at this scale.

That puts micro-batch 8 at roughly 3.0 GB of activations plus workspace — probably fine, but the widened `window_only` arm stores larger window attention maps than `dual` and will be the first to fall over. **Do not guess: `scripts/bench.py` sweeps micro-batch per arm and reports the largest that fits under the ceiling.**

### 6.3 Effective batch is fixed; micro-batch is not

This is the one rule in this section that is not negotiable.

> **`effective_batch = 16` for every arm of every phase. The micro-batch is whatever fits, and gradient accumulation makes up the difference.**

The three ablation arms do not consume the same VRAM per image. If memory pressure is absorbed by changing the batch size, then `dual` beating `window_only` could be an artifact of `dual` having trained at batch 8 while `window_only` trained at batch 4 — a confound that would not show up anywhere in the results table and would invalidate the entire study. Absorb it in the micro-batch, where it costs wall-clock and nothing else.

`Trainer` enforces `effective_batch % micro_batch == 0` and, on OOM, halves the micro-batch and doubles the accumulation rather than dropping the effective batch. The backoff is written to the ledger so it is visible afterwards.

### 6.4 Settings

| Setting | Value | Note |
|---|---|---|
| Base width | 32 | ~6 M params for `dual`; the single-axis arms are widened to match at run time |
| Blocks | [2, 3] enc / 4 bottleneck / [3, 2] dec / 2 refine | |
| Heads | [1, 2, 4] | head dim is 32 at every level |
| Window size | 8 | L3 is 32×32 at a 128² patch — sixteen windows |
| GDFN expansion | 2.66 | |
| Patch | 128² (SR: 64² LR → 256² HR) | fixed; no progressive resizing |
| **Effective batch** | **16** | constant across all arms — see §6.3 |
| Micro-batch | 4 (starting point; `bench.py` decides) | |
| AMP | **bf16** | Ada-native; no loss scaling, so no scaler-skipped steps to reason about |
| Optimiser | AdamW, lr 3e-4 (SR 2e-4), cosine → 1e-6, 5 k warmup, wd 1e-4, clip 1.0 | |
| EMA | decay 0.999, evaluated from EMA weights | worth a few tenths of a dB on GoPro alone |
| Loss | L1 (+ FFT at weight 0.05 for deblurring only) | |

**Deliberately not defaults:**

- `torch.compile` — **off**. Windows support is still the flaky path, and a compile failure at hour six of a run is a worse trade than the 20–30% it might buy. Try it once, on a short segment, after Phase 1 works.
- `channels_last` — **off until measured**. It helps conv-dominated networks; this one is reshape- and attention-dominated, where it can cost more in layout conversions than it saves. `bench.py` can settle it in ten minutes.
- Gradient checkpointing — **off for Phases 1–4, on for Phase 5**. It trades ~30% throughput for a large memory saving. Phases 1–4 should not need it at 128²; Phase 5 carries four tails and a conditioning head and will.

### 6.5 Throughput — MEASURED, not projected

The v1 draft guessed 3-5 it/s from published desktop setups, then a revised
guess put it at 1.2-2.5. Both are now superseded by measurement on this card, on
mains power (`docs/CHECKPOINT.md` §4 carries the full tables).

| | width 32 (2.04 M) | width 56 (6.10 M) |
|---|---|---|
| best micro-batch | 8, accum 2 | 8, accum 2 |
| peak VRAM, `dual` | 3.74 GB | 6.43 GB |
| `dual`, synthetic inputs | **2.14 opt-it/s** | **1.15 opt-it/s** |
| 3 arms x 100 k | **37.1 GPU-h** | **63.1 GPU-h** |

Three things the measurement changed:

1. **Mains power is worth 30%.** The same benchmark on battery gave 1.64 it/s
   for `dual` instead of 2.14. Never build a schedule on a battery number.
2. **FLOPs badly overstate the cost of going wider.** Width 56 has 3x the FLOPs
   of width 32 but costs only 1.7x the wall-clock, because width 32 does not
   saturate the card. An earlier estimate in this plan said ~140 GPU-h for width
   56; the measured figure is 63.1. Do not size runs from FLOP ratios here.
3. **Real training runs at ~1.67 it/s, and the gap to 2.14 is not fixable.**
   Pre-cropping (10x less decode) bought 5%; removing EMA bought nothing;
   removing gradient clipping bought nothing; workers saturate at 2. In-loop
   validation and checkpointing explain ~12%, and the remaining ~22% is most
   likely boost-vs-sustained clocks on a laptop GPU -- `bench.py` times 12 steps
   from idle. This is why it is documented as an upper bound.

**Budget from the real number, not the synthetic one.** At 1.67 it/s a 100 k run
is 16.6 h, not 13.0.

### 6.6 Training in sections

A 200 k-iteration run is not a thing you start and watch. It is a **queue of bounded segments**, each of which resumes exactly where the last one stopped.

**Definitions used throughout the code:**

| Term | Meaning |
|---|---|
| **Segment** | One `train.py` invocation. Trains at most `--segment-iters`, checkpoints, exits. Default 10 k ≈ 1.5–2.5 h — one evening. |
| **Section (run)** | One (phase, variant, seed) triple trained to `total_iters`. Lives in `runs/<phase>__<variant>__seed<N>/`. |
| **Queue** | An ordered list of sections, driven by `run_segments.ps1`. |

**What makes a resume exact.** The checkpoint carries the model, EMA shadow, AdamW moments, schedule position, AMP scaler state, and **all four RNG streams** (Python, NumPy, torch CPU, torch CUDA). Without the RNG state, every resume re-draws the same augmentations and noise realisations it already saw — harmless once, a real bias over a run stopped and restarted thirty times, which is exactly the regime a laptop imposes. Checkpoints are written to a temp file and `os.replace`d, so a power cut mid-save cannot destroy the last good one.

**What the ledger buys.** Every segment appends a JSON line with iterations done, wall seconds, achieved it/s, and peak VRAM. `Ledger.summary()` turns that into percent complete and a measured ETA. After two segments you know what the run actually costs on your card, under your thermals, and can schedule the rest instead of guessing.

**Practical cadence:**

```powershell
# an evening: 4 segments, then stop, whatever state it is in
.\scripts\run_segments.ps1 -Config configs\phase1_denoise.yaml -Variants dual `
    -SegmentIters 10000 -MaxHours 8 -CoolDownSeconds 90
```

`-CoolDownSeconds` is not superstition. Sustained load pushes a laptop 4060 into thermal throttling; a short idle gap between segments measurably holds clocks higher over a multi-day queue. Watch the per-segment `it_per_s` in the ledger — if it decays monotonically across a night, raise the cooldown.

**Order the queue by information, not by phase number.** Run `dual` first in every phase. If `dual` does not beat the single-axis arms, Phase 1 is a kill switch (§4) and the other two arms of later phases were never worth running.

---

## 7. Timeline, budgeted in GPU-hours

The earlier draft budgeted in calendar weeks, which hides the thing that actually binds: this is one 8 GB laptop, and every phase runs **three** arms. Budget in GPU-hours first, convert to calendar second.

Figures below assume the midpoint of the §6.5 band, **1.8 optimiser-steps/second**. Replace them with your measured number after `bench.py`.

### 7.1 Compute budget — from measured throughput

Figures below use the **real** width-32 rate of **1.43 opt-it/s** measured on a
200-iteration Phase-1 segment against uncropped DIV2K (see §6.5). Pre-cropping
should lift this toward the 2.14 synthetic ceiling; re-derive the table if it
does.

| Phase | Runs | Iterations each | GPU-hours @ 1.43 it/s | Notes |
|---|---|---|---|---|
| 0 — scaffolding | — | — | **done** | sanity, bench, smoke test all green |
| 1 — denoising | 3 arms | 60 k | **35** | screening length; enough to rank three arms |
| 1b — confirm | 1 (`dual`) | +40 k | **8** | only if the Phase-1 gate passes |
| 2 — deblurring | 3 arms | 100 k | **58** | do not shorten — the gate-flip experiment |
| 3 — super-resolution | 3 arms | 100 k | **58** | x4 only |
| 4 — deraining | 3 arms | 50 k | **29** | optional |
| 5 — all-in-one | 2 (oracle, blind) | 150 k | **58** | grad-checkpointing on, so ~30% slower per step |
| | | **core (0–3)** | **~160** | |
| | | **full (0–5)** | **~246** | |

Add roughly 30-40% for evaluation passes, failed runs and reruns after a bug:
call it **215 GPU-hours core, 330 full**.

**If Phase 2 onward runs at width 56** (6.10 M params, the size this plan
actually describes), multiply the phase figures by ~1.7: core becomes ~250
GPU-hours before overhead. That is the real cost of the model-size decision in
§11, and it is a 1.7x multiplier rather than the 3x the FLOP count implies.

### 7.2 Calendar

| Usable GPU time per day | Core (phases 0–3) | Full (0–5) |
|---|---|---|
| 6 h/day | ~4 weeks | ~6.5 weeks |
| 10 h/day (overnight queues) | ~2.5 weeks | ~4 weeks |

Add 2 weeks for writing, ablation tables and figures, and roughly 1 week of slack for the SR protocol work (§4 Phase 3), which is engineering rather than training and does not parallelise with the GPU queue — except that it *can* be written while a Phase-2 queue runs overnight. Do that.

**Realistic total: 9–12 weeks at 6 h/day, 7–9 weeks if you run overnight queues.** The original 12-week estimate survives, but only because Phase 1 is shortened to a screening run and the whole thing assumes the GPU is genuinely busy most nights.

### 7.3 Screening versus confirmation

The change that makes the budget fit: **an ablation does not need a publishable number, it needs a reliable ranking.**

- **Screening runs** (60–100 k) answer "does `dual` beat both single-axis arms at matched parameters?" and "where does the gate settle?" Both are comparisons between three models trained identically, so a shorter budget costs absolute PSNR but not the comparison.
- **Confirmation runs** (the full length) are for the numbers that go in a table, and are only spent on configurations that already survived screening.

The one place this does not apply is **Phase 2**. The gate-flip figure is the central claim, and a gate that has not converged is indistinguishable from a gate that does not move. Give Phase 2 its full 100 k per arm and check the gate trajectory in `gates.jsonl` — if the per-block values are still drifting at the end, the run was too short and the negative result would be an artifact.

### 7.4 Suggested queue order

Ordered so that the cheapest kill switch fires first and nothing expensive runs before the thing that could invalidate it.

| # | What | GPU-h | Kill condition |
|---|---|---|---|
| 1 | Read X-Restormer and PromptIR in full | 0 | differentiation does not hold → re-frame before writing code |
| 2 | Phase 0: sanity + bench + protocol tests | ~2 | any sanity check fails |
| 3 | Phase 1 `dual` @ 60 k | 9 | — |
| 4 | Phase 1 `channel_only`, `window_only` @ 60 k | 19 | `dual` ≤ max(single) → rerun `dual` at 2× before concluding, then §8 |
| 5 | **Gate check**: plot Phase-1 gates | 0 | gates all sit at their initialisation → the gate is not learning; debug before spending Phase 2 |
| 6 | Phase 2, three arms @ 100 k | 46 | gate does not flip → fall back to the all-in-one contribution (§8) |
| 7 | **Central figure**: `plot_gates.py` phase 1 vs phase 2 | 0 | |
| 8 | Phase 3 SR, three arms @ 100 k | 46 | |
| 9 | Phase 5 oracle, then blind | 46 | |
| 10 | Phase 4 derain, if time remains | 23 | |

Step 5 is new and costs nothing. A gate pinned at its initialisation across all of Phase 1 means the fusion is not receiving gradient — a wiring bug, not a research result — and catching that before spending 46 hours on Phase 2 is worth the two minutes.

### 7.5 Cut list, in order

When the timeline slips, cut from the top:

1. Phase 4 (deraining) — 23 h, predicted to be the least informative outcome.
2. The `blind` conditioning arm of Phase 5 — halves Phase 5; report `oracle` as an upper bound and say so.
3. Phase 3 down to two arms (`dual` vs the better single-axis arm from Phase 1) — 15 h.
4. Phase 1b confirmation — report the 60 k screening number with the length stated.

Do **not** cut: the three-way ablation in Phase 2, the parameter matching, or the per-task evaluation protocols. Each of those, removed, makes the remaining results unpublishable rather than merely weaker.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `dual` ≤ single-axis at matched params | Medium | Kill switch at end of Phase 1. Rerun at 2× iterations first to rule out a convergence artifact. If confirmed, pivot to the pure all-in-one routing story, which does not depend on dual beating single on any individual task. X-Restormer's results make outright failure here unlikely — it already showed the combination helps. |
| Gate does not flip between tasks | **Medium-High** | This is the load-bearing risk. If the gate does not move between noise and blur, then applying both attentions unconditionally (X-Restormer) was already correct and conditioning buys nothing. Fall back to the all-in-one-including-SR contribution, which is independent of the gate result. Report the negative result honestly in the ablation section — it saves the next person the experiment. |
| X-Restormer is too close | **High** | Already partly realised: it covers all five tasks and combines both attention axes. Surviving differentiators are the *learned, conditioned* gate and the single all-in-one checkpoint including SR (§2.1). Read the full paper in week 1 and confirm those two hold before Phase 0 is finished. |
| SR phase overruns | High | Ship ×4 only. Drop ×2/×3. Drop Manga109 if eval is slow. |
| OOM on the 8 GB card | Medium | Automatic: `Trainer` catches OOM, halves the micro-batch and doubles accumulation, keeping the effective batch at 16, and records the backoff in the ledger. Manual escalation: gradient checkpointing, then width 24. Never lower `effective_batch` — that confounds the ablation (§6.3). |
| GoPro overfitting (2 k pairs) | Medium | EMA weights, heavier augmentation, early stopping on test PSNR. |
| Total time overrun | High | Phases 4 and 5 are both cuttable; phases 0–3 alone constitute a complete study. Cut in the order given in §7.5. |
| **GPU-hours exceed the calendar** | **High** | The real budget is ~170 GPU-hours for phases 0–3 and ~270 for the full plan (§7.1) — three arms per phase on one laptop. Mitigations, in order: screening-length runs (§7.3), overnight segment queues, and the §7.5 cut list. Re-derive the budget from `bench.py` before trusting any of these figures. |
| **Data pipeline starves the GPU** | Medium | 15.4 GB system RAM with Windows spawn workers means no in-RAM dataset cache (§6.1). Pre-crop DIV2K to 480×480 sub-images (`scripts/prepare_subimages.py`) and cap `num_workers` at 4. Symptom to watch: `it_per_s` in the ledger insensitive to micro-batch size. |
| **Thermal throttling on a laptop** | Medium | Sustained load drops sustained clocks, so a benchmark measured over ten minutes overstates a ten-hour run. Segment queues with `-CoolDownSeconds` (§6.6); watch for monotonically decaying per-segment `it_per_s`. |
| Phase-5 SR gap is uninterpretable | Medium | The joint model uses the U-Net for SR while Phase 3 uses the flat trunk (§11.1). Run a single-task U-Net SR control so the backbone change can be separated from the cost of joint training. |

---

## 9. Immediate next actions

Phase 0 is now written but unrun (§11.1). In order:

1. **Create the environment.** `conda create -n datnet python=3.12` and install torch from the CUDA index — the system Python is 3.14, which has no torch wheels. See `README.md`.
2. **Run the three checks.** `tests/test_metrics.py`, then `scripts/sanity.py`, then `scripts/bench.py --patch 128 --vram-ceiling 7.0`. Nothing in §6 or §7 is real until `bench.py` has produced measured it/s and VRAM numbers on this card; paste them back into the phase configs.
3. **Read X-Restormer and PromptIR in full.** This still gates everything. The differentiation to confirm is the *learned, conditioned* gate and the single all-in-one checkpoint including SR (§2.1).
4. **Download the data.** DIV2K HR (~3.3 GB) plus the small test sets — CBSD68, Kodak24, Set5, Set14, Urban100 — which are tens of MB in total. Pre-crop DIV2K to 480×480 sub-images before Phase 1 (§6.1); it is the largest CPU-side throughput win available.
5. **Force the display onto the Intel iGPU** so the 4060 carries nothing but training (§6.2).
6. **Launch the Phase-1 queue, `dual` first.** `run_segments.ps1 -Config configs\phase1_denoise.yaml -Variants dual -SegmentIters 10000`. Check `gates.jsonl` after the first 20 k iterations: if every gate is still sitting exactly at its initialisation, the fusion is not receiving gradient and it is a bug, not a result.

---

## 10. Repository map

```
datnet/
  models/
    attention.py    MDTA (channel axis) + shifted-window MSA (spatial axis)
    gate.py         AxisGate -- fixed and degradation-conditioned variants
    block.py        DATB: norm -> fused attention -> norm -> GDFN
    ffn.py          GDFN
    layers.py       bias-free LayerNorm2d, pixel-(un)shuffle resampling, skip fusion
    datnet.py       backbone (unet | flat) + restoration / SR tails
    allinone.py     Phase 5: shared backbone, per-task tails, conditioned gates
    conditioning.py oracle and blind degradation encoders
    build.py        config -> model
  data/
    datasets.py     Gaussian-denoise (on the fly), paired, SR, full-image test sets
    bicubic.py      MATLAB-compatible imresize
    transforms.py   paired crop, D4 augmentation
    multitask.py    sqrt-proportional task sampling + per-task loss balancing
    build.py        config -> DataLoader factory
    io.py           cv2 fast path, PIL fallback
  engine/
    trainer.py      segmented loop: AMP, accumulation, EMA, OOM backoff, gate logging
    checkpoint.py   atomic save/load including all four RNG streams
    ledger.py       per-segment JSONL: it/s, peak VRAM, measured ETA
    losses.py       L1 / Charbonnier, FFT, auxiliary classifier term
    ema.py, scheduler.py
  evaluation/
    protocols.py    THE per-task convention table (colour space, border shave)
    metrics.py      PSNR / SSIM, MATLAB-equivalent
    inference.py    whole-image, falling back to feathered tiling on OOM
    evaluate.py     test-set loop + a light in-training validation callable
  utils/
    complexity.py   parameter and FLOP counting
    matching.py     width matching across the three ablation arms
configs/            one YAML per phase
scripts/
  sanity.py         the four Phase-0 checks
  bench.py          measure it/s and VRAM per arm; size the runs from the result
  prepare_subimages.py  pre-crop DIV2K to 480x480 tiles (§6.1)
  smoke_test.py     full training path on synthetic data, incl. resume
  fetch_data.py     resumable DIV2K download
  fetch_reference.py  official implementations, for the equivalence tests
  train.py          one segment of one single-task run
  train_multitask.py Phase 5
  plot_gates.py     the axis-preference figure
  run_segments.ps1  segment queue runner
tests/test_metrics.py  pins the evaluation conventions
tests/test_reference_equivalence.py  our blocks vs Restormer / SwinIR
```

---

## 11. Decisions the implementation had to settle

The plan left five things underspecified. The code commits to an answer for each; these are the places to revisit if a result looks wrong.

**1. Phase 5 uses the U-Net for SR too.** Phases 1–4 use a 3-level U-Net for same-size tasks and a flat trunk for SR — two different backbones. A single all-in-one checkpoint cannot have two backbones, so the joint model uses the U-Net for everything and switches only the tail. This is viable because SR consumes the *LR* image: at a 96×96 LR patch the L3 map is 24×24, well above the 8×8 window minimum. The 48×48-collapsing-to-6×6 problem that motivated the flat topology is avoided by training joint SR at a larger LR patch.

> **This creates a confound that must be controlled.** Any gap between the Phase-3 SR number and the Phase-5 SR column mixes "all-in-one is hard" with "the backbone changed". Run a single-task U-Net SR control before attributing the gap to joint training. It is one extra run and it is the difference between a claim and a guess.

**2. The gate blends attention outputs only, not the FFN.** `out = g·MDTA(x) + (1−g)·SWMSA(x)`, then a shared GDFN. Gating the FFN as well would double the parameter cost of `dual` and make the width matching against the single-axis arms much less clean.

**3. The conditioned gate starts identical to the fixed gate.** The final linear layer of the conditioning MLP is zero-initialised with its bias set to `logit(gate_init)`. An untrained conditioned gate is therefore numerically the fixed gate at its resolution-aware prior, which makes Phase 5 a strict extension of Phases 1–4 rather than a different initialisation that happens to share a name.

**4. bf16, not fp16.** Ada supports bf16 natively. bf16 has fp32's exponent range, so there is no `GradScaler`, no scale-overflow steps silently skipped, and no chance of an inf in the FFT loss quietly poisoning the deblurring run.

**5. Head dimension is held at 32 across all levels.** `heads=[1,2,4]` against widths `[32,64,128]` gives head dim 32 everywhere, so MDTA's attention map is 32×32 at every level and the temperature parameter has the same meaning throughout. When width matching widens an arm to, say, 40, the head count stays fixed and the head dim moves — worth noting in the ablation table, since it is a small uncontrolled difference between arms.

### 11.1 State of the code

Everything above is written but **has not been executed**: there is no `torch` installed on this machine, and Python 3.14 (the only interpreter present) has no torch wheels. Set up the 3.12 environment per `README.md`, then run, in this order:

```powershell
python tests/test_metrics.py    # protocol conventions
python scripts/sanity.py        # shapes, gate identity, width matching, overfit
python scripts/bench.py --patch 128 --vram-ceiling 7.0
```

Treat the first green `sanity.py` as the real end of Phase 0. Until then the parameter counts, the ~6 M estimate, and every hour figure in §7 are projections.

---

## Appendix A — Naming

`DATNet` (Dual-Axis Transformer Network) is a placeholder. Check it against existing published names before committing to it in writing — note that "DAT" is already taken by *Dual Aggregation Transformer* for image super-resolution (ICCV 2023), which is itself another channel/spatial-attention combination and should be added to your reading list.

## Appendix B — Key references

- **X-Restormer** — A Comparative Study of Image Restoration Networks for General Backbone Network Design, ECCV 2024 — [arXiv:2310.11881](https://arxiv.org/abs/2310.11881) · [code](https://github.com/Andrew0613/X-Restormer)
- **PromptIR** — Prompting for All-in-One Blind Image Restoration, NeurIPS 2023 — [arXiv:2306.13090](https://arxiv.org/abs/2306.13090) · [code](https://github.com/va1shn9v/PromptIR)
- **Restormer** — Efficient Transformer for High-Resolution Image Restoration, CVPR 2022
- **SwinIR** — Image Restoration Using Swin Transformer, ICCVW 2021
- **Uformer** — A General U-Shaped Transformer for Image Restoration, CVPR 2022
- **NAFNet** — Simple Baselines for Image Restoration, ECCV 2022
- **AirNet** — All-in-One Image Restoration for Unknown Corruption, CVPR 2022
