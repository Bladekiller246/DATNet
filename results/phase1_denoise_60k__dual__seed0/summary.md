# phase1_denoise_60k__dual__seed0

- checkpoint: `best.pth` at iteration **57,500**
- parameters: **2.037 M**
- variant: **dual**  seed 0

## Test results (EMA weights, RGB PSNR, no border shave)

| test set | sigma | PSNR (dB) | SSIM |
|---|---|---|---|
| CBSD68 | 15 | 33.779 | 0.9253 |
| CBSD68 | 25 | 31.239 | 0.8810 |
| CBSD68 | 50 | 28.039 | 0.7889 |
| McMaster | 15 | 34.299 | 0.9135 |
| McMaster | 25 | 32.279 | 0.8833 |
| McMaster | 50 | 29.275 | 0.8154 |

## Validation during training (McMaster, sigma 25)

| iteration | PSNR | SSIM |
|---|---|---|
| 2,500 | 29.815 | 0.7848 |
| 5,000 | 31.390 | 0.8496 |
| 7,500 | 31.848 | 0.8612 |
| 10,000 | 32.120 | 0.8667 |
| 12,500 | 32.279 | 0.8682 |
| 15,000 | 32.395 | 0.8696 |
| 17,500 | 32.399 | 0.8656 |
| 20,000 | 32.396 | 0.8608 |
| 22,500 | 32.519 | 0.8648 |
| 25,000 | 32.575 | 0.8653 |
| 27,500 | 32.610 | 0.8655 |
| 30,000 | 32.629 | 0.8656 |
| 32,500 | 32.664 | 0.8670 |
| 35,000 | 32.667 | 0.8661 |
| 37,500 | 32.628 | 0.8644 |
| 40,000 | 32.755 | 0.8708 |
| 42,500 | 32.746 | 0.8703 |
| 45,000 | 32.775 | 0.8718 |
| 47,500 | 32.814 | 0.8721 |
| 50,000 | 32.861 | 0.8769 |
| 52,500 | 32.885 | 0.8781 |
| 55,000 | 32.899 | 0.8782 |
| 57,500 | 32.908 | 0.8785 |
| 60,000 | 32.904 | 0.8784 |
| 60,000 | 32.904 | 0.8784 |

## Files

- `gates.png` -- mean gate value per block
- `samples/` -- 4 noisy | restored | ground-truth strips
- `metrics.json` -- machine readable

**This is one arm.** `channel_only` and `window_only` are needed before any of it is interpretable as a result.
