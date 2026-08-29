# phase1_denoise__dual__seed0

- checkpoint: `best.pth` at iteration **22,500**
- parameters: **2.037 M**
- variant: **dual**  seed 0

## Test results (EMA weights, RGB PSNR, no border shave)

| test set | sigma | PSNR (dB) | SSIM |
|---|---|---|---|
| CBSD68 | 15 | 33.740 | 0.9279 |
| CBSD68 | 25 | 31.170 | 0.8821 |
| CBSD68 | 50 | 27.941 | 0.7878 |
| McMaster | 15 | 34.203 | 0.9170 |
| McMaster | 25 | 32.133 | 0.8843 |
| McMaster | 50 | 29.117 | 0.8155 |

## Validation during training (McMaster, sigma 25)

| iteration | PSNR | SSIM |
|---|---|---|
| 2,500 | 30.068 | 0.8049 |
| 5,000 | 31.313 | 0.8491 |
| 7,500 | 32.002 | 0.8658 |
| 10,000 | 32.370 | 0.8743 |
| 12,500 | 32.605 | 0.8791 |
| 15,000 | 32.741 | 0.8820 |
| 17,500 | 32.814 | 0.8831 |
| 20,000 | 32.856 | 0.8835 |
| 22,500 | 32.875 | 0.8834 |
| 25,000 | 32.854 | 0.8823 |
| 27,500 | 32.750 | 0.8775 |
| 30,000 | 32.667 | 0.8746 |

## Files

- `gates.png` -- mean gate value per block
- `samples/` -- 4 noisy | restored | ground-truth strips
- `metrics.json` -- machine readable

**This is one arm.** `channel_only` and `window_only` are needed before any of it is interpretable as a result.
