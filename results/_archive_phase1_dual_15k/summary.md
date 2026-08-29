# phase1_denoise__dual__seed0

- checkpoint: `best.pth` at iteration **15,000**
- parameters: **2.037 M**
- variant: **dual**  seed 0

## Test results (EMA weights, RGB PSNR, no border shave)

| test set | sigma | PSNR (dB) | SSIM |
|---|---|---|---|
| CBSD68 | 15 | 33.650 | 0.9271 |
| CBSD68 | 25 | 31.012 | 0.8814 |
| CBSD68 | 50 | 27.819 | 0.7836 |
| McMaster | 15 | 34.159 | 0.9156 |
| McMaster | 25 | 32.066 | 0.8822 |
| McMaster | 50 | 29.023 | 0.8110 |

## Validation during training (McMaster, sigma 25)

| iteration | PSNR | SSIM |
|---|---|---|
| 5,000 | 31.300 | 0.8488 |
| 10,000 | 32.535 | 0.8772 |
| 15,000 | 32.810 | 0.8820 |
| 20,000 | 20.438 | 0.4836 |
| 25,000 | nan | nan |
| 30,000 | nan | nan |

## Files

- `gates.png` -- mean gate value per block
- `samples/` -- 4 noisy | restored | ground-truth strips
- `metrics.json` -- machine readable

**This is one arm.** `channel_only` and `window_only` are needed before any of it is interpretable as a result.
