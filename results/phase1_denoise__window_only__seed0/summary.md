# phase1_denoise__window_only__seed0

- checkpoint: `best.pth` at iteration **30,000**
- parameters: **2.042 M**
- variant: **window_only**  seed 0

## Test results (EMA weights, RGB PSNR, no border shave)

| test set | sigma | PSNR (dB) | SSIM |
|---|---|---|---|
| CBSD68 | 15 | 33.777 | 0.9275 |
| CBSD68 | 25 | 31.191 | 0.8820 |
| CBSD68 | 50 | 27.974 | 0.7902 |
| McMaster | 15 | 34.234 | 0.9170 |
| McMaster | 25 | 32.151 | 0.8845 |
| McMaster | 50 | 29.141 | 0.8171 |

## Validation during training (McMaster, sigma 25)

| iteration | PSNR | SSIM |
|---|---|---|
| 2,500 | 29.099 | 0.7518 |
| 5,000 | 31.171 | 0.8415 |
| 7,500 | 31.878 | 0.8606 |
| 10,000 | 32.271 | 0.8698 |
| 12,500 | 32.498 | 0.8749 |
| 15,000 | 32.636 | 0.8778 |
| 17,500 | 32.741 | 0.8803 |
| 20,000 | 32.810 | 0.8819 |
| 22,500 | 32.856 | 0.8830 |
| 25,000 | 32.883 | 0.8834 |
| 27,500 | 32.895 | 0.8836 |
| 30,000 | 32.901 | 0.8839 |

## Files

- `gates.png` -- mean gate value per block
- `samples/` -- 4 noisy | restored | ground-truth strips
- `metrics.json` -- machine readable

**This is one arm.** `channel_only` and `window_only` are needed before any of it is interpretable as a result.
