# phase1_denoise__channel_only__seed0

- checkpoint: `best.pth` at iteration **30,000**
- parameters: **2.063 M**
- variant: **channel_only**  seed 0

## Test results (EMA weights, RGB PSNR, no border shave)

| test set | sigma | PSNR (dB) | SSIM |
|---|---|---|---|
| CBSD68 | 15 | 33.761 | 0.9280 |
| CBSD68 | 25 | 31.166 | 0.8816 |
| CBSD68 | 50 | 27.944 | 0.7884 |
| McMaster | 15 | 34.243 | 0.9170 |
| McMaster | 25 | 32.084 | 0.8830 |
| McMaster | 50 | 29.037 | 0.8147 |

## Validation during training (McMaster, sigma 25)

| iteration | PSNR | SSIM |
|---|---|---|
| 2,500 | 30.744 | 0.8312 |
| 5,000 | 31.684 | 0.8580 |
| 7,500 | 32.162 | 0.8692 |
| 10,000 | 32.415 | 0.8748 |
| 12,500 | 32.577 | 0.8782 |
| 15,000 | 32.691 | 0.8804 |
| 17,500 | 32.743 | 0.8807 |
| 20,000 | 32.795 | 0.8820 |
| 22,500 | 32.823 | 0.8823 |
| 25,000 | 32.847 | 0.8829 |
| 27,500 | 32.856 | 0.8832 |
| 30,000 | 32.859 | 0.8831 |

## Files

- `gates.png` -- mean gate value per block
- `samples/` -- 4 noisy | restored | ground-truth strips
- `metrics.json` -- machine readable

**This is one arm.** `channel_only` and `window_only` are needed before any of it is interpretable as a result.
