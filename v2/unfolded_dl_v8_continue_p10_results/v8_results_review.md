# V8 P=10 continuation results

## Source

Recovered original run outputs from the timestamped Drive-download archive into this folder: per-seed summary JSON, final checkpoints, resumable checkpoints, and mean CSV. The completed `DL_V8.ipynb` reported a Tesla T4 run and 100 continuation epochs per seed. The suffixed `_reconstructed.json` remains a separate local reconstruction; use the original `summary_unfolded_v8_p10_continuation.json` for the saved run record.

## Mean test NMSE comparison

Negative DL-minus-GS means DL is better. Values are dB; V8 is averaged across the three seeds.

| SNR (dB) | GS | GD | HS-GS | V8 mean DL | V8 minus GS | Seeds beating GS |
|---:|---:|---:|---:|---:|---:|---:|
| -5 | 7.868 | 7.624 | 4.425 | -1.413 | -9.281 | 3/3 |
| 0 | 3.060 | 3.028 | 0.809 | -2.906 | -5.966 | 3/3 |
| 5 | -2.119 | -2.165 | -3.317 | -5.304 | -3.185 | 3/3 |
| 10 | -7.286 | -7.280 | -7.356 | -8.431 | -1.145 | 3/3 |
| 15 | -12.159 | -11.822 | -11.691 | -12.792 | -0.633 | 3/3 |
| 20 | -17.184 | -15.792 | -15.381 | -17.285 | -0.101 | 3/3 |
| 25 | -22.235 | -19.117 | -18.022 | -22.191 | +0.044 | 0/3 |
| 30 | -27.353 | -20.651 | -19.654 | -26.764 | +0.590 | 0/3 |

V8 beats GD and HS-GS at all eight SNR points. It beats GS at six points and remains behind at 25 dB (0.044 dB) and 30 dB (0.590 dB), consistently across all three seeds.

Against V7, the mean gap to GS improved by approximately 0.024 dB at 25 dB and 0.128 dB at 30 dB. The worst validation gap fell from about +0.831 to +0.832 dB at the selected V7 checkpoints to +0.7018, +0.7031, and +0.7020 dB for the three V8 seeds. Each seed selected its epoch-100 checkpoint. Continued training improved the high-SNR test result modestly, but did not beat GS at those points.

## Files in this local folder

- `v8_test_by_seed.csv`: per-seed results and comparisons with GS, GD, and HS-GS.
- `v8_test_mean_vs_classical.csv`: three-seed mean by SNR.
- `summary_unfolded_v8_p10_continuation_reconstructed.json`: recorded per-seed test gaps and run metadata.

The comparator curves come from the existing local final-four-method results CSV. GS also matches the cached P=10 test curve used by the V6/V7 comparisons. Test outcomes are exploratory if they have informed later model choices; use a fresh test split for a final confirmatory claim.

