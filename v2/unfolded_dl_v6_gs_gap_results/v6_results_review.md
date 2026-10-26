# v6 GS gap experiment results

Imported from Google Drive on 2026-10-25: six trained checkpoints, the run summary, and validation/test GS curves for P=10 and P=30.

## Test comparison

| P | Seed | Beats GS | Beats GD | 30 dB DL | 30 dB GS | DL?GS at 30 dB |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20261026 | 6/8 | 8/8 | -26.426 dB | -27.353 dB | +0.927184 dB |
| 10 | 20261027 | 6/8 | 8/8 | -26.426 dB | -27.353 dB | +0.927500 dB |
| 10 | 20261028 | 6/8 | 8/8 | -26.427 dB | -27.353 dB | +0.926420 dB |
| 30 | 20261026 | 7/8 | 8/8 | -35.844 dB | -35.845 dB | +0.000261 dB |
| 30 | 20261027 | 7/8 | 8/8 | -35.844 dB | -35.845 dB | +0.000225 dB |
| 30 | 20261028 | 7/8 | 8/8 | -35.844 dB | -35.845 dB | +0.000250 dB |

## What the run shows

- P=10: every seed beats GD at all eight SNR points and GS at six of eight. At 25 dB, GS leads by about 0.091 dB; at 30 dB, GS leads by about 0.927 dB. The v6 result at 30 dB is about 0.63 dB better than v3, but it still does not beat GS.
- P=30: every seed beats GD at all eight points and GS at seven of eight. At 30 dB, DL and GS differ by only about 0.0002?0.0003 dB, with GS numerically lower. Treat this as a tie at practical precision, not a demonstrated DL win.
- Seed outcomes are nearly identical. The tiny P=30 high-SNR differences have no confidence intervals and do not support a strict all-SNR superiority claim.
- All three P=10 runs selected epoch 50, the training limit; the validation gap was still decreasing in the last five epochs. More training could help, but it needs a fresh validation selection and a genuinely new test set for a final claim.
- For P=30, selected epochs were 34?35; the later worst validation gaps rose before early stopping.
- The independently recomputed GS test NMSE matches the existing MATLAB GS CSV to 6 decimal places for both pilot counts and all SNR points.
- This remains exploratory: earlier model choices used the same test set. Use a newly generated held-out test set before reporting a confirmatory thesis claim.

Full seed/SNR values are in `comparison_v6_test_vs_classical.csv`.
