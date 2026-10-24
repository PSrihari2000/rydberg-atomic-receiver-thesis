# HS-GS V9 findings (RSR 25 dB)

V9 selected settings on the existing validation data, froze the rule, then generated and evaluated a new paired test set. Each point has 500 channel samples. The test compared HS-GS with biased GS and linearized GD on the same observations using ratio-of-sums NMSE.

The frozen validation rule requires a nonzero Hankel setting to improve aggregate validation NMSE by at least 0.1 dB and to improve at least four of five disjoint validation groups. Otherwise it uses zero Cadzow sweeps. This threshold and group rule are local experiment choices, not paper settings.

| SNR (dB) | P=10 HS-GS gain over GS (dB) | P=30 HS-GS gain over GS (dB) |
| ---: | ---: | ---: |
| -5 | 4.284 | 1.693 |
| 0 | 3.198 | 0.620 |
| 5 | 1.868 | 0.002 |
| 10 | 0.181 | 0.001 |
| 15 | 0.117 | -0.001 |
| 20 | 0.174 | -0.001 |
| 25 | -0.225 | approximately 0 |
| 30 | approximately 0 | approximately 0 |

Positive gain means lower HS-GS NMSE. At P=10, 25 dB, the paired 95% bootstrap interval for HS-GS gain over GS is [-0.417, -0.038] dB. HS-GS beats GD numerically at every tested point, but it does **not** beat GS at every SNR.

The frozen rule applies one Cadzow sweep at P=10 for -5, 0, and 5 dB and at P=30 for -5 and 0 dB. All other points use zero sweeps. Consequently, high-SNR differences from biased GS do not demonstrate a Hankel benefit; the zero-sweep HS-GS path still uses an EM update. In particular, its update count and stopping rule differ from the GS baseline. Further high-SNR tuning belongs on validation data; the V9 fresh test has already been used for final assessment.

Reproduce from `HS_GS_V9_VALIDATE_FREEZE.m`, `HS_GS_V9_GENERATE_FRESH_TEST.m`, and `HS_GS_V9_FRESH_TEST.m`. Saved settings, per-sample errors, NMSEs, and confidence intervals are in `results/HS_GS_V9_frozen_rule.mat` and `results/HS_GS_V9_fresh_test_results.mat`. The newly generated paired test arrays are in `results/HS_GS_V9_fresh_test_RSR25dB/`.
