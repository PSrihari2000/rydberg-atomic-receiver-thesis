# N=8 V3 classical estimator comparison

This is the first V3 benchmark to run because it matches the N=8, K=3, P=10/30 scale used by the 1D comparison in the channel-estimation reference. It uses the already generated AoA U(-90,90 deg), RSR=25 dB V3 data. N=16 is a separate extension and must not be combined with these results as though antenna count were the only changed setting.

## Methods

- GS: magnitude-only biased Gerchberg-Saxton baseline.
- GD: the reference paper's one-dimensional linearized-gradient method. The paper's rank-projection PGD is a 2D method and is not added as a 1D comparator.
- EM-GS: local expectation-maximization variant.
- Soft HS-GS: EM-GS with Hankel-domain shrinkage. The multiplier is selected using validation data only.

All methods receive the same saved observations, pilots, reference signal, and noise variance for each sample. The settings in these scripts, including the GD initialization, convergence tolerance, update cap, and Hankel split, are explicit local choices and should be reported alongside the results rather than attributed to the reference paper.

## Run order

Run the validation-only policy selection first:

```matlab
cd('D:\MTech_Thesis_Results\v3\scripts'); DL_V3_N8_SELECT_HSGS_VALIDATION
```

Review `v3/results/N8_CLASSICAL_COMPARISON_RSR25DB/N8_HSGS_validation_selection.csv`. The script uses 250 validation samples per SNR to select among the predeclared Hankel shrinkage multipliers and a separate 250 per SNR to confirm the selected setting. It does not load training or test files.

Only after reviewing and freezing that policy, run the reserved test evaluation once:

```matlab
DL_V3_N8_FROZEN_CLASSICAL_TEST
```

The test script loads the frozen validation policy and reserved test files only. It writes resumable checkpoints and a per-SNR NMSE/gain/paired-bootstrap interval table. Do not change settings based on these test results and rerun the same test as a tuning loop.

Both scripts refuse to overwrite existing result files. Inspect/remove or archive an existing output folder manually before intentionally starting a new, separately documented experiment.
