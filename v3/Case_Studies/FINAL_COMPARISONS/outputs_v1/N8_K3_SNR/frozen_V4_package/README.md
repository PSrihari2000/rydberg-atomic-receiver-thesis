# Frozen V3 N=8 V4 baseline package

This folder is the canonical, self-contained snapshot for the current N=8, K=3, RSR=25 dB five-method comparison. It groups the selected DL models, hard-rank policy, exact V4 test inputs, MATLAB comparison script, validation-selection evidence, and final V4 results. The original V3 scripts and result folders remain untouched.

## Frozen DL models

- **P=10:** 16-stage spectrum-adaptive PI-DU-RACE model, copied from `v3/results/N8_PI_DU_RACE_16STAGE_SPECTRUM_ADAPTIVE_HANKEL_RSR25DB/selected_model_P10.mat`.
- **P=30:** 16-stage spectrum-adaptive PI-DU-RACE continuation model, copied from `v3/results/N8_PI_DU_RACE_P30_SPECTRUM_CONTINUATION_BALANCED_RSR25DB/selected_model_P30.mat`.
- The validation-by-SNR tables and training histories used to document the selections are in `validation_selection/`.

The P=10 and P=30 models are separate, frozen estimators. They are not a dimension-general model and must not be assumed to work for other N or K.

## Package contents

- `DL_V3_N8_FROZEN_V4_COMPARISON.m`: runnable comparison script with all five estimator implementations in one MATLAB script. Its paths point inside this package.
- `models/P10/` and `models/P30/`: selected inference model files.
- `policy/`: validation-frozen hard-rank policy used by the V4 comparison.
- `test_data/`: exact independent V4 test files, 500 samples per SNR for each pilot count.
- `validation_selection/`: training histories and validation tables for both selected DL models.
- `evidence/`: final V4 NMSE CSV, paired confidence-interval CSV, MAT summary, PNG, and FIG.
- No custom MATLAB helper `.m` file is required by the packaged comparison script; its estimator code is embedded. The script uses MATLAB built-ins.

## Frozen comparison result

The V4 output has 16 P/SNR conditions and compares GS, GD, EM-GS, validation-frozen hard-rank HS-GS, and DL. The DL point estimate is lower than all four classical NMSE estimates at every condition. The paired pointwise 95% CI lower bounds are positive at all 16 conditions for each baseline. Under simultaneous 95% intervals across all 64 DL-versus-baseline contrasts, positive gains remain for 14/16 GS, 16/16 GD, 14/16 EM-GS, and 12/16 hard-rank HS-GS contrasts. Therefore, report universal pointwise gains, but do not claim simultaneous significance at every condition.

The V4 test set has already been used for this result. It is retained here for provenance and exact reproduction only; do not use it for further model or policy tuning. For the N and K sweeps, generate separate datasets and freeze their settings using training/validation data before final testing.

## Reproduce the saved V4 comparison

In MATLAB, run:

```matlab
run('D:\MTech_Thesis_Results\v3\baselines\N8_FROZEN_V4_BASELINE\DL_V3_N8_FROZEN_V4_COMPARISON.m')
```

The script writes a new `reproduction_output` folder and stops if that folder already exists. Reproduction re-evaluates the already-used V4 test set; it does not authorize tuning on it.
