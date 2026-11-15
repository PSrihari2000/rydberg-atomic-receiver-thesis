# Confirmatory shared-K DL experiment (V2)

This is a separate, fresh user-count experiment. It does not modify or reuse the old user-sweep test set or overwrite the existing shared-K model/results.

## Frozen setup

- N=8, RSR=25 dB, SNR=[-10,0,10,20,30] dB.
- P=10: K=1:5; P=30: K=1:7.
- 2,000 training, 500 validation, and 500 test samples per SNR and dimension cell.
- Separate 16-stage models for P=10 and P=30, initialized from the matching frozen V4 weights.
- GS initialization cap=20,000 updates and relative-change tolerance=1e-10 for DL training, validation, and test scoring. The classical sweep uses the same cap/tolerance. Convergence fractions are retained in outputs.
- Fresh split seed bases are 2026120101/2026120201/2026120301 for train/validation/test.

## Run order in MATLAB

1. Run `Users_count_confirmatory_v2_classical.m` with `workflow_phase = 'GENERATE'` (the file is set this way). This creates new independent train/validation/test MAT files under `data/USER_COUNT_CONFIRMATORY_V2_...`. Do not open or inspect the test files.
2. Set that file's `workflow_phase` to `'VALIDATE'` and run it. This selects and freezes hard-rank HS-GS policy from validation only.
3. Run `Users_count_confirmatory_v2_DL_train.m`. It trains/selects the shared-K DL models using only train/validation files and the same 20,000-update GS initialization rule.
4. Once both P-specific selected-model files exist, set the classical script phase to `'TEST'` and run it to score GS, GD, EM-GS, and validation-frozen hard-rank HS-GS on the fresh test split.
5. Run `Users_count_confirmatory_v2_DL_test.m`. It verifies the new split/model metadata, scores V4 warm-start and shared-K DL on that same fresh test, and joins the four classical rows from step 4. It does not select or tune parameters.

Only inspect test figures/tables after step 5 completes. If any test result informs a later change, that test is no longer confirmatory; create another fresh test release after freezing the next candidate.

## Output locations

- Data: `data/USER_COUNT_CONFIRMATORY_V2_P10_K1_2_3_4_5_P30_K1_2_3_4_5_6_7_RSR25_SNR_m10_0_10_20_30/`
- Classical validation policy and test results: `results/USER_COUNT_CONFIRMATORY_V2/`
- DL checkpoints, selected models, validation history, and test results: `results/USER_COUNT_SHARED_PI_DU_RACE_K_SWEEP_CONFIRMATORY_V2/`

The 20,000-update GS initialization can substantially increase training time. Do not lower it after inspecting test results; any runtime-driven alternative must be chosen using train/validation and then tested on a new untouched split.
