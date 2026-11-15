# Frozen PI-DU-RACE V4 vs nonlearned soft-rank HS-GS

This study compares the unchanged P-specific N=8, K=3 PI-DU-RACE V4 checkpoints against the existing 16-stage EM-GS plus conventional Hankel soft-threshold estimator on the already generated zero-shot antenna and user sweep observations.

## Important interpretation

The exact sweep test releases and V4 test results have already been viewed in this project. Re-scoring those observations is a paired reanalysis, not a new blind or untouched confirmation. Do not use its results to tune either estimator or claim a fresh confirmatory test. A later confirmatory claim needs a new independent test release after all choices are frozen.

## Source inspection

- `../V4_Frozen_Transfer/V4_frozen_classical_sweeps.m`: configuration, data release, scoring path, V4 forward pass, and convergence-based GS initialization used for the frozen sweep.
- `../Baseline_Validation_Diagnostic/compare_v4_soft_hsgs_validation.m`: established soft-rank rule, original threshold scale and candidate grid, validation partitioning, and original N=8,K=3 common-start diagnostic.
- `../../baselines/N8_FROZEN_V4_BASELINE/models/P10/selected_model_P10.mat` and `P30/selected_model_P30.mat`: frozen checkpoints (recorded SHA-256 hashes in `Experiment_logs/source_manifest.txt`).
- Existing sweep releases under `../V4_Frozen_Transfer/data/V4_FROZEN_CLASSICAL_FRESH_TEST_V1`: sample tensors include `Y_data`, `G_data`, `S_data`, `B_data`, `sigma2_data`, and `SNR_dB_data`. The observation `Y_data` carries the realized measurement/noise; explicit `W_data` is not saved, so the same exact observation is the reproducible pairing key.
- Validation releases are the matching antenna sweep validation files and Users Confirmatory V2 validation files. See paths in `MATLAB/run_validation_tuning.m`.

## Method and fairness notes

The established soft estimator performs 16 repeated steps: EM-GS update, singular-value soft threshold `max(sigma-tau,0)` for each per-user Hankel matrix, and anti-diagonal averaging. Its noise-scaled threshold is `tau=lambda*sqrt(sigma2)*norm(S'/(S*S'),'fro')*sqrt(max(N-p_H,p_H+1)/K)`. The old N=8,K=3 comparison tuned `lambda` by P and SNR from `0:0.1:2` and used a shared biased-GS initialization with a 5,000-update cap. The frozen zero-shot sweep instead initialized V4 from the sweep's GS estimate with a 20,000-update cap. This new test scorer follows the sweep GS path for both methods, so its threshold policy is re-tuned on validation using that same start. It uses `p_H=floor((N-1)/2)` as in the sweep.

Three validation-only threshold policies are reported: configuration-specific (strong, configuration-tuned), sweep-wide (one lambda per P within each sweep, selected across all dimensions in that sweep), and N=8,K=3 transfer (one lambda per P/SNR, tuned only at the original training configuration). Only the last is a zero-shot classical threshold transfer outside N=8,K=3. V4 is unchanged in every policy.

## Run order

1. Run `MATLAB/run_validation_tuning.m`.
2. Inspect `Validation_results/soft_rank_selected_thresholds.csv`. The scorer evaluates all three prespecified policies, each frozen using validation only.
3. Run `MATLAB/run_test_scoring.m` for the already-used release, understanding that this is exploratory reanalysis. It requires V4 to reproduce the corrected original per-condition CSV before writing result outputs.
4. Run `MATLAB/run_statistical_analysis_and_figures.m`.

The expected evaluation volume is 20 dimension/P configurations × 5 SNRs × 500 samples = 50,000 saved test observations. Validation selection evaluates 21 threshold multipliers over 30 validation samples per dimension/P/SNR condition (3,000 validation sample-condition instances across the 20 configurations and five SNRs). At 16 soft-refinement stages, this is roughly 3.38 million per-user Hankel stage evaluations for the validation grid, before the final test scoring. The test scorer also recomputes the sweep GS initialization on all 50,000 observations and runs V4 plus three soft-policy variants; actual duration depends on GS convergence and machine speed. Both scripts use up to six process workers.

All scripts are new and self-contained. MATLAB has not been run by the assistant. Output folders are guarded against overwrite. Since the existing test release has already been viewed, the resulting scores cannot be called blind confirmation.
