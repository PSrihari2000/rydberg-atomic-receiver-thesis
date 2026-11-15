# PI-DU-RACE V4 vs. nonlearned soft-rank HS-GS: validation plan

## Purpose and limits

This is a validation-only diagnostic of the frozen current V4 checkpoints against a nonlearned soft-rank HS-GS baseline. It does not modify or resave either V4 checkpoint, and does not load any test file. Because the historical V4 test split has already been inspected, these results cannot be presented as a new confirmatory test result. A final confirmatory comparison requires a new test split generated only after the method and threshold policy are frozen.

## What the repository currently implements

- The current V4 comparison script is test-only and includes GS, GD, EM-GS, hard-rank HS-GS, and V4 PI-DU-RACE. It does not include soft-rank HS-GS.
- The archived six-method comparison includes soft-rank HS-GS, but its DL checkpoint has 13 parameters per stage rather than the current V4 15, and it uses a linearized DL initializer while soft HS-GS starts from a spectral initializer. Its reported comparison therefore does not answer the current V4 question fairly.
- The V4 trainer builds a biased-GS initial estimate and passes that estimate into the 16-stage model. The new comparison computes that estimate once per observation and supplies the exact same matrix to both estimators.
- The nonlearned comparator uses a scalar soft threshold, scaled by the local noise/pilot factor, held fixed across its 16 updates. Its multiplier is selected separately for each P/SNR cell on validation data.

## Prespecified comparison protocol

1. Settings: N=8, K=3, RSR=25 dB; separate P=10 and P=30 models; the original V4 SNR grid `-5:5:30` dB; Hankel parameter `p_H=3`.
2. Inputs: for each validation observation, both methods use the same recorded channel truth, pilot matrix, reference matrix, magnitude observation, and noise variance.
3. Initialization: compute biased GS once using the V4 cap of 5,000 updates and relative-change tolerance `1e-10`. Both methods receive that same estimate.
4. Updates: run exactly 16 stages/updates for each method. PI-DU-RACE executes the frozen V4 stage; soft HS-GS executes one EM-GS update followed by fixed-threshold Hankel singular-value soft shrinkage at every update.
5. Validation partition: reconstruct and exclude both deterministic 64-per-SNR subsets reserved by the V4 warm-start and checkpoint-selection rules. Split the remaining samples by SNR into a threshold-tuning subset and a disjoint paired diagnostic subset using a fixed seed. This reduces leakage from tuning the soft policy and avoids rescoring the samples used for V4 checkpoint selection.
6. Threshold selection: evaluate multipliers `0:0.1:2` on the tuning subset; select the multiplier minimizing the pooled linear NMSE ratio for each P/SNR cell. Freeze this table before scoring the diagnostic subset.
7. NMSE and uncertainty: report per-SNR ratio-of-sums NMSE in dB for each method. Compute paired 95% percentile bootstrap intervals for `soft-HS-GS NMSE - V4 NMSE` by resampling matched observations within each P/SNR cell. Positive gain favors V4.
8. Runtime: warm up both code paths, then time GS initialization, each 16-stage forward pass, and the total per-observation cost. Count the same measured GS initialization in each method's total. Run the benchmark serially and alternate method order across samples; report mean seconds/sample and the speed ratio. These are local MATLAB/hardware timings, not platform-independent complexity claims.

## Files added

- `compare_v4_soft_hsgs_validation.m`: self-contained MATLAB validation comparison and report generation. It writes only to `results/PI_DU_RACE_VS_SOFT_HSGS_VALIDATION_V1/`; it refuses to overwrite an existing output path.
- This plan.

Expected outputs are the frozen soft-threshold policy CSV, per-SNR NMSE and paired-CI CSV, runtime CSV, paired validation-sample CSV, MATLAB MAT file, and separate P=10/P=30 figures.

## Run and review sequence

No MATLAB job has been run by Codex. After reviewing the script and this plan, run from the repository root in MATLAB:

```matlab
run('Case_Studies/Baseline_Validation_Diagnostic/compare_v4_soft_hsgs_validation.m')
```

Review output metadata, validation partition sizes, selected multipliers, convergence status of the GS initialization, per-SNR paired intervals, and runtime before deciding whether a new untouched test is warranted. This validation diagnostic alone must not be described as confirmatory test evidence.
