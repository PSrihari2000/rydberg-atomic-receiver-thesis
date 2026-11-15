# PI-DU-RACE vs. soft-rank HS-GS confirmatory test

This is a new, independent test release for the frozen N=8, K=3, RSR=25 dB comparison. It does not reuse the previously inspected V4 test split or the validation diagnostic subset.

## MATLAB run order

From the repository root in MATLAB, run:

```matlab
run('Case_Studies/Baseline_Validation_Diagnostic/generate_confirmatory_test_six_methods_v1.m')
run('Case_Studies/Baseline_Validation_Diagnostic/evaluate_six_methods_confirmatory_test_v1.m')
```

The generator writes P=10 and P=30 test files under `Case_Studies/data/PI_DU_RACE_SIX_METHOD_CONFIRMATORY_TEST_V1/`. It uses 500 samples per SNR and a new seed base (2026122500, then +P). The evaluator verifies the release metadata before scoring. Both scripts refuse to overwrite an existing output location.

## Frozen comparison protocol

- N=8, K=3, RSR=25 dB, P={10,30}, SNR=-5:5:30 dB.
- Methods: GS, GD, EM-GS, validation-frozen hard-rank HS-GS, validation-frozen soft-rank HS-GS, and frozen V4 PI-DU-RACE.
- V4 and soft-rank HS-GS use one identical biased-GS initialization per observation and exactly 16 updates each. The V4 checkpoint, hard-rank policy, and soft-threshold policy are read-only.
- NMSE is ratio-of-summed-error-energy to summed-channel-energy per P/SNR condition. Paired bootstrap intervals compare each method with V4; positive `MethodGain_vs_V4_dB` favors that method.
- Classical iteration cap is 20,000 with relative-change tolerance 1e-10. The common V4/soft initialization cap is 5,000 with the same tolerance.
- Paired runtime is benchmarked serially on 10 observations per SNR and includes the shared GS initialization. Forward passes are repeated three times per observation with alternating order. Runtime is illustrative for this MATLAB/hardware setup, not a platform-independent complexity result.

## Outputs

The evaluator writes CSV tables, a MAT results file, and separate P=10/P=30 figures under `Case_Studies/Baseline_Validation_Diagnostic/results/PI_DU_RACE_SIX_METHOD_CONFIRMATORY_TEST_V1/`. It does not train, tune, or select any model or policy.

The user should inspect the generated test metadata and figures after running MATLAB. No MATLAB execution was performed while preparing these files.
