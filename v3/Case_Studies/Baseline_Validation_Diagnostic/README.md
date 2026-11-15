# V4 stage and gate validation diagnosis

This read-only MATLAB script diagnoses the frozen N=8, K=3, RSR=25 dB V4 model using the fresh V5 validation data. It does not train a model, alter a checkpoint, or load a test file.

The V4 baseline uses SNR `-5:5:30` dB. This diagnostic intentionally follows that grid; the separate antenna and user-count case studies use a different SNR grid.

## Validation samples

The V5 screening script first reserves one deterministic set of 32 validation samples per SNR, then selects a second deterministic set of 32 fresh samples per SNR for checkpoint selection. The original 16-sample diagnostic output was generated before its selector correctly removed the first reserved set. Treat that output as exploratory; do not use it as the disjoint confirmation.

The `V4_stage_gate_validation_confirmation.m` script excludes both 32-sample sets and all sample indices listed in the earlier diagnostic CSV, then evaluates 64 additional samples per SNR. The confirmation script saves the exact sample indices and checks that they do not overlap the excluded sets. It remains validation-only.

## Run in MATLAB

```matlab
cd('D:\MTech_Thesis_Results\v3\Case_Studies\Baseline_Validation_Diagnostic')
run('V4_stage_gate_validation_diagnostic.m')
```

For the disjoint 64-sample confirmation, run:

```matlab
run('V4_stage_gate_validation_confirmation.m')
```

The script compares the iterative biased-GS initialization, EM-GS, validation-frozen hard-rank HS-GS, and V4 on the same validation samples. It records the V4 stage outputs, EM-GS and Hankel candidate NMSE at each stage, the three gate weights, paired sample NMSE differences, and estimator convergence/update counts. Classical iterative estimators use the V4 comparison's 5,000-update cap and `1e-10` relative-update tolerance.

The initial diagnostic outputs are under `results/V4_STAGE_GATE_VALIDATION/`. The confirmation outputs are under `results/V4_STAGE_GATE_VALIDATION_CONFIRMATION/`:

- `V4_validation_paired_sample_diagnostic.csv`: paired validation NMSE and convergence information per sample.
- `V4_validation_condition_summary.csv`: mean NMSE by P and SNR, with paired V4 gains over EM-GS and hard-rank HS-GS.
- `V4_validation_stage_gate_summary.csv`: stage 0 GS initialization and stages 1–16, candidate branch NMSE, and mean gate weights.
- `V4_validation_stage_gate_diagnostic.mat`: MATLAB tables and diagnostic configuration.

If any classical estimator reaches the update cap, report that condition as nonconverged. The diagnostic does not increase update limits or tune estimator settings.
