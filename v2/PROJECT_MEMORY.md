# Rydberg channel-estimation project handover

Updated 2026-10-24. This file is the Git-synced handover for work on the Asus and Dell computers. Read the current scripts, outputs, and Git status before acting; this snapshot can become stale.

## Aim and comparison rules

- Estimate the channel from magnitude-only observations `Y = abs(G*S + B + W)` for `N=8`, `K=3`, `P=10` and `P=30`, SNR `-5:5:30` dB, and a locally chosen `RSR=25` dB. The numeric RSR is an experiment choice, not a paper-stated value.
- Compare biased GS, linearized GD, a separate Hankel-structured HS-GS estimator, and a separate unfolded DL estimator. Lower ratio-of-sums NMSE is better. The immediate research target is to improve HS-GS and DL against GS/GD across the full SNR grid, but the saved experiments do not yet establish an all-SNR HS-GS win over GS.
- Use matching observations for paired method comparisons. Select parameters using training/validation only; freeze them before an independent final test. Do not tune against the V9 fresh test, which has already been evaluated. Keep older RSR=12/50 dB experiments separate.

## Source and data

- Dataset generator and original train/validation/test splits: `../GS-GS-HS-DL/DL_dataset_RSR25dB/`. It contains files for both pilot counts. The older four-method comparison is in `../GS-GS-HS-DL/final_four_method_results_RSR25dB/`.
- Standalone MATLAB HS-GS work is in this `v2/` folder. V5-V8 scripts/results trace adaptive sweeps, Hankel shape, and validation selection. Their settings and exact outputs must be read from the corresponding scripts and `results/*.mat` files before quoting numbers.
- V9 source: `HS_GS_V9_VALIDATE_FREEZE.m`, `HS_GS_V9_GENERATE_FRESH_TEST.m`, `HS_GS_V9_FRESH_TEST.m`, and `hs_gs_v9_sample.m`. Validation, generation, and test save checkpoints for resume. On the Asus laptop the user ran long MATLAB jobs, including parallel jobs, themselves; avoid starting expensive local simulations without a direct request.

## Verified V9 state

V9 selected settings on 500 validation samples per P/SNR. Its local robustness rule accepts a nonzero Cadzow setting only if it improves aggregate validation NMSE over zero sweeps by at least 0.1 dB and improves at least four of five disjoint validation groups. It then froze the rule, generated a separate new 500-sample-per-point test set, and evaluated HS-GS, biased GS, and linearized GD on paired samples. These facts and numbers were checked against `results/HS_GS_V9_frozen_rule.mat` and `results/HS_GS_V9_fresh_test_results.mat` on 2026-10-24.

| SNR (dB) | HS-GS gain over GS, P=10 (dB) | HS-GS gain over GS, P=30 (dB) |
| ---: | ---: | ---: |
| -5 | 4.284 | 1.693 |
| 0 | 3.198 | 0.620 |
| 5 | 1.868 | 0.002 |
| 10 | 0.181 | 0.001 |
| 15 | 0.117 | -0.001 |
| 20 | 0.174 | -0.001 |
| 25 | -0.225 | approximately 0 |
| 30 | approximately 0 | approximately 0 |

Gain is `GS_NMSE_dB - HS_NMSE_dB`, so positive favors HS-GS. At P=10 and 25 dB, the paired bootstrap 95% interval is `[-0.417, -0.038]` dB: GS wins there. HS-GS beats GD numerically at all 16 points, but does not beat GS at every SNR. The frozen rule uses one Cadzow sweep only at P=10 for -5/0/5 dB and P=30 for -5/0 dB; all other points use zero sweeps. At zero sweeps, HS-GS still performs EM updates, and its update count/stopping rule differs from the biased GS comparator. Thus high-SNR performance cannot be attributed to Hankel projection. See `HS_GS_V9_FINDINGS.md` for the short result summary and `results/HS_GS_V9_fresh_test_results.mat` for exact per-sample errors and confidence intervals.

## DL state

- `colab_unfolded_dl_v2.py` is a 40-layer, eight-scalar-per-layer, noise-conditioned unfolded estimator. It trains separate P=10 and P=30 models in Colab. `colab_unfolded_dl_v3_highsnr.py` is a separate high-SNR experiment; check its Colab outputs before assigning it a result.
- The user previously reported V2 Colab test NMSE curves of approximately P=10 `[-1.341, -2.993, -5.479, -8.620, -12.979, -17.153, -21.624, -24.465]` dB and P=30 `[-3.613, -6.727, -11.164, -15.999, -21.006, -25.876, -30.797, -35.648]` dB over the SNR grid. Those numbers come from a prior Colab output, not a model checkpoint verified in this repository. They do not establish a DL win over GS at every high-SNR point.
- Colab data and output paths in the Python files are machine/session dependent. Check the mounted Drive path and actual model files before rerunning or comparing DL. Keep HS-GS and DL as separate estimators.

## Where to continue

1. For HS-GS, investigate high-SNR update counts and the zero-sweep EM baseline using validation data. Compare a fixed-budget EM baseline with the same initialization and update rule to isolate any true Hankel effect. Freeze a new rule before generating another untouched test set.
2. For DL, inspect current Colab V3 logs/checkpoints and decide on architecture or training changes using train/validation data. Report frozen models on paired independent test observations.
3. Preserve V2-V9 and their saved outputs. Add future experiments under new versioned filenames with explicit provenance, checkpoint/resume for long runs, and no changes to unrelated folders.

The V5-V9 files and V9 fresh data/results were pushed as commit `91ea71b` on branch `fresh-figure-rebuild`. Pull that branch on Dell to obtain this repository state and later handover updates. Local Codex memory on the Asus laptop is separate from Git and may not appear on Dell.
