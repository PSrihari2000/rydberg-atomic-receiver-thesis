# Final Rydberg channel-estimation comparison package

This folder is the central index and MATLAB packaging script for the three completed comparisons. `outputs_v1/` now contains copies of the comparison results plus the corresponding datasets, frozen models/checkpoints, validation policies, and source MATLAB scripts. The package is about 0.55 GB. The original files remain in their source folders unchanged.

## Contents created by the MATLAB script

The existing `outputs_v1/` package is already assembled. To build a fresh release from the source folders, run this from MATLAB after choosing a new output-folder name in the script (it intentionally refuses to overwrite an existing release):

```matlab
run('D:\MTech_Thesis_Results\v3\Case_Studies\FINAL_COMPARISONS\assemble_final_comparison_package.m')
```

The script reads saved CSV summaries and copies related assets. It does not rerun estimators, train, select settings, or score samples. The package contains:

1. `N8_K3_SNR/`: frozen V4 local test comparison against GS, GD, EM-GS, hard-rank HS-GS, and PI-DU-RACE DL, with its paired confidence-interval CSV and figure. This local benchmark uses SNR `-5:5:30` dB, `N=8`, `K=3`, `RSR=25` dB, and separate `P=10/30` trained checkpoints. It is the project's local benchmark using the paper-motivated setup; it is **not** a reproduction of the paper's published curves or an exact claim that every simulator parameter matches the paper.
2. `Antenna_count/`: for each `P` and each SNR in `[-10, 0, 10, 20, 30]` dB, NMSE versus `N={8,16,24,32}` for GS, GD, EM-GS, hard-rank HS-GS, and the shared-N DL model. It also creates a separate-per-P SNR-averaged summary, computed by averaging linear NMSE over SNR and converting to dB afterward.
3. `User_count/`: for each `P` and each SNR in `[-10, 0, 10, 20, 30]` dB, NMSE versus K for the same four classical estimators and the shared-K DL model. The sweep is `K=1:5` for `P=10` and `K=1:7` for `P=30`. It also creates separate-per-P SNR-averaged summaries using the same linear-domain convention.

Each case-study folder includes `data/`, `model_training_validation.../`, `hard_rank_validation_policy/`, source test-evaluation files, and archived `.m` scripts. The antenna package separates the training data release from its fresh validation/test release. The V4 package is copied as a complete, self-contained baseline folder.

The two sweep DL curves are from **different shared-dimension models**, each with separate P=10/P=30 checkpoints. They are not the frozen N=8/K=3 V4 checkpoint evaluated at unsupported dimensions. The `V4 weights (N=8)` or `V4 warm start` comparator is excluded from the sweep plots so each sweep has exactly five plotted methods; it is not one of the four classical estimators.

## Source evidence

The package includes raw `.mat` split files, selected model and checkpoint `.mat` files, CSV/PNG/FIG result evidence, policies, and MATLAB source copies. The archived case-study scripts are for reference; their canonical runnable locations remain `Case_Studies/Antenna_sweep/` and `Case_Studies/Users_sweep/` because those scripts use paths relative to their original locations.

All test results in this package are historical, already-used test evidence. This assembly is for reporting and does not authorize tuning against those tests. The P=10 and P=30 figures remain separate.
