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

## V10-V21 follow-up and latest findings

The subsequent versioned scripts and outputs are retained in `HS_GS_V10_*.m` through `HS_GS_V21_*.m` and `results/`. They diagnose high-SNR behavior, projection schedules, rank choice, convergence, and the effect of increasing the receive dimension. The most recent paired fresh-test files are V19 (N=8), V20 (N=16), and V21 (focused N=8 high-SNR test). All report gains as `GS_NMSE_dB - HS_NMSE_dB`; positive values favor HS-GS.

- **V19, N=8, RSR=25 dB:** On 500 paired test samples per P/SNR, HS-GS has positive numerical gain over converged biased GS at most points, but the advantage approaches zero at high SNR. The V19 confidence intervals include zero at P=10/25 and 30 dB and at P=30/15 and 25 dB; this does not establish a robust all-SNR win. The algorithm also outperforms GD in this comparison. See `results/HS_GS_V19_fresh_test_table.csv` and the MAT result/checkpoint/policy files.
- **V20, N=16, RSR=25 dB:** On 500 paired test samples per P/SNR, the frozen rank policy gives positive observed HS-GS gain over GS at all points. The paired 95% intervals are positive at every P=30 point; at P=10, the 25 and 30 dB intervals include zero. This is evidence that the tested Hankel prior is more useful in this N=16 setup, not proof of a universal or all-condition advantage. The run used six workers. See `results/HS_GS_V20_fresh_test_table.csv` and the V20 frozen policy/results.
- **V21, N=8, P=10/30, SNR=25/30 dB:** Validation selected rank 2 at all four points despite the candidate set including rank 4. On the separate 500-sample paired test, all HS-GS versus GS paired 95% intervals include zero; the high-SNR result is effectively parity. HS-GS remains substantially better than GD at these points. See `results/HS_GS_V21_fresh_test_table.csv` and the V21 policy/results.
- `results/HS_GS_V19_V20_gain_over_GS.png` and `.pdf`, generated by `plot_v19_v20_gain_over_gs.py`, visualize paired gains with confidence intervals. They compare different N values, so interpret the panels as separate scenarios rather than a controlled N-only claim unless all other data-generation settings are verified.

**Current interpretation:** For N=8, further tuning has not produced statistically supported HS-GS superiority over GS at every SNR. At high SNR, the rank-2 Hankel projection is usually neutral because the underlying channel is already well estimated by GS/EM-GS; V21 found no measurable benefit from adding rank-4 candidates. N=16 is a promising regime, especially for P=30, but repeatability across independent seeds and a carefully matched N=8/N=16 data-generation setup should be checked before making a broad claim. Keep the estimators, channel dimension, baselines, and RSR explicit in any claim.

**Next research step:** Do not launch another broad rank/iteration sweep. First analyze the paired per-sample errors and selected V20 policies, then replicate the most promising N=16 P=30 condition with a new independent seed and frozen validation policy. Keep N=8 and N=16 conclusions separate. Preserve all old versioned files and never choose settings using a fresh test set.

## DL state

- `colab_unfolded_dl_v2.py` is a 40-layer, eight-scalar-per-layer, noise-conditioned unfolded estimator. It trains separate P=10 and P=30 models in Colab. `colab_unfolded_dl_v3_highsnr.py` is a separate high-SNR experiment; check its Colab outputs before assigning it a result.
- The user previously reported V2 Colab test NMSE curves of approximately P=10 `[-1.341, -2.993, -5.479, -8.620, -12.979, -17.153, -21.624, -24.465]` dB and P=30 `[-3.613, -6.727, -11.164, -15.999, -21.006, -25.876, -30.797, -35.648]` dB over the SNR grid. Those numbers come from a prior Colab output, not a model checkpoint verified in this repository. They do not establish a DL win over GS at every high-SNR point.
- Colab data and output paths in the Python files are machine/session dependent. Check the mounted Drive path and actual model files before rerunning or comparing DL. Keep HS-GS and DL as separate estimators.

## Where to continue

1. For HS-GS, inspect V19-V21 paired error arrays and V20's N=16 policy; if continuing, independently replicate the promising N=16 P=30 setup with a frozen validation-only rule. Do not claim N=8 all-SNR superiority over GS based on these results.
2. For DL, inspect current Colab V3 logs/checkpoints and decide on architecture or training changes using train/validation data. Report frozen models on paired independent test observations.
3. Preserve V2-V9 and their saved outputs. Add future experiments under new versioned filenames with explicit provenance, checkpoint/resume for long runs, and no changes to unrelated folders.

The V5-V9 files and V9 fresh data/results were pushed as commit `91ea71b` on branch `fresh-figure-rebuild`. V10-V21 scripts and their result artifacts are the current follow-up package; check the latest Git commit after this handover is pushed. Pull `fresh-figure-rebuild` on Dell to obtain Git-synced files and handover updates. Local Codex memory on the Asus laptop is separate from Git and may not appear on Dell.

## 2026-10-27 handover: N=8 RSR=12 comparison and DL next phase

### Latest three-estimator script

- `GS_GD_HS_GS_FINAL_N8_RSR12.m` is the current simple MATLAB comparison for biased GS, linearized GD, and hard-rank HS-GS. It uses N=8, K=3, P=10/30, SNR=-5:5:30 dB, RSR=12 dB, 500 Monte Carlo trials per P/SNR, six `parpool` workers, GS/GD caps of 1000 with relative-change tolerance 1e-10, candidate Hankel ranks 1:4 for the 5x4 Hankel matrix, 25 updates for rank selection, 100 final HS-GS updates, and four Cadzow sweeps.
- For each simulated sample, the first 70% of pilot slots are used to estimate the channel and the remaining 30% choose rank by held-out magnitude prediction error. This is an internal per-sample pilot split, not an external validation MAT file. True channel is used only for NMSE scoring. If maximum candidate rank 4 is selected, the final update is EM-GS without a Hankel projection.
- The script now uses `parpool` and per-trial deterministic random substreams. It was not rerun by Codex after the parallel edit; the user ran it and supplied the CSV values below.

### User-reported result from `results/GS_GD_HS_GS_FINAL_N8_RSR12_table.csv`

At P=10, HS-GS beats GS at -5, 0, 5, and 10 dB; GS is better at 15 through 30 dB. HS-GS beats GD through 10 dB, while GD is better at 15 through 30 dB. At P=30, HS-GS beats GS only at -5 and 0 dB, is close but worse at 5 through 30 dB, and beats GD from 5 through 30 dB (but not at -5/0). These are ratio-of-sums NMSE point estimates without confidence intervals, so small gaps should not be presented as statistically significant. This does not establish HS-GS as uniformly superior.

Exact reported rows (GS/GD/HS-GS NMSE dB):
- P=10: -5 (7.077, 7.633, 5.022); 0 (2.847, 3.259, 0.990); 5 (-1.987, -1.205, -3.192); 10 (-6.798, -4.435, -7.154); 15 (-11.795, -6.779, -10.644); 20 (-16.634, -7.760, -14.103); 25 (-20.517, -8.178, -15.992); 30 (-23.638, -8.277, -16.670).
- P=30: -5 (1.847, -0.502, -0.719); 0 (-4.833, -5.089, -5.527); 5 (-10.540, -9.185, -10.397); 10 (-15.770, -12.112, -15.553); 15 (-20.791, -13.739, -20.542); 20 (-25.813, -14.391, -25.619); 25 (-30.914, -14.628, -30.841); 30 (-35.861, -14.718, -35.603).

### Current DL direction

- The user is moving focus to a stronger, standalone DL channel estimator after the HS-GS diagnostics. Keep DL independent from HS-GS.
- Existing Colab files are `colab_unfolded_dl_v2.py` (40-layer noise-aware unfolded estimator; eight learned scalar parameters/layer) and `colab_unfolded_dl_v3_highsnr.py` (same model with high-SNR-weighted training loss). Prior V2 metrics reported by the user are historical and not independently verified against checkpoints here.
- Do not claim the old DL model beats GS/GD at every SNR. Preserve the 40-layer unfolded baseline and create a new versioned trainer for any architectural/objective changes.
- Fair next comparison requires matching RSR, channel/pilot/noise generation, sample splits, and test observations across DL and baselines. The currently available DL dataset is RSR=25 dB whereas the latest three-method script is RSR=12 dB; resolve this explicitly before claiming a direct comparison. Prefer keeping the paper-motivated RSR=12 dB study coherent by generating separate train/validation/test data at RSR=12 dB, without overwriting RSR=25 dB files, then train and compare on held-out test samples. Use validation for model selection only and test only once for the final report.
- Recommended DL experiment: first establish a matched RSR data package and validate MAT/HDF5 complex-array decoding/shapes; then train separate P=10 and P=30 models with a per-SNR-balanced NMSE objective and early stopping on validation; compare to GS/GD/HS-GS on paired test samples, reporting per-SNR NMSE and paired confidence intervals. An unfolded physics-informed baseline is the starting point; compare any residual-CNN add-on as a separate ablation rather than replacing the baseline prematurely.

### Git handover

- New v2 scripts and outputs are intended to be shared with Dell via branch `fresh-figure-rebuild`. Do not include the unrelated pre-existing top-level deletions/modifications in the commit.
