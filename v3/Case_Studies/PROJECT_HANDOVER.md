# V3 Rydberg Channel Estimation: Case Studies Handover

Updated: 2026-11-09

## Project and experimental rules

- The active work is the MATLAB-only V3 PI-DU-RACE Rydberg channel-estimation thesis.
- The frozen reference is the N=8, K=3, RSR=25 dB V4 model, with separate P=10 and P=30 models and a 16-stage physics-informed unfolded architecture.
- The V4 test has already been used. Treat it as historical evidence only; do not use it to choose or tune a new model.
- Select changes from training and validation data. After freezing a candidate, evaluate it on a new untouched test split.
- Keep all new case-study code and reports in `Case_Studies`. The user runs MATLAB, so do not run MATLAB jobs.
- Keep P=10 and P=30 results separate in figures and analysis. Do not describe a larger DL-vs-classical gap as success unless the DL estimate improves under a fair, same-observation comparison.

## Completed case studies

### Antenna-count classical sweep

- Settings: N={8,16,24,32}, K=3, RSR=25 dB, P={10,30}, SNR={-10,0,10,20,30} dB.
- Classical estimators were evaluated on the generated test splits. Results and per-P summary figures/tables are under `Antenna_sweep/results/ANTENNA_COUNT_CLASSICAL/test_evaluation/`.
- The validation-frozen hard-rank HS-GS policy and all-ranks convergence diagnostic are saved alongside the other sweep outputs.
- The plots show hard-rank HS-GS improving as N grows; GS, GD, and EM-GS change much less. This is an observed result for this simulator and setup, not a universal scaling law.

### User-count classical sweep

- The recommended sweep uses N=8, RSR=25 dB, SNR={-10,0,10,20,30} dB, K=1..5 for P=10, and K=1..7 for P=30.
- Validation-frozen rank policies, test results, individual-SNR plots, and SNR-averaged P=10/P=30 figures are under `Users_sweep/results/USER_COUNT_RECOMMENDED_SWEEP/`.
- K=5 showed an unusual GD excursion in the original plot. Validation-only K-load, convergence, identifiability, matched-sample, and GD checkpoint diagnostics were run; an independent test replication showed a similar pattern. The cause was not conclusively isolated. Do not attribute it to a particular seed or claim it is resolved.

### Shared-K PI-DU-RACE model

- Separate shared-K models for P=10 and P=30 were initialized from the matching V4 model, trained on training/validation splits, then evaluated on the user-count test splits.
- Training outputs and test tables, per-sample paired errors, figures, and saved scoring checkpoints are under `Users_sweep/results/USER_COUNT_SHARED_PI_DU_RACE_K_SWEEP_FROM_V4/`.
- The sweep demonstrates that the learned model does not automatically retain the N=8,K=3 baseline behavior as K changes. Treat its test plots as a completed case study, not as permission to tune on those test results.

## Baseline improvement attempts

### V5 continuation

- V5 warm-started from V4 and kept the 16-stage architecture while fine-tuning on fresh training/validation data. It did not load or score a reserved test split.
- Its screening validation deltas were very small: roughly 0.001–0.039 dB by SNR for P=10 and near zero for P=30. This is not evidence of a meaningful gain; do not promote V5 or report a new test claim from it.
- Scripts, local helper copies, validation outputs, and training data are under `Baseline_V5/`.

### Deleted V6 measurement-consistency attempt

- The user deleted the V6 measurement-consistency experiment. Its last reported validation summary was slightly worse than V4 (about 0.005 dB for P=10 and 0.010 dB for P=30 on the reported SNR-averaged metric). Its files are no longer part of the active project.
- Do not restore or treat that candidate as an accepted model without a new, deliberate validation study.

## Recommended next step

Before changing depth or adding another branch, create a read-only, validation-only diagnostic for the frozen V4 model. For each P and SNR, compare its output with EM-GS and the frozen hard-rank HS-GS policy, record paired per-sample NMSE differences, inspect stagewise NMSE and gate weights, and check whether the initialization or either update branch is responsible for weak gains. Then test one isolated training change on validation while keeping the existing 16-stage structure. Do not optimize the reported gap itself; optimize a declared channel-estimation metric and require per-condition validation reporting. Only after freezing a candidate should a new untouched test be generated and scored.

## Git / data note

- This handover and the case-study scripts and result artifacts are intended for GitHub.
- The generated raw `.mat` datasets under `Case_Studies/*/data/` are intentionally kept local for this push; together they occupy about 652 MB. Recreate them with the included generator scripts when needed.
- The V4 baseline, its historical test, and unrelated workspace changes are not included in the case-study commit.
