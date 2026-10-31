# Rydberg Channel Estimation - V3

This folder is the workspace for the next experiment package. The existing `References` directory is retained. V2 datasets, checkpoints, and outputs remain separate. A small set of unmodified V2 MATLAB source snapshots is kept under `scripts/v2_reference_templates`; those files are references and are not ready to run as V3 scripts. See `docs/V2_REUSE_MAP.md` before reusing them.

## Reference and model choices

- Primary channel and measurement model: Xu et al., *Channel Estimation for Rydberg Atomic Receivers* (`References/Channel Estimation Paper.pdf`).
- AoA distribution: use incident angles uniformly distributed over `[-90 deg, 90 deg]`, as listed in Cui et al., *Towards Atomic MIMO Receivers* (`References/Towards_Atomic_MIMO_Receivers.pdf`). Xu's 1D channel phase uses the cosine of the AoA. Implement the angle in radians in MATLAB and keep the convention consistent with that equation.
- The Hankel paper is a conceptual reference only. Do not import its simulation parameters or treat its claims as verified for this project.
- Other settings (array size, path counts, pilot counts, SNR, RSR, polarization, and noise) must be stated explicitly in the V3 generator and results.

## Data and comparison rules

- Generate independent training, validation, and test splits under one versioned simulator.
- For any comparison split, all estimators use the exact same channel, pilots, reference, and noise realization for each sample.
- Train DL on training data; select models and HS-GS settings on validation data only.
- Freeze all model and algorithm settings before opening the independent test split.
- Preserve V2 datasets and outputs; do not mix them with V3 results.

## Initial scenario to carry forward

Unless deliberately changed and recorded in a new configuration, begin with the established RSR=25 dB scenario: `N=8`, `K=3`, `P=[10 30]`, and `SNR=-5:5:30 dB`. Treat `N=16` as a separately generated scenario with its own correctly dimensioned data.

## Status

The N=8 V3 dataset is present under `data/N8_K3_RSR25dB_AoA_Uminus90to90/`. The N=16 classical benchmark scripts are prepared; see `docs/N16_1D_CLASSICAL_BENCHMARK.md` for the run order. No N=16 data has been generated and no N=16 MATLAB job has been run. The copied V2 source snapshots have not been adapted or executed.
