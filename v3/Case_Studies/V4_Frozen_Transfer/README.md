# Frozen V4 transfer sweeps

`V4_frozen_classical_sweeps.m` compares the original, frozen N=8/K=3 V4 checkpoint against GS, GD, EM-GS, and validation-frozen hard-rank HS-GS on fresh antenna-count and user-count test data.

## Scope

- Antenna sweep: N = {8, 16, 24, 32}, K = 3.
- User sweep: N = 8, K = 1:5 at P=10 and K = 1:7 at P=30.
- Both: RSR = 25 dB, P = {10, 30}, SNR = {-10, 0, 10, 20, 30} dB.
- P=10 and P=30 use their respective original V4 checkpoints. No weights are retrained or updated for the sweeps.
- The script uses the already validation-frozen antenna and user hard-rank policies, read-only. It does not tune any method on the newly generated test data.
- All five methods score the same channel, pilot, reference, and noise realization in each cell.
- Per-condition NMSE is pooled: sum of squared channel error divided by sum of true-channel energy. The SNR aggregate is the arithmetic mean of the five per-SNR pooled linear NMSE values, converted to dB afterward.

## MATLAB workflow

Set `workflow_phase` near the top of the script to one phase at a time:

1. `GENERATE_TEST` creates a new independent test-only data release under this folder's `data/` directory. It refuses to overwrite an existing release.
2. `SCORE_TEST` scores that release with the frozen V4 checkpoints and frozen validation policies. It writes per-condition tables and sample-level score data under `results/`.
3. `PLOT_ONLY` creates, separately for P=10 and P=30, a five-panel per-SNR figure and an SNR-aggregated figure for each sweep.

Run the script once per phase, changing only `workflow_phase` between runs. MATLAB is run by the project owner; this repository handoff does not run MATLAB.

## Interpretation

The sweep is a zero-shot transfer experiment for V4: the checkpoints were trained for N=8, K=3, and are applied unchanged at other listed dimensions. N=8/K=3 is the in-domain reference cell. The other tested dimensions are finite, specified transfer points; they do not establish support for arbitrary N or K. Existing sweep tests have been viewed and are historical; this release uses new test data.
