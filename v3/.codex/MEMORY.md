# V3 Rydberg Channel-Estimation Project Memory

Updated: 2026-11-10

## Project context

- MATLAB-only M.Tech thesis work on channel estimation for Rydberg atomic MIMO receivers.
- Active baseline: frozen V4 PI-DU-RACE, `N=8`, `K=3`, `RSR=25 dB`, separate `P=10` and `P=30` models, 16 unrolled stages, 15 learned scalar parameters per stage (240 per model).
- Each V4 stage combines a biased-GS/EM-GS update, per-user Hankel spectral shrinkage, and a learned softmax blend of previous, EM-GS, and Hankel estimates.
- Classical estimators: GS, GD, EM-GS, and validation-frozen hard-rank HS-GS. Keep hard-rank HS-GS as the classical method; soft Hankel shrinkage belongs to the learned model.

## Settled workflow rules

- User runs MATLAB jobs. Do not run MATLAB.
- Keep new case-study scripts, reports, data, and results under `Case_Studies`.
- Preserve user changes. Do not commit or push unless explicitly requested.
- Keep `P=10` and `P=30` models, result tables, and figures separate.
- Select model/policy changes on training and validation only. After freezing, use a new untouched test split.
- V4 baseline test and previously reported antenna/user sweep tests have already been used. Treat them as frozen historical evidence; do not tune against them.
- Compare methods on the same observations and channel truth with a declared NMSE convention. Do not optimize merely for a larger DL-versus-classical gap.
- Report actual dimensions and model support. Shared-N covers only `N={8,16,24,32}` in its trained setup; shared-K covers the explicit trained K grid, not arbitrary dimensions.

## Completed case studies and interpretation

- Antenna classical sweep: `N={8,16,24,32}`, `K=3`, `RSR=25 dB`, `P={10,30}`, `SNR={-10,0,10,20,30} dB`. Hard-rank HS-GS improved as N increased in this simulator; do not generalize beyond these settings.
- User classical sweep: `N=8`, `RSR=25 dB`, same SNR grid; `K=1..5` at `P=10` and `K=1..7` at `P=30`. An unusual GD excursion near `K=5` was diagnosed but not conclusively explained.
- Shared-K and shared-N DL models have been trained and evaluated on their corresponding sweeps. Their test results are completed evidence, not model-selection data.
- Paired confidence-interval analysis for shared-N re-scored 500 samples per condition on the already-used shared-N test split. This provides uncertainty estimates for that split, not independent test confirmation.
- V5 continuation had only tiny validation changes and is not promoted. Deleted V6/V7 experiments should not be restored without a deliberate new validation study.

## Novelty and closest related work

- Xu et al., “Channel Estimation for Rydberg Atomic Receivers,” *IEEE Wireless Communications Letters*, 2025, DOI `10.1109/LWC.2025.3584040`: rank-constrained PGD for Rydberg channel estimation, including 1D/2D arrays. Consider PGD as a primary classical comparator after matching signal model and settings.
- Xiao et al., “Channel Estimation for Rydberg Atomic Quantum Receivers: Unrolled Phase Retrieval From Holographic Snapshots,” *IEEE Signal Processing Letters*, 2026, DOI `10.1109/LSP.2026.3685170`: URformer unrolls EM-GS and has a trainable Bessel-filter replacement, learned gate, and channel Transformer residual correction. Thus EM-GS unrolling or a learned gate alone is not a defensible novelty claim.
- Candidate distinction to investigate: per-user Hankel spectral-shrinkage conditioned on noise and relative singular spectrum, integrated with the previous/EM-GS/Hankel three-way gate. Candidate classical contribution: hard-rank Hankel refinement within biased-GS/EM-GS. These are not yet verified first-ever claims; conduct a broader literature review and fair algorithm comparison.
- Compatibility audit: URformer's scalar-gain magnitude equation `c*abs(H*s+b+w)` with complex Gaussian noise matches the local `abs(G*S+B+W)` observation after gain calibration. Channel/path model, steering (`sin` versus local `cos`), and paper operating settings differ, so published NMSE is not directly comparable. A local same-split method comparison is feasible.
- Xu et al. use a linearized magnitude objective; their 1D method reduces to GD, while rank-projection PGD applies to the 2D setting. Local GD is structurally aligned but needs gradient/step-size normalization checked before claiming exact reproduction.
- The current source-and-literature comparison is in `Case_Studies/NOVELTY_AND_BASELINE_COMPATIBILITY.md`.

## Recommended next work

1. Implement a validation-only comparison of V4, EM-GS, hard-rank HS-GS, Xu-style 1D GD, and URformer on common local observations; first verify GD gradient normalization and URformer gain calibration.
2. Keep paper-parameter reproductions separate from common-local-data comparisons; do not compare their published NMSE values directly with local curves.
3. Run validation ablations that isolate the Hankel branch, learned shrinkage, and learned gate. Report per-condition NMSE and computation cost.
4. Freeze the candidate using validation. Only then generate/use a new untouched test split for final evaluation.

## Repository note

The detailed historical case-study status is in `Case_Studies/PROJECT_HANDOVER.md`. Generated raw datasets may be large and should not be pushed without a specific request. No MATLAB jobs were run while creating this memory file.
