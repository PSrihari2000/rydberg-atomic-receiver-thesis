# Rydberg Channel Estimation: Novelty and Baseline Compatibility Audit

Audit date: 2026-11-10

This is a source-and-literature audit, not a MATLAB run or a test-data analysis. It records candidate contribution claims and the comparisons still needed before claiming novelty or publication readiness.

## Current implementation, verified from repository sources

| Track | Code-grounded description | What it can support |
|---|---|---|
| Frozen V4 baseline | Separate `P=10` and `P=30` models; `N=8, K=3, RSR=25 dB`; 16 unrolled stages; 15 learned scalar weights per stage (240 per model). Each stage computes an EM-GS update, a per-user Hankel spectral-shrinkage estimate, and a three-candidate softmax blend of previous/EM-GS/Hankel estimates. | The established N=8/K=3 DL reference and its frozen historical test comparison. Do not tune on that test. |
| Hard-rank HS-GS | Repeated EM-GS channel update followed by per-user Hankel matrix SVD, hard truncation to a validation-selected rank, and anti-diagonal averaging. The rank policy is frozen before test scoring. | Candidate classical-algorithm contribution. The contribution must be distinguished from generic low-rank projection and existing Hankel methods. |
| Shared-N DL | One 16-stage, 240-parameter model per pilot count, trained jointly over `N={8,16,24,32}`, `K=3`, `RSR=25 dB`, and `SNR={-10,0,10,20,30} dB`; Hankel dimensions adapt with N. Warm-starts from V4. | Dimension-generalization case study over the four trained N values only; it does not establish operation at arbitrary antenna counts. |
| Shared-K DL | One 16-stage, 240-parameter model per pilot count, trained across `K=1..5` for `P=10` and `K=1..7` for `P=30`, with `N=8`. Warm-starts from V4. | User-load generalization case study over the specified K values only. |

The frozen V4 package reports that its DL point NMSE beats GS, GD, EM-GS, and the frozen hard-rank policy in the original V4 conditions. That test has been used; it is evidence for the reported configuration, not a tuning or model-selection set.

## Closest published work found

| Work | Method reported by the paper | Overlap with this project | Comparison implication |
|---|---|---|---|
| Xu et al., "Channel Estimation for Rydberg Atomic Receivers," *IEEE Wireless Communications Letters*, 2025, [DOI: 10.1109/LWC.2025.3584040](https://doi.org/10.1109/LWC.2025.3584040), [arXiv](https://arxiv.org/abs/2503.08985) | Rank-constrained phase retrieval solved by projected gradient descent (PGD), for 1D and 2D arrays. In the paper's 1D case the projection step is absent, leaving gradient descent (GD); the low-rank projection is relevant to the 2D formulation. | The local exact observation is also a biased magnitude measurement. The local GD uses the paper's first-order/linearized magnitude objective, while the generator supplies exact magnitude observations. | For the current 1D ULA case, compare the local GD implementation with the paper's 1D GD after checking the gradient/step-size normalization. A PGD implementation is the appropriate paper method when evaluating the paper's 2D/rank-constrained setting; do not label an unmodified 1D run as PGD. |
| Xiao et al., "Channel Estimation for Rydberg Atomic Quantum Receivers: Unrolled Phase Retrieval From Holographic Snapshots," *IEEE Signal Processing Letters*, 2026, [DOI: 10.1109/LSP.2026.3685170](https://doi.org/10.1109/LSP.2026.3685170), [arXiv version](https://arxiv.org/abs/2509.12586) | URformer unrolls EM-GS, learns a filter in place of the fixed Bessel-ratio step, uses a learned sigmoid gate, and adds a Transformer residual correction. | Its observation equation is a scalar receiver gain times `|H s + b + w|`, with circular complex Gaussian `w`; this is the same magnitude-measurement form as local `Y=abs(G*S+B+W)` after absorbing/calibrating the scalar gain. | The measurement model is compatible. The channel/path and steering conventions and operating settings differ, so published NMSE values are not directly comparable. A fair method comparison can train/run URformer on the same local splits and observations, with only validation used for choices. |

This was a focused comparison of two close works, not an exhaustive novelty search or legal assessment. The absence of an exact match in these sources does not prove priority for the hard-rank or Hankel-shrinkage details.

## Measurement-model compatibility details

The current data generator uses `Y = abs(G*S + B + W)`, with circular complex Gaussian noise `W`. Its channel is a sum of 3–7 random paths per user with complex Gaussian gains and polarization terms; the ULA phase uses `cos(theta)` and incident angles are uniform on `[-90,90]` degrees. Pilots are iid circular complex Gaussian. SNR is based on aggregate `mean(abs(G*S).^2)` divided by the complex-noise variance. RSR is set as reference power divided by the average single-user received-signal power, not the aggregate multiuser signal power.

URformer's paper measurement is `z = c |H s + b + w|` (notation simplified), with circular complex Gaussian noise. The known scalar receiver gain `c` can be absorbed by consistent calibration/normalization, so "holographic snapshots" do not by themselves make its magnitude observation incompatible with this simulator. However, the paper uses a clustered Saleh-Valenzuela channel, a `sin(theta)` steering convention, and a different numerical operating point (including RSR and pilot settings). To compare algorithms rather than paper-specific setups, preserve the local generated observations and train/evaluate a faithful URformer implementation on the local train/validation/test partitions. Do not compare its published curve numerically against this project's curves.

Xu et al.'s received field is also converted to a magnitude observation and then linearized under a strong reference. Their 1D objective is the linearized magnitude least-squares objective; their 1D method reduces to GD, whereas rank-projection PGD is the 2D extension. The local GD update is structurally aligned with that 1D objective, but exact reproduction still requires checking the factor in the gradient and using matching initialization, step-size, stopping rule, and evaluation settings. The local generator's exact nonlinear magnitude samples are suitable inputs for this approximate GD baseline, provided that limitation is stated.

In short: **URformer is measurement-model compatible but not paper-setting matched; Xu's 1D GD is the direct model-based comparator for this 1D study, while 2D PGD is a separate extension.** Both can be compared fairly on common local observations, but published NMSE values cannot be compared directly because the channel, RSR convention, pilot/SNR settings, and sometimes geometry differ.

## Candidate novelty claims and their status

1. **Hard-rank HS-GS:** plausible candidate algorithmic contribution: combine the biased-GS/EM-GS update with a per-user Hankel low-rank projection and validation-frozen rank selection. To defend it, compare against the paper's PGD, EM-GS, and appropriate Hankel/low-rank alternatives; give an ablation showing what the Hankel step adds, complexity/runtime, and sensitivity to rank policy.
2. **PI-DU-RACE Hankel branch:** a plausible architecture-specific distinction is the learned, noise- and relative-spectrum-conditioned soft singular-value shrinkage on each user’s Hankel matrix, followed by a gate over previous, EM-GS, and Hankel-refined estimates. URformer’s abstract reports a learned Bessel filter, gate, and Transformer residual block; it does not describe this same Hankel-shrinkage/gating formulation. Treat this as a candidate distinction, not a verified first-ever claim, until full-text equations and related Hankel/unfolding literature are compared.
3. **N/K sweeps:** useful empirical evaluation, not standalone algorithmic novelty. State the exact N/K/P/SNR/RSR grids and limitations. The shared models only demonstrate generalization over their trained grid and simulated assumptions.

## Fairness and evidence checks before a paper claim

- Compare methods on identical generated observations and channel truth, with the same measurement model and a declared NMSE convention.
- URformer's scalar-gain magnitude observation is compatible after calibration; implement it faithfully on the same local data splits if feasible. Match its channel/steering and training choices only when reproducing its published setting, not when comparing methods on the local setting.
- For the local 1D setup, include the paper's GD specialization and verify its gradient scaling. Use rank-constrained PGD only for a matching 2D experiment. Do not claim the current GD is a verbatim reproduction until that code-level check is done.
- Keep local common-data comparisons distinct from reproducing paper figures. Never compare published NMSE values directly when channel assumptions, RSR, pilots, or SNR grids differ.
- Ablate the proposed elements separately: EM-GS-only unrolling; no Hankel branch; fixed versus learned shrinkage; no learned gate / fixed blend. Select based on validation.
- Report per-((P,SNR,N/K)) results, confidence intervals from paired samples, computation/runtime, convergence behavior, and multiple independent training seeds where practical.
- Do not use the already-consumed V4, antenna-sweep, or user-sweep test outcomes to make further model choices. Freeze a candidate using training/validation, then evaluate it on a new untouched test split.
- Keep claims specific: simulated channel-estimation performance under the stated Rydberg receiver model is not a hardware demonstration.

## Recommended paper framing (provisional)

Lead with **a Hankel-structured hard-rank refinement for biased phase retrieval in multiuser Rydberg MIMO channel estimation**, with PI-DU-RACE as a learned soft-shrinkage extension. Avoid claiming that the overall idea of learned EM-GS unrolling or a trainable gate is novel. Finalize this framing only after the PGD/URformer compatibility check, ablations, and a fresh frozen test.

## Files audited

- `Case_Studies/PROJECT_HANDOVER.md`
- `baselines/N8_FROZEN_V4_BASELINE/README.md`
- `baselines/N8_FROZEN_V4_BASELINE/DL_V3_N8_FROZEN_V4_COMPARISON.m`
- `Case_Studies/Antenna_sweep/Antenna_count_sweep.m`
- `Case_Studies/Antenna_sweep/Antenna_count_shared_N_generate_data.m`
- `Case_Studies/Antenna_sweep/Antenna_count_shared_N_DL_train.m`
- `Case_Studies/Users_sweep/Users_count_shared_DL_train.m`
- `Case_Studies/Users_sweep/Users_count_recommended_sweep.m`
- `.codex/MEMORY.md`

Repository handover and current scripts were inspected. No MATLAB jobs were run, and no test data were opened.
