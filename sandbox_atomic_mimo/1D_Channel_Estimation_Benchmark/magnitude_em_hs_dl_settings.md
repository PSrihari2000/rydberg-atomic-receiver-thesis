# Magnitude-only EM-GS, HS-GS and plain DL benchmark

Entry point: `run_magnitude_em_hs_dl_comparison.m`. MATLAB Deep Learning Toolbox is required. Outputs are written to `magnitude_em_hs_dl_results/`.

This is a Fig.3-style extension using an explicitly specified combination of settings. It is not a literal reproduction of Xu or the Hankel draft. The plot has six curves: three estimators at each of P=10 and P=30. Lower NMSE is better. No vertical error bars are displayed.

## Simulation settings and attribution

| Setting | Value / definition | Source |
|---|---|---|
| Measurement | Z = abs(G*S + B + W), entrywise magnitudes | Xu Eq.(8); nonlinear EM-GS/Hankel model |
| Users, cells | K=3, N=8, 1D ULA | Xu Section V / Fig.3 |
| Pilots | P=10 and 30, iid CN(0,1) | Xu Section V / Fig.3 |
| SNR grid | -5:5:30 dB | Xu Fig.3 |
| Noise | W iid CN(0,sigma2); each real component variance sigma2/2 | Xu Eq.(8) noise convention |
| SNR definition | mean(abs(G*S).^2) / sigma2, averaged across cells and pilots; reference excluded | Aggregate user-field convention borrowed from Hankel; sample-power implementation choice |
| RSR | 12 dB, reference power / average single-user received pilot-field power | Hankel setting; NOT a fixed RSR reported by Xu |
| RSR enforcement | Rescale B once per realization/P; preserve its spatial phase; fixed across SNR | Implementation choice |
| Path counts | L_k uniform integers 3 through 7 | Xu Section V |
| User path gains | Independent CN(0,1), without dividing by sqrt(L_k) | Xu Section V; differs from equal-power normalized Hankel simulations |
| Polarization | 3 Gaussian components with variance 1/3, not unit-normalized | Xu component distribution |
| Spatial polarization | One independent vector per user path, shared across cells; one independent reference-path vector shared across cells | Deliberate structure-compatible adaptation of Xu cell-indexed polarization |
| Reference sharing | Spatially constant within reference path; user and reference vectors are NOT forced to be the same draw | Modeling choice |
| AoAs | Independent broadside beta uniform (-90,90) degrees; axis angle theta=90-beta | Hankel broadside angle range, converted to Xu axis convention; uniform sampling is a local choice because the draft does not explicitly name the distribution |
| Spatial phase | phi=2*pi*(d/lambda)*cos(theta_axis); response exp(-j*(i-1)*phi), i=1:N | Xu phase convention |
| Spacing | d/lambda=0.5 | Hankel half-wavelength ULA |
| Reference | One known plane wave, raw alpha_b CN(0,10), rescaled to RSR12 | Xu Eq.(9) and Section V for raw gain; target scaling from Hankel regime |
| Reference symbols | s_b,p=1 | Implementation choice; distribution unspecified in Xu |
| Unit conversion | Divide all fields/channels by the fixed dipole_y/hbar factor; implement dipole_y/hbar=1 | Numerical choice equivalent to Xu scalar coupling; NMSE, SNR and RSR invariant |
| Final EM/HS budget | 100 EM updates; fixed budget, no convergence guarantee | Hankel final estimate setting |
| HS rank testing | 25 updates per candidate rank | Hankel candidate setting |
| Cadzow | 4 sweeps per user after EVERY EM update | Hankel interleaved projection setting |
| Hankel embedding | p=floor((N-1)/2)=3; matrix 5x4; ranks 1..4 | Balanced Hankel construction |
| HS selection | Random 70/30 pilot split: 7/3 and 21/9; minimum held-out magnitude residual | Hankel rank-selection procedure |
| Final HS estimate | Fresh fit using all pilots at selected rank | Hankel procedure |
| Test count | 500 independent channels, each evaluated at all SNR points and both P cases | User choice |
| NMSE | sum_m ||Ghat_m-G_m||_F^2 / sum_m ||G_m||_F^2, then 10log10 | Monte Carlo ratio of expectations, Xu NMSE definition |
| Pairing | Exact same Z,S,B,sigma2,G for all three test methods | Fair comparison choice |

## Plain deep-learning estimator: implementation choices

A separate fully connected MLP is trained for P=10 and P=30. Input features are vec(Z), real/imaginary vec(S), real/imaginary vec(B), and log10(sigma2). Output is real and imaginary vec(G), with dimension 2*N*K=48. There is no Hankel embedding, SVD, rank input, rank loss, or projection in this network. It may still learn channel correlations implicitly from its training distribution; Hankel unaware means no explicit Hankel mechanism.

| Training setting | Value |
|---|---|
| Training channels per P | 12,000 |
| Validation channels per P | 2,000 |
| Hidden layers | 256 ReLU units, then 128 ReLU units |
| Optimizer | Adam, initial learning rate 0.001 |
| Epoch cap | 50 |
| Mini-batch size | 256 |
| Training SNR | Uniform continuous draw over -5 to 30 dB |
| Training RSR | 12 dB, same generator as testing |
| Training loss | Squared channel regression error, scaled by one training-derived constant |
| Model selection | Lowest validation loss among saved epoch checkpoints |
| Input scaling | Per feature mean/std from training data only |
| Target scaling | One global RMS constant from training labels only |
| Seeds | Training 2026101401+P-case; validation 2026101402+P-case; testing 2026101403 |
| Execution | CPU; toolbox trainNetwork |

The independent DL validation channels select the network checkpoint. The held-out HS validation pilots select rank within a test realization. These are different validation procedures. Test true channels are used only for final scoring, while training true channels are legitimate supervised labels. All methods receive the same known noise variance. Network training time is saved separately from inference time; HS timing includes rank search and final fit.

## Limitations to report

With N=8 and 3..7 paths, the Hankel rank often reaches its maximum 4. HS can then match EM-GS. Enforcing smaller rank can bias the estimate. Three validation pilots at P=10 may yield unstable selection. Fixed 100/25 iteration budgets measure finite-budget performance. The plain MLP is a first baseline and has no guaranteed advantage; training-set size and architecture need an independent validation study before publication. Reference values and noise variance are assumed perfectly known. Training and testing use the same simulator distribution, so this does not establish robustness to hardware/model mismatch. No uncertainty intervals are inferred from the line plot.

## Sources

- Bokai Xu et al., *Channel Estimation for Rydberg Atomic Receivers*: https://arxiv.org/html/2503.08985, Eqs.(7)-(9), Section V, Fig.3.
- Abdullah Haatim, *Hankel-Structured Channel Estimation for Magnitude-Only Rydberg Atomic Receivers: When Does Structure Help?*, local draft: `Channel_Estimation_for_Rydberg_Atomic_Receivers/ThesisProj__Copy__Haatim.pdf`, simulation settings/Table I and interleaved EM-GS/Cadzow rank selection. Checked against the local PDF: model and SNR/RSR definitions on pages 2-3, rank selection on page 3, budgets/RSR in Table I on page 6; this combined experiment is an adaptation.
- Existing implementation used for channel and classical estimator functions: `run_classic_hankel_comparison.m`. The existing file is not modified.

The script reuses saved networks and resumes compatible checkpoints when increasing the test count. Completed trial data are preserved. It replays only their seeded simulator draws to advance the random state, then evaluates the additional trials. The latest requested test count is 500.

## Completed 500-trial result

The first plain MLP did not learn useful channel recovery: its NMSE remains near 0 dB, comparable to a zero-channel estimate. Its architecture/data/training setup therefore needs improvement using independent validation, and this experiment does not establish that deep learning is generally inferior. HS-GS helps at low SNR but can introduce rank-selection bias at high SNR, particularly for P=10. The small N=8, L=3..7 setting leaves little unused Hankel rank. No superiority was forced into the curves.

At 30 dB SNR, P=10: EM-GS -22.803 dB, HS-GS -15.262 dB, MLP 0.005 dB. P=30: EM-GS -35.899 dB, HS-GS -35.820 dB, MLP 0.006 dB. These values measure the declared fixed 100/25 iteration budgets, not guaranteed convergence. The trained networks were selected with independent validation data before the test run.
