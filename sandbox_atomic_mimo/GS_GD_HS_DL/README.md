# Four-method magnitude-observation experiment

Target curves: biased GS, linearized GD, interleaved HS-GS, direct DL.
EM-GS is an internal HS-GS update, not a separate plotted baseline.

## Current stage: noiseless learning diagnostic

1. Run `export_learning_check.m` in MATLAB. It creates independent train,
   validation, and test channels for N=8, K=3, P=10/30.
2. Preferred first check: run `train_learning_check_matlab.m`. It trains on CPU
   for P=10 and P=30, saves models/history and `matlab_learning_check/summary.csv`,
   and measures training time. No Python dependency is required.
   Alternatively, use Python/PyTorch locally or upload the script and generated
   MAT files to Colab. Run `python train_learning_check.py --pilots 10`, then P=30.
3. Inspect training fit and independent validation NMSE. The tiny MLP is an
   implementation check, not the final proposed DL architecture. Fitting these
   64 training examples does not establish generalization. Test data stays unused.

Current diagnostic choices: 64/32/32 realizations; noiseless observations;
RSR=50 dB relative to aggregate user-field power; Gaussian varying pilots;
existing spatially shared path-polarization channel generator. The reference
scaling and uniform angle sampling are local choices, not literal Xu settings.
Noiseless variance zero is for DL checks only; do not pass it to Rician EM.

Next: address any learning/validation failure, freeze a model, generate a larger
noisy dataset, and evaluate all four methods on identical saved test arrays.
Use SNR=-5:5:30 dB and 500 test trials per point for the proposed main comparison;
final settings must be recorded before running. Train/validation partitions stay
separate by channel realization. GS/GD stopping rules and HS iteration budgets
must be reported. The historical finite-budget tuned figure is preserved.

The final benchmark and final DL architecture are not implemented in these
diagnostic files. Existing results and scripts are unchanged.

## Pilot-conditioned development experiment

Run `export_pilot_conditioned_data.m`, then `train_pilot_conditioned.m`.
This uses 1,024 training channel realizations and 256 independent validation
realizations for each P. All eight cell rows of a channel stay in the same split.
The previous small dataset, previous models, and final test files are preserved.

For each cell, construct D from the real and imaginary parts of the known
reference-phase-weighted pilots. The network inputs are D'*(Y-|B|)/P and the
21 upper-triangle entries of D'*D/P: 27 real features. The output is six real
numbers representing three complex channel coefficients. No true channel or
classical-estimator output enters inference. The Gram matrix is not inverted
in the learned method. Its use of Y-|B| and these statistics incorporates a
strong-reference approximation and discards some nonlinear information.
This variant must be labeled a strong-reference feature-based DL diagnostic,
not a demonstrated general nonlinear-model estimator. Training uses a compact
128/128 network for 60 epochs and selects by validation loss. Final test data
remains unused; results must not be presented as final benchmark performance.

## Noisy, complete-magnitude-input development experiment

Run `export_nonlinear_development.m`, then `train_nonlinear_development.m`.
Two independent datasets per P: 2,048 training channels, 256 validation channels;
balanced SNR levels -5:5:30 dB; RSR 50 dB relative to aggregate user-field power.
This is a local strong-reference setting, not literal Xu reproduction.

The rowwise input retains every magnitude and all known measurement-design
information: Y-|B|, |B|, real/imaginary reference-phase-rotated pilots, and noise
standard deviation. Reference phase rotation is exact for nonzero B under the
circular complex noise model. The original noiseless predicted magnitude is
|rotatedPilots*g + |B||. Auxiliary Gram/backprojection statistics are included
but do not replace the complete observations. No GS/GD/EM/HS prediction is used.
The loss is supervised channel MSE, not a Rician likelihood. This per-cell network
does not exploit spatial correlation across the eight cells; that is a limitation.

Models use 128/128 hidden units, 80 epochs, and best-validation-loss selection.
Validation-by-SNR CSVs are development results (32 channels/point), not final
test curves. Hyperparameter selection uses this validation set, so it must not
be relabeled as an untouched test set. The original held-out test remains unused.

## Exact-magnitude learned unfolding

`train_unfolded_magnitude.m` is an alternative model-based DL architecture,
not another name for the plain MLP. It starts at zero and applies 30 exact
magnitude-residual gradient layers. Each layer learns one positive step factor
and one bounded momentum factor (60 trained scalars total), using supervised
channel MSE and automatic differentiation. It does not take a GS, GD, EM-GS or
HS-GS output as input and uses no Hankel projection. Its gradient follows the
nonlinear amplitude objective, not Xu's linearized GD or the Rician likelihood.

The first run uses 256 development-training channels and 256 independent
validation channels, with all their cell rows kept in their own partition.
Twenty epochs; batch size 256 cell rows; separate models for P=10 and P=30.
Each row's step is normalized by the known pilot spectral norm squared. The
noise variance is not used explicitly by this initial architecture; its learned
schedule is shared over the training SNR range. Per-cell processing also omits
spatial structure. Include the untrained schedule as an ablation so improvements
cannot be attributed entirely to deterministic model-based iterations.

The loop starts from a noiseless field equal to the known reference and uses
observed noisy magnitudes in every layer. The real/imaginary gradient formulas
were independently checked by finite differences. Final tests remain untouched.

## Noise-aware unfolding development

`train_noise_aware_unfolding.m` preserves the previous model and adds known
noise variance to the update schedule and proximal L2 shrinkage. Channel prior
power is estimated from training labels only and frozen. Each layer conditions
step/momentum on log(1 + (sigma2/priorPower)/pilotSpectralNormSquared).
The proximal denominator is
`1 + step * learnedPositiveRidgeWeight * (sigma2/priorPower) / bound`.
This is a local regularized architecture, not a claim of exact Rician EM/MAP.
It has five learned scalars per layer, 150 total. Zero-start inference remains;
the step/momentum parameter schedule is warm-started from the previous trained
model. No per-sample classical channel estimate is supplied.

Training/validation channels and SNRs are unchanged from the 256-channel
unfolding experiment. Use the same validation data for paired development
comparisons and checkpoint selection; final tests are still withheld.

## Frozen four-method test

`run_four_method_test('integration',5)` runs an independent wiring check.
`run_four_method_test('final',500)` runs the paired main test, resuming from
its checkpoint when called again with the same mode and trial count.
Actual G/Y/S/B/noise variance arrays are saved per trial; no test labels enter
any estimator. The DL prior is frozen from training. All models are copied to
`frozen_noise_aware_models` and fingerprinted in `sha256.json` before testing.
Do not tune the architecture against the final test results.

Fixed settings: N=8, K=3, P=10/30, SNR=-5:5:30 dB, RSR=50 dB relative to
aggregate user-field power. GS uses spectral initialization; GD uses CN(0,0.1)
initialization. Both have a 1,000-update cap and relative squared-change stop
1e-10, with convergence not presumed from the cap alone. HS uses 25 updates
for each rank candidate, a 70/30 pilot split, 100 final updates, and four Cadzow
sweeps. DL uses the frozen 30-layer noise-aware schedule. There is no separate
EM-GS curve. All timings are CPU wall times; DL includes feature packing and
HS includes rank selection. Model loading and shared data generation are excluded.
Training cost is reported separately. Curves use ratio-of-sums NMSE.

The experiment is a disclosed Xu-Fig.3-style extension, not literal reproduction.
Original channel geometry/polarization adaptations are inherited from the
standalone sandbox simulator. Final observations use independent trial seeds;
per-trial RNG reset ensures deterministic checkpoint/resume.
# Version-2 all-data development check

`train_noise_aware_unfolding_v2.m` uses all 2,048 training channels per pilot
count, a linearized strong-reference initialization, and a per-channel
normalized NMSE loss. Outputs are written separately to
`noise_aware_v2_development_results`. Both `P = 10` and `P = 30` retained epoch
0 under validation early stopping, so this check did not replace the frozen
version-1 estimator. See the result folder's `RESULTS.md` for the measured
values and interpretation.

