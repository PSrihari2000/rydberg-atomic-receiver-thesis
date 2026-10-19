# Version-2 development result

This development experiment tested whether the unfolded estimator improved by
using all 2,048 available training channels, rather than the 256 channels used
for version 1. It also used a strong-reference linearized channel estimate as
the initial state and optimized a per-channel normalized NMSE loss. Model
selection used the existing 256-channel validation split. No frozen test files
were loaded.

Training stopped after 12 epochs for both pilot counts because the validation
ratio-of-sums NMSE did not improve for eight consecutive epochs. Early stopping
therefore retained epoch 0.

| Pilots | Training channels | Best epoch | Validation NMSE |
|---:|---:|---:|---:|
| 10 | 2,048 | 0 | -6.655 dB |
| 30 | 2,048 | 0 | -10.200 dB |

The result is effectively unchanged from version 1. The normalized training
objective and mean per-channel validation NMSE improved slightly, but the
ratio-of-sums validation metric used for the final comparison did not. This
shows that increasing the training sample count alone is not sufficient for
the present 30-layer parameterization.

A justified next architecture is a noise-conditioned residual/skip connection
that blends the linearized initialization with the nonlinear unfolded output.
It can retain the unfolded estimator's low-SNR denoising while protecting the
accurate linearized solution at high SNR. This change must be selected on
training and validation data, then evaluated on a newly generated independent
test set after freezing.
