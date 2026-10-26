# V9 high-SNR fine-tuning audit

## Artifacts and run integrity

The original V9 summary JSON, three final model checkpoints, and three resumable checkpoints were recovered from the timestamped Drive-download archive and copied into this folder. The corresponding original V8 summary and checkpoints are in `../unfolded_dl_v8_continue_p10_results/`.

The V9 JSON contains three distinct seeds. Each selected epoch 100, and each seed names its matching V8 checkpoint as its starting model. The V8 and V9 model tensors have distinct SHA-256 hashes across seeds; comparing each matched V8/V9 pair shows 318 of 320 learned scalars changed. Thus the nearly identical seed metrics are not duplicate model files: these independently different parameter vectors converged to very similar aggregate NMSE.

The local P=10 validation and test files each match the SHA-256 fingerprint recorded in the cached GS JSON. The notebook's training code uses three different shuffle seeds. The V9 summary does not record a training-data hash or runtime/device metadata, so exact training-file identity and the CPU runtime are not independently attested by that JSON. The user reported the run used CPU.

The V9 code selects checkpoints on validation and only loads the test split after selection. All three seeds selected epoch 100. The test split has nevertheless been examined across multiple model versions during development; treat these results as exploratory, not as a fresh confirmatory test.

The validation and test split SHA-256 fingerprints match the cached GS curves exactly. The training split has 20,000 examples, but the V9 summary did not save its fingerprint, so the exact training-file version used in that run cannot be proven from the saved artifacts.

## Reproduction check and per-example validation audit

`audit_v8_v9_validation.py` reads the learned 40x8 float32 tensors from the PyTorch checkpoint archives and reproduces the unfolded estimator and the V6 GS update using NumPy. On 500 validation examples at each of 25 and 30 dB, its aggregate DL-minus-GS gaps agree with the saved V8/V9 validation summaries to within 0.000013 dB. The validation data fingerprint also matches the GS cache.

At 25 dB, V9 is better than GS on about 51% of individual validation examples by per-example NMSE, yet its aggregate validation gap is about +0.613 dB. At 30 dB it is better on 35% of examples, with an aggregate gap about +0.672 dB. The aggregate NMSE weights error by target channel energy, so per-example win rate and aggregate NMSE can differ substantially.

The 30 dB deficit is larger among examples with more poorly conditioned pilot matrices: by condition-number quartile, the V9 aggregate gap rises from about +0.29 dB in the lowest quartile to +0.87 dB in the highest. The channel-norm relationship is not monotonic at 25 dB. This points to heterogeneous difficult examples; it does not establish that architecture alone is the cause. One of 500 validation examples at 25 dB reached the GS 1,000-iteration cap; none at 30 dB did.

Detailed artifacts:

- `v8_v9_validation_highsnr_per_example.csv`: per-example, per-seed DL and GS NMSE, gap, channel norm, pilot conditioning, and GS iteration count.
- `v8_v9_validation_highsnr_summary.csv`: per-seed validation summary and agreement with saved JSON metrics.
- `v8_v9_checkpoint_integrity.csv`: full checkpoint and learned-tensor hashes, changed-parameter counts, and seed/source mapping.
- `p10_dataset_fingerprints.csv`: local train/validation/test hashes and validation/test GS-cache checks.
- `../unfolded_dl_v8_continue_p10_results/summary_unfolded_v8_p10_continuation.json` and `summary_unfolded_v9_highsnr_p10.json`: original run summaries.

## Test comparison

Negative DL-minus-GS favors DL; positive favors GS. Means below are recomputed from the original per-seed JSON summaries.

| SNR (dB) | V8 mean gap (dB) | V9 mean gap (dB) | V9 minus V8 (dB) | V9 seeds beating GS |
|---:|---:|---:|---:|---:|
| -5 | -9.281 | -9.286 | -0.006 | 3/3 |
| 0 | -5.966 | -5.952 | +0.014 | 3/3 |
| 5 | -3.185 | -3.129 | +0.056 | 3/3 |
| 10 | -1.145 | -1.077 | +0.067 | 3/3 |
| 15 | -0.633 | -0.584 | +0.049 | 3/3 |
| 20 | -0.101 | -0.083 | +0.018 | 3/3 |
| 25 | +0.044 | +0.051 | +0.007 | 0/3 |
| 30 | +0.590 | +0.564 | -0.026 | 0/3 |

V9 remains behind GS at 25 and 30 dB. Its 30 dB test gain over V8 is small; at 25 dB it regresses slightly. On validation, V9 worsens the 25 dB gap by roughly 0.256 dB while improving 30 dB by about 0.030 dB. This indicates a tradeoff and a validation/test generalization difference, not a broad high-SNR fix.

## Conclusion

The V9 loss reweighting did not solve the high-SNR GS gap. The strongest next step is to diagnose or target difficult, pilot-conditioned cases using validation only, with a single controlled model change. Keep the existing test split out of future tuning and reserve a fresh test set for the final claim.
