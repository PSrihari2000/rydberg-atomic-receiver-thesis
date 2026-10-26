# V11 continuation results (P=10)

## Provenance and completion

- Retrieved from the personal Google Drive folder `unfolded_dl_v11_continue_v10_p10_results` on 2026-10-26.
- The run completed 100 continuation epochs for each of three seeds (20261026?20261028); the selected checkpoint for each seed is epoch 100.
- V11 continued the V10 pilot-conditioned 40-layer unfolded model. The architecture was unchanged; the continuation learning rate was 1e-6.
- Checkpoint selection used validation DL?GS gaps. The summary marks `test_evaluated: false`; these are validation results, not final held-out test results.
- The downloaded summary, three selected `.pt` model files, and three `.pt` resume files are preserved in this folder.

## Validation comparison

Negative DL?GS means DL has lower NMSE than GS. Positive means DL is worse. Values below average the three seeds; see `v10_v11_validation_comparison.csv` for full precision and seed counts.

| SNR (dB) | V10 mean DL?GS (dB) | V11 mean DL?GS (dB) | V11 seeds beating GS |
|---:|---:|---:|---:|
| -5 | -9.4784 | -9.4785 | 3/3 |
| 0 | -5.6556 | -5.6556 | 3/3 |
| 5 | -3.3453 | -3.3452 | 3/3 |
| 10 | -1.3991 | -1.3988 | 3/3 |
| 15 | -0.6864 | -0.6858 | 3/3 |
| 20 | +0.0425 | +0.0427 | 0/3 |
| 25 | +0.5970 | +0.5972 | 0/3 |
| 30 | +0.6388 | +0.6368 | 0/3 |

## Interpretation

V11 is better than GS on validation from ?5 through 15 dB for all three seeds. It is still worse than GS at 20, 25, and 30 dB for all three seeds. At 20 dB the deficit is small (about 0.043 dB); it grows to about 0.597 dB at 25 dB and 0.637 dB at 30 dB.

Against V10, continuation changes are very small. The mean 30 dB gap improves by about 0.00194 dB, while the 25 dB gap worsens by about 0.00022 dB; other SNR changes are also tiny. So simply running the same model for another 100 epochs at a lower learning rate did not close the high-SNR gap. This points away from ?just train longer? as a sufficient fix and motivates diagnosing the model/objective or the GS baseline and then changing the estimator deliberately. The result alone does not identify which cause is responsible.

This experiment covers P=10 only. It does not establish performance at P=30, on the independent test set, or across new random test channels. Do not present these validation numbers as final generalization performance. The next fair comparison should evaluate the selected model and the matched GS baseline once on the untouched test split, using the same samples and exact NMSE definition; keep that test out of further checkpoint/model selection.
