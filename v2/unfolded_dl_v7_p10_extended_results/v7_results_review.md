# V7 continued training result review (P=10)

## Source and reconstruction status

The completed Colab cell output was inspected under the college account. It reports a Tesla T4 run, 100 additional epochs for each of seeds 20261026, 20261027, and 20261028, and output saved to `/content/gdrive/MyDrive/unfolded_dl_v7_p10_extended_results`. The output folder is in the college account's My Drive and is not exposed in the connected Drive account, so the saved JSON and checkpoint files were not downloaded in this review. The files in this local folder are a transparent reconstruction from the exact per-seed test-gap dictionaries printed by Colab, combined with the already-verified local GS test curve. Do not treat them as original Colab artifacts.

## Test result

Gap = DL NMSE (dB) minus GS NMSE (dB); a negative value favors DL. Mean over three seeds:

| SNR (dB) | GS (dB) | GD (dB) | HS-GS (dB) | V7 DL mean (dB) | V7 minus GS (dB) | Seeds beating GS |
|---:|---:|---:|---:|---:|---:|---:|
| -5 | 7.868 | 7.624 | 4.425 | -1.411 | -9.279 | 3/3 |
| 0 | 3.060 | 3.028 | 0.809 | -2.906 | -5.966 | 3/3 |
| 5 | -2.119 | -2.165 | -3.317 | -5.303 | -3.184 | 3/3 |
| 10 | -7.286 | -7.280 | -7.356 | -8.425 | -1.139 | 3/3 |
| 15 | -12.159 | -11.822 | -11.691 | -12.800 | -0.641 | 3/3 |
| 20 | -17.184 | -15.792 | -15.381 | -17.286 | -0.102 | 3/3 |
| 25 | -22.235 | -19.117 | -18.022 | -22.167 | +0.068 | 0/3 |
| 30 | -27.353 | -20.651 | -19.654 | -26.635 | +0.718 | 0/3 |

V7 beats GD and HS-GS at all eight tested SNRs. It beats GS at six of eight SNRs for all three seeds, but still loses to GS at 25 and 30 dB. The 25 dB loss is small (about 0.068 dB); the 30 dB loss remains material (about 0.718 dB).

Compared with the local V6 three-seed test summary, V7 improves the mean DL-minus-GS gap by about 0.023 dB at 25 dB and 0.209 dB at 30 dB. This is progress, but does not meet the target of beating GS at every SNR.

## Interpretation and caveats

- Training loss/validation selection improved steadily through the full 100-epoch continuation, with the worst per-SNR validation gap ending near +0.831 to +0.832 dB. The 30 dB region remains the bottleneck.
- The three seed outputs are very close, indicating low run-to-run variation for this continuation setup; this is not evidence of generalization across new data splits.
- The outputs are test-set results. If these same test results influenced architecture or hyperparameter choices, use them as exploratory evidence, not as a fresh final evaluation.
- Classical curves are taken from the existing local final-four-method CSV, while GS is also cross-checked against the cached test JSON. The local v6 review records that the GS curve matched the MATLAB output to six decimals.
- Exact selected checkpoints and the original Colab summary JSON remain in the college Drive output folder and were not pulled into this workspace in this pass.
