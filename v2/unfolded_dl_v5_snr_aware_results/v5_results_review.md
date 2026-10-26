# v5 SNR-aware unfolded estimator: result review

Source: Google Drive `unfolded_dl_v5_snr_aware_results`, imported 2026-10-24. Six model `.pt` files and the original summary JSON are in this folder.

Comparator files: `../unfolded_dl_v3_highsnr_results/summary_unfolded_v2.json`, `../unfolded_dl_v4_pure_results/summary_unfolded_v4_pure.json`, and `../../GS-GS-HS-DL/final_four_method_results_RSR25dB/final_four_method_results.csv`. Lower NMSE dB is better.

The classical comparison script loads saved `test_P*.mat` files, and v5 points to the Drive dataset folder of the same name. File-level identity of the Drive and local datasets was not independently checked.

| P | v5 variant | Beats GS | Beats GD | Beats HS-GS | 30 dB NMSE | GS at 30 dB |
|---:|---|---:|---:|---:|---:|---:|
| 10 | initializer_only | 5/8 | 6/8 | 8/8 | -20.405 | -27.353 |
| 10 | snr_conditioned_only | 5/8 | 7/8 | 8/8 | -23.802 | -27.353 |
| 10 | initializer_plus_snr_conditioning | 5/8 | 7/8 | 8/8 | -23.424 | -27.353 |
| 30 | initializer_only | 3/8 | 8/8 | 5/8 | -33.668 | -35.845 |
| 30 | snr_conditioned_only | 3/8 | 8/8 | 8/8 | -35.839 | -35.845 |
| 30 | initializer_plus_snr_conditioning | 3/8 | 8/8 | 8/8 | -35.839 | -35.845 |

## Interpretation

- No v5 variant beats GS at every SNR for either P.
- The SNR-conditioned-only variant is strongest overall among v5 variants; the initializer offers no consistent gain.
- For P=10 at 30 dB, SNR-conditioned-only v5 is -23.802 dB versus GS at -27.353 dB (3.551 dB gap). v3 remains stronger at -25.793 dB.
- For P=30 at 30 dB, SNR-conditioned-only v5 is -35.839 dB versus GS at -35.845 dB (0.006 dB gap). This difference is too small to interpret without uncertainty estimates.
- Checkpoint selection minimizes the largest per-SNR validation NMSE in dB. For these runs the worst SNR is -5 dB, so selection emphasizes the low-SNR endpoint. The training loss remains high-SNR weighted.
- Report each trained model separately. Picking the best v5 variant at each SNR using these test results would leak test information into model selection.
