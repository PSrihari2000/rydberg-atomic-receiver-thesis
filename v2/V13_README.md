# V13 shared per-antenna MLP pilot

## Goal

Test a direct supervised ANN channel estimator as a separate architecture from the existing 40-layer unfolded estimator. The experiment targets the P=10 high-SNR weakness observed in the V10-V12 validation pilots while also training a separate P=30 model.

## Model

For each receive antenna, the model consumes that antenna's normalized magnitude observations and background field, the known complex pilot matrix, noise/pilot conditioning features, and SNR. A shared residual MLP predicts the real and imaginary parts of the K=3 channel coefficients. It has width 384 and five residual blocks. The MLP is shared across N=8 receive antennas; P=10 and P=30 use separate networks.

Training uses channel labels and an SNR-balanced log-NMSE objective. Validation selects the checkpoint by minimizing the worst per-SNR validation gap relative to the cached biased-GS validation curve. The code checks that cached curve's validation data SHA-256 matches the current validation MAT file. The test split is never loaded.

## Colab setup

1. Upload `DL_V13_shared_antenna_MLP_Colab.ipynb` to Colab and open it.
2. Keep the Drive mount at `/content/gdrive` and run the mount cell.
3. Ensure Drive has the following exact paths:
   - `MyDrive/DL_dataset_RSR25dB/train_P10.mat`
   - `MyDrive/DL_dataset_RSR25dB/validation_P10.mat`
   - `MyDrive/DL_dataset_RSR25dB/train_P30.mat`
   - `MyDrive/DL_dataset_RSR25dB/validation_P30.mat`
   - `MyDrive/unfolded_dl_v6_gs_gap_results/gs_validation_P10.json`
   - `MyDrive/unfolded_dl_v6_gs_gap_results/gs_validation_P30.json`
4. Run the code-writing cell, then the file-check cell. Resolve any reported missing paths before training.
5. Run the training cell. It saves an epoch-resume checkpoint and selected model/summary files under `MyDrive/unfolded_dl_v13_shared_mlp_results`.
6. Review per-SNR validation gaps. Do not describe them as final test performance. Only after architecture/settings are frozen should a separate paired test evaluation be built and run.

## Research caveats

- The source dataset and cached GS reference are RSR=25 dB. Do not compare these directly with the current RSR=12 dB MATLAB HS-GS results as if they shared observations/settings.
- This is a validation pilot, not evidence of generalization or a guaranteed win. It is a new ANN baseline to test whether a direct supervised mapping helps where the unfolded model struggled.
- The scale is computed from training channels only. Separate P models use the matching P-specific training split.