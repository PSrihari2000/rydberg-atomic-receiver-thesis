# V5 validation-only DL improvement track

This isolated track leaves the frozen V4 baseline and its already-used test results unchanged. It continues the selected V4 P=10 and P=30 16-stage models using newly generated independent training and validation sets. N=8, K=3, RSR=25 dB, the architecture, pilot counts, and SNR grid stay fixed. The goal is to test whether more independent training diversity and a lower-rate, longer validation-controlled fine-tune improve NMSE; it does not target an artificially larger gap over classical methods.

## Files

- `V5_generate_train_validation_data.m` creates only train and validation files under `Case_Studies/Baseline_V5/data/N8_K3_RSR25dB`. It creates 5,000 samples per SNR for training and 500 per SNR for validation, with fresh seeds. It refuses to overwrite any existing split. No test split is generated.
- `V5_PI_DU_RACE_train_parfor.m` warm-starts from the frozen V4 P=10 and P=30 models. It trains each pilot-count model independently and saves checkpoints, histories, selected models, per-SNR validation metrics, and same-sample V4 versus V5 validation comparisons under this folder's `results` directory.
- The classical estimator and differentiable Bessel-ratio helper are local copies, so the training script dependencies are contained here.

## MATLAB commands

Run these one at a time in MATLAB:

```matlab
cd('D:\MTech_Thesis_Results\v3\Case_Studies\Baseline_V5')
run('V5_generate_train_validation_data.m')
run('V5_PI_DU_RACE_train_parfor.m')
```

The trainer uses up to two parallel workers for P=10 and P=30. It can take substantial time. The script refuses to reuse an existing result folder. Checkpoints are saved for inspection, but this script does not automatically resume from them; an interrupted run must be restarted with a new output-folder name.

In `validation_V4_vs_V5_P*.csv`, positive `V5GainOverV4_dB` means V5 has lower NMSE on that SNR's fresh validation samples. Decide using all SNR rows for both P values, plus training stability and validation consistency. Only after freezing the choice should a separate untouched V5 test split be generated and scored. Never use the V4 test results to select or tune this run.

For a quicker screening run, use `V5_PI_DU_RACE_train_parfor_screening.m` after data generation. It keeps the V4 warm start and architecture, but limits training to 500 samples per epoch for at most 20 epochs, uses 32 fresh validation samples per SNR for selection, and prints GS-cache progress. Its outputs go to a separate `V5_PI_DU_RACE_16STAGE_SCREENING` folder. Treat this as an initial validation screen; only a promising candidate should be followed by a larger validation confirmation and then a new untouched test.
