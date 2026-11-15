# Balanced-SNR log-NMSE screening candidate

This is a separate validation-only candidate. It leaves the frozen V4 models, the V5 files, and all prior test results unchanged.

## Fixed model and changed choice

- N=8, K=3, RSR=25 dB, P=10 and P=30, and SNR=-5:5:30 dB.
- Each P uses its own V4-warm-started 16-stage PI-DU-RACE model. The architecture, 240 trainable parameters, optimizer, learning rate, batch size, and screening budget match the V5 screening trainer.
- The only modeling change is the loss: minimize the mean per-sample `log(NMSE)` instead of the arithmetic mean NMSE. With equal training counts at every SNR, this gives each SNR equal representation in the training objective and is equivalent to minimizing mean per-sample NMSE in dB. The frozen classical baselines are unchanged, so lower model dB NMSE increases the measured gain over any fixed baseline on the same samples.

The objective is a screening choice, not a guarantee that the model will improve. Report per-SNR validation NMSE against V4 and the classical estimators; do not accept the candidate based on one pooled number.

## MATLAB run order

Run these commands separately in MATLAB:

```matlab
cd('D:\MTech_Thesis_Results\v3\Case_Studies\Baseline_V6_BalancedSNR')
run('V6_generate_train_validation_data.m')
run('V6_PI_DU_RACE_balanced_log_nmse_train.m')
```

The generator makes independent train and validation splits only. The trainer uses a 500-sample-per-epoch screening budget and selects checkpoints from 32 fresh validation samples per SNR. It does not load any test file. Output is written to `results/V6_PI_DU_RACE_BALANCED_LOG_NMSE_SCREENING`; scripts refuse to overwrite existing data or result folders.

The selected candidate still requires a confirmation on unused samples from this fresh validation split before it is frozen. The selected sample indices are saved in each model file. Only then should a new untouched test split be generated and scored. Never use the previously used V4 test to select this candidate.

If the trainer is interrupted after a checkpoint was saved, do not rerun the full trainer into the existing output folder. Run `V6_finalize_saved_checkpoint.m` to score and export the best checkpoint from each pilot count on the same validation-selection indices.

The two copied helper files in this folder (`V3_BesselRatioFunction.m` and `V3_CLASSICAL_ESTIMATOR.m`) keep the MATLAB run self-contained. They are support code, not modified V4 baseline files.
