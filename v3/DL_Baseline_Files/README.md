# V3 DL baseline bundle

This folder collects copies of the V3 N=8 DL baseline scripts and their MATLAB helper files. The original files under `v3/scripts` are unchanged.

## Main comparison scripts

- `DL_V3_N8_FRESH_TEST_HATIM_HARD_DL_COMPARISON.m`: latest fresh-test comparison; GS, GD, EM-GS, pilot-split Hatim-style HS-GS, validation-frozen hard-rank HS-GS, and the GS-initialized 16-stage DL model.
- `DL_V3_N8_FINAL_FIVE_METHOD_TEST_COMPARISON.m`: earlier five-method comparison. It loads the earlier adaptive-gate 16-stage model, not the later GS-initialized model.

## DL model training chain

- `DL_V3_PI_DU_RACE_16STAGE_ADAPTIVE_GATE_TRAIN_PARFOR.m`: creates the first 16-stage adaptive-gate model from the saved 8-stage warm start.
- `DL_V3_PI_DU_RACE_16STAGE_FULLDATA_TRAIN_PARFOR.m`: fine-tunes that model using the full training set and saves the checkpoint used by the GS-initialized trainer.
- `DL_V3_PI_DU_RACE_16STAGE_GS_INIT_TRAIN_PARFOR.m`: trains the selected GS-initialized 16-stage model.
- `DL_V3_PI_DU_RACE_16STAGE_GS_INIT_CONFIRMATION_VALIDATION.m`: independent validation check for that model.
- `V3_BesselRatioFunction.m` and `V3_CLASSICAL_ESTIMATOR.m`: helper dependencies used by the trainers.
- `DL_V3_N8_SELECT_HSGS_VALIDATION.m`: creates the validation policy/settings required by the GS-initialized training script.

## Data generation

- `DL_V3_GENERATE_DATASET_RSR25DB.m` creates the original train, validation, and reserved test data.
- `DL_V3_GENERATE_FRESH_TEST_ONLY_V2_RSR25DB.m` creates a separate V2 test-only dataset.

The scripts calculate the V3 root as the parent of this bundle folder. They therefore use the existing shared folders `v3/data` and `v3/results`; those large MAT files are not duplicated here. Existing datasets, policies, and model checkpoints must be present at the paths expected by each script. Do not rerun a generator over an existing dataset or rerun a comparison whose output folder already exists without first choosing a new output folder.

This is a collected source bundle, not a single monolithic MATLAB file. No MATLAB job was run while preparing it.
