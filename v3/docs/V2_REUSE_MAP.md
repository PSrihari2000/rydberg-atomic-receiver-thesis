# V2 to V3 reuse map

## Recommendation

Carry forward V2's implementation experience, equations, and comparison workflow, but generate all V3 datasets under the corrected AoA distribution and train V3 networks from scratch. Keep V2 intact as the historical experiment package. V2's trained networks and selected HS-GS policies were fitted or selected under a different AoA distribution, so they should not be the headline V3 models/settings.

For V3, regenerate independent training, validation, and reserved test splits. Re-run GS, GD, and HS-GS on the same V3 observations; these classical methods do not require training. Train the DL model using only V3 training data, choose its checkpoint and HS-GS settings using only V3 validation data, then freeze everything before opening V3 test data. Transfer learning from V2 could be a later, separately labeled ablation, not the primary comparison.

## Source references copied into V3

The unmodified source snapshots are in [`scripts/v2_reference_templates`](../scripts/v2_reference_templates/README.md).

| V3 reference copy | What is useful | What must change before any V3 use |
|---|---|---|
| `V2_DATASET_GENERATOR_RSR25dB.m` | Dataset array shapes, simulator bookkeeping, split organization, and measurement-generation flow | AoA sampling; V3 output paths and version/config metadata; ensure independent V3 seeds and do not overwrite V2 files |
| `V2_FRESH_VALTEST_GENERATOR_RSR25dB.m` | Example of separately generated validation and reserved test files with overwrite protection | AoA sampling and V3 folder/configuration; this generates no training split |
| `V2_V25_UNROLLED_EMGS_REFERENCE.m` | Most developed model-driven DL starting point: four unrolled EM-GS layers, learned filtering/gates, channel correction MLP, and MATLAB training/checkpoint pattern | New V3 data and anchors; V3 paths; retrain from fresh initialization; review the architecture/objective after V3 baseline diagnostics |
| `V2_V36_SOFT_HS_GS_REFERENCE.m` | Soft Hankel projection and validation-only multiplier-selection logic | It bundles synthetic sample generation and test evaluation; replace with paired V3 observations and isolate selection from reserved test evaluation |
| `V2_V37_FROZEN_REPLICATION_REFERENCE.m` | Checkpointed, paired evaluation and replication bookkeeping | Its generated AoAs and imported frozen policy are V2-specific; adapt only the reusable evaluation mechanics |
| `V2_FINAL_FOUR_METHOD_COMPARISON_REFERENCE.m` | Paired four-method table/figure and uncertainty-reporting structure | It loads V2 test data, a V2 DL model, and a V2 HS-GS policy; point all V3 evaluation to V3 artifacts only after settings are frozen |

## Do not carry into V3 as current evidence

- V2 train/validation/test `.mat` files, including V23, V25, and other confirmation splits.
- V2 GS-anchor caches, since they depend on the corresponding observations and settings.
- V2 model weights, optimizer/checkpoint state, best epochs, and validation-selected per-SNR policies.
- V2 plots, tables, and test metrics as V3 performance claims.

The V2 scripts remain in place. These V3 copies are reference snapshots and have not been adapted or executed. The next implementation step is a V3 simulator/data generator with the corrected AoA distribution and explicit metadata, followed by V3-only training and validation workflows.
