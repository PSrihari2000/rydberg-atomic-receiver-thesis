# V3 initial experiment configuration

## Fixed first scenario

- Array/users: `N=8`, `K=3`.
- Pilot counts: `P=10` and `P=30`; these are separate datasets.
- SNR points: `-5:5:30 dB`.
- Reference-to-signal ratio: `RSR=25 dB` under the generator's documented average single-user received-power definition.
- Array spacing: `d/lambda=0.5` (local implementation choice).
- Per-user path count: discrete uniform on `{3,4,5,6,7}` (retained from the V2 generator).
- Incident AoAs for both channel paths and the reference: continuous uniform on `[-90 deg, 90 deg]`.
- Spatial phase: `phi = 2*pi*(d/lambda)*cos(theta)`, with `theta` converted to radians.
- Measurement: `Y = abs(G*S + B + W)`.

## Splits

The generator creates each split with a distinct fixed random seed, separate from the known V2 V23 validation/test seed bases, and cycles evenly over the eight SNR points. The pilot count is added to the split seed to keep P-specific generation deterministic.

| Split | Seed base (P is added) |
|---|---:|
| Train | `2026110101` |
| Validation | `2026110201` |
| Reserved test | `2026110301` |

| Split | Samples per SNR | Total per P |
|---|---:|---:|
| Train | 2500 | 20000 |
| Validation | 500 | 4000 |
| Reserved test | 500 | 4000 |

Output files are written to `v3/data/N8_K3_RSR25dB_AoA_Uminus90to90/` as `{train,validation,test}_P{10,30}.mat`. The generator stops if any destination exists, so remove nothing automatically; inspect existing files before choosing a new output version.

## Status and provenance

`scripts/DL_V3_GENERATE_DATASET_RSR25DB.m` implements this configuration. It has not been run in MATLAB. Counts and most simulator equations follow the V2 generator; the AoA distribution and versioned V3 output location are updated. The angle range is the chosen Cui-style incident-angle assumption; the cosine spatial phase follows the Xu 1D model. This is a new V3 simulation dataset and should not be described as a continuation of the V2 training data.
