# N=16 one-dimensional classical benchmark

## Why this comparison is relevant

Xu et al. define a 1D magnitude-only Rydberg channel model and compare biased GS with their gradient-descent estimator. Their PGD rank-projection step is for the 2D-array extension; for 1D the paper states that only gradient descent is applied. Thus an N=16 linear array should compare GS and the Xu-form 1D GD update, not add a separate 2D-PGD curve. We also include EM-GS and the V2-developed soft HS-GS variant.

Xu's 1D simulation reports N=8, K=3, random path counts from 3 to 7, and pilot cases including P=10 and P=30. V3's N=8 scenario follows these core dimensions/path choices, with the selected Cui-style AoA prior and local RSR/noise implementation recorded separately. The N=16 experiment is a new array-size scenario, not an exact reproduction of Xu's plotted 1D setup.

## N=16 experiment

- One-dimensional ULA: N=16, K=3, P=[10 30], SNR=-5:5:30 dB, RSR=25 dB.
- AoA is Uniform(-90 deg, 90 deg); the channel uses the Xu cosine spatial phase.
- Generate independent N=16 validation and reserved test datasets only. Training data is unnecessary for this classical comparison and will be generated separately only if DL is later selected for N=16.
- Choose soft-HS-GS multipliers on half of validation samples and require confirmation on the other half. If the confirmation interval does not support a positive HS-GS gain over EM-GS, freeze the EM-GS fallback at that P/SNR point.
- Test GS, Xu-form 1D GD, EM-GS, and the frozen soft-HS-GS policy on the same test observations. Report paired NMSE gains, confidence intervals, convergence rates, iteration counts, and runtimes.
- All iterative methods use the same relative-change tolerance and maximum update cap. The GD initialization and step-size are local implementation choices and must be reported as such.

## MATLAB run order

Run these in order. The test script requires the validation-frozen policy file.

```matlab
cd('D:\MTech_Thesis_Results\v3\scripts'); DL_V3_GENERATE_N16_VALTEST_RSR25DB
```

```matlab
DL_V3_N16_SELECT_HSGS_VALIDATION
```

```matlab
DL_V3_N16_FROZEN_CLASSICAL_TEST
```

The test evaluator does not load N=16 training or validation observations. It loads the frozen validation policy first, then the reserved test files. Do not alter the policy after viewing the test results.

## Publication interpretation

Beating GS and Xu's 1D GD on the frozen test split would support a performance claim; it does not establish novelty by itself. The contribution must be a distinct, justified HS-GS or DL method, and the DL component must be differentiated from existing URformer work. Do not claim an exact Xu reproduction unless all simulator, angle, RSR, SNR, initialization, and stopping choices match their reported setup.
