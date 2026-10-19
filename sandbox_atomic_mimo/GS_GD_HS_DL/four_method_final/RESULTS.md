# Frozen four-method test results

This folder contains the completed independent 500-trial comparison for
`N = 8`, `K = 3`, `P = 10` and `P = 30`, and SNR from `-5 dB` to `30 dB`.
Every method received the same saved magnitude observations in each trial.
The reported NMSE uses a ratio of accumulated squared error to accumulated
true-channel energy across the 500 test channels.

The four reported estimators are Biased GS, Linearized GD, HS-GS, and the
frozen noise-aware unfolded DL estimator. EM-GS is used only inside HS-GS and
is not reported as a separate curve. The DL model was frozen before this test
and the true channel was used only to compute final NMSE.

At `P = 10`, the DL method gives the lowest NMSE from `-5 dB` through `15 dB`.
At `20 dB` and above, Linearized GD and Biased GS give lower NMSE than DL.
HS-GS improves substantially over GS/GD at low SNR, but develops a high-SNR
floor for this `N = 8` configuration.

At `P = 30`, DL gives the lowest NMSE from `-5 dB` through `10 dB`. The four
methods are close around `15--20 dB`; at `25--30 dB`, Biased GS and Linearized
GD give the lowest NMSE.

These results use a `50 dB` reference-to-aggregate-user-field power ratio and
the local shared-path-polarization channel generator. They are a controlled
Fig. 3-style extension, not a literal reproduction of Xu et al.'s Figure 3.
The individual generated observations, checkpoint, CSV table, MATLAB figure,
and PNG figure are retained in this folder for audit and reproduction.
