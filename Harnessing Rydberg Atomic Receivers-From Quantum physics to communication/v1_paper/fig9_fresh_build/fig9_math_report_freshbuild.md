# Fig. 9 math report -- independent rebuild, SER vs SNR

NEW file, does not touch the existing fig9_ser_vs_snr.py/.png, fig9_data.npz,
fig9_lofree_quantum_detector.py/.npz, fig9_math_report.md.

Reuses `fig6_fresh_build/fig6_data.csv`'s real, already-computed SNR values (no new
QuTiP solves) -- Conventional and LO-dressed use the full real distance sweep at
PTx=-10dBm as the SNR axis; LO-dressed uses Practical (not Theoretical) SNR so the
real distortion-driven decline is preserved. SER formulas are standard closed-form
results (Proakis, *Digital Communications*):

    M-PAM SER = 2(1-1/M) * Q(sqrt(6*SNR/(M^2-1)))              (Eq.5.2-46)
    M-QAM SER = 1 - [1 - 2(1-1/sqrt(M))*Q(sqrt(3*SNR/(M-1)))]^2  (Eq.5.2-79)

Q(x) = 0.5*erfc(x/sqrt(2)), SNR is the linear (not dB) average SNR per symbol.

## (a) Conventional -- 8/16/64/256-QAM (log2M=2,4,6,8)

Straight formula application over the real Conventional SNR sweep,
[-103.33,36.67]dB across 281 points.

## (b) LO-dressed -- QAM up to log2M=10 (1024-QAM, matching the paper's own stated order)

Uses Practical LO-dressed's real SNR values, [-73.66,
54.12]dB across 280 points -- includes the
real short-distance region where SNR is degraded by distortion (below Fig.5's linear
range), so SER should show the same non-monotonic character already established for
Practical LO-dressed's SNR curve itself in Fig.6.

## (c) LO-free -- 4/16/64/256-PAM (log2M=2,4,6,8), honest ceiling-gated treatment

LO-free's SNR is NOT a continuous sweepable quantity in this model -- it is a hard
ceiling with no real signal available below it (established multiple independent ways
earlier in this project, including a full Monte-Carlo "quantum detector" that confirmed
a sophisticated receiver cannot extract signal that isn't physically present). The
ceiling value used here, 46.0206dB, is read directly from
fig6_data.csv's real LO-free column (mean over 56 resolved points,
std=0.00e+00dB -- confirms it is genuinely flat, not assumed), matching
the algebraic 1/epsilon_tilde^2 prediction (epsilon_tilde=0.5%) but derived from the
loaded data rather than hardcoded. Modelled here as: below 46.0206dB,
no real detection is possible, so SER = random guessing = (M-1)/M; at/above
46.0206dB, the real PAM SER formula applies.

This produces a near-vertical "wall" at the ceiling, not the paper's own published
smooth SER-vs-SNR curve spanning a wide range. This is a deliberate, honest
consequence of the real physics under the paper's own stated Eq.16/epsilon_tilde=0.5%
model, not a bug -- reported as-is, consistent with this project's standing policy of
not forcing curves to resemble the paper's published shape.

## Parameters/data sources

    Fig.4 threshold (from fig6_data.csv metadata): 5.1250 MHz
    All underlying SNR values: real, from fig6_fresh_build's independently-rebuilt
    Fig.6 pipeline (today's Fig.4/Fig.5 data, not the older frozen fig6 build).
