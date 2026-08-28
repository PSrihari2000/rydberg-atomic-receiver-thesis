# Fig. 7 — Math Report

Mutual information versus SNR (SNR swept directly, -20 to 40dB, per the paper's own Fig.7
x-axis — not mapped through distance the way Fig.6 was).

## 1. Paper text — extracted directly via `pdftotext`, not memory

Equations (22) and (38) were re-verified character-for-character from
`References/Harnessing Rydberg Atomic Receivers-From Quantum physics to communication.pdf`
(extracted to `References/paper_text.txt`), not recalled from earlier session notes, since the
prior `fig7_reinvestigation/` audit files no longer exist (deleted).

**Eq.(22)** (LO-free MI, nats):
```
I(z;x) = ln[(r+1)/2] - ln(I0(r)) + r·I1(r)/I0(r)
```
with `r ≈ SNR_Ry` — the paper's own explicitly stated approximation ("we approximate the
Rician factor by the SNR"). `I0`, `I1` = modified Bessel functions of the first kind, order
0 and 1.

**Eq.(38)** (LO-dressed MI):
```
I(z_LO;x) = e^(1/SNR) · E1(1/SNR) · ln(2)
```
`E1` = exponential integral function.

**No equation is given anywhere in the paper for "Conventional"'s MI.** The text states only:
*"The LO-free system follows the mutual information formulated in (22), while the LO-dressed
system obeys the result presented in (38)"* — Conventional is not attributed to either.

## 2. Data sources — REUSED, not recomputed

- `fig4_fresh_build/fig4_classification.npz`: real threshold, `5.5000 MHz`.
- `fig5_fresh_build/fig5_data.npz`: real static response, `P0_bar`, `Ω_LO`. `κ` re-fit fresh
  here (same algorithm/data as `fig5_fresh_build`/`fig6_fresh_build`, reproduces the identical
  number: `-9.493466e-07 W/MHz`).
- `A_LO`, `σ²_Ry,LO` (Eq.36/37): re-derived fresh from the same real constants used in
  `fig6_fresh_build`, reproduces the identical numbers (`A_LO=4.478045e-07`,
  `σ²_Ry,LO=2.542319e-14`) — no PRx/distance dependence, since Fig.7's x-axis is SNR directly.

**No new QuTiP solves.**

## 3. Model 1 — Conventional (INFERENCE, not paper-cited)

Standard Shannon capacity, `MI = log2(1+SNR)`. Chosen because it's the universal default when
no other formula is given, and because it is *information-theoretically consistent* with
Eq.(38): Eq.(38) is the ergodic capacity of a Rayleigh-faded channel, which by Jensen's
inequality must sit at or below the unfaded `log2(1+SNR)` at the same average SNR — and our
computed numbers do satisfy that (Section 6). This is a plausibility check, not proof the
paper used exactly this formula.

## 4. Model 2 — LO-free, Eq.(22)

Implemented with `scipy.special.ive` (exponentially-scaled Bessel functions) rather than raw
`iv`/`I0`/`I1`, because `r=SNR_linear` reaches `10^4` at SNR=40dB and `I0(10^4)` overflows
double precision directly; `ive(0,r)=I0(r)·e^-r` stays finite, and `ln(I0(r))` is recovered as
`ln(ive(0,r))+r` without ever forming the overflowing intermediate.

**Gating — a real, non-obvious finding (not a bug):**

- **First attempt**: reuse the same Fig.4/Fig.6 AT-splitting threshold (5.500MHz), inverting
  Eq.(16) to find what `PRx`/`Ω_RF` a given SNR value implies. Result: **this gate is empty
  across the entire -20..40dB window** — because the SNR value at which the implied `Ω_RF`
  first reaches 5.5MHz turns out to sit almost exactly at the same `1/ε̃²=40000` (46.02dB) hard
  ceiling already established in `fig6_fresh_build` (both are driven by the same `ε̃`-limited
  asymptote). So the distance-domain resolvability criterion, moved into SNR-space, tells us
  almost nothing over the paper's own plotted range.
- **Actual gate used**: `MI_nats(r) ≥ 0`. Eq.(22), evaluated directly (ungated), is
  well-behaved and increasing but goes **negative** below roughly `SNR≈-1.2dB` — physically
  impossible for a real mutual information, indicating the `r≈SNR_Ry` approximation (or the
  formula's domain of validity) breaks down there, not a coding error. This matches the
  paper's own description of an "LO-free distortion region" at low SNR (Sec. V-C) far more
  naturally than the AT-splitting gate does.

## 5. Model 3 — Theoretical LO-dressed, Eq.(38)

`scipy.special.exp1` for `E1`; no numerical stability issues in this SNR range (`1/SNR` maxes
out at 100, well within double precision for both `exp` and `exp1`).

Both constants shown, same ambiguity already flagged for this equation earlier in the project:
paper-literal `×ln(2)` and the standard nats→bits `×log2(e)`. `log2(e)=1/ln(2)`, so the two
curves differ by a factor of `~2.08×` at every SNR.

## 6. What this figure DOES NOT do — Practical LO-dressed and Fig.8 DEFERRED (user decision)

**Practical LO-dressed** is not built here. The paper's text ("closely aligned with theoretical
at low SNR... precipitously declines beyond a certain threshold, entering its distortion
region") describes a divergence *at the same SNR value* between Practical and Theoretical —
which cannot come from plugging a different (degraded) real SNR into Eq.(38), since Eq.(38) is
a pure function of SNR and any SNR value fed to it lands on the same parent curve. Reproducing
the paper's described behavior faithfully would require a genuine Monte-Carlo nonlinear-channel
mutual-information estimate (real waveform distortion via `fig6_fresh_build`'s phasor-sum +
real static-curve interpolation + Fourier projection, plus a numerical differential-entropy
estimator for `h(z)`) — a substantially larger, separate task with real design choices, set
aside for later per the user's explicit request (2026-08-20: "let's revisit unfinished plots
later, let's go with whatever we can now").

**Fig.8** (achievable capacity for 8-QAM/8-PAM, saturating at `log2(8)=3` bits/s/Hz) needs a
constellation-constrained-capacity formula that is not given anywhere in this paper's text
either (only SER formulas, cited from an external textbook, are given — for Fig.9, not Fig.8).
Deferred in full for the same reason.

## 7. Actual numeric results — HONEST discrepancies with the paper, not fixed or hidden

Checkpoints (bits/s/Hz):

| SNR | Conventional | LO-free (Eq.22, gated) | Theo. LO-dressed (paper ×ln2) | Theo. LO-dressed (standard ×log2e) |
|---|---|---|---|---|
| -10dB | 0.1375 | NaN (below MI≥0 region) | 0.0635 | 0.1321 |
| 0dB | 1.0000 | 0.3037 | 0.4134 | 0.8603 |
| 10dB | 3.4594 | 4.6855 | 1.3964 | 2.9065 |
| 20dB | 6.6582 | 9.5809 | 2.8270 | 5.8840 |
| 40dB | 13.2879 | 19.5361 | 5.9847 | 12.4564 |

**Discrepancy 1 — magnitude**: our curves top out around 13-20 bits/s/Hz at 40dB; the paper's
Fig.7 y-axis is drawn up to 30 bits/s/Hz with curves visibly using most of that range. Not
force-scaled to match — reported as-is.

**Discrepancy 2 — Theoretical LO-dressed sits BELOW Conventional at every SNR we checked**,
under both constant conventions. This is actually *expected* given Eq.(38)'s own structure
(ergodic Rayleigh-fading capacity is provably ≤ unfaded `log2(1+SNR)` at equal average SNR —
Jensen's inequality), so it is internally consistent with our Conventional formula choice
(Section 3) — but it does not match the visual impression from the paper's own Fig.7, where
"Theoretical LO-dressed" appears to sit at or above "Conventional" for most of the range. Given
that Conventional's own formula isn't paper-cited, this could mean either our Conventional
guess is wrong, or the paper's Eq.(38) constant/derivation differs from what's printed — not
resolved here, reported honestly.

**Discrepancy 3 — LO-free's own AT-splitting-based "distortion region" (Fig.4/6's criterion)
does not transfer into SNR-space** at all over the plotted range (Section 4) — a genuinely new
finding from this figure, not seen in Fig.3-6's distance-domain work.

None of these were adjusted, curve-fit, or tuned toward the paper's numbers — they are the real
output of the equations exactly as printed, applied to this project's already-established real
constants.
