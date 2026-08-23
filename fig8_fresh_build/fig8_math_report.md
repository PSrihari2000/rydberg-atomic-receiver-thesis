# Fig. 8 — Math Report

Achievable capacity versus SNR: 8-QAM (Conventional, Theoretical/Practical LO-dressed), 8-PAM
(LO-free).

## 1. Paper text — confirmed via `pdftotext`, no formula given

No equation number or formula is given anywhere for this figure. Only the modulation
assignment is stated: "*the LO-dressed and the conventional systems utilize 8-order quadrature
amplitude modulation (8-QAM), while the LO-free systems employ 8-order pulse amplitude
modulation (8-PAM) due to its fixed-amplitude nature*" and "*the theoretical performance limit
for both 8-QAM and 8-PAM is log2 8 = 3 bits/s/Hz*". Fig.9's SER formulas cite Proakis
explicitly; Fig.8 does not cite anything for its capacity computation.

## 2. What's used instead — standard constellation-constrained MI (CCMI)

Standard, textbook result (Ungerboeck/Forney — the same "well-known, not paper-specific"
category as Fig.9's Proakis SER formulas, not fabricated for this project):
```
I(X;Y) = log2(M) - (1/M)·Σᵢ E_n[ log2( Σⱼ exp( -(|xᵢ-xⱼ+n|²-|n|²)/N0 ) ) ]
```
evaluated via Monte Carlo (`n_mc=30000` per SNR point) over a unit-average-energy constellation,
`N0=1/SNR` (Es/N0 convention, matching Fig.9). Verified against known limiting behavior before
use: →0 as SNR→-∞, →`log2(M)` exactly as SNR→+∞ (checked at SNR=60dB, Section 6).

**Constellations** — the paper does not specify exact 8-QAM geometry (8 is not a perfect
square, so it isn't the simple separable-PAM-product layout Fig.9 used for 4/16/64/256-QAM).
Used the standard rectangular 8-QAM (`I∈{-3,-1,1,3}`, `Q∈{-1,1}`, the common/default layout),
and standard equally-spaced 8-PAM (`{-7,-5,...,7}`), both normalized to unit average energy.
**Flagged as an inference** — a different 8-QAM constellation geometry (cross/star) would shift
the numbers somewhat, though the qualitative saturating-sigmoid shape would not change.

## 3. Data sources — REUSED, not recomputed

Same as `fig7/9_fresh_build`: `fig4_fresh_build/fig4_classification.npz` (threshold),
`fig5_fresh_build/fig5_data.npz` (κ re-fit fresh, `P0_bar`, static curve, `Ω_LO`), `A_LO`/
`σ²_Ry,LO` re-derived fresh (matches `fig6/7/9_fresh_build` exactly). **No new QuTiP solves.**

## 4. The four curves — same "club closed-form + real data" mechanism as Fig.9

- **Conventional**: `CCMI_8QAM(SNR_nominal)` directly.
- **Theoretical LO-dressed**: `CCMI_8QAM(SNR_nominal)` — **same formula, same x-axis SNR as
  Conventional.** Unlike Fig.7 (where Eq.(38) explicitly gave LO-dressed a Rayleigh-fading-
  averaged formula distinct from Conventional's plain Shannon capacity), no paper equation
  singles out a different treatment for LO-dressed in this constellation-constrained context —
  so no such distinction was invented here. Result: **these two curves come out numerically
  identical** in this reproduction (Section 6). Reported honestly, not forced apart.
- **Practical LO-dressed**: nominal SNR → invert Eq.(37) for `PRx`/`Ω_RF` → real phasor-sum +
  static-curve interpolation + Fourier projection (identical machinery to `fig6/7/9_fresh_build`)
  → real achieved SNR → `CCMI_8QAM(SNR_real)`. This is what actually differentiates it from
  Theoretical.
- **LO-free**: nominal SNR → invert Eq.(16) → real `Ω_RF` → gated by the real Fig.4 threshold
  (`SNR_real=SNR_nominal` if resolvable, else `SNR_real→-∞` equivalent, i.e. `CCMI→0`) →
  `CCMI_8PAM(SNR_real)`.

## 5. What this figure does NOT do

- No new QuTiP solves.
- Does not import old `fig8` anything or `fig7_reinvestigation/`.
- Does not force Conventional and Theoretical LO-dressed apart artificially, and does not
  force LO-free to show a nonzero operating region — both report the real, direct consequence
  of the paper's own Eq.(16)/(37) plus this project's already-established real threshold/noise
  constants.

## 6. Actual numeric results — HONEST findings, not tuned

CCMI sanity check (Monte Carlo, before use): `CCMI_8QAM(20dB)=3.0000`, `CCMI_8PAM(20dB)=2.9924`
(both correctly saturating at `log2(8)=3.0000`).

Checkpoints (bits/s/Hz):

| SNR | Conventional | Theoretical LO-dressed | Practical LO-dressed | LO-free |
|---|---|---|---|---|
| -10dB | 0.1347 | 0.1347 | 0.1558 | 0.0001 |
| 0dB | 0.8963 | 0.8963 | 0.9985 | 0.0001 |
| 10dB | 2.6839 | 2.6839 | 2.7739 | 0.0001 |
| 20dB+ | 3.0000 | 3.0000 | 3.0000 | 0.0001 |

**Finding 1 — Conventional and Theoretical LO-dressed are numerically identical**
(`max|difference|=0.00e+00`) throughout the sweep. Both trace the same saturating-sigmoid CCMI
curve, reaching `~50%` of the 3-bit ceiling around SNR≈0dB and saturating by ~20dB. This
follows directly from Section 4's reasoning (no paper-given formula distinguishes them for this
particular figure) — reported as a genuine feature of this reproduction, not a bug.

**Finding 2 — Practical LO-dressed is very slightly BETTER than Theoretical**, not worse
(e.g. `0.9985` vs `0.8963` at nominal SNR=0dB — Practical needs about 1dB less nominal SNR to
reach the same capacity). Root cause: the real static curve's LOCAL slope at the operating
point is slightly steeper than `κ`, the globally-fit AVERAGE slope over the whole linear
window (consistent with the very small real-vs-nominal SNR gap already seen in
`fig9_math_report.md` Section 1, "nominal SNR=10dB → real SNR=10.70dB"). No trace of the
paper's claimed high-SNR distortion penalty appears anywhere in the real data range (matches
`fig9_math_report.md` Section 5's finding for the same underlying pipeline).

**Finding 3 — LO-free is pinned near zero (`CCMI≈0.0001`) across the ENTIRE -20..40dB range.**
Direct, expected consequence of the now four-times-independently-confirmed finding (Fig.7,
Fig.9, and the underlying Fig.4/6 threshold math) that Eq.(16) + the real Fig.4 threshold +
the paper's own `ε̃=0.5%` make LO-free structurally unresolvable anywhere close to this SNR
range — it only becomes resolvable in a ~1e-5 dB sliver just below the 46.02dB hard ceiling
(`fig9_math_report.md` Section 5), far outside this figure's -20..40dB window. The paper's own
Fig.8 shows a clearly non-trivial LO-free 8-PAM curve reaching the 3-bit ceiling in this same
range — a substantial, consistent, now well-triangulated discrepancy with our reproduction.

None of these three findings were adjusted, curve-fit, or tuned toward the paper's numbers —
all are the direct, real output of the CCMI formula applied to this project's already-verified
real constants and pipelines.
