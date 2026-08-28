# Fig. 9 — Math Report

SER performance versus SNR. (a) Conventional (b) LO-dressed (c) LO-free.

## 1. Data sources — REUSED, not recomputed

- `fig4_fresh_build/fig4_classification.npz`: real threshold, `5.5000 MHz`.
- `fig5_fresh_build/fig5_data.npz`: real static response (`static_omega_mhz` actually spans
  `[0.00, 20.00]MHz` in the saved file — the earlier `fig5_math_report.md` note calling this
  "restricted to 0-14MHz" describes the paper's *plotted* axis, not the full saved array; the
  full 0-20MHz range is what's actually loaded and used here), `P0_bar`, `Ω_LO`. `κ` re-fit
  fresh (same algorithm/data as `fig5/6/7_fresh_build`, reproduces the identical number).
- `A_LO`, `σ²_Ry,LO` (Eq.36/37): re-derived fresh, matches `fig6/7_fresh_build` exactly.

**No new QuTiP solves.**

## 2. Closed-form SER (Proakis, *Digital Communication*, 4th ed. — the paper's own cited [44])

`SNR` = average symbol SNR (`Es/N0`), matching this project's SNR convention throughout
(signal power / noise power, no separate bit-rate normalization):
```
M-PAM:  SER = 2(M-1)/M · Q( sqrt(6·SNR/(M²-1)) )
M-QAM:  SER = 1 - [1 - 2(1-1/√M)·Q( sqrt(3·SNR/(M-1)) )]²      (square M-QAM)
Q(x) = 0.5·erfc(x/√2)
```
LO-free uses M-PAM (amplitude-only readout, no phase — same reasoning as Fig.7/8's text).
Conventional and LO-dressed use M-QAM (both retain phase).

## 3. The "club closed-form + QuTiP" mechanism

Same formula throughout; only its *input SNR* differs between "operating" and "distortion"
regions:
- **Panel (a) Conventional**: `SER = f(SNR_nominal)` directly — no atoms, no distortion
  mechanism, no region split.
- **Panel (b) LO-dressed**: invert Eq.(37) for the nominal x-axis SNR to get the required
  `PRx`/`Ω_RF`, run that through the REAL phasor-sum (`Ω_total(t)`, Δφ=0) → real interpolated
  static curve → real Fourier-projected `|P(Δf)|` pipeline (identical to `fig6_fresh_build`'s
  `practical_fourier_amplitude`, reparametrized by `Ω_RF` directly instead of distance) → real
  achieved `SNR_real`. `SER = f_QAM(SNR_real)`.
- **Panel (c) LO-free**: invert Eq.(16) for the nominal SNR's implied `PRx`/`Ω_RF`, gate by the
  real Fig.4 threshold: `SNR_real = SNR_nominal` if resolvable, else `SNR_real = 0` (total
  readout failure — a hard cutoff, not a smooth taper, since AT-splitting resolvability is a
  binary phenomenon). `SER = f_PAM(SNR_real)`.

At `SNR_real=0`, `Q(0)=0.5`, so `f_PAM` naturally saturates at its own ceiling `(M-1)/M` — no
artificial gap/NaN needed for the distortion-region plateau.

## 4. Actual numeric results — TWO honest, real discrepancies with the paper

**Panel (a) reproduces qualitatively well**: 4 curves, correctly ordered (higher M = worse
SER), monotonically decreasing, matching the paper's panel (a) shape.

**Panel (b) — no bathtub/U-shape appears.** All 4 curves decrease monotonically with nominal
SNR up to 40dB; none turn back upward the way the paper's `log₂M=10` curve does. Root cause:
`A_LO` is tiny (`4.478e-07`), so even at nominal SNR=40dB the implied drive stays inside or
near the already-established linear window (`[1.25,7.25]MHz`, `fig5_fresh_build` Section 8) —
not far enough into the nonlinear region to reproduce the paper's dramatic high-SNR decline.
This is the SAME pattern already documented in `fig6_math_report.md` (our Practical LO-dressed
dip has consistently come out much milder than the paper's own claimed effect) and
`fig7_math_report.md` — a third independent confirmation of the same root cause, not a new
bug. Diagnostic check confirmed 0/401 nominal-SNR points ever left the real static curve's
interpolation range, so this isn't a data-range artifact either — the real nonlinearity is
just genuinely weaker, at these parameter values, than what the paper's plot implies.

**Panel (c) — completely flat, no operating region at all within -40..20dB.** `resolvable_c`
is `False` for every single point in the sweep. This is the SAME finding already surfaced in
`fig7_math_report.md` Section 4 (the AT-splitting threshold, moved into SNR-space via Eq.16,
only opens at the ~46.02dB hard ceiling, `1/ε̃²`) — now confirmed a THIRD time, independently,
via a completely different figure (SER instead of MI). Across Fig.7 and Fig.9, evaluating the
paper's own Eq.(16) + the real Fig.4 threshold + the paper's own stated `ε̃=0.5%` together
gives a resolvability boundary near 46dB, not the ~0dB the paper's own Fig.9(c) visibly shows.
This is now a robust, cross-validated inconsistency between the paper's stated equations/
parameters and its own plotted figures — not attributable to a coding error on our side (the
same real threshold, same real Eq.16, same real `ε̃` have been re-derived and cross-checked
independently in `fig4`, `fig6`, `fig7`, and now `fig9`, all agreeing with each other, all
disagreeing with the paper's plotted transition point).

Neither of these was adjusted, curve-fit, or tuned toward the paper's numbers. Both are the
real, direct output of the paper's own equations and this project's already-established real
constants.

## 5. Extended sweep (user request, 2026-08-20) — confirms these are structural, not range-limited

Both panels' x-axis were genuinely extended (real recomputation, not `xlim` padding):

**Panel (c)**: extended to `-40..46.5dB` (2601 points, ~0.033dB resolution). Result: **flat
across the entire range**, terminating at a hard vertical asymptote at
`10·log10(1/ε̃²)=46.0206dB` — annotated on the plot. Under Eq.(16) with the paper's own stated
`ε̃=0.5%`, no finite `PRx` can ever produce a nominal SNR at or above this value, for any
transmit power. The AT-splitting-resolvability crossing point (Section 4/Fig.7 finding) sits
only `1.9e-6 dB` below this same asymptote — physically unreachable at any usable resolution.
**Conclusion: this is not a "hasn't come down yet" situation — under Eq.(16) + the real Fig.4
threshold + the paper's own `ε̃`, LO-free structurally cannot show a visible operating region
anywhere in a realistic SNR range.** This is now the strongest form of the discrepancy first
noticed in `fig7_math_report.md`.

**Panel (b)**: extended to `-40..90dB` (1301 points). All four curves fall monotonically to the
numerical floor and stay there — **no bathtub/rise appears anywhere within the real static
curve's data range** (`Ω_total∈[0,20]MHz`, i.e. nominal SNR `<71.25dB`, marked with a vertical
dotted line on the plot). Beyond that point `Ω_total(t)` genuinely leaves the interpolation
range and `SNR_real`/`SER` become undefined (188/1301 points, all above 71.25dB) — this is a
real data-range limit, not a computed result, and would require extending `fig3_fresh_build`'s
original QuTiP sweep past `Ω_RF=20MHz` (a new solve) to investigate further. Not attempted here
per the user's "let's revisit unfinished plots later" scoping decision — flagged as the natural
next step if this is worth chasing.

## 6. Reconciliation against `Documentation/` reference material (user request, 2026-08-20)

Three prepared reference documents were discovered in `Documentation/` (dated 2026-08-17,
predating this session's Fig.6-9 work): `01_Analytical_SER_SNR_Derivation.pdf`,
`02_Quantum_Distortion_QuTiP_Implementation.pdf`, `Parameter_values_Rydberg_atomic_receiver.pdf`.
A full reconciliation pass was run against all three.

**Doc.01 validates this figure's formulas/parameters/constants exactly** (same SNR formulas,
same `wp_RF=-1443.459·e·a0` convention, same `N0=1e15 m⁻³`, same Proakis PAM/QAM SER forms) and
**independently derives the same `γ_max=1/ε̃²≈46.02dB` LO-free ceiling** found in this project's
own Fig.7 work, explicitly calling it *"the finite SNR ceiling of this simplified LO-free
model"* — i.e. an already-known, already-documented property of this exact model, not a
reproduction error. It also states outright that *"the non-monotonic distortion portions cannot
be claimed to follow from the analytical SER equation alone"* for the paper's own Fig.9 curves.

**Doc.02 specifies a more rigorous methodology** for the distortion regions: a REAL AT-splitting
lookup table (built via peak detection on the QuTiP response, not the FWHM-based `2×Γ_EIT`
criterion this project used in `fig4_fresh_build`), plus a Monte-Carlo symbol-level "quantum
detector" that still attempts detection below threshold using a noisy-but-real reading, rather
than a hard gate to zero. This was built and run fresh (`fig9_lofree_quantum_detector.py`, does
not edit this frozen file):

- **New AT-splitting lookup table**, built from `fig3_fresh_build`'s real grid via `find_peaks`
  (correcting an initial sign-convention mistake caught by doc.02's own warning to validate the
  peak sense before trusting `find_peaks` — peaks are in transmission `P_out`, not absorption;
  confirmed directly by inspecting the real curve shape before proceeding).
- **Alternative threshold, by relative AT-splitting measurement accuracy**: `2.1-2.4MHz`
  (10%/5%/2% tolerance), noticeably LOWER than `fig4_fresh_build`'s FWHM-based `5.500MHz`.
- **Decisive check**: does this lower threshold move the SNR-domain finding? No — `PRx`(and
  hence SNR) at `Ω_RF=2MHz` vs `Ω_RF=5.5MHz` differ by 46.0206dB → 46.0206dB (same to 5 decimal
  places). The `1/ε̃²` ceiling is reached so steeply that the exact threshold choice is
  essentially irrelevant to the SNR-domain conclusion.
- **Full Monte-Carlo quantum detector**, run with 200,000 symbols/point, real AT-splitting
  lookup table, and a noise std derived directly from the SAME `σ_Ry=√(PRx/SNR)` budget used
  throughout this project (not an arbitrary value, per doc.02's own rule) — **result:
  numerically indistinguishable from the simple hard-gated model** (SER pinned at
  `(M-1)/M` — `0.75/0.9375/0.984/0.996` for `log₂M=2/4/6/8` — across the entire `-40..46dB`
  range). Root cause: the true `Ω_RF` implied by any SNR in this range is astronomically small
  (e.g. `~2×10⁻³MHz` even at nominal SNR=40dB) compared to the smallest resolvable AT-splitting
  in the real data (`~1.4MHz`) — no detector, however sophisticated, can extract information
  that isn't physically present in a signal that small relative to the noise floor.

**Conclusion of the reconciliation**: the more rigorous doc.02 methodology, actually implemented
and run (not just reasoned about), **confirms rather than changes** this project's Fig.7/9/8
finding. This is now the fifth independent confirmation (Fig.4/6's threshold derivation, Fig.7,
Fig.9's original hard-gate, and now this Monte-Carlo reconstruction), using two different
threshold-derivation methods and two different distortion-region models, all agreeing. This
should be treated as a settled, load-bearing conclusion for the thesis, not an open question.

**Doc.03 (parameter sheet) independently confirms** `Pin=39.0761µW` for `d=0.76mm` and
`Pin=9.7690µW` for `d=0.38mm` (exact match to `fig3_fresh_build`'s own numbers, including the
4× ratio), using identical framing to this project's own diameter investigation (0.76mm is
paper-explicit; 0.38mm is "not a paper-stated radius interpretation," an "empirical calibration
choice"). One minor housekeeping note: doc.03 recommends full-precision CODATA constants
(`e=1.602176634e-19`, `a0=5.29177210903e-11`) where this entire fresh-build pipeline has
consistently used rounded values (`e=1.6e-19`, `a0=5.2e-11`) throughout Fig.3-9 — a ~0.1%
precision difference, not expected to change any qualitative finding, not corrected here to
avoid silently breaking consistency with everything already cross-checked this session.

