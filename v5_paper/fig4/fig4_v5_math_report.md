# Fig. 4 (v5 paper) — Math Report

Probe laser transmission heatmap as a function of coupling detuning Δc/2π and
the dimensionless ratio R = ΩRF/ΓFWHM (Eq.48).
Reference: `References/v5_text.txt` and the rendered PDF page (page 10, Fig.4 image + caption).

## 1. Relationship to Fig.3

Fig.4 is the *same* physical quantity as Fig.3 (Pout(Δc, ΩRF)) — same Hamiltonian
(Eq.6), same Lindblad master equation, same real `qutip.steadystate()` pipeline,
no Doppler averaging (see `fig3_v5_math_report.md` for that decision's full
justification), Pin=20.7µW used directly. Fig.4 only re-expresses the y-axis
(ΩRF/2π → R) and re-renders as a 2D heatmap instead of a 3D surface. **No new
QuTiP computation was run for Fig.4** — it reuses `fig3_v5_qutip_response.npz`
in full.

## 2. Definitions

    R = ΩRF / ΓFWHM   (Eq.48)

ΓFWHM is never given a closed-form formula in the v5 main text — it is only
ever referred to as "the FWHM of the EIT spectrum." Consistent with this
project's established practice (v1's Fig.4 rebuild — see
`v1_paper/fig3_fresh_build`), ΓFWHM was **measured directly** from our own real
data (the ΩRF=0 row of `fig3_v5_qutip_response.npz`), not taken from an
analytic small-signal formula:

    Gamma_FWHM = 1.2231 MHz  (half-max crossings, linear interpolation)

This is much narrower than the ~5.2MHz value found in the earlier v1 work,
consistent with v5's real N0 being ~49,000x higher than v1's placeholder
N0 — same mechanism already documented for Fig.3's sharp, spiky peak.

## 3. Plot construction

- Cropped to the paper's own plotted window: Δc/2π ∈ [-8,8] MHz, R ∈ [1, ~9.8]
  (R only reaches ~9.8, not 10, because Fig.3's existing grid tops out at
  ΩRF/2π=12MHz = 12/1.2231 ≈ 9.81 in R — close enough to the paper's R=10 that
  no grid extension was needed).
- Colorbar in units of x10⁻⁶ W, matching the paper's own Fig.4 colorbar
  convention (not x10⁻⁵ like our Fig.3 plot — chosen here to match Fig.4
  specifically since its native scale is smaller).
- Yellow dotted line at R=1 (paper's stated resolvability threshold).
- White dashed peak-loci lines: **measured** from the real data via
  `scipy.signal.find_peaks` on each R-row (two most prominent peaks), NOT
  assumed from the paper's caption text.

## 4. Real findings (this run)

- **Splitting resolved starting at R ≈ 1.226** (first R value where
  `find_peaks` finds two distinct maxima) — close to, but not exactly at,
  the paper's stated critical threshold R=1. A modest, honestly-reported
  difference, not tuned to match.
- **Measured slope of the peak-locus line vs R ≈ 0.625.** The paper's own
  caption states the loci as "Δc/2π = ±R" (i.e. slope exactly 1). Our real,
  measured slope is meaningfully different (0.625, not 1.0) — flagged as an
  open discrepancy, not adjusted to fit. Not fully explained by this build;
  worth further investigation if precision on this specific ratio matters.
- **Peak Pout in this cropped window = 5.08 µW** (~2.8x higher than the
  paper's own Fig.4 colorbar max of ~1.8µW, read off the actual PDF image) —
  consistent with the same overall peak-scale gap already found and
  documented for Fig.3 (traced to the still-unresolved Pin/diameter
  inconsistency, see `fig3_v5_math_report.md` section 3.1).

## 5. Known open flags (not resolved here)

- The 0.625-vs-1.0 slope discrepancy on the peak loci is unexplained.
- The R≈1.226-vs-1.0 threshold-onset discrepancy is unexplained.
- Both may be downstream consequences of the same Pin/diameter/N0 scale
  issues already tracked for Fig.3, or may indicate ΓFWHM should be measured
  differently (e.g. from a different ΩRF slice, or via a different linewidth
  definition) — not investigated further in this build.
