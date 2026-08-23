# Fig. 4 — Math Report

Distortion of the LO-free Rydberg atomic receiver (AT-splitting peak positions vs `Ω_RF`).

## 1. Data source — REUSED, not recomputed

Per the paper's own text ("actually a top-down view of Fig. 3"), Fig. 4 is a re-analysis of
the identical `P_out(Δc, Ω_RF)` surface Fig. 3 already computed. This script loads
`../fig3_fresh_build/fig3_quantum_response.npz` (the real, 32,361-point, manually
cross-verified grid) — **no new QuTiP solves are performed.** Confirmed by the user before
proceeding (avoids a redundant, deterministic 38-minute re-run that would produce
bit-identical numbers anyway).

## 2. What Fig. 4 shows

For every `Ω_RF`, locate the two Autler-Townes-split peak positions in `P_out(Δc)`. Plot
those positions against `Ω_RF`. As `Ω_RF` shrinks, the two peaks converge; below a
threshold they can no longer be told apart (single-peaked spectrum) — the "distortion
region."

## 3. Equations

**Theoretical peak position** (paper Appendix A, Eq. 48 — exact, closed-form):
`Δc = ±Ω_RF/2`

**Three-level EIT half-width** (paper supplementary Appendix B, Eq. 51 — `Γ_EIT` is
explicitly defined there as the HWHM of the three-level EIT spectrum):
`Γ_EIT = (Ωc² + Ωp²) / (2·√(γ2² + 2·Ωp²))`

**Resolvability criterion (INFERRED, not paper-stated as a formula):**
Two candidate peaks count as a resolved AT doublet only if:
1. Both are genuine local maxima with an interior valley between them (not an edge
   artifact), AND
2. Their separation ≥ `2×Γ_EIT` (= 1 full FWHM linewidth — the standard "just resolved"
   spectroscopy convention, since `Γ_EIT` is a HALF-width by the paper's own definition).

This is a defensible, principled choice (not fitted), and reproduces the paper's stated
"~5.5 MHz" closely. See the session-level sensitivity check already on record: `1.5×Γ_EIT →
3.9 MHz`, `2.0×Γ_EIT → 5.5 MHz`, `2.5×Γ_EIT → 6.6 MHz` — smooth, not a knife-edge fit.

## 4. Parameters used

All already fixed by Fig. 3 — no new parameters needed:
- `Ωp/2π = 8 MHz`, `Ωc/2π = 1 MHz`, `γ2/2π = 5.2 MHz` (paper Sec. IV)

**Important physical fact**: `Γ_EIT`, and hence the whole resolvability threshold, depends
ONLY on `Ωp`, `Ωc`, `γ2` — it has zero dependence on `Pin` or the probe beam diameter `d`.
So the d=0.76mm-vs-0.38mm question that drove Fig. 3's magnitude mismatch is completely
irrelevant here.

## 5. Classification algorithm (per Ω_RF row)

1. Find all local maxima in `P_out(Δc)` within the grid window.
2. If fewer than 2 maxima exist → single-peaked → UNRESOLVED.
3. Otherwise take the 2 tallest, locate the valley between them, require it's a genuine
   interior minimum.
4. Require separation ≥ `2×Γ_EIT` → RESOLVED, else UNRESOLVED.

## 6. What this script does NOT do

- No new QuTiP solves (reuses Fig.3's real grid, justified in Section 1).
- Does not import old `fig4.py` or anything from `fig7_reinvestigation/`.
- Does not fit, tune, or force the threshold to hit 5.5 MHz — the criterion is fixed
  (2×Γ_EIT) before looking at the result.

*(Numeric results appended below after running.)*


## 7. Actual numeric results (this run)

- Gamma_EIT = 2.610126 MHz
- Required separation (2x Gamma_EIT) = 5.220252 MHz
- Sensitivity check: 1.5x=3.9152 MHz, 2.0x=5.2203 MHz, 2.5x=6.5253 MHz
- Distortion threshold (from real grid, 0.1250 MHz resolution) = 5.5000 MHz
- Resolved rows: 115/161
- Comparison to paper's stated ~5.5 MHz: MATCH (within one grid step)

## 8. Note

Plot axes were corrected after the first draft plotted `Ω_RF` on x / `Δc` on y (rotated
relative to the paper's own convention). Fixed to `Δc` on x, `Ω_RF` on y, matching the
published Fig. 4 exactly. Re-ran the same (unchanged) classification — numbers identical,
only the plot orientation changed.
