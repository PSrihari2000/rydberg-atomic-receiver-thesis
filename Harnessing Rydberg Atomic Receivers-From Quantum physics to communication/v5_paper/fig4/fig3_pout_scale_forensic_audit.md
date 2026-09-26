# Fig.3/Fig.4 Absolute Pout Scale — Forensic Audit

Read-only audit. No code, parameters, or frozen figures modified.
Source of truth: `v5_paper/fig3/fig3_v5_qutip.py` (the actual code that
produced `fig3_v5_qutip_response.npz`, currently reused by Fig.4).

## STEP 1 — Absolute power values (measured directly from the real data)

| Quantity | Value |
|---|---|
| Pin (used in code) | 20.7 µW (2.07×10⁻⁵ W) |
| Max Pout (entire grid) | 7.6836 µW |
| Min Pout (entire grid) | 4.29×10⁻⁷ µW (≈0) |
| Max transmission Pout/Pin | 0.3712 (37.1%) |
| Pout at ΩRF=0, Δc=0 exactly | 3.2281 µW (transmission 15.6%) — **note: this is NOT the grid's global maximum**, which occurs at ΩRF/2π=0.12MHz, Δc=0 |
| Peak Pout *within the paper's actual plotted Fig.4 window* (R≥1, \|Δc/2π\|≤8MHz, using Γ_ref=1.2231MHz) | **5.0810 µW** |

Paper's approximate Fig.4 peak (from visual read): ~2×10⁻⁶ W = 2µW.

**Important scope correction before comparing further**: the "7.7µW" figure
you quoted is the *global* grid maximum, which sits at ΩRF/2π≈0.12MHz —
i.e. R≈0.1, *outside* the paper's own plotted R∈[1,10] window entirely.
Comparing that number to the paper's plotted peak is not apples-to-apples.
The fairer comparison is the peak actually inside the plotted window:
**5.08µW vs ~2µW — a 2.54× ratio**, not 3.85×. Still a real, substantial
gap, just smaller than the raw numbers suggested.

## STEP 2 — Exact Pout pipeline (symbolic chain, as actually coded)

```
QuTiP: rho_ss = qt.steadystate(H, C_OPS)      # full 4x4 Lindblad solve
rho21 = rho_ss[1,0]                            # the |2><1| coherence (Eq.4's rho21)
chi   = C0 * rho21                             # Eq.4: chi(t) = C0 * rho21(t)
C0    = -2*N0*wp_12 / (eps0*hbar*Omega_p)      # Eq.4's C0, exact form
OD    = kp * L_cell * Im(chi)                  # optical depth (exponent of Eq.2)
Pout  = Pin * exp(-OD)                         # Eq.2 exactly
```

1. Density-matrix coherence used: ρ21 = ρ_ss[1,0] (QuTiP 0-indexed → paper's ⟨2|ρ|1⟩).
2. Susceptibility: χ = C0·ρ21 (Eq.4, exact).
3. Numerical prefactors: none beyond what Eq.4/Eq.2 state — no extra
   scaling constants anywhere in the code.
4. N0 = 4.89×10¹⁶ m⁻³ (converted from paper's 4.89×10¹⁰ cm⁻³ — verified
   correct: 1 cm⁻³ = 10⁶ m⁻³, so 4.89e10×1e6 = 4.89e16 ✓).
5. Dipole moment: wp_12 = (2.5·e·a0)² — paper's pre-squared convention
   (independently confirmed against the rendered PDF page, not garbled
   text — see `fig3_v5_math_report.md`).
6. Probe Rabi frequency: Ωp/2π = 8MHz, used directly (not derived from a
   field amplitude — Eq.3/diameter is bypassed entirely, see below).
7. kp = 2π/λp, λp = 852nm → kp = 7,374,630.64 rad/m.
8. L = 1cm = 0.01m (paper-stated vapor cell length).
9. Doppler averaging: **NOT used** — confirmed by direct code inspection
   (`fig3_v5_qutip.py` has no velocity loop at all) and by design decision
   (see `project_rydberg_v5_freshbuild` memory / math report).
10. Beer-Lambert: `Pout = Pin * exp(-kp*L*Im(chi))` — exact literal
    implementation of Eq.2, no extra factor.

## STEP 3 — Parameter-by-parameter audit

| Parameter | Paper value (Sec.V-A) | Code value | Units | Match? |
|---|---|---|---|---|
| λp | 852 nm | 852e-9 m | m | ✓ |
| λc | 510 nm | *(not used — Ωc taken directly, λc never enters the Pout pipeline)* | — | N/A |
| Pin | 20.7 µW (stated, itself inconsistent with paper's own Eq.3+d+Ωp — see prior audits) | 20.7e-6 W, used directly | W | ✓ (matches the paper's stated number; the paper's own internal inconsistency is separate, not introduced by our code) |
| probe beam diameter d | 0.76 mm | *(not used at all — Eq.3 bypassed)* | — | N/A by design |
| Ωp/2π | 8 MHz | 8.0e6 Hz | Hz | ✓ |
| Ωc/2π | 1 MHz | 1.0e6 Hz | Hz | ✓ |
| N0 | 4.89×10¹⁰ cm⁻³ | 4.89e16 m⁻³ | m⁻³ | ✓ (unit-converted correctly) |
| L | 1 cm (stated for the vapor cell) | 0.01 m | m | ✓ |
| Temperature | 290 K | *(not used — no Doppler, T never enters the Pout pipeline)* | — | N/A by design |
| ℘12 | (2.5·e·a0)² (pre-squared, confirmed from PDF render) | (2.5*e_charge*a0)**2 | C²·m² | ✓ |
| ℘RF | -1443.459 e·a0 | *(not used in Fig.3/4 — only enters the LO-dressed/RF-dipole context, Fig.6+ )* | — | N/A for this figure |
| γ2/2π | 5.2 MHz | 5.2e6 Hz | Hz | ✓ |
| γ3/2π | 3.9 kHz | 3.9e3 Hz | Hz | ✓ |
| γ4/2π | 1.7 kHz | 1.7e3 Hz | Hz | ✓ |
| γ1 | 0 (implied, ground state) | not explicitly set but never appears (H has no |1⟩ decay term, matching γ1=0) | — | ✓ |
| kp | derived, = 2π/λp | 2π/852e-9 | rad/m | ✓ (2π/852nm = 7,374,630.64 rad/m, matches) |
| Doppler | not used in the derivation chain that reaches Fig.3/4 (see prior audit — mentioned once in Sec.II-C, never applied in Sec.V) | not used | — | ✓ by the earlier documented decision |

**No assumed/inferred values used anywhere in this pipeline** — every
parameter that actually enters the Pout(Δc,ΩRF) calculation is either a
literal paper-stated number or a direct, checked unit conversion of one.
d, λc, T, ℘RF are paper-stated but genuinely don't enter this particular
calculation (confirmed by inspecting every line of `fig3_v5_qutip.py` —
none of those four symbols appear anywhere in the code).

## STEP 4 — Susceptibility / Rabi-frequency convention audit

Checked against the actual rendered PDF pages (not garbled `pdftotext`):

- **Hamiltonian (Eq.6)**: off-diagonal couplings are Ωp/2, Ωc/2, ΩRF/2 —
  standard RWA two-level-coupling form. Code: `(Omega_p/2.0)*(k1*k2.dag()+k2*k1.dag())`
  — **exact match**, no missing/extra factor of 2.
- **C0 (Eq.4)**: C0 = −2N0℘12/(ε0ℏΩp). Code:
  `-2.0*N0*wp_12/(eps0*hbar*Omega_p)` — **exact match**.
- **Beer-Lambert (Eq.2)**: Pout = Pin·exp(−kpL·Im(χ)). Code:
  `Pin*np.exp(-kp*L_cell*np.imag(chi))` — **exact match**.
- **Eq.3 (Pin from d, Ωp, ℘12)**: NOT used at all in this pipeline (Pin is
  a hard-coded direct value) — so any factor-of-2/convention issue that
  might exist inside Eq.3 (there's a √℘12 in it, already audited
  separately) **cannot propagate into the Pout(Δc,ΩRF) shape or scale**,
  because Eq.3 is never called. This rules out Eq.3's own internal
  convention as a cause of the Pout discrepancy.

**No factor-of-2 (or any other) convention inconsistency found** between
our code and the paper's own printed equations, checked against the
actual PDF renders for all three equations that matter (2, 4, 6). This
was checked line-by-line, not assumed.

## STEP 5 — Doppler audit

Confirmed definitively by direct code inspection: `fig3_v5_qutip.py`
contains no velocity variable, no Maxwell-Boltzmann weighting, no
integration loop over vz anywhere — **Doppler averaging is not used** in
the data currently reused by Fig.4.

Whether the paper's *own* Fig.3/4 are supposed to include it: **genuinely
ambiguous, previously audited at length** (see
`fig4_linewidth_forensic_audit.md` and the Fig.3 math report) — Sec.II-C
introduces the Doppler formalism once, but it's never referenced again in
Sec.V's actual simulation description, and the paper's own cited source
[23] explicitly states it uses "the cold atomic model" for this exact
kind of spectrum plot.

**Qualitative effect if Doppler WERE included** (already tested this
session, not reapplied here): a properly-converged Doppler average
*raises* the off-resonance baseline substantially (from ~0 to several µW)
and *lowers* the sharp on-resonance peak — net effect on the peak/baseline
contrast is a flattening, not a uniform rescaling. It would change the
peak value in some direction, but it was already found to destroy the
AT-splitting shape entirely (0% dip at Δc=0) when properly converged, so
it's not a viable explanation for "our peak is too high" without
reintroducing an already-rejected problem.

## STEP 6 — Transmission / optical-depth sanity check

Using Pin=20.7µW throughout:

| | Pout | Transmission (Pout/Pin) | Optical depth OD=−ln(T) |
|---|---|---|---|
| Paper (approx., visual read) | ~2 µW | 0.0966 | 2.337 |
| Ours — global grid max | 7.684 µW | 0.3712 | 0.991 |
| Ours — Δc=0,ΩRF=0 exactly | 3.228 µW | 0.1559 | 1.858 |
| Ours — peak within paper's plotted window (R≥1, \|Δc\|≤8MHz) | 5.081 µW | 0.2455 | 1.405 |

Our model is **less absorptive than the paper's plotted result at every
comparison point** — OD ratios (ours/paper) range from 0.42 (global max)
to 0.79 (at Δc=ΩRF=0) depending which point is compared. There is no
single point where our OD matches the paper's implied ~2.34.

## STEP 7 — Final diagnosis (ranked, no changes made)

1. **(c) Paper-internal ambiguity — MOST LIKELY, well-supported.** The
   already-documented Pin/diameter/Ωp inconsistency (paper's own Eq.3
   doesn't reproduce its own stated 20.7µW from its own stated d=0.76mm
   and Ωp/2π=8MHz — off by 1.89×) means we already know the authors'
   *actual* generating code almost certainly used a self-consistent
   parameter combination that differs from what ended up printed in
   Sec.V-A text. Since Pout depends on Pin only as a linear prefactor,
   *and* on N0/℘12 through the exponent (nonlinearly, via C0), a genuinely
   different internal Pin and/or N0 in their real code would shift both
   the scale and the absorptiveness in ways we cannot reconstruct from
   the published text alone. **Mechanism**: any of Pin, N0, or ℘12 being
   internally different changes Pout multiplicatively (Pin) or through
   the exponent (N0, ℘12 via C0) — both directions are mathematically
   capable of producing a 2-4× shift. **Evidence supports it**: yes, this
   exact inconsistency is independently, separately documented and dated
   to v3's editorial insertion of the "20.7µW" clause.

2. **(d) Expected consequence of different assumptions — PARTIALLY
   explains the discrepancy, well-supported.** The naive "7.7 vs 2µW"
   comparison conflates two different (Δc,ΩRF) regions — our global max
   sits outside the paper's own plotted R≥1 window. Restricting to the
   actual matching window already shrinks the apparent ratio from 3.85×
   to 2.54×. **Mechanism**: not a physics error, a plot-window mismatch.
   **Evidence supports it**: directly confirmed by the Step 1 recomputation
   above — this is not speculative, it's measured.

3. **(b) Parameter mismatch — NOT supported by current evidence.** Every
   parameter that actually enters this pipeline (N0, Ωp, Ωc, γ2/3/4, ℘12,
   λp, L) was checked line-by-line against the paper's stated Sec.V-A
   values (Step 3 table) and matches exactly, including a verified-correct
   unit conversion for N0. No transcription error found.

4. **(a) Model/equation error in our code — NOT supported by current
   evidence.** All three equations that matter (Eq.2 Beer-Lambert, Eq.4
   susceptibility, Eq.6 Hamiltonian) were checked against the actual
   rendered PDF pages (not the unreliable `pdftotext` extraction) and
   match exactly, with no missing or extra factor of 2 anywhere in the
   chain from ρ21 to Pout.

**Bottom line**: the residual ~2.5× gap (after correcting for the
window-mismatch part of the discrepancy) is best explained by category
(c) — an already-documented, paper-internal parameter inconsistency we
cannot resolve without the authors' actual simulation code — compounded
with (d), a genuine but partial window-comparison artifact. No evidence
supports a bug in this project's implementation (categories a/b both
checked and ruled out). No scaling, normalization, or parameter change
has been applied — this is a diagnosis only, per your instruction.
