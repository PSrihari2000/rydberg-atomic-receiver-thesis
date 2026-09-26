# Fig. 5 (v5 paper) — Math Report

Probe transmission Pout vs coupling detuning Δc and free-space link distance
dTx-Rx, for the **LO-free** Rydberg receiver (confirmed directly from the
paper's own text, Sec.V-B — "SNR performance of LO-free Rydberg atomic
receiver, SNRRy").

## 1. Why this build follows the paper's stated method exactly

Earlier attempts in this project reused the real, frozen Fig.3 QuTiP data
directly (via interpolation) instead of the paper's own described method.
That was reconsidered: since this figure exists to be compared directly
against the published paper, the goal is to match the paper's own model,
parameters, equations, and assumptions — not substitute a different
(even if arguably more rigorous) approach. This build therefore follows
the paper's literal stated method.

## 2. The paper's stated method (Sec.V-B, exact quote)

*"For each distance dTx-Rx, we extract a distance-dependent RF Rabi
frequency ΩRF(dTx-Rx) and then use QuTip toolkit to generate the EIT-AT
spectrum as two Lorentzian peaks (half-width ΓFWHM/2) centered at
detunings Δc=±½ΩRF(dTx-Rx)."*

## 3. Equation chain implemented

**Shape** (paper's literal formula, normalized — see Section 6 for why):

    raw(Δc) = L(Δc − ΩRF/2) + L(Δc + ΩRF/2),   L(x) = (Γ/2)² / (x² + (Γ/2)²)
    Pout(Δc) = A · raw(Δc) / max(raw)

using Γ = Γ_ref = 1.2231 MHz (our settled project-wide ΓFWHM convention —
see `fig4_linewidth_forensic_audit.md` FINAL DECISION), so half-width =
0.61155 MHz. The division by max(raw) forces the curve's peak to equal
exactly A regardless of how much the two Lorentzians overlap — see
Section 6 for the artifact this fixes.

**Amplitude A**: the paper says QuTiP generates this spectrum — so we use
**our own real QuTiP data** in that role ("if they used their QuTiP, we
use our QuTiP"). A(ΩRF) is the actual simulated peak Pout at that same
ΩRF, read from the frozen `fig3_v5_qutip_response.npz` grid (interpolated
between real grid points as needed). Nothing about the amplitude is
invented — every value traces back to a genuine master-equation solve.

**Distance → ΩRF(d)**: dTx-Rx → SRx=PTx·GTx/(4πd²) → PRx=SRx·Aeff →
ERF=√(2Z0·PRx) → ΩRF=℘RF·ERF/ℏ, with Aeff=2Z0·Natoms·℘RF²·ωRF/(ℏΓFWHM)
(Eq.17). Algebraically this collapses to:

    ΩRF(d) = K · √Natoms / d

where K bundles every paper-confirmed constant (PTx, GTx, ℘RF, fRF,
Γ_ref, Z0, ℏ) — **Natoms is the only unknown in the entire chain, and it
only ever appears as √Natoms.**

## 4. The Natoms gap — why the distance axis is a reference, not a calibration

Natoms has no numeric value or formula anywhere in the v5 paper — confirmed
by exhaustive search of the paper text (11 symbolic occurrences, zero
numeric values) and of all three cited reference papers ([12] Sedlacek
2012, [20] Jing 2020 — has a similarly-named but different quantity, an
atoms/second *rate*, not a count — [23] Wu 2023 — zero mentions of
effective aperture/atom count at all). Appendix B explicitly states Aeff
scales with Natoms "rather than the physical cross-section of the vapor
cell," ruling out the natural geometric fallback in the authors' own words.

Rather than silently guess a number and present it as settled, this build
uses **one reference value** — Natoms_ref = N0 × π(d_beam/2)²×L_cell =
**2.2183×10⁸** — purely to draw a concrete curve. Because ΩRF(d) ∝
√Natoms, the real distance for the true (unknown) Natoms is simply:

    d_real = d_plotted × √(Natoms_true / Natoms_ref)

The plotted axis is not claimed to be the paper's actual calibration —
only the shape and relative behavior are asserted as correct.

## 5. Parameter table

| Parameter | Value | Source |
|---|---|---|
| N0 | 4.89×10¹⁶ m⁻³ | paper-stated |
| ℘RF | 1443.459 e·a0 | paper-stated |
| fRF | 6.9 GHz | paper-stated |
| PTx | 30 dBm = 1 W | paper-stated |
| GTx | 2.15 dBi | paper-stated |
| Γ_ref (=ΓFWHM) | 1.2231 MHz | measured, project-wide standing convention |
| A(ΩRF) | 2.40–7.68 µW (varies with ΩRF) | measured, real QuTiP data (Fig.3) |
| d_beam, L_cell (Veff inputs only) | 0.76mm, 1cm | paper-stated, repurposed as our reference assumption's input |
| **Natoms_ref** | **2.2183×10⁸** | **OUR REFERENCE — not paper-confirmed** |

## 6. The overlap-doubling artifact — found, explained, then fixed

**First version of this build was unnormalized**: Pout(Δc)=A·[L(Δc−ΩRF/2)+L(Δc+ΩRF/2)],
no division by max(raw). That gave a **global peak of 15.18 µW** at
Δc=0, ΩRF/2π=0.131 MHz — about 2× our own real single-EIT-peak amplitude
(7.68µW) and ~7-8× the paper's own plotted peak (~2µW).

**Root cause, verified numerically**: at small ΩRF, the two Lorentzian
centers (±ΩRF/2) sit closer together than the half-width (0.61MHz), so
each curve is still near its own maximum at the *other* curve's center —
e.g. at ΩRF/2π=0.125MHz, each Lorentzian alone evaluates to ~0.99 of its
own peak at Δc=0, so the raw sum reaches ~1.98×A instead of the intended
A. This is a pure arithmetic property of adding two curves that nearly
coincide, not real physics — the true single EIT peak never actually
doubles like that as splitting disappears.

**Fix applied** (per user + independent second-review consensus, 2026-08-29):
normalize by the shape's own maximum (Section 3's formula) so the curve's
peak equals exactly A(ΩRF), the real QuTiP value, regardless of overlap.
Re-run confirms this: **new global peak = 7.68 µW**, matching the real
QuTiP maximum almost exactly (7.6836µW, residual difference is just
interpolation between grid points) — the artificial doubling is gone.

**What remains, still real and still open**: our peak (7.68µW) is still
~3-4× the paper's own plotted peak (~2µW) — this is the pre-existing
Fig.3 peak-scale gap, traced to the paper's own Pin/diameter/Ωp
inconsistency (see `../fig3/fig3_v5_math_report.md` and
`../fig4/fig3_pout_scale_forensic_audit.md`), not something this fix
addresses or is meant to address. Not corrected, rescaled, or hidden —
reported as a separate, already-documented, genuine limitation.

## 7. Files

- `fig5_v5_two_lorentzian.py` — build script
- `fig5_v5_two_lorentzian.png` — plot (paper-style axes: Δc direction and
  10ⁿ-style distance ticks matching the paper's own presentation)
- `fig5_v5_two_lorentzian_data.csv` — full (Δc, distance, ΩRF, Pout) grid
- `fig5_v5_math_report.md` — this file
