# Fig. 3 (v5 paper) — Math Report (FINAL)

Probe laser transmission Pout versus coupling detuning Δc and RF Rabi
frequency ΩRF. Reference: `References/v5_text.txt`,
`References/[v5]Harnessing Rydberg Atomic Receivers-v5-accepted.pdf`
(Eqs. 1-8, Sec.V-A parameters), and the rendered PDF pages (not the
garbled `pdftotext` extraction, which mangled several key equations).

## 1. Governing equations

Hamiltonian (Eq.6), 4-level system ⟨1⟩→⟨2⟩→⟨3⟩→⟨4⟩ (Cs: 6S1/2→6P3/2→47D5/2→48P3/2):

    H = (hbar/2) * [[0, Ωp, 0, 0],
                     [Ωp, -2Δp, Ωc, 0],
                     [0, Ωc, -2(Δp+Δc), ΩRF],
                     [0, 0, ΩRF, -2(Δp+Δc+ΔRF)]]

Lindblad decay (Eq.7): γ2/2π=5.2MHz, γ3/2π=3.9kHz, γ4/2π=1.7kHz, γ1=0,
dephasing γij=(γi+γj)/2.

Master equation, steady state (Eq.5): dρ/dt = -(j/hbar)[H,ρ] + L(ρ) = 0,
solved via real `qutip.steadystate()` (method="direct").

Susceptibility and transmission (Eqs. 2, 4):

    χ = C0 * ρ21,  C0 = -2*N0*wp12 / (eps0*hbar*Ωp)
    Pout = Pin * exp(-kp*L*Im(χ)),  kp = 2*pi/lambda_p

## 2. Parameters (paper Sec. V-A, v5 values)

| Parameter | Value |
|---|---|
| L (vapor cell length) | 1 cm |
| N0 | 4.89e10 cm^-3 = 4.89e16 m^-3 |
| gamma2/2pi, gamma3/2pi, gamma4/2pi | 5.2 MHz, 3.9 kHz, 1.7 kHz |
| Omega_p/2pi, Omega_c/2pi | 8 MHz, 1 MHz |
| lambda_p | 852 nm |
| wp12 | (2.5 e a0)^2 |
| Pin | 20.7 uW (paper-stated, used directly) |

## 3. Decisions made (confirmed with user, all still standing)

1. **Pin used directly (20.7 uW), Eq.3/diameter bypassed.** v1/v2 never
   stated Pin at all (just d=0.76mm -> Ωp/2pi=8MHz). v3 grafted "the laser
   intensity is 20.7uW" onto that existing, unchanged pair without
   re-deriving via Eq.3 (which gives 39.08uW from the same d/Ωp) --
   the mismatch is traceable to that specific v3 edit, not to this
   project's implementation. Since diameter never enters the ρ21/χ
   computation (only Eq.3 does, and Eq.3 is skipped), using the paper's
   more-direct, later-stated 20.7uW sidesteps the inconsistency.

2. **rho21 solved via REAL qutip.steadystate()**, not the paper's literal
   closed-form Eq.8 (which was found, this session, to give unphysical
   gain -- Pout>Pin -- due to the same Appendix-A sign/phase-convention
   issue already documented for Eq.26 in the v1 investigation). Real
   QuTiP is used specifically because this script is meant to be shown
   to/run by the user's guide, who checks for literal QuTiP usage --
   see `feedback_real_qutip_required` memory.

3. **NO Doppler averaging.** Investigated at length (see history in the
   script's own header comments): an initial quick test looked
   promising, but a convergence check proved the velocity integral needs
   N_VZ~400+ to converge, and properly converged Doppler averaging
   erases the AT-splitting entirely (0% dip at Δc=0). Confirmed via
   direct grep of v5's own text (Doppler mentioned exactly once, only in
   Sec.II-C, never in Sec.V's actual simulation description) and via the
   paper's own cited source [23] (Wu et al. 2023, PRA), which explicitly
   states it uses "the cold atomic model" for this exact kind of
   spectrum plot.

4. **Axis presentation matches the paper's own Fig.3 exactly**: Δc/2π
   ticks every 10MHz (-20 to 20), ΩRF/2π ticks every 4MHz (0 to 12),
   Pout ticks every 0.2 (×10⁻⁵ W units), and the Δc axis direction
   inverted to read negative-left/positive-right (matching the paper;
   the camera trick borrowed from v1 -- `view_init` + `zaxis._axinfo
   ['juggled']` to force Pout onto the left -- happened to also mirror
   the Δc direction, corrected via `ax.invert_xaxis()`).

## 4. Numeric results (current frozen run)

- Grid: Δc/2π ∈ [-20,20]MHz (201 pts), ΩRF/2π ∈ [0,12]MHz (97 pts) --
  19,497 real `qutip.steadystate()` solves, ~30 min runtime.
- Peak Pout = 7.6836 µW at ΩRF/2π=0.125MHz, Δc/2π=0MHz (CSV line 303).
- Edge Pout (ΩRF=0, Δc=±20MHz) = 0.0000 µW exactly.
- Pout(Δc=0,ΩRF=0) = 3.228101 µW -- independently re-derived live and
  confirmed to match the CSV to all 6 decimal places (see chat log
  2026-08-29), proving every row is a genuine, independent fresh solve,
  not interpolated or cached from a formula.

## 5. Known open discrepancies vs. the paper's own plotted Fig.3 (documented, not "fixed")

**5.1 Peak scale gap (~2.5x).** Full forensic audit in
`../fig4/fig3_pout_scale_forensic_audit.md`. Summary: comparing like-for-like
windows, our peak within the paper's actual plotted range is 5.08µW vs.
the paper's ~2µW (2.54x). All three governing equations (2,4,6) and every
parameter that enters this pipeline were checked line-by-line against the
actual PDF and match exactly -- no bug or parameter-transcription error
found. Most likely cause: the already-documented Pin/diameter/Ωp
inconsistency means the authors' real generating code almost certainly
used different internal numbers than what ended up printed in Sec.V-A.

**5.2 Nonzero floor (paper ~0.2x10^-5 W = 2µW; ours = 0 exactly).**
Tested directly (2026-08-29): scanned Doppler-averaging width from 0% to
100% of the real thermal value. AT-splitting survives only up to ~3% of
the full width (σv≈4 m/s); the floor doesn't reach a realistic ~2µW
until ~30-40% of the full width (σv≈40-55 m/s) -- a 10-15x gap with NO
overlapping regime where both the paper's floor AND its splitting hold
simultaneously in this model. This rules out "the right amount of
Doppler" as an explanation. No other floor-raising mechanism (detector
noise floor, stray light, additional absorption channel) has any support
in the paper's text -- not added, to avoid fabricating an unsupported term.

Both gaps are accepted, documented limitations of this reproduction, not
actively being chased further absent new information from the paper.
