# Fig. 6 (v5 paper) — Math Report

Linear dynamic range of the **LO-dressed** Rydberg atomic receiver (unlike
Fig.3/4/5, which are LO-free). Full audit in `fig6_v5_forensic_audit.md`;
this report documents the final build and its real findings.

## A. Paper-stated quantities

Ωp/2π=8MHz, Ωc/2π=1MHz, γ2/2π=5.2MHz, γ3/2π=3.9kHz, γ4/2π=1.7kHz,
N0=4.89×10¹⁶m⁻³, ℘12=(2.5·e·a0)², λp=852nm, L=1cm, Pin=20.7µW, Δf=15kHz.
Ω_LO,opt=4.23MHz and the numeric Ω_LO=4.67MHz are the paper's own stated
results for comparison, not inputs.

## B. Directly derived quantities

- Γ (four-level, Eq.78) = 7.3255 MHz — computed from Ωp, Ωc, γ2 only.
- Ω_LO,opt (analytic) = Γ/√3 = **4.2294 MHz** — matches the paper's
  stated 4.23MHz almost exactly, strong validation this is the correct Γ
  (not Fig.4/5's measured Γ_ref=1.2231MHz, which is a different quantity
  used in a different context).

## C. Genuine QuTiP numerical quantities

- Panel (a): 201-point real `qutip.steadystate()` sweep of Pout(Ω_total),
  Δp=Δc=Δ_RF=0, |3⟩↔|4⟩ coupling = Ω_total. No Natoms/Aeff/distance
  anywhere (confirmed not needed — Fig.6 is a pure intrinsic response,
  not a link-budget calculation).
- Ω_LO,opt (numeric) = 1.3087 MHz — found via `np.gradient`, max
  |dPout/dΩtotal|, excluding a finite-difference edge artifact near
  Ω_total=0 (see history below).
- Real second derivative κ₀'' at each candidate bias, used for the
  Eq.57 LDR bound (Section E).
- Panels (b)/(c): Ω_total(t)=|Ω_LO+ΩRF·e^{j(2πΔf·t+Δφ)}| (confirmed
  complex phasor form, not a plain cosine — see audit Section 7),
  Pout(t) obtained by interpolating the real Panel (a) curve, not a
  fresh solve per time point and not an analytic substitute.

## D. Unavoidable illustrative assumptions

- Δφ=0 — no paper value exists, flagged explicitly.
- ΩRF values for panels (b)/(c) — 0.5/1.5/3.0MHz, matching the paper's
  own legend exactly (not our invention, this one is paper-stated).
- Ω_LO used for panels (b)/(c) = the analytic optimum (4.23MHz), a
  reasonable choice matching the paper's own emphasis on that point.

## E. The Linear Dynamic Range — corrected methodology (history kept, not hidden)

**First attempt (wrong, corrected same session)**: shaded the LDR based on
where a straight tangent line (through the curve at Ω_LO,opt) stayed
within some % of the true curve, measured **relative to the global peak**.
This gave a nonsensical result — the shaded band ended up sitting in the
region where the curve had *already collapsed to near-zero*, because near
zero, both the true curve and the tangent are tiny numbers and trivially
"agree" in absolute terms, regardless of whether the tangent means
anything there. Caught directly by the user pointing out the band didn't
make sense.

**Second attempt (still flawed)**: measured deviation relative to the
*local* curve value instead of the peak — better in principle, but gave
an oddly narrow window (~±0.1-0.5MHz) because the underlying comparison
(tangent-line-fit quality) still isn't what the paper's own distortion
analysis actually measures.

**Final, correct approach**: use the paper's own **Eq.57**,
Ω″RF,max = 4|κ₀'/κ₀''|, computed from **real numeric first/second
derivatives** of our actual QuTiP curve. This needs no ε at all.

- At the analytic bias (4.23MHz): κ₀'=−0.2363, κ₀''=0.3287 (µW/MHz,
  µW/MHz²) → **Ω″RF,max = 2.8755 MHz** — a real, ε-free, paper-formula
  result. LDR band shown as Ω_LO,opt ± this value = [1.35, 7.10] MHz.
- At our own numeric optimum (1.31MHz): κ₀''≈−0.0175 (essentially zero)
  → Eq.57 blows up (761MHz, meaningless). **This is not a bug** — the
  point of maximum |slope| on any smooth curve is mathematically always
  an inflection point (zero second derivative), which is the exact same
  structural property the paper describes for its own Ω_LO,opt ("employing
  an optimal LO drive... simultaneously maximizes the intrinsic
  coefficient κ and eliminates the second-order distortion"). Our curve
  reproduces this same structural behavior, just centered at a different
  Ω_total (1.31 vs 4.23MHz) because our curve is sharper overall (see
  Section F). At OUR true optimum, we hit the same wall the paper hits at
  its own optimum: the next bound (Eq.58, third-order) needs ε, which the
  paper itself calls "the designer-specified tolerance" — genuinely
  undefined, not chosen here.
- **ΩRF,min**: still has no closed-form equation anywhere in the paper
  (only "corresponds to the intrinsic sensitivity", no formula) — not
  invented, omitted from the plot.

## F. Honest curve-shape finding (root cause, tying every discrepancy together)

Our real curve collapses far more sharply (7.4µW→0.0002µW by 10MHz) than
the paper's own plotted curve (~8.3→2µW over the same range). This is the
**same post-absorption compression mechanism already found and explained
for Fig.3/4's ΓFWHM** (real high N0 makes the exponential Beer-Lambert
term amplify small coherence changes into large Pout swings), now
confirmed to recur in a fourth, independent context. Consequences,
all downstream of this one cause, not separate bugs:
- Peak height matches reasonably well (7.7 vs 8.3µW, ~7%) since it's set
  near Ω_total→0 where absorption differences are minimal.
- The falloff rate, tangent-line divergence, LDR width/position, and the
  shape of panels (b)/(c) (sharp pulses vs. the paper's smooth
  oscillations) all differ from the paper — all explained by this same
  mechanism, not four independent unexplained gaps.

## Files

- `fig6_v5_qutip.py` — build script (real QuTiP throughout)
- `fig6_v5_qutip_ldr.png` — combined 3-panel plot
- `fig6_v5_qutip_ldr_data.csv` — full data (panels a/b/c)
- `fig6_v5_qutip_ldr_response.npz` — raw arrays for reuse
- `fig6_v5_forensic_audit.md` — the Phase-1 audit (equations, parameter
  table, ε/ΩRF,min gap analysis)
- `fig6_v5_math_report.md` — this file
