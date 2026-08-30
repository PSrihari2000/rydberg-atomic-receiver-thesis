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

**Second attempt**: used the paper's own **Eq.57**, Ω″RF,max=4|κ₀'/κ₀''|,
computed from real numeric first/second derivatives. At the analytic bias
(4.23MHz): κ₀'=−0.2363, κ₀''=0.3287 → Ω″RF,max=2.8755MHz, a real,
ε-free, paper-formula number. But plotting this as a band centered at
4.23MHz was ALSO wrong, caught by inspecting the real paper figure
side-by-side: on our curve, 4.23MHz already sits deep in the decayed
tail (Section F), so the shaded band mostly covered flat, near-zero
curve — visibly not a "linear dynamic range" the way the paper's own
Fig.6(a) shows one (there, the curve is still clearly declining across
the whole shaded band).

**Third attempt, also considered**: at our own numeric optimum
(1.31MHz — where the curve is actually steep/quasi-linear), κ₀''≈0 (an
inflection point — mathematically the same property that makes
2nd-order distortion vanish at the paper's own optimum), so Eq.57 is
undefined there. A "local relative-deviation-from-tangent" metric WAS
tried at this point (valid here, unlike near the tail) and reaches a
band of roughly [0.26, 2.16]MHz at a 10% threshold — visually closer to
the paper's own band width. **Not used in the final build**: this is our
own invented criterion, not one the paper specifies, and the project's
standing rule (established during the Fig.5 rebuild) is to match the
paper's stated equations over a better-looking but non-paper metric
whenever the two are in tension.

**Final, adopted approach**: use the paper's own **Eq.58** (the equation
that actually governs at an optimum, since Eq.57's 2nd-order term has
vanished there), Ω‴RF,max=√(6ε|κ₀'/κ₀'''|), evaluated at our real
numeric optimum (1.31MHz) using genuine third derivatives of the QuTiP
curve. ε is genuinely undefined by the paper ("designer-specified
tolerance"), so three explicitly labeled ILLUSTRATIVE values are shown,
none claimed as paper-confirmed:

| ε (illustrative) | Ω‴RF,max |
|---|---|
| 1% | 0.23 MHz |
| 5% | 0.52 MHz |
| 10% | 0.74 MHz |

These are narrow compared to the paper's own LDR (several MHz) — this is
an honest, direct consequence of our curve being genuinely sharper (see
Section F and the closed root-cause audit below), not a new bug. Eq.57
at the analytic bias (2.8755MHz) is still reported as a diagnostic
number in the console output, but is no longer used for the plotted
band, since that bias point doesn't correspond to a genuinely
quasi-linear region on our own curve.

- **ΩRF,min**: still has no closed-form equation anywhere in the paper
  (only "corresponds to the intrinsic sensitivity", no formula) — not
  invented, omitted from the plot.

## E.1 Root-cause audit for the sharpness mismatch (2026-08-30, closed)

Ran a dedicated forensic audit (`../fig3/fig3_sharpness_root_cause_audit.md`)
specifically hunting for a second hidden N0/L-style paper inconsistency
(in the same family as the already-found Pin/diameter one) that might
explain why our curve is sharper than the paper's plotted one, and which
— if found — would likely cascade-fix Fig.3-10 at once. Checked: Eq.2/4's
exact form (matches our implementation exactly), the paper's own
Rydberg-blockade justification (Nb≈0.02≪1, which explicitly confirms raw,
uncorrected N0 should be used — the opposite of a hidden correction), and
N0/L consistency everywhere in the text (identical every time). **Found
nothing fixable.** Combined with the independent QuTiP-vs-numpy
cross-check (<0.12% agreement, see Fig.3 report), this confirms our
computation is correct given the paper's stated inputs — the sharpness
gap is a real, unresolved property of the paper's own published figures,
not a parameter we're getting wrong. This line of investigation is
closed; see [[feedback_deviation_metrics_and_recurring_sharpening]] in
project memory.

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

## G. Presentation additions (2026-08-30): cross-panel markers, shared axes, pipeline schematic

Following the same cross-referencing style used in this project's v1
Fig.5 presentation, and the paper's own Fig.6 layout:

- **Cross-panel colored markers**: for each Ω_RF trace (0.5/1.5/3.0MHz),
  the phasor Ω_total(t) swings between Ω_LO−Ω_RF (at t=period/2, the
  curve's HIGHEST Pout point) and Ω_LO+Ω_RF (at t=0 and t=period, the
  LOWEST Pout point). These two real, already-computed extremes are
  marked with matching colored dots in all three panels, connected by
  thin dotted horizontal/vertical guide lines — no new computation, just
  re-reading points already on the real curves.
- **Shared axes across panels (important fix, caught by inspection
  against the real paper figure)**: initially each panel auto-scaled
  independently, so panel (b)'s Ω_total axis and panel (c)'s Pout axis
  drifted away from panel (a)'s ranges — this breaks the whole point of
  the guide lines, since a line at a given Ω_total or Pout value no
  longer lands at the same visual position across panels. Fixed by
  explicitly forcing panel (b)'s x-axis = panel (a)'s x-axis (0-10MHz),
  panel (c)'s y-axis = panel (a)'s y-axis (0-9µW), and panel (b)'s
  y-axis = panel (c)'s x-axis (0 to one real period) — matching exactly
  how the paper's own Fig.6 shares axes between its three panels.
- **Panel (d) — pipeline schematic**: reproduces the paper's own small
  Input→Quantum procedure→Output diagram (the L-shaped arrow figure
  next to the paper's real Fig.6), placed in the previously-empty
  bottom-right slot. This is a static annotation, not a data plot — it
  restates Eq. for Ω_total(t) and labels which panel is which stage.

None of this changes any computed value — it only fixes how the same
real numbers are displayed and cross-referenced.

## Files

- `fig6_v5_qutip.py` — build script (real QuTiP throughout)
- `fig6_v5_qutip_ldr.png` — combined 3-panel plot
- `fig6_v5_qutip_ldr_data.csv` — full data (panels a/b/c)
- `fig6_v5_qutip_ldr_response.npz` — raw arrays for reuse
- `fig6_v5_forensic_audit.md` — the Phase-1 audit (equations, parameter
  table, ε/ΩRF,min gap analysis)
- `fig6_v5_math_report.md` — this file
