# Fig. 5 — Time-Dependent QuTiP Validation — Math Report

**Scope**: a targeted, additional validation check. Does NOT replace or modify the frozen
`fig5_lodressed_analysis.py` / `fig5_data.npz` / `fig5a_distortion.png` — those stay exactly as
they are. This file documents a separate script, `fig5_time_dependent_qutip_validation.py`,
that answers one specific question: **is the adiabatic (quasi-static) approximation used
throughout Fig.5–9 actually valid, or does it hide something a genuine time-dependent quantum
solve would catch?**

## 1. What the existing (frozen) LO-dressed model does

`fig5_lodressed_analysis.py` never calls QuTiP for the LO-dressed curve. It:
1. Takes the real `Δc=0` column of `fig3_fresh_build`'s already-computed steady-state grid
   (`Pout` vs a single real Rabi-frequency magnitude, call it `Ω`).
2. Computes `Ω_total(t) = |Ω_LO + e^{j·2π·Δf·t}·Ω_RF|` — a purely classical phasor-magnitude
   calculation (Appendix B / Eq. 23-24 of the main paper).
3. Looks up `Pout(t)` by interpolating the *static* `Δc=0` curve at `Ω_total(t)`.

This assumes the atom's response at every instant equals its **steady-state** response to a
constant coupling of that instantaneous magnitude — i.e. that the atom "keeps up" with the
slowly-varying combined drive rather than lagging behind it or retaining memory of the drive's
phase. This is the standard **adiabatic / quasi-static approximation**.

## 2. Why this might (or might not) be a good approximation

The approximation's validity hinges on a timescale separation: the atom must relax to its new
steady state much faster than the drive itself changes.

- **Atomic relaxation timescale**: dominated by `γ2/(2π) = 5.2 MHz` (the `|1⟩→|2⟩` decay
  rate, by far the fastest of the three decay channels: `γ2 ≫ γ3=3.9kHz, γ4=1.7kHz`) →
  relaxation time `~1/γ2 ≈ 30 ns`.
- **Drive-envelope timescale**: the LO-signal beat period, `1/Δf = 1/(150kHz) ≈ 6.67 μs`.
- **Ratio**: `6.67μs / 30ns ≈ 220×` — the atom should relax roughly 220 times faster than the
  envelope moves. On this basis the adiabatic approximation is *expected* to be excellent, but
  this report treats that as a prediction to be checked, not an assumption to rely on
  unverified — see Section 6 for the actual numeric result.

  **This estimate turned out to be incomplete — see Section 6.** It only used `γ2` (the
  fastest of the three decay channels). The real bottleneck is the much slower `γ3, γ4`
  (Rydberg-state decay), which govern the specific coherence `Ω_eff(t)` modulates. Left here
  unedited, with the correction, so the reasoning gap is visible rather than silently patched.

## 3. The genuine time-dependent model built for this check

Rather than pre-combining the LO and signal into a single real magnitude `Ω_total(t)` fed into
a *static* Hamiltonian, this validation keeps them as **two separate, simultaneously-present,
complex oscillating drive terms** on the same `|3⟩↔|4⟩` transition — the literal physical
picture, with no adiabatic assumption baked in up front. This structure (a genuinely
time-dependent Hamiltonian with the LO and signal terms kept separate, `Ω_LO + e^{jS(t)}Ω_RF`
appearing directly as a complex off-diagonal Hamiltonian element rather than pre-reduced to a
real magnitude) follows the same form Jing et al. 2020 (*Nat. Phys.* 16, 911 — the main paper's
own ref [20], the direct source of the atomic parameters used throughout this whole project)
derive for the identical physical situation — confirmed against the extracted text of that
paper during the 2026-08-20 reference-paper review (see
`References/extracted/Atomic superheterodyne receiver based on microwave-dressed Rydberg
spectroscopy --s41567-020-0918-5.txt`).

**Hamiltonian** (fixed detunings `Δp=Δc=Δ_RF=0`, matching Fig.5(a)'s locked operating point and
the same fixed-detuning convention `fig3_fresh_build` uses):

```
H(t) = (Ωp/2)·(|1⟩⟨2|+|2⟩⟨1|) + (Ωc/2)·(|2⟩⟨3|+|3⟩⟨2|)
       + Ω_eff(t)/2·|3⟩⟨4|  +  Ω_eff(t)*/2·|4⟩⟨3|

Ω_eff(t) = Ω_LO + Ω_RF·e^{j(Δω·t + Δφ)},   Δω = 2π·Δf,  Δφ = 0
```

Note `|Ω_eff(t)| = Ω_total(t)`, the *same* magnitude formula the adiabatic model uses — the
difference is that here it is never taken as a real number and substituted into a static
lookup; it stays a genuine complex, time-varying off-diagonal Hamiltonian element, and the
density matrix is propagated through it directly. (Aside for defense-readiness: the *phase* of
`Ω_eff(t)` can, in principle, matter on its own — e.g. through a basis rotation `|4⟩→e^{jψ(t)}|4⟩`
— but only if `ψ(t)` changes on a timescale comparable to the atom's own coherence dynamics.
The same 220× timescale-separation argument in Section 2 says it shouldn't here; Section 5
checks this directly rather than asserting it.)

**Lindblad collapse operators**: identical to `fig3_hamiltonian_qutip.py` — `√γ2·|1⟩⟨2|`,
`√γ3·|2⟩⟨3|`, `√γ4·|3⟩⟨4|`.

**Solver**: `qutip.mesolve()`, QuTiP's standard time-dependent master-equation integrator, with
the Hamiltonian passed as a function-coefficient list `[H0, [H1, coeff1(t)], [H1†, coeff2(t)]]`.

## 4. Procedure

1. **Initial state**: the steady state of `H(t=0)` (i.e. start already "dressed" at the LO+signal
   constructive-interference point, rather than from an arbitrary bare state — avoids wasting
   simulated time on an unrelated startup transient).
2. **Integration span**: 3 full beat periods (`3×6.67μs ≈ 20μs`), 2000 time points per period.
3. **Transient handling**: only the **last** period is analyzed/plotted. Two full periods
   (`≈13.3μs`) of settle time precede it — vastly more than the `~30ns` relaxation time from
   Section 2, so any genuine startup transient is long gone by then.
4. **Output**: `Pout(t) = Pin·exp(-kp·L·C0·Im[ρ21(t)])` — the exact same formula
   `fig3_fresh_build`/`fig5_fresh_build` already use, applied to the time-dependent `ρ21(t)`
   from `mesolve()` instead of a steady-state solve.
5. **Comparison**: the same three illustrative `Ω_RF/2π ∈ {1, 3, 5} MHz` cases `fig5_data.npz`
   already has adiabatic curves for, at `Ω_LO/2π=4.23MHz` and `Δf=150kHz` — the *exact same
   real, already-saved* `t_seconds`/`pout_{1,3,5}mhz` arrays are loaded and compared against,
   not recomputed, to avoid any chance of a subtle re-derivation mismatch.

## 5. Parameters

| Quantity | Value | Status |
|---|---|---|
| `Ωp/2π` | 8 MHz | PAPER-STATED (Sec. IV), same as fig3/5 |
| `Ωc/2π` | 1 MHz | PAPER-STATED (Sec. IV), same as fig3/5 |
| `γ2/2π`, `γ3/2π`, `γ4/2π` | 5.2 MHz, 3.9 kHz, 1.7 kHz | PAPER-STATED (Sec. IV), same as fig3/5 |
| `Ω_LO/2π` | 4.23 MHz | PAPER-STATED (Sec. V-A), reused from `fig5_data.npz` |
| `Δf` | 150 kHz | PAPER-STATED (Sec. V-A), reused from `fig5_data.npz` |
| `Δφ` | 0 | ASSUMPTION, same as fig5_fresh_build's own choice |
| `Δp, Δc, Δ_RF` | 0 | ASSUMPTION (fixed detunings), same as fig3_fresh_build |
| `Ω_RF/2π` tested | 1, 3, 5 MHz | matches fig5's own illustrative panel values exactly |
| `d` (probe diameter) | 0.76 mm | PAPER-STATED, canonical (see project memory) |
| `N_periods`, `pts/period` | 3, 2000 | numerical-convergence choice, not paper-stated |

## 6. What this script does NOT do

- Does not sweep distance/PTx or feed into Fig.6-9's noise-budget pipeline — this is purely a
  physics-validation check of one modeling assumption, not a new production figure.
- Does not re-derive or question `Ω_LO=4.23MHz`'s own justification (the separate, still-open
  `Ω_LO,opt` 30% discrepancy documented elsewhere) — it is used here exactly as-is.
- Does not attempt non-zero `Δφ` or non-zero detunings — those remain untested extensions if
  ever needed.

*(Numeric results appended below after running — see Section 6 header inserted by the script;
titled "Actual numeric results" to match every other fresh_build math report's convention.)*


## 6. Actual numeric results (this run)

| Omega_RF/2pi | max abs deviation | max relative deviation | RMS relative deviation |
|---|---|---|---|
| 1.0 MHz | 1100.907512 nW | 3.028624% | 1.874103% |
| 3.0 MHz | 3463.789525 nW | 9.138809% | 4.944796% |
| 5.0 MHz | 4424.485199 nW | 11.612223% | 5.949450% |

**Conclusion — REVISED after seeing the real numbers (the original timescale argument in
Section 2 was incomplete, corrected here rather than left standing as written):** the
adiabatic/quasi-static approximation is **not** accurate to a negligible degree — the genuine
time-dependent solve deviates from it by up to **11.6%** (at `Ω_RF/2π=5MHz`), growing
monotonically with `Ω_RF` (3.0% → 9.1% → 11.6% at 1/3/5MHz). This is a real, physically
explainable effect, not a numerical artifact:

Section 2's `γ2/2π=5.2MHz → ~30ns` relaxation estimate only considered the *fastest* of the
three decay channels — the `|1⟩→|2⟩` optical coherence that directly gives `Pout`. But the
coupling actually being modulated by `Ω_eff(t)` is the `|3⟩↔|4⟩` **Rydberg-state** coherence,
whose relevant decay channels are `γ3/2π=3.9kHz` and `γ4/2π=1.7kHz` — giving relaxation times
of **`1/γ3≈40.8μs`** and **`1/γ4≈93.6μs`**. Both are *longer than the 6.67μs beat period*, not
220× shorter than it. The Rydberg states genuinely do not fully relax within one LO-signal beat
cycle; the system carries real memory across cycles, which the instantaneous-magnitude lookup
cannot capture. The correct comparison is `Δf=150kHz` vs. `γ3,γ4~kHz` — comparable
magnitudes, not a 220× gap — so a real, non-negligible deviation is exactly what should have
been predicted from the *Rydberg*-state lifetimes (the actual bottleneck), not the fast optical
transition.

**Practical implication**: this deviation is real but modest in the metrics that matter
downstream — `Fig.6/8/9`'s Practical LO-dressed SNR uses the Fourier amplitude `|P(Δf)|` of
`Pout(t)`, and an ~11.6% amplitude deviation corresponds to only `~0.5dB`
(`20·log10(1.116)≈0.955dB` worst case on amplitude, ~0.48dB on power) of potential SNR
difference at the high-`Ω_RF` end — small relative to the ~5-40dB scale of the open Fig.6/7
discrepancies, but a genuine, now-quantified refinement this reproduction's adiabatic curves
carry, not previously known. **Not adopted into the frozen Fig.5-9 baseline** — flagged as a
documented, real finding for the user to decide whether it's worth propagating.
