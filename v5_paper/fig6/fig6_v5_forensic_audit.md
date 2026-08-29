# Fig. 6 (v5 paper) — Forensic Audit (Step A, read-only)

Source: `References/v5_text.txt` + rendered PDF pages 11-12 (pymupdf,
not the garbled `pdftotext` extraction, which mangles subscripts/
superscripts on this page badly).

## 1. Equation map (Fig.6-relevant only)

| Eq. | Content | Relevant to |
|---|---|---|
| 28 | Ω_total definition context (referenced, not re-derived here — see Sec.6 below, real form confirmed as the phasor Ω_total(t)=Ω_LO+Ω_RF·e^{jθ(t)}) | Panel (b) |
| 53 | Pout = Pin·e^{−κ(Ω_total,Δ)} (restructured probe transmission) | Panel (a)/(c) functional form |
| 54 | Taylor expansion of κ(Ω_total,Δ) around Ω_LO in powers of δ=Ω_total−Ω_LO: κ₀, κ₀', κ₀'' | LDR derivation |
| 55 | Pout expanded to 3rd order: desired term (∝κ₀'·ΩRF·cosθ), 2nd-order distortion term (∝κ₀''·ΩRF²), 3rd-order distortion term (∝κ₀'''·ΩRF³·cos3θ) | LDR derivation |
| 56 | THD ε defined via ratio of harmonic amplitudes (cites external ref [57] for the THD definition itself) | ε definition |
| 57 | Ω″RF,max = 4|κ₀'/κ₀''| (upper bound when 2nd-order distortion is the limiting term) | LDR upper bound (off-optimum) |
| 58 | Ω‴RF,max = √(6ε·\|κ₀'/κ₀'''\|) (upper bound when 2nd-order is nulled, i.e. at Ω_LO,opt, so 3rd-order dominates) | LDR upper bound (at optimum) |
| 75-78 | Appendix C: Λ(a,b)=b²/(b²+a²) Lorentzian model; Γ⁽³⁾_HWHM (Eq.77, three-level); Γ (Eq.78, four-level, = HWHM parameter *b* of Λ(Ω_LO,Γ)) | Γ definition |
| 79 | Ω_LO,opt = Γ/√3 (equivalently (√3/3)Γ — same number, two ways of writing it) | Analytic optimum |

## 2. Linewidth convention — resolved, not re-inherited from Fig.4/5

**Confirmed: Γ in Eq.79 is the Appendix C four-level Γ (Eq.78), an HWHM,
and is NOT the same quantity as Fig.4/5's measured Γ_ref=1.2231MHz.**

Evidence:
- Eq.78's Γ is explicitly introduced as the HWHM parameter *b* inside
  Λ(Ω_LO,Γ)=Γ²/(Γ²+Ω_LO²) — i.e. it's the width of the LO-dressed
  Lorentzian as a function of Ω_LO specifically, a different physical
  role than Fig.4's ΓFWHM (which normalizes ΩRF in the LO-*free* Δc-domain
  context, Eq.48).
- Numerically: Γ (Eq.78) = 7.3255 MHz (computed from Ωp=8MHz, Ωc=1MHz,
  γ2=5.2MHz — paper-stated, already validated in this project's earlier
  work). Ω_LO,opt = Γ/√3 = 7.3255/1.7321 = **4.2294 MHz**, matching the
  paper's own stated "Ω_LO,opt=4.23MHz" to within rounding — strong
  confirmation this is the correct Γ.
- Fig.4/5's Γ_ref (1.2231MHz) was deliberately *not* used here — using
  it would give Ω_LO,opt=1.2231/√3=0.706MHz, nowhere near the paper's
  stated 4.23MHz, confirming the two Γ's are genuinely different
  quantities and must not be conflated.

## 3. Parameters used (all already validated in Fig.3, reused unchanged)

Ωp/2π=8MHz, Ωc/2π=1MHz, γ2/2π=5.2MHz, γ3/2π=3.9kHz, γ4/2π=1.7kHz,
Δp=Δc=Δ_RF=0 (all on-resonance — Fig.6 sweeps coupling *strength*, not
detuning), N0=4.89×10¹⁶m⁻³, ℘12=(2.5·e·a0)², λp=852nm, L=1cm,
Pin=20.7µW direct. **No Natoms, Aeff, PTx, GTx, or distance anywhere** —
confirmed Fig.6 is a pure intrinsic atomic-response calculation, not a
link-budget one (matches the prompt's Check 5).

## 4. Hamiltonian confirmed

Same 4-level Hamiltonian as `fig3_v5_qutip.py`, with the |3⟩↔|4⟩
coupling term set to Ω_total instead of Ω_RF, and Δp=Δc=Δ_RF=0 throughout
(already implemented this way in `fig6_v5_qutip.py`, built before this
audit — consistent with what the audit confirms is correct).

## 5. ε (THD tolerance) — genuinely undefined, confirmed by the paper's own wording

Full-text search for "epsilon"/"THD"/"distortion tolerance"/"harmonic
distortion" found the exact defining sentence (page 11): *"Above this
limit, the harmonic distortion exceeds **the designer-specified
tolerance ε**, although the detector may still operate with
post-calibration at the expense of linearity."*

**The paper itself calls ε "designer-specified"** — i.e. it is explicitly
framed as a free engineering choice, not a fixed physical constant the
authors computed or measured. There is no numeric value for ε anywhere
in the paper. Per the prompt's own instruction: **STOPPING here — the
upper LDR bound Ω‴RF,max=√(6ε|κ₀'/κ₀'''|) (Eq.58) cannot be assigned a
specific number without choosing ε ourselves.** This will need to be
either (a) presented as a formula with ε left symbolic / swept over a
few illustrative values, clearly labeled as our choice, or (b) omitted
from the shaded LDR band entirely, showing only what's paper-confirmed.

## 6. ΩRF,min — no explicit equation found, only qualitative description

Exact quote (page 11): *"the upper bound of the linear dynamic range is
determined by Ω″RF,max (or Ω‴RF,max), shown in (57)-(58). By contrast,
its **lower bound corresponds to the intrinsic sensitivity** of the
LO-dressed Rydberg atomic receiver."*

No equation number is given for ΩRF,min. "Intrinsic sensitivity" suggests
a noise-floor/SNR-based quantity (analogous to Emin for the LO-free
receiver, Eq.51) rather than a distortion-based algebraic bound like the
upper limit — but the paper never writes this out as a formula for
ΩRF,min specifically anywhere found in this audit. **Classified: paper
supports only a qualitative description, not a reproducible closed-form
value.** Not invented here.

## 7. Ω_total(t) — the exact form used for Panel (b)

Confirmed (page 10, Sec.IV-B-2, and consistent with Eq.28's context):

  Ω_total(t) = Ω_LO + Ω_RF · e^{jθ(t)},   θ(t) = 2π·Δf·t + Δφ

This is the **complex phasor form** (magnitude |Ω_total(t)|² =
Ω_LO²+Ω_RF²+2Ω_LOΩ_RF·cos(θ(t))), not the plain real-cosine approximation
— confirmed against the same phasor form already validated in this
project's earlier v1 Fig.5 work (documented in
`project_rydberg_paper_findings` memory: a plain real offset-cosine was
tried first and failed to reproduce the paper's beat-pattern shape; only
the phasor-sum form matched). Using the plain-cosine approximation here
would repeat a mistake already found and corrected once before.

**Δf = 15 kHz is paper-stated** (Sec.V-A: "the frequency difference is
given by Δf=15kHz, comfortably below the 3-dB bandwidth"). **Δφ has no
numeric value anywhere in the paper** — confirmed by full-text search.
Any Δφ used will be labeled explicitly as OUR ILLUSTRATIVE CHOICE (a
natural default, Δφ=0, since the paper never assigns it meaning as a
swept/critical parameter).

## 8. Panel-by-panel interpretation

- **(a) Pout vs Ω_total**: genuine QuTiP sweep (already built), analytic
  Ω_LO,opt=4.23MHz (validated), numeric Ω_LO,opt from real
  |dPout/dΩ_total| (already built, found at 1.31MHz — see Section 9 below
  for the honest discrepancy this revealed), LDR shading blocked on the
  ε gap (Section 5).
- **(b) Ω_total vs time**: |Ω_LO + ΩRF·e^{jθ(t)}| using the confirmed
  phasor form, Δf=15kHz paper-stated, Δφ=0 as our labeled illustrative
  choice, Ω_LO at whichever optimum we choose to demonstrate, ΩRF at an
  illustrative in-LDR value.
- **(c) Pout vs time**: Pout(t) = [real Fig.6(a) QuTiP curve] evaluated
  at Ω_total(t) from panel (b), via interpolation on the genuine dense
  curve — no analytic approximation, no re-solving QuTiP per time point
  (the static curve already captures the full nonlinear response;
  interpolating it is equivalent and much cheaper, same principle as
  Fig.5's amplitude lookup).

## 9. Honest finding already surfaced (from the pre-audit build, folded in)

The real QuTiP Pout(Ω_total) curve collapses far more sharply (7.4µW→
~0.0002µW by 10MHz) than the paper's own plotted curve (~8.3→2µW over
the same range) — consistent with the same high-N0 sharpening pattern
already documented for Fig.3/4/5. Consequence: the genuine numeric
Ω_LO,opt (steepest real slope, 1.31MHz after excluding a finite-
difference edge artifact near Ω_total=0) does not land near the paper's
stated 4.67MHz, even though the **analytic** Ω_LO,opt (N0-independent)
matches beautifully. This is a real, reportable gap, not a bug — same
category as every other real/paper shape mismatch found this session.

## 10. Implementation plan (pending your review before proceeding)

1. Panel (a): keep the already-built real QuTiP curve + both Ω_LO,opt
   lines (no changes needed, already validated). LDR shading: present
   Ω‴RF,max as a formula with ε swept over a few illustrative values
   (e.g. 1%, 5%, 10%), explicitly labeled as illustrative, NOT a single
   paper-confirmed shaded band. ΩRF,min: omitted from the plot (no
   equation to compute it from) — noted as a documented gap instead of
   invented.
2. Panel (b): build Ω_total(t) via the confirmed phasor form, Δf=15kHz,
   Δφ=0 (labeled ours), Ω_LO=analytic optimum (4.23MHz, paper-matching),
   ΩRF at one illustrative in-LDR value.
3. Panel (c): interpolate the real Panel (a) curve at Ω_total(t) from
   panel (b) — no new QuTiP calls, no analytic substitute.
4. Combine into one 3-panel figure matching the paper's layout.
5. Math report separating paper-stated / derived / genuine-QuTiP /
   illustrative-assumption quantities, per the requested A/B/C/D split.

Stopping here per your instruction — let me know if this plan is
approved before I generate the final plot code.
