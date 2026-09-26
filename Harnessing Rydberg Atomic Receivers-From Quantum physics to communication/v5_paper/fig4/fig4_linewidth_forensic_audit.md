# Fig. 4 (v5 paper) — Linewidth Forensic Audit

## FINAL DECISION (2026-08-29, confirmed with user)

**Adopted: Γ_ref = 1.2231 MHz (measured, from our own real ΩRF=0 Pout(Δc)
row), as the standing project-wide convention for ΓFWHM**, used
consistently in Fig.4 and Fig.5. Reasoning, in order of weight:

1. **Empirically validated against the paper's own stated behavior.**
   Tested robustly across 4 dip-depth thresholds (2/5/10/20%): the
   measured value gives splitting resolving at R≈1.2-1.6, just above 1 —
   matching the paper's own claim "below R=1 unresolvable, beyond R=1
   splitting disappears." The best analytic candidate (Γ⁽³⁾_FWHM=5.22MHz)
   failed this test decisively — real splitting was already visible at
   R≈0.3-0.4, well below the paper's own claimed threshold. Matching a
   stated *behavior* outweighs matching a single descriptive word
   ("intrinsic") in the text.
2. **The best analytic candidate has a hidden 3-level-vs-4-level mismatch.**
   Γ⁽³⁾_HWHM=(Ωc²+Ωp²)/(2√(γ2²+2Ωp²)) contains no γ3 or γ4 at all — it's
   derived for an idealized 3-level system, not the real 4-level system
   (with real, nonzero γ3=3.9kHz, γ4=1.7kHz) that both our simulation and
   the paper's own QuTiP-based figures actually use. Same category of
   mismatch already found and rejected in this project's earlier v1 work.
   The measured value has no such gap — it comes directly from the real
   4-level QuTiP output.
3. **Consistency**: one ΓFWHM value used everywhere it appears (Fig.4's
   R-axis, Fig.5's Aeff and Lorentzian half-width), rather than switching
   definitions per figure.

**Standing caveat, not resolved, just accepted**: Γ_ref is a
post-absorption (measured-from-Pout) quantity, while Eq.48 calls ΓFWHM
"intrinsic" — suggesting the authors may have meant a pre-absorption
(χ-space) quantity. This tension is real; the decision above chooses
behavioral/model fidelity over the literal word "intrinsic."

---

Purpose: determine, before choosing/freezing a ΓFWHM value for Fig.4's R-axis,
what "ΓFWHM" in Eq.48 (R ≡ ΩRF/ΓFWHM) scientifically and mathematically refers
to in the v5 paper. No conclusion here is used to modify the Fig.4 plot.
This is a read-only forensic pass over `References/v5_text.txt` and the
rendered PDF pages (pymupdf, not the garbled `pdftotext` extraction).

## A. Every linewidth definition found in the paper

Full-text search of `v5_text.txt` for Gamma/linewidth/FWHM/HWHM/2 MHz
turned up these distinct definitions (page numbers from the rendered PDF):

1. **Sec.IV-B intro (page 9), Eq.48**: "we introduce a dimensionless R for
   measuring the RF coupling strength in units of the intrinsic EIT
   linewidth: R ≡ ΩRF/ΓFWHM." No formula given here or anywhere near Eq.48-50
   — ΓFWHM is used symbolically only, described only as "intrinsic EIT
   linewidth" / "the EIT linewidth ΓFWHM."

2. **Eq.17 (Appendix B context, page 5)**: "ΓFWHM denotes the full-width at
   half-maximum (FWHM) of the EIT spectrum. See Appendix B for its detailed
   derivation." (Appendix B, however, derives Aeff, not a ΓFWHM formula — it
   uses ΓFWHM symbolically too, inherited from elsewhere.)

3. **Eq.68 (Appendix A, page 15)**, cited from external ref [62, Sec.4.2]
   (not derived in this paper): ρee = (ΩRF/ΓFWHM)² / [1+4(Δc/ΓFWHM)²+2(ΩRF/ΓFWHM)²].
   Same symbol, no numeric formula — sourced from a textbook, used for a
   scattering-rate argument unrelated to Fig.4 directly.

4. **Eq.77 (Appendix C, page 16)** — the first (and only) place an actual
   closed-form linewidth formula is derived in this paper:

       Γ⁽³⁾_HWHM = (Ωc² + Ωp²) / (2·√(γ2² + 2Ωp²))

   explicitly named "the HWHM of the **three-level** EIT spectrum."

5. **Eq.78 (Appendix C, page 16)**:

       Γ = (2√2·Ωp / √(Ωc²+Ωp²)) · Γ⁽³⁾_HWHM
         = Ωp·√(2(Ωc²+Ωp²)/(2Ωp²+γ2²))   (equivalent direct form)

   explicitly named "the HWHM of a **four-level** system, which is related
   to the HWHM of the three-level EIT spectrum." This Γ is specifically the
   HWHM parameter *b* in the LO-dressed Lorentzian Λ(ΩLO,Γ)=Γ²/(Γ²+ΩLO²) used
   throughout Appendix C's LO-dressed derivation (Eq.75-83, including
   ΩLO,opt=(√3/3)Γ, Eq.79). **This Γ is tied to scanning ΩLO in the
   LO-dressed system, not to Δc in the LO-free system.**

6. **Fig.5 caption / Sec.V-B text (page 9-10)**: "...use QuTip toolkit to
   generate the EIT-AT spectrum as two Lorentzian peaks (half-width
   ΓFWHM/2) centered at detunings Δc = ±½ΩRF(dTx-Rx)." Confirms (a) ΓFWHM is
   being used as a FULL width (its half is ΓFWHM/2), and (b) peaks sit at
   Δc=±ΩRF/2 exactly, matching what we already found in our real Fig.3 data.
   Still no numeric value given.

No other "2 MHz" or equivalent numeric linewidth statement exists anywhere
in the paper. No supplementary material file was found separate from the
main PDF (all appendices A-D are inside the single accepted PDF).

## B. Analytical numerical values (paper's own formulas, paper's own stated parameters: Ωp/2π=8MHz, Ωc/2π=1MHz, γ2/2π=5.2MHz)

| Quantity | Formula | Value |
|---|---|---|
| Γ⁽³⁾_HWHM (three-level, Eq.77) | (Ωc²+Ωp²)/(2√(γ2²+2Ωp²)) | **2.6101 MHz** |
| Γ⁽³⁾_FWHM (2×above) | — | **5.2203 MHz** |
| Γ (four-level HWHM, Eq.78, LO-dressed context) | Ωp√(2(Ωc²+Ωp²)/(2Ωp²+γ2²)) | **7.3255 MHz** |
| Γ (four-level "FWHM", 2×above) | — | **14.6511 MHz** |

(Direct-formula cross-check for Γ matches Eq.78's two-step form exactly —
confirms no arithmetic error in transcribing Eq.78.)

## C. Measured QuTiP Pout linewidth (this project, not the paper's formula)

**1.2231 MHz** — measured directly from `fig3_v5_qutip_response.npz`'s real
ΩRF=0 row (half-max crossings of the actual computed Pout(Δc) curve, which
already includes the full Beer-Lambert exponential and v5's real N0=4.89e16
m⁻³). This is a genuinely different kind of quantity from all four analytic
candidates in section B — see section G.

## D. Fig.4 slope-implied linewidth

From the actual peak positions (Δc=±ΩRF/2, confirmed both analytically and
in our real data) and R≡ΩRF/ΓFWHM, algebra forces:

    Delta_c/2pi = R * (Gamma_FWHM/2pi) / 2   =>   slope = Gamma_FWHM/(2 x 2pi)

For the paper's literal caption "Δc/2π = ±R" (slope exactly 1) to hold
exactly, this REQUIRES:

    Gamma_FWHM/(2pi) = 2.0 MHz  exactly

This is **not equal to any of the four analytically-derived candidates**
in section B (2.61, 5.22, 7.33, 14.65 MHz) nor to the measured 1.2231 MHz.
Ratio of each candidate to this implied 2.0 MHz: 0.61 (measured), 1.31
(Γ⁽³⁾_HWHM), 2.61 (Γ⁽³⁾_FWHM), 3.66 (Γ four-level HWHM), 7.33 (Γ four-level
FWHM) — none close to 1.

## E. Dimensional / unit consistency check

R is dimensionless by definition (ratio of two frequencies). Δc/2π is
explicitly plotted in MHz (confirmed directly off the rendered Fig.4 image:
axis ticks -8,-4,0,4,8, labeled "Δc/2π"). The caption's literal equation
"Δc/2π = ±R" therefore equates a MHz-valued axis to a dimensionless number
— only self-consistent if a hidden unit-carrying conversion factor
(ΓFWHM/2, which must carry MHz) has implicitly been set to exactly 1 MHz
(so that Δc/2π = R×1 numerically). This is NOT a case of Δc secretly being
plotted in normalized/dimensionless units — the real image's axis ticks
(-8...8) match literal MHz values consistent with the known ΩRF/2π range
(0-12MHz gives Δc/2π=ΩRF/2π/2, i.e. 0-6MHz, extended a bit for the R-scaled
version) — so the inconsistency is real, not a mislabeled axis.

## F. Is 2 MHz paper-stated, derived, implied, or unexplained?

**Implied only, and unexplained.** It is not stated anywhere in the paper
text (section A's exhaustive search found no "2 MHz" or "2.0 MHz" linewidth
statement). It is not derivable from any of the paper's own closed-form
linewidth formulas (section B — nearest is Γ⁽³⁾_HWHM=2.61MHz, 30% off). It
is only reverse-engineered from taking the caption's "Δc/2π=±R" completely
literally and solving for what ΓFWHM would have to be. Classified: **an
implied-but-unconfirmed value, not a value we have independent evidence the
authors actually computed or used.**

## G. Can 1.22 MHz (measured Pout FWHM) legitimately be used in Eq.48?

**Conceptually questionable, on reflection — not a clean fit.** Eq.48 calls
ΓFWHM "the *intrinsic* EIT linewidth," and every analytic candidate in
section B is a property of the linear susceptibility / excited-state
population (χ or ρee), independent of N0 and Pin entirely. Our measured
1.2231 MHz, by contrast, is the width of the actual *post-Beer-Lambert*
Pout(Δc) curve — Pout = Pin·exp(-kpL·Im(χ)), which is a highly nonlinear
(exponentially sharpened) function of the same underlying χ once N0 is as
large as v5's real 4.89×10¹⁶ m⁻³. The measured value is real and
reproducible, but it answers a different question ("how wide does the
*observed transmission spike* look") than what "intrinsic EIT linewidth"
in Eq.48 most likely means ("how wide is the underlying atomic coherence
resonance"). This is a materially different regime from v1's earlier
Fig.4 work, where N0 was a tiny placeholder (1e15 m⁻³) and the medium sat
in the weak-absorption limit — there, Pout ≈ Pin(1-kpL·Im(χ)) is
approximately *linear* in χ, so a measured-from-Pout HWHM would have
closely tracked the intrinsic χ-space HWHM. At v5's real, much higher N0,
that approximation breaks down (confirmed numerically: 1.22 vs 2.61 MHz,
a genuine ~2.1x gap, not rounding noise) — so the "measure it from data"
practice that was well-justified for v1 does not automatically transfer to
v5's regime with the same justification.

## H. Final recommendation

Given the state of this evidence, none of the five candidate linewidths
(measured 1.22, Γ⁽³⁾_HWHM 2.61, Γ⁽³⁾_FWHM 5.22, Γ_4level_HWHM 7.33,
Γ_4level_FWHM 14.65) reproduces the paper's literal slope-1 caption
relationship, and the implied 2.0MHz value has no independent support.
**No single choice here is unambiguously "correct" from the paper's own
text — this is a genuine, unresolved definitional gap in the published
paper**, not a computational mistake in this project's pipeline (confirmed
separately: our real vs fast-numpy Fig.3 pipelines agree to <0.12%, and our
peak positions independently verify Δc=±ΩRF/2 exactly, matching the paper's
own Fig.5 text).

Recommendation for how to proceed (decision deferred to you, not applied
yet): the most textually-defensible *analytic* candidate, if one is to be
used, is **Γ⁽³⁾_FWHM = 5.2203 MHz** — it is (a) explicitly named a FWHM
(matching Eq.48's own "ΓFWHM" naming, unlike the HWHM-labeled candidates),
(b) derived from the three-level EIT spectrum specifically (matching the
LO-free, Δc-domain context Fig.4 actually operates in, unlike Eq.78's
four-level Γ which is explicitly tied to the LO-dressed ΩLO-scanning
Lorentzian), and (c) is a paper-internal closed-form result (Eq.77), not
sourced from an external reference like Eq.68's ρee formula. It still does
not reproduce the caption's literal slope=1, but it is the candidate with
the strongest textual justification among the five. The measured 1.2231MHz
remains a legitimate, real, reproducible number — just probably answering
a different question than the one Eq.48 is asking.
