# Fig.3 curve-sharpness mismatch — forensic audit for a hidden N0/L-style inconsistency

Triggered by: after finding the same "our curve is sharper than the
paper's plotted curve" pattern recur in Fig.3, 4, 5, AND 6, checked
whether — analogous to the already-found Pin/diameter inconsistency
(see `fig3_v5_math_report.md`) — there is a SECOND, undiscovered paper-
internal inconsistency specifically affecting the exponent's effective
optical depth (N0×L), which could explain the sharpness mismatch and,
if found, would likely fix Fig.4-10 at once since they all inherit from
the same exponential transmission formula.

## What was checked (all via rendered PDF pages / full-text search, not
garbled pdftotext, per project convention)

1. **Eq.2/Eq.4 exact functional form**: `Pout(t) = Pin·exp{-kpL·Im(χ(t))}`,
   `χ(t)=C0·ρ21(t)`, `C0=-2N0℘12/(ε0ħΩp)`. Confirmed to match our
   implementation exactly — no missing factor of 2, no alternate
   normalization of χ or kp.
2. **Rydberg-blockade / "independent-atom" correction** — the paper
   itself explicitly addresses whether raw N0 is valid to use in the
   master equation: it computes the blockade figure of merit
   Nb=ρR·N0·(4πrb³/3)≈**0.02≪1** and concludes atom-atom interactions
   are negligible, i.e. the FULL, uncorrected N0=4.89×10¹⁰cm⁻³ should be
   used directly. This is the opposite of a hidden correction — it's
   the paper's own justification for exactly what we already do.
3. **N0/L value consistency** — searched every occurrence of N0 and L in
   the paper; both appear identically (N0=4.89×10¹⁰cm⁻³=4.89×10¹⁶m⁻³,
   L=1cm) everywhere they're stated. No second/different numeric pair
   found anywhere (unlike Pin/diameter, which IS genuinely inconsistent
   — see the existing audit).
4. **Other efficiency-like parameters** — found η_eff=0.5 and PD
   responsivity D=0.55 A/W in the same parameter list (Sec.V-A). These
   belong to the photodetector/noise chain used in later SNR/mutual-
   information figures (Fig.7-9), NOT to the Pout(Δc) transmission
   formula (Eq.2) itself — confirmed irrelevant to Fig.3-6's shape, but
   flagged for reuse when building Fig.7+.

## Conclusion

**No hidden, fixable second inconsistency was found.** Every parameter
in the exponent is paper-stated, self-consistent, and (for N0
specifically) explicitly validated by the paper's own blockade
calculation. Combined with the already-established fact that our QuTiP
computation is independently cross-validated (<0.12% agreement against
an independent numpy Liouvillian solve), this means:

- Our Fig.3 computation is correct given the paper's literally-stated
  inputs.
- The sharpness/linewidth mismatch against the paper's own plotted
  curves is NOT something we can resolve by finding a different
  "correct" parameter value — there isn't one hiding in the text.
- The mismatch must originate from something in how the paper's own
  figures were actually generated that isn't recoverable from the text
  (e.g. the authors' own plotting pipeline may not literally match
  Sec.V-A's stated parameter list, similar in spirit to the already-
  documented Pin/diameter case, just for a different, unlocated
  factor) — or from a plotting-side scaling never mentioned in the
  paper. This is now a **closed line of investigation**, not an open
  gap — do not re-run this same search for Fig.7-10 without new
  evidence (e.g. a newer paper version, an author response).

This audit does not change any existing frozen result. It's a negative
finding: it rules out a candidate explanation, and confirms the
existing recurring-sharpness note (see [[feedback_deviation_metrics_and_recurring_sharpening]])
is the correct level of explanation to keep using in later figures.
