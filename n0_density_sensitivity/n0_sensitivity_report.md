# N0 (atom density) sensitivity check

Paper literally states N0 = 1e-15 m^-3 (an impossible negative exponent); this project's
established fix is a sign correction to +1e15 m^-3, used throughout fig3/4/5/6_fresh_build.
Jing et al. 2020 (the paper this main paper's own parameters are borrowed from) reports
~3.5e16 m^-3 for a comparable Cs vapor cell -- ~35x higher.

## Method: exact algebraic rescaling, not new QuTiP solves

N0 enters only through C0 = -2*N0*wp_12/(eps0*hbar*Omega_p), a classical constant applied
AFTER the quantum steady-state rho21 is solved -- rho21 itself never depends on N0. Since
Pout = Pin*exp(-kp*L*C0*Im(rho21)) and C0 is proportional to N0, this gives an exact identity:

    Pout_new = Pin * (Pout_old/Pin)^(N0_new/N0_old)

Verified against genuine fresh QuTiP spot-checks at 3 points before trusting it: relative
difference between the direct QuTiP solve and the algebraic prediction was ~1e-15 (machine
precision) at every point checked -- confirmed exact, not approximate. This let every
already-computed real Pout value (Fig.3's full grid, Fig.5's real curve) be rescaled to any
N0 with zero new QuTiP solves.

## Results

| N0 | Fig.4 threshold | Fig.5 linear range | Fig.5 kappa | Fig.6 gap @ 100m,-10dBm |
|---|---|---|---|---|
| 1.0e15 (baseline, used throughout) | 5.1250 MHz | [1.0344,7.3627]MHz (6.33MHz wide) | -1.054e-6 W/MHz | 28.8864 dB |
| 1.0e16 | 3.1250 MHz | [3.7064,4.6908]MHz (0.98MHz wide) | -5.130e-6 W/MHz | 45.1597 dB |
| 3.5e16 (Jing et al.'s real value) | 1.8750 MHz | **none found** (R²≥0.998 fails even at the smallest 3-point window) | -- | cannot compute (no kappa exists) |

(Paper's own claims for reference: Fig.4 threshold ~5.5MHz, Fig.6 gap ~44dB.)

## Verdict: not a clean improvement -- a real tension

**Fig.6's gap** does move toward the paper's 44dB at N0=1e16 -- in fact it overshoots to
45.16dB. Taken alone, this might look like "more realistic N0 = better match."

**But Fig.4's threshold moves the wrong way** -- further from the paper's ~5.5MHz, not
closer (5.125→3.125→1.875MHz as N0 increases). And **Fig.5's linear dynamic range collapses**
-- shrinking by 85% at N0=1e16, and vanishing entirely at N0=3.5e16 (Jing et al.'s own real
density). At that realistic density, under the same strict criterion used everywhere else in
this project, there is no meaningfully linear operating region left at all near Omega_LO --
which undermines the entire premise of the LO-dressed receiver design (a receiver that's
supposed to operate in a linear region, but the region has disappeared).

So a higher N0 doesn't uniformly improve the reproduction -- it trades a better Fig.6 number
for a worse Fig.4 number and a collapsing Fig.5 linear range. No single N0 value makes every
figure simultaneously agree better with the paper; this is the same kind of structural tension
already found between Fig.6's 44dB gap and Fig.7's distortion-decline visibility (see
`project_rydberg_fig6_fig7_findings.md`). Given this, N0=1e15 is not obviously "wrong" just
because it's numerically far from typical literature values -- it may simply be the value
that keeps Fig.4/5/6 self-consistent with each other, even if it doesn't match any single
external reference exactly.

Deliberately NOT searched for an N0 that hits exactly 44dB -- that would be tuning to match
the paper's headline number, which this project does not do.
