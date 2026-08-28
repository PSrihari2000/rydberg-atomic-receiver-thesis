# *** CASE STUDY -- MASTER SUMMARY, explicitly NOT a validated result ***

Reverse-engineering exercise across every ambiguous/flagged formula and parameter found in
this project, searching for what would need to change to reproduce the paper's claimed ~44dB
Fig.6 gap and ~1500m extended coverage. Six sub-investigations, all in this folder, none
touching any existing frozen file. Bottom line up front: **one clean, mathematically rigorous
finding emerged, and it points away from the LO-dressed side entirely and toward the
Conventional receiver's own absolute level.**

## The six investigations, in brief

| # | Test | Result |
|---|---|---|
| 01 | Eq.26 sign flip (paper-literal vs QuTiP's physical sign) | Dead end -- breaks Fig.4's peak-finding entirely (unphysical gain), barely moves Fig.6's gap (28.89→30.10dB) |
| 02 | Omega_LO anchor sweep (1-9MHz, including Eq.58's "optimal" 2.9576MHz) | Dead end -- gap stays flat, 23.97-29.11dB across the WHOLE range |
| 03 | Conventional formula variants (2 new readings beyond A/B already tested) | Confirms Reading B (already used) is closest to 44dB of every variant tried; no improvement found |
| 04 | Eq.35 Ibar literal reading (|P(df)|^2 vs P0_bar) | Interesting but physically circular (noise depends on the unmeasured signal) -- gap rises to 35.62dB, worth noting but not adopted |
| 05 | Literal unsigned N0/e (testing if the sign corrections were wrong) | Confirms both corrections are MANDATORY, not optional -- literal values give zero absorption / total overflow, no usable physics |
| 06 | Joint N0 x d_probe 2D sweep, ~170 real combinations tested | **The key finding -- see below** |

## The key finding (from #06, verified to 4-5 significant figures)

Because Conventional and Theoretical LO-dressed are both *exactly* -20dB/decade parallel lines
in this model (both scale as PRx~1/d^2), the relationship between the gap-at-100m and the
extended-coverage distance is a fixed, deterministic formula, not two independent degrees of
freedom:

    coverage = conv_cross * (10^(gap_dB/20) - 1)

where `conv_cross` is Conventional's OWN 0dB crossing distance -- a quantity that depends
*only* on Conventional-side parameters (fRF, GTx, GRx, GLNA, T, B, sigma_BGN) and is
**completely unaffected by N0, d_probe, Omega_LO, or any other LO-dressed-side parameter**.

Verified against real computed data: at conv_cross=3.4346m (fRF fix applied), gap=34.8395dB
predicts coverage=186.171m -- matches the actual computed 186.1723m to 4 decimal places. Same
match at the un-fixed baseline. This is an exact relationship, not an approximation.

**Solving what conv_cross the paper's own pair (44dB, 1500m) requires:**

    conv_cross_required = 1500 / (10^(44/20) - 1) = 9.5245 m

**Our real, computed conv_cross values:**

    With the verified fRF fix (6.9458GHz):  3.4346 m  (2.77x too small)
    Without it (baseline, 3.5GHz):          6.8160 m  (1.40x too small)

Both are too small -- meaning **no amount of N0/d_probe/Omega_LO tuning on the LO-dressed side
can ever jointly satisfy both paper numbers**, because coverage is mathematically locked to
gap once conv_cross is fixed, and conv_cross doesn't depend on any of those parameters at all.
This is why the 2D sweep in #06 kept finding the same pattern: whenever coverage approached
1500m, gap had already overshot to ~52-53dB; whenever gap sat near 44dB, coverage was stuck
around 515-585m. That's not bad luck or an incomplete search -- it's a structural
impossibility given conv_cross's real value.

**A genuinely interesting wrinkle**: the verified fRF fix, while real and independently
grounded (traced to Jing et al. 2020's primary source), actually makes conv_cross SMALLER
(6.82m -> 3.43m) -- moving further from the 9.52m that would be needed for coverage
compatibility, even though it moves the gap closer to 44dB. The two paper numbers pull in
different directions under this one correction. This mirrors the earlier-documented tension
between Fig.6's gap and Fig.7's distortion-decline visibility -- another case where a single
defensible fix helps one paper-matching target while hurting another.

## What this actually points to

The real, unresolved discrepancy is on the **Conventional receiver's own absolute SNR level**,
not on anything about the Rydberg/LO-dressed model. To reach conv_cross=9.52m, Conventional's
SNR would need to be roughly 8.85dB higher than what our real, defensible formula gives at any
fixed distance -- a change to Conventional's OWN parameters (fRF, GTx, GRx, GLNA, T, B, or
sigma_BGN), not to N0, d_probe, or Omega_LO. None of the Conventional-formula variants tested
in #03 move in that direction; all of them (including the currently-used Reading B) sit at or
below the real, currently-used value's distance-implied gap. This case study did not find what
that missing ~8.85dB on the Conventional side would be -- flagged as the most promising
remaining direction if this is pursued further, but not resolved here.

## Investigation #07 -- what specific Conventional-side change would close it?

Followed the #06 lead directly: solved for the exact value of each Conventional-side
parameter (fRF, GTx/GRx, sigma_BGN, and the aperture formula's own structure) that alone would
push conv_cross from its real value up to the required 9.5245m.

    fRF alone:        needs 2.5047 GHz  (paper states 3.5GHz; Jing et al.'s real value is
                       6.9458GHz; 2.50GHz matches no known transition or reference checked)
    GTx (or GRx):      needs 5.0562 dBi (paper states 2.15dBi, a standard half-wave-dipole
                       value, not an arbitrary number)
    sigma_BGN:         needs -92.9062 dBm (paper states -90dBm; not a round number)
    Aperture formula:  dropping the /(4*pi) entirely overshoots to 24.16m (vs required 9.52m)
                       -- and this "fix" isn't plausible anyway, since A_eff=G*lambda^2/(4*pi)
                       is a rigorous antenna-theory identity, not something a paper could
                       realistically misprint the way a transition frequency can be typo'd.
    Combined with the real fRF fix: still ~8.86dB of unidentified Conventional-side headroom
                       needed, with no single candidate found that supplies exactly this.

**None of these land on anything recognizable** -- no round number, no value matching another
paper, no plausible typo pattern. Unlike fRF (where 6.9458GHz was independently verified
against Jing et al.'s own Methods section), nothing in this follow-up search turned up a real,
external justification. This confirms investigation #06's conclusion rather than resolving it:
the Conventional side is where the discrepancy structurally lives, but this case study did not
find what specifically explains it.

## Investigation #08 -- combining the two best partial leads (fRF fix + Eq.35 literal)

Combined the verified fRF fix with test #04's Eq.35 literal reading (physically circular, but
numerically the closest single LO-dressed-side reading found): gap climbs to **41.58dB** --
very close to the paper's 44dB. But coverage is only **408.42m**, confirming the #06 pattern
one more time: it doesn't matter how the LO-dressed noise floor is read, or how close gap gets
to 44dB by tuning that side -- coverage stays locked to conv_cross via the exact formula, and
conv_cross itself never moves unless a Conventional-side parameter changes. This is the
strongest gap-match found in the whole case study, and it still doesn't come with the coverage
number the paper claims alongside it.

## Investigation #09 -- which diameter matches the paper's own Fig.3 plot, and a web-search confirmation

**Diameter check.** Today's own real Fig.3 run (d=0.76mm, the paper's literally-stated value)
gives peak Pout=38.29μW -- still a 3.95x mismatch against the paper's own published Fig.3 peak
(~9.6-9.8μW), confirming a finding already on record from `fig3_fresh_build`'s own math report.
Solving for what diameter WOULD match: d=0.3825mm -- 1.0066x of 0.76mm's own radius (0.38mm).
So the paper's own published Fig.3 plot matches using the RADIUS, not the diameter it states
in its own text.

**But this doesn't help Fig.6.** Using d=0.38mm (which matches Fig.3's plot) instead of 0.76mm
(which matches Fig.3's text), combined with the verified fRF fix: gap drops from 34.84dB to
**26.68dB** -- moving further from the paper's 44dB claim, not closer. Whatever diameter the
paper's authors actually used for Fig.3 appears to be inconsistent with whatever they used for
Fig.6, if they even used the same figures/values across sections consistently at all -- another
paper-internal inconsistency, on top of the fRF/dipole-moment one already found.

**Web search confirmation (new, independent of prior project memory).** Searched for other
reproductions (none found -- no public GitHub code exists for this paper) and for the origin of
the "1500m" claim. Found: Gong et al. 2412.05554 (same author group as our main paper) states
*"At a distance of 1500m with a pathloss exponent of 2.0, the amplitude of the received RF
signal is -71.8dBV/m"* -- used there as a fixed INPUT for an unrelated Doppler-broadening-free
analysis, not a derived 0dB-SNR crossing point. This independently reconfirms (via a fresh web
search, not just re-reading prior memory) the earlier-documented suspicion that our paper's
Fig.6 "~1500m" may be reused from this companion paper rather than derived from Fig.6's own
stated parameters -- now supported by TWO independent lines of evidence: this web-search
finding, and #06's own mathematical proof that 1500m is unreachable under Fig.6's own model
regardless of parameter tuning.

## Investigation #10 -- Fig.4's saturation mechanism (two-peak fit vs find_peaks)

Tested whether a genuine two-Lorentzian fit (matching Jing et al. 2020's own real
methodology -- a fit, not simple peak-counting) reproduces the paper's described "saturation
around 5.5MHz" better than our find_peaks()-based approach. It does not: the fit curve rises
smoothly from near-zero with no plateau near 5.5MHz, tracking slightly below the theoretical
separation=Omega_RF line throughout. Both methods land in the same general neighborhood as our
existing 5.125MHz threshold, not at a distinctly different, cleanly-explained value. The paper's
"~5.5MHz" remains unexplained by either literal peak-finding method available to us -- most
likely an unstated detail of their specific numerical procedure, not a criterion we're missing.

## Investigation -- LO-free ceiling vs. the paper's own Fig.6/7/9(c) text (closed, not a lead)

A promising-looking visual read of Fig.7/9(c)'s images suggested the paper's own LO-free
transition might sit well below our algebraic 46.02dB ceiling. Checked rigorously instead of
trusting the pixel-read: the paper's own text states an "~25dB SNR advantage" for LO-free over
Conventional at "close proximity (e.g. dTx-Rx<10m)". Computing Conventional's real SNR at d=5-10m
and adding 25dB gives an implied LO-free ceiling of 41.7-47.7dB -- bracketing our 46.02dB
closely. This lead does not hold up; the visual impression was an imprecise read of a small
image, not a real discrepancy. Reported as a closed dead end, not chased further.

## Investigation #11 -- exhaustive remaining single-parameter and joint Conventional-side sweep

Completed the systematic single-parameter search from #07 (which covered fRF, GTx/GRx,
sigma_BGN) by testing the 3 remaining untested Conventional-side inputs:

    GLNA alone:  needs 22.9062dB (paper states 20dB) -- a +2.91dB change, the SMALLEST
                 single-parameter change found across the entire case study
    T alone:     no solution in any physically sane range (confirms sigma_BGN dominates
                 Conventional's noise floor by orders of magnitude, as found earlier)
    B alone:     no solution in any physically sane range

Then ran a JOINT optimization across all 4 free Conventional-side parameters simultaneously
(fRF, GTx=GRx, GLNA, sigma_BGN), minimizing total relative deviation subject to hitting the
required conv_cross=9.5245m exactly. Result: fRF -9.58% (3.50->3.16GHz), GTx +0.20dB, GLNA
-1.01dB, sigma_BGN -1.02dB -- a combination where every individual change is modest (all under
10%), unlike any single-knob solution found elsewhere in this case study. Still, no external
evidence supports this specific combination over any other combination that would also solve
the same one equation (conv_cross has 4 free inputs and only 1 constraint -- infinitely many
combinations satisfy it; this is the "smallest total deviation" one by construction, not a
uniquely correct one). Unlike fRF's real fix (independently verified against Jing et al.'s
primary source), nothing here is externally corroborated.

Omega_p/Omega_c sensitivity (LO-dressed side) was not run as a fresh QuTiP sweep -- #06 already
proves coverage cannot move independently of conv_cross (a Conventional-only quantity), so this
branch could only affect gap (already known reachable, e.g. via N0), not close the coverage gap.
Skipped the expensive new sweep since the structural result already answers the question.

## Overall verdict

No parameter or formula variant tested across all six investigations reproduces both of the
paper's headline Fig.6 numbers, and the joint-sweep analysis shows *why* not, precisely, rather
than just reporting more failed attempts: the two numbers are locked together by a fixed
formula once Conventional's own crossing point is set, and every real, defensible calculation
of that crossing point comes out too small by a factor of 1.4-2.8x. This is a genuine, honest,
well-quantified dead end for the LO-dressed-side parameters -- and a clear, specific pointer
toward where a real explanation (if one exists) would have to live: the Conventional receiver's
own absolute noise/signal budget, not the atomic/Rydberg side of the model at all.

Nothing here is adopted into any frozen baseline. This entire folder is a documented,
labeled case study only.
