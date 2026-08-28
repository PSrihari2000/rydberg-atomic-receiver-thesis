# *** CASE STUDY -- explicitly NOT a validated result, NOT for submission ***

Reverse-engineering exercise requested to see whether the parameter value needed to hit the
paper's claimed ~44dB gap / ~1500m coverage lands anywhere near something independently
plausible (a typo, a literature value, a round number) -- a lead-generation technique, done
on top of the already-verified fRF/dipole-moment fix (real, traced to Jing et al. 2020's
primary source). Does not touch any frozen fresh_build file or the n0_density_sensitivity/
investigation; reuses only their already-computed real data via the exact N0-rescaling
identity (proven to ~1e-15 against fresh QuTiP solves in that earlier check).

## Reference points (real, no tuning)

    Baseline (N0=1e15, fRF=3.5GHz, i.e. today's actual fig6_fresh_build):
        gap=28.8864dB, coverage=182.79m, Conventional's own 0dB crossing=6.82m
    + fRF fix only (N0=1e15, fRF=6.9458GHz, the transition-consistent value):
        gap=34.8395dB, coverage=186.17m, Conventional's own 0dB crossing=3.43m
    Paper's claims: gap~44dB, coverage~1500m

## Step 2 -- solving for the N0 that hits EXACTLY 44dB (fRF fix applied)

    SOLUTION: N0 = 3.0346e15 m^-3  (3.03x the baseline, 0.087x Jing et al.'s 3.5e16)

This is not a round number, not close to any literature value checked so far, and not
obviously explainable by a decimal-point or unit-conversion typo. At this N0, coverage comes
out to only 540.9m -- badly short of the paper's ~1500m claim, and the linear range has
already shrunk to [2.16,6.24]MHz (vs baseline's [1.03,7.36]MHz).

## Step 3 -- solving for the N0 that hits EXACTLY 1500m coverage instead

**No solution exists.** Swept log10(N0) from 15.0 to 17.0 (N0 from 1e15 to 1e17, i.e. up to
~2900x the baseline and ~3x even Jing et al.'s own literature value) -- coverage never reaches
1500m anywhere in this range:

    Coverage climbs from 183m (N0=1e15) up to a PEAK of ~1338m around N0=1.4-1.6e16,
    then DECLINES again as N0 increases further, and the linear region collapses entirely
    (no kappa exists) beyond N0~2.5e16.

So 1500m coverage is not just "not yet found" -- it appears to be **structurally unreachable**
by tuning N0 alone in this model, at any value, however unrealistic.

## Verdict

No single, defensible tuning of N0 (even combined with the already-real fRF fix) reproduces
both of the paper's headline Fig.6 numbers simultaneously:
- The N0 that hits 44dB (3.03e15) badly undershoots coverage (541m vs 1500m).
- No N0 at all reaches 1500m coverage -- it caps out around 1338m before the model breaks down.
- The "required" N0 values in both directions are not recognizable/round numbers, not close to
  Jing et al.'s real literature value, and not explainable by any obvious typo pattern already
  found elsewhere in the paper (unlike fRF, where a specific, independently-documented
  inconsistency was found and verified against a primary source).

This is consistent with (not new evidence against, but not resolving) the earlier lead already
on record: the paper's own "~1500m" figure may be a borrowed illustrative distance from a
companion paper (Gong et al. 2412.05554, same author group, uses L=1500m as a fixed simulation
INPUT for an unrelated comparison, not a derived 0dB crossing) rather than a number this
specific Fig.6 model, with these specific parameters, actually produces. Nothing found in this
case study contradicts that; if anything, the structural unreachability of 1500m via N0 makes
it a bit more plausible that the number came from somewhere other than a direct derivation
under this paper's own stated Sec.IV/V-A parameters.

**No lead found here worth pursuing further as a genuine explanation.** The honest conclusion
of this exercise is negative: N0 is not the missing piece, and reverse-engineering it does not
point to anything suspicious the way the fRF/dipole-moment mismatch did. Filed as a documented
dead end, not a result.
