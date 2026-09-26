# Fig. 6 — exploratory findings summary

NEW file. Consolidates 9 exploratory scripts run against today's independently-rebuilt
`fig6_snr_vs_distance.py` (real Fig.4/Fig.5 data, not the older frozen `fig6_snr_distance.py`).
None of these were folded into the main build — they're comparison/verification checks, kept
separate per this project's standing convention. Every number below is a real, printed result
from running the corresponding script, not estimated.

**LO-free: formula vs. Fig.4's real measured splitting** (`fig6_lofree_qutip_measured.py`).
Feeding Fig.4's actual jagged simulated splitting into Eq.16 instead of the clean Eq.8
theoretical value gives SNR values identical to the formula-based approach to ~0.00001dB — the
46.02dB ceiling washes out the difference. The only real effect of switching is a *narrower*
usable distance window (Fig.4's own simulation only covers Ω_RF/2π up to 20MHz, so short
distances where the physical Ω_RF exceeds that lose coverage they'd otherwise have). Not worth
adopting — same values, less coverage.

**LO-free: sensitivity to ε̃** (`fig6_lofree_epsilon_sensitivity.py`). Swept 0.1%-5% (paper
uses 0.5%, cited to Sedlacek 2012's specific 2012 apparatus, not a universal constant). The
curve stays perfectly flat at every value tested — only the ceiling height moves, by exactly
-20·log10(ε̃) as expected, e.g. 46.02dB→26.02dB going from 0.5% to 5%. The cutoff distance never
moves (governed entirely by Fig.4's threshold, which doesn't involve ε̃ at all). No functional
form for how ε̃ should vary with distance/power exists in the paper or its 7 cited/reviewed
references — changing it further would be inventing an unstated relationship, not something
this project does.

**Practical LO-dressed: is the short-distance "hump" above LO-free's ceiling real?**
(`fig6_practical_convergence_check.py`). Doubled both the QuTiP sweep density (350→700 points)
and the Fourier-extraction sampling (512→2048 points) — two independent numerical refinements.
Every tested distance gave identical values to 4 decimal places under both refinements. The
hump (and the single-point NaN gap where Ω_total(t) briefly touches zero, at the exact distance
where Ω_RF crosses Ω_LO) is real, converged physics, not an interpolation or sampling artifact.

**Angle 1 — is σ²_QPN (Eq.12) actually negligible?** Natoms computed for real from N0=1e15m⁻³
and the probe beam's interaction volume: 4.5365e6 atoms. Eq.12 (Ceff=1, best case) gives
σ²_QPN=5.51e-8 — but this is a *dimensionless* number (Natoms is a pure count), while σ²_BGN and
σ²_UN are power-valued (W). Eq.15 adds them directly, which isn't dimensionally consistent as
literally written, and no conversion factor is given anywhere in the paper. This can't be
honestly verified one way or the other from what's stated — flagged as a genuine, unresolved gap
in the paper's own equation, not patched by guessing a conversion factor.

**Angle 2 — σ²_BGN: direct value or squared?** (`fig6_angle2_bgn_convention.py`). Convention A
(the main build: -90dBm converted to W and used directly, matching the paper's own text calling
it "the noise power") vs. Convention B (squaring that value again). A constant 40.1dB swing
between them at every distance. Convention B is ruled out, not just disfavoured: it makes the
Conventional receiver's SNR (+16.78dB) exceed Theoretical LO-dressed's (+5.56dB) at the
checkpoint, directly contradicting the paper's central claim that Rydberg receivers dramatically
outperform conventional ones. Convention A stands as the only self-consistent reading.

**Angle 3 — κ's sensitivity to Fig.5's R² fit threshold** (`fig6_angle3_kappa_r2_sensitivity.py`).
Re-fit Fig.5's real saved curve at R²=0.990/0.995/0.998(paper's choice)/0.999/0.9995. κ shifts
only ±5-7% across this whole range, and the downstream Theoretical LO-dressed SNR checkpoint
moves by under 1dB (4.96-5.96dB). Stable; not a meaningful lever.

**Angle 4 — does probe beam diameter affect Fig.6's gap?** (`fig6_angle4_probe_diameter.py`).
Fig.4's threshold was already known to be diameter-independent, but Fig.6's absolute SNR is not:
ran genuinely fresh LO-dressed QuTiP sweeps at d=0.76mm (paper-stated) and d=0.36mm (this
project's other diameter-check value). κ scales exactly as d² (ratio 0.2244, matching (0.36/0.76)²
to 4 decimals), so A_LO (∝κ²) scales exactly as d⁴ (ratio 0.0503, matching (0.36/0.76)⁴ to 4
decimals) — both confirmed against real QuTiP data, not just algebra. Net effect: the checkpoint
gap shrinks from 28.87dB (d=0.76mm) to 19.99dB (d=0.36mm) — a real ~9dB sensitivity. Confirms
using the paper-stated 0.76mm in the main build was the right call.

**Angle 5 — extending the wide QuTiP sweep beyond 35MHz** (`fig6_angle5_extend_domain.py`).
Pushing the cap from 35MHz to 80MHz (same Hamiltonian, just wider) recovers real coverage for
PTx=+10dBm down to d≈0.175m (was d≈0.42m), with the newly-revealed region continuing smoothly
into the already-known curve — no new artifacts. A hard floor remains even at 80MHz (d<0.175m
still uncovered), and pushing further mostly just delays the same question: at those distances
Ω_RF reaches hundreds of MHz, far outside the regime the 4-level model (Ω_p=8MHz, Ω_c=1MHz) was
built to describe.

**Angle 6 — the fRF/dipole-moment consistency fix, re-verified and applied**
(`fig6_angle6_fRF_consistency_fix.py`). An earlier session's finding, independently re-verified
here against Jing et al. 2020's own Methods section (not just trusted from memory): the paper's
dipole moment ℘RF=-1443.459 e·a0 is confirmed, verbatim, to be Jing et al.'s value for the
47D5/2→48P3/2 transition driven "at 6.94GHz" — but the main paper's own Fig.4 caption separately
states fRF=3.5GHz, without adjusting the borrowed dipole moment to a transition actually
resonant there. fRF enters this project's Fig.6 pipeline in exactly one place (the Conventional
receiver's effective aperture λRF²/(4π)) and nowhere else (LO-free/LO-dressed depend on ℘RF/ℏ
only, not fRF). Substituting fRF=6.9458GHz:

    Checkpoint gap (d=100m, PTx=-10dBm): 28.8864dB -> 34.8395dB  (+5.9531dB, exactly the
        predicted 20*log10(6.9458/3.5) shift)
    Remaining unexplained vs. the paper's ~44dB claim: 15.1136dB -> 9.1605dB

This is the single largest, best-grounded lever found across all 9 checks — real (traced to a
primary source, not invented), and specific to a single, isolated term (only the Conventional
curve moves; LO-free and both LO-dressed curves are provably unaffected).

## Net picture

Of the paper's claimed ~44dB gap, today's independently-rebuilt pipeline plus this one
well-grounded correction accounts for 34.84dB, leaving 9.16dB genuinely unresolved. Not applied
to the main build's frozen output (`fig6_snr_vs_distance.py`/`.png`/`.csv`) — kept here as a
documented, verified lead, consistent with this project's standing policy of reporting
discrepancies rather than silently folding fixes into a baseline without an explicit decision to
do so.
