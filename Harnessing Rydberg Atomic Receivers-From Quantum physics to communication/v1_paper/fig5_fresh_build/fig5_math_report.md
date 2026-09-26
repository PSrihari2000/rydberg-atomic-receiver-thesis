# Fig. 5 math report — LO-dressed Rydberg atomic receiver, independent QuTiP reproduction

## 1. Paper equations used

- Eq. (2)/(4): `Pout(t) = Pin*exp(-kp*L*Im(chi(t)))`, `chi(t)=C0*rho21(t)` — the general adiabatic probe-transmission relation, Sec. II-B.
- Eq. (6): four-level Hamiltonian, with `Omega` in the (3,4)/(4,3) slot refined for the LO-dressed case.
- Eq. (23a/b)-(25): field definitions and the strong-LO envelope approximation.
- Sec. III-B (unnumbered, right before Eq. 26): `Omega_total = |Omega_LO + e^{j(2*pi*Delta_f*t+Delta_phi)}*Omega_RF|`.
- Eq. (26): closed-form `rho21` under `gamma3=gamma4=0`, `Delta_p=Delta_c=Delta_LO=0` — used for validation only (Sec. 13 below), never to generate a plotted point.
- Eq. (51)-(58) (Appendix B): `Gamma_EIT`, `Gamma` (4-level HWHM), `Abar`, and the closed-form `Omega_LO,opt` — used fresh in Sec. 10 below.

## 2. LO-dressed Hamiltonian

```
H = hbar/2 * [[0, Omega_p, 0, 0],
              [Omega_p, -2*Delta_p, Omega_c, 0],
              [0, Omega_c, -2*(Delta_p+Delta_c), Omega_total],
              [0, 0, Omega_total, -2*(Delta_p+Delta_c+Delta_LO)]]
```
with `Omega_total` occupying the (3,4)/(4,3) slot (real, non-negative magnitude — the phase is never placed directly in H, per the instruction; only the scalar magnitude enters the coupling term).

## 3. LO-dressed assumptions (verified from Sec. III-B, not carried over from Fig.3/4)

- `gamma3 = gamma4 = 0` — paper's own words: *"the states |3> and |4> are metastable... it is reasonable to set gamma3=gamma4=0."* This OVERRIDES Fig.3/4's real small gamma3=2pi*3.9kHz, gamma4=2pi*1.7kHz.
- `Delta_p = Delta_c = Delta_LO = 0` — paper's own words: *"we consider only the resonant case... this implies that their detunings are Delta_p=Delta_c=Delta_LO=0."*
- `gamma2 = 2pi*5.2MHz` — retained; the paper does not override this one for the LO-dressed case, it is SHARED (Sec. IV parameters apply to both receiver types "unless otherwise specified").

## 4. All numerical parameters

| Quantity | Value | Source |
|---|---|---|
| Omega_p/2pi | 8.0 MHz | SHARED, Sec. IV |
| Omega_c/2pi | 1.0 MHz | SHARED, Sec. IV |
| gamma2/2pi | 5.2 MHz | SHARED, Sec. IV |
| gamma3, gamma4 | 0, 0 | LO-DRESSED-SPECIFIC, Sec. III-B |
| Delta_p, Delta_c, Delta_LO | 0, 0, 0 | LO-DRESSED-SPECIFIC, Sec. III-B |
| wp_RF, wp_12 | -1443.459*e*a0, (2.5*e*a0)^2 | SHARED, Sec. IV |
| d_probe, lambda_p | 0.76mm, 852nm | SHARED, Sec. IV |
| L, N0 | 1cm, 1e15 m^-3 | SHARED, Sec. IV |
| Omega_LO/2pi | 4.23 MHz | PAPER-STATED, Sec. IV (operating value) |
| Delta_f | 150 kHz | PAPER-STATED, Sec. IV |
| Illustrative Omega_RF/2pi | 1, 3, 5 MHz | PAPER-STATED (Fig.5(b)/(c) legend) |
| Delta_phi | 0 rad | ASSUMPTION (phase-origin only, doesn't affect shape) |
| Pin | 39.076 microW | derived, Eq.(3) |
| C0 | -1.8436e-05 | derived, Eq.(4) |

## 5. Definition of Omega_total

```
Omega_complex(t) = Omega_LO + Omega_RF * exp(j*(2*pi*Delta_f*t + Delta_phi))
Omega_total(t)   = |Omega_complex(t)|
```
The modulus is part of the definition and is never removed. `Omega_complex(t)` (both real and imaginary parts) is computed and retained; only its magnitude is fed into the Hamiltonian. Verified numerically never negative (Check 3, all three curves — see script output).

## 6. Fresh QuTiP calculation method

For every `Omega_total` value in the sweep (0.05 MHz to Omega_LO+max(Omega_RF)+2MHz = 11.23 MHz, 160 points):
1. Build the LO-dressed Hamiltonian (Sec. 2 above) with that `Omega_total` in the coupling slot.
2. Solve `qutip.steadystate(H, c_ops, method="direct")` (c_ops built from gamma2 only, since gamma3=gamma4=0).
3. Extract `rho21 = rho_ss[1,0]`.

**163 independent QuTiP steady-state solves performed this run** (160 sweep points + 3 validation-check points). None of Fig.3/4's `Pout_surface` data, peak data, or any precomputed quantum-response surface was loaded or reused.

## 7. Pout calculation — verified which equation applies

Checked explicitly: Eq. (2) is the paper's general adiabatic relation, stated in Sec. II-B **before** the LO-free/LO-dressed split. The paper's own Eq. (55) derivation for the LO-dressed case starts from `Pout(t)=Pin*e^{-kp*L*C0*Im(rho21)}` — i.e. it re-derives Eq.(2)/(4) with the LO-dressed `rho21` (Eq. 26) substituted in, not a different Pout formula. **Confirmed: the same Eq.(2)/(4) relation is used for both receiver types; only rho21 differs.** This script uses `Pout = Pin*exp(-kp*L*Im(C0*rho21))` throughout, with `rho21` obtained fresh from QuTiP at every point — never the linearized Eq.(27) form for generating curves (see Sec. 12).

## 8. Linear-fit methodology

**Criterion**: expanding-window linear regression, anchored at `Omega_LO` (the paper's own operating point — the paper doesn't specify a numeric anchor, but Sec. IV-B explicitly says the linear range is judged by where "the orange line coincides with the black curve" around the operating condition, so anchoring at `Omega_LO` is the most literal reading). The window grows outward symmetrically while R² stays above a threshold; stops at the last radius meeting it.

**Threshold**: R² ≥ 0.998, chosen because it is a strict-but-round statistical bar, fixed **before** comparing the resulting range to the paper's own box — not tuned to match it.

**Why reasonable**: R² directly measures how well a straight line explains the actual simulated curve's shape in the window — a standard, transparent, reproducible goodness-of-fit statistic.

## 9. OUR measured linear dynamic range

```
[1.0344, 7.3627] MHz
slope     = -1.053982e-06 W/MHz
intercept =  4.014310e-05 W
R^2       = 0.998178
RMS residual = 8.3172e-08 W
```
This is **not** the paper's own range — it was never loaded, referenced, or targeted. It emerged purely from the fit criterion above applied to this run's fresh QuTiP curve.

## 10. Omega_LO,opt — verified, not hardcoded

Computed fresh from this run's own `Omega_p`, `Omega_c`, `gamma2` via the paper's Eq.(58) chain (Eqs. 51-54 build up to it):
```
Gamma (4-level HWHM)      = 4.7799 MHz
Abar (3-level EIT amp.)   = 0.268318
Omega_LO,opt (Eq.58)      = 2.9576 MHz
```
**This does not equal the paper's Sec. IV operating value (4.23 MHz)** — a 30.1% discrepancy. This is a known paper-internal inconsistency (also recorded independently in this project's memory from earlier analysis): Eq.(58)'s own closed-form "optimal" LO value does not match what the paper actually used for its simulation. Not resolved here — both numbers are reported, and the point labeled `(Omega_LO)_opt` in the figure uses the **operating** value (4.23 MHz), since that is the actual parameter generating the Panel (b)/(c) trajectories being cross-checked against it; using the Eq.(58) alternative would mark a point unrelated to the trajectories actually plotted.

## 11. Construction of Omega_total(t)

Panel (b) uses the exact definition from Sec. 5 above, evaluated over one full real beat period `t in [0, 1/Delta_f] = [0, 6.667us]`, 4000 points, for each of the three paper-stated illustrative `Omega_RF` values (1, 3, 5 MHz). `Delta_phi=0` throughout (assumption, phase-origin only).

## 12. Mapping from Fig.5(b) to Fig.5(c)

`Pout(t) = cubic_spline(Panel (a)'s Omega_total_MHz, Pout_MHz)` evaluated at `Panel (b)'s Omega_total(t)/2pi` values, for each of the three curves — **interpolation only**, using a cubic spline built from the 160 actually-simulated QuTiP points. Explicitly **not** the linearized Eq.(27) form, since that would assume away exactly the nonlinearity/asymmetry Panel (c) exists to show.

**Verified, not assumed** (Check 4/4b in the script): the sweep domain (0.05-11.23 MHz) was checked to fully cover all three trajectories' actual Omega_total(t) ranges before interpolating (confirmed: no extrapolation needed, `extrapolation_used=False`), and three random time-samples per curve were independently re-interpolated and confirmed to exactly match the stored Panel (c) values.

## 13. Validation against paper analytics (Check 1)

Fresh QuTiP `rho21` was compared against the paper's closed-form Eq.(26) at Omega_total/2pi = 1, 4, 8 MHz. **Magnitudes agree exactly; the sign of Im(rho21) is flipped** (relative difference of exactly ~2.000 at every point — the signature of a pure sign flip, not a numerical discrepancy). Checked which sign is physical: with `C0<0` (as defined), only `Im(rho21)<=0` gives `Pout<=Pin` (required for a passive absorptive vapor with no gain mechanism). QuTiP's sign satisfies this (every computed Panel (a) point is below Pin=39.08 microW); Eq.(26) taken literally would imply gain. **QuTiP's result was kept as solved, not flipped to match the paper** — this is reported as a likely sign/phase convention detail in the paper's own Appendix B derivation chain, not a defect in this simulation. (Consistent with the same finding from this project's earlier independent Fig.5 build, see project memory.)

## 14. Differences between our results and the paper

| | This run | Paper (visual/stated) |
|---|---|---|
| Linear dynamic range | [1.0344, 7.3627] MHz | ~[1.3, 7.2] MHz (visual estimate from Fig.5, not a stated number) |
| Omega_LO,opt marked | 4.23 MHz (operating value) | Appears near ~4.2 MHz in the figure, consistent |
| Eq.(26) sign | Im(rho21) < 0 | Literally written with Im(rho21) > 0 (see Sec. 13) |
| Strong-LO validity | Violated for Omega_RF=5MHz (ratio 1.18) | Not checked/flagged in the paper itself |
| Panel (c) shape | Real interpolation of fresh data — visible asymmetry for Omega_RF=5MHz | Same qualitative asymmetry visible in the published figure |

The overall shape and qualitative behavior (decreasing Pout vs. Omega_total, near-linear middle region, growing asymmetry for larger Omega_RF) matches the paper well; the specific numeric range and the Eq.(26) sign are the two documented, unforced differences.

## 15. Limitations

- Adiabatic assumption: Panel (c) treats the atom as reaching steady state instantaneously at each moment's Omega_total(t) (the paper's own stated approximation, Eq. 2 — "instantaneous steady-state"). A separate, earlier time-dependent QuTiP check in this project (`fig5_time_dependent_qutip_validation.py`) found this approximation deviates by 3-11.6% from a genuinely time-dependent solve, growing with Omega_RF — not re-derived here, referenced for context.
- `Delta_phi=0` is an assumption; only the phase origin of the periodic curves is affected, not their shape, since a full period is always plotted.
- The Omega_RF=5MHz illustrative curve sits outside the strong-LO regime (Eq.25's own stated validity condition) — its Panel (c) shape is genuinely more nonlinear/asymmetric as a direct, expected consequence, not an artifact.

## 16. Approximations not fully satisfied — stated explicitly, not hidden

1. **Strong-LO approximation (Eq. 25)** requires `Omega_RF/Omega_LO << 1`. For Omega_RF=1MHz: ratio=0.24 (holds reasonably). For Omega_RF=3MHz: ratio=0.71 (marginal). For **Omega_RF=5MHz: ratio=1.18 — the condition is violated** (Omega_RF is larger than Omega_LO, not smaller). This is the paper's own illustrative curve, not a case we chose adversarially.
2. **Eq.(58)'s Omega_LO,opt (2.96MHz) does not match the paper's own operating Omega_LO (4.23MHz)** used to generate every trajectory in Fig.5(b)/(c) — see Sec. 10.
3. Trajectory-vs-linear-range consistency (script Section 9): the Omega_RF=1 and 3 MHz trajectories stay **100% inside** our measured linear range for the whole period; the **Omega_RF=5MHz trajectory is only 53.8% inside** it, spending nearly half the period outside the linear regime — directly consistent with finding #1 above.
