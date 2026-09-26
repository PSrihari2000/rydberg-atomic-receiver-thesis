# Fig. 6 — Math Report

SNR performance versus distance `d_Tx-Rx`, for 4 receiver types: Conventional, LO-free,
Theoretical LO-dressed, Practical LO-dressed.

## 1. Data sources — REUSED, not recomputed

- **`fig4_fresh_build/fig4_classification.npz`**: real, already-verified AT-splitting
  resolvability threshold, `threshold_mhz = 5.5000 MHz`. Used to gate the LO-free curve
  (Eq.16 SNR is only physically meaningful where the underlying `Ω_RF` clears this
  threshold).
- **`fig5_fresh_build/fig5_data.npz`**: real `static_omega_mhz`/`static_pout_w` (the QuTiP
  static response curve, `Δc=0`, from `fig3_fresh_build`) and `P0_bar` (already computed).
  `κ` (the linear-region slope) is NOT saved in that file (an oversight when it was built),
  so it is re-fit here from the same real `static_omega_mhz`/`static_pout_w` arrays, using
  the identical algorithm (R²-threshold expanding-window linear fit, anchored at
  `Ω_LO=4.23MHz`) — same real data, same method, re-typed fresh in this script rather than
  importing across files, consistent with every other fresh-build script's self-contained
  style. Expected to reproduce `fig5_fresh_build`'s own numbers exactly.

**No new QuTiP solves anywhere in this figure.**

## 2. Shared link budget (Eq. 10)

```
PRx = PTx · GTx · η0 / (4π·d²)
```

## 3. Model 1 — Conventional RF receiver (Sec. V-A + Fig. 2)

```
S = PTx·GTx / (4π·d²)                    (flux density)
A_eff = λRF² / (4π)                       (paper-stated effective aperture)
Pc = S · A_eff
σ²_Conv = GRx·GLNA·σ²_BGN + 4·kB·T·B
SNR_Conv = Pc / σ²_Conv
```
`SNR_Conv = Pc|h|²/σ²_Conv` with the standard `|h|²=1` average-SNR convention (dropped,
consistent with every other receiver in this figure — Fig.2's own signal model
`y=√Pc·h·x+n_Conv` implies this directly; not a separately-numbered paper equation but the
direct, unambiguous consequence of the stated model).

## 4. Model 2 — LO-free receiver (Eq. 8-16)

```
Ω_RF = |ERF|·℘RF/ℏ                        (Eq.8)
PRx = |ERF|²                              (Eq.9)
σ²_UN = (ε̃·ERF)²                          (Eq.14)
σ²_Ry = σ²_UN + σ²_QPN + σ²_BGN            (Eq.15), σ²_QPN≈0 (paper-licensed to ignore)
SNR_Ry = PRx / (σ²_BGN + ε̃²·PRx)          (Eq.16, |h|²=1)
```
**Gated by Fig.4's real threshold**: plotted only where the resulting `Ω_RF ≥ 5.500MHz`;
elsewhere the curve is a gap (NaN), matching the paper's own "discontinues beyond about 10
meters" description (Sec. V-B) — not fabricated, a direct physical consequence of the real
threshold applied to this real formula.

## 5. Model 3 — Theoretical LO-dressed receiver (Eq. 30-37)

```
A_LO = GLNA·RL·D²·κ²·(℘RF/ℏ)²
Ī = ηeff·P0_bar/(ℏ·fp)·e                  (Eq.35, using P0_bar as the DC operating level —
                                            same resolution of the |P(Δf)| vs average-power
                                            ambiguity used throughout this project)
σ²_PSN = 2e·B·Ī                            (Eq.34)
σ²_TN = 4kB·T·B                            (Eq.13)
σ²_Ry,LO = GLNA·RL·D²·κ²·(℘RF/ℏ)²·σ²_BGN + GLNA·RL·D²·σ²_PSN + σ²_TN     (Eq.36)
SNR_Ry,LO = A_LO·PRx / σ²_Ry,LO             (Eq.37, |h|²=1)
```
`κ` and `P0_bar` are the real numbers already established by `fig5_fresh_build` (re-fit here
from the same real data, see Section 1).

## 6. Model 4 — Practical LO-dressed receiver

Uses the real nonlinear response (Fig.5(a)/(b)/(c)'s exact method) instead of the linear `κ`
approximation, swept across every distance:
```
For each distance d:
  PRx = link budget (Section 2)
  Ω_RF = |ERF|·℘RF/ℏ, |ERF|²=PRx
  Ω_total(t) = |Ω_LO + e^{j(2πΔf·t)}·Ω_RF|      (Fig.5(b)'s exact formula, Δφ=0)
  P_out(t) = interpolate(real static curve, Ω_total(t))    (Fig.5(c)'s exact method)
  |P(Δf)| = single-sided Fourier amplitude of P_out(t) at Δf     (Eq.28, new step here)
  signal = GLNA·RL·D²·|P(Δf)|²
  SNR_practical = signal / σ²_Ry,LO              (same noise floor as Model 3)
```

## 7. New parameters (paper Sec. IV/V-A, none used before this figure)

| Quantity | Value | Status |
|---|---|---|
| `GTx`, `GRx` | 2.15 dBi | PAPER-STATED |
| `GLNA` | 20 dB | PAPER-STATED |
| `RL` | 50 Ω | PAPER-STATED |
| `D` | 0.55 A/W | PAPER-STATED |
| `T` | 290 K | PAPER-STATED |
| `ηeff` | 0.8 | PAPER-STATED |
| `σ_BGN` | -90 dBm | PAPER-STATED |
| `ε̃` | 0.5% | PAPER-STATED |
| `fRF` | 3.5 GHz | PAPER-STATED |
| `B` | 1 MHz | PAPER-STATED |
| `d` sweep | 1 to 10⁶ m, log-spaced | matches paper's plotted x-axis range |
| `PTx` sets | Conventional: [-10,10]dBm; LO-free: [10,20]dBm; Theoretical/Practical LO-dressed: [-10,10]dBm | matches the paper's own Fig.6 legend exactly (verified against user-supplied image) |

## 8. What this script does NOT do

- No new QuTiP solves (reuses Fig.3/4/5's real, already-verified data/thresholds throughout).
- Does not import old `fig6.py` or anything from `fig7_reinvestigation/`.
- Does not fit, tune, or force any curve to hit the paper's claimed ~44dB gap or ~1500m
  crossing — those numbers are reported honestly, whatever this real computation gives.

*(Numeric results appended below after running.)*


## 9. Actual numeric results

- kappa (re-fit, matches fig5_fresh_build) = -9.493466e-07 W/MHz
- A_LO = 4.478045e-07, sigma_Ry_LO^2 = 2.542319e-14
- Checkpoint @ PTx=-10dBm, d=100m: Conventional=-23.3296dB, Theoretical LO-dressed=9.3799dB, gap=32.7095dB (paper claims ~44dB)
- 0dB crossing: Conventional@6.82m, Theoretical LO-dressed@294.44m, extended coverage=287.62m (paper claims ~1500m)

## 10. Legend style fix (user request, after seeing the first plot)

Switched from 4 distinct linestyles per color (hard to tell apart) to the paper's own
convention: ONE linestyle+color per receiver type; the two PTx values distinguished only by
a marker (plain line vs. same line with open circle markers). No data changed, presentation
only.

## 11. Marker-density fix (user request, after seeing LO-free/Practical curves looked missing)

Investigated with real numbers, not assumed: LO-free PTx=10dBm and PTx=20dBm curves sit at
the EXACT SAME SNR value (46.0206dB, the hard LO-free ceiling already established this
session) for their entire resolvable range -- they only differ in how far that flat segment
extends (2.24m vs 7.08m). Practical LO-dressed PTx=-10dBm/10dBm curves are confirmed
genuinely different (verified: PTx=10dBm's value at d=10m exactly equals PTx=-10dBm's value
at d=1m -- a clean one-decade shift, as expected from PRx~PTx/d^2) but nearly overlap with
each other and with the theoretical curve visually. Root cause of both being hard to see:
markevery was computed from the full 241-point array, leaving too few markers on short
resolvable segments (LO-free) to visually distinguish the two PTx values. Fixed with a
per-curve dynamic markevery (based on each curve's own count of finite/resolvable points, not
the full array) -- data unchanged, presentation only.

## 12. Axis start -- corrected to a REAL sweep extension, not fake padding (user request)

First attempt used a purely visual `set_xlim(0.4, 1e6)` with no data below `d=1m` -- user
correctly rejected this as fake padding. Fixed properly: `DISTANCE_M` itself now runs from
`d=0.1m` to `1e6m` (281 points, same ~40 points/decade density as before, extended by one
real decade), so the curves genuinely extend to `10⁻¹m` with real computed SNR values, not an
empty margin. `xlim` set to `(0.1, 1e6)` to match the sweep's own real start exactly. This
also reveals more of the Practical LO-dressed near-field behavior (the short-distance
peak/decline) that a `d>=1m` sweep was cutting off. Checkpoint numbers at `d=100m` are
unaffected (that part of the sweep is unchanged).


## 9. Actual numeric results (this run)

- kappa (re-fit, matches fig5_fresh_build) = -9.493466e-07 W/MHz
- A_LO = 4.478045e-07, sigma_Ry_LO^2 = 2.542319e-14
- Checkpoint @ PTx=-10dBm, d=100m: Conventional=-23.3296dB, Theoretical LO-dressed=9.3799dB, gap=32.7095dB (paper claims ~44dB)
- 0dB crossing: Conventional@6.82m, Theoretical LO-dressed@294.44m, extended coverage=287.62m (paper claims ~1500m)


## 9. Actual numeric results (this run)

- kappa (re-fit, matches fig5_fresh_build) = -9.493466e-07 W/MHz
- A_LO = 4.478045e-07, sigma_Ry_LO^2 = 2.542319e-14
- Checkpoint @ PTx=-10dBm, d=100m: Conventional=-23.3296dB, Theoretical LO-dressed=9.3799dB, gap=32.7095dB (paper claims ~44dB)
- 0dB crossing: Conventional@6.82m, Theoretical LO-dressed@294.44m, extended coverage=287.62m (paper claims ~1500m)


## 9. Actual numeric results (this run)

- kappa (re-fit, matches fig5_fresh_build) = -9.493466e-07 W/MHz
- A_LO = 4.478045e-07, sigma_Ry_LO^2 = 2.542319e-14
- Checkpoint @ PTx=-10dBm, d=100m: Conventional=-23.3296dB, Theoretical LO-dressed=9.3799dB, gap=32.7095dB (paper claims ~44dB)
- 0dB crossing: Conventional@6.82m, Theoretical LO-dressed@294.44m, extended coverage=287.62m (paper claims ~1500m)
