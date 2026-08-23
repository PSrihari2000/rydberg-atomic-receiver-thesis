# Fig. 5 — Math Report

Distortion of the LO-dressed Rydberg atomic receiver. Three panels: (a) quantum procedure,
(b) input, (c) output.

## 1. Data source for panel (a) — REUSED, not recomputed

Per paper Sec. III-B ("Ω in (6) is further refined as Ω_total"), evaluating `P_out` at fixed
`Δc=0` as a function of `Ω_total` is mathematically identical to `fig3_fresh_build`'s own
`Δc=0` column (Ω_total simply occupies the same Hamiltonian slot as `Ω_RF` did in Fig.3).
Loaded from `../fig3_fresh_build/fig3_quantum_response.npz`, restricted to the paper's
plotted range (0-14 MHz). **No new QuTiP solves.**

## 2. Equations

**Panel (a)** — real QuTiP data (via reuse above), no closed form needed for the actual
curve.

**Panel (b) / (c)** — the exact phasor-magnitude formula (paper Sec. III-B, confirmed against
the user-supplied screenshot of the boxed equation):
```
Ω_total = |Ω_LO + e^{j(2πΔf·t+Δφ)}·Ω_RF|
        = √(Ω_LO² + Ω_RF² + 2·Ω_LO·Ω_RF·cos(2πΔf·t+Δφ))
```
Panel (c)'s `P_out(t)` is obtained by REAL interpolation of panel (a)'s data at the
`|Ω_total(t)|` values from panel (b) — not the linearized Eq.(27) formula, since panel (c) is
specifically meant to show departure from linearity outside the linear dynamic range.

**Linear dynamic range** (paper gives no formula, only "where the orange line coincides with
the curve"): operating-point-anchored linear fit, R² ≥ threshold (see Section 8 for the
threshold sensitivity check and the final choice of 0.998), anchored at `Ω_LO=4.23MHz`,
window expanded outward until R² drops below threshold.

## 3. Panel layout (CORRECTED after checking the user's screenshot of the actual paper figure)

- (a): x = `Ω_total/2π` (MHz, 0-14), y = `P_out`
- (b): x = `|Ω_total|/2π` (MHz, 0-14, SAME scale as (a), stacked directly below it), y = `t`
  — axes transposed relative to the "obvious" reading; this is why each curve folds into a
  "hook" shape (one `Ω_total` value maps to two `t` values per period).
- (c): x = `t`, y = `P_out`

## 4. Parameters

| Quantity | Value | Status |
|---|---|---|
| `Ω_LO/2π` | 4.23 MHz | PAPER-STATED |
| `Δf` | 150 kHz | PAPER-STATED |
| `Δc` | 0 (fixed) | PAPER-STATED |
| `Δφ` | 0 | ASSUMPTION |
| Illustrative `Ω_RF` (panels b/c) | 1, 3, 5 MHz | INFERRED interpretation of the plotted labels (see prior discussion) |
| Time range | one full real period, `t ∈ [0, 1/Δf] ≈ [0, 6.667µs]` | Uses the REAL physical period, NOT the paper's plotted `0-3×10⁻¹⁰s` axis (a known, unresolved ~10⁴x inconsistency in the paper itself — flagged, not silently rescaled) |

## 5. What this script does NOT do

- No new QuTiP solves (reuses Fig.3's verified `Δc=0` column).
- Does not import old `fig5.py` or anything from `fig7_reinvestigation/`.
- Does not rescale the time axis to match the paper's plotted (physically inconsistent)
  numbers.
- Does not use the linearized Eq.(27) for panel (c) — uses real interpolation of real data.

*(Numeric results appended below after running.)*


## 6. Actual numeric results (superseded by Section 8's threshold change; kept for history)

Original run used R^2>=0.995 -> [0.5000, 8.0000] MHz. Superseded below.

## 7. Panels (b)/(c) DEFERRED (user decision, 2026-08-20)

Only panel (a) is presented as the current deliverable (`fig5a_distortion.png`). The user
asked to drop (b)/(c) from the result for now, citing the ambiguity already on record for
them:
- The time-axis mismatch (Section 4): paper's plotted `0-3e-10s` vs. the real physical
  period `6.667us`, a ~22,222x discrepancy with no resolution found anywhere in the paper.
- `Delta_phi=0` is an ASSUMPTION, not paper-stated.
- The illustrative `Omega_RF=1,3,5MHz` labels are an INFERRED interpretation of the
  published Fig.5(b)/(c) labels, not a paper-stated operating point (Sec.V-A's actual stated
  value is `Omega_RF=20kHz`, never plotted).

The underlying data for panels (b)/(c) is still computed by this script and saved in
`fig5_data.npz` (`omega_total_1mhz/3mhz/5mhz`, `pout_1mhz/3mhz/5mhz`, `t_seconds`) in case
they are revisited later -- it is simply not rendered as part of the current plot. The
straight-line "Linear dynamic range" overlay in panel (a) was also corrected in this pass:
it is now the actual fitted line (`slope*x+intercept`), not the real curve re-colored, so it
visibly diverges from the gray data at the edges of the fitted window -- matching the
paper's own visual convention.


## 8. Threshold sensitivity check + final numeric results (user request, 2026-08-20)

R^2 threshold sensitivity (computed on the real static curve, no refitting to any target):

| R^2 threshold | Range (MHz) | Actual R^2 | # points |
|---|---|---|---|
| 0.999 | [1.625, 6.875] | 0.999035 | 43 |
| 0.998 | [1.250, 7.250] | 0.998248 | 49 |
| 0.995 (original choice) | [0.500, 8.000] | 0.995148 | 61 |
| 0.99 | [0.125, 8.375] | 0.992455 | 67 |
| 0.98 and looser | [0.000, 8.500] | 0.986978 | 69 (hits array edge Omega_total=0, not a true R^2 plateau) |

Center of the range is stable (~4-4.5 MHz, near Omega_LO) across all thresholds; the exact
edges shift by 1-2 MHz depending on the threshold. Switched the primary result from
R^2>=0.995 to **R^2>=0.998** (stricter, still round, gives a tighter/cleaner presentation) --
this was NOT chosen to reproduce the paper's own box pixel-for-pixel, it was chosen before
comparing shapes, purely for a stricter statistical criterion.

Final numeric results (R^2>=0.998):
- Linear dynamic range: [1.2500, 7.2500] MHz, R^2=0.998248
- P0_bar = Pout(Omega_LO) = 35.4032 microW
- Real period (1/Delta_f) = 6.6667 microseconds (paper's plotted axis reads 0-3e-10s, a ~22222x mismatch, not silently corrected)

**Shading fix**: the "linear dynamic range" box is now bounded by the FIT LINE's own y-range
at its two x-edges (a tight rectangle), not the full plot height -- matches the paper's own
box style, which hugs the linear segment rather than spanning axis-to-axis.

**Caveat, restated from the chat discussion**: R^2 is a statistical statement about how well
a straight line explains the STATIC curve's shape -- it is not a direct measurement of how
accurately the receiver's actual dynamic signal readout (Eq.27-29) would work in this range.
That would require the deferred panels (b)/(c) analysis (comparing the true Omega_RF against
the Fourier-extracted apparent Omega_RF for various assumed signal strengths). Indirect
support that a real breakdown region exists (not just a statistical artifact) comes from
this project's earlier Fig.6/7 work, where the "practical" (real, nonlinear) LO-dressed
SNR/MI curves genuinely peak and decline once pushed outside roughly this same region, while
the "theoretical" (linear-everywhere) curves do not.

## 9. Panel (b) — Input, `Ω_total(t)` vs `t` (`fig5b_input_plot.py`)

**Un-deferred 2026-08-20**: panels (b)/(c) were set aside in Section 7, then revisited
per-panel with their own math-first review, per the user's request.

**Equation** (paper Sec. III-B, verified character-for-character against a user-supplied
screenshot of the boxed equation):
```
Ω_total(t) = |Ω_LO + e^{j(2πΔf·t+Δφ)}·Ω_RF|
           = √(Ω_LO² + Ω_RF² + 2·Ω_LO·Ω_RF·cos(2πΔf·t+Δφ))
```
Pure closed-form equation evaluation — no QuTiP, no atoms involved in this panel at all (it
is classical two-tone beating, upstream of anything atomic; the atomic response only enters
in panel (a)/(c)).

**Parameters**: identical to panel (a) — `Ω_LO=4.23MHz`, `Δf=150kHz` (both PAPER-STATED),
`Δφ=0` (ASSUMPTION, justified: only shifts the phase origin of a periodic curve, not its
shape, and we plot a full period regardless), `Ω_RF=1,3,5MHz` (INFERRED interpretation of the
paper's illustrative labels), `t ∈ [0, 1/Δf] ≈ [0, 6.667µs]` (real physical period, NOT the
paper's inconsistent `0-3×10⁻¹⁰s` plotted axis — see the chat discussion: that axis span is
incompatible with `Δf=150kHz` by a factor of ~22,222x, most plausibly traced to a mix-up with
`1/fRF = 2.857×10⁻¹⁰s`, the RF *carrier* period, a different paper-stated number entirely;
speculative, not confirmed against the authors' own code).

**Axis convention — DELIBERATELY DIFFERENT FROM THE PAPER**, per explicit user request:
`x = |Ω_total|/2π`, `y = t`. This matches the paper's own transposed layout (confirmed via
user screenshot) and is why the curves fold into a "hook" shape (one `Ω_total` value maps to
two `t` values per period) — a genuine consequence of the real equation plotted this way, not
a forced/traced match.

**No new computation**: reuses `omega_total_1mhz/3mhz/5mhz` and `t_seconds`, already computed
and saved in `fig5_data.npz` by `fig5_lodressed_analysis.py`'s original run.

**Result**: same "C"/hook shape as the paper's published Fig.5(b), each curve's horizontal
extent set by `Ω_LO±Ω_RF` (e.g. `Ω_RF=5MHz` curve spans `[0.77,9.23]MHz`, `Ω_RF=1MHz` spans
`[3.23,5.23]MHz`) — wider `Ω_RF` gives a wider swing, visibly confirmed in the plot.

## 10. Panel (c) — Output, `P_out(t)` vs `t` (`fig5c_output_plot.py`)

**Not a new physics equation — function composition** of panel (a) and panel (b), both
already real:
```
P_out(t) = f( Ω_total(t) )
```
where `f` = panel (a)'s real QuTiP static response curve (cubic-spline interpolated), and
`Ω_total(t)` = panel (b)'s real phasor-magnitude values, at the SAME `t` grid.

**Explicitly NOT using the paper's Eq.(27)** linearized form
(`P_out(t) = P̄₀+κΩ_RF·cos(2πΔf·t+Δφ)`) — that is a small-signal approximation valid only
inside the linear region (Section 8's `[1.25,7.25]MHz`), and would assume away exactly the
nonlinear distortion this panel exists to show. Real interpolation of the real static curve
is used instead, so genuine nonlinearity (e.g. for `Ω_RF=5MHz`, whose `Ω_total(t)` swings
down to `0.77MHz`, well outside the linear region) shows up as visible curve asymmetry, not
by construction.

**Parameters**: identical to panels (a)/(b), nothing new.

**Axis convention**: `x=t`, `y=P_out` — matches the paper's own panel (c) layout directly (no
transposition needed here, unlike panel (b)).

**No new computation**: reuses `pout_1mhz/3mhz/5mhz`, already computed and saved in
`fig5_data.npz` by the original `fig5_lodressed_analysis.py` run (the script always computed
all three panels' data; only the rendering was deferred).
