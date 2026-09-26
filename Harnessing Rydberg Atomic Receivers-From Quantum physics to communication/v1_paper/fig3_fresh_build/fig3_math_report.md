# Fig. 3 — Math Report

Probe laser transmission `P_out` versus coupling detuning `Δc` and RF Rabi frequency `Ω_RF`.

This is a fresh, self-contained rebuild. It does not import `fig3.py` or anything from
`fig7_reinvestigation/`. Its own Hamiltonian, collapse operators, and QuTiP steady-state
calls are written from scratch in `fig3_hamiltonian_qutip.py`.

## 1. Physical picture

A cesium atom is modeled as a 4-level ladder `|1⟩→|2⟩→|3⟩→|4⟩`
(`6S₁/₂ → 6P₃/₂ → 47D₅/₂ → 48P₃/₂`):

- Probe laser drives `|1⟩↔|2⟩` — this is the field actually measured at the photodetector.
- Coupling laser drives `|2⟩↔|3⟩` — its detuning `Δc` is one of the two swept axes.
- RF field drives `|3⟩↔|4⟩` — its Rabi frequency `Ω_RF` is the other swept axis.

Sweeping the coupling laser reveals a single EIT transparency peak when `Ω_RF=0`; turning
on the RF field splits that peak into two (Autler-Townes splitting). Fig. 3 is this whole
family of spectra, stacked as a 3D surface over `(Δc, Ω_RF)`.

## 2. Governing equations (paper Eqs. 2, 4, 5, 6, 7 — used exactly as printed)

**Master equation (Eq. 5):**
`ρ̇ = -(i/ℏ)[H,ρ] + L`

No closed-form solution exists (stated explicitly by the paper); the steady state
(`ρ̇=0`) is obtained numerically.

**Hamiltonian (Eq. 6), rotating frame, angular-frequency units:**
```
H = (ℏ/2) *
[  0      Ωp        0              0          ]
[  Ωp    -2Δp       Ωc             0          ]
[  0      Ωc    -2(Δp+Δc)         Ω_RF        ]
[  0      0        Ω_RF    -2(Δp+Δc+Δ_RF)     ]
```

**Collapse (Lindblad) operators, from the decay structure in Eq. 7:**
`C2=√γ2 |1⟩⟨2|`, `C3=√γ3 |2⟩⟨3|`, `C4=√γ4 |3⟩⟨4|` — a cascade `|2⟩→|1⟩→`, `|3⟩→|2⟩`, `|4⟩→|3⟩`.

**Steady state:** `ρ_ss = steadystate(H, [C2,C3,C4])` (QuTiP `direct` method — solves the
16×16 Liouvillian superoperator's null space, a linear-algebra problem, not a stochastic
simulation).

**Coherence extracted:** `ρ21 = ⟨2|ρ_ss|1⟩`

**Susceptibility (Eq. 4):** `χ = C0·ρ21`, `C0 = -2·N0·℘12 / (ε0·ℏ·Ωp)`

**Output power (Eq. 2):** `P_out = P_in · exp(-kp·L·Im(χ))`, `kp = 2π/λp`

**Input power (Eq. 3):** `P_in = π/(2η0) · (d·Ωp·ℏ / (2·√℘12))²`

## 3. Parameter values used (paper Sec. IV, all cross-checked against the user-supplied
reference parameter sheet)

| Quantity | Value | Status |
|---|---|---|
| `L` | 1 cm | PAPER-STATED |
| `N0` | `+1×10¹⁵ m⁻³` | PAPER-STATED, sign-corrected — see Flag 2 below |
| `γ1` | 0 | PAPER-STATED |
| `γ2/2π` | 5.2 MHz | PAPER-STATED |
| `γ3/2π` | 3.9 kHz | PAPER-STATED |
| `γ4/2π` | 1.7 kHz | PAPER-STATED |
| `Ωp/2π` | 8 MHz | PAPER-STATED |
| `Ωc/2π` | 1 MHz | PAPER-STATED |
| `℘_RF` | `-1443.459·e·a0` | PAPER-STATED |
| `℘12` | `(2.5·e·a0)²` | PAPER-STATED |
| `λp` | 852 nm | PAPER-STATED |
| `d` | **0.76 mm** | PAPER-STATED, used literally — CONFIRMED by user |
| `e` | `1.6×10⁻¹⁹ C` | PAPER-STATED rounded value — CONFIRMED by user |
| `a0` | `5.2×10⁻¹¹ m` | PAPER-STATED rounded value — CONFIRMED by user |
| `η0` | 377 Ω | PAPER-STATED |
| `ε0` | `8.854×10⁻¹² F/m` | PAPER-STATED |
| `Δp` | 0 (fixed) | ASSUMPTION — Fig.3 is a 2-variable plot over `(Δc,Ω_RF)` only; probe-on-resonance is the standard EIT convention and matches Eq.(1)'s "scanning the coupling laser" framing |
| `Δ_RF` | 0 (fixed) | ASSUMPTION — not listed as a swept or stated parameter for Fig.3; RF-on-resonance is the natural reading |

## 4. Grid

`Δc/2π ∈ [-20, 20] MHz`, 201 points
`Ω_RF/2π ∈ [0, 20] MHz`, 161 points
Total: 32,361 fresh QuTiP steady-state solves.

## 5. Flags raised BEFORE generating anything (per user instruction to flag inconsistencies first)

1. **Expected magnitude mismatch vs. the published Fig. 3.** `d=0.76mm` gives
   `Pin ≈ 39.08 µW` via Eq.(3), so the computed `P_out` surface is expected to peak around
   `~38 µW` — about 4× the paper's own plotted peak (~9.6–9.8 µW). This is a known,
   previously-investigated internal inconsistency between the paper's own Eq.(3) (evaluated
   with its own stated diameter) and its own published figure — not something this
   reproduction should silently correct. The **shape** (AT-splitting structure) is expected
   to match; the **z-axis scale** is not.
2. **`N0` sign.** The paper text as supplied renders `N0 = 1×10⁻¹⁵ m⁻³`, which is physically
   meaningless (sub-atomic density). Used `+1×10¹⁵ m⁻³` instead (matches the user's own
   reference parameter sheet), most likely an exponent-sign loss in text extraction (the
   same passage also drops the sign from `e=1.6×10¹⁹`, which should read `10⁻¹⁹`).

## 6. What this script does NOT do

- Does not import `fig3.py`, `fig4.py`, or anything under `fig7_reinvestigation/`.
- Does not fit, tune, rescale, or otherwise adjust any output to visually match the
  published Fig. 3.
- Does not use any cached `.npz` from prior work — every point is a fresh QuTiP solve.

*(Numeric results and the actual peak value will be appended below after the sweep completes.)*


## 7. Actual numeric results (this run)

- Pin = 3.907609e-05 W = 39.0761 microW
- C0 = -1.843619e-05
- Sanity check point (Omega_RF/2pi=6MHz, Delta_c=0): Pout = 33.642881 microW, Tr(rho)=1.0, hermiticity error=0.000e+00
- Grid: 161 x 201 = 32361 points
- Sweep time: 2320.4s (71.70 ms/solve average)
- Peak Pout = 38.2921 microW at Omega_RF/2pi=0.125 MHz, Delta_c/2pi=0.000 MHz
- Comparison to paper's published Fig.3 peak (~9.6-9.8 microW): ratio = 3.95x (Flag 1 in Section 5 confirmed/refuted by this number)

## 8. Companion plot: EIT/AT spectrum slices

`fig3_eit_at_slices.py` extracts 7 rows (Omega_RF/2pi = 0,2,4,6,8,12,16 MHz) directly from
the same `fig3_quantum_response.npz` grid above -- no new QuTiP solves, no fitting. All 7
requested Omega_RF values landed exactly on real grid points (0.125 MHz spacing divides
each of them exactly), confirmed by the script's own exact-match check at runtime (no
"WARNING: ... nearest available" lines were printed). Output: `fig3_eit_at_slices.png`.
Same shape as the reference slice plot (single peak at 0 MHz splitting into progressively
wider doublets), y-axis scaled ~4x higher (27-38 microW vs ~6.8-9.5 microW), consistent
with Flag 1.

## 9. Companion plots: Pout vs Omega_RF, and top-down map

Both extracted from the same `fig3_quantum_response.npz` -- no new QuTiP solves, no fitting.

`fig3_pout_vs_omegarf.py` -> `fig3_pout_vs_omegarf.png`: the Delta_c=0 column of the grid.
Min=28.1277 microW, max=38.2921 microW at Omega_RF/2pi=0.125 MHz.

`fig3_topdown.py` -> `fig3_topdown.png`: full top-down contour map. Colorbar range
27.2134-38.2921 microW. Same "V" AT-splitting structure as the reference top-down plot,
scaled ~4x on the color axis (27-38 microW vs ~6.8-9.5 microW), consistent with Flag 1.

## 10. Note on folder naming

This folder was renamed from `fig3/` to `fig3_fresh_build/` by the user (2026-08-19) to avoid
confusion with other fig3-named files elsewhere in the project. No data was lost in the
rename -- confirmed by checking all files were present at the new path before continuing.
