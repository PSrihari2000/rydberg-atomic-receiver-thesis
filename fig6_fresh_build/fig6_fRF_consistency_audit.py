# ============================================================
# FIG. 6 -- TARGETED FORENSIC AUDIT (does NOT modify or regenerate
# fig6_snr_distance.py or its outputs; read-only investigation).
#
# QUESTION: the main paper (paper_text.txt) states, in two separate
# places:
#   (a) Sec.IV "parameters are taken from [12],[20],[23]... the
#       following four states |1>|2>|3>|4> in a Cs atom:
#       6S1/2->6P3/2->47D5/2->48P3/2 ... dipole moment associated
#       with |3>->|4> transition is wp_RF = -1443.459*e*a0"
#       -- ref [20] is confirmed (paper_text.txt line 996) to be
#       Jing et al. 2020, Nat. Phys. 16, 911 -- whose OWN paper
#       states this exact transition is driven by a REAL microwave
#       field "at 6.94GHz" (Methods, "We applied a local MW field
#       at 6.94GHz to resonantly drive the Rydberg transition
#       47D5/2->48P3/2").
#   (b) Sec.IV Fig.4 caption paragraph, separately: "carrier
#       frequency of the RF signal is fRF = 3.5 GHz."
#
# So the paper borrows Jing et al.'s entire atomic parameter set
# (transition scheme, decay rates, dipole moment -- all specific to
# the REAL 6.9458GHz transition) while separately modeling Fig.6's
# RF carrier at a DIFFERENT, unrelated 3.5GHz -- without adjusting
# the dipole moment to a transition actually resonant at 3.5GHz.
#
# fRF enters our fig6_snr_distance.py pipeline in EXACTLY ONE place:
# the Conventional receiver's effective aperture A_eff=lambda_RF^2/(4pi).
# It does NOT appear anywhere in the LO-free/LO-dressed SNR formulas
# (Eq.8-37), which depend on wp_RF/hbar only. So this is a single,
# cleanly isolated substitution to test.
#
# This script reuses fig6_snr_distance.py's own real data sources
# (fig4/fig5 real npz) and re-derives kappa/A_LO/sigma_Ry_LO_sq with
# the IDENTICAL method -- only fRF_Hz is varied. Provenance-classified
# throughout. Creates ONLY this file + a .txt summary; the frozen
# fig6_snr_distance.py and its .png/.md are untouched.
# ============================================================

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

OUTPUT_DIR = Path(__file__).resolve().parent
FIG4_DATA = OUTPUT_DIR.parent / "fig4_fresh_build" / "fig4_classification.npz"
FIG5_DATA = OUTPUT_DIR.parent / "fig5_fresh_build" / "fig5_data.npz"

print("=" * 70)
print("FIG. 6 -- fRF/dipole-moment CONSISTENCY AUDIT (read-only)")
print("=" * 70)

# ------------------------------------------------------------
# Physical constants -- IDENTICAL to fig6_snr_distance.py
# ------------------------------------------------------------

e_charge = 1.6e-19
a0 = 5.2e-11
hbar = 1.054571817e-34
eta0 = 377.0
kB = 1.380649e-23
c_light = 2.99792458e8

wp_RF = -1443.459 * e_charge * a0        # PAPER-STATED (Sec.IV, from ref [20]=Jing 2020)
lambda_p = 852e-9
fp_optical = c_light / lambda_p

GTx_dBi, GRx_dBi = 2.15, 2.15
GLNA_dB = 20.0
RL_ohm = 50.0
D_resp = 0.55
T_kelvin = 290.0
eta_eff = 0.8
sigma_BGN_dBm = -90.0
epsilon_tilde = 0.005
B_Hz = 1.0e6

# --- The two fRF candidates under test ---
FRF_PAPER_STATED_HZ = 3.5e9        # PAPER-STATED (Sec.IV Fig.4 caption, literal)
FRF_TRANSITION_CONSISTENT_HZ = 6.9458e9   # DERIVED: Jing et al. 2020's own real
                                            # resonance frequency for the SAME
                                            # 47D5/2->48P3/2 transition whose dipole
                                            # moment (wp_RF) and decay rates the main
                                            # paper explicitly borrows via ref [20].


def dbm_to_watts(p_dbm):
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


def db_to_linear(g_db):
    return 10.0 ** (g_db / 10.0)


GTx_lin = db_to_linear(GTx_dBi)
GRx_lin = db_to_linear(GRx_dBi)
GLNA_lin = db_to_linear(GLNA_dB)
sigma_BGN_sq_W = dbm_to_watts(sigma_BGN_dBm)

# ------------------------------------------------------------
# Reuse Fig.4/Fig.5's real data -- identical to fig6_snr_distance.py
# ------------------------------------------------------------

fig4_data = np.load(FIG4_DATA)
threshold_mhz = float(fig4_data["threshold_mhz"])

fig5_data = np.load(FIG5_DATA)
static_omega_mhz = fig5_data["static_omega_mhz"]
static_pout_w = fig5_data["static_pout_w"]
P0_bar = float(fig5_data["P0_bar"])
OMEGA_LO_MHZ = float(fig5_data["OMEGA_LO_MHZ"])


def linear_fit_score(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def find_linear_dynamic_range(omega_values, response_values, omega_lo, r2_threshold=0.998):
    center = int(np.argmin(np.abs(omega_values - omega_lo)))
    best = None
    max_radius = min(center, len(omega_values) - center - 1)
    for radius in range(5, max_radius + 1):
        left, right = center - radius, center + radius + 1
        slope, intercept, r2 = linear_fit_score(omega_values[left:right], response_values[left:right])
        if r2 >= r2_threshold:
            best = dict(slope=slope, r2=r2)
        else:
            break
    return best


fit = find_linear_dynamic_range(static_omega_mhz, static_pout_w, OMEGA_LO_MHZ, r2_threshold=0.998)
kappa_W_per_MHz = fit["slope"]
kappa_rad_per_s = kappa_W_per_MHz / (2.0 * np.pi * 1e6)
print(f"Re-fit kappa (identical method/data to fig6_snr_distance.py): "
      f"{kappa_W_per_MHz:.6e} W/MHz -- must match 32.7095dB baseline's -9.493466e-07")

# ------------------------------------------------------------
# Noise budget for Theoretical LO-dressed -- fRF-INDEPENDENT
# (identical regardless of which fRF candidate is used, since fRF
# never appears in this block)
# ------------------------------------------------------------

wp_over_hbar = wp_RF / hbar
A_LO = GLNA_lin * RL_ohm * D_resp ** 2 * kappa_rad_per_s ** 2 * wp_over_hbar ** 2
I_bar = eta_eff * P0_bar / (hbar * (2.0 * np.pi * fp_optical)) * e_charge
sigma_PSN_sq = 2.0 * e_charge * B_Hz * I_bar
sigma_TN_sq = 4.0 * kB * T_kelvin * B_Hz
sigma_Ry_LO_sq = (
    GLNA_lin * RL_ohm * D_resp ** 2 * kappa_rad_per_s ** 2 * wp_over_hbar ** 2 * sigma_BGN_sq_W
    + GLNA_lin * RL_ohm * D_resp ** 2 * sigma_PSN_sq
    + sigma_TN_sq
)


def PRx_rydberg(PTx_watts, d_m):
    return PTx_watts * GTx_lin * eta0 / (4.0 * np.pi * d_m ** 2)


def snr_lo_dressed_theoretical_dB(PTx_watts, d_m):
    PRx = PRx_rydberg(PTx_watts, d_m)
    return 10.0 * np.log10(A_LO * PRx / sigma_Ry_LO_sq)


def snr_conventional_dB(PTx_watts, d_m, fRF_hz):
    S = PTx_watts * GTx_lin / (4.0 * np.pi * d_m ** 2)
    lambda_RF_m = c_light / fRF_hz
    A_eff = lambda_RF_m ** 2 / (4.0 * np.pi)
    sigma_Conv_sq_W = GRx_lin * GLNA_lin * sigma_BGN_sq_W + 4.0 * kB * T_kelvin * B_Hz
    Pc = S * A_eff
    return 10.0 * np.log10(Pc / sigma_Conv_sq_W)


def find_0db_crossing(distance, snr):
    valid = np.isfinite(snr)
    x = np.log10(distance[valid])
    y = snr[valid]
    diff = y - 0.0
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    return 10.0 ** (x0 - y0 * (x1 - x0) / (y1 - y0))


DISTANCE_M = np.logspace(-1.0, 6.0, 281)
PTx_watts = dbm_to_watts(-10.0)
i100 = int(np.argmin(np.abs(DISTANCE_M - 100.0)))

theo_curve = snr_lo_dressed_theoretical_dB(PTx_watts, DISTANCE_M)
d_theo_0db = find_0db_crossing(DISTANCE_M, theo_curve)

print("\n" + "-" * 70)
print("NOTE (structural, applies to BOTH cases below): since both "
      "SNR_Conv and SNR_theo scale as PRx~1/d^2 identically, the dB "
      "GAP between them is mathematically distance-invariant in this "
      "model -- it is the same number at d=100m as at d=1500m or any "
      "other distance. Only the 0dB-CROSSING distances (an absolute, "
      "not relative, quantity) depend on distance.")
print("-" * 70)

results = {}
for label, fRF_hz in [
    ("A) paper-literal fRF=3.5GHz (CURRENT frozen baseline)", FRF_PAPER_STATED_HZ),
    ("B) transition-consistent fRF=6.9458GHz (Jing 2020's real resonance for the SAME dipole moment/transition the paper borrows)", FRF_TRANSITION_CONSISTENT_HZ),
]:
    conv_curve = snr_conventional_dB(PTx_watts, DISTANCE_M, fRF_hz)
    gap = theo_curve[i100] - conv_curve[i100]
    d_conv_0db = find_0db_crossing(DISTANCE_M, conv_curve)
    coverage = d_theo_0db - d_conv_0db
    results[label] = dict(fRF_GHz=fRF_hz / 1e9, gap_100m=gap, d_conv_0db=d_conv_0db,
                           d_theo_0db=d_theo_0db, coverage=coverage,
                           conv_100m=conv_curve[i100])
    print(f"\n{label}")
    print(f"  fRF = {fRF_hz/1e9:.4f} GHz -> lambda_RF = {c_light/fRF_hz*100:.4f} cm")
    print(f"  Conventional SNR @ d=100m = {conv_curve[i100]:.4f} dB")
    print(f"  Theoretical LO-dressed SNR @ d=100m = {theo_curve[i100]:.4f} dB (fRF-independent, unchanged)")
    print(f"  Gap @ 100m (== gap at ANY distance, see note above) = {gap:.4f} dB  (paper claims ~44dB)")
    print(f"  0dB crossing: Conventional@{d_conv_0db:.2f}m, Theoretical LO-dressed@{d_theo_0db:.2f}m")
    print(f"  Extended coverage = {coverage:.2f}m  (paper claims ~1500m)")

delta_gap = results[list(results.keys())[1]]["gap_100m"] - results[list(results.keys())[0]]["gap_100m"]
print("\n" + "=" * 70)
print(f"EFFECT OF THE CORRECTION: gap changes by {delta_gap:+.4f} dB")
print(f"  Remaining unexplained gap (vs paper's ~44dB), case A: {44.0 - results[list(results.keys())[0]]['gap_100m']:.4f} dB")
print(f"  Remaining unexplained gap (vs paper's ~44dB), case B: {44.0 - results[list(results.keys())[1]]['gap_100m']:.4f} dB")
print(f"  Effect on 1500m coverage claim: negligible (LO-dressed 0dB crossing point does not depend on fRF at all;")
print(f"  Conventional's own crossing point is already tiny -- {results[list(results.keys())[0]]['d_conv_0db']:.2f}m vs {results[list(results.keys())[1]]['d_conv_0db']:.2f}m -- compared to {d_theo_0db:.0f}m)")
print("=" * 70)

summary_path = OUTPUT_DIR / "fig6_fRF_consistency_audit.txt"
with open(summary_path, "w") as f:
    f.write("FIG. 6 -- fRF / dipole-moment consistency audit\n")
    f.write("=" * 70 + "\n\n")
    f.write("SCOPE: targeted, read-only forensic check. Does not modify "
            "fig6_snr_distance.py or its outputs.\n\n")
    f.write("FINDING (PAPER-STATED, both halves verified against paper_text.txt):\n")
    f.write("  - Main paper Sec.IV states its atomic parameters (transition scheme "
            "6S1/2->6P3/2->47D5/2->48P3/2, decay rates gamma3/(2pi)=3.9kHz, "
            "gamma4/(2pi)=1.7kHz, dipole moment wp_RF=-1443.459*e*a0) are "
            "\"taken from [12],[20],[23]\" (paper_text.txt line 615).\n")
    f.write("  - Reference [20] is confirmed (paper_text.txt line 996) to be "
            "Jing et al. 2020, Nat. Phys. 16, 911 -- the ORIGIN of this exact "
            "dipole moment value (their Methods: radial matrix element "
            "2946.512*e*a0, dipole moment 1443.450*e*a0 for the 47D5/2->48P3/2 "
            "transition, matching the main paper's -1443.459*e*a0 to 4 sig figs).\n")
    f.write("  - Jing et al. 2020's own Methods section states this transition "
            "is driven by \"a local MW field at 6.94GHz\" (i.e. this dipole "
            "moment/decay-rate set is SPECIFIC to a transition resonant at "
            "6.9458GHz, not an arbitrary/generic number).\n")
    f.write("  - Separately, the main paper's Sec.IV Fig.4 caption paragraph "
            "states (paper_text.txt line 657): \"carrier frequency of the RF "
            "signal is fRF = 3.5 GHz.\"\n\n")
    f.write("CLASSIFICATION: this is a genuine PAPER-INTERNAL parameter "
            "inconsistency (borrowed atomic parameters specific to a real "
            "6.9458GHz transition, reused for a separately-declared 3.5GHz "
            "RF carrier) -- not an error introduced by this reproduction. "
            "Both numbers are independently paper-stated; the inconsistency "
            "is between them, verified via ref [20]'s own source paper.\n\n")
    f.write("WHY IT MATTERS FOR THE 44dB GAP: fRF enters fig6_snr_distance.py's "
            "pipeline in exactly one place -- the Conventional receiver's "
            "effective aperture A_eff=lambda_RF^2/(4*pi) -- and nowhere in the "
            "LO-free/LO-dressed SNR formulas (Eq.8-37 depend on wp_RF/hbar "
            "only, never fRF). So substituting the transition-consistent "
            "fRF=6.9458GHz changes ONLY the Conventional receiver's SNR "
            "(smaller wavelength -> smaller effective aperture -> worse "
            "classical SNR), which WIDENS the gap.\n\n")
    f.write("RESULTS (PTx=-10dBm, d=100m checkpoint, diameter-corrected d=0.76mm "
            "baseline -- same real fig4/fig5 data as the frozen fig6_snr_distance.py):\n\n")
    for label, r in results.items():
        f.write(f"  {label}\n")
        f.write(f"    fRF={r['fRF_GHz']:.4f}GHz, Conventional SNR@100m={r['conv_100m']:.4f}dB, "
                f"gap={r['gap_100m']:.4f}dB, 0dB-coverage={r['coverage']:.2f}m\n\n")
    f.write(f"NET EFFECT: correcting fRF widens the gap by {delta_gap:+.4f}dB "
            f"(from {results[list(results.keys())[0]]['gap_100m']:.4f}dB to "
            f"{results[list(results.keys())[1]]['gap_100m']:.4f}dB), closing the "
            f"previously-unresolved portion of the paper's 44dB claim from "
            f"~{44.0 - results[list(results.keys())[0]]['gap_100m']:.2f}dB down to "
            f"~{44.0 - results[list(results.keys())[1]]['gap_100m']:.2f}dB.\n\n")
    f.write("EFFECT ON THE ~1500m COVERAGE CLAIM: negligible. The Theoretical "
            "LO-dressed 0dB-crossing distance does not depend on fRF at all "
            f"(fixed at {d_theo_0db:.2f}m by the real fig4/fig5 data, unchanged "
            "in both cases above). The Conventional receiver's own crossing "
            f"point moves from {results[list(results.keys())[0]]['d_conv_0db']:.2f}m "
            f"to {results[list(results.keys())[1]]['d_conv_0db']:.2f}m, which barely "
            f"affects the {d_theo_0db:.0f}m-scale 'extended coverage' number. "
            "The 1500m claim remains genuinely unresolved by this correction -- "
            "closing it would require a ~14dB improvement in the LO-dressed "
            "receiver's OWN noise floor / A_LO gain, which this audit does not "
            "attempt to justify from the paper's text.\n\n")
    f.write("STATUS: this correction is offered as a documented, provenance-"
            "verified lead, not adopted into the frozen fig6_snr_distance.py "
            "baseline. Adopting fRF=6.9458GHz project-wide (it would also "
            "affect any other figure that independently uses fRF=3.5GHz, if "
            "any) is a decision for the user, not applied silently here.\n")

print(f"\nSaved: {summary_path.name}")
print("DONE. fig6_snr_distance.py and its outputs were NOT modified.")
