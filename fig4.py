# ============================================================
# FIGURE 4 REPRODUCTION
# Harnessing Rydberg Atomic Receivers
#
# Distortion of the LO-free Rydberg atomic receiver.
#
# Paper description (Sec. IV-A):
#
#   "If we fix Pout in Fig. 3 and scan Delta_c, we obtain the
#    trend of the AT splitting interval upon varying the Rabi
#    frequency Omega_RF, as shown in Fig. 4 (actually a
#    top-down view of Fig. 3). ... the dashed gray line
#    represents the theoretical interval between a pair of AT
#    splitting peaks, while the jagged solid red curves are
#    obtained using the QuTiP toolkit. ... as the Rabi
#    frequency decreases, the interval between the two AT
#    splitting peaks gradually narrows. However, this interval
#    does not reduce to zero; instead, it saturates at a
#    certain Rabi frequency, around 5.5 MHz. Below this
#    threshold, the EIT spectrum no longer exhibits two peaks,
#    ... indicating that the LO-free system has entered its
#    distortion region."
#
# This script does NOT run a new QuTiP sweep and does NOT
# modify the Hamiltonian / master equation / physical
# parameters in fig3.py. Fig. 4 is explicitly a re-analysis of
# the same Pout(Delta_c, Omega_RF) surface computed for Fig. 3
# -- it extracts the AT peak POSITIONS at each Omega_RF from
# the already-computed, real fig3_quantum_response.npz.
#
# PEAK-RESOLVABILITY CRITERION
#
# The only thing this script adds beyond raw peak-finding is a
# defensible test for whether two candidate peaks are a
# genuinely resolvable AT doublet, as opposed to numerical
# shoulders on a single broadened EIT line. The test is a
# Rayleigh-style linewidth criterion:
#
#   Two candidate peaks are RESOLVED only if their separation
#   is at least 2 x Gamma_EIT, where Gamma_EIT is the natural
#   EIT transparency-window half-width
#
#       Gamma_EIT = (Omega_c^2 + Omega_p^2)
#                   / (2 sqrt(gamma2^2 + 2 Omega_p^2))
#
#   computed from the SAME atomic parameters already used in
#   fig3.py (Omega_c, Omega_p, gamma2 -- imported directly from
#   fig3, not re-typed). This is Documentation Eq. (68).
#
# Physical justification: an AT-split peak only becomes
# distinguishable from the unsplit EIT line once it has moved
# out from line center by roughly its own natural linewidth.
# Closer than that, the "splitting" is just a shoulder on a
# single broadened feature, not a resolvable doublet -- this is
# the same logic as the classical Rayleigh criterion for
# resolving two close spectral lines.
#
# This threshold is NOT fitted or hard-coded to hit the paper's
# ~5.5 MHz. It falls out of Omega_c, Omega_p, gamma2 alone.
#
# Outputs (all in fig4_results/):
#
#   1. fig4_distortion.png
#      Paper-style AT peak position vs Omega_RF plot.
#
#   2. fig4_diagnostics.png
#      Per-Omega_RF diagnostic panels showing the spectrum,
#      candidate peaks, valley, and resolved/unresolved
#      classification with reasoning.
#
#   3. fig4_peak_data.npz
#      Saved peak-position / classification arrays.
#
#   4. fig4_summary.csv
#      Per-Omega_RF text summary of the classification.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import contextlib
import csv
import io

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import find_peaks

# Reuse the EXACT atomic parameters already defined in fig3.py
# (Omega_c, Omega_p, gamma2) -- no re-typed / duplicated
# constants. fig3.py prints a parameter banner on import; that
# banner is suppressed here to keep fig4's own output readable.

with contextlib.redirect_stdout(io.StringIO()):
    import fig3


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("fig4_results")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD THE REAL FIG. 3 QUANTUM SURFACE
# ============================================================

def load_fig3_surface():
    """
    Load the already-computed, real Pout(Delta_c, Omega_RF)
    surface from Fig. 3. No new QuTiP solves happen here.
    """

    data_file = Path("fig3_results") / "fig3_quantum_response.npz"

    if not data_file.exists():
        raise FileNotFoundError(
            "fig3_results/fig3_quantum_response.npz not found. "
            "Run fig3.py first to generate the quantum surface "
            "that Fig. 4 is derived from."
        )

    data = np.load(data_file)

    return (
        data["delta_c_mhz"],
        data["omega_rf_mhz"],
        data["Pout_surface"]
    )


# ============================================================
# NATURAL EIT LINEWIDTH (RESOLUTION SCALE)
# ============================================================

def natural_eit_linewidth_mhz():
    """
    Gamma_EIT = (Omega_c^2 + Omega_p^2)
                / (2 sqrt(gamma2^2 + 2 Omega_p^2))

    Documentation Eq. (68). Uses Omega_c, Omega_p, gamma2
    exactly as defined in fig3.py (imported, not duplicated).
    """

    omega_c_mhz = fig3.Omega_c / (2.0 * np.pi * 1e6)
    omega_p_mhz = fig3.Omega_p / (2.0 * np.pi * 1e6)
    gamma2_mhz = fig3.gamma2 / (2.0 * np.pi * 1e6)

    return (
        (omega_c_mhz ** 2 + omega_p_mhz ** 2)
        / (
            2.0
            * np.sqrt(
                gamma2_mhz ** 2
                + 2.0 * omega_p_mhz ** 2
            )
        )
    )


# ============================================================
# CLASSIFY A SINGLE EIT/AT SPECTRUM
# ============================================================

def classify_spectrum(
    delta_c_mhz,
    pout_row,
    gamma_eit_mhz,
    resolvability_linewidths=2.0,
    search_half_width_mhz=10.0
):
    """
    Determine whether Pout(Delta_c) at one fixed Omega_RF shows
    a genuinely resolvable AT doublet.

    Procedure:

      1. Find ALL local maxima in the search window (purely
         topological candidates -- no arbitrary absolute-Watt
         prominence filter).
      2. If fewer than two local maxima exist, the spectrum is
         genuinely single-peaked: UNRESOLVED.
      3. Otherwise take the two highest local maxima as the
         AT-doublet candidates, and locate the valley between
         them.
      4. Require the valley to be a genuine INTERIOR minimum
         (not coincident with either peak).
      5. Require the peak separation to be at least
         `resolvability_linewidths` x Gamma_EIT (Rayleigh-style
         linewidth criterion, see module docstring).

    Returns a dict with the full diagnostic trail so the
    classification can be plotted and audited.
    """

    window = (
        np.abs(delta_c_mhz)
        <= search_half_width_mhz
    )

    dc_w = delta_c_mhz[window]
    row_w = pout_row[window]

    all_max_idx, _ = find_peaks(row_w)

    diag = {
        "delta_c_window": dc_w,
        "pout_window": row_w,
        "candidate_delta": dc_w[all_max_idx],
        "candidate_pout": row_w[all_max_idx],
    }

    if len(all_max_idx) < 2:

        diag.update(
            resolved=False,
            reason="single_peak",
            left=np.nan,
            right=np.nan,
            separation=np.nan,
            required_separation=(
                resolvability_linewidths
                * gamma_eit_mhz
            ),
            valley_delta=np.nan,
            valley_pout=np.nan,
        )

        return diag

    heights = row_w[all_max_idx]

    top2 = all_max_idx[
        np.argsort(heights)[-2:]
    ]

    top2 = np.sort(top2)

    left_idx, right_idx = top2

    delta_left = dc_w[left_idx]
    delta_right = dc_w[right_idx]

    separation = delta_right - delta_left

    between = row_w[left_idx:right_idx + 1]

    valley_local_idx = np.argmin(between)

    valley_delta = dc_w[left_idx + valley_local_idx]
    valley_pout = between[valley_local_idx]

    valley_is_interior = (
        0 < valley_local_idx < (right_idx - left_idx)
    )

    required_separation = (
        resolvability_linewidths
        * gamma_eit_mhz
    )

    resolved = (
        valley_is_interior
        and (separation >= required_separation)
    )

    if resolved:
        reason = "resolved"
    elif not valley_is_interior:
        reason = "no_interior_valley"
    else:
        reason = "below_linewidth_criterion"

    diag.update(
        resolved=resolved,
        reason=reason,
        left=delta_left if resolved else np.nan,
        right=delta_right if resolved else np.nan,
        separation=separation,
        required_separation=required_separation,
        valley_delta=valley_delta,
        valley_pout=valley_pout,
    )

    return diag


# ============================================================
# SWEEP OVER OMEGA_RF (REAL DATA, NO NEW QUTIP SOLVES)
# ============================================================

def characterize_distortion(
    delta_c_mhz,
    omega_rf_mhz,
    Pout_surface,
    gamma_eit_mhz,
    resolvability_linewidths=2.0
):
    """
    For every Omega_RF row in the real Fig. 3 surface, classify
    the spectrum and extract AT peak positions where resolved.
    """

    n_omega = len(omega_rf_mhz)

    delta_left = np.full(n_omega, np.nan)
    delta_right = np.full(n_omega, np.nan)
    separation = np.full(n_omega, np.nan)
    resolved = np.zeros(n_omega, dtype=bool)
    reason = np.empty(n_omega, dtype=object)

    for i in range(n_omega):

        diag = classify_spectrum(
            delta_c_mhz,
            Pout_surface[i, :],
            gamma_eit_mhz,
            resolvability_linewidths=resolvability_linewidths
        )

        delta_left[i] = diag["left"]
        delta_right[i] = diag["right"]
        separation[i] = diag["separation"]
        resolved[i] = diag["resolved"]
        reason[i] = diag["reason"]

    return {
        "delta_left": delta_left,
        "delta_right": delta_right,
        "separation": separation,
        "resolved": resolved,
        "reason": reason,
    }


# ============================================================
# EMPIRICAL DISTORTION THRESHOLD
# ============================================================

def find_distortion_threshold(
    omega_rf_mhz,
    resolved
):
    """
    Lowest Omega_RF at which the spectrum is still classified
    as a resolved AT doublet.
    """

    if not np.any(resolved):
        return np.nan

    return np.min(
        omega_rf_mhz[resolved]
    )


# ============================================================
# DIAGNOSTIC PANELS
# ============================================================

def plot_diagnostics(
    delta_c_mhz,
    omega_rf_mhz,
    Pout_surface,
    gamma_eit_mhz,
    resolvability_linewidths,
    target_omega_rf_mhz
):
    """
    For a set of Omega_RF values, show Pout(Delta_c), the
    candidate peaks, the valley, and the resolved/unresolved
    verdict with reasoning.
    """

    n = len(target_omega_rf_mhz)

    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.6 * nrows)
    )

    axes = np.atleast_1d(axes).ravel()

    for ax, target in zip(axes, target_omega_rf_mhz):

        idx = np.argmin(
            np.abs(omega_rf_mhz - target)
        )

        actual_omega = omega_rf_mhz[idx]

        diag = classify_spectrum(
            delta_c_mhz,
            Pout_surface[idx, :],
            gamma_eit_mhz,
            resolvability_linewidths=resolvability_linewidths
        )

        dc_w = diag["delta_c_window"]
        row_w = diag["pout_window"] / 1e-6

        ax.plot(
            dc_w,
            row_w,
            color="black",
            linewidth=1.2,
            label="Pout"
        )

        ax.scatter(
            diag["candidate_delta"],
            diag["candidate_pout"] / 1e-6,
            color="gray",
            s=22,
            zorder=3,
            label="candidates"
        )

        if diag["resolved"]:

            ax.scatter(
                [diag["left"], diag["right"]],
                [
                    row_w[
                        np.argmin(np.abs(dc_w - diag["left"]))
                    ],
                    row_w[
                        np.argmin(np.abs(dc_w - diag["right"]))
                    ],
                ],
                color="red",
                s=55,
                zorder=4,
                label="chosen AT peaks"
            )

            ax.scatter(
                diag["valley_delta"],
                diag["valley_pout"] / 1e-6,
                color="blue",
                marker="v",
                s=55,
                zorder=4,
                label="valley"
            )

        verdict = (
            "RESOLVED"
            if diag["resolved"]
            else "UNRESOLVED"
        )

        sep_text = (
            f"sep={diag['separation']:.2f} MHz"
            if np.isfinite(diag["separation"])
            else "sep=n/a"
        )

        req_text = (
            f"req={diag['required_separation']:.2f} MHz"
        )

        ax.set_title(
            f"$\\Omega_{{\\rm RF}}/2\\pi$={actual_omega:.2f} MHz\n"
            f"{verdict} ({diag['reason']})\n"
            f"{sep_text}, {req_text}",
            fontsize=9
        )

        ax.set_xlim(-10, 10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=9
    )

    fig.suptitle(
        "Fig. 4 peak-resolvability diagnostics\n"
        r"(criterion: separation $\geq$ "
        f"{resolvability_linewidths:.1f}"
        r"$\times\Gamma_{\rm EIT}$"
        f" = {resolvability_linewidths * gamma_eit_mhz:.2f} MHz)",
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.90])

    output_file = OUTPUT_DIR / "fig4_diagnostics.png"

    plt.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight"
    )

    print("Saved:", output_file)

    plt.close(fig)


# ============================================================
# PLOT — PAPER-STYLE FIG. 4
# ============================================================

def plot_fig4_distortion(
    omega_rf_mhz,
    delta_left,
    delta_right,
    omega_rf_th,
    gamma_eit_mhz,
    resolvability_linewidths
):
    """
    Paper-style plot: Delta_c on the x-axis, Omega_RF on the
    y-axis. Red solid = real QuTiP-extracted, resolvability-
    filtered AT peak positions. Gray dashed = theoretical
    Delta_c = +/- Omega_RF/2.
    """

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.plot(
        -omega_rf_mhz / 2.0,
        omega_rf_mhz,
        color="gray",
        linestyle="--",
        linewidth=1.3,
        label="Theoretical"
    )

    ax.plot(
        omega_rf_mhz / 2.0,
        omega_rf_mhz,
        color="gray",
        linestyle="--",
        linewidth=1.3
    )

    ax.plot(
        delta_left,
        omega_rf_mhz,
        color="red",
        linewidth=1.6,
        label="Obtained by QuTiP"
    )

    ax.plot(
        delta_right,
        omega_rf_mhz,
        color="red",
        linewidth=1.6
    )

    if np.isfinite(omega_rf_th):

        ax.axhspan(
            0,
            omega_rf_th,
            color="0.85",
            zorder=0
        )

        ax.annotate(
            "Distortion region\n"
            f"(unresolved below "
            rf"$\Omega_{{\rm RF}}/2\pi \approx {omega_rf_th:.2f}$ MHz)",
            xy=(0, omega_rf_th * 0.5),
            xytext=(-7.8, omega_rf_th + 4.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10
        )

    ax.set_xlabel(
        r"Coupling detuning, $\Delta_c/2\pi$ (MHz)",
        fontsize=12
    )

    ax.set_ylabel(
        r"RF Rabi frequency, $\Omega_{\rm RF}/2\pi$ (MHz)",
        fontsize=12
    )

    ax.set_title(
        "Distortion of the LO-free Rydberg atomic receiver\n"
        r"(Rayleigh criterion: separation $\geq$ "
        f"{resolvability_linewidths:.1f}"
        r"$\times\Gamma_{\rm EIT}$"
        f" = {resolvability_linewidths * gamma_eit_mhz:.2f} MHz)",
        fontsize=11
    )

    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 16)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", fontsize=10)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "fig4_distortion.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print()
    print("Saved:", output_file)

    plt.close(fig)


# ============================================================
# CSV SUMMARY
# ============================================================

def save_summary_csv(
    omega_rf_mhz,
    result
):

    output_file = OUTPUT_DIR / "fig4_summary.csv"

    with open(output_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "omega_rf_mhz",
            "delta_left_mhz",
            "delta_right_mhz",
            "separation_mhz",
            "resolved",
            "reason",
        ])

        for i, omega in enumerate(omega_rf_mhz):

            writer.writerow([
                f"{omega:.4f}",
                f"{result['delta_left'][i]:.4f}",
                f"{result['delta_right'][i]:.4f}",
                f"{result['separation'][i]:.4f}",
                result["resolved"][i],
                result["reason"][i],
            ])

    print("Saved:", output_file)


# ============================================================
# MAIN
# ============================================================

def main():

    RESOLVABILITY_LINEWIDTHS = 2.0

    print()
    print("=" * 70)
    print("FIGURE 4 REPRODUCTION (real re-analysis of Fig. 3 data)")
    print("=" * 70)

    delta_c_mhz, omega_rf_mhz, Pout_surface = load_fig3_surface()

    print(f"Loaded Pout_surface shape: {Pout_surface.shape}")
    print(
        f"Delta_c range: {delta_c_mhz.min():.1f} to "
        f"{delta_c_mhz.max():.1f} MHz"
    )
    print(
        f"Omega_RF range: {omega_rf_mhz.min():.1f} to "
        f"{omega_rf_mhz.max():.1f} MHz"
    )

    gamma_eit_mhz = natural_eit_linewidth_mhz()

    print()
    print("-" * 70)
    print("Peak-resolvability criterion (Rayleigh-style)")
    print("-" * 70)
    print(f"Omega_c/(2pi)          = {fig3.Omega_c/(2*np.pi*1e6):.4f} MHz")
    print(f"Omega_p/(2pi)          = {fig3.Omega_p/(2*np.pi*1e6):.4f} MHz")
    print(f"gamma2/(2pi)           = {fig3.gamma2/(2*np.pi*1e6):.4f} MHz")
    print(f"Gamma_EIT              = {gamma_eit_mhz:.4f} MHz")
    print(
        f"Required separation    = "
        f"{RESOLVABILITY_LINEWIDTHS} x Gamma_EIT = "
        f"{RESOLVABILITY_LINEWIDTHS * gamma_eit_mhz:.4f} MHz"
    )
    print("-" * 70)

    result = characterize_distortion(
        delta_c_mhz,
        omega_rf_mhz,
        Pout_surface,
        gamma_eit_mhz,
        resolvability_linewidths=RESOLVABILITY_LINEWIDTHS
    )

    omega_rf_th = find_distortion_threshold(
        omega_rf_mhz,
        result["resolved"]
    )

    print()
    print("=" * 70)
    print(
        f"Empirical distortion threshold (from real data, "
        f"linewidth criterion): Omega_RF/(2pi) = {omega_rf_th:.3f} MHz"
    )
    print(
        "Paper's stated value (Fig. 4 text): approx. 5.5 MHz"
    )
    print("=" * 70)

    plot_fig4_distortion(
        omega_rf_mhz,
        result["delta_left"],
        result["delta_right"],
        omega_rf_th,
        gamma_eit_mhz,
        RESOLVABILITY_LINEWIDTHS
    )

    target_omega_rf_mhz = [
        1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 8.0, 12.0, 16.0
    ]

    plot_diagnostics(
        delta_c_mhz,
        omega_rf_mhz,
        Pout_surface,
        gamma_eit_mhz,
        RESOLVABILITY_LINEWIDTHS,
        target_omega_rf_mhz
    )

    save_summary_csv(
        omega_rf_mhz,
        result
    )

    np.savez(
        OUTPUT_DIR / "fig4_peak_data.npz",
        omega_rf_mhz=omega_rf_mhz,
        delta_left=result["delta_left"],
        delta_right=result["delta_right"],
        separation=result["separation"],
        resolved=result["resolved"],
        gamma_eit_mhz=gamma_eit_mhz,
        resolvability_linewidths=RESOLVABILITY_LINEWIDTHS,
        omega_rf_threshold=omega_rf_th
    )

    print("Saved:", OUTPUT_DIR / "fig4_peak_data.npz")


if __name__ == "__main__":
    main()
