# ============================================================
# FIG. 4 -- CLEAN PRESENTATION VERSION
#
# Reuses the already-computed data-derived results (no new QuTiP
# solves, no data modification) -- just a simplified, paper-style
# rendering: only "Obtained by QuTiP" and "Theoretical" in the
# legend, one threshold line at the measured FWHM (~5 MHz).
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import csv

OUTPUT_DIR = Path(__file__).resolve().parent

omega_rf_mhz, left_x, right_x = [], [], []
with open(OUTPUT_DIR / "fig4_data_derived_threshold.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        omega_rf_mhz.append(float(row["Omega_RF_MHz"]))
        left_x.append(float(row["left_MHz"]) if row["left_MHz"] else np.nan)
        right_x.append(float(row["right_MHz"]) if row["right_MHz"] else np.nan)

omega_rf_mhz = np.array(omega_rf_mhz)
left_x = np.array(left_x)
right_x = np.array(right_x)

# Measured-HWHM Rayleigh threshold, from fig4_data_derived_threshold.py's output
FWHM_threshold_mhz = 5.0

fig, ax = plt.subplots(figsize=(7, 6.5))

has_valley = ~np.isnan(left_x)
ax.plot(left_x[has_valley], omega_rf_mhz[has_valley], "-", color="red", linewidth=1.8,
        label="Obtained by QuTiP")
ax.plot(right_x[has_valley], omega_rf_mhz[has_valley], "-", color="red", linewidth=1.8)

ax.plot(omega_rf_mhz / 2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2, label="Theoretical")
ax.plot(-omega_rf_mhz / 2.0, omega_rf_mhz, "--", color="gray", linewidth=1.2)

ax.axhline(FWHM_threshold_mhz, color="k", linestyle=":", alpha=0.7)
ax.fill_between([-10, 10], 0, FWHM_threshold_mhz, color="gray", alpha=0.15)
ax.annotate("Distortion region", xy=(-8.5, FWHM_threshold_mhz * 0.5), fontsize=10)

ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_title("Fig. 4. Distortion of the LO-free Rydberg atomic receiver")
ax.set_xlim(-10, 10)
ax.set_ylim(0, 10)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_clean_presentation.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig4_clean_presentation.png")
