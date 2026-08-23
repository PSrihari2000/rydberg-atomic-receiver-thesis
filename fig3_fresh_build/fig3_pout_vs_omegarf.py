# ============================================================
# FIG. 3 companion plot -- Pout vs Omega_RF at Delta_c=0,
# extracted from the SAME fresh grid (fig3_quantum_response.npz).
# No new QuTiP solves, no fitting.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

data = np.load(OUTPUT_DIR / "fig3_quantum_response.npz")
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"]

dc0_idx = int(np.argmin(np.abs(delta_c_mhz)))
exact = np.isclose(delta_c_mhz[dc0_idx], 0.0, atol=1e-9)
if not exact:
    print(f"WARNING: no exact Delta_c=0 grid point, using nearest = {delta_c_mhz[dc0_idx]} MHz")

pout_vs_omega = Pout_surface[:, dc0_idx]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(
    omega_rf_mhz, pout_vs_omega / 1e-6,
    color="red", linewidth=2.2, marker="x", markersize=5, markeredgewidth=1.4,
    markevery=max(1, len(omega_rf_mhz) // 35),
    label=r"$P_{out}$ vs $\Omega_{RF}$ ($\Delta_c=0$)"
)
ax.set_xlabel(r"$\Omega_{RF}/2\pi$ (MHz)")
ax.set_ylabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax.set_title(r"$P_{out}$ versus $\Omega_{RF}$ at $\Delta_c=0$" "\n(d = 0.76 mm, fresh rebuild)")
ax.grid(True, alpha=0.35)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_pout_vs_omegarf.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("Saved: fig3_pout_vs_omegarf.png")
print(f"Delta_c=0 slice: min={pout_vs_omega.min()/1e-6:.4f} microW, "
      f"max={pout_vs_omega.max()/1e-6:.4f} microW (at Omega_RF/2pi="
      f"{omega_rf_mhz[np.argmax(pout_vs_omega)]:.3f} MHz)")
