# ============================================================
# FIG. 3 -- TOP-DOWN VIEW (bird's-eye contour map)
#
# Same real data as fig3_hamiltonian_qutip.py's own 3D plot (loaded
# from the already-computed fig3_quantum_response.npz -- NO new QuTiP
# solve, just a different viewing angle: straight down onto the
# (Delta_c, Omega_RF) plane instead of the oblique 3D surface, matching
# the paper's own top-down presentation of this same data and lining
# up with Fig.4's own (Delta_c, Omega_RF) axis convention.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
data = np.load(OUTPUT_DIR / "fig3_quantum_response.npz")

delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"] / 1e-6  # microW

fig, ax = plt.subplots(figsize=(7.5, 6.5))
mesh = ax.pcolormesh(delta_c_mhz, omega_rf_mhz, Pout_surface, cmap="turbo", shading="gouraud")
fig.colorbar(mesh, ax=ax, label=r"$P_{out}$ ($\mu$W)")

ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_title(r"Fig. 3. $P_{out}$ versus $\Delta_c$ and $\Omega_{RF}$" "\n(d = 0.76 mm)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_topdown.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print(f"Saved: fig3_topdown.png")
print(f"Colorbar range: {Pout_surface.min():.4f}-{Pout_surface.max():.4f} microW")
