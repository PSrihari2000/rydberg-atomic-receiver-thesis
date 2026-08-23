# ============================================================
# FIG. 3 -- 3D SURFACE, PAPER-STYLE VIEWING ANGLE
#
# Same real data as fig3_hamiltonian_qutip.py's own 3D plot
# (loaded from the already-computed fig3_quantum_response.npz --
# NO new QuTiP solve, just a camera-angle change to match the
# paper's own layout: Pout axis on the left, Delta_c receding on
# the left, Omega_RF receding on the right -- matching the
# user-supplied screenshot of the paper's actual Fig.3).
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
data = np.load(OUTPUT_DIR / "fig3_quantum_response.npz")

delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"]

X, Y = np.meshgrid(delta_c_mhz, omega_rf_mhz)
Z = Pout_surface / 1e-6

dc0 = np.argmin(np.abs(delta_c_mhz))

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="turbo", linewidth=0, antialiased=True, alpha=0.93)
ax.plot(
    np.full_like(omega_rf_mhz, delta_c_mhz[dc0]), omega_rf_mhz, Z[:, dc0],
    color="red", linestyle="--", linewidth=2, marker="x", markersize=5,
    label=r"$P_{out}$ vs $\Omega_{RF}$ ($\Delta_c=0$)"
)

ax.set_xlabel(r"Coupling detuning, $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"RF Rabi frequency, $\Omega_{RF}/2\pi$ (MHz)")
ax.set_zlabel("")  # set_zlabel gets clipped with the juggled axis pane -- use fig.text instead
ax.set_title(r"Fig. 3. $P_{out}$ versus $\Delta_c$ and $\Omega_{RF}$" "\n(d = 0.76 mm)")
ax.set_xticks(np.arange(-20, 21, 10))   # Delta_c axis: 10MHz steps, matches paper's own ticks
ax.set_yticks(np.arange(0, 21, 5))   # RF axis: 5MHz steps instead of 2.5MHz (less cluttered)
ax.view_init(elev=23, azim=-230)   # slightly adjusted tilt
ax.zaxis._axinfo['juggled'] = (1, 2, 0)   # force the z (Pout) axis pane onto the left side
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$P_{out}$ ($\mu$W)")
ax.legend(fontsize=9, loc="upper left")
fig.subplots_adjust(left=0.15)
fig.text(0.04, 0.55, r"$P_{out}$ ($\times10^{-6}$ W)", rotation=90, va="center", fontsize=11)
fig.savefig(OUTPUT_DIR / "fig3_pout_surface_paperstyle.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig3_pout_surface_paperstyle.png")
