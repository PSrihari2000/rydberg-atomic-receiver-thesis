# ============================================================
# FIG. 3 -- 3D SURFACE, PAPER-STYLE VIEWING ANGLE -- d = 0.36 mm
#
# Same styling as qutip_d076mm/fig3_pout_surface_paperstyle.py,
# applied to the genuine d=0.36mm QuTiP grid computed in this
# folder (fig3_qutip_atomic_response_d036mm.npz). No new QuTiP
# solve -- just the camera angle / axis styling matching the
# paper's own Fig.3 layout: Pout axis pane forced next to the
# Delta_c axis, RF Rabi frequency ticks at 5 MHz steps.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
data = np.load(OUTPUT_DIR / "fig3_qutip_atomic_response_d036mm.npz")

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
ax.set_zlabel("")
ax.set_title(r"Fig. 3. $P_{out}$ versus $\Delta_c$ and $\Omega_{RF}$" "\n(d = 0.36 mm)")
ax.set_xticks(np.arange(-20, 21, 10))
ax.set_yticks(np.arange(0, 21, 5))   # RF axis: 5 MHz steps, matches paper's own ticks
ax.view_init(elev=23, azim=-230)
ax.zaxis._axinfo['juggled'] = (1, 2, 0)   # Pout axis pane forced onto the left side
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=r"$P_{out}$ ($\mu$W)")
ax.legend(fontsize=9, loc="upper left")
fig.subplots_adjust(left=0.15)
fig.text(0.04, 0.55, r"$P_{out}$ ($\times10^{-6}$ W)", rotation=90, va="center", fontsize=11)
fig.savefig(OUTPUT_DIR / "fig3_pout_surface_paperstyle_d036mm.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved: fig3_pout_surface_paperstyle_d036mm.png")
