# ============================================================
# FIG. 3 companion plot -- EIT/AT spectrum slices at fixed
# Omega_RF, extracted from the SAME fresh grid computed in
# fig3_hamiltonian_qutip.py (fig3_quantum_response.npz).
#
# No new QuTiP solves. No fitting/rescaling. Rows are picked
# from the real, already-saved Pout_surface array at the exact
# grid indices matching each requested Omega_RF (the 0.125 MHz
# grid spacing divides 0,2,4,6,8,12,16 MHz exactly -- verified
# below, not assumed).
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

data = np.load(OUTPUT_DIR / "fig3_quantum_response.npz")
delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"]

selected_omega_mhz = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]

fig, ax = plt.subplots(figsize=(11, 6.5))

for omega_target in selected_omega_mhz:
    idx = int(np.argmin(np.abs(omega_rf_mhz - omega_target)))
    actual_omega = omega_rf_mhz[idx]
    exact_match = np.isclose(actual_omega, omega_target, atol=1e-9)
    if not exact_match:
        print(f"WARNING: requested Omega_RF/2pi={omega_target} MHz has no exact grid "
              f"point -- using nearest available, {actual_omega:.4f} MHz, off by "
              f"{abs(actual_omega-omega_target):.4f} MHz")
    ax.plot(
        delta_c_mhz, Pout_surface[idx, :] / 1e-6,
        linewidth=2.0,
        label=rf"$\Omega_{{RF}}/2\pi = {actual_omega:g}$ MHz"
    )

ax.set_xlabel(r"Coupling detuning $\Delta_c/2\pi$ (MHz)")
ax.set_ylabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax.set_title("EIT/AT spectrum slices at fixed $\\Omega_{RF}$\n(d = 0.76 mm, fresh rebuild)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_eit_at_slices.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("Saved: fig3_eit_at_slices.png")
print(f"Peak value across all slices: {Pout_surface.max()/1e-6:.4f} microW "
      f"(same real data as fig3_pout_surface.png -- no new computation)")
