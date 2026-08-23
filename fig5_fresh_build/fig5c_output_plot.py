# ============================================================
# FIG. 5(c) -- Output: P_out(t) vs t
#
# P_out(t) = f( Omega_total(t) ), where f is panel (a)'s real
# QuTiP static response curve and Omega_total(t) is panel (b)'s
# real phasor-magnitude curve. Function composition of two
# already-real results -- NOT the paper's linearized Eq.(27),
# so genuine nonlinear distortion (for large Omega_RF) shows up
# honestly rather than being assumed away.
#
# Uses pout_1mhz/3mhz/5mhz and t_seconds, already computed and
# saved by fig5_lodressed_analysis.py -- no new computation.
# Axis convention matches the paper directly: x=t, y=Pout.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
data = np.load(OUTPUT_DIR / "fig5_data.npz")

t_seconds = data["t_seconds"]
pout_curves = {
    1.0: data["pout_1mhz"],
    3.0: data["pout_3mhz"],
    5.0: data["pout_5mhz"],
}

print("=" * 70)
print("FIG. 5(c) -- Output, reusing already-computed real Pout(t)")
print("=" * 70)
for orf, curve in pout_curves.items():
    print(f"Omega_RF/2pi={orf}MHz: Pout range=[{np.nanmin(curve)/1e-6:.4f},"
          f"{np.nanmax(curve)/1e-6:.4f}] microW")

fig, ax = plt.subplots(figsize=(7, 6.5))
colors = {1.0: "red", 3.0: "green", 5.0: "blue"}
for orf in [1.0, 3.0, 5.0]:
    ax.plot(t_seconds, pout_curves[orf] / 1e-6, color=colors[orf],
            label=rf"$\Omega_{{total}}/2\pi={orf:g}$ MHz")

ax.set_xlabel("t (s)")
ax.set_ylabel(r"$P_{out}$ ($\times10^{-6}$ W)")
ax.set_title("Fig. 5(c). Output\n(real physical period, $1/\\Delta f$)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig5c_output.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig5c_output.png")
