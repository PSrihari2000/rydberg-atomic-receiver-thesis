# ============================================================
# FIG. 5(b) -- Input: |Omega_total(t)| vs t
#
# Uses the REAL Omega_total(t) values already computed and
# saved by fig5_lodressed_analysis.py (fig5_data.npz) -- pure
# equation evaluation of the phasor-magnitude formula, no
# fitting, no tracing, no new computation here.
#
# Axis convention: |Omega_total|/2pi on x, t on y -- matches
# the PAPER's own panel (b) layout (confirmed against the
# user-supplied screenshot earlier), which is why this produces
# the "hook" shape -- a genuine mathematical consequence of the
# phasor formula plotted this way, not a forced/traced match.
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
data = np.load(OUTPUT_DIR / "fig5_data.npz")

t_seconds = data["t_seconds"]
omega_total_curves = {
    1.0: data["omega_total_1mhz"],
    3.0: data["omega_total_3mhz"],
    5.0: data["omega_total_5mhz"],
}

print("=" * 70)
print("FIG. 5(b) -- Input, reusing already-computed real Omega_total(t)")
print("=" * 70)
for orf, curve in omega_total_curves.items():
    print(f"Omega_RF/2pi={orf}MHz: Omega_total range=[{curve.min():.4f},{curve.max():.4f}]MHz")

fig, ax = plt.subplots(figsize=(7, 6.5))
colors = {1.0: "red", 3.0: "green", 5.0: "blue"}
for orf in [1.0, 3.0, 5.0]:
    ax.plot(omega_total_curves[orf], t_seconds, "--", color=colors[orf],
            label=rf"$\Omega_{{total}}/2\pi={orf:g}$ MHz")

ax.set_xlabel(r"$|\Omega_{total}|/2\pi$ (MHz)")
ax.set_ylabel("t (s)")
ax.set_title("Fig. 5(b). Input\n(real physical period, $1/\\Delta f$)")
ax.set_xlim(0, 14)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fig5b_input.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig5b_input.png")
