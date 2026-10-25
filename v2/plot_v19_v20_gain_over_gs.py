"""Plot paired HS-GS test-set gains over converged biased GS for N=8 and N=16."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
INPUTS = {
    8: RESULTS / "HS_GS_V19_fresh_test_table.csv",
    16: RESULTS / "HS_GS_V20_fresh_test_table.csv",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True, constrained_layout=True)
colors = {10: "#1769aa", 30: "#d55e00"}
markers = {10: "o", 30: "s"}
styles = {10: "-", 30: "--"}

for axis, (n_antennas, filename) in zip(axes, INPUTS.items()):
    data = pd.read_csv(filename)
    for pilots in (10, 30):
        part = data.loc[data["P"] == pilots].sort_values("SNR_dB")
        gain = part["gain_over_GS_dB"].to_numpy()
        lower = part["CI_low_GS"].to_numpy()
        upper = part["CI_high_GS"].to_numpy()
        yerr = [gain - lower, upper - gain]
        axis.errorbar(
            part["SNR_dB"], gain, yerr=yerr,
            color=colors[pilots], marker=markers[pilots], linestyle=styles[pilots],
            linewidth=1.8, markersize=5.5, capsize=3,
            label=f"P = {pilots}",
        )
    axis.axhline(0, color="black", linewidth=1, linestyle=":")
    axis.set_title(f"N = {n_antennas}")
    axis.set_xlabel("SNR (dB)")
    axis.set_xticks(range(-5, 31, 5))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", frameon=True)

axes[0].set_ylabel("HS-GS gain over converged GS (dB)")
axes[0].text(-0.16, 0.5, "Positive values favor HS-GS", transform=axes[0].transAxes,
             rotation=90, va="center", ha="center", fontsize=9, color="#444444")
fig.suptitle("Paired test-set HS-GS gain over biased GS\n(pointwise 95% paired bootstrap intervals; 500 samples per point)",
             fontsize=13)

png_path = RESULTS / "HS_GS_V19_V20_gain_over_GS.png"
pdf_path = RESULTS / "HS_GS_V19_V20_gain_over_GS.pdf"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(png_path)
print(pdf_path)
