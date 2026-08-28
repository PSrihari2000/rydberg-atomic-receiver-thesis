# ============================================================
# Export the real fig3_quantum_response.npz grid to a plain CSV
# so it can be opened directly in Excel/Sheets and browsed by
# hand. No new computation -- just a format conversion of the
# already-saved, real data.
# ============================================================

from pathlib import Path
import numpy as np
import csv

OUTPUT_DIR = Path(__file__).resolve().parent
data = np.load(OUTPUT_DIR / "fig3_quantum_response.npz")

delta_c_mhz = data["delta_c_mhz"]
omega_rf_mhz = data["omega_rf_mhz"]
Pout_surface = data["Pout_surface"] / 1e-6   # convert to microW for readability

with open(OUTPUT_DIR / "fig3_quantum_response_grid.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # header row: first cell labels the row axis, rest are Delta_c values
    writer.writerow(["Omega_RF_MHz \\ Delta_c_MHz"] + [f"{v:.2f}" for v in delta_c_mhz])
    for i, omega in enumerate(omega_rf_mhz):
        writer.writerow([f"{omega:.4f}"] + [f"{v:.6f}" for v in Pout_surface[i, :]])

print(f"Saved: fig3_quantum_response_grid.csv")
print(f"Shape: {len(omega_rf_mhz)} rows x {len(delta_c_mhz)} columns (+ header row/col)")
print(f"All values in microwatts. Open directly in Excel/Sheets/Numbers.")
