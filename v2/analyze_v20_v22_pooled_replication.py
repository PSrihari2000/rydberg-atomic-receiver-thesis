"""Pool the independent V20 and V22 N=16 paired test samples.

No model or policy selection is performed here. The V22 test uses the same
frozen validation policy as V20, with a distinct test seed stream. This script
combines their per-sample squared errors and channel energies, then computes
ratio-of-sums NMSE and pointwise paired bootstrap confidence intervals.
"""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
V20_FILE = RESULTS / "HS_GS_V20_fresh_test_results.mat"
V22_FILE = RESULTS / "HS_GS_V22_fresh_test_results.mat"
POLICY_FILE = RESULTS / "HS_GS_V20_frozen_policy.mat"
OUTPUT_CSV = RESULTS / "HS_GS_V20_V22_pooled_replication.csv"
OUTPUT_PNG = RESULTS / "HS_GS_V20_V22_pooled_gain_over_GS.png"
OUTPUT_PDF = RESULTS / "HS_GS_V20_V22_pooled_gain_over_GS.pdf"

METHODS = ["HS_GS", "EM_GS", "Biased_GS", "GD"]
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261025


def read_matlab_result(path: Path) -> dict:
    """Load the result arrays from a MATLAB v7.3 file (HDF5 axis order reversed)."""
    with h5py.File(path, "r") as handle:
        required = [
            "test_squared_error",
            "test_channel_energy",
            "P_list",
            "SNR_dB",
            "selected_rank",
            "selected_projection_updates",
            "selected_total_updates",
            "test_seed_base",
        ]
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{path.name} is missing required fields: {missing}")

        errors = np.asarray(handle["test_squared_error"], dtype=np.float64)
        energy = np.asarray(handle["test_channel_energy"], dtype=np.float64)
        # MATLAB [P, SNR, samples, methods] -> h5py [methods, samples, SNR, P].
        errors = errors.transpose(3, 2, 1, 0)
        # MATLAB [P, SNR, samples] -> h5py [samples, SNR, P].
        energy = energy.transpose(2, 1, 0)

        def scalar(name: str) -> float:
            return float(np.asarray(handle[name]).reshape(-1)[0])

        return {
            "errors": errors,
            "energy": energy,
            "P_list": np.asarray(handle["P_list"]).reshape(-1).astype(int),
            "SNR_dB": np.asarray(handle["SNR_dB"]).reshape(-1).astype(int),
            # MATLAB [P, SNR] -> h5py [SNR, P].
            "selected_rank": np.asarray(handle["selected_rank"]).T.astype(int),
            "selected_projection_updates": np.asarray(
                handle["selected_projection_updates"]
            ).T.astype(int),
            "selected_total_updates": np.asarray(
                handle["selected_total_updates"]
            ).T.astype(int),
            "test_seed_base": scalar("test_seed_base"),
            "samples": errors.shape[2],
        }


def validate_pair(v20: dict, v22: dict) -> None:
    for name in ("P_list", "SNR_dB"):
        if not np.array_equal(v20[name], v22[name]):
            raise ValueError(f"V20 and V22 {name} do not match.")
    for name in ("N", "K", "RSR_dB", "p_H"):
        if v20[name] != v22[name]:
            raise ValueError(f"V20 and V22 {name} do not match.")
    for name in (
        "selected_rank",
        "selected_projection_updates",
        "selected_total_updates",
    ):
        if not np.array_equal(v20[name], v22[name]):
            raise ValueError(f"V20 and V22 frozen policies differ in {name}.")
    if v20["test_seed_base"] == v22["test_seed_base"]:
        raise ValueError("V20 and V22 use the same test seed base; refusing to pool.")
    if v20["errors"].shape[:2] != (len(v20["P_list"]), len(v20["SNR_dB"])):
        raise ValueError(f"Unexpected V20 error array shape: {v20['errors'].shape}")
    if v22["errors"].shape[:2] != (len(v22["P_list"]), len(v22["SNR_dB"])):
        raise ValueError(f"Unexpected V22 error array shape: {v22['errors'].shape}")
    for run in (v20, v22):
        if run["errors"].shape[3] != len(METHODS):
            raise ValueError(f"Expected {len(METHODS)} methods in saved errors.")
        if not np.isfinite(run["errors"]).all() or not np.isfinite(run["energy"]).all():
            raise ValueError("A test result contains nonfinite sample errors or energy.")
        if np.any(run["energy"] <= 0):
            raise ValueError("A test sample has nonpositive channel energy.")


def nmse_db(errors: np.ndarray, energy: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(errors.sum(axis=0) / energy.sum())


def paired_bootstrap_gain_ci(
    errors: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap paired ratio-of-sums dB gains of HS-GS over other methods."""
    sample_count = errors.shape[0]
    gains = np.empty((replicates, len(METHODS) - 1), dtype=np.float64)
    chunk_size = 100
    for begin in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - begin)
        indices = rng.integers(0, sample_count, size=(count, sample_count))
        sampled = errors[indices].sum(axis=1)
        gains[begin : begin + count, :] = 10.0 * np.log10(
            sampled[:, 1:] / sampled[:, [0]]
        )
    low, high = np.percentile(gains, [2.5, 97.5], axis=0)
    return low, high


def main() -> None:
    if not V20_FILE.is_file() or not V22_FILE.is_file() or not POLICY_FILE.is_file():
        raise FileNotFoundError("V20, V22, and the V20 frozen-policy MAT files are required.")

    v20 = read_matlab_result(V20_FILE)
    v22 = read_matlab_result(V22_FILE)
    policy = loadmat(POLICY_FILE)
    for run in (v20, v22):
        for name in ("N", "K", "RSR_dB", "p_H"):
            run[name] = float(np.asarray(policy[name]).reshape(-1)[0])
    for run, label in ((v20, "V20"), (v22, "V22")):
        for name in (
            "selected_rank",
            "selected_projection_updates",
            "selected_total_updates",
        ):
            expected = np.asarray(policy[name], dtype=int)
            if not np.array_equal(run[name], expected):
                raise ValueError(f"{label} result does not match the frozen V20 {name}.")
    validate_pair(v20, v22)

    pooled_errors = np.concatenate((v20["errors"], v22["errors"]), axis=2)
    pooled_energy = np.concatenate((v20["energy"], v22["energy"]), axis=2)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    gains_for_plot = {}
    intervals_for_plot = {}

    for p_index, pilot_count in enumerate(v20["P_list"]):
        gains_for_plot[int(pilot_count)] = np.zeros((len(v20["SNR_dB"]), 2))
        intervals_for_plot[int(pilot_count)] = np.zeros(
            (len(v20["SNR_dB"]), 2, 2)
        )
        for snr_index, snr_db in enumerate(v20["SNR_dB"]):
            # Sample axis is moved first so each bootstrap resample keeps all
            # methods paired for the same channel, pilots, and observations.
            errors = pooled_errors[p_index, snr_index, :, :]
            energy = pooled_energy[p_index, snr_index, :]
            metrics = nmse_db(errors, energy)
            ci_low, ci_high = paired_bootstrap_gain_ci(
                errors, BOOTSTRAP_REPLICATES, rng
            )
            gain = metrics[1:] - metrics[0]

            rows.append(
                {
                    "P": int(pilot_count),
                    "SNR_dB": int(snr_db),
                    "samples_pooled": int(errors.shape[0]),
                    "HS_rank": int(v20["selected_rank"][p_index, snr_index]),
                    "HS_projected_updates": int(
                        v20["selected_projection_updates"][p_index, snr_index]
                    ),
                    "HS_total_updates": int(
                        v20["selected_total_updates"][p_index, snr_index]
                    ),
                    "HS_GS_NMSE_dB": metrics[0],
                    "EM_GS_NMSE_dB": metrics[1],
                    "biased_GS_NMSE_dB": metrics[2],
                    "GD_NMSE_dB": metrics[3],
                    "gain_over_EM_dB": gain[0],
                    "gain_over_GS_dB": gain[1],
                    "gain_over_GD_dB": gain[2],
                    "CI_low_EM_dB": ci_low[0],
                    "CI_high_EM_dB": ci_high[0],
                    "CI_low_GS_dB": ci_low[1],
                    "CI_high_GS_dB": ci_high[1],
                    "CI_low_GD_dB": ci_low[2],
                    "CI_high_GD_dB": ci_high[2],
                    "V20_gain_over_GS_dB": float(
                        (nmse_db(v20["errors"][p_index, snr_index], v20["energy"][p_index, snr_index])[2]
                         - nmse_db(v20["errors"][p_index, snr_index], v20["energy"][p_index, snr_index])[0])
                    ),
                    "V22_gain_over_GS_dB": float(
                        (nmse_db(v22["errors"][p_index, snr_index], v22["energy"][p_index, snr_index])[2]
                         - nmse_db(v22["errors"][p_index, snr_index], v22["energy"][p_index, snr_index])[0])
                    ),
                }
            )
            gains_for_plot[int(pilot_count)][snr_index, :] = gain[[1, 0]]
            intervals_for_plot[int(pilot_count)][snr_index, 0, :] = ci_low[[1, 0]]
            intervals_for_plot[int(pilot_count)][snr_index, 1, :] = ci_high[[1, 0]]

    fieldnames = list(rows[0])
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    snr = v20["SNR_dB"]
    colors = {"GS": "#1769aa", "EM-GS": "#d97706"}
    for ax, pilot_count in zip(axes, v20["P_list"]):
        gains = gains_for_plot[int(pilot_count)]
        ci = intervals_for_plot[int(pilot_count)]
        for method_index, label in enumerate(("GS", "EM-GS")):
            marker = "o" if label == "GS" else "s"
            ax.plot(
                snr,
                gains[:, method_index],
                marker=marker,
                linewidth=1.5,
                label=f"HS-GS gain over {label}",
                color=colors[label],
            )
            # Draw percentile intervals explicitly. A percentile-bootstrap
            # interval need not contain the original point estimate.
            ax.vlines(
                snr,
                ci[:, method_index, 0],
                ci[:, method_index, 1],
                color=colors[label],
                linewidth=1.0,
                alpha=0.85,
            )
            ax.scatter(snr, gains[:, method_index], marker=marker, color=colors[label], s=25)
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_title(f"P = {int(pilot_count)}")
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks(snr)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("NMSE gain (dB); positive favors HS-GS")
    fig.suptitle("Pooled independent paired tests: V20 + V22, N=16, RSR=25 dB")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    print(f"Pooled {v20['samples']} V20 + {v22['samples']} V22 samples per P/SNR.")
    print(f"Bootstrap replicates per point: {BOOTSTRAP_REPLICATES} (pointwise 95% CIs).")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Figure: {OUTPUT_PNG}")
    for row in rows:
        decision = "SUPPORTED" if row["CI_low_GS_dB"] > 0 else "INCLUDES ZERO"
        print(
            f"P={row['P']:2d}, SNR={row['SNR_dB']:>3d} dB: "
            f"gain over GS={row['gain_over_GS_dB']:+.3f} dB "
            f"[{row['CI_low_GS_dB']:+.3f}, {row['CI_high_GS_dB']:+.3f}] "
            f"{decision}; gain over EM-GS={row['gain_over_EM_dB']:+.3f} dB"
        )


if __name__ == "__main__":
    main()
