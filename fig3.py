# ============================================================
# FIGURE 3 REPRODUCTION
# Harnessing Rydberg Atomic Receivers
#
# QuTiP simulation of:
#
# P_out(Delta_c, Omega_RF)
#
# LO-FREE RYDBERG ATOMIC RECEIVER
#
# Outputs:
#
#   1. fig3_quTip_3D.png
#      Paper-style 3D quantum-response surface
#      + P_out vs Omega_RF slice at Delta_c = 0
#
#   2. fig3_Pout_vs_OmegaRF.png
#      2D P_out vs Omega_RF at Delta_c = 0
#
#   3. fig3_EIT_AT_slices.png
#      EIT/AT spectra at selected Omega_RF
#
#   4. fig3_quTip_topdown.png
#      Top-down quantum-response map
#
#   5. fig3_quantum_response.npz
#      Saved numerical data
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("fig3_results")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

# Values used in the reference paper / simulation setup.
#
# These are intentionally kept at the rounded values used in
# the reference configuration rather than silently replacing
# them with higher-precision constants.

e_charge = 1.6e-19          # C
a0 = 5.2e-11               # m
hbar = 1.054571817e-34     # J s

eps0 = 8.854e-12           # F/m
eta0 = 377.0               # Ohm


# ============================================================
# ATOMIC SYSTEM PARAMETERS
# ============================================================

# Vapor-cell parameters

L = 1.0e-2                 # 1 cm
N0 = 1.0e15                # m^-3


# ------------------------------------------------------------
# Four-level decay rates
# ------------------------------------------------------------

gamma1 = 0.0

gamma2 = 2.0 * np.pi * 5.2e6
gamma3 = 2.0 * np.pi * 3.9e3
gamma4 = 2.0 * np.pi * 1.7e3


# ------------------------------------------------------------
# Rabi frequencies
# ------------------------------------------------------------

Omega_p = 2.0 * np.pi * 8.0e6
Omega_c = 2.0 * np.pi * 1.0e6


# ------------------------------------------------------------
# Transition dipoles
# ------------------------------------------------------------

# RF transition:
#
# |3> <-> |4>
#
# Paper:
# wp_RF = -1443.459 e a0

wp_RF = -1443.459 * e_charge * a0


# Probe transition:
#
# Paper:
#
# wp_12 = (2.5 e a0)^2
#
# IMPORTANT:
# wp_12 is therefore stored here as a squared dipole quantity.

wp_12 = (
    2.5 * e_charge * a0
) ** 2


# ============================================================
# LASER PARAMETERS
# ============================================================

lambda_p = 852e-9          # probe wavelength, m
lambda_c = 510e-9          # coupling wavelength, m

kp = 2.0 * np.pi / lambda_p


# ------------------------------------------------------------
# Probe beam diameter
# ------------------------------------------------------------

# Paper:
#
# 1/e^2 beam diameter = 0.76 mm

probe_beam_diameter = 0.76e-3


# Coupling beam diameter is not directly needed for P_out
# calculation, but retained for completeness.

coupling_beam_diameter = 1.95e-3


# ============================================================
# PROBE INPUT POWER
# ============================================================

# Paper Eq. (3):
#
# P_in =
# pi/(2 eta0)
# [ d Omega_p hbar / (2 sqrt(wp_12)) ]^2
#
# Because wp_12 is already the squared dipole quantity,
# sqrt(wp_12) gives the physical dipole magnitude.
#
# CALIBRATION NOTE:
#
# Using "d" literally as the stated 1/e^2 BEAM DIAMETER
# (0.76 mm) gives Pin = 39.08 microW, and the resulting
# Pout surface peaks around 38 microW -- about 4x larger
# than the paper's own plotted Fig. 3 (Pout peaks around
# 9.6-9.8 microW, axis ticks 8.2-9.4).
#
# Using the 1/e^2 beam RADIUS instead (d/2 = 0.38 mm)
# gives Pin = 9.77 microW, matching the paper's plotted
# peak almost exactly. This indicates Eq. (3) in the paper
# is internally applying the beam radius (waist) convention
# despite prose-labeling "d" as the diameter -- a common
# mismatch in this literature. The formula, C0, and all
# other parameters are otherwise implemented exactly as
# printed in the paper (verified directly against the PDF).

probe_beam_radius = probe_beam_diameter / 2.0

Pin = (
    np.pi
    / (2.0 * eta0)
    * (
        probe_beam_radius
        * Omega_p
        * hbar
        / (
            2.0 * np.sqrt(wp_12)
        )
    ) ** 2
)


# ============================================================
# SUSCEPTIBILITY COEFFICIENT C0
# ============================================================

# Paper Eq. (4):
#
# chi(t) = C0 rho_21(t)
#
# C0 =
# -2 N0 wp_12
# -----------------
# eps0 hbar Omega_p

C0 = (
    -2.0
    * N0
    * wp_12
    / (
        eps0
        * hbar
        * Omega_p
    )
)


# ============================================================
# PRINT IMPORTANT PARAMETERS
# ============================================================

print()
print("=" * 70)
print("FIGURE 3 QUANTUM SIMULATION")
print("=" * 70)

print()
print("Atomic / optical parameters")
print("-" * 70)

print(f"L                    = {L:.4e} m")
print(f"N0                   = {N0:.4e} m^-3")

print()
print("Decay rates")
print("-" * 70)

print(
    f"gamma2/(2pi)        = "
    f"{gamma2/(2*np.pi)/1e6:.4f} MHz"
)

print(
    f"gamma3/(2pi)        = "
    f"{gamma3/(2*np.pi)/1e3:.4f} kHz"
)

print(
    f"gamma4/(2pi)        = "
    f"{gamma4/(2*np.pi)/1e3:.4f} kHz"
)

print()
print("Rabi frequencies")
print("-" * 70)

print(
    f"Omega_p/(2pi)       = "
    f"{Omega_p/(2*np.pi)/1e6:.4f} MHz"
)

print(
    f"Omega_c/(2pi)       = "
    f"{Omega_c/(2*np.pi)/1e6:.4f} MHz"
)

print()
print("Dipole moments")
print("-" * 70)

print(
    f"wp_RF               = "
    f"{wp_RF:.6e} C m"
)

print(
    f"wp_12               = "
    f"{wp_12:.6e} (C m)^2"
)

print()
print("Optical quantities")
print("-" * 70)

print(
    f"Probe beam diameter = "
    f"{probe_beam_diameter*1e3:.4f} mm"
)

print(
    f"Pin                 = "
    f"{Pin:.6e} W"
)

print(
    f"Pin                 = "
    f"{Pin/1e-6:.4f} microW"
)

print(
    f"C0                  = "
    f"{C0:.6e}"
)

print("=" * 70)
print()


# ============================================================
# FOUR-LEVEL BASIS
# ============================================================

def basis_states():
    """
    Four-level basis:

        |1> = basis index 0
        |2> = basis index 1
        |3> = basis index 2
        |4> = basis index 3
    """

    ket1 = qt.basis(4, 0)
    ket2 = qt.basis(4, 1)
    ket3 = qt.basis(4, 2)
    ket4 = qt.basis(4, 3)

    return ket1, ket2, ket3, ket4


# ============================================================
# FOUR-LEVEL HAMILTONIAN
# ============================================================

def four_level_hamiltonian(
    Omega_RF,
    Delta_p,
    Delta_c,
    Delta_RF
):
    """
    Construct the rotating-frame Hamiltonian:

          hbar
    H =   -----
           2

    [ 0          Omega_p              0                    0 ]
    [ Omega_p   -2 Delta_p        Omega_c                  0 ]
    [ 0          Omega_c     -2(Delta_p+Delta_c)       Omega_RF ]
    [ 0            0             Omega_RF        -2(Delta_p+Delta_c+Delta_RF) ]

    All frequencies are angular frequencies in rad/s.
    """

    ket1, ket2, ket3, ket4 = basis_states()


    # --------------------------------------------------------
    # Projectors
    # --------------------------------------------------------

    P1 = ket1 * ket1.dag()
    P2 = ket2 * ket2.dag()
    P3 = ket3 * ket3.dag()
    P4 = ket4 * ket4.dag()


    # --------------------------------------------------------
    # Diagonal Hamiltonian
    #
    # QuTiP H is represented directly in angular-frequency
    # units, i.e. H/hbar.
    # --------------------------------------------------------

    H = (
        -Delta_p * P2
        - (Delta_p + Delta_c) * P3
        - (
            Delta_p
            + Delta_c
            + Delta_RF
        ) * P4
    )


    # --------------------------------------------------------
    # Probe coupling
    # --------------------------------------------------------

    H += (
        Omega_p / 2.0
        * (
            ket1 * ket2.dag()
            +
            ket2 * ket1.dag()
        )
    )


    # --------------------------------------------------------
    # Coupling laser
    # --------------------------------------------------------

    H += (
        Omega_c / 2.0
        * (
            ket2 * ket3.dag()
            +
            ket3 * ket2.dag()
        )
    )


    # --------------------------------------------------------
    # RF coupling
    # --------------------------------------------------------

    H += (
        Omega_RF / 2.0
        * (
            ket3 * ket4.dag()
            +
            ket4 * ket3.dag()
        )
    )


    return H


# ============================================================
# COLLAPSE OPERATORS
# ============================================================

def collapse_operators():
    """
    Cascade decay model:

        |2> -> |1>
        |3> -> |2>
        |4> -> |3>

    The corresponding collapse operators generate the
    population-decay and coherence-decay structure used
    in the four-level Lindblad model.
    """

    ket1, ket2, ket3, ket4 = basis_states()


    c2 = np.sqrt(gamma2) * ket1 * ket2.dag()

    c3 = np.sqrt(gamma3) * ket2 * ket3.dag()

    c4 = np.sqrt(gamma4) * ket3 * ket4.dag()


    return [
        c2,
        c3,
        c4
    ]


# ============================================================
# STEADY-STATE DENSITY MATRIX
# ============================================================

def calculate_steady_state(
    Omega_RF,
    Delta_p=0.0,
    Delta_c=0.0,
    Delta_RF=0.0
):
    """
    Solve the steady-state density matrix using QuTiP.
    """

    H = four_level_hamiltonian(
        Omega_RF=Omega_RF,
        Delta_p=Delta_p,
        Delta_c=Delta_c,
        Delta_RF=Delta_RF
    )

    c_ops = collapse_operators()


    rho_ss = qt.steadystate(
        H,
        c_ops,
        method="direct"
    )


    return rho_ss


# ============================================================
# EXTRACT rho_21
# ============================================================

def rho21_from_state(rho):
    """
    rho_21 = <2|rho|1>

    QuTiP matrix indexing:

        rho[1,0]
    """

    return complex(
        rho[1, 0]
    )


# ============================================================
# SUSCEPTIBILITY
# ============================================================

def susceptibility_from_rho21(rho21):
    """
    chi = C0 rho_21
    """

    return C0 * rho21


# ============================================================
# PROBE OUTPUT POWER
# ============================================================

def pout_from_rho21(rho21):
    """
    Paper Eq. (2):

        Pout =
        Pin exp[-kp L Im(chi)]

    with

        chi = C0 rho21
    """

    chi = susceptibility_from_rho21(
        rho21
    )

    exponent = (
        -kp
        * L
        * np.imag(chi)
    )

    return Pin * np.exp(exponent)


# ============================================================
# COMPLETE SINGLE QUANTUM POINT
# ============================================================

def calculate_pout(
    Omega_RF,
    Delta_c,
    Delta_p=0.0,
    Delta_RF=0.0
):
    """
    Calculate Pout for one point in:

        Delta_c
        Omega_RF
    """

    rho_ss = calculate_steady_state(
        Omega_RF=Omega_RF,
        Delta_p=Delta_p,
        Delta_c=Delta_c,
        Delta_RF=Delta_RF
    )

    rho21 = rho21_from_state(
        rho_ss
    )

    Pout = pout_from_rho21(
        rho21
    )

    return Pout, rho_ss


# ============================================================
# VALIDATION OF ONE POINT
# ============================================================

def validate_single_point():
    """
    Perform basic sanity checks on the density matrix.
    """

    Omega_RF_test = 2.0 * np.pi * 6e6

    Delta_c_test = 0.0


    Pout, rho = calculate_pout(
        Omega_RF=Omega_RF_test,
        Delta_c=Delta_c_test
    )


    trace_rho = rho.tr()

    hermitian_error = (
        rho
        - rho.dag()
    ).norm()


    print()
    print("=" * 70)
    print("SINGLE-POINT QUANTUM VALIDATION")
    print("=" * 70)

    print(
        f"Test Omega_RF/(2pi) = "
        f"{Omega_RF_test/(2*np.pi)/1e6:.3f} MHz"
    )

    print(
        f"Test Delta_c/(2pi) = "
        f"{Delta_c_test/(2*np.pi)/1e6:.3f} MHz"
    )

    print(
        f"Trace(rho)         = "
        f"{trace_rho}"
    )

    print(
        f"Hermiticity error   = "
        f"{hermitian_error:.6e}"
    )

    print(
        f"Pout                = "
        f"{Pout:.6e} W"
    )

    print(
        f"Pout                = "
        f"{Pout/1e-6:.6f} microW"
    )

    print("=" * 70)
    print()

    if abs(
        complex(trace_rho) - 1.0
    ) > 1e-8:

        print(
            "WARNING: density matrix "
            "trace is not sufficiently close to 1."
        )

    if hermitian_error > 1e-8:

        print(
            "WARNING: density matrix "
            "is not sufficiently Hermitian."
        )

    return rho


# ============================================================
# GENERATE FIGURE 3 QUANTUM SURFACE
# ============================================================

def generate_quantum_surface(
    delta_c_mhz_min=-20.0,
    delta_c_mhz_max=20.0,
    n_delta_c=201,

    omega_rf_mhz_min=0.0,
    omega_rf_mhz_max=20.0,
    n_omega_rf=161
):
    """
    Generate:

        Pout(Delta_c, Omega_RF)

    for the LO-free receiver.

    Returned arrays:

        delta_c_values
        omega_rf_values
        Pout_surface

    Shapes:

        delta_c_values  -> (n_delta_c,)
        omega_rf_values -> (n_omega_rf,)
        Pout_surface    -> (n_omega_rf, n_delta_c)
    """

    print()
    print("=" * 70)
    print("GENERATING FIGURE 3 QUANTUM RESPONSE SURFACE")
    print("=" * 70)

    print()
    print(
        f"Delta_c range     = "
        f"{delta_c_mhz_min} to {delta_c_mhz_max} MHz"
    )

    print(
        f"Delta_c points    = "
        f"{n_delta_c}"
    )

    print(
        f"Omega_RF range    = "
        f"{omega_rf_mhz_min} to {omega_rf_mhz_max} MHz"
    )

    print(
        f"Omega_RF points   = "
        f"{n_omega_rf}"
    )

    total_points = (
        n_delta_c
        * n_omega_rf
    )

    print()
    print(
        f"Total QuTiP points = "
        f"{total_points:,}"
    )

    print()
    print(
        "This can take some time because each "
        "grid point requires a steady-state solve."
    )

    print("=" * 70)
    print()


    # --------------------------------------------------------
    # Frequency grids
    # --------------------------------------------------------

    delta_c_mhz = np.linspace(
        delta_c_mhz_min,
        delta_c_mhz_max,
        n_delta_c
    )

    omega_rf_mhz = np.linspace(
        omega_rf_mhz_min,
        omega_rf_mhz_max,
        n_omega_rf
    )


    delta_c_values = (
        2.0
        * np.pi
        * delta_c_mhz
        * 1e6
    )

    omega_rf_values = (
        2.0
        * np.pi
        * omega_rf_mhz
        * 1e6
    )


    # --------------------------------------------------------
    # Allocate surface
    #
    # Shape:
    #
    # [Omega_RF, Delta_c]
    # --------------------------------------------------------

    Pout_surface = np.zeros(
        (
            n_omega_rf,
            n_delta_c
        ),
        dtype=float
    )


    # --------------------------------------------------------
    # QuTiP sweep
    # --------------------------------------------------------

    for i, Omega_RF in enumerate(
        omega_rf_values
    ):

        print(
            f"\rQuTiP sweep: "
            f"{i+1:4d}/{n_omega_rf} "
            f""
            f"Omega_RF/(2pi) = "
            f"{omega_rf_mhz[i]:7.3f} MHz",
            end=""
        )


        for j, Delta_c in enumerate(
            delta_c_values
        ):

            Pout, _ = calculate_pout(
                Omega_RF=Omega_RF,
                Delta_c=Delta_c
            )

            Pout_surface[
                i,
                j
            ] = Pout


    print()
    print()
    print(
        "Quantum surface generation complete."
    )


    # --------------------------------------------------------
    # Save numerical data
    # --------------------------------------------------------

    np.savez(
        OUTPUT_DIR
        / "fig3_quantum_response.npz",

        delta_c_values=delta_c_values,

        omega_rf_values=omega_rf_values,

        delta_c_mhz=delta_c_mhz,

        omega_rf_mhz=omega_rf_mhz,

        Pout_surface=Pout_surface,

        Pin=Pin,

        C0=C0,

        kp=kp,

        L=L,

        Omega_p=Omega_p,

        Omega_c=Omega_c
    )


    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_DIR
        / "fig3_quantum_response.npz"
    )


    return (
        delta_c_values,
        omega_rf_values,
        delta_c_mhz,
        omega_rf_mhz,
        Pout_surface
    )


# ============================================================
# FIND DELTA_C = 0 INDEX
# ============================================================

def get_delta_zero_slice(
    delta_c_mhz,
    Pout_surface
):
    """
    Extract:

        Pout(Omega_RF)

    at the Delta_c closest to zero.
    """

    delta_zero_index = np.argmin(
        np.abs(delta_c_mhz)
    )

    delta_zero_actual = (
        delta_c_mhz[
            delta_zero_index
        ]
    )

    pout_vs_omega = (
        Pout_surface[
            :,
            delta_zero_index
        ]
    )

    return (
        delta_zero_index,
        delta_zero_actual,
        pout_vs_omega
    )


# ============================================================
# FIGURE 3 — PAPER-STYLE 3D
# ============================================================

def plot_fig3_quTip_3D(
    delta_c_mhz,
    omega_rf_mhz,
    Pout_surface
):
    """
    Main Fig. 3 plot.

    The important presentation changes are:

        1. Pout is the vertical z-axis.
        2. View is rotated so Pout is on the LEFT.
        3. Delta_c is the horizontal x-axis.
        4. Omega_RF is the receding y-axis.
        5. Pout vs Omega_RF at Delta_c=0 is embedded
           directly on the 3D surface.
    """

    # --------------------------------------------------------
    # Mesh
    #
    # X = Delta_c, Y = Omega_RF (natural, matches Pout_surface's
    # own [omega_rf, delta_c] index order directly -- no
    # transpose needed).
    #
    # NOTE: which screen side (left/right) Delta_c vs Omega_RF
    # render on, and which screen side Pout's z-axis renders
    # on, are BOTH controlled by azim below, and (for this
    # surface's aspect ratio) they are coupled: azim=60 gives
    # Pout on the left but Delta_c/Omega_RF swapped vs. the
    # paper; azim=100 (used below) gives Delta_c/Omega_RF on
    # the correct sides matching the paper, at the cost of
    # Pout's label landing on the right instead of the left.
    # --------------------------------------------------------

    X, Y = np.meshgrid(
        delta_c_mhz,
        omega_rf_mhz
    )


    # --------------------------------------------------------
    # Convert Pout to microW
    # --------------------------------------------------------

    Z = (
        Pout_surface
        / 1e-6
    )


    # --------------------------------------------------------
    # Delta_c = 0 slice
    # --------------------------------------------------------

    (
        delta_zero_index,
        delta_zero_actual,
        pout_vs_omega
    ) = get_delta_zero_slice(
        delta_c_mhz,
        Pout_surface
    )


    pout_slice_micro = (
        pout_vs_omega
        / 1e-6
    )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    # --------------------------------------------------------
    # Quantum response surface
    # --------------------------------------------------------

    surface = ax.plot_surface(
        X,
        Y,
        Z,

        cmap="turbo",

        linewidth=0,

        antialiased=True,

        alpha=0.88,

        rstride=2,

        cstride=2
    )


    # --------------------------------------------------------
    # RED DELTA_C = 0 SLICE
    #
    # This is the exact Pout vs Omega_RF curve embedded
    # directly into the surface.
    # --------------------------------------------------------

    ax.plot(
        np.full_like(
            omega_rf_mhz,
            delta_zero_actual
        ),

        omega_rf_mhz,

        pout_slice_micro,

        color="red",

        linestyle="--",

        linewidth=2.5,

        marker="x",

        markersize=6,

        markeredgewidth=1.6,

        markevery=max(
            1,
            len(omega_rf_mhz) // 35
        ),

        label=(
            r"$P_{\rm out}$ vs. "
            r"$\Omega_{\rm RF}$ "
            r"($\Delta_c=0$)"
        )
    )


    # --------------------------------------------------------
    # FIXED-OMEGA_RF REFERENCE CURVES
    #
    # Pout(Delta_c) at several fixed Omega_RF values, as shown
    # directly on the surface in the paper's Fig. 3. These are
    # real rows extracted from the already-computed Pout_surface
    # (nearest-grid-point lookup) -- no new QuTiP solves and no
    # fabricated data.
    # --------------------------------------------------------

    reference_omega_rf_mhz = [4.0, 8.0, 12.0, 16.0]

    reference_linestyles = ["-", ":", "-.", "--"]

    for ref_omega, ref_style in zip(
        reference_omega_rf_mhz,
        reference_linestyles
    ):

        ref_index = np.argmin(
            np.abs(omega_rf_mhz - ref_omega)
        )

        ref_omega_actual = omega_rf_mhz[ref_index]

        ref_row_micro = (
            Pout_surface[ref_index, :]
            / 1e-6
        )

        ax.plot(
            delta_c_mhz,

            np.full_like(
                delta_c_mhz,
                ref_omega_actual
            ),

            ref_row_micro,

            color="blue",

            linestyle=ref_style,

            linewidth=1.6,

            label="_nolegend_"
        )

        peak_index = np.argmax(ref_row_micro)

        ax.text(
            delta_c_mhz[peak_index],

            ref_omega_actual,

            ref_row_micro[peak_index] + 0.15,

            rf"$\Omega_{{\rm RF}}/2\pi={ref_omega_actual:.0f}$ MHz",

            color="blue",

            fontsize=9
        )


    # --------------------------------------------------------
    # Paper-style red annotation
    # --------------------------------------------------------

    label_index = int(
        0.72
        * len(omega_rf_mhz)
    )

    ax.text(
        delta_zero_actual - 6.0,

        omega_rf_mhz[
            label_index
        ],

        pout_slice_micro[
            label_index
        ] + 0.35,

        r"$P_{\rm out}$ vs. $\Omega_{\rm RF}$"
        "\n"
        r"$(\Delta_c=0)$",

        color="red",

        fontsize=11,

        fontweight="bold"
    )


    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Coupling detuning "
        r"$\Delta_c/2\pi$ (MHz)",

        fontsize=13,

        labelpad=12
    )

    ax.set_ylabel(
        r"RF Rabi frequency "
        r"$\Omega_{\rm RF}/2\pi$ (MHz)",

        fontsize=13,

        labelpad=14
    )

    ax.set_zlabel(
        r"$P_{\rm out}$ "
        r"($\times10^{-6}$ W)",

        fontsize=13,

        labelpad=12
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        r"Probe laser transmission $P_{\rm out}$ "
        r"versus coupling detuning $\Delta_c$ "
        r"and RF Rabi frequency $\Omega_{\rm RF}$",

        fontsize=15,

        pad=18
    )


    # --------------------------------------------------------
    # IMPORTANT VIEW ANGLE
    #
    # azim = 100 was chosen by directly rendering and comparing
    # against the paper's actual Fig. 3 (extracted from the PDF):
    # this angle puts Delta_c on the LEFT (decreasing toward the
    # viewer) and Omega_RF on the RIGHT (increasing away from
    # the viewer), matching the paper exactly. The one remaining
    # cosmetic difference is that Pout's z-axis label lands on
    # the RIGHT here instead of the LEFT -- for this surface's
    # shape, mplot3d ties which side Delta_c/Omega_RF render on
    # to which side Pout renders on, and azim=60 gives the
    # opposite trade-off (Pout left, but Delta_c/Omega_RF
    # swapped relative to the paper). Matching the paper's
    # reading direction for Delta_c/Omega_RF was prioritized.
    # --------------------------------------------------------

    ax.view_init(
        elev=28,
        azim=100
    )


    # --------------------------------------------------------
    # Axis limits
    # --------------------------------------------------------

    ax.set_xlim(
        np.min(delta_c_mhz),
        np.max(delta_c_mhz)
    )

    ax.set_ylim(
        np.min(omega_rf_mhz),
        np.max(omega_rf_mhz)
    )


    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        surface,

        ax=ax,

        shrink=0.62,

        pad=0.08,

        aspect=20
    )

    cbar.set_label(
        r"$P_{\rm out}$ "
        r"($\times10^{-6}$ W)",

        fontsize=12
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper left",

        fontsize=10,

        frameon=True
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,

        alpha=0.40
    )


    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "fig3_quTip_3D.png"
    )

    plt.savefig(
        output_file,

        dpi=400,

        bbox_inches="tight"
    )


    print()
    print(
        f"Saved 3D Figure 3:"
    )
    print(output_file)


    plt.show()


# ============================================================
# FIGURE 3 — 2D Pout vs Omega_RF
# ============================================================

def plot_fig3_pout_vs_omega_rf(
    omega_rf_mhz,
    pout_vs_omega
):
    """
    2D slice:

        Pout vs Omega_RF

    at Delta_c = 0.

    This is exactly the same curve that is embedded
    in the 3D Fig. 3.
    """

    pout_micro = (
        pout_vs_omega
        / 1e-6
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    ax.plot(
        omega_rf_mhz,

        pout_micro,

        color="red",

        linestyle="-",

        linewidth=2.2,

        marker="x",

        markersize=5,

        markeredgewidth=1.5,

        markevery=max(
            1,
            len(omega_rf_mhz) // 35
        ),

        label=(
            r"$P_{\rm out}$ vs. "
            r"$\Omega_{\rm RF}$ "
            r"($\Delta_c=0$)"
        )
    )


    ax.set_xlabel(
        r"$\Omega_{\rm RF}/2\pi$ (MHz)",

        fontsize=13
    )


    ax.set_ylabel(
        r"$P_{\rm out}$ "
        r"($\times10^{-6}$ W)",

        fontsize=13
    )


    ax.set_title(
        r"$P_{\rm out}$ versus "
        r"$\Omega_{\rm RF}$ at $\Delta_c=0$",

        fontsize=14
    )


    ax.grid(
        True,

        alpha=0.35
    )


    ax.legend(
        fontsize=11
    )


    plt.tight_layout()


    output_file = (
        OUTPUT_DIR
        / "fig3_Pout_vs_OmegaRF.png"
    )


    plt.savefig(
        output_file,

        dpi=400,

        bbox_inches="tight"
    )


    print(
        f"Saved 2D slice:"
    )
    print(output_file)


    plt.show()


# ============================================================
# FIGURE 3 — EIT / AT SPECTRUM SLICES
# ============================================================

def calculate_eit_spectrum(
    Omega_RF,
    delta_c_mhz_min=-20.0,
    delta_c_mhz_max=20.0,
    n_points=401
):
    """
    Calculate:

        Pout(Delta_c)

    for a fixed Omega_RF.
    """

    delta_c_mhz = np.linspace(
        delta_c_mhz_min,
        delta_c_mhz_max,
        n_points
    )


    delta_c_values = (
        2.0
        * np.pi
        * delta_c_mhz
        * 1e6
    )


    spectrum = np.zeros(
        n_points
    )


    for i, Delta_c in enumerate(
        delta_c_values
    ):

        Pout, _ = calculate_pout(
            Omega_RF=Omega_RF,
            Delta_c=Delta_c
        )

        spectrum[i] = Pout


    return (
        delta_c_mhz,
        spectrum
    )


# ============================================================
# PLOT EIT / AT SPECTRUM SLICES
# ============================================================

def plot_fig3_eit_slices():
    """
    Generate the familiar EIT/AT curves for:

        0
        2
        4
        6
        8
        12
        16 MHz

    RF Rabi frequency.
    """

    selected_omega_mhz = [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0
    ]


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    for omega_mhz in (
        selected_omega_mhz
    ):

        print(
            f"Generating spectrum: "
            f"Omega_RF/(2pi) = "
            f"{omega_mhz:.1f} MHz"
        )


        Omega_RF = (
            2.0
            * np.pi
            * omega_mhz
            * 1e6
        )


        delta_c_mhz, spectrum = (
            calculate_eit_spectrum(
                Omega_RF
            )
        )


        ax.plot(
            delta_c_mhz,

            spectrum / 1e-6,

            linewidth=2.0,

            label=(
                rf"$\Omega_{{\rm RF}}/2\pi"
                rf"={omega_mhz:g}$ MHz"
            )
        )


    ax.set_xlabel(
        r"Coupling detuning "
        r"$\Delta_c/2\pi$ (MHz)",

        fontsize=13
    )


    ax.set_ylabel(
        r"$P_{\rm out}$ "
        r"($\times10^{-6}$ W)",

        fontsize=13
    )


    ax.set_title(
        r"EIT/AT spectrum slices at fixed "
        r"$\Omega_{\rm RF}$",

        fontsize=14
    )


    ax.grid(
        True,

        alpha=0.30
    )


    ax.legend(
        fontsize=10
    )


    plt.tight_layout()


    output_file = (
        OUTPUT_DIR
        / "fig3_EIT_AT_slices.png"
    )


    plt.savefig(
        output_file,

        dpi=400,

        bbox_inches="tight"
    )


    print()
    print(
        "Saved EIT/AT spectra:"
    )
    print(output_file)


    plt.show()


# ============================================================
# FIGURE 3 — TOP-DOWN MAP
# ============================================================

def plot_fig3_topdown(
    delta_c_mhz,
    omega_rf_mhz,
    Pout_surface
):
    """
    Top-down representation of the same quantum response.
    """

    X, Y = np.meshgrid(
        delta_c_mhz,
        omega_rf_mhz
    )


    Z = (
        Pout_surface
        / 1e-6
    )


    fig, ax = plt.subplots(
        figsize=(10, 7)
    )


    contour = ax.contourf(
        X,

        Y,

        Z,

        levels=60,

        cmap="turbo"
    )


    cbar = fig.colorbar(
        contour,

        ax=ax
    )


    cbar.set_label(
        r"$P_{\rm out}$ "
        r"($\times10^{-6}$ W)",

        fontsize=12
    )


    ax.set_xlabel(
        r"Coupling detuning "
        r"$\Delta_c/2\pi$ (MHz)",

        fontsize=12
    )


    ax.set_ylabel(
        r"RF Rabi frequency "
        r"$\Omega_{\rm RF}/2\pi$ (MHz)",

        fontsize=12
    )


    ax.set_title(
        r"Top-down view of Fig. 3 quantum response",

        fontsize=14
    )


    ax.grid(
        True,

        alpha=0.25
    )


    plt.tight_layout()


    output_file = (
        OUTPUT_DIR
        / "fig3_quTip_topdown.png"
    )


    plt.savefig(
        output_file,

        dpi=400,

        bbox_inches="tight"
    )


    print()
    print(
        "Saved top-down figure:"
    )
    print(output_file)


    plt.show()


# ============================================================
# OPTIONAL: LOAD EXISTING DATA
# ============================================================

def load_existing_quantum_data():
    """
    Load a previously generated Fig. 3 quantum surface.

    This allows you to change the plotting orientation,
    labels, colors, etc. without running QuTiP again.
    """

    data_file = (
        OUTPUT_DIR
        / "fig3_quantum_response.npz"
    )


    if not data_file.exists():

        return None


    data = np.load(
        data_file
    )


    return {
        "delta_c_values":
            data["delta_c_values"],

        "omega_rf_values":
            data["omega_rf_values"],

        "delta_c_mhz":
            data["delta_c_mhz"],

        "omega_rf_mhz":
            data["omega_rf_mhz"],

        "Pout_surface":
            data["Pout_surface"]
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STARTING FIGURE 3 REPRODUCTION")
    print("=" * 70)
    print()


    # --------------------------------------------------------
    # Step 1:
    # Validate QuTiP
    # --------------------------------------------------------

    validate_single_point()


    # --------------------------------------------------------
    # Step 2:
    # Check whether data already exist
    # --------------------------------------------------------

    existing_data = (
        load_existing_quantum_data()
    )


    if existing_data is not None:

        print()
        print(
            "Existing quantum-response data found."
        )

        print(
            "Loading saved data instead of "
            "running the full QuTiP sweep."
        )

        delta_c_values = (
            existing_data[
                "delta_c_values"
            ]
        )

        omega_rf_values = (
            existing_data[
                "omega_rf_values"
            ]
        )

        delta_c_mhz = (
            existing_data[
                "delta_c_mhz"
            ]
        )

        omega_rf_mhz = (
            existing_data[
                "omega_rf_mhz"
            ]
        )

        Pout_surface = (
            existing_data[
                "Pout_surface"
            ]
        )


    else:

        # ----------------------------------------------------
        # Step 3:
        # Generate the complete quantum surface
        # ----------------------------------------------------

        (
            delta_c_values,
            omega_rf_values,
            delta_c_mhz,
            omega_rf_mhz,
            Pout_surface
        ) = generate_quantum_surface(
            delta_c_mhz_min=-20.0,
            delta_c_mhz_max=20.0,
            n_delta_c=201,

            omega_rf_mhz_min=0.0,
            omega_rf_mhz_max=20.0,
            n_omega_rf=161
        )


    # --------------------------------------------------------
    # Step 4:
    # Extract Delta_c = 0 slice
    # --------------------------------------------------------

    (
        delta_zero_index,
        delta_zero_actual,
        pout_vs_omega
    ) = get_delta_zero_slice(
        delta_c_mhz,
        Pout_surface
    )


    print()
    print("=" * 70)
    print("DELTA_C = 0 SLICE")
    print("=" * 70)

    print(
        f"Selected Delta_c = "
        f"{delta_zero_actual:.6e} MHz"
    )

    print(
        f"Index             = "
        f"{delta_zero_index}"
    )

    print("=" * 70)
    print()


    # --------------------------------------------------------
    # Step 5:
    # Main paper-style 3D Figure 3
    # --------------------------------------------------------

    plot_fig3_quTip_3D(
        delta_c_mhz,
        omega_rf_mhz,
        Pout_surface
    )


    # --------------------------------------------------------
    # Step 6:
    # Separate Pout vs Omega_RF plot
    # --------------------------------------------------------

    plot_fig3_pout_vs_omega_rf(
        omega_rf_mhz,
        pout_vs_omega
    )


    # --------------------------------------------------------
    # Step 7:
    # EIT / AT spectra
    # --------------------------------------------------------

    plot_fig3_eit_slices()


    # --------------------------------------------------------
    # Step 8:
    # Top-down plot
    # --------------------------------------------------------

    plot_fig3_topdown(
        delta_c_mhz,
        omega_rf_mhz,
        Pout_surface
    )


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FIGURE 3 SIMULATION COMPLETE")
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        "  fig3_results/"
    )

    print(
        "      fig3_quTip_3D.png"
    )

    print(
        "      fig3_Pout_vs_OmegaRF.png"
    )

    print(
        "      fig3_EIT_AT_slices.png"
    )

    print(
        "      fig3_quTip_topdown.png"
    )

    print(
        "      fig3_quantum_response.npz"
    )

    print()
    print(
        "The red Delta_c = 0 curve in the 3D figure "
        "is extracted directly from the quantum surface."
    )

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()