# CAMB v2 LateDE -- cubic spline consistency tests
#
# DEmodel = 5 : fixed-node natural cubic spline in redshift
#
# Intended LateDE behavior:
#
#   spline_z = [0.0, 0.2, 0.57, 0.8, 1.3, 2.33]
#
#   - w(z) is a natural cubic spline through the supplied node values.
#   - w(z_i) = w_i exactly at every node.
#   - w(z) = -1 exactly for z > z_max.
#   - rho_DE is obtained from the continuity equation.
#
# This script tests those properties independently.

import numpy as np
import matplotlib.pyplot as plt
import camb
import os


# ============================================================
# Basic setup
# ============================================================

print(
    "Using CAMB %s installed at %s"
    % (camb.__version__, os.path.dirname(camb.__file__))
)

base_params = dict(
    H0=70,
    ombh2=0.02238280,
    omch2=0.1201075,
    TCMB=2.7255,
    dark_energy_model="ppf",

    # Neutrinos
    omnuh2=0,
    num_nu_massless=3.044,
    num_nu_massive=0,
    nu_mass_degeneracies=[0],
    nu_mass_numbers=[0],

    # Primordial spectrum
    As=2.100549e-9,
    ns=0.9660499,
    YHe=0.246,

    WantTransfer=True,
)

SPLINE_Z = np.array([
    0.0,
    0.2,
    0.57,
    0.8,
    1.3,
    2.33,
])

ZMAX = SPLINE_Z[-1]

redshift = np.linspace(
    0.0,
    10.0,
    30001,
)


# ============================================================
# Helpers
# ============================================================

def banner(name):
    print()
    print("=" * 78)
    print(name)
    print("=" * 78)


def camb_rho_w_at_z(results, z_values):
    """
    Query CAMB at arbitrary redshifts.

    CAMB's get_dark_energy_rho_w takes scale factor a.
    """

    z_values = np.atleast_1d(
        z_values
    ).astype(float)

    a_values = (
        1.0
        / (1.0 + z_values)
    )

    order = np.argsort(
        a_values
    )

    rho_sorted, w_sorted = (
        results.get_dark_energy_rho_w(
            a_values[order]
        )
    )

    rho_sorted = np.asarray(
        rho_sorted,
        dtype=float,
    )

    w_sorted = np.asarray(
        w_sorted,
        dtype=float,
    )

    rho = np.empty_like(
        rho_sorted
    )

    w = np.empty_like(
        w_sorted
    )

    rho[order] = rho_sorted
    w[order] = w_sorted

    return rho, w


def rho_ratio_from_w(z, w):
    """
    Independent reconstruction of

        rho_DE(z)/rho_DE(0)
          = exp[
                3 integral_0^z
                (1+w(z'))/(1+z') dz'
              ].

    This deliberately does not reproduce the analytic spline integral
    implemented in LateDarkEnergy.f90.
    """

    z = np.asarray(
        z,
        dtype=float,
    )

    w = np.asarray(
        w,
        dtype=float,
    )

    if np.any(np.diff(z) <= 0):
        raise ValueError(
            "z must be strictly increasing"
        )

    integrand = (
        (1.0 + w)
        / (1.0 + z)
    )

    cumulative = np.zeros_like(
        z
    )

    dz = np.diff(
        z
    )

    cumulative[1:] = np.cumsum(
        0.5
        * (
            integrand[:-1]
            + integrand[1:]
        )
        * dz
    )

    return np.exp(
        3.0 * cumulative
    )


def compare_results(
        label,
        results_a,
        results_b,
        z,
        atol_w=1.0e-11,
        atol_rho=1.0e-11):

    rho_a, w_a = camb_rho_w_at_z(
        results_a,
        z,
    )

    rho_b, w_b = camb_rho_w_at_z(
        results_b,
        z,
    )

    rho_a = rho_a / rho_a[0]
    rho_b = rho_b / rho_b[0]

    max_w = np.max(
        np.abs(
            w_a
            - w_b
        )
    )

    max_rho = np.max(
        np.abs(
            rho_a
            - rho_b
        )
    )

    print(
        f"{label}: "
        f"max |Delta w|={max_w:.3e}, "
        f"max |Delta rho/rho0|={max_rho:.3e}"
    )

    assert max_w < atol_w
    assert max_rho < atol_rho


# ============================================================
# Reference LambdaCDM
# ============================================================

results_lcdm = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=1,
        w0=-1.0,
    )
)


# ============================================================
# TEST 1: LambdaCDM limit
# ============================================================

banner("TEST 1: all spline nodes at -1 give LambdaCDM")

w_lcdm = np.full(
    len(SPLINE_Z),
    -1.0,
)

pars_lcdm = camb.set_params(
    **base_params,
    DEmodel=5,
    spline_z=SPLINE_Z,
    spline_w=w_lcdm,
)

results_spline_lcdm = camb.get_results(
    pars_lcdm
)

compare_results(
    "cubic spline LambdaCDM vs constant-w LambdaCDM",
    results_spline_lcdm,
    results_lcdm,
    redshift,
)

print("PASS: cubic spline reproduces LambdaCDM")


# ============================================================
# TEST 2: exact interpolation at fixed nodes
# ============================================================

banner("TEST 2: spline passes exactly through every input node")

w_nodes = np.array([
    -1.00,
    -0.80,
    -1.15,
    -0.90,
    -1.10,
    -1.00,
])

pars = camb.set_params(
    **base_params,
    DEmodel=5,
    spline_z=SPLINE_Z,
    spline_w=w_nodes,
)

stored_w = np.asarray(
    pars.DarkEnergy.spline_w
)

print(
    "max |stored spline_w - input| = "
    f"{np.max(np.abs(stored_w-w_nodes)):.3e}"
)

assert np.allclose(
    stored_w,
    w_nodes,
    atol=0,
    rtol=0,
)

results = camb.get_results(
    pars
)

_, w_at_nodes = camb_rho_w_at_z(
    results,
    SPLINE_Z,
)

for zi, expected, actual in zip(
        SPLINE_Z,
        w_nodes,
        w_at_nodes):

    print(
        f"z={zi:6.3f}   "
        f"input={expected:+.12f}   "
        f"CAMB={actual:+.12f}   "
        f"diff={actual-expected:+.3e}"
    )

max_node_error = np.max(
    np.abs(
        w_at_nodes
        - w_nodes
    )
)

print(
    "max node interpolation error = "
    f"{max_node_error:.3e}"
)

assert max_node_error < 1.0e-12

print("PASS: cubic spline interpolates every node exactly")


# ============================================================
# TEST 3: independent rho_DE reconstruction
# ============================================================

banner("TEST 3: independent rho_DE reconstruction")

rho_camb, w_camb = camb_rho_w_at_z(
    results,
    redshift,
)

rho_camb_ratio = (
    rho_camb
    / rho_camb[0]
)

rho_python_ratio = rho_ratio_from_w(
    redshift,
    w_camb,
)

max_abs_error = np.max(
    np.abs(
        rho_camb_ratio
        - rho_python_ratio
    )
)

max_rel_error = np.max(
    np.abs(
        rho_camb_ratio
        - rho_python_ratio
    )
    / np.maximum(
        np.abs(rho_camb_ratio),
        1.0e-30,
    )
)

max_log_error = np.max(
    np.abs(
        np.log(rho_camb_ratio)
        - np.log(rho_python_ratio)
    )
)

print(
    "max absolute error = "
    f"{max_abs_error:.3e}"
)

print(
    "max relative error = "
    f"{max_rel_error:.3e}"
)

print(
    "max |Delta ln(rho/rho0)| = "
    f"{max_log_error:.3e}"
)

assert max_log_error < 5.0e-5

print("PASS: spline rho_DE obeys the continuity equation")


# ============================================================
# TEST 4: C1 smoothness at internal spline knots
# ============================================================

banner("TEST 4: first derivative is continuous at internal knots")

# A cubic spline must have continuous first derivative across internal
# nodes.  We estimate the left- and right-sided derivatives directly
# from CAMB w(z).
#
# We do NOT apply this test to z_max because the implementation imposes
# an external hard boundary w=-1 above z_max.

internal_knots = SPLINE_Z[1:-1]

h_values = [
    1.0e-3,
    3.0e-4,
    1.0e-4,
    3.0e-5,
]

for zk in internal_knots:

    print()
    print(
        f"internal knot z = {zk:.6f}"
    )

    best_difference = np.inf

    for h in h_values:

        z_sample = np.array([
            zk - 2.0*h,
            zk - h,
            zk,
            zk + h,
            zk + 2.0*h,
        ])

        _, ws = camb_rho_w_at_z(
            results,
            z_sample,
        )

        # Second-order one-sided derivative estimates:
        #
        # left:
        #   f'(x) = [3f(x)-4f(x-h)+f(x-2h)]/(2h)
        #
        # right:
        #   f'(x) = [-3f(x)+4f(x+h)-f(x+2h)]/(2h)

        deriv_left = (
            3.0 * ws[2]
            - 4.0 * ws[1]
            + ws[0]
        ) / (
            2.0 * h
        )

        deriv_right = (
            -3.0 * ws[2]
            + 4.0 * ws[3]
            - ws[4]
        ) / (
            2.0 * h
        )

        difference = abs(
            deriv_left
            - deriv_right
        )

        best_difference = min(
            best_difference,
            difference,
        )

        print(
            f"h={h:.1e}   "
            f"dw/dz left={deriv_left:+.9e}   "
            f"right={deriv_right:+.9e}   "
            f"|Delta|={difference:.3e}"
        )

    # The finite-difference mismatch converges toward zero as h shrinks.
    assert best_difference < 1.0e-5

print()
print("PASS: dw/dz is continuous across all internal spline knots")


# ============================================================
# TEST 5: natural spline boundary condition at z=0
# ============================================================

banner("TEST 5: natural-spline second derivative vanishes at z=0")

# A natural cubic spline has w''=0 at its spline endpoints.
#
# At the low-z endpoint z=0 we can test this directly without any
# complication from the imposed high-z LambdaCDM boundary.

h_values = [
    1.0e-2,
    5.0e-3,
    2.0e-3,
    1.0e-3,
]

second_derivatives = []

for h in h_values:

    z_sample = np.array([
        0.0,
        h,
        2.0*h,
        3.0*h,
    ])

    _, ws = camb_rho_w_at_z(
        results,
        z_sample,
    )

    # One-sided O(h^2) approximation:
    #
    # f''(0) ~ [2f0 - 5f1 + 4f2 - f3] / h^2

    second = (
        2.0 * ws[0]
        - 5.0 * ws[1]
        + 4.0 * ws[2]
        - ws[3]
    ) / (
        h**2
    )

    second_derivatives.append(
        second
    )

    print(
        f"h={h:.1e}   "
        f"d2w/dz2(z=0)={second:+.9e}"
    )

best_second = np.min(
    np.abs(second_derivatives)
)

assert best_second < 1.0e-4

print("PASS: low-z endpoint satisfies the natural-spline condition")


# ============================================================
# TEST 6: high-z boundary
# ============================================================

banner("TEST 6: exact w=-1 above zmax")

z_high = np.array([
    ZMAX + 1.0e-8,
    ZMAX + 1.0e-6,
    ZMAX + 1.0e-4,
    3.0,
    5.0,
    10.0,
    30.0,
    100.0,
])

rho_high, w_high = camb_rho_w_at_z(
    results,
    z_high,
)

for zi, wi in zip(
        z_high,
        w_high):

    print(
        f"z={zi:10.6f}   "
        f"w={wi:+.12f}   "
        f"|w+1|={abs(wi+1.0):.3e}"
    )

assert np.allclose(
    w_high,
    -1.0,
    atol=0,
    rtol=0,
)

print("PASS: w(z)=-1 exactly for z>zmax")


# ============================================================
# TEST 7: rho_DE is constant above zmax
# ============================================================

banner("TEST 7: rho_DE is constant above zmax")

z_rho_high = np.array([
    ZMAX + 1.0e-6,
    3.0,
    5.0,
    10.0,
    30.0,
    100.0,
])

rho_high, _ = camb_rho_w_at_z(
    results,
    z_rho_high,
)

rho_high_ratio = (
    rho_high
    / rho_high[0]
)

for zi, ratio in zip(
        z_rho_high,
        rho_high_ratio):

    print(
        f"z={zi:10.6f}   "
        f"rho/rho(first high-z point)={ratio:.12e}"
    )

max_high_rho_difference = np.max(
    np.abs(
        rho_high_ratio
        - 1.0
    )
)

print(
    "max high-z rho variation = "
    f"{max_high_rho_difference:.3e}"
)

assert max_high_rho_difference < 1.0e-11

print("PASS: rho_DE remains constant above zmax")


# ============================================================
# TEST 8: behavior at the high-z boundary itself
# ============================================================

banner("TEST 8: inspect behavior immediately around zmax")

# This is primarily diagnostic.  Because the implementation switches to
# w=-1 above zmax, continuity at zmax requires the final spline node to
# be exactly -1.  We chose w_nodes[-1] = -1 above precisely for this reason.

h_values = [
    1.0e-3,
    1.0e-4,
    1.0e-5,
    1.0e-6,
]

for h in h_values:

    z_sample = np.array([
        ZMAX - h,
        ZMAX,
        ZMAX + h,
    ])

    _, ws = camb_rho_w_at_z(
        results,
        z_sample,
    )

    deriv_left = (
        ws[1]
        - ws[0]
    ) / h

    deriv_right = (
        ws[2]
        - ws[1]
    ) / h

    print(
        f"h={h:.1e}   "
        f"w(zmax-h)={ws[0]:+.12f}   "
        f"w(zmax)={ws[1]:+.12f}   "
        f"w(zmax+h)={ws[2]:+.12f}"
    )

    print(
        f"           "
        f"dw/dz left={deriv_left:+.6e}   "
        f"right={deriv_right:+.6e}"
    )

assert abs(
    camb_rho_w_at_z(
        results,
        [ZMAX],
    )[1][0]
    + 1.0
) < 1.0e-12

print("PASS: w(z) is continuous through the imposed high-z boundary")


# ============================================================
# Plots
# ============================================================

banner("GENERATING CONSISTENCY PLOTS")

z_plot = np.linspace(
    0.0,
    6.0,
    4001,
)

rho_plot, w_plot = camb_rho_w_at_z(
    results,
    z_plot,
)

rho_plot = (
    rho_plot
    / rho_plot[0]
)

rho_python_plot = rho_ratio_from_w(
    z_plot,
    w_plot,
)


# ------------------------------------------------------------
# Plot 1: w(z)
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    z_plot,
    w_plot,
    lw=2,
    label="CAMB cubic spline",
)

plt.scatter(
    SPLINE_Z,
    w_nodes,
    zorder=5,
    label="input nodes",
)

plt.axhline(
    -1.0,
    ls="--",
    lw=1,
)

plt.axvline(
    ZMAX,
    ls=":",
    lw=1,
    label=r"$z_{\rm max}$",
)

plt.xlabel(
    r"$z$"
)

plt.ylabel(
    r"$w_{\rm DE}(z)$"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "cubic_spline_consistency_w.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# Plot 2: rho_DE
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    z_plot,
    rho_plot,
    lw=2,
    label="CAMB",
)

plt.plot(
    z_plot,
    rho_python_plot,
    ls="--",
    lw=1.5,
    label="Python continuity integral",
)

plt.axvline(
    ZMAX,
    ls=":",
    lw=1,
)

plt.xlabel(
    r"$z$"
)

plt.ylabel(
    r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "cubic_spline_consistency_rho.pdf",
    bbox_inches="tight",
)

plt.show()


banner("ALL CUBIC-SPLINE TESTS PASSED")
