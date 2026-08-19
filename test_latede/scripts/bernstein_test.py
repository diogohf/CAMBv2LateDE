import numpy as np
import matplotlib.pyplot as plt
import camb
import os

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

    omnuh2=0,
    num_nu_massless=3.044,
    num_nu_massive=0,
    nu_mass_degeneracies=[0],
    nu_mass_numbers=[0],

    As=2.100549e-9,
    ns=0.9660499,
    YHe=0.246,

    WantTransfer=True,
)

def banner(name):
    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

def camb_rho_w_at_z(results, z_values):
    z_values = np.atleast_1d(z_values).astype(float)
    a_values = 1.0 / (1.0 + z_values)

    order = np.argsort(a_values)

    rho_sorted, w_sorted = results.get_dark_energy_rho_w(
        a_values[order]
    )

    rho_sorted = np.asarray(rho_sorted, dtype=float)
    w_sorted = np.asarray(w_sorted, dtype=float)

    rho = np.empty_like(rho_sorted)
    w = np.empty_like(w_sorted)

    rho[order] = rho_sorted
    w[order] = w_sorted

    return rho, w

def rho_ratio_from_w(z, w):
    """
    Independent numerical reconstruction:

        rho_DE(z)/rho_DE(0)
          = exp[3 int_0^z (1+w)/(1+z) dz].
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)

    integrand = (1.0 + w) / (1.0 + z)

    cumulative = np.zeros_like(z)
    dz = np.diff(z)

    cumulative[1:] = np.cumsum(
        0.5 * (integrand[:-1] + integrand[1:]) * dz
    )

    return np.exp(3.0 * cumulative)

def compare_results(
        label,
        results_a,
        results_b,
        z,
        atol_w=1.0e-10,
        atol_rho=1.0e-10):

    rho_a, w_a = camb_rho_w_at_z(results_a, z)
    rho_b, w_b = camb_rho_w_at_z(results_b, z)

    rho_a = rho_a / rho_a[0]
    rho_b = rho_b / rho_b[0]

    max_w = np.max(np.abs(w_a - w_b))
    max_rho = np.max(np.abs(rho_a - rho_b))

    print(
        f"{label}: "
        f"max |Delta w|={max_w:.3e}, "
        f"max |Delta rho/rho0|={max_rho:.3e}"
    )

    assert max_w < atol_w
    assert max_rho < atol_rho


# ============================================================
# BERNSTEIN: DEmodel = 7
# ============================================================

ZMAX = 3.5
redshift = np.linspace(0.0, 10.0, 20001)

results_lcdm = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=1,
        w0=-1.0,
    )
)


def bern_results(
        b0=0.0,
        b1=0.0,
        b2=0.0,
        b3=0.0,
        delta=1.0):

    pars = camb.set_params(
        **base_params,
        DEmodel=7,
        bern_b0=b0,
        bern_b1=b1,
        bern_b2=b2,
        bern_b3=b3,
        bern_zmax=ZMAX,
        bern_delta=delta,
    )

    return camb.get_results(pars)


# ------------------------------------------------------------
# TEST 1: LambdaCDM limit
# ------------------------------------------------------------

banner("TEST 1: b0=b1=b2=b3=-1 give LambdaCDM")

results = bern_results(
    b0=-1.0,
    b1=-1.0,
    b2=-1.0,
    b3=-1.0,
)

compare_results(
    "Bernstein LambdaCDM vs constant-w LambdaCDM",
    results,
    results_lcdm,
    redshift,
)

print("PASS: Bernstein expansion reproduces LambdaCDM")


# ------------------------------------------------------------
# TEST 2: equal coefficients -> constant function
# ------------------------------------------------------------

banner("TEST 2: equal coefficients reproduce constant w below zmax")

results_const = bern_results(
    b0=-0.8,
    b1=-0.8,
    b2=-0.8,
    b3=-0.8,
)

z_low = np.linspace(
    0.0,
    0.999 * ZMAX,
    501,
)

_, w_low = camb_rho_w_at_z(
    results_const,
    z_low,
)

max_error = np.max(
    np.abs(
        w_low + 0.8
    )
)

print(
    "max |w(z)+0.8| below zmax = "
    f"{max_error:.3e}"
)

assert max_error < 1.0e-12

print("PASS: equal Bernstein coefficients give a constant function")


# ------------------------------------------------------------
# TEST 3: Bernstein basis sanity checks
# ------------------------------------------------------------

banner("TEST 3: cubic Bernstein basis sanity checks")

z_basis = np.linspace(
    0.0,
    ZMAX,
    501,
)

results_zero = bern_results()

_, w_zero = camb_rho_w_at_z(
    results_zero,
    z_basis,
)

unit_results = [
    bern_results(b0=1.0),
    bern_results(b1=1.0),
    bern_results(b2=1.0),
    bern_results(b3=1.0),
]

B = []

for res in unit_results:

    _, wi = camb_rho_w_at_z(
        res,
        z_basis,
    )

    B.append(
        wi - w_zero
    )

B = np.asarray(B)

# Partition of unity:
#
#   B0 + B1 + B2 + B3 = 1.

partition_error = np.max(
    np.abs(
        np.sum(B, axis=0)
        - 1.0
    )
)

minimum_basis_value = np.min(B)
maximum_basis_value = np.max(B)

print(
    f"partition-of-unity error = {partition_error:.3e}"
)

print(
    "basis range = "
    f"[{minimum_basis_value:.12f}, "
    f"{maximum_basis_value:.12f}]"
)

assert partition_error < 1.0e-12
assert minimum_basis_value > -1.0e-12
assert maximum_basis_value < 1.0 + 1.0e-12

# Endpoint property:
# at either edge exactly one Bernstein basis function is 1.
endpoint_target = np.array([
    0.0,
    0.0,
    0.0,
    1.0,
])

endpoint0_error = np.max(
    np.abs(
        np.sort(B[:, 0])
        - endpoint_target
    )
)

endpoint1_error = np.max(
    np.abs(
        np.sort(B[:, -1])
        - endpoint_target
    )
)

print(
    f"endpoint errors = "
    f"{endpoint0_error:.3e}, "
    f"{endpoint1_error:.3e}"
)

assert endpoint0_error < 1.0e-12
assert endpoint1_error < 1.0e-12

# For cubic Bernstein at x=1/2:
#
#   [B0, B1, B2, B3]
#     = [1/8, 3/8, 3/8, 1/8].
imid = len(z_basis) // 2

midpoint_target = np.sort(
    np.array([
        1.0 / 8.0,
        3.0 / 8.0,
        3.0 / 8.0,
        1.0 / 8.0,
    ])
)

midpoint_error = np.max(
    np.abs(
        np.sort(B[:, imid])
        - midpoint_target
    )
)

print(
    f"midpoint cubic-basis error = {midpoint_error:.3e}"
)

assert midpoint_error < 1.0e-12

print("PASS: Bernstein basis has the correct cubic structure")


# ------------------------------------------------------------
# TEST 4: arbitrary coefficients and convex hull
# ------------------------------------------------------------

banner("TEST 4: arbitrary coefficients obey basis reconstruction and convex hull")

coeff = np.array([
    -0.90,
    -0.70,
    -1.20,
    -0.90,
])

results = bern_results(
    b0=coeff[0],
    b1=coeff[1],
    b2=coeff[2],
    b3=coeff[3],
)

_, w_bern_basis = camb_rho_w_at_z(
    results,
    z_basis,
)

w_expected = (
    w_zero
    + np.sum(
        coeff[:, None] * B,
        axis=0,
    )
)

reconstruction_error = np.max(
    np.abs(
        w_bern_basis
        - w_expected
    )
)

print(
    "max basis-reconstruction error = "
    f"{reconstruction_error:.3e}"
)

assert reconstruction_error < 1.0e-12

# Convex-hull property:
#
#   min(b_i) <= w(z) <= max(b_i).

lower_violation = max(
    0.0,
    np.min(coeff)
    - np.min(w_bern_basis),
)

upper_violation = max(
    0.0,
    np.max(w_bern_basis)
    - np.max(coeff),
)

print(
    f"convex-hull lower violation={lower_violation:.3e}"
)
print(
    f"convex-hull upper violation={upper_violation:.3e}"
)

assert lower_violation < 1.0e-12
assert upper_violation < 1.0e-12

print("PASS: Bernstein curve remains inside coefficient convex hull")


# ------------------------------------------------------------
# TEST 5: independent rho_DE reconstruction
# ------------------------------------------------------------

banner("TEST 5: independent rho_DE reconstruction")

rho_camb, w_camb = camb_rho_w_at_z(
    results,
    redshift,
)

rho_camb = rho_camb / rho_camb[0]

rho_python = rho_ratio_from_w(
    redshift,
    w_camb,
)

max_log_error = np.max(
    np.abs(
        np.log(rho_camb)
        - np.log(rho_python)
    )
)

print(
    "max |Delta ln(rho/rho0)| = "
    f"{max_log_error:.3e}"
)

assert max_log_error < 5.0e-5

print("PASS: rho_DE obeys the continuity equation")


# ------------------------------------------------------------
# TEST 6: high-z transition
# ------------------------------------------------------------

banner("TEST 6: bern_delta only controls the high-z transition")

deltas = [
    0.25,
    0.5,
    1.0,
    2.0,
]

z_low = np.linspace(
    0.0,
    0.999 * ZMAX,
    501,
)

low_curves = {}

for delta in deltas:

    res = bern_results(
        b0=coeff[0],
        b1=coeff[1],
        b2=coeff[2],
        b3=coeff[3],
        delta=delta,
    )

    _, wlow = camb_rho_w_at_z(
        res,
        z_low,
    )

    low_curves[delta] = wlow

reference = low_curves[deltas[0]]

for delta in deltas[1:]:

    maxdiff = np.max(
        np.abs(
            low_curves[delta]
            - reference
        )
    )

    print(
        f"delta={delta:4.2f}: "
        f"max low-z difference={maxdiff:.3e}"
    )

    assert maxdiff < 1.0e-12

print("PASS: bern_delta leaves the polynomial region unchanged")


# ------------------------------------------------------------
# High-z asymptotic approach to LambdaCDM
# ------------------------------------------------------------

z_high = np.array([
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
])

for delta in deltas:

    res = bern_results(
        b0=coeff[0],
        b1=coeff[1],
        b2=coeff[2],
        b3=coeff[3],
        delta=delta,
    )

    _, whigh = camb_rho_w_at_z(
        res,
        z_high,
    )

    deviation = np.abs(
        1.0 + whigh
    )

    print()
    print(
        f"delta = {delta:.2f}"
    )

    for zi, wi, di in zip(
            z_high,
            whigh,
            deviation):

        print(
            f"z={zi:8.1f}   "
            f"w={wi:+.12f}   "
            f"|1+w|={di:.3e}"
        )

    assert deviation[-1] < deviation[0]

print()
print(
    "PASS: Bernstein high-z transition approaches LambdaCDM"
)


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

banner("GENERATING DIAGNOSTIC PLOTS")

z_plot = np.linspace(
    0.0,
    100.0,
    20001,
)


# ------------------------------------------------------------
# Reference constant-w model
# ------------------------------------------------------------

cosmology_w = camb.set_params(
    **base_params,
    DEmodel=1,
    w0=-0.8,
)

results_w = camb.get_results(
    cosmology_w
)

rho_w, wde_w = camb_rho_w_at_z(
    results_w,
    z_plot,
)

rho_w = (
    rho_w
    / rho_w[0]
)


# ------------------------------------------------------------
# Bernstein LambdaCDM realization
# ------------------------------------------------------------

cosmology_bern_lcdm = camb.set_params(
    **base_params,
    DEmodel=7,
    bern_b0=-1.0,
    bern_b1=-1.0,
    bern_b2=-1.0,
    bern_b3=-1.0,
    bern_zmax=ZMAX,
    bern_delta=1.0,
)

results_bern_lcdm = camb.get_results(
    cosmology_bern_lcdm
)

rho_bern_lcdm, wde_bern_lcdm = camb_rho_w_at_z(
    results_bern_lcdm,
    z_plot,
)

rho_bern_lcdm = (
    rho_bern_lcdm
    / rho_bern_lcdm[0]
)


# ------------------------------------------------------------
# Bernstein constant-w realization
# ------------------------------------------------------------

cosmology_bern_w = camb.set_params(
    **base_params,
    DEmodel=7,
    bern_b0=-0.8,
    bern_b1=-0.8,
    bern_b2=-0.8,
    bern_b3=-0.8,
    bern_zmax=ZMAX,
    bern_delta=1.0,
)

results_bern_w = camb.get_results(
    cosmology_bern_w
)

rho_bern_w, wde_bern_w = camb_rho_w_at_z(
    results_bern_w,
    z_plot,
)

rho_bern_w = (
    rho_bern_w
    / rho_bern_w[0]
)


# ------------------------------------------------------------
# Non-trivial Bernstein realization
# ------------------------------------------------------------

cosmology_bern = camb.set_params(
    **base_params,
    DEmodel=7,
    bern_b0=-0.9,
    bern_b1=-0.7,
    bern_b2=-1.2,
    bern_b3=-0.9,
    bern_zmax=ZMAX,
    bern_delta=1.0,
)

results_bern = camb.get_results(
    cosmology_bern
)

rho_bern, wde_bern = camb_rho_w_at_z(
    results_bern,
    z_plot,
)

rho_bern = (
    rho_bern
    / rho_bern[0]
)


# ------------------------------------------------------------
# FIGURE 1:
# LambdaCDM and constant-w reduction tests
# ------------------------------------------------------------

fig, ax = plt.subplots(
    2,
    2,
    figsize=(10, 7),
)

ax[0, 0].plot(
    z_plot,
    wde_bern_lcdm,
    lw=2,
    label=r"Bernstein: $b_i=-1$",
)

ax[0, 0].axhline(
    -1.0,
    ls="--",
    lw=1.5,
    label=r"$\Lambda$CDM",
)

ax[0, 0].set_xlim(
    0,
    10,
)

ax[0, 0].set_ylabel(
    r"$w_{\rm DE}(z)$"
)

ax[0, 0].set_title(
    r"$\Lambda$CDM limit"
)

ax[0, 0].legend(
    loc="best"
)


ax[0, 1].plot(
    z_plot,
    rho_bern_lcdm,
    lw=2,
    label="Bernstein",
)

ax[0, 1].axhline(
    1.0,
    ls="--",
    lw=1.5,
    label=r"$\Lambda$CDM",
)

ax[0, 1].set_xlim(
    0,
    10,
)

ax[0, 1].set_ylabel(
    r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$"
)

ax[0, 1].set_title(
    r"$\Lambda$CDM density"
)

ax[0, 1].legend(
    loc="best"
)


ax[1, 0].plot(
    z_plot,
    wde_w,
    lw=2,
    label=r"constant $w=-0.8$",
)

ax[1, 0].plot(
    z_plot,
    wde_bern_w,
    lw=2,
    ls="--",
    label=r"Bernstein: $b_i=-0.8$",
)

ax[1, 0].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[1, 0].set_xlim(
    0,
    10,
)

ax[1, 0].set_xlabel(
    r"$z$"
)

ax[1, 0].set_ylabel(
    r"$w_{\rm DE}(z)$"
)

ax[1, 0].set_title(
    r"constant-$w$ reduction below $z_{\max}$"
)

ax[1, 0].legend(
    loc="best"
)


ax[1, 1].plot(
    z_plot,
    rho_w,
    lw=2,
    label=r"constant $w=-0.8$",
)

ax[1, 1].plot(
    z_plot,
    rho_bern_w,
    lw=2,
    ls="--",
    label=r"Bernstein: $b_i=-0.8$",
)

ax[1, 1].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[1, 1].set_xlim(
    0,
    10,
)

ax[1, 1].set_xlabel(
    r"$z$"
)

ax[1, 1].set_ylabel(
    r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$"
)

ax[1, 1].set_title(
    r"density comparison"
)

ax[1, 1].legend(
    loc="best"
)

fig.tight_layout()

fig.savefig(
    "bernstein_reduction_tests.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 2:
# Non-trivial Bernstein realization and transition at zmax
# ------------------------------------------------------------

fig, ax = plt.subplots(
    1,
    3,
    figsize=(13, 4),
)

ax[0].plot(
    z_plot,
    wde_bern,
    lw=2,
)

ax[0].axhline(
    -1.0,
    ls="--",
    lw=1,
)

ax[0].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[0].set_xlim(
    0,
    10,
)

ax[0].set_xlabel(
    r"$z$"
)

ax[0].set_ylabel(
    r"$w_{\rm DE}(z)$"
)

ax[0].set_title(
    r"$b=(-0.9,-0.7,-1.2,-0.9)$"
)


ax[1].plot(
    z_plot,
    rho_bern,
    lw=2,
)

ax[1].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[1].set_xlim(
    0,
    10,
)

ax[1].set_xlabel(
    r"$z$"
)

ax[1].set_ylabel(
    r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$"
)

ax[1].set_title(
    r"dark-energy density"
)


transition_mask = (
    (z_plot > 2.8)
    & (z_plot < 6.0)
)

ax[2].plot(
    z_plot[transition_mask],
    wde_bern[transition_mask],
    lw=2,
)

ax[2].axhline(
    -1.0,
    ls="--",
    lw=1,
)

ax[2].axvline(
    ZMAX,
    ls=":",
    lw=1,
    label=r"$z_{\max}=3.5$",
)

ax[2].set_xlabel(
    r"$z$"
)

ax[2].set_ylabel(
    r"$w_{\rm DE}(z)$"
)

ax[2].set_title(
    r"high-$z$ matching region"
)

ax[2].legend(
    loc="best"
)

fig.tight_layout()

fig.savefig(
    "bernstein_nontrivial_transition.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 3:
# Explicit cubic Bernstein basis functions B0--B3
# ------------------------------------------------------------

fig, ax = plt.subplots(
    1,
    1,
    figsize=(7, 5),
)

for i in range(4):

    ax.plot(
        z_basis,
        B[i],
        lw=2,
        label=rf"$B_{i,3}$",
    )

ax.axvline(
    ZMAX,
    lw=1,
    ls=":",
)

ax.set_xlim(
    0,
    ZMAX,
)

ax.set_ylim(
    -0.05,
    1.05,
)

ax.set_xlabel(
    r"$z$"
)

ax.set_ylabel(
    r"$B_{i,3}[x(z)]$"
)

ax.set_title(
    "Cubic Bernstein basis used by LateDE"
)

ax.legend(
    loc="best"
)

fig.tight_layout()

fig.savefig(
    "bernstein_basis_functions.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 4:
# Convex-hull interpretation
# ------------------------------------------------------------

fig, ax = plt.subplots(
    1,
    1,
    figsize=(7, 5),
)

ax.plot(
    z_basis,
    w_bern_basis,
    lw=2,
    label=r"$w_{\rm DE}(z)$",
)

for bi in coeff:

    ax.axhline(
        bi,
        ls=":",
        lw=1,
    )

ax.axhline(
    np.min(coeff),
    ls="--",
    lw=1.5,
    label=r"$\min(b_i)$",
)

ax.axhline(
    np.max(coeff),
    ls="--",
    lw=1.5,
    label=r"$\max(b_i)$",
)

ax.set_xlim(
    0,
    ZMAX,
)

ax.set_xlabel(
    r"$z$"
)

ax.set_ylabel(
    r"$w_{\rm DE}(z)$"
)

ax.set_title(
    "Bernstein convex-hull property"
)

ax.legend(
    loc="best"
)

fig.tight_layout()

fig.savefig(
    "bernstein_convex_hull.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 5:
# Effect of bern_delta on the high-z transition
# ------------------------------------------------------------

z_delta = np.linspace(
    ZMAX,
    30.0,
    4000,
)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(10, 4),
)

for delta in deltas:

    res_delta = bern_results(
        b0=coeff[0],
        b1=coeff[1],
        b2=coeff[2],
        b3=coeff[3],
        delta=delta,
    )

    _, w_delta = camb_rho_w_at_z(
        res_delta,
        z_delta,
    )

    ax[0].plot(
        z_delta,
        w_delta,
        lw=2,
        label=rf"$\Delta={delta}$",
    )

    ax[1].plot(
        z_delta,
        np.abs(1.0 + w_delta),
        lw=2,
        label=rf"$\Delta={delta}$",
    )

ax[0].axhline(
    -1.0,
    ls="--",
    lw=1,
)

ax[0].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[0].set_xlabel(
    r"$z$"
)

ax[0].set_ylabel(
    r"$w_{\rm DE}(z)$"
)

ax[0].set_title(
    r"transition to $w=-1$"
)

ax[0].legend(
    loc="best"
)


ax[1].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[1].set_yscale(
    "log"
)

ax[1].set_xlabel(
    r"$z$"
)

ax[1].set_ylabel(
    r"$|1+w_{\rm DE}(z)|$"
)

ax[1].set_title(
    r"asymptotic approach to $\Lambda$CDM"
)

ax[1].legend(
    loc="best"
)

fig.tight_layout()

fig.savefig(
    "bernstein_delta_transition.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 6:
# Continuity equation residual
#
#   d ln rho_DE / d ln(1+z) = 3(1+w)
# ------------------------------------------------------------

z_cons = np.linspace(
    0.0,
    100.0,
    50001,
)

rho_cons, w_cons = camb_rho_w_at_z(
    results_bern,
    z_cons,
)

rho_cons = (
    rho_cons
    / rho_cons[0]
)

lnrho = np.log(
    rho_cons
)

ln1pz = np.log1p(
    z_cons
)

dlnrho_dln1pz = np.gradient(
    lnrho,
    ln1pz,
    edge_order=2,
)

rhs = (
    3.0
    * (1.0 + w_cons)
)

residual = (
    dlnrho_dln1pz
    - rhs
)

mask_cons = (
    (z_cons > 0.02)
    & (z_cons < 90.0)
)

max_conservation_error = np.max(
    np.abs(
        residual[mask_cons]
    )
)

print(
    "max differential conservation error = "
    f"{max_conservation_error:.3e}"
)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(10, 4),
)

ax[0].plot(
    z_cons[mask_cons],
    dlnrho_dln1pz[mask_cons],
    lw=2,
    label=r"$d\ln\rho_{\rm DE}/d\ln(1+z)$",
)

ax[0].plot(
    z_cons[mask_cons],
    rhs[mask_cons],
    lw=1.5,
    ls="--",
    label=r"$3(1+w)$",
)

ax[0].set_xlim(
    0,
    10,
)

ax[0].set_xlabel(
    r"$z$"
)

ax[0].set_ylabel(
    "continuity-equation terms"
)

ax[0].set_title(
    "direct conservation check"
)

ax[0].legend(
    loc="best"
)


ax[1].plot(
    z_cons[mask_cons],
    residual[mask_cons],
    lw=1.5,
)

ax[1].axhline(
    0.0,
    ls="--",
    lw=1,
)

ax[1].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[1].set_xlim(
    0,
    10,
)

ax[1].set_xlabel(
    r"$z$"
)

ax[1].set_ylabel(
    r"$d\ln\rho/d\ln(1+z)-3(1+w)$"
)

ax[1].set_title(
    "conservation residual"
)

fig.tight_layout()

fig.savefig(
    "bernstein_conservation_check.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 7:
# CAMB rho_DE versus independent numerical integral
# ------------------------------------------------------------

z_rho = np.linspace(
    0.0,
    10.0,
    10001,
)

rho_camb_plot, w_camb_plot = camb_rho_w_at_z(
    results_bern,
    z_rho,
)

rho_camb_plot = (
    rho_camb_plot
    / rho_camb_plot[0]
)

rho_python_plot = rho_ratio_from_w(
    z_rho,
    w_camb_plot,
)

fractional_rho_error = (
    rho_camb_plot
    / rho_python_plot
    - 1.0
)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(10, 4),
)

ax[0].plot(
    z_rho,
    rho_camb_plot,
    lw=2,
    label="CAMB",
)

ax[0].plot(
    z_rho,
    rho_python_plot,
    lw=1.5,
    ls="--",
    label="independent integral",
)

ax[0].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[0].set_xlabel(
    r"$z$"
)

ax[0].set_ylabel(
    r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$"
)

ax[0].set_title(
    r"$\rho_{\rm DE}$ reconstruction"
)

ax[0].legend(
    loc="best"
)


ax[1].plot(
    z_rho,
    fractional_rho_error,
    lw=1.5,
)

ax[1].axhline(
    0.0,
    ls="--",
    lw=1,
)

ax[1].axvline(
    ZMAX,
    ls=":",
    lw=1,
)

ax[1].set_xlabel(
    r"$z$"
)

ax[1].set_ylabel(
    r"$\rho_{\rm CAMB}/\rho_{\rm int}-1$"
)

ax[1].set_title(
    "density reconstruction residual"
)

fig.tight_layout()

fig.savefig(
    "bernstein_rho_reconstruction.pdf",
    bbox_inches="tight",
)

plt.show()


banner("ALL BERNSTEIN TESTS PASSED")
