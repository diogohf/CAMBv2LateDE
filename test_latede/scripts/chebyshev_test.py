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
# CHEBYSHEV: DEmodel = 6
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


def cheb_results(
        c0=0.0,
        c1=0.0,
        c2=0.0,
        c3=0.0,
        delta=1.0):

    pars = camb.set_params(
        **base_params,
        DEmodel=6,
        cheb_c0=c0,
        cheb_c1=c1,
        cheb_c2=c2,
        cheb_c3=c3,
        cheb_zmax=ZMAX,
        cheb_delta=delta,
    )

    return camb.get_results(pars)


# ------------------------------------------------------------
# TEST 1: LambdaCDM limit
# ------------------------------------------------------------

banner("TEST 1: c0=1 and c1=c2=c3=0 give LambdaCDM")

results = cheb_results(
    c0=1.0,
)

compare_results(
    "Chebyshev LambdaCDM vs constant-w LambdaCDM",
    results,
    results_lcdm,
    redshift,
)

print("PASS: Chebyshev expansion reproduces LambdaCDM")


# ------------------------------------------------------------
# TEST 2: pure c0 mode
# ------------------------------------------------------------

banner("TEST 2: pure c0 gives constant w below zmax")

results_const = cheb_results(
    c0=0.8,
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

print("PASS: c0 controls the constant Chebyshev mode")


# ------------------------------------------------------------
# TEST 3: basis identities
# ------------------------------------------------------------

banner("TEST 3: T0,T1,T2,T3 satisfy Chebyshev identities")

z_basis = np.linspace(
    0.0,
    ZMAX,
    501,
)

results_zero = cheb_results()

_, w_zero = camb_rho_w_at_z(
    results_zero,
    z_basis,
)

unit_results = [
    cheb_results(c0=1.0),
    cheb_results(c1=1.0),
    cheb_results(c2=1.0),
    cheb_results(c3=1.0),
]

responses = []

for res in unit_results:
    _, wi = camb_rho_w_at_z(
        res,
        z_basis,
    )
    responses.append(
        wi - w_zero
    )

R0, R1, R2, R3 = responses

max_r0_error = np.max(
    np.abs(
        R0 + 1.0
    )
)

print(
    "max |unit-c0 response + 1| = "
    f"{max_r0_error:.3e}"
)

assert max_r0_error < 1.0e-12

# Divide out the common implementation sign.
T1 = R1 / R0
T2 = R2 / R0
T3 = R3 / R0

err_T2 = np.max(
    np.abs(
        T2
        - (
            2.0 * T1**2
            - 1.0
        )
    )
)

err_T3 = np.max(
    np.abs(
        T3
        - (
            4.0 * T1**3
            - 3.0 * T1
        )
    )
)

print(
    f"max |T2-(2T1^2-1)| = {err_T2:.3e}"
)
print(
    f"max |T3-(4T1^3-3T1)| = {err_T3:.3e}"
)

assert err_T2 < 1.0e-11
assert err_T3 < 1.0e-11

# Check that T1 spans x=-1 to x=+1, regardless of orientation.
endpoint_abs_error = max(
    abs(abs(T1[0]) - 1.0),
    abs(abs(T1[-1]) - 1.0),
)

endpoint_opposite_error = abs(
    T1[0] + T1[-1]
)

print(
    f"T1 endpoint |x|-error = {endpoint_abs_error:.3e}"
)
print(
    f"T1 opposite-endpoint error = {endpoint_opposite_error:.3e}"
)

assert endpoint_abs_error < 1.0e-12
assert endpoint_opposite_error < 1.0e-12

print("PASS: CAMB uses the intended Chebyshev basis through order 3")


# ------------------------------------------------------------
# TEST 4: coefficient superposition
# ------------------------------------------------------------

banner("TEST 4: arbitrary coefficients obey linear superposition")

coeff = np.array([
    1.00,
    0.20,
    -0.10,
    0.05,
])

results = cheb_results(
    c0=coeff[0],
    c1=coeff[1],
    c2=coeff[2],
    c3=coeff[3],
)

_, w_camb = camb_rho_w_at_z(
    results,
    z_basis,
)

w_expected = (
    w_zero
    + coeff[0] * R0
    + coeff[1] * R1
    + coeff[2] * R2
    + coeff[3] * R3
)

superposition_error = np.max(
    np.abs(
        w_camb
        - w_expected
    )
)

print(
    "max superposition error = "
    f"{superposition_error:.3e}"
)

assert superposition_error < 1.0e-12

print("PASS: Chebyshev coefficients enter linearly")


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

banner("TEST 6: cheb_delta only controls the high-z transition")

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

    res = cheb_results(
        c0=coeff[0],
        c1=coeff[1],
        c2=coeff[2],
        c3=coeff[3],
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

print("PASS: cheb_delta leaves the polynomial region unchanged")


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

    res = cheb_results(
        c0=coeff[0],
        c1=coeff[1],
        c2=coeff[2],
        c3=coeff[3],
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
    "PASS: Chebyshev high-z transition approaches LambdaCDM"
)


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

banner("GENERATING DIAGNOSTIC PLOTS")

# A redshift grid ordered from low -> high z.
z_plot = np.linspace(
    0.0,
    100.0,
    20001,
)

# CAMB expects scale factor ordered from low -> high a.  We therefore use
# camb_rho_w_at_z throughout, which handles the ordering and returns arrays
# in increasing redshift.

# ------------------------------------------------------------
# Reference models used in direct visual reductions
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

rho_w = rho_w / rho_w[0]


cosmology_cheb_lcdm = camb.set_params(
    **base_params,
    DEmodel=6,
    cheb_c0=1.0,
    cheb_c1=0.0,
    cheb_c2=0.0,
    cheb_c3=0.0,
    cheb_zmax=ZMAX,
    cheb_delta=1.0,
)

results_cheb_lcdm = camb.get_results(
    cosmology_cheb_lcdm
)

rho_cheb_lcdm, wde_cheb_lcdm = camb_rho_w_at_z(
    results_cheb_lcdm,
    z_plot,
)

rho_cheb_lcdm = (
    rho_cheb_lcdm
    / rho_cheb_lcdm[0]
)


cosmology_cheb_w = camb.set_params(
    **base_params,
    DEmodel=6,
    cheb_c0=0.8,
    cheb_c1=0.0,
    cheb_c2=0.0,
    cheb_c3=0.0,
    cheb_zmax=ZMAX,
    cheb_delta=1.0,
)

results_cheb_w = camb.get_results(
    cosmology_cheb_w
)

rho_cheb_w, wde_cheb_w = camb_rho_w_at_z(
    results_cheb_w,
    z_plot,
)

rho_cheb_w = (
    rho_cheb_w
    / rho_cheb_w[0]
)


cosmology_cheb = camb.set_params(
    **base_params,
    DEmodel=6,
    cheb_c0=1.0,
    cheb_c1=0.2,
    cheb_c2=-0.1,
    cheb_c3=0.05,
    cheb_zmax=ZMAX,
    cheb_delta=1.0,
)

results_cheb = camb.get_results(
    cosmology_cheb
)

rho_cheb, wde_cheb = camb_rho_w_at_z(
    results_cheb,
    z_plot,
)

rho_cheb = (
    rho_cheb
    / rho_cheb[0]
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

# LambdaCDM reduction
ax[0, 0].plot(
    z_plot,
    wde_cheb_lcdm,
    lw=2,
    label=r"Chebyshev: $c_0=1$",
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
    rho_cheb_lcdm,
    lw=2,
    label="Chebyshev",
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


# Constant-w reduction
ax[1, 0].plot(
    z_plot,
    wde_w,
    lw=2,
    label=r"constant $w=-0.8$",
)

ax[1, 0].plot(
    z_plot,
    wde_cheb_w,
    lw=2,
    ls="--",
    label=r"Chebyshev: $c_0=0.8$",
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
    rho_cheb_w,
    lw=2,
    ls="--",
    label=r"Chebyshev: $c_0=0.8$",
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
    "chebyshev_reduction_tests.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 2:
# Non-trivial Chebyshev realization and transition at zmax
# ------------------------------------------------------------

fig, ax = plt.subplots(
    1,
    3,
    figsize=(13, 4),
)

ax[0].plot(
    z_plot,
    wde_cheb,
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
    r"$c=(1,0.2,-0.1,0.05)$"
)


ax[1].plot(
    z_plot,
    rho_cheb,
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
    wde_cheb[transition_mask],
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
    "chebyshev_nontrivial_transition.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 3:
# Explicit Chebyshev basis functions T0--T3
# ------------------------------------------------------------

# Reconstruct the implementation's x-coordinate from T1.
# This remains independent of whether the implementation uses
# x = 2z/zmax - 1 or x = 1 - 2z/zmax.

fig, ax = plt.subplots(
    1,
    1,
    figsize=(7, 5),
)

ax.plot(
    z_basis,
    np.ones_like(z_basis),
    lw=2,
    label=r"$T_0$",
)

ax.plot(
    z_basis,
    T1,
    lw=2,
    label=r"$T_1$",
)

ax.plot(
    z_basis,
    T2,
    lw=2,
    label=r"$T_2$",
)

ax.plot(
    z_basis,
    T3,
    lw=2,
    label=r"$T_3$",
)

ax.axhline(
    0.0,
    lw=1,
    ls="--",
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

ax.set_xlabel(
    r"$z$"
)

ax.set_ylabel(
    r"$T_n[x(z)]$"
)

ax.set_title(
    "Chebyshev basis used by LateDE"
)

ax.legend(
    loc="best"
)

fig.tight_layout()

fig.savefig(
    "chebyshev_basis_functions.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 4:
# Effect of cheb_delta on the high-z transition
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

    res_delta = cheb_results(
        c0=coeff[0],
        c1=coeff[1],
        c2=coeff[2],
        c3=coeff[3],
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
    "chebyshev_delta_transition.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 5:
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
    results_cheb,
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

# Exclude the grid edges where numerical derivatives are least accurate.
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
    "chebyshev_conservation_check.pdf",
    bbox_inches="tight",
)


# ------------------------------------------------------------
# FIGURE 6:
# CAMB rho_DE versus independent numerical integral
# ------------------------------------------------------------

z_rho = np.linspace(
    0.0,
    10.0,
    10001,
)

rho_camb_plot, w_camb_plot = camb_rho_w_at_z(
    results_cheb,
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
    "chebyshev_rho_reconstruction.pdf",
    bbox_inches="tight",
)

plt.show()


banner("ALL CHEBYSHEV TESTS PASSED")
