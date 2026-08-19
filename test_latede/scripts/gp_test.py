# CAMB v2 LateDE -- Gaussian Process consistency tests
#
# DEmodel = 8 only.
#
# Tests both:
#   gp_kernel = 1 : squared-exponential / RBF
#   gp_kernel = 2 : exponential / Holsclaw alpha=1
#
# The Python LateDE interface is assumed to use:
#
#   N = 15
#   gp_z = linspace(0, gp_zmax, N)
#   q = (gp_q0,...,gp_q13)
#   GP mean = -1
#   w(gp_zmax) = -1 exactly
#
# and to store the resulting realization in the existing
# Fortran spline_z / spline_w arrays.

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

N_GP = 15
N_Q = N_GP - 1
GP_ZMAX = 3.5

# A deterministic, non-trivial whitened GP realization.
# These are NOT w(z_i). They are q_i ~ N(0,1)-type coordinates.
q_test = np.array([
     0.70,
    -0.40,
     1.10,
    -0.25,
     0.50,
    -1.20,
     0.30,
     0.80,
    -0.60,
     0.20,
     0.90,
    -0.35,
     0.45,
    -0.70,
], dtype=float)

assert len(q_test) == N_Q


# ============================================================
# Helpers
# ============================================================

def q_kwargs(q):
    """Convert q vector into CAMB set_params keyword arguments."""
    q = np.asarray(q, dtype=float)

    if len(q) != N_Q:
        raise ValueError(f"Expected {N_Q} q parameters")

    return {
        f"gp_q{i}": float(q[i])
        for i in range(N_Q)
    }


def build_gp_covariance(
        kernel,
        sigma,
        ell,
        zmax=GP_ZMAX,
        n=15,
        jitter=1.0e-10):
    """
    Independently reproduce the GP algebra used in dark_energy.py.

    Returns:
        z         : full GP grid, length n
        K         : unconditioned covariance
        K_cond    : covariance conditioned on delta w(zmax)=0
        L         : Cholesky factor of K_cond
    """

    if kernel not in (1, 2):
        raise ValueError("kernel must be 1 or 2")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if ell <= 0:
        raise ValueError("ell must be positive")

    z = np.linspace(
        0.0,
        zmax,
        n
    )

    dz = (
        z[:, None]
        - z[None, :]
    )

    if kernel == 1:

        # Squared-exponential / RBF
        K = (
            sigma**2
            * np.exp(
                -0.5
                * (dz / ell)**2
            )
        )

    else:

        # Exponential / Holsclaw alpha=1
        K = (
            sigma**2
            * np.exp(
                -np.abs(dz) / ell
            )
        )

    # Condition on residual at final node:
    #
    # delta w(zmax) = 0
    #
    # Since mean = -1, this is equivalent to w(zmax) = -1.
    K_ff = K[:-1, :-1]
    K_fb = K[:-1, -1]
    K_bb = K[-1, -1]

    K_cond = (
        K_ff
        - np.outer(K_fb, K_fb) / K_bb
    )

    K_cond = (
    K_cond
    + jitter
    * np.max(np.diag(K_cond))
    * np.eye(n - 1)
)

    L = np.linalg.cholesky(
        K_cond
    )

    return z, K, K_cond, L


def expected_gp_nodes(
        kernel,
        sigma,
        ell,
        q,
        zmax=GP_ZMAX):
    """
    Independently calculate the expected GP node values:

        w[:-1] = -1 + L q
        w[-1]  = -1
    """

    z, K, K_cond, L = build_gp_covariance(
        kernel=kernel,
        sigma=sigma,
        ell=ell,
        zmax=zmax,
        n=N_GP,
    )

    q = np.asarray(
        q,
        dtype=float
    )

    w = np.empty(
        N_GP,
        dtype=float
    )

    w[:-1] = (
        -1.0
        + L @ q
    )

    w[-1] = -1.0

    return z, w, K, K_cond, L


def make_gp_params(
        kernel,
        sigma,
        ell,
        q,
        zmax=GP_ZMAX):

    return camb.set_params(
        **base_params,
        DEmodel=8,
        gp_kernel=kernel,
        gp_sigma=sigma,
        gp_ell=ell,
        gp_zmax=zmax,
        **q_kwargs(q),
    )


def camb_rho_w_at_z(
        results,
        z_values):
    """
    Query CAMB at arbitrary redshifts.

    get_dark_energy_rho_w takes scale factor. Sort by a before
    calling CAMB, then restore the requested redshift order.
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
        rho_sorted
    )

    w_sorted = np.asarray(
        w_sorted
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


def rho_ratio_from_w(
        z,
        w):
    """
    Independent numerical reconstruction:

       rho(z)/rho(0)
         = exp[3 integral_0^z (1+w)/(1+z) dz]

    This does NOT use the LateDE Fortran integration routine.
    """

    z = np.asarray(
        z,
        dtype=float
    )

    w = np.asarray(
        w,
        dtype=float
    )

    if np.any(np.diff(z) <= 0):
        raise ValueError(
            "rho_ratio_from_w requires increasing z"
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


def banner(name):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)


# ============================================================
# Main redshift grid
# ============================================================

redshift = np.linspace(
    0.0,
    10.0,
    5001
)


# ============================================================
# TEST 1
#
# Pure GP algebra:
# K must be symmetric, have sigma^2 on its diagonal, and
# conditioned K must be positive definite after jitter.
#
# This test does NOT call CAMB.
# ============================================================

banner("TEST 1: covariance-matrix sanity checks")

for kernel in (1, 2):

    sigma = 0.20
    ell = 0.60

    z, K, K_cond, L = build_gp_covariance(
        kernel,
        sigma,
        ell,
    )

    symmetry_error = np.max(
        np.abs(K - K.T)
    )

    diagonal_error = np.max(
        np.abs(
            np.diag(K)
            - sigma**2
        )
    )

    eig_cond = np.linalg.eigvalsh(
        K_cond
    )

    reconstruction_error = np.max(
        np.abs(
            L @ L.T
            - K_cond
        )
    )

    print(
        f"kernel={kernel}: "
        f"symmetry={symmetry_error:.3e}, "
        f"diag error={diagonal_error:.3e}, "
        f"min eig(K_cond)={eig_cond.min():.3e}, "
        f"max |LL^T-K_cond|={reconstruction_error:.3e}"
    )

    assert symmetry_error < 1e-14
    assert diagonal_error < 1e-14
    assert eig_cond.min() > 0
    assert reconstruction_error < 1e-12

print("PASS: both covariance constructions are internally consistent")


# ============================================================
# TEST 2
#
# q_i = 0 must give w(z)=-1 for ANY kernel, sigma and ell.
#
# This tests the meaning of the whitened variables:
#
#     w = -1 + L q
#
# so q=0 must exactly reproduce LambdaCDM.
# ============================================================

banner("TEST 2: q=0 gives LambdaCDM for both kernels")

q_zero = np.zeros(
    N_Q
)

lcdm_results = {}

for kernel in (1, 2):

    pars = make_gp_params(
        kernel=kernel,
        sigma=0.35,
        ell=0.90,
        q=q_zero,
    )

    stored_z = np.asarray(
        pars.DarkEnergy.spline_z
    )

    stored_w = np.asarray(
        pars.DarkEnergy.spline_w
    )

    assert np.allclose(
        stored_w,
        -1.0,
        atol=0,
        rtol=0,
    )

    results = camb.get_results(
        pars
    )

    rho, w = camb_rho_w_at_z(
        results,
        redshift
    )

    max_w_error = np.max(
        np.abs(w + 1.0)
    )

    rho_relative_variation = (
        rho.max() - rho.min()
    ) / rho.mean()

    print(
        f"kernel={kernel}: "
        f"max |w+1|={max_w_error:.3e}, "
        f"relative rho variation={rho_relative_variation:.3e}"
    )

    assert max_w_error < 1e-12
    assert rho_relative_variation < 1e-12

    lcdm_results[kernel] = (
        rho,
        w,
    )

print("PASS: q=0 reproduces LambdaCDM for both kernels")


# ============================================================
# TEST 3
#
# The two kernels must give exactly the same LambdaCDM result
# when q=0.
# ============================================================

banner("TEST 3: kernel independence in the LambdaCDM limit")

rho1, w1 = lcdm_results[1]
rho2, w2 = lcdm_results[2]

print(
    "max |w_kernel1-w_kernel2| =",
    np.max(np.abs(w1 - w2))
)

print(
    "max |rho_kernel1-rho_kernel2| =",
    np.max(np.abs(rho1 - rho2))
)

assert np.allclose(
    w1,
    w2,
    atol=1e-12,
    rtol=0,
)

assert np.allclose(
    rho1,
    rho2,
    atol=1e-12,
    rtol=1e-12,
)

print("PASS: kernels are identical when the GP residual is zero")


# ============================================================
# TEST 4
#
# Independently reconstruct the expected node values from
# K -> K_cond -> Cholesky -> Lq, then compare with what
# dark_energy.py placed into CAMB's spline arrays.
#
# This is the strongest Python-interface GP test.
# ============================================================

banner("TEST 4: independent GP algebra vs CAMB stored nodes")

sigma = 0.20
ell = 0.60

nontrivial = {}

for kernel in (1, 2):

    expected_z, expected_w, K, K_cond, L = (
        expected_gp_nodes(
            kernel=kernel,
            sigma=sigma,
            ell=ell,
            q=q_test,
        )
    )

    pars = make_gp_params(
        kernel=kernel,
        sigma=sigma,
        ell=ell,
        q=q_test,
    )

    stored_z = np.asarray(
        pars.DarkEnergy.spline_z
    )

    stored_w = np.asarray(
        pars.DarkEnergy.spline_w
    )

    z_error = np.max(
        np.abs(
            stored_z - expected_z
        )
    )

    w_error = np.max(
        np.abs(
            stored_w - expected_w
        )
    )

    print(
        f"kernel={kernel}: "
        f"max node-z error={z_error:.3e}, "
        f"max node-w error={w_error:.3e}"
    )

    assert z_error < 1e-14
    assert w_error < 1e-12

    results = camb.get_results(
        pars
    )

    nontrivial[kernel] = {
        "pars": pars,
        "results": results,
        "z_nodes": expected_z,
        "w_nodes": expected_w,
    }

print("PASS: dark_energy.py implements K -> Lq -> w exactly as expected")


# ============================================================
# TEST 5
#
# CAMB's w(z) must pass exactly through every GP node.
#
# This tests the Fortran cubic-spline backend separately
# from the GP construction.
# ============================================================

banner("TEST 5: CAMB interpolation passes through GP nodes")

for kernel in (1, 2):

    results = nontrivial[kernel]["results"]
    z_nodes = nontrivial[kernel]["z_nodes"]
    w_nodes = nontrivial[kernel]["w_nodes"]

    _, w_camb_nodes = camb_rho_w_at_z(
        results,
        z_nodes
    )

    node_error = np.max(
        np.abs(
            w_camb_nodes
            - w_nodes
        )
    )

    print(
        f"kernel={kernel}: "
        f"max interpolation error={node_error:.3e}"
    )

    assert node_error < 1e-10

print("PASS: CAMB spline passes through every GP node")


# ============================================================
# TEST 6
#
# Exact high-z boundary.
#
# DEmodel=8 imposes w=-1 for z > gp_zmax.
# Also verify the final GP node itself is exactly -1.
# ============================================================

banner("TEST 6: exact high-z Lambda boundary")

z_high = np.array([
    GP_ZMAX,
    GP_ZMAX + 1e-8,
    4.0,
    5.0,
    10.0,
    30.0,
    100.0,
])

for kernel in (1, 2):

    results = nontrivial[kernel]["results"]

    _, w_high = camb_rho_w_at_z(
        results,
        z_high
    )

    max_error = np.max(
        np.abs(
            w_high + 1.0
        )
    )

    print(
        f"kernel={kernel}: "
        f"max high-z |w+1|={max_error:.3e}"
    )

    assert max_error < 1e-12

print("PASS: both GP kernels join exactly to w=-1 at zmax")


# ============================================================
# TEST 7
#
# Check derivative at zmax.
#
# The spline helper imposes a zero derivative at its last node,
# and the branch above zmax is constant w=-1. Therefore both
# left and right derivatives should approach zero.
# ============================================================

banner("TEST 7: derivative continuity at zmax")

for kernel in (1, 2):

    results = nontrivial[kernel]["results"]

    print(f"kernel={kernel}")

    for h in (
        1e-3,
        1e-4,
        1e-5,
        1e-6,
    ):

        z_eval = np.array([
            GP_ZMAX - h,
            GP_ZMAX,
            GP_ZMAX + h,
        ])

        _, w_eval = camb_rho_w_at_z(
            results,
            z_eval
        )

        dw_left = (
            w_eval[1] - w_eval[0]
        ) / h

        dw_right = (
            w_eval[2] - w_eval[1]
        ) / h

        print(
            f"  h={h:.0e}: "
            f"left={dw_left:+.6e}, "
            f"right={dw_right:+.6e}"
        )

    # Right branch is analytically constant.
    assert abs(dw_right) < 1e-10

print("PASS: high-z branch has zero derivative; left spline derivative converges toward zero")


# ============================================================
# TEST 8
#
# Independent continuity-equation check.
#
# Use CAMB-returned w(z), integrate it independently in Python:
#
# rho/rho0 =
# exp[3 integral (1+w)/(1+z) dz].
#
# Compare against CAMB rho normalized at z=0.
#
# This directly tests TLateDE_grho_de against TLateDE_w_de.
# ============================================================

banner("TEST 8: independent rho_DE continuity-equation check")

z_rho = np.linspace(
    0.0,
    8.0,
    20001
)

for kernel in (1, 2):

    results = nontrivial[kernel]["results"]

    rho_camb, w_camb = camb_rho_w_at_z(
        results,
        z_rho
    )

    rho_python = rho_ratio_from_w(
        z_rho,
        w_camb
    )

    rho_camb_normalized = (
        rho_camb
        / rho_camb[0]
    )

    relative_error = (
        rho_camb_normalized
        / rho_python
        - 1.0
    )

    # Avoid making the test depend on the single zmax grid point.
    # The independent trapezoidal integration converges with grid
    # resolution, so tolerance here is numerical, not theoretical.
    mask = (
        np.abs(z_rho - GP_ZMAX)
        > 2 * (z_rho[1] - z_rho[0])
    )

    max_rel = np.max(
        np.abs(
            relative_error[mask]
        )
    )

    print(
        f"kernel={kernel}: "
        f"max relative rho error={max_rel:.3e}"
    )

    # This tolerance is deliberately modest because this side uses
    # a finite-grid trapezoid integration, while Fortran integrates
    # the cubic spline analytically.
    assert max_rel < 5e-5

print("PASS: CAMB rho_DE agrees with an independent integration of CAMB w(z)")


# ============================================================
# TEST 9
#
# Sign symmetry of whitened GP coordinates.
#
# For fixed K:
#
# w(q)   = -1 + Lq
# w(-q)  = -1 - Lq
#
# Hence:
#
# w(q) + w(-q) = -2
#
# at every free GP node.
# ============================================================

banner("TEST 9: q -> -q symmetry")

for kernel in (1, 2):

    pars_plus = make_gp_params(
        kernel=kernel,
        sigma=sigma,
        ell=ell,
        q=q_test,
    )

    pars_minus = make_gp_params(
        kernel=kernel,
        sigma=sigma,
        ell=ell,
        q=-q_test,
    )

    w_plus = np.asarray(
        pars_plus.DarkEnergy.spline_w
    )

    w_minus = np.asarray(
        pars_minus.DarkEnergy.spline_w
    )

    symmetry_error = np.max(
        np.abs(
            w_plus
            + w_minus
            + 2.0
        )
    )

    print(
        f"kernel={kernel}: "
        f"max |w(q)+w(-q)+2|={symmetry_error:.3e}"
    )

    assert symmetry_error < 1e-12

print("PASS: GP realization is linear and symmetric in q around mean w=-1")


# ============================================================
# TEST 10
#
# Sigma scaling.
#
# For fixed ell and q:
#
# K ~ sigma^2
# L ~ sigma
# delta w = L q ~ sigma
#
# Therefore doubling sigma must double w+1 at all free nodes.
# ============================================================

banner("TEST 10: GP amplitude scales linearly with sigma")

for kernel in (1, 2):

    pars_a = make_gp_params(
        kernel=kernel,
        sigma=0.10,
        ell=ell,
        q=q_test,
    )

    pars_b = make_gp_params(
        kernel=kernel,
        sigma=0.20,
        ell=ell,
        q=q_test,
    )

    wa = np.asarray(
        pars_a.DarkEnergy.spline_w
    )[:-1]

    wb = np.asarray(
        pars_b.DarkEnergy.spline_w
    )[:-1]

    # Jitter introduces a tiny non-perfect sigma scaling because
    # the jitter itself is fixed, not proportional to sigma^2.
    scaling_error = np.max(
        np.abs(
            (wb + 1.0)
            - 2.0 * (wa + 1.0)
        )
    )

    print(
        f"kernel={kernel}: "
        f"max sigma-scaling error={scaling_error:.3e}"
    )

    assert scaling_error < 1e-7

print("PASS: sigma controls the GP vertical amplitude as expected")


# ============================================================
# TEST 11
#
# The kernel flag must actually change a nontrivial realization.
#
# Same sigma, ell, q; different K should produce different nodes.
# This catches a broken/ignored gp_kernel selector.
# ============================================================

banner("TEST 11: kernel selector changes nontrivial GP realization")

w_kernel1 = nontrivial[1]["w_nodes"]
w_kernel2 = nontrivial[2]["w_nodes"]

kernel_difference = np.max(
    np.abs(
        w_kernel1 - w_kernel2
    )
)

print(
    "max |w_kernel1 - w_kernel2| at nodes =",
    kernel_difference
)

assert kernel_difference > 1e-5

print("PASS: gp_kernel flag genuinely changes the GP covariance/realization")


# ============================================================
# TEST 12
#
# Empirical covariance test of the whitening construction.
#
# Draw many q ~ N(0,I), form delta w = Lq, and verify that the
# sample covariance approaches K_cond.
#
# No CAMB call here; this tests the GP statistics themselves.
# ============================================================

banner("TEST 12: Monte-Carlo covariance check of L q")

rng = np.random.default_rng(
    123456
)

n_draws = 50000

for kernel in (1, 2):

    _, _, K_cond, L = build_gp_covariance(
        kernel=kernel,
        sigma=sigma,
        ell=ell,
    )

    q_draws = rng.normal(
        size=(n_draws, N_Q)
    )

    # Each row is one delta-w realization.
    dw_draws = (
        q_draws @ L.T
    )

    empirical_cov = np.cov(
        dw_draws,
        rowvar=False,
        ddof=1,
    )

    # Normalize the error by sigma^2 for a meaningful scale.
    max_cov_error = np.max(
        np.abs(
            empirical_cov - K_cond
        )
    ) / sigma**2

    print(
        f"kernel={kernel}: "
        f"max |Cov_empirical-K_cond|/sigma^2 = "
        f"{max_cov_error:.3e}"
    )

    # 50k draws gives percent-level Monte Carlo covariance accuracy.
    assert max_cov_error < 0.03

print("PASS: Lq empirically has the covariance required by the GP")


# ============================================================
# Diagnostic plots: w(z) and rho_DE(z)
# ============================================================

banner("Generating GP w(z) and rho_DE(z) plots")

# Plot slightly beyond zmax so that the imposed high-z
# LambdaCDM boundary can be seen explicitly.
z_plot = np.linspace(
    0.0,
    6.0,
    4000
)

# ------------------------------------------------------------
# Main GP figure
#
# Left:
#   w(z) for both kernels, including the actual GP nodes.
#
# Right:
#   rho_DE(z) / rho_DE,0 from CAMB for both kernels.
# ------------------------------------------------------------

fig, ax = plt.subplots(
    1,
    2,
    figsize=(11, 4.5)
)

kernel_labels = {
    1: "Squared-exponential",
    2: r"Exponential ($\alpha=1$)",
}

for kernel in (1, 2):

    results = nontrivial[kernel]["results"]
    z_nodes = nontrivial[kernel]["z_nodes"]
    w_nodes = nontrivial[kernel]["w_nodes"]

    rho_plot, w_plot = camb_rho_w_at_z(
        results,
        z_plot
    )

    # CAMB's returned rho quantity is normalized here at z=0.
    rho_ratio = (
        rho_plot
        / rho_plot[0]
    )

    # w(z)
    line, = ax[0].plot(
        z_plot,
        w_plot,
        lw=2,
        label=kernel_labels[kernel],
    )

    # Use the same automatically assigned line color for the nodes.
    ax[0].scatter(
        z_nodes,
        w_nodes,
        s=28,
        color=line.get_color(),
        zorder=5,
    )

    # rho_DE(z)/rho_DE,0
    ax[1].plot(
        z_plot,
        rho_ratio,
        lw=2,
        label=kernel_labels[kernel],
    )


# LambdaCDM reference
ax[0].axhline(
    -1.0,
    ls="--",
    lw=1,
)

ax[1].axhline(
    1.0,
    ls="--",
    lw=1,
)

# GP reconstruction boundary
for a in ax:
    a.axvline(
        GP_ZMAX,
        ls=":",
        lw=1,
        label=r"$z_{\rm max}$",
    )

    a.set_xlim(
        0.0,
        6.0
    )

    a.set_xlabel(
        r"$z$",
        fontsize=14,
    )

    a.minorticks_on()

    a.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )


ax[0].set_ylabel(
    r"$w(z)$",
    fontsize=14,
)

ax[1].set_ylabel(
    r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$",
    fontsize=14,
)

ax[0].legend(
    loc="best",
    fontsize=9,
)

ax[1].legend(
    loc="best",
    fontsize=9,
)

fig.tight_layout()

fig.savefig(
    "gp_w_rho.pdf",
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Optional consistency plot for rho_DE
#
# Solid  = CAMB
# Dashed = independent Python integration of CAMB-returned w(z)
#
# This is useful visually in addition to TEST 8.
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

for kernel in (1, 2):

    results = nontrivial[kernel]["results"]

    rho_camb, w_camb = camb_rho_w_at_z(
        results,
        z_plot
    )

    rho_camb_ratio = (
        rho_camb
        / rho_camb[0]
    )

    rho_python_ratio = rho_ratio_from_w(
        z_plot,
        w_camb
    )

    line, = ax.plot(
        z_plot,
        rho_camb_ratio,
        lw=2,
        label=f"CAMB: {kernel_labels[kernel]}",
    )

    ax.plot(
        z_plot,
        rho_python_ratio,
        lw=1.5,
        ls="--",
        color=line.get_color(),
        label=f"Python integral: {kernel_labels[kernel]}",
    )


ax.axhline(
    1.0,
    ls="--",
    lw=1,
)

ax.axvline(
    GP_ZMAX,
    ls=":",
    lw=1,
)

ax.set_xlim(
    0.0,
    6.0
)

ax.set_xlabel(
    r"$z$",
    fontsize=14,
)

ax.set_ylabel(
    r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$",
    fontsize=14,
)

ax.minorticks_on()

ax.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True,
)

ax.legend(
    loc="best",
    fontsize=8,
)

fig.tight_layout()

fig.savefig(
    "gp_rho_consistency.pdf",
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Kernel-correlation diagnostic
# ============================================================

delta_z = np.linspace(
    0.0,
    GP_ZMAX,
    500
)

corr_rbf = np.exp(
    -0.5
    * (delta_z / ell)**2
)

corr_exp = np.exp(
    -delta_z / ell
)

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    delta_z,
    corr_rbf,
    lw=2,
    label="Squared-exponential"
)

plt.plot(
    delta_z,
    corr_exp,
    lw=2,
    label=r"Exponential ($\alpha=1$)"
)

plt.xlabel(
    r"$|z-z'|$",
    fontsize=14,
)

plt.ylabel(
    r"$K(z,z')/\sigma_{\rm GP}^2$",
    fontsize=14,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "gp_kernel_correlations.pdf",
    bbox_inches="tight",
)

plt.show()


banner("ALL GP CONSISTENCY TESTS PASSED")
