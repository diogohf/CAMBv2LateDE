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
# FLEXKNOTS: DEmodel = 4
# ============================================================

redshift = np.linspace(0.0, 10.0, 20001)

# References
results_lcdm = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=1,
        w0=-1.0,
    )
)

results_const = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=1,
        w0=-0.8,
    )
)

CPL_W0 = -0.8
CPL_WA = -0.3

results_cpl = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=2,
        w0=CPL_W0,
        w1=CPL_WA,
    )
)


# ------------------------------------------------------------
# TEST 1: LambdaCDM limit
# ------------------------------------------------------------

banner("TEST 1: all FlexKnot values at -1 give LambdaCDM")

a_knots = np.array([
    0.0,
    0.2,
    0.55,
    0.8,
    1.0,
])

w_knots = np.full(
    len(a_knots),
    -1.0,
)

results = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=4,
        a_flexknot=a_knots,
        w_flexknot=w_knots,
    )
)

compare_results(
    "FlexKnot LambdaCDM vs constant-w LambdaCDM",
    results,
    results_lcdm,
    redshift,
)

print("PASS: FlexKnots reproduce LambdaCDM")


# ------------------------------------------------------------
# TEST 2: one knot -> constant w
# ------------------------------------------------------------

banner("TEST 2: one knot reproduces constant-w")

results_one = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=4,
        a_flexknot=[1.0],
        w_flexknot=[-0.8],
    )
)

compare_results(
    "one FlexKnot vs constant w=-0.8",
    results_one,
    results_const,
    redshift,
)

print("PASS: one FlexKnot reproduces constant-w")


# ------------------------------------------------------------
# TEST 3: two endpoint knots -> CPL
# ------------------------------------------------------------

banner("TEST 3: two endpoint knots reproduce CPL")

# CPL:
#
#   w(a) = w0 + (1-a) wa
#
# so
#
#   w(a=0) = w0 + wa
#   w(a=1) = w0.

results_two = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=4,
        a_flexknot=[0.0, 1.0],
        w_flexknot=[
            CPL_W0 + CPL_WA,
            CPL_W0,
        ],
    )
)

compare_results(
    "two FlexKnots vs CPL",
    results_two,
    results_cpl,
    redshift,
)

print("PASS: two FlexKnots reproduce CPL")


# ------------------------------------------------------------
# TEST 4: arbitrary piecewise-linear realization
# ------------------------------------------------------------

banner("TEST 4: arbitrary realization is linear in scale factor")

a_knots = np.array([
    0.0,
    0.20,
    0.55,
    0.80,
    1.0,
])

w_knots = np.array([
    -1.20,
    -0.75,
    -1.35,
    -0.85,
    -1.05,
])

pars = camb.set_params(
    **base_params,
    DEmodel=4,
    a_flexknot=a_knots,
    w_flexknot=w_knots,
)

stored_a = np.asarray(
    pars.DarkEnergy.a_flexknot
)

stored_w = np.asarray(
    pars.DarkEnergy.w_flexknot
)

assert np.allclose(
    stored_a,
    a_knots,
    atol=0,
    rtol=0,
)

assert np.allclose(
    stored_w,
    w_knots,
    atol=0,
    rtol=0,
)

results = camb.get_results(pars)

a_probe = np.array([
    0.05,
    0.20,
    0.35,
    0.55,
    0.70,
    0.80,
    0.92,
    1.00,
])

z_probe = 1.0 / a_probe - 1.0

_, w_probe = camb_rho_w_at_z(
    results,
    z_probe,
)

expected = np.interp(
    a_probe,
    a_knots,
    w_knots,
)

max_error = np.max(
    np.abs(
        w_probe
        - expected
    )
)

print(
    "max |CAMB w(a) - linear interpolation| = "
    f"{max_error:.3e}"
)

assert max_error < 1.0e-12

print("PASS: FlexKnot interpolation is piecewise linear in a")


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
# Plot
# ------------------------------------------------------------

banner("GENERATING PLOTS")

z_plot = np.linspace(0.0, 10.0, 3001)
rho_plot, w_plot = camb_rho_w_at_z(results, z_plot)
rho_plot = rho_plot / rho_plot[0]
rho_py = rho_ratio_from_w(z_plot, w_plot)

plt.figure(figsize=(7, 5))
plt.plot(z_plot, w_plot, lw=2)
plt.axhline(-1.0, ls="--", lw=1)
plt.xlabel(r"$z$")
plt.ylabel(r"$w_{\rm DE}(z)$")
plt.tight_layout()
plt.savefig("flexknot_consistency_w.pdf", bbox_inches="tight")

plt.figure(figsize=(7, 5))
plt.plot(z_plot, rho_plot, lw=2, label="CAMB")
plt.plot(z_plot, rho_py, ls="--", lw=1.5, label="Python integral")
plt.xlabel(r"$z$")
plt.ylabel(r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$")
plt.legend()
plt.tight_layout()
plt.savefig("flexknot_consistency_rho.pdf", bbox_inches="tight")

plt.show()

banner("ALL FLEXKNOT TESTS PASSED")
