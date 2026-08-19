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
# BINNED w(z): DEmodel = 3
# ============================================================

ZMAX = 3.5
redshift = np.linspace(0.0, 10.0, 20001)

# Reference LambdaCDM
results_lcdm = camb.get_results(
    camb.set_params(
        **base_params,
        DEmodel=1,
        w0=-1.0,
    )
)

# ------------------------------------------------------------
# TEST 1: LambdaCDM limit
# ------------------------------------------------------------

banner("TEST 1: all bins at -1 give LambdaCDM")

bin_z = np.array([0.4, 1.0, 2.0, ZMAX])
bin_w_lcdm = np.full(len(bin_z), -1.0)

pars = camb.set_params(
    **base_params,
    DEmodel=3,
    z_knot=bin_z,
    w_knot=bin_w_lcdm,
)

results = camb.get_results(pars)

compare_results(
    "binned LambdaCDM vs constant-w LambdaCDM",
    results,
    results_lcdm,
    redshift,
)

print("PASS: binned w(z) reproduces LambdaCDM")


# ------------------------------------------------------------
# TEST 2: piecewise-constant values
# ------------------------------------------------------------

banner("TEST 2: correct value inside every bin")

bin_w = np.array([
    -0.75,
    -1.25,
    -0.90,
    -1.10,
])

pars = camb.set_params(
    **base_params,
    DEmodel=3,
    z_knot=bin_z,
    w_knot=bin_w,
)

stored_z = np.asarray(pars.DarkEnergy.z_knot)
stored_w = np.asarray(pars.DarkEnergy.w_knot)

print(
    "max |stored z_knot - input| =",
    np.max(np.abs(stored_z - bin_z)),
)
print(
    "max |stored w_knot - input| =",
    np.max(np.abs(stored_w - bin_w)),
)

assert np.allclose(stored_z, bin_z, atol=0, rtol=0)
assert np.allclose(stored_w, bin_w, atol=0, rtol=0)

results = camb.get_results(pars)

# Probe strictly inside bins, not on boundaries.
z_probe = np.array([
    0.20,
    0.70,
    1.50,
    2.70,
    4.00,
])

expected = np.array([
    bin_w[0],
    bin_w[1],
    bin_w[2],
    bin_w[3],
    -1.0,
])

_, w_probe = camb_rho_w_at_z(
    results,
    z_probe,
)

for zi, wi_expected, wi in zip(
        z_probe,
        expected,
        w_probe):

    print(
        f"z={zi:5.2f}   "
        f"expected={wi_expected:+.8f}   "
        f"CAMB={wi:+.8f}   "
        f"diff={wi-wi_expected:+.3e}"
    )

assert np.allclose(
    w_probe,
    expected,
    atol=1.0e-12,
    rtol=0,
)

print("PASS: each redshift interval uses the intended w_i")


# ------------------------------------------------------------
# TEST 3: independent rho_DE reconstruction
# ------------------------------------------------------------

banner("TEST 3: independent rho_DE reconstruction")

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

# Slightly looser because w(z) is genuinely discontinuous.
assert max_log_error < 5.0e-4

print("PASS: rho_DE obeys the continuity equation")


# ------------------------------------------------------------
# TEST 4: exact high-z boundary
# ------------------------------------------------------------

banner("TEST 4: exact high-z LambdaCDM boundary")

z_high = np.array([
    ZMAX + 1.0e-6,
    5.0,
    10.0,
    30.0,
    100.0,
])

_, w_high = camb_rho_w_at_z(
    results,
    z_high,
)

for zi, wi in zip(z_high, w_high):
    print(
        f"z={zi:9.4f}   "
        f"w={wi:+.12f}   "
        f"|w+1|={abs(wi+1.0):.3e}"
    )

assert np.allclose(
    w_high,
    -1.0,
    atol=0,
    rtol=0,
)

print("PASS: w(z)=-1 exactly above the last bin")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

banner("GENERATING PLOTS")

z_plot = np.linspace(0.0, 6.0, 3001)
rho_plot, w_plot = camb_rho_w_at_z(results, z_plot)
rho_plot = rho_plot / rho_plot[0]
rho_py = rho_ratio_from_w(z_plot, w_plot)

plt.figure(figsize=(7, 5))
plt.plot(z_plot, w_plot, lw=2)
plt.axhline(-1.0, ls="--", lw=1)
plt.xlabel(r"$z$")
plt.ylabel(r"$w_{\rm DE}(z)$")
plt.tight_layout()
plt.savefig("binw_consistency_w.pdf", bbox_inches="tight")

plt.figure(figsize=(7, 5))
plt.plot(z_plot, rho_plot, lw=2, label="CAMB")
plt.plot(z_plot, rho_py, ls="--", lw=1.5, label="Python integral")
plt.xlabel(r"$z$")
plt.ylabel(r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$")
plt.legend()
plt.tight_layout()
plt.savefig("binw_consistency_rho.pdf", bbox_inches="tight")

plt.show()

banner("ALL BINNED-w TESTS PASSED")
