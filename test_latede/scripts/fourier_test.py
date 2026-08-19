# CAMB v2 LateDE -- Fourier consistency tests
#
# DEmodel = 10
#
# Fourier model based on Tamayo & Vazquez 2019, arXiv:1901.08679,
# with a re-centered constant coefficient:
#
#   w_F(a) = -1 + c0
#            + sum_n [
#                A_n sin(n theta (a-a_med))
#              + B_n cos(n theta (a-a_med))
#              ]
#
#   theta = 2*pi/(1-a_med)
#
# For z_med < z <= z_ini, w is linearly joined to -1 in scale
# factor a.  For z > z_ini, w=-1 exactly.
#
# Our convention gives exact LambdaCDM when
#
#   c0 = A_n = B_n = 0.
#
# Relation to the paper's constant coefficient:
#
#   c0 = 1 + w0_paper/2.


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


ZMED = 2.8
ZINI = 3.0


def banner(name):
    print()
    print("=" * 78)
    print(name)
    print("=" * 78)


def exact_fourier_w(
        z,
        c0=0.0,
        an=(0.0, 0.0),
        bn=(0.0, 0.0),
        zmed=ZMED,
        zini=ZINI):
    """Independent implementation of the intended DEmodel=10 w(z)."""

    z = np.asarray(z, dtype=float)
    an = np.asarray(an, dtype=float)
    bn = np.asarray(bn, dtype=float)

    if len(an) != len(bn):
        raise ValueError("an and bn must have equal lengths")

    a = 1.0 / (1.0 + z)
    amed = 1.0 / (1.0 + zmed)
    aini = 1.0 / (1.0 + zini)

    theta = 2.0 * np.pi / (1.0 - amed)

    w = np.empty_like(z)

    mf = z <= zmed
    ml = (z > zmed) & (z <= zini)
    mh = z > zini

    w[mf] = -1.0 + c0

    for i in range(len(an)):
        n = i + 1
        phase = n * theta * (a[mf] - amed)
        w[mf] += an[i] * np.sin(phase) + bn[i] * np.cos(phase)

    wmed = -1.0 + c0 + np.sum(bn)
    slope = (wmed + 1.0) / (amed - aini)

    w[ml] = -1.0 + slope * (a[ml] - aini)
    w[mh] = -1.0

    return w


def make_params(
        n=2,
        c0=0.0,
        an=(0.0, 0.0, 0.0, 0.0),
        bn=(0.0, 0.0, 0.0, 0.0),
        zmed=ZMED,
        zini=ZINI,
        nint=4097):

    an = list(an) + [0.0] * (4-len(an))
    bn = list(bn) + [0.0] * (4-len(bn))

    return camb.set_params(
        **base_params,
        DEmodel=10,
        fourier_n=n,
        fourier_c0=c0,
        fourier_a1=an[0],
        fourier_a2=an[1],
        fourier_a3=an[2],
        fourier_a4=an[3],
        fourier_b1=bn[0],
        fourier_b2=bn[1],
        fourier_b3=bn[2],
        fourier_b4=bn[3],
        fourier_zmed=zmed,
        fourier_zini=zini,
        fourier_nint=nint,
    )


def camb_rho_w_at_z(results, z_values):
    z_values = np.atleast_1d(z_values).astype(float)
    a_values = 1.0 / (1.0 + z_values)

    order = np.argsort(a_values)

    rho_sorted, w_sorted = results.get_dark_energy_rho_w(
        a_values[order]
    )

    rho_sorted = np.asarray(rho_sorted)
    w_sorted = np.asarray(w_sorted)

    rho = np.empty_like(rho_sorted)
    w = np.empty_like(w_sorted)

    rho[order] = rho_sorted
    w[order] = w_sorted

    return rho, w


def rho_ratio_from_w(z, w):
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)

    integrand = (1.0 + w) / (1.0 + z)

    cumulative = np.zeros_like(z)
    dz = np.diff(z)

    cumulative[1:] = np.cumsum(
        0.5 * (integrand[:-1] + integrand[1:]) * dz
    )

    return np.exp(3.0 * cumulative)


# Nontrivial deterministic model.
C0 = -0.08
AN = np.array([-0.18, -0.12, 0.0, 0.0])
BN = np.array([-0.30,  0.05, 0.0, 0.0])


# ======================================================================
# TEST 1: exact LambdaCDM limit
# ======================================================================

banner("TEST 1: all-zero Fourier coefficients give exact LambdaCDM")

pars_lcdm = make_params(
    n=2,
    c0=0.0,
    an=(0.0, 0.0),
    bn=(0.0, 0.0),
)

results_lcdm = camb.get_results(pars_lcdm)

z = np.linspace(0.0, 10.0, 5001)
rho_lcdm, w_lcdm = camb_rho_w_at_z(results_lcdm, z)

max_w_error = np.max(np.abs(w_lcdm + 1.0))
rho_relative_variation = (rho_lcdm.max()-rho_lcdm.min()) / rho_lcdm.mean()

print(f"max |w+1| = {max_w_error:.3e}")
print(f"relative rho variation = {rho_relative_variation:.3e}")

assert max_w_error < 1e-12
assert rho_relative_variation < 1e-12

print("PASS: Fourier origin is exact LambdaCDM")


# ======================================================================
# TEST 2: stored coefficients and integral boundary
# ======================================================================

banner("TEST 2: Python interface stores Fourier inputs correctly")

pars = make_params(
    n=2,
    c0=C0,
    an=AN[:2],
    bn=BN[:2],
)

stored_a = np.asarray(pars.DarkEnergy.fourier_an)
stored_b = np.asarray(pars.DarkEnergy.fourier_bn)
stored_zint = np.asarray(pars.DarkEnergy.fourier_zint)
stored_Iint = np.asarray(pars.DarkEnergy.fourier_Iint)

print("stored A_n =", stored_a)
print("stored B_n =", stored_b)
print("integral grid points =", len(stored_zint))

assert np.allclose(stored_a, AN[:2], atol=0, rtol=0)
assert np.allclose(stored_b, BN[:2], atol=0, rtol=0)
assert stored_zint[0] == 0.0
assert abs(stored_zint[-1]-ZINI) < 1e-14
assert abs(stored_Iint[0]) < 1e-15

print("PASS: Python-to-Fortran Fourier state is consistent")


# ======================================================================
# TEST 3: exact Fortran w(z) versus independent formula
# ======================================================================

banner("TEST 3: CAMB w(z) matches independent Fourier formula")

results = camb.get_results(pars)

z_check = np.linspace(0.0, 6.0, 10001)

_, w_camb = camb_rho_w_at_z(results, z_check)

w_exact = exact_fourier_w(
    z_check,
    c0=C0,
    an=AN[:2],
    bn=BN[:2],
)

w_error = np.max(np.abs(w_camb-w_exact))

print(f"max |w_CAMB-w_exact| = {w_error:.3e}")

assert w_error < 2e-11

print("PASS: Fortran evaluates the intended Fourier expression")


# ======================================================================
# TEST 4: continuity at z_med and z_ini
# ======================================================================

banner("TEST 4: continuity at Fourier/linear and linear/Lambda boundaries")

for zb, label in ((ZMED, "z_med"), (ZINI, "z_ini")):
    eps = 1.0e-9
    zz = np.array([zb-eps, zb, zb+eps])

    _, ww = camb_rho_w_at_z(results, zz)

    left_jump = abs(ww[1]-ww[0])
    right_jump = abs(ww[2]-ww[1])

    print(
        f"{label}: w(left)={ww[0]:+.12e}, "
        f"w(boundary)={ww[1]:+.12e}, "
        f"w(right)={ww[2]:+.12e}"
    )

    # Differences here are derivative*eps, not literal discontinuities.
    assert left_jump < 1e-7
    assert right_jump < 1e-7

print("PASS: w(z) is continuous at both transition points")


# ======================================================================
# TEST 5: exact high-z Lambda boundary
# ======================================================================

banner("TEST 5: exact high-z w=-1 and constant rho_DE")

z_high = np.array([
    ZINI,
    ZINI + 1e-8,
    4.0,
    5.0,
    10.0,
    30.0,
    100.0,
])

rho_high, w_high = camb_rho_w_at_z(results, z_high)

max_high_w_error = np.max(np.abs(w_high + 1.0))
rho_high_variation = (rho_high.max()-rho_high.min()) / np.mean(rho_high)

print(f"max high-z |w+1| = {max_high_w_error:.3e}")
print(f"relative high-z rho variation = {rho_high_variation:.3e}")

assert max_high_w_error < 1e-12
assert rho_high_variation < 1e-12

print("PASS: model is exactly Lambda-like above z_ini")


# ======================================================================
# TEST 6: independent continuity-equation rho check
# ======================================================================

banner("TEST 6: CAMB rho_DE agrees with independent integration of CAMB w(z)")

z_rho = np.linspace(0.0, 8.0, 60001)

rho_camb, w_camb_rho = camb_rho_w_at_z(results, z_rho)

rho_python = rho_ratio_from_w(
    z_rho,
    w_camb_rho,
)

rho_camb_ratio = rho_camb / rho_camb[0]

relative_error = rho_camb_ratio/rho_python - 1.0

max_rel = np.max(np.abs(relative_error))

print(f"max relative rho error = {max_rel:.3e}")

# The CAMB side interpolates a 4097-point precomputed I(z), whereas this
# independent reconstruction uses a much finer 60001-point integration.
assert max_rel < 2e-6

print("PASS: Fourier rho_DE satisfies the continuity equation")


# ======================================================================
# TEST 7: one-harmonic identities
# ======================================================================

banner("TEST 7: individual sine and cosine harmonics")

# A1 only
pars_a1 = make_params(
    n=1,
    c0=0.0,
    an=(0.2,),
    bn=(0.0,),
)
res_a1 = camb.get_results(pars_a1)

# B1 only
pars_b1 = make_params(
    n=1,
    c0=0.0,
    an=(0.0,),
    bn=(0.2,),
)
res_b1 = camb.get_results(pars_b1)

z_low = np.linspace(0.0, ZMED, 3001)

_, wa1 = camb_rho_w_at_z(res_a1, z_low)
_, wb1 = camb_rho_w_at_z(res_b1, z_low)

wa1_exact = exact_fourier_w(
    z_low, c0=0.0, an=(0.2,), bn=(0.0,)
)
wb1_exact = exact_fourier_w(
    z_low, c0=0.0, an=(0.0,), bn=(0.2,)
)

err_a1 = np.max(np.abs(wa1-wa1_exact))
err_b1 = np.max(np.abs(wb1-wb1_exact))

print(f"A1-only max error = {err_a1:.3e}")
print(f"B1-only max error = {err_b1:.3e}")

assert err_a1 < 2e-11
assert err_b1 < 2e-11

print("PASS: individual Fourier harmonics are evaluated correctly")


# ======================================================================
# TEST 8: linearity around LambdaCDM in Fourier region
# ======================================================================

banner("TEST 8: coefficient sign symmetry around w=-1 in Fourier region")

pars_plus = make_params(
    n=2, c0=C0, an=AN[:2], bn=BN[:2]
)
pars_minus = make_params(
    n=2, c0=-C0, an=-AN[:2], bn=-BN[:2]
)

res_plus = camb.get_results(pars_plus)
res_minus = camb.get_results(pars_minus)

z_sym = np.linspace(0.0, ZMED, 3001)

_, wplus = camb_rho_w_at_z(res_plus, z_sym)
_, wminus = camb_rho_w_at_z(res_minus, z_sym)

sym_error = np.max(np.abs(wplus+wminus+2.0))

print(f"max |w(+p)+w(-p)+2| = {sym_error:.3e}")

assert sym_error < 2e-11

print("PASS: Fourier reconstruction is linear around LambdaCDM")


# ======================================================================
# TEST 9: relation to the paper's constant normalization
# ======================================================================

banner("TEST 9: paper normalization maps correctly to our c0 convention")

# Paper: w_F contains w0_paper/2.
# Our model: -1 + c0.
# Therefore c0 = 1 + w0_paper/2.

w0_paper = -2.50
c0_here = 1.0 + 0.5*w0_paper

assert abs(c0_here + 0.25) < 1e-15

z_demo = np.array([0.0, 0.5, 1.0, ZMED])

w_ours = exact_fourier_w(
    z_demo,
    c0=c0_here,
    an=(-0.4, -0.7),
    bn=(-0.65, 0.1),
)

# Direct paper-form Fourier expression in its Fourier region.
a = 1.0/(1.0+z_demo)
amed = 1.0/(1.0+ZMED)
theta = 2.0*np.pi/(1.0-amed)

w_paper = np.full_like(z_demo, w0_paper/2.0)
for n, (aa, bb) in enumerate(
        zip((-0.4, -0.7), (-0.65, 0.1)), start=1):
    phase = n*theta*(a-amed)
    w_paper += aa*np.sin(phase) + bb*np.cos(phase)

mapping_error = np.max(np.abs(w_ours-w_paper))

print(f"max mapped normalization error = {mapping_error:.3e}")

assert mapping_error < 1e-14

print("PASS: our re-centering is algebraically identical to the paper")


# ======================================================================
# Diagnostic plots
# ======================================================================

banner("Generating Fourier diagnostic plots")

z_plot = np.linspace(0.0, 6.0, 5000)

rho_plot, w_plot = camb_rho_w_at_z(results, z_plot)

rho_plot_ratio = rho_plot/rho_plot[0]

w_exact_plot = exact_fourier_w(
    z_plot,
    c0=C0,
    an=AN[:2],
    bn=BN[:2],
)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

ax[0].plot(z_plot, w_plot, lw=2, label="CAMB Fourier")
ax[0].plot(z_plot, w_exact_plot, ls="--", lw=1.2, label="Independent formula")
ax[0].axhline(-1.0, ls=":", lw=1)
ax[0].axvline(ZMED, ls=":", lw=1)
ax[0].axvline(ZINI, ls=":", lw=1)
ax[0].set_xlabel(r"$z$")
ax[0].set_ylabel(r"$w(z)$")
ax[0].legend()

rho_python_plot = rho_ratio_from_w(z_plot, w_plot)

ax[1].plot(z_plot, rho_plot_ratio, lw=2, label="CAMB")
ax[1].plot(z_plot, rho_python_plot, ls="--", lw=1.2, label="Python integral")
ax[1].axhline(1.0, ls=":", lw=1)
ax[1].axvline(ZMED, ls=":", lw=1)
ax[1].axvline(ZINI, ls=":", lw=1)
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$")
ax[1].legend()

for aplot in ax:
    aplot.minorticks_on()
    aplot.tick_params(which="both", direction="in", top=True, right=True)

fig.tight_layout()
fig.savefig("fourier_w_rho.pdf", bbox_inches="tight")
plt.show()


banner("ALL FOURIER CONSISTENCY TESTS PASSED")
