# CAMB v2 LateDE

import numpy as np
import matplotlib.pyplot as plt
import camb
import os

# Check which CAMB is installed
print(
    "Using CAMB %s installed at %s"
    % (camb.__version__, os.path.dirname(camb.__file__))
)


# Common cosmological parameters
base_params = dict(
    H0=70,ombh2=0.02238280,omch2=0.1201075,TCMB=2.7255,dark_energy_model="ppf",
    # Neutrinos
    omnuh2=0,num_nu_massless=3.044,num_nu_massive=0,nu_mass_degeneracies=[0],nu_mass_numbers=[0],
    # Initial power spectrum
    As=2.100549e-9,ns=0.9660499,YHe=0.246,WantTransfer=True
    )

# ------------------------------------------------------------
# DEmodel = 1: constant w
# ------------------------------------------------------------
cosmology_w = camb.set_params(**base_params,DEmodel=1,w0=-0.8)

# ------------------------------------------------------------
# DEmodel = 2: CPL
# w(a) = w0 + (1-a) w1
# ------------------------------------------------------------
cosmology_cpl = camb.set_params(**base_params,DEmodel=2,w0=-0.8,w1=-0.3)

# ------------------------------------------------------------
# DEmodel = 3: piecewise-constant binned w(z)
# ------------------------------------------------------------
z_knot = [0.5 * (i + 1) for i in range(20)]
w_knot = [-1 + 0.3 * np.sin(i) for i in range(20)]

cosmology_bins = camb.set_params(**base_params,DEmodel=3,z_knot=z_knot,w_knot=w_knot)

# ------------------------------------------------------------
# DEmodel = 4: Flexknots
# ------------------------------------------------------------
a_flexknot=[0.0,0.35,0.60,0.80,1.0]
w_flexknot=[-1.0,-0.7,-1.4,-0.8,-1.1]
cosmology_flexknot = camb.set_params(**base_params,DEmodel=4,a_flexknot=a_flexknot,w_flexknot=w_flexknot)

# Background sampling
scale_factor = np.logspace(-2, 0, 2000)
redshift = 1 / scale_factor - 1

# ------------------------------------------------------------
# Compute dark-energy background evolution
# ------------------------------------------------------------
results_w = camb.get_results(cosmology_w)
rho_w, wde_w = results_w.get_dark_energy_rho_w(scale_factor)

results_cpl = camb.get_results(cosmology_cpl)
rho_cpl, wde_cpl = results_cpl.get_dark_energy_rho_w(scale_factor)

results_bins = camb.get_results(cosmology_bins)
rho_bins, wde_bins = results_bins.get_dark_energy_rho_w(scale_factor)

results_flexknot = camb.get_results(cosmology_flexknot)
rho_flexnot, wde_flexknot = results_flexknot.get_dark_energy_rho_w(scale_factor)


# ------------------------------------------------------------
# Plot w(z)
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))

plt.plot(redshift,wde_w,lw=2,label=r"DEmodel=1: constant $w$")
plt.plot(redshift,wde_cpl,lw=2,label=r"DEmodel=2: CPL")
plt.plot(redshift,wde_bins,lw=2,label=r"DEmodel=3: binned $w(z)$")
plt.plot(redshift,wde_flexknot,lw=2,label=r"DEmodel=4: Flexknots")

plt.axhline(-1, lw=1, ls="--")

plt.xlim(0, 15)
plt.ylabel(r"$w(z)$", fontsize=15)
plt.xlabel(r"$z$", fontsize=15)
plt.legend(loc="best")
plt.tight_layout()

# plt.savefig("latede_models_w.pdf")
plt.show()

# ------------------------------------------------------------
# Plot rho_DE(a) / rho_DE(a=1)
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))

plt.plot(redshift,rho_w / rho_w[-1],lw=2,label=r"DEmodel=1: constant $w$")
plt.plot(redshift,rho_cpl / rho_cpl[-1],lw=2,label=r"DEmodel=2: CPL")
plt.plot(redshift,rho_bins / rho_bins[-1],lw=2,label=r"DEmodel=3: binned $w(z)$")
plt.plot(redshift,rho_flexnot / rho_flexnot[-1],lw=2,label=r"DEmodel=4: Flexknots")

# plt.xlim(0, 1)
plt.ylabel(r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$", fontsize=15)
plt.xlabel(r"$z$", fontsize=15)
plt.legend(loc="best")
plt.tight_layout()
# plt.savefig("latede_models_rho.pdf")
plt.show()


# One knot vs constant w
cosmology_1flexknot = camb.set_params(**base_params,DEmodel=4,a_flexknot=[1.],w_flexknot=[-.8])

# Compute
results_1flexknot = camb.get_results(cosmology_1flexknot)
rho_1flexnot, wde_1flexknot = results_1flexknot.get_dark_energy_rho_w(scale_factor)

fig,ax = plt.subplots(2,2,figsize=(8,6))
ax[0,0].plot(redshift,wde_w,lw=2,label=r"DEmodel=1: constant w")
ax[0,0].plot(redshift,wde_1flexknot,lw=2,ls='--',label=r"DEmodel=4: one flexknots")
ax[0,1].plot(redshift,rho_w / rho_1flexnot[-1],lw=2)
ax[0,1].plot(redshift,rho_w / rho_1flexnot[-1],lw=2,ls='--')
ax[1,0].plot(redshift,wde_w/wde_1flexknot-1,lw=2)
ax[1,1].plot(redshift,rho_w/rho_1flexnot-1,lw=2)
[ax[1,i].set_xlabel("z") for i in [0,1]]
ax[0,0].set_ylabel("w_de")
ax[0,1].set_ylabel("rho_de")
ax[1,0].set_ylabel("frac in w")
ax[1,1].set_ylabel("frac in rho")
ax[0,0].legend(loc='upper center')


# Two knots vs CPL
w0=-0.8
wa=-0.3

a_flexknot=[0., 1.]
w_flexknot=[w0+wa, w0]

cosmology_2flexknot = camb.set_params(**base_params,DEmodel=4,a_flexknot=a_flexknot,w_flexknot=w_flexknot)

# Compute
results_2flexknot = camb.get_results(cosmology_2flexknot)
rho_2flexnot, wde_2flexknot = results_2flexknot.get_dark_energy_rho_w(scale_factor)

fig,ax = plt.subplots(2,2,figsize=(8,6))
ax[0,0].plot(redshift,wde_cpl,lw=2,label=r"DEmodel=2: CPL")
ax[0,0].plot(redshift,wde_2flexknot,lw=2,ls='--',label=r"DEmodel=4: two flexknots")
ax[0,1].plot(redshift,rho_cpl / rho_cpl[-1],lw=2)
ax[0,1].plot(redshift,rho_2flexnot / rho_2flexnot[-1],lw=2,ls='--')
ax[1,0].plot(redshift,wde_cpl/wde_2flexknot-1,lw=2)
ax[1,1].plot(redshift,rho_cpl/rho_2flexnot-1,lw=2)
[ax[1,i].set_xlabel("z") for i in [0,1]]
ax[0,0].set_ylabel("w_de")
ax[0,1].set_ylabel("rho_de")
ax[1,0].set_ylabel("frac in w")
ax[1,1].set_ylabel("frac in rho")
ax[0,0].legend(loc='upper center')


# Pure ΛCDM flexknot
a_flexknot=[0., 0.2, 0.6, 0.9, 1.]
w_flexknot=[-1., -1., -1., -1., -1.]

cosmology_lflexknotcdm = camb.set_params(**base_params,DEmodel=4,a_flexknot=a_flexknot,w_flexknot=w_flexknot)

# Compute
results_lflexknotcdm = camb.get_results(cosmology_lflexknotcdm)
rho_lflexknotcdm, wde_lflexknotcdm = results_lflexknotcdm.get_dark_energy_rho_w(scale_factor)

fig,ax = plt.subplots(1,2,figsize=(8,4))
ax[0].plot(redshift,wde_lflexknotcdm,lw=2,ls='--',label=r"DEmodel=4: L flexknot CDM")
ax[1].plot(redshift,rho_lflexknotcdm / rho_lflexknotcdm[-1],lw=2,ls='--')
[ax[i].set_xlabel("z") for i in [0,1]]
ax[0].set_ylabel("w_de")
ax[1].set_ylabel("rho_de")
ax[0].legend(loc='upper center')


# ------------------------------------------------------------
# Approximate knot coordinates read from Fig. 1 of
# Ormondroyd et al. 2025, arXiv:2503.08658
# ------------------------------------------------------------

examples = [
    (
        [0.0, 0.20, 0.70, 1.0],
        [-1.55, -0.50, -1.00, -1.45],
    ),
    (
        [0.0, 0.50, 0.80, 0.90, 1.0],
        [-0.85, -1.20, -0.80, -1.00, -1.75],
    ),
    (
        [0.0, 0.30, 0.60, 0.90, 1.0],
        [-1.85, -1.20, -2.60, -1.30, -2.05],
    ),
]

agrid = np.linspace(1e-4, 1.0, 1000)
fig, ax = plt.subplots(figsize=(6, 6))

for a_knot, w_knot in examples:
    pars = camb.set_params(**base_params,DEmodel=4,a_flexknot=a_knot,w_flexknot=w_knot)
    results = camb.get_results(pars)
    rho, wde = results.get_dark_energy_rho_w(agrid)
    ax.plot(agrid,wde,'--',lw=1.5)
    ax.plot(a_knot,w_knot,'+',ms=10,mew=1.5)

ax.set_xlim(0, 1)
ax.set_ylim(-3, 0)

ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$w$')

ax.minorticks_on()
ax.tick_params(which='both',direction='in',top=True,right=True)

plt.tight_layout()
plt.show()


# CAMB v2 LateDE

import numpy as np
import matplotlib.pyplot as plt
import camb
import os


# ============================================================
# Check which CAMB is installed
# ============================================================

print(
    "Using CAMB %s installed at %s"
    % (camb.__version__, os.path.dirname(camb.__file__))
)


# ============================================================
# Common cosmological parameters
# ============================================================

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

    # Initial power spectrum
    As=2.100549e-9,
    ns=0.9660499,
    YHe=0.246,

    WantTransfer=True
)


# ============================================================
# Background sampling
# ============================================================

redshift = np.linspace(
    0.0,
    10.0,
    3000
)

scale_factor = 1.0 / (
    1.0 + redshift
)


# ============================================================
# Fixed cubic-spline redshift nodes
# ============================================================

spline_z = np.array([
    0.0,
    0.2,
    0.57,
    0.8,
    1.3,
    2.33
])

input_w = np.array([
    -1.00,
    -0.80,
    -1.15,
    -0.90,
    -1.10,
    -1.00
])

zmax = spline_z[-1]


# ============================================================
# MODEL 1
#
# Constant-w LambdaCDM control
# ============================================================

cosmology_const_lcdm = camb.set_params(
    **base_params,
    DEmodel=1,
    w0=-1.0
)


# ============================================================
# MODEL 2
#
# Cubic-spline LambdaCDM
# ============================================================

cosmology_spline_lcdm = camb.set_params(
    **base_params,
    DEmodel=5,
    spline_z=spline_z,
    spline_w=[
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0
    ]
)


# ============================================================
# MODEL 3
#
# Non-trivial cubic spline
# ============================================================

cosmology_spline = camb.set_params(
    **base_params,
    DEmodel=5,
    spline_z=spline_z,
    spline_w=input_w
)


# ============================================================
# Compute CAMB results
# ============================================================

results_const_lcdm = camb.get_results(
    cosmology_const_lcdm
)

results_spline_lcdm = camb.get_results(
    cosmology_spline_lcdm
)

results_spline = camb.get_results(
    cosmology_spline
)


# ============================================================
# Helper:
# query CAMB rho and w at arbitrary redshifts
#
# We sort internally by increasing scale factor and then restore
# the original requested order.
# ============================================================

def camb_rho_w_at_z(results, z_values):

    z_values = np.atleast_1d(
        z_values
    ).astype(float)

    a_values = 1.0 / (
        1.0 + z_values
    )

    order = np.argsort(
        a_values
    )

    a_sorted = a_values[
        order
    ]

    rho_sorted, w_sorted = (
        results
        .get_dark_energy_rho_w(
            a_sorted
        )
    )

    rho_sorted = np.asarray(
        rho_sorted
    )

    w_sorted = np.asarray(
        w_sorted
    )

    rho_values = np.empty_like(
        rho_sorted
    )

    w_values = np.empty_like(
        w_sorted
    )

    rho_values[order] = rho_sorted
    w_values[order] = w_sorted

    return rho_values, w_values


# ============================================================
# Helper:
# independent Python reconstruction of rho/rho0
#
# rho/rho0 =
#
# exp[
#     3 integral_0^z
#     (1+w)/(1+z) dz
# ]
# ============================================================

def rho_ratio_from_w(z, w):

    z = np.asarray(
        z,
        dtype=float
    )

    w = np.asarray(
        w,
        dtype=float
    )

    integrand = (
        1.0 + w
    ) / (
        1.0 + z
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
            integrand[1:]
            + integrand[:-1]
        )
        * dz
    )

    return np.exp(
        3.0 * cumulative
    )


# ============================================================
# Evaluate all three models on the main redshift grid
# ============================================================

rho_const_camb, w_const_camb = camb_rho_w_at_z(
    results_const_lcdm,
    redshift
)

rho_spline_lcdm_camb, w_spline_lcdm_camb = camb_rho_w_at_z(
    results_spline_lcdm,
    redshift
)

rho_spline_camb, w_spline_camb = camb_rho_w_at_z(
    results_spline,
    redshift
)


# ============================================================
# TEST 1
#
# Cubic-spline LambdaCDM must reproduce constant-w LambdaCDM
#
# This is the primary control test.
# ============================================================

print()
print("==========================================")
print("TEST 1: spline LambdaCDM vs constant-w")
print("==========================================")

max_w_diff = np.max(
    np.abs(
        w_spline_lcdm_camb
        - w_const_camb
    )
)

max_rho_diff = np.max(
    np.abs(
        rho_spline_lcdm_camb
        - rho_const_camb
    )
)

print(
    "max |w_spline - w_const| =",
    max_w_diff
)

print(
    "max |rho_spline - rho_const| =",
    max_rho_diff
)

assert np.allclose(
    w_spline_lcdm_camb,
    w_const_camb,
    atol=1e-12,
    rtol=0
)

assert np.allclose(
    rho_spline_lcdm_camb,
    rho_const_camb,
    atol=1e-12,
    rtol=1e-12
)

print(
    "PASS: spline LambdaCDM reproduces "
    "constant-w LambdaCDM inside CAMB"
)


# ============================================================
# TEST 2
#
# Inspect what CAMB actually returns as rho
# ============================================================

print()
print("==========================================")
print("TEST 2: CAMB rho diagnostic")
print("==========================================")

z_diag = np.array([
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0
])

a_diag = 1.0 / (
    1.0 + z_diag
)

rho_const_diag, w_const_diag = camb_rho_w_at_z(
    results_const_lcdm,
    z_diag
)

rho_spline_diag, w_spline_diag = camb_rho_w_at_z(
    results_spline_lcdm,
    z_diag
)

print()
print("Constant-w DEmodel=1:")
print()

for zi, ai, rhoi, wi in zip(
        z_diag,
        a_diag,
        rho_const_diag,
        w_const_diag):

    print(
        f"z={zi:5.1f}   "
        f"a={ai:.8f}   "
        f"rho_CAMB={rhoi:.12e}   "
        f"w={wi:+.12f}   "
        f"a^2={ai**2:.12e}   "
        f"a^4={ai**4:.12e}"
    )

print()
print("Spline LambdaCDM DEmodel=5:")
print()

for zi, ai, rhoi, wi in zip(
        z_diag,
        a_diag,
        rho_spline_diag,
        w_spline_diag):

    print(
        f"z={zi:5.1f}   "
        f"a={ai:.8f}   "
        f"rho_CAMB={rhoi:.12e}   "
        f"w={wi:+.12f}"
    )


# ============================================================
# TEST 3
#
# Check exact spline interpolation at supplied nodes
# ============================================================

print()
print("==========================================")
print("TEST 3: spline interpolation at nodes")
print("==========================================")

rho_nodes, w_nodes = camb_rho_w_at_z(
    results_spline,
    spline_z
)

for zi, wi, wi_camb in zip(
        spline_z,
        input_w,
        w_nodes):

    print(
        f"z = {zi:6.3f}   "
        f"input w = {wi: .8f}   "
        f"CAMB w = {wi_camb: .8f}   "
        f"diff = {wi_camb-wi:+.3e}"
    )

    assert np.isclose(
        wi_camb,
        wi,
        atol=1e-10,
        rtol=0
    )

print(
    "PASS: spline passes through all nodes"
)


# ============================================================
# Plot w(z)
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    redshift,
    w_spline_camb,
    label="Fixed-knot cubic spline"
)

plt.plot(
    redshift,
    w_spline_lcdm_camb,
    "--",
    label=r"$\Lambda$CDM spline"
)

plt.scatter(
    spline_z,
    input_w,
    marker="o",
    label="Spline nodes"
)

plt.axhline(
    -1.0,
    linestyle=":",
    linewidth=1
)

plt.axvline(
    zmax,
    linestyle=":",
    linewidth=1,
    label=r"$z_{\rm max}$"
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
    "cubic_spline_1.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# TEST 4
#
# Independent Python rho/rho0 from CAMB-returned w(z)
# ============================================================

rho_spline_python = rho_ratio_from_w(
    redshift,
    w_spline_camb
)

rho_lcdm_python = rho_ratio_from_w(
    redshift,
    w_spline_lcdm_camb
)


print()
print("==========================================")
print("TEST 4: Python LambdaCDM continuity")
print("==========================================")

print(
    "max |rho_python/rho0 - 1| =",
    np.max(
        np.abs(
            rho_lcdm_python - 1.0
        )
    )
)

assert np.allclose(
    rho_lcdm_python,
    1.0,
    atol=1e-10,
    rtol=0
)

print(
    "PASS: independent Python integration gives "
    "constant rho_DE for LambdaCDM"
)


# ============================================================
# Plot Python reconstructed rho/rho0
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    redshift,
    rho_spline_python,
    label="Fixed-knot cubic spline"
)

plt.plot(
    redshift,
    rho_lcdm_python,
    "--",
    label=r"$\Lambda$CDM"
)

plt.axhline(
    1.0,
    linestyle=":",
    linewidth=1
)

plt.axvline(
    zmax,
    linestyle=":",
    linewidth=1,
    label=r"$z_{\rm max}$"
)

plt.xlabel(
    r"$z$"
)

plt.ylabel(
    r"$\rho_{\rm DE}(z)/\rho_{\rm DE}(0)$"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "cubic_spline_2.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# TEST 5
#
# Diagnose whether CAMB rho can be converted to a simple ratio
#
# First normalize CAMB's returned quantity at z=0.
# This is diagnostic only.
# ============================================================

print()
print("==========================================")
print("TEST 5: CAMB normalized rho diagnostic")
print("==========================================")

rho_spline_camb_normalized = (
    rho_spline_camb
    / rho_spline_camb[0]
)

difference = (
    rho_spline_camb_normalized
    - rho_spline_python
)

relative_difference = (
    difference
    / rho_spline_python
)

max_abs_index = np.argmax(
    np.abs(
        difference
    )
)

max_rel_index = np.argmax(
    np.abs(
        relative_difference
    )
)

print(
    "rho_CAMB raw at z=0 =",
    rho_spline_camb[0]
)

print(
    "rho_CAMB normalized at z=0 =",
    rho_spline_camb_normalized[0]
)

print(
    "rho_Python at z=0 =",
    rho_spline_python[0]
)

print()

print(
    "max absolute difference =",
    np.max(
        np.abs(
            difference
        )
    )
)

print(
    "at z =",
    redshift[
        max_abs_index
    ]
)

print(
    "CAMB normalized =",
    rho_spline_camb_normalized[
        max_abs_index
    ]
)

print(
    "Python          =",
    rho_spline_python[
        max_abs_index
    ]
)

print()

print(
    "max relative difference =",
    np.max(
        np.abs(
            relative_difference
        )
    )
)

print(
    "at z =",
    redshift[
        max_rel_index
    ]
)


# ============================================================
# TEST 6
#
# Numerical derivative dw/dz at zmax
#
# Expected:
#
# dw/dz -> 0 from below
#
# dw/dz = 0 above zmax
# ============================================================

print()
print("==========================================")
print("TEST 6: derivative at zmax")
print("==========================================")

for h in [
        1e-3,
        1e-4,
        1e-5,
        1e-6]:

    z_eval = np.array([
        zmax - h,
        zmax,
        zmax + h
    ])

    _, w_eval = camb_rho_w_at_z(
        results_spline,
        z_eval
    )

    w_left = w_eval[0]
    w_center = w_eval[1]
    w_right = w_eval[2]

    dw_left = (
        w_center
        - w_left
    ) / h

    dw_right = (
        w_right
        - w_center
    ) / h

    print(
        f"h={h:.0e}   "
        f"left={dw_left:+.10e}   "
        f"right={dw_right:+.10e}   "
        f"diff={dw_right-dw_left:+.3e}"
    )


# ============================================================
# TEST 7
#
# Exact high-z LambdaCDM behavior
#
# For z > zmax:
#
# w(z) = -1 exactly
# ============================================================

print()
print("==========================================")
print("TEST 7: exact high-z LambdaCDM behavior")
print("==========================================")

z_high = np.array([
    zmax + 1e-6,
    3.0,
    5.0,
    10.0,
    30.0,
    100.0
])

rho_high, w_high = camb_rho_w_at_z(
    results_spline,
    z_high
)

for zi, wi in zip(
        z_high,
        w_high):

    print(
        f"z={zi:8.4f}   "
        f"w={wi:+.12f}   "
        f"|w+1|={abs(wi+1.0):.3e}"
    )

assert np.allclose(
    w_high,
    -1.0,
    atol=1e-12,
    rtol=0
)

print(
    "PASS: w(z) = -1 exactly above zmax"
)


# ============================================================
# TEST 8
#
# Compare high-z CAMB rho behavior with constant-w control
#
# Do NOT assume yet that CAMB's first returned quantity is
# directly rho/rho0.
# ============================================================

print()
print("==========================================")
print("TEST 8: high-z CAMB rho diagnostic")
print("==========================================")

rho_high_spline, _ = camb_rho_w_at_z(
    results_spline,
    z_high
)

rho_high_const, _ = camb_rho_w_at_z(
    results_const_lcdm,
    z_high
)

for zi, rho_s, rho_c in zip(
        z_high,
        rho_high_spline,
        rho_high_const):

    print(
        f"z={zi:8.4f}   "
        f"rho_spline={rho_s:.12e}   "
        f"rho_const={rho_c:.12e}   "
        f"ratio={rho_s/rho_c:.12e}"
    )


cosmology_cheb_lcdm = camb.set_params(
    **base_params,
    DEmodel=6,
    cheb_c0=1.0,
    cheb_c1=0.0,
    cheb_c2=0.0,
    cheb_c3=0.0,
    cheb_zmax=3.5,
    cheb_delta=1.0,
)

results_cheb_lcdm = camb.get_results(cosmology_cheb_lcdm)
rho_cheb_lcdm, wde_cheb_lcdm = results_cheb_lcdm.get_dark_energy_rho_w(scale_factor)

# ------------------------------------------------------------
# Plot w(z)
# ------------------------------------------------------------
fig,ax = plt.subplots(1,2,figsize=(8,4))

ax[0].plot(redshift,wde_cheb_lcdm,lw=2,label=r"DEmodel=5: Chebyshev")
ax[1].plot(redshift,rho_cheb_lcdm/rho_cheb_lcdm[-1],lw=2,label=r"DEmodel=5: Chebyshev")
[ax[i].legend(loc='upper center') for i in [0,1]]
[ax[i].set_xlabel('z') for i in [0,1]]
ax[0].set_ylabel('w')
ax[1].set_ylabel('rho / rho_today')


cosmology_cheb_w = camb.set_params(
    **base_params,
    DEmodel=6,
    cheb_c0=0.8,
    cheb_c1=0.0,
    cheb_c2=0.0,
    cheb_c3=0.0,
    cheb_zmax=3.5,
    cheb_delta=1.0,
)

results_cheb_w = camb.get_results(cosmology_cheb_w)
rho_cheb_w, wde_cheb_w = results_cheb_w.get_dark_energy_rho_w(scale_factor)

# ------------------------------------------------------------
# Plot w(z)
# ------------------------------------------------------------
fig,ax = plt.subplots(1,2,figsize=(8,4))

ax[0].plot(redshift,wde_w,lw=2,label=r"DEmodel=1: constant $w$")
ax[1].plot(redshift,rho_w/rho_w[-1],lw=2)
ax[0].plot(redshift,wde_cheb_w,lw=2,ls='--',label=r"DEmodel=5: Chebyshev")
ax[1].plot(redshift,rho_cheb_w/rho_cheb_w[-1],lw=2,ls='--')
ax[0].set_xlim(0,50)
ax[1].set_xlim(0,50)
ax[0].legend(loc='best')
[ax[i].set_xlabel('z') for i in [0,1]]
ax[0].set_ylabel('w')
ax[1].set_ylabel('rho / rho_today')



cosmology_cheb = camb.set_params(
    **base_params,
    DEmodel=6,
    cheb_c0=1.0,
    cheb_c1=0.2,
    cheb_c2=-0.1,
    cheb_c3=0.05,
    cheb_zmax=3.5,
    cheb_delta=1.0,
)

results_cheb = camb.get_results(cosmology_cheb)
rho_cheb, wde_cheb = results_cheb.get_dark_energy_rho_w(scale_factor)

# ------------------------------------------------------------
# Plot w(z)
# ------------------------------------------------------------
fig,ax = plt.subplots(1,2,figsize=(8,4))

ax[0].plot(redshift,wde_cheb,lw=2,ls='-',label=r"DEmodel=5: Chebyshev")
ax[1].plot(redshift,rho_cheb/rho_cheb[-1],lw=2,ls='-',label=r"DEmodel=5: Chebyshev")
ax[0].set_xlim(0,50)
ax[1].set_xlim(0,50)
ax[0].set_xlim(0,50)
ax[1].set_xlim(0,50)
[ax[i].legend(loc='best') for i in [0,1]]
[ax[i].set_xlabel('z') for i in [0,1]]
ax[0].set_ylabel('w')
ax[1].set_ylabel('rho / rho_today')


mask = (redshift > 3.0) & (redshift < 4.0)

plt.plot(
    redshift[mask],
    wde_cheb[mask]
)
plt.axvline(3.5, ls="--")



lnrho = np.log(rho_cheb)
ln1pz = np.log1p(redshift)

dlnrho_dln1pz = np.gradient(
    lnrho,
    ln1pz
)

rhs = 3.0 * (1.0 + wde_cheb)

mask = (redshift > 0.02) & (redshift < 90)

print(
    "max conservation error =",
    np.max(
        np.abs(
            dlnrho_dln1pz[mask] - rhs[mask]
        )
    )
)



cosmology_bern_lcdm = camb.set_params(
    **base_params,
    DEmodel=7,
    bern_b0=-1.0,
    bern_b1=-1.0,
    bern_b2=-1.0,
    bern_b3=-1.0,
    bern_zmax=3.5,
    bern_delta=1.0,
)

results_bern_lcdm = camb.get_results(cosmology_bern_lcdm)
rho_bern_lcdm, wde_bern_lcdm = results_bern_lcdm.get_dark_energy_rho_w(scale_factor)

# ------------------------------------------------------------
# Plot w(z)
# ------------------------------------------------------------
fig,ax = plt.subplots(1,2,figsize=(8,4))

ax[0].plot(redshift,wde_bern_lcdm,lw=2,label=r"DEmodel=6: Bernstein")
ax[1].plot(redshift,rho_bern_lcdm/rho_bern_lcdm[-1],lw=2,label=r"DEmodel=6: Bernstein")
[ax[i].legend(loc='upper center') for i in [0,1]]
[ax[i].set_xlabel('z') for i in [0,1]]
ax[0].set_ylabel('w')
ax[1].set_ylabel('rho / rho_today')




cosmology_bern_w = camb.set_params(
    **base_params,
    DEmodel=7,
    bern_b0=-0.8,
    bern_b1=-0.8,
    bern_b2=-0.8,
    bern_b3=-0.8,
    bern_zmax=3.5,
    bern_delta=1.0,
)

results_bern_w = camb.get_results(cosmology_bern_w)
rho_bern_w, wde_bern_w = results_bern_w.get_dark_energy_rho_w(scale_factor)

# ------------------------------------------------------------
# Plot w(z)
# ------------------------------------------------------------
fig,ax = plt.subplots(1,2,figsize=(8,4))

ax[0].plot(redshift,wde_w,lw=2,label=r"DEmodel=1: constant $w$")
ax[1].plot(redshift,rho_w/rho_w[-1],lw=2)
ax[0].plot(redshift,wde_bern_w,lw=2,ls='--',label=r"DEmodel=6: Bernstein")
ax[1].plot(redshift,rho_bern_w/rho_bern_w[-1],lw=2,ls='--')
ax[0].set_xlim(0,50)
ax[1].set_xlim(0,50)
ax[0].legend(loc='best')
[ax[i].set_xlabel('z') for i in [0,1]]
ax[0].set_ylabel('w')
ax[1].set_ylabel('rho / rho_today')



cosmology_bern = camb.set_params(
    **base_params,
    DEmodel=7,
    bern_b0=-0.9,
    bern_b1=-0.7,
    bern_b2=-1.2,
    bern_b3=-0.9,
    bern_zmax=3.5,
    bern_delta=1.0,
)
results_bern = camb.get_results(cosmology_bern)
rho_bern, wde_bern = results_bern.get_dark_energy_rho_w(scale_factor)

# ------------------------------------------------------------
# Plot w(z)
# ------------------------------------------------------------
fig,ax = plt.subplots(1,2,figsize=(8,4))

ax[0].plot(redshift,wde_bern,lw=2,ls='-',label=r"DEmodel=6: Bernstein")
ax[1].plot(redshift,rho_bern/rho_bern[-1],lw=2,ls='-',label=r"DEmodel=6: Bernstein")
ax[0].set_xlim(0,50)
ax[1].set_xlim(0,50)
ax[0].set_xlim(0,50)
ax[1].set_xlim(0,50)
[ax[i].legend(loc='best') for i in [0,1]]
[ax[i].set_xlabel('z') for i in [0,1]]
ax[0].set_ylabel('w')
ax[1].set_ylabel('rho / rho_today')