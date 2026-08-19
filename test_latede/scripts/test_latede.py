import numpy as np
import camb
import os
import sys

print(
    "Using CAMB %s installed at %s"
    % (camb.__version__, os.path.dirname(camb.__file__)),
    flush=True,
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


# ============================================================
# TEST 1: constant-w baseline
# ============================================================

print("[1] Building constant-w model...", flush=True)

p0 = camb.set_params(
    **base_params,
    DEmodel=1,
    w0=-1.0,
)

print("[2] Constant-w set_params OK", flush=True)

r0 = camb.get_results(p0)

print("[3] Constant-w get_results OK", flush=True)


# ============================================================
# TEST 2: GP realization corresponding exactly to LambdaCDM
# ============================================================

gp_z = np.linspace(0.0, 3.5, 15)

gp_w = -np.ones_like(gp_z)

print("[4] Building GP LambdaCDM realization...", flush=True)

print(
    "    gp_z =",
    gp_z,
    flush=True,
)

print(
    "    gp_w =",
    gp_w,
    flush=True,
)


pgp = camb.set_params(
    **base_params,
    DEmodel=8,
    gp_z=gp_z,
    gp_w=gp_w,
)

print("[5] GP set_params OK", flush=True)


# ============================================================
# Check parameters stored inside CAMB
#
# IMPORTANT:
#
# DEmodel=8 accepts gp_z/gp_w at the Python interface,
# but internally stores them in the existing
#
#     spline_z
#     spline_w
#
# Fortran arrays.
# ============================================================

print(
    "    CAMB DEmodel =",
    pgp.DarkEnergy.DEmodel,
    flush=True,
)

stored_gp_z = np.array(
    pgp.DarkEnergy.spline_z
)

stored_gp_w = np.array(
    pgp.DarkEnergy.spline_w
)

print(
    "    CAMB GP z nodes =",
    stored_gp_z,
    flush=True,
)

print(
    "    CAMB GP w nodes =",
    stored_gp_w,
    flush=True,
)


# Check that Python -> Fortran transfer was exact

assert np.allclose(
    stored_gp_z,
    gp_z,
)

assert np.allclose(
    stored_gp_w,
    gp_w,
)

print(
    "    PASS: GP nodes transferred correctly to spline storage",
    flush=True,
)


# ============================================================
# Run CAMB
# ============================================================

rgp = camb.get_results(pgp)

print("[6] GP get_results OK", flush=True)


# ============================================================
# Evaluate dark-energy background
# ============================================================

a = np.logspace(
    -2,
    0,
    500,
)

rho, w = rgp.get_dark_energy_rho_w(a)

print(
    "[7] get_dark_energy_rho_w OK",
    flush=True,
)


# ============================================================
# Sanity checks
# ============================================================

assert np.all(
    np.isfinite(rho)
)

assert np.all(
    np.isfinite(w)
)

print(
    "    PASS: rho_DE and w(z) are finite",
    flush=True,
)


# ------------------------------------------------------------
# LambdaCDM GP realization:
#
# every GP node is exactly w=-1.
#
# Therefore spline interpolation must return exactly -1.
# ------------------------------------------------------------

max_w_error = np.max(
    np.abs(w + 1.0)
)

print(
    "    max |w+1| =",
    max_w_error,
    flush=True,
)

assert max_w_error < 1e-10

print(
    "    PASS: GP LambdaCDM reproduces w=-1",
    flush=True,
)


# ------------------------------------------------------------
# rho_DE must be constant for w=-1
# ------------------------------------------------------------

relative_rho_variation = (
    rho.max() - rho.min()
) / rho.mean()

print(
    "    relative rho variation =",
    relative_rho_variation,
    flush=True,
)

assert relative_rho_variation < 1e-10

print(
    "    PASS: GP LambdaCDM gives constant rho_DE",
    flush=True,
)


# ============================================================
# Check GP nodes directly
# ============================================================

a_nodes = 1.0 / (
    1.0 + gp_z
)

rho_nodes, w_nodes = (
    rgp.get_dark_energy_rho_w(
        a_nodes
    )
)

node_error = np.max(
    np.abs(
        w_nodes - gp_w
    )
)

print(
    "    max GP-node interpolation error =",
    node_error,
    flush=True,
)

assert node_error < 1e-10

print(
    "    PASS: CAMB spline passes through every GP node",
    flush=True,
)


print(
    "\nALL BASIC GP TESTS PASSED",
    flush=True,
)