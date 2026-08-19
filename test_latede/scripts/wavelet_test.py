# CAMB v2 LateDE -- Wavelet consistency tests
#
# DEmodel = 9 only.
#
# Tests both:
#   wave_type = 1 : Daubechies D4 (PyWavelets "db2")
#   wave_type = 2 : Haar
#
# The Python LateDE interface is assumed to use:
#
#   N = 16 wavelet coefficients
#   wavelet reconstruction of delta_w = 1 + w
#   coefficient hierarchy:
#
#       [cA4, cD4, cD3(2), cD2(4), cD1(8)]
#
#   giving 1 + 1 + 2 + 4 + 8 = 16 coefficients.
#
# The inverse DWT uses mode="periodization" on 16 interior
# samples z in [0, wave_zmax), followed by one explicit
# high-z boundary node:
#
#       (wave_zmax, w=-1).
#
# The resulting 17 samples are stored in the existing
# Fortran spline_z / spline_w arrays.
#
# Reference:
#   Hojjati, Pogosian & Zhao 2010, arXiv:0912.4843
#
# IMPORTANT:
#   In the older wavelet literature, "D4" usually means the
#   Daubechies wavelet with four filter coefficients.  In
#   PyWavelets this is named "db2", not "db4".


import numpy as np
import matplotlib.pyplot as plt
import camb
import os

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "test_wavelet_consistency.py requires PyWavelets. "
        "Install with `pip install PyWavelets` or "
        "`conda install pywavelets`."
    ) from exc


# ============================================================
# Basic setup
# ============================================================

print(
    "Using CAMB %s installed at %s"
    % (camb.__version__, os.path.dirname(camb.__file__))
)

print(
    "Using PyWavelets %s"
    % pywt.__version__
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

N_WAVE = 16
WAVE_ZMAX = 3.5

# Deterministic non-trivial wavelet coefficient vector.
#
# These are NOT values of w(z_i).
# They are coefficients in the multiresolution wavelet basis.
#
# Keep amplitudes moderate so the test probes a visibly dynamical
# w(z) without producing pathological dark-energy histories.
p_test = np.array([
     0.20,   # cA4
    -0.15,   # cD4
     0.12,   # cD3[0]
    -0.10,   # cD3[1]
     0.08,   # cD2[0]
    -0.06,   # cD2[1]
     0.05,   # cD2[2]
    -0.04,   # cD2[3]
     0.035,  # cD1[0]
    -0.030,  # cD1[1]
     0.025,  # cD1[2]
    -0.020,  # cD1[3]
     0.015,  # cD1[4]
    -0.012,  # cD1[5]
     0.010,  # cD1[6]
    -0.008,  # cD1[7]
], dtype=float)

assert len(p_test) == N_WAVE


# ============================================================
# Helpers
# ============================================================

def p_kwargs(p):
    """Convert a 16-vector into CAMB set_params wave_p* keywords."""

    p = np.asarray(
        p,
        dtype=float
    )

    if len(p) != N_WAVE:
        raise ValueError(
            f"Expected {N_WAVE} wavelet coefficients"
        )

    return {
        f"wave_p{i}": float(p[i])
        for i in range(N_WAVE)
    }


def wavelet_name(wave_type):
    """
    Match the convention implemented in dark_energy.py.

        wave_type = 1 -> Daubechies D4 = PyWavelets db2
        wave_type = 2 -> Haar
    """

    if wave_type == 1:
        return "db2"

    if wave_type == 2:
        return "haar"

    raise ValueError(
        "wave_type must be 1 (D4/db2) or 2 (Haar)"
    )


def reconstruct_wavelet(
        wave_type,
        p,
        zmax=WAVE_ZMAX):
    """
    Independently reproduce the Python-side DEmodel=9 construction.

    The 16 coefficients are interpreted as

        [cA4,
         cD4,
         cD3(2),
         cD2(4),
         cD1(8)]

    and PyWavelets reconstructs

        delta_w = 1 + w.

    The 16 reconstructed samples live in [0,zmax), and an
    explicit final node (zmax,-1) is appended.

    Returns
    -------
    z_full : length 17
    w_full : length 17
    delta_w_inner : length 16
    """

    p = np.asarray(
        p,
        dtype=float
    )

    if p.shape != (N_WAVE,):
        raise ValueError(
            "p must contain exactly 16 coefficients"
        )

    if zmax <= 0:
        raise ValueError(
            "zmax must be positive"
        )

    coeffs = [
        p[0:1],    # cA4
        p[1:2],    # cD4
        p[2:4],    # cD3
        p[4:8],    # cD2
        p[8:16],   # cD1
    ]

    delta_w_inner = pywt.waverec(
        coeffs,
        wavelet=wavelet_name(wave_type),
        mode="periodization",
    )

    delta_w_inner = np.asarray(
        delta_w_inner,
        dtype=float,
    )

    if delta_w_inner.size != N_WAVE:
        raise RuntimeError(
            "Independent wavelet reconstruction did not "
            "return 16 samples"
        )

    z_inner = np.linspace(
        0.0,
        zmax,
        N_WAVE,
        endpoint=False,
        dtype=float,
    )

    w_inner = -1.0 + delta_w_inner

    z_full = np.concatenate([
        z_inner,
        np.array([zmax], dtype=float),
    ])

    w_full = np.concatenate([
        w_inner,
        np.array([-1.0], dtype=float),
    ])

    return z_full, w_full, delta_w_inner


def build_basis_matrix(
        wave_type):
    """
    Construct the discrete 16x16 inverse-wavelet transform matrix Psi.

        delta_w = Psi @ P

    Each column is obtained by setting exactly one wavelet
    coefficient to unity and all others to zero.
    """

    Psi = np.zeros(
        (N_WAVE, N_WAVE),
        dtype=float,
    )

    for i in range(N_WAVE):

        p = np.zeros(
            N_WAVE,
            dtype=float,
        )

        p[i] = 1.0

        _, _, delta_w = reconstruct_wavelet(
            wave_type=wave_type,
            p=p,
        )

        Psi[:, i] = delta_w

    return Psi


def make_wavelet_params(
        wave_type,
        p,
        zmax=WAVE_ZMAX):

    return camb.set_params(
        **base_params,
        DEmodel=9,
        wave_type=wave_type,
        wave_zmax=zmax,
        **p_kwargs(p),
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


wave_labels = {
    1: "Daubechies D4 (db2)",
    2: "Haar",
}


# ============================================================
# Main redshift grid
# ============================================================

redshift = np.linspace(
    0.0,
    10.0,
    5001,
)


# ============================================================
# TEST 1
#
# Pure wavelet-transform sanity check.
#
# Build the inverse-transform matrix Psi independently and verify
# that PyWavelets reconstruction is linear:
#
#       IDWT(P) = Psi P.
#
# Also inspect the Gram matrix Psi^T Psi.  For the orthogonal
# D4/Haar transforms with periodization it should be the identity
# to numerical precision.
#
# This test does NOT call CAMB.
# ============================================================

banner("TEST 1: wavelet transform and basis-matrix sanity checks")

basis_matrices = {}

for wave_type in (1, 2):

    Psi = build_basis_matrix(
        wave_type
    )

    basis_matrices[wave_type] = Psi

    _, _, delta_direct = reconstruct_wavelet(
        wave_type=wave_type,
        p=p_test,
    )

    delta_matrix = (
        Psi @ p_test
    )

    linearity_error = np.max(
        np.abs(
            delta_direct
            - delta_matrix
        )
    )

    gram = (
        Psi.T @ Psi
    )

    orthogonality_error = np.max(
        np.abs(
            gram
            - np.eye(N_WAVE)
        )
    )

    print(
        f"wave_type={wave_type} "
        f"({wave_labels[wave_type]}): "
        f"max |IDWT(P)-Psi P|={linearity_error:.3e}, "
        f"max |Psi^T Psi-I|={orthogonality_error:.3e}"
    )

    assert linearity_error < 1e-13
    assert orthogonality_error < 1e-12

print("PASS: both wavelet transforms are linear and orthonormal")


# ============================================================
# TEST 2
#
# P_i = 0 must give exact LambdaCDM for either wavelet family.
#
# Since the parameterization is
#
#       1 + w = sum_i P_i psi_i,
#
# all coefficients equal to zero must give w=-1 exactly.
# ============================================================

banner("TEST 2: all-zero coefficients give LambdaCDM")

p_zero = np.zeros(
    N_WAVE
)

lcdm_results = {}

for wave_type in (1, 2):

    pars = make_wavelet_params(
        wave_type=wave_type,
        p=p_zero,
    )

    stored_z = np.asarray(
        pars.DarkEnergy.spline_z
    )

    stored_w = np.asarray(
        pars.DarkEnergy.spline_w
    )

    print(
        f"wave_type={wave_type}: "
        f"stored node count={len(stored_z)}, "
        f"max stored |w+1|={np.max(np.abs(stored_w + 1.0)):.3e}"
    )

    assert len(stored_z) == N_WAVE + 1

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
        f"wave_type={wave_type}: "
        f"max CAMB |w+1|={max_w_error:.3e}, "
        f"relative rho variation={rho_relative_variation:.3e}"
    )

    assert max_w_error < 1e-12
    assert rho_relative_variation < 1e-12

    lcdm_results[wave_type] = (
        rho,
        w,
    )

print("PASS: P=0 reproduces exact LambdaCDM for D4 and Haar")


# ============================================================
# TEST 3
#
# Wavelet-family independence in the LambdaCDM limit.
#
# D4 and Haar must become identical when P=0.
# ============================================================

banner("TEST 3: wavelet-family independence in LambdaCDM limit")

rho1, w1 = lcdm_results[1]
rho2, w2 = lcdm_results[2]

max_w_difference = np.max(
    np.abs(
        w1 - w2
    )
)

max_rho_difference = np.max(
    np.abs(
        rho1 - rho2
    )
)

print(
    "max |w_D4-w_Haar| =",
    max_w_difference
)

print(
    "max |rho_D4-rho_Haar| =",
    max_rho_difference
)

assert max_w_difference < 1e-12
assert max_rho_difference < 1e-12

print("PASS: D4 and Haar are identical when all coefficients vanish")


# ============================================================
# TEST 4
#
# Independently perform the PyWavelets inverse transform and
# compare the resulting 17 nodes against what dark_energy.py
# stored in CAMB's spline_z / spline_w arrays.
#
# This is the strongest Python-interface test.
# ============================================================

banner("TEST 4: independent IDWT vs CAMB stored wavelet nodes")

nontrivial = {}

for wave_type in (1, 2):

    expected_z, expected_w, _ = reconstruct_wavelet(
        wave_type=wave_type,
        p=p_test,
    )

    pars = make_wavelet_params(
        wave_type=wave_type,
        p=p_test,
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
        f"wave_type={wave_type}: "
        f"max node-z error={z_error:.3e}, "
        f"max node-w error={w_error:.3e}"
    )

    assert z_error < 1e-14
    assert w_error < 1e-13

    results = camb.get_results(
        pars
    )

    nontrivial[wave_type] = {
        "pars": pars,
        "results": results,
        "z_nodes": expected_z,
        "w_nodes": expected_w,
    }

print("PASS: dark_energy.py stores exactly the independently reconstructed wavelet nodes")


# ============================================================
# TEST 5
#
# CAMB w(z) must pass through every stored wavelet/spline node.
#
# This tests the Fortran cubic-spline backend separately from
# the Python PyWavelets reconstruction.
# ============================================================

banner("TEST 5: CAMB interpolation passes through all wavelet nodes")

for wave_type in (1, 2):

    results = nontrivial[wave_type]["results"]
    z_nodes = nontrivial[wave_type]["z_nodes"]
    w_nodes = nontrivial[wave_type]["w_nodes"]

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
        f"wave_type={wave_type}: "
        f"max interpolation error={node_error:.3e}"
    )

    assert node_error < 1e-10

print("PASS: CAMB spline passes through every wavelet node")


# ============================================================
# TEST 6
#
# Exact high-z LambdaCDM boundary.
#
# DEmodel=9 appends w(wave_zmax)=-1 and the Fortran branch
# imposes w=-1 above wave_zmax.
# ============================================================

banner("TEST 6: exact high-z Lambda boundary")

z_high = np.array([
    WAVE_ZMAX,
    WAVE_ZMAX + 1e-8,
    4.0,
    5.0,
    10.0,
    30.0,
    100.0,
])

for wave_type in (1, 2):

    results = nontrivial[wave_type]["results"]

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
        f"wave_type={wave_type}: "
        f"max high-z |w+1|={max_error:.3e}"
    )

    assert max_error < 1e-12

print("PASS: D4 and Haar join exactly to w=-1 at zmax")


# ============================================================
# TEST 7
#
# Check behavior of dw/dz at zmax.
#
# The right branch is exactly constant.  If the current spline
# helper imposes zero derivative at its final node, the left
# derivative should also converge toward zero.
#
# This is printed as a diagnostic because it checks a property
# of the spline boundary convention rather than of wavelets.
# ============================================================

banner("TEST 7: derivative behavior at zmax")

for wave_type in (1, 2):

    results = nontrivial[wave_type]["results"]

    print(
        f"wave_type={wave_type} ({wave_labels[wave_type]})"
    )

    final_right = None

    for h in (
        1e-3,
        1e-4,
        1e-5,
        1e-6,
    ):

        z_eval = np.array([
            WAVE_ZMAX - h,
            WAVE_ZMAX,
            WAVE_ZMAX + h,
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

        final_right = dw_right

        print(
            f"  h={h:.0e}: "
            f"left={dw_left:+.6e}, "
            f"right={dw_right:+.6e}"
        )

    # The high-z branch itself must be exactly constant.
    assert abs(final_right) < 1e-10

print("PASS: high-z branch has exactly zero derivative")


# ============================================================
# TEST 8
#
# Independent continuity-equation check:
#
#   rho(z)/rho(0)
#     = exp[3 integral_0^z (1+w)/(1+z) dz].
#
# Integrate CAMB-returned w(z) numerically in Python and compare
# with CAMB rho normalized at z=0.
# ============================================================

banner("TEST 8: independent rho_DE continuity-equation check")

z_rho = np.linspace(
    0.0,
    8.0,
    30001,
)

for wave_type in (1, 2):

    results = nontrivial[wave_type]["results"]

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

    # Exclude the immediate vicinity of zmax, where a finite-grid
    # trapezoidal integral samples the explicit branch transition.
    dz_rho = (
        z_rho[1] - z_rho[0]
    )

    mask = (
        np.abs(z_rho - WAVE_ZMAX)
        > 2 * dz_rho
    )

    max_rel = np.max(
        np.abs(
            relative_error[mask]
        )
    )

    print(
        f"wave_type={wave_type}: "
        f"max relative rho error={max_rel:.3e}"
    )

    # Fortran integrates the cubic spline analytically while this
    # independent side uses a finite-grid trapezoid rule.
    assert max_rel < 5e-5

print("PASS: CAMB rho_DE agrees with independent integration of CAMB w(z)")


# ============================================================
# TEST 9
#
# P -> -P symmetry.
#
# Because the inverse wavelet transform is linear around w=-1:
#
#       w(P)  = -1 + Psi P
#       w(-P) = -1 - Psi P
#
# hence at all 16 interior wavelet samples:
#
#       w(P) + w(-P) = -2.
#
# The appended zmax boundary is also exactly -1 in both cases.
# ============================================================

banner("TEST 9: P -> -P symmetry around w=-1")

for wave_type in (1, 2):

    pars_plus = make_wavelet_params(
        wave_type=wave_type,
        p=p_test,
    )

    pars_minus = make_wavelet_params(
        wave_type=wave_type,
        p=-p_test,
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
        f"wave_type={wave_type}: "
        f"max |w(P)+w(-P)+2|={symmetry_error:.3e}"
    )

    assert symmetry_error < 1e-13

print("PASS: wavelet reconstruction is symmetric and linear around LambdaCDM")


# ============================================================
# TEST 10
#
# Coefficient-amplitude scaling.
#
# For any scalar A:
#
#       delta_w(A P) = A delta_w(P).
#
# Check A=2 directly in the stored spline nodes.
# ============================================================

banner("TEST 10: wavelet amplitude scales linearly with coefficients")

for wave_type in (1, 2):

    pars_a = make_wavelet_params(
        wave_type=wave_type,
        p=p_test,
    )

    pars_b = make_wavelet_params(
        wave_type=wave_type,
        p=2.0 * p_test,
    )

    wa = np.asarray(
        pars_a.DarkEnergy.spline_w
    )[:-1]

    wb = np.asarray(
        pars_b.DarkEnergy.spline_w
    )[:-1]

    scaling_error = np.max(
        np.abs(
            (wb + 1.0)
            - 2.0 * (wa + 1.0)
        )
    )

    print(
        f"wave_type={wave_type}: "
        f"max coefficient-scaling error={scaling_error:.3e}"
    )

    assert scaling_error < 1e-13

print("PASS: scaling all P_i scales w+1 linearly")


# ============================================================
# TEST 11
#
# The wave_type selector must genuinely change a nontrivial
# realization.
#
# Same coefficient vector, different basis family -> generally
# different reconstructed node values.
# ============================================================

banner("TEST 11: D4 and Haar give different nontrivial realizations")

w_d4 = nontrivial[1]["w_nodes"]
w_haar = nontrivial[2]["w_nodes"]

wavelet_difference = np.max(
    np.abs(
        w_d4 - w_haar
    )
)

print(
    "max |w_D4 - w_Haar| at nodes =",
    wavelet_difference
)

assert wavelet_difference > 1e-6

print("PASS: wave_type selector genuinely changes the reconstruction")


# ============================================================
# TEST 12
#
# One-hot coefficient test.
#
# For each P_i, set only that coefficient to one.  The result
# must equal column i of Psi exactly.
#
# This checks coefficient ordering and is especially useful for
# catching accidental changes in the cA4/cD4/... bookkeeping.
# ============================================================

banner("TEST 12: one-hot coefficient ordering")

for wave_type in (1, 2):

    Psi = basis_matrices[wave_type]

    max_one_hot_error = 0.0

    for i in range(N_WAVE):

        p = np.zeros(
            N_WAVE,
            dtype=float,
        )

        p[i] = 1.0

        _, _, delta_w = reconstruct_wavelet(
            wave_type=wave_type,
            p=p,
        )

        err = np.max(
            np.abs(
                delta_w
                - Psi[:, i]
            )
        )

        max_one_hot_error = max(
            max_one_hot_error,
            err,
        )

    print(
        f"wave_type={wave_type}: "
        f"max one-hot basis error={max_one_hot_error:.3e}"
    )

    assert max_one_hot_error < 1e-14

print("PASS: coefficient ordering is internally consistent")


# ============================================================
# TEST 13
#
# Scaling coefficient only.
#
# For an orthonormal 16-point transform, the coarsest scaling
# coefficient cA4 must reconstruct a constant delta_w.
#
# This is a useful physical interpretation check: P0 controls
# the broadest/global offset in 1+w.
# ============================================================

banner("TEST 13: scaling coefficient reconstructs a constant mode")

p_scaling = np.zeros(
    N_WAVE,
    dtype=float,
)

p_scaling[0] = 1.0

for wave_type in (1, 2):

    _, _, delta_w = reconstruct_wavelet(
        wave_type=wave_type,
        p=p_scaling,
    )

    variation = (
        delta_w.max()
        - delta_w.min()
    )

    mean_value = np.mean(
        delta_w
    )

    print(
        f"wave_type={wave_type}: "
        f"mean delta_w={mean_value:.12e}, "
        f"max-min={variation:.3e}"
    )

    assert variation < 1e-12

print("PASS: P0 is the coarsest constant/scaling mode")


# ============================================================
# Diagnostic plot 1: all 16 basis functions
# ============================================================

banner("Generating wavelet basis-function plots")

z_inner = np.linspace(
    0.0,
    WAVE_ZMAX,
    N_WAVE,
    endpoint=False,
)

for wave_type in (1, 2):

    Psi = basis_matrices[wave_type]

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(12, 9),
        sharex=True,
    )

    axes = axes.ravel()

    for i in range(N_WAVE):

        axes[i].step(
            z_inner,
            Psi[:, i],
            where="mid",
            lw=1.5,
        )

        axes[i].axhline(
            0.0,
            ls=":",
            lw=0.8,
        )

        axes[i].set_title(
            rf"$P_{{{i}}}=1$",
            fontsize=10,
        )

        axes[i].minorticks_on()

        axes[i].tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
        )

    for ax in axes[-4:]:
        ax.set_xlabel(
            r"$z$",
            fontsize=11,
        )

    fig.suptitle(
        f"Wavelet basis: {wave_labels[wave_type]}",
        fontsize=14,
    )

    fig.tight_layout()

    filename = (
        "wavelet_basis_d4.pdf"
        if wave_type == 1
        else "wavelet_basis_haar.pdf"
    )

    fig.savefig(
        filename,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# Diagnostic plot 2: w(z) and rho_DE(z)
# ============================================================

banner("Generating wavelet w(z) and rho_DE(z) plots")

z_plot = np.linspace(
    0.0,
    6.0,
    5000,
)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
)

for wave_type in (1, 2):

    results = nontrivial[wave_type]["results"]
    z_nodes = nontrivial[wave_type]["z_nodes"]
    w_nodes = nontrivial[wave_type]["w_nodes"]

    rho_plot, w_plot = camb_rho_w_at_z(
        results,
        z_plot,
    )

    rho_ratio = (
        rho_plot
        / rho_plot[0]
    )

    # w(z)
    line, = ax[0].plot(
        z_plot,
        w_plot,
        lw=2,
        label=wave_labels[wave_type],
    )

    ax[0].scatter(
        z_nodes,
        w_nodes,
        s=25,
        color=line.get_color(),
        zorder=5,
    )

    # rho_DE(z)/rho_DE,0
    ax[1].plot(
        z_plot,
        rho_ratio,
        lw=2,
        label=wave_labels[wave_type],
    )


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

for a in ax:

    a.axvline(
        WAVE_ZMAX,
        ls=":",
        lw=1,
        label=r"$z_{\rm max}$",
    )

    a.set_xlim(
        0.0,
        6.0,
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
    "wavelet_w_rho.pdf",
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Diagnostic plot 3: rho consistency
#
# Solid  = CAMB
# Dashed = independent Python integration of CAMB w(z)
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5),
)

for wave_type in (1, 2):

    results = nontrivial[wave_type]["results"]

    rho_camb, w_camb = camb_rho_w_at_z(
        results,
        z_plot,
    )

    rho_camb_ratio = (
        rho_camb
        / rho_camb[0]
    )

    rho_python_ratio = rho_ratio_from_w(
        z_plot,
        w_camb,
    )

    line, = ax.plot(
        z_plot,
        rho_camb_ratio,
        lw=2,
        label=f"CAMB: {wave_labels[wave_type]}",
    )

    ax.plot(
        z_plot,
        rho_python_ratio,
        lw=1.5,
        ls="--",
        color=line.get_color(),
        label=f"Python integral: {wave_labels[wave_type]}",
    )


ax.axhline(
    1.0,
    ls="--",
    lw=1,
)

ax.axvline(
    WAVE_ZMAX,
    ls=":",
    lw=1,
)

ax.set_xlim(
    0.0,
    6.0,
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
    "wavelet_rho_consistency.pdf",
    bbox_inches="tight",
)

plt.show()


banner("ALL WAVELET CONSISTENCY TESTS PASSED")
