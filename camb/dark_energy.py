from .baseconfig import F2003Class, fortran_class, numpy_1d, CAMBError, np, \
    AllocatableArrayDouble, f_pointer
from ctypes import c_int, c_double, byref, POINTER, c_bool

# Optional dependency used only by DEmodel=9 (wavelets).
# CAMB can still be imported and used without PyWavelets when DEmodel != 9.
try:
    import pywt
except ImportError:
    pywt = None

class DarkEnergyModel(F2003Class):
    """
    Abstract base class for dark energy model implementations.
    """
    _fields_ = [
        ("__is_cosmological_constant", c_bool),
        ("__num_perturb_equations", c_int),
        ("w_lam", c_double),
        ("wa",c_double),
        ("cs2_lam", c_double),
        ("no_perturbations",c_bool)]

    def validate_params(self):
        return True

@fortran_class
#DHFS
class LateDE(DarkEnergyModel):
    """
    Base class for late-time dark-energy parameterizations.

    Supports constant-w, CPL, and piecewise-constant binned w(z) models.
    For binned models, ``z_knot`` defines the upper bin boundaries and
    ``w_knot`` the corresponding w(z) values. Above the last bin, w = -1.
    """
    
    _fortran_class_module_ = 'LateDE'
    _fortran_class_name_ = 'TLateDE'

    _fields_ = [
        ("DEmodel", c_int, "Dark-energy parameterization"),
        # Constant w and CPL
        ("w0", c_double, "Constant-w or CPL w0"),
        ("w1", c_double, "CPL wa"),
        # Bin w
        ("z_knot", AllocatableArrayDouble, "Bin upper redshift boundaries"),
        ("w_knot", AllocatableArrayDouble, "Equation of state in each bin"),
        # Flexknots
        ("a_flexknot", AllocatableArrayDouble, "Flexknot scale-factor positions"),
        ("w_flexknot", AllocatableArrayDouble, "Flexknot equation-of-state values"),
        # Cubic-spline
        ("spline_z", AllocatableArrayDouble,"Fixed cubic-spline redshift nodes"),
        ("spline_w", AllocatableArrayDouble,"Equation of state at cubic-spline nodes"),
        # Chebyshev
        ("cheb_c0", c_double, "Chebyshev C0"),
        ("cheb_c1", c_double, "Chebyshev C1"),
        ("cheb_c2", c_double, "Chebyshev C2"),
        ("cheb_c3", c_double, "Chebyshev C3"),
        ("cheb_zmax", c_double, "Maximum Chebyshev redshift"),
        ("cheb_delta", c_double, "High-z transition width"),
        # Bernstein
        ("bern_b0", c_double, "Bernstein b0"),
        ("bern_b1", c_double, "Bernstein b1"),
        ("bern_b2", c_double, "Bernstein b2"),
        ("bern_b3", c_double, "Bernstein b3"),
        ("bern_zmax", c_double, "Maximum Bernstein redshift"),
        ("bern_delta", c_double, "High-z transition width"),
        # Gaussian-Process (DEmodel=8): 
        # The GP kernel is constructed on the Python side:
        #   gp_kernel = 1 : squared-exponential / RBF
        #   gp_kernel = 2 : exponential / Holsclaw alpha=1
        # We reuse the spline interpolation
        # GP references:
        # [1] Holsclaw et al. 2010, arXiv: 1009.5443 >>> See appendix A
        # [2] Holsclaw et al. 2010, arXiv: 1011.3079
        # [3] Seikel et al. 2012, arXiv: 1204.2832
        #
        # Wavelets (DEmodel=9):
        # The inverse discrete wavelet transform is constructed on the Python
        # side and the resulting w(z) samples are passed to the same
        # spline_z/spline_w backend used by DEmodel=5 and DEmodel=8.
        #
        # wave_type = 1 : Daubechies D4, i.e. PyWavelets "db2"
        # wave_type = 2 : Haar
        #
        # Reference:
        # Hojjati, Pogosian & Zhao 2010, arXiv:0912.4843
        #
        # Fourier - Tamayo & Vazquez 2019, arXiv:1901.08679
        ("fourier_n", c_int, "Number of Fourier harmonics"),
        ("fourier_c0", c_double, "Mean displacement of w from -1"),
        ("fourier_zmed", c_double, "End of Fourier region"),
        ("fourier_zini", c_double, "End of linear transition to w=-1"),
        ("fourier_an", AllocatableArrayDouble, "Fourier sine coefficients A_n"),
        ("fourier_bn", AllocatableArrayDouble, "Fourier cosine coefficients B_n"),
        ("fourier_zint", AllocatableArrayDouble, "Grid for Fourier continuity integral"),
        ("fourier_Iint", AllocatableArrayDouble, "Precomputed Fourier continuity integral"),
    ]

    def set_params(self,
                    DEmodel=1,
                    # Cosntant-w and w0wa
                    w0=-1,
                    w1=0,
                    # Bin-w
                    z_knot=None,
                    w_knot=None,
                    # Flexknots
                    a_flexknot=None,
                    w_flexknot=None,
                    # Cubic-spline
                    spline_z=None,
                    spline_w=None,
                    # Chebyshev
                    cheb_c0=1.0,
                    cheb_c1=0.0,
                    cheb_c2=0.0,
                    cheb_c3=0.0,
                    cheb_zmax=3.5,
                    cheb_delta=1.0,
                    # Bernstein
                    bern_b0=-1.0,
                    bern_b1=-1.0,
                    bern_b2=-1.0,
                    bern_b3=-1.0,
                    bern_zmax=3.5,
                    bern_delta=1.0,
                    # Gaussian-process
                    gp_z=None,
                    gp_w=None,
                    gp_kernel=1,
                    gp_sigma=0.2,
                    gp_ell=0.6,
                    gp_q0=0.0,
                    gp_q1=0.0,
                    gp_q2=0.0,
                    gp_q3=0.0,
                    gp_q4=0.0,
                    gp_q5=0.0,
                    gp_q6=0.0,
                    gp_q7=0.0,
                    gp_q8=0.0,
                    gp_q9=0.0,
                    gp_q10=0.0,
                    gp_q11=0.0,
                    gp_q12=0.0,
                    gp_q13=0.0,
                    gp_zmax=3.5,
                    # Wavelets
                    wave_p0=0.0,
                    wave_p1=0.0,
                    wave_p2=0.0,
                    wave_p3=0.0,
                    wave_p4=0.0,
                    wave_p5=0.0,
                    wave_p6=0.0,
                    wave_p7=0.0,
                    wave_p8=0.0,
                    wave_p9=0.0,
                    wave_p10=0.0,
                    wave_p11=0.0,
                    wave_p12=0.0,
                    wave_p13=0.0,
                    wave_p14=0.0,
                    wave_p15=0.0,
                    wave_zmax=3.5,
                    wave_type=1,
                    # Fourier
                    fourier_n=2,
                    fourier_c0=0.0,
                    fourier_a1=0.0,
                    fourier_b1=0.0,
                    fourier_a2=0.0,
                    fourier_b2=0.0,
                    fourier_a3=0.0,
                    fourier_b3=0.0,
                    fourier_a4=0.0,
                    fourier_b4=0.0,
                    fourier_zmed=2.8,
                    fourier_zini=3.0,
                    fourier_nint=4097,
                   ):
        self.DEmodel = DEmodel
        self.w0 = w0
        self.w1 = w1

        if DEmodel == 3:
            if z_knot is None or w_knot is None:
                raise ValueError(
                    "DEmodel=3 requires z_knot and w_knot"
                )

            z_knot = np.asarray(z_knot, dtype=np.float64)
            w_knot = np.asarray(w_knot, dtype=np.float64)

            if len(z_knot) != len(w_knot):
                raise ValueError(
                    "z_knot and w_knot must have the same length"
                )

            if np.any(np.diff(z_knot) <= 0):
                raise ValueError(
                    "z_knot must be strictly increasing"
                )

            self.z_knot = z_knot
            self.w_knot = w_knot

        elif DEmodel == 4:
            a_flexknot = np.asarray(
                a_flexknot,
                dtype=np.float64
            )

            w_flexknot = np.asarray(
                w_flexknot,
                dtype=np.float64
            )

            if a_flexknot.ndim != 1:
                raise ValueError(
                    "a_flexknot must be one-dimensional"
                )

            if w_flexknot.ndim != 1:
                raise ValueError(
                    "w_flexknot must be one-dimensional"
                )

            if len(a_flexknot) != len(w_flexknot):
                raise ValueError(
                    "a_flexknot and w_flexknot "
                    "must have the same length"
                )

            if len(a_flexknot) == 0:
                raise ValueError(
                    "At least one flexknot is required"
                )

            if len(a_flexknot) > 1:

                if not np.isclose(a_flexknot[0], 0.0):
                    raise ValueError(
                        "First flexknot must be at a=0"
                    )

                if not np.isclose(a_flexknot[-1], 1.0):
                    raise ValueError(
                        "Last flexknot must be at a=1"
                    )

                if np.any(np.diff(a_flexknot) <= 0):
                    raise ValueError(
                        "a_flexknot must be strictly increasing"
                    )

            self.a_flexknot = a_flexknot
            self.w_flexknot = w_flexknot

        elif DEmodel == 5:

            if spline_z is None or spline_w is None:
                raise ValueError(
                    "DEmodel=5 requires spline_z and spline_w"
                )

            spline_z = np.asarray(
                spline_z,
                dtype=np.float64
            )

            spline_w = np.asarray(
                spline_w,
                dtype=np.float64
            )

            if spline_z.ndim != 1:
                raise ValueError(
                    "spline_z must be one-dimensional"
                )

            if spline_w.ndim != 1:
                raise ValueError(
                    "spline_w must be one-dimensional"
                )

            if len(spline_z) != len(spline_w):
                raise ValueError(
                    "spline_z and spline_w must have the same length"
                )

            if len(spline_z) < 3:
                raise ValueError(
                    "Cubic spline requires at least 3 nodes"
                )

            if not np.isclose(
                    spline_z[0],
                    0.0):
                raise ValueError(
                    "First cubic-spline node must be z=0"
                )

            if np.any(
                    np.diff(spline_z) <= 0):
                raise ValueError(
                    "spline_z must be strictly increasing"
                )

            # High-z LambdaCDM boundary condition:
            #
            # w(zmax) = -1
            if not np.isclose(
                    spline_w[-1],
                    -1.0,
                    atol=1e-12,
                    rtol=0):
                raise ValueError(
                    "For DEmodel=5, the last spline node must "
                    "satisfy spline_w[-1] = -1 so that the "
                    "cubic spline joins LambdaCDM at high redshift"
                )

            self.spline_z = spline_z
            self.spline_w = spline_w

        elif DEmodel == 6:
            if cheb_zmax <= 0:
                raise ValueError(
                    "cheb_zmax must be positive"
                )

            if cheb_delta <= 0:
                raise ValueError(
                    "cheb_delta must be positive"
                )

            self.cheb_c0 = cheb_c0
            self.cheb_c1 = cheb_c1
            self.cheb_c2 = cheb_c2
            self.cheb_c3 = cheb_c3
            self.cheb_zmax = cheb_zmax
            self.cheb_delta = cheb_delta
        
        elif DEmodel == 7:

            if bern_zmax <= 0:
                raise ValueError(
                    "bern_zmax must be positive"
                )

            if bern_delta <= 0:
                raise ValueError(
                    "bern_delta must be positive"
                )

            self.bern_b0 = bern_b0
            self.bern_b1 = bern_b1
            self.bern_b2 = bern_b2
            self.bern_b3 = bern_b3
            self.bern_zmax = bern_zmax
            self.bern_delta = bern_delta

        elif DEmodel == 8:

            # --------------------------------------------------------
            # GP kernel selector
            #
            # gp_kernel = 1:
            #   Squared-exponential / RBF kernel
            #
            #   K_ij = sigma^2 exp[-0.5 ((zi-zj)/ell)^2]
            #
            # gp_kernel = 2:
            #   Exponential kernel, equivalent to the Holsclaw
            #   covariance family with alpha=1 after writing
            #
            #   rho = exp(-1/ell)
            #
            #   so that
            #
            #   K_ij = sigma^2 exp[-|zi-zj|/ell].
            #
            # References:
            # Holsclaw et al. 2010, arXiv:1009.5443
            # Holsclaw et al. 2010, arXiv:1011.3079
            # --------------------------------------------------------

            if gp_kernel not in (1, 2):
                raise ValueError(
                    "gp_kernel must be 1 (squared-exponential) "
                    "or 2 (exponential/Holsclaw alpha=1)"
                )

            if gp_sigma <= 0:
                raise ValueError("gp_sigma must be positive")

            if gp_ell <= 0:
                raise ValueError("gp_ell must be positive")

            # --------------------------------------------------------
            # Fixed GP redshift grid
            # --------------------------------------------------------

            N = 15

            gp_z = np.linspace(
                0.0,
                gp_zmax,
                N
            )

            # --------------------------------------------------------
            # Latent independent Gaussian parameters
            #
            # q_i ~ N(0,1)
            #
            # Last w node is constrained to -1, so we need N-1 q's.
            # --------------------------------------------------------

            q = np.array([
                gp_q0,
                gp_q1,
                gp_q2,
                gp_q3,
                gp_q4,
                gp_q5,
                gp_q6,
                gp_q7,
                gp_q8,
                gp_q9,
                gp_q10,
                gp_q11,
                gp_q12,
                gp_q13,
            ], dtype=np.float64)

            # --------------------------------------------------------
            # GP covariance matrix
            # --------------------------------------------------------

            dz = gp_z[:, None] - gp_z[None, :]

            if gp_kernel == 1:
                # Squared-exponential / RBF kernel:
                #
                # K_ij = sigma^2 exp[-0.5 ((zi-zj)/ell)^2]
                #
                # This gives a very smooth GP prior.
                K = (
                    gp_sigma**2
                    * np.exp(
                        -0.5 * (dz / gp_ell)**2
                    )
                )

            elif gp_kernel == 2:
                # Exponential kernel:
                #
                # K_ij = sigma^2 exp[-|zi-zj|/ell]
                #
                # This is equivalent to the Holsclaw et al.
                # covariance family
                #
                # K = kappa^2 rho^{|zi-zj|^alpha}
                #
                # for alpha=1, kappa=sigma, and
                # rho = exp(-1/ell).
                #
                # It permits rougher node-to-node GP realizations
                # than the squared-exponential kernel.
                K = (
                    gp_sigma**2
                    * np.exp(
                        -np.abs(dz) / gp_ell
                    )
                )

            # --------------------------------------------------------
            # CONDITION THE GP ON
            #
            # w(zmax) = -1
            #
            # Since the GP mean is also -1, this means the residual
            # delta w(zmax) = 0.
            # --------------------------------------------------------

            K_ff = K[:-1, :-1]
            K_fb = K[:-1, -1]
            K_bb = K[-1, -1]

            K_cond = (K_ff - np.outer(K_fb, K_fb) / K_bb)

            # Numerical jitter for Cholesky stability
            jitter = 1.0e-10

            K_cond = (
            K_cond
            + jitter
            * np.max(np.diag(K_cond))
            * np.eye(N - 1)
        )

            # --------------------------------------------------------
            # Cholesky:
            #
            # K_cond = L L^T
            # --------------------------------------------------------

            L = np.linalg.cholesky(K_cond)

            # --------------------------------------------------------
            # GP realization
            #
            # mean = -1
            # --------------------------------------------------------

            delta_w = L @ q

            gp_w = np.empty(N)

            gp_w[:-1] = -1.0 + delta_w

            # Exact high-z boundary
            gp_w[-1] = -1.0

            # --------------------------------------------------------
            # Send realization to existing Fortran spline backend
            # --------------------------------------------------------

            self.spline_z = gp_z
            self.spline_w = gp_w
            
        elif DEmodel == 9:

            # --------------------------------------------------------
            # WAVELET PARAMETERIZATION OF w(z)
            #
            # Hojjati, Pogosian & Zhao (2010), arXiv:0912.4843:
            #
            #     1 + w(z_j) = sum_i P_i psi_i(z_j)
            #
            # Therefore all P_i = 0 gives exact LambdaCDM:
            #
            #     w(z) = -1.
            #
            # The discrete inverse wavelet transform is performed
            # here in Python.  CAMB/Fortran only receives the resulting
            # redshift samples through spline_z and spline_w.
            #
            # We use 16 coefficients, matching the 16-parameter
            # resolution used in Hojjati et al.
            #
            # wave_type = 1 : Daubechies D4 = PyWavelets "db2"
            # wave_type = 2 : Haar
            #
            # Boundary convention:
            #   - inverse DWT uses "periodization" on the 16 interior
            #     samples, giving an exactly 16-dimensional orthogonal
            #     discrete transform;
            #   - z = wave_zmax is then appended explicitly with w=-1;
            #   - the Fortran layer keeps w=-1 for z > wave_zmax.
            # --------------------------------------------------------

            if pywt is None:
                raise ImportError(
                    "DEmodel=9 requires PyWavelets. "
                    "Install it with `pip install PyWavelets` "
                    "or `conda install pywavelets`."
                )

            if wave_zmax <= 0:
                raise ValueError(
                    "wave_zmax must be positive"
                )

            if wave_type == 1:
                # Daubechies D4 in the convention where D4 denotes
                # four filter coefficients.  PyWavelets calls this db2.
                wavelet_name = "db2"

            elif wave_type == 2:
                wavelet_name = "haar"

            else:
                raise ValueError(
                    "wave_type must be 1 (Daubechies D4 / db2) "
                    "or 2 (Haar)"
                )

            # --------------------------------------------------------
            # Wavelet coefficients
            #
            # For N = 16 = 2^4 we use a complete four-level
            # multiresolution coefficient vector:
            #
            #   P = [ cA4,
            #         cD4,
            #         cD3(1:2),
            #         cD2(1:4),
            #         cD1(1:8) ]
            #
            # giving
            #
            #   1 + 1 + 2 + 4 + 8 = 16 coefficients.
            #
            # This ordering is explicit and reproducible and does not
            # depend on PyWavelets' automatic maximum-level choice.
            # --------------------------------------------------------

            P = np.array([
                wave_p0,
                wave_p1,
                wave_p2,
                wave_p3,
                wave_p4,
                wave_p5,
                wave_p6,
                wave_p7,
                wave_p8,
                wave_p9,
                wave_p10,
                wave_p11,
                wave_p12,
                wave_p13,
                wave_p14,
                wave_p15,
            ], dtype=np.float64)

            # Explicit coefficient hierarchy for a 16-point signal.
            coeffs = [
                P[0:1],    # cA4 : scaling coefficient
                P[1:2],    # cD4 : coarsest detail
                P[2:4],    # cD3 : 2 coefficients
                P[4:8],    # cD2 : 4 coefficients
                P[8:16],   # cD1 : 8 finest-scale coefficients
            ]

            # --------------------------------------------------------
            # Reconstruct delta w = 1 + w on 16 equally spaced
            # redshift samples below wave_zmax.
            # --------------------------------------------------------

            delta_w = pywt.waverec(
                coeffs,
                wavelet=wavelet_name,
                mode="periodization",
            )

            delta_w = np.asarray(
                delta_w,
                dtype=np.float64,
            )

            if delta_w.size != 16:
                raise RuntimeError(
                    "Wavelet reconstruction did not return 16 samples"
                )

            wave_w_inner = -1.0 + delta_w

            # --------------------------------------------------------
            # Redshift grid
            #
            # The 16 wavelet samples occupy [0, wave_zmax), and the
            # exact LambdaCDM boundary point (wave_zmax, -1) is appended.
            #
            # This avoids overwriting any of the 16 wavelet samples
            # merely to impose the high-z boundary condition.
            # --------------------------------------------------------

            wave_z_inner = np.linspace(
                0.0,
                wave_zmax,
                16,
                endpoint=False,
                dtype=np.float64,
            )

            wave_z = np.concatenate([
                wave_z_inner,
                np.array([wave_zmax], dtype=np.float64),
            ])

            wave_w = np.concatenate([
                wave_w_inner,
                np.array([-1.0], dtype=np.float64),
            ])

            # --------------------------------------------------------
            # Send the reconstructed wavelet w(z) to the existing
            # Fortran cubic-spline backend.
            # --------------------------------------------------------

            self.spline_z = wave_z
            self.spline_w = wave_w
            
        elif DEmodel == 10:

            # --------------------------------------------------------
            # FOURIER PARAMETERIZATION OF w(a)
            #
            # Based on:
            #   Tamayo & Vazquez 2019, arXiv:1901.08679
            #
            # We use the same Fourier-in-scale-factor construction,
            # but shift the constant term so LambdaCDM is at the
            # origin of parameter space:
            #
            #   w_F(a) = -1 + c0
            #            + sum_n [
            #                A_n sin(n theta (a-a_med))
            #              + B_n cos(n theta (a-a_med))
            #              ]
            #
            # with
            #
            #   theta = 2*pi/(1-a_med).
            #
            # The original paper writes w0/2 as the constant term;
            # our convention is related by
            #
            #   c0 = 1 + w0_paper/2.
            #
            # Thus c0=A_n=B_n=0 is exactly LambdaCDM.
            #
            # For z_med < z <= z_ini, a straight line connects the
            # Fourier value at z_med to w=-1 at z_ini.  For z>z_ini,
            # w=-1 exactly.
            # --------------------------------------------------------

            if fourier_n not in (1, 2, 3, 4):
                raise ValueError(
                    "fourier_n must be 1, 2, 3, or 4"
                )

            if fourier_zmed <= 0:
                raise ValueError(
                    "fourier_zmed must be positive"
                )

            if fourier_zini <= fourier_zmed:
                raise ValueError(
                    "fourier_zini must be larger than fourier_zmed"
                )

            if int(fourier_nint) != fourier_nint or fourier_nint < 257:
                raise ValueError(
                    "fourier_nint must be an integer >= 257"
                )

            # Fixed scalar keywords are convenient for Cobaya/YAML.
            # The active first four harmonics are packed into arrays
            # passed to the Fortran layer.
            all_a = np.array([
                fourier_a1,
                fourier_a2,
                fourier_a3,
                fourier_a4,
            ], dtype=np.float64)

            all_b = np.array([
                fourier_b1,
                fourier_b2,
                fourier_b3,
                fourier_b4,
            ], dtype=np.float64)

            fourier_an = all_a[:fourier_n].copy()
            fourier_bn = all_b[:fourier_n].copy()

            self.fourier_n = fourier_n
            self.fourier_c0 = fourier_c0
            self.fourier_zmed = fourier_zmed
            self.fourier_zini = fourier_zini
            self.fourier_an = fourier_an
            self.fourier_bn = fourier_bn

            # --------------------------------------------------------
            # Precompute the continuity integral
            #
            #   I(z) = integral_0^z (1+w)/(1+z') dz'
            #
            # The Fortran layer evaluates w(a) from the exact Fourier
            # expression, but interpolates this precomputed I(z) for
            # rho_DE.  This avoids repeated sine/cosine-integral
            # special-function evaluations during CAMB evolution.
            # --------------------------------------------------------

            fourier_zint = np.linspace(
                0.0,
                fourier_zini,
                int(fourier_nint),
                dtype=np.float64,
            )

            agrid = 1.0 / (1.0 + fourier_zint)
            amed = 1.0 / (1.0 + fourier_zmed)
            aini = 1.0 / (1.0 + fourier_zini)

            theta = 2.0 * np.pi / (1.0 - amed)

            fourier_w = np.empty_like(fourier_zint)

            mask_fourier = fourier_zint <= fourier_zmed
            mask_linear = ~mask_fourier

            af = agrid[mask_fourier]

            wf = np.full(
                af.shape,
                -1.0 + fourier_c0,
                dtype=np.float64,
            )

            for iharm in range(fourier_n):
                n = iharm + 1
                phase = n * theta * (af - amed)

                wf += (
                    fourier_an[iharm] * np.sin(phase)
                    + fourier_bn[iharm] * np.cos(phase)
                )

            fourier_w[mask_fourier] = wf

            # At a=a_med: sin(0)=0 and cos(0)=1.
            wmed = (
                -1.0
                + fourier_c0
                + np.sum(fourier_bn)
            )

            slope = (
                (wmed + 1.0)
                / (amed - aini)
            )

            fourier_w[mask_linear] = (
                -1.0
                + slope * (agrid[mask_linear] - aini)
            )

            integrand = (
                (1.0 + fourier_w)
                / (1.0 + fourier_zint)
            )

            fourier_Iint = np.zeros_like(
                fourier_zint
            )

            dz = np.diff(
                fourier_zint
            )

            fourier_Iint[1:] = np.cumsum(
                0.5
                * (
                    integrand[:-1]
                    + integrand[1:]
                )
                * dz
            )

            self.fourier_zint = fourier_zint
            self.fourier_Iint = fourier_Iint

@fortran_class
class DarkEnergyPPF(LateDE):
    """
    VM: CLASS IMPLEMENTS w, w0wa and binw

    """
    # cannot declare c_Gamma_ppf directly here as have not defined all fields in DarkEnergyEqnOfState (TCubicSpline)
    _fortran_class_module_ = 'DarkEnergyPPF'
    _fortran_class_name_ = 'TDarkEnergyPPF'


# short names for models that support w/wa
F2003Class._class_names.update({'ppf': DarkEnergyPPF})
