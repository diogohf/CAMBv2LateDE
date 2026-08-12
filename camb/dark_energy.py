from .baseconfig import F2003Class, fortran_class, numpy_1d, CAMBError, np, \
    AllocatableArrayDouble, f_pointer
from ctypes import c_int, c_double, byref, POINTER, c_bool


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
        # Chebyshev
        ("cheb_c0", c_double, "Chebyshev C0"),
        ("cheb_c1", c_double, "Chebyshev C1"),
        ("cheb_c2", c_double, "Chebyshev C2"),
        ("cheb_c3", c_double, "Chebyshev C3"),
        ("cheb_zmax", c_double, "Maximum Chebyshev redshift"),
        ("cheb_delta", c_double, "High-z transition width"),
    ]

    def set_params(self,
                    DEmodel=1,
                    w0=-1,
                    w1=0,
                    z_knot=None,
                    w_knot=None,
                    a_flexknot=None,
                    w_flexknot=None,
                    cheb_c0=1.0,
                    cheb_c1=0.0,
                    cheb_c2=0.0,
                    cheb_c3=0.0,
                    cheb_zmax=3.5,
                    cheb_delta=1.0,
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
