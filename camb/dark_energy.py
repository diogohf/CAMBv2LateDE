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
        ("w0", c_double, "Constant-w or CPL w0"),
        ("w1", c_double, "CPL wa"),
        ("z_knot", AllocatableArrayDouble, "Bin upper redshift boundaries"),
        ("w_knot", AllocatableArrayDouble, "Equation of state in each bin"),
    ]

    def set_params(self,DEmodel=1,w0=-1,w1=0,z_knot=None,w_knot=None):
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
