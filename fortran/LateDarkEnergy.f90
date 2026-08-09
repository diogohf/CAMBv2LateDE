module LateDE
    use DarkEnergyInterface
    use results
    use constants
    use classes
    implicit none

    type, extends(TDarkEnergyModel) :: TLateDE
        real(dl) :: w_n = 1._dl !Effective equation of state when oscillating
        real(dl) :: fde_zc = 0._dl ! energy density fraction at a_c (not the same as peak dark energy fraction)
        real(dl) :: zc  !transition redshift (scale factor a_c)
        real(dl) :: theta_i = const_pi/2 !Initial value
        !om is Omega of the early DE component today (assumed to be negligible compared to omega_lambda)
        !omL is the lambda component of the total dark energy omega
        real(dl), private :: a_c, pow, om, omL, acpow, freq, n !cached internally
        
        contains

        procedure :: ReadParams =>  TLateDE_ReadParams
        procedure, nopass :: PythonClass => TLateDE_PythonClass
        procedure, nopass :: SelfPointer => TLateDE_SelfPointer
        procedure :: Init => TLateDE_Init
        procedure :: w_de => TLateDE_w_de
        procedure :: grho_de => TLateDE_grho_de
        procedure :: PerturbedStressEnergy => TLateDE_PerturbedStressEnergy
        procedure :: PerturbationEvolve => TLateDE_PerturbationEvolve
    end type TLateDE

    public TLateDE

    contains

    subroutine TLateDE_ReadParams(this, Ini)
        use IniObjects
        class(TLateDE) :: this
        class(TIniFile), intent(in) :: Ini

        call this%TDarkEnergyModel%ReadParams(Ini)
        if (Ini%HasKey('AxionEffectiveFluid_a_c')) then
            error stop 'AxionEffectiveFluid inputs changed to AxionEffectiveFluid_fde_zc and AxionEffectiveFluid_zc'
        end if
        this%w_n  = Ini%Read_Double('AxionEffectiveFluid_w_n')
        this%fde_zc  = Ini%Read_Double('AxionEffectiveFluid_fde_zc')
        this%zc  = Ini%Read_Double('AxionEffectiveFluid_zc')
        call Ini%Read('AxionEffectiveFluid_theta_i', this%theta_i)
    end subroutine TLateDE_ReadParams


    function TLateDE_PythonClass()
        character(LEN=:), allocatable :: TLateDE_PythonClass

        TLateDE_PythonClass = 'AxionEffectiveFluid'
    end function TLateDE_PythonClass

    subroutine TLateDE_SelfPointer(cptr,P)
        use iso_c_binding
        Type(c_ptr) :: cptr
        Type (TLateDE), pointer :: PType
        class (TPythonInterfacedClass), pointer :: P

        call c_f_pointer(cptr, PType)
        P => PType
    end subroutine TLateDE_SelfPointer

    subroutine TLateDE_Init(this, State)
        use classes
        class(TLateDE), intent(inout) :: this
        class(TCAMBdata), intent(in), target :: State
        real(dl) :: grho_rad, F, p, mu, xc, n

        select type(State)
        class is (CAMBdata)
            this%is_cosmological_constant = this%fde_zc==0
            this%pow = 3*(1+this%w_n)
            this%a_c = 1/(1+this%zc)
            this%acpow = this%a_c**this%pow
            !Omega in early de at z=0
            this%om = 2*this%fde_zc/(1-this%fde_zc)*&
                (State%grho_no_de(this%a_c)/this%a_c**4/State%grhocrit + State%Omega_de)/(1 + 1/this%acpow)
            this%omL = State%Omega_de - this%om !Omega_de is total dark energy density today
            this%num_perturb_equations = 2
            if (this%w_n < 0.9999) then
                ! n <> infinity
                !get (very) approximate result for sound speed parameter; arXiv:1806.10608 Eq 30
                !(but mu may not exactly agree with what they used)
                n = nint((1+this%w_n)/(1-this%w_n))
                !Assume radiation domination, standard neutrino model; H0 factors cancel
                grho_rad = (kappa/c**2*4*sigma_boltz/c**3*State%CP%tcmb**4*Mpc**2*(1+default_nnu*7._dl/8*(4._dl/11)**(4._dl/3)))
                xc = this%a_c**2/2/sqrt(grho_rad/3)
                F=7./8
                p=1./2
                mu = 1/xc*(1-cos(this%theta_i))**((1-n)/2.)*sqrt((1-F)*(6*p+2)*this%theta_i/n/sin(this%theta_i))
                this%freq =  mu*(1-cos(this%theta_i))**((n-1)/2.)* &
                    sqrt(const_pi)*Gamma((n+1)/(2.*n))/Gamma(1+0.5/n)*2.**(-(n**2+1)/(2.*n))*3.**((1./n-1)/2)*this%a_c**(-6./(n+1)+3) &
                    *( this%a_c**(6*n/(n+1.))+1)**(0.5*(1./n-1))
                this%n = n
            end if
        end select
    end subroutine TLateDE_Init


    function TLateDE_w_de(this, a)
        class(TLateDE) :: this
        real(dl) :: TLateDE_w_de
        real(dl), intent(IN) :: a
        real(dl) :: om_t, apow, apc

        apow = a**this%pow
        apc = apow + this%acpow
        om_t = this%om*(1+this%acpow)/apc !early DE contribution to relative density
        TLateDE_w_de = om_t*(1+this%w_n)*apow/(apc*(this%omL+om_t)) - 1
    end function TLateDE_w_de

    function TLateDE_grho_de(this, a)  !relative density (8 pi G a^4 rho_de /grhov)
        class(TLateDE) :: this
        real(dl) :: TLateDE_grho_de, apc
        real(dl), intent(IN) :: a

        if(a == 0.d0)then
            TLateDE_grho_de = 0.d0
        else
            apc = a**this%pow + this%acpow
            TLateDE_grho_de = (this%omL*apc+this%om*(1+this%acpow))*a**4 &
                /(apc*(this%omL+this%om))
        endif
    end function TLateDE_grho_de

    subroutine TLateDE_PerturbationEvolve(this, ayprime, w, w_ix, &
        a, adotoa, k, z, y)
        class(TLateDE), intent(in) :: this
        real(dl), intent(inout) :: ayprime(:)
        real(dl), intent(in) :: a, adotoa, w, k, z, y(:)
        integer, intent(in) :: w_ix
        real(dl) Hv3_over_k, deriv, apow, acpow, apc, cs2, fac, k2

        k2 = k**2
        apow = a**this%pow
        acpow = this%acpow
        if (this%w_n < 0.9999) then
            !a**(2-6*this%w_n) = a**8/apow**2 since pow = 3*(1+w_n)
            fac = 2*this%freq**2*a**8/apow**2
            cs2 = (fac*(this%n-1) + k2)/(fac*(this%n+1) + k2)
        else
            cs2 = 1
        end if
        apc = apow + acpow
        Hv3_over_k =  3*adotoa* y(w_ix + 1) / k
        ! dw/dlog a/(1+w)
        deriv  = (acpow**2*(this%om+this%omL)+this%om*acpow-apow**2*this%omL)*this%pow &
            /(apc*(this%omL*apc+this%om*(1+acpow)))
        !density perturbation
        ayprime(w_ix) = -3 * adotoa * (cs2 - w) *  (y(w_ix) + Hv3_over_k) &
            -   k * y(w_ix + 1) - (1 + w) * k * z  - adotoa*deriv* Hv3_over_k
        !(1+w)v
        ayprime(w_ix + 1) = -adotoa * (1 - 3 * cs2 - deriv) * y(w_ix + 1) + &
            k * cs2 * y(w_ix)
    end subroutine TLateDE_PerturbationEvolve


    subroutine TLateDE_PerturbedStressEnergy(this, dgrhoe, dgqe, &
        a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1, ay, ayprime, w_ix)
        class(TLateDE), intent(inout) :: this
        real(dl), intent(out) :: dgrhoe, dgqe
        real(dl), intent(in) :: a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1
        real(dl), intent(in) :: ay(*)
        real(dl), intent(inout) :: ayprime(*)
        integer, intent(in) :: w_ix

        dgrhoe = ay(w_ix) * grhov_t
        dgqe = ay(w_ix + 1) * grhov_t
    end subroutine TLateDE_PerturbedStressEnergy

end module LateDE
