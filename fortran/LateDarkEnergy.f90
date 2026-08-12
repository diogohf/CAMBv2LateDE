module LateDE
    use DarkEnergyInterface
    use results
    use constants
    use classes
    implicit none

    private
    real(dl) :: grho_de_today
    type, extends(TDarkEnergyModel) :: TLateDE
        integer  :: DEmodel
        ! Constant w and CPL
        real(dl) :: w0
        real(dl) :: w1
        ! Bin w
        real(dl), allocatable :: z_knot(:)
        real(dl), allocatable :: w_knot(:)
        ! Flexknots - Ormondroyd et al. 2025, arXiv:2503.08658
        real(dl), allocatable :: a_flexknot(:)
        real(dl), allocatable :: w_flexknot(:)
        ! Chebyshev - Calderon et al. 2024, arXiv:2405.04216v1
        real(dl) :: cheb_c0
        real(dl) :: cheb_c1
        real(dl) :: cheb_c2
        real(dl) :: cheb_c3
        real(dl) :: cheb_zmax
        real(dl) :: cheb_delta
        contains
        procedure :: ReadParams => TLateDE_ReadParams
        procedure :: Init => TLateDE_Init
        procedure :: PrintFeedback => TLateDE_PrintFeedback
        procedure :: w_de => TLateDE_w_de
        procedure :: grho_de => TLateDE_grho_de
        procedure :: Effective_w_wa => TLateDE_Effective_w_wa   !VM: wont be called with CASARINI (our mod)         
        procedure, nopass :: SelfPointer => TLateDE_SelfPointer
        procedure :: BackgroundDensityAndPressure => TLateDE_density 
    end type TLateDE

    public TLateDE

    contains

    function TLateDE_w_de(this, a) result(w_de)
        class(TLateDE) :: this
        real(dl), intent(in) :: a    
        real(dl) :: w_de, z
        real(dl) :: x
        real(dl) :: T0, T1, T2, T3
        real(dl) :: B0, B1, u
        integer :: i

        w_de = 0
        z = 1.0_dl/a - 1.0_dl

        select case (this%DEmodel)
            case(1) 
                ! Constant w
                w_de = this%w0
            
            case(2)
                ! CPL parametrization w0wa
                w_de = this%w0 + this%w1*(1._dl - a)
            
            case(3)
                ! Piecewise-constant binned w
                z = 1._dl / a - 1._dl
                w_de = -1._dl
                do i = 1, size(this%z_knot)
                    if (z < this%z_knot(i)) then
                        w_de = this%w_knot(i)
                        exit
                    end if
                end do
            
            case(4)
                ! Flexknots
                ! Appendix A2 of Ormondroyd et al. 2025, 2503.08658v2
                ! w(a) = m_i * a + c_i                 >>> Equation of state
                ! m_i  = (w_i+1 - w_i) / (a_i+1 - a_i) >>> Slope
                ! c_i  = w_i - m_i * a_i               >>> Intercept
                if (size(this%a_flexknot) == 1) then
                    w_de = this%w_flexknot(1)
                else if (a <= this%a_flexknot(1)) then
                    w_de = this%w_flexknot(1)
                else if (a >= this%a_flexknot(size(this%a_flexknot))) then 
                    w_de = this%w_flexknot(size(this%w_flexknot))
                else
                    do i = 1, size(this%a_flexknot)-1
                        if (a <= this%a_flexknot(i+1)) then
                            w_de = this%w_flexknot(i) + &
                                (this%w_flexknot(i+1)-this%w_flexknot(i)) * &
                                (a-this%a_flexknot(i)) / &
                                (this%a_flexknot(i+1)-this%a_flexknot(i))
                            exit
                        end if
                    end do       
                end if

            case(5)
                ! Chebyshev 
                ! Calderon et al. 2024, arXiv:2405.04216v1
                ! Eqs. (2.1), (2.2), (2.5)
                if (z <= this%cheb_zmax) then
                    x = -1._dl + 2._dl*z/this%cheb_zmax ! Because zmin=0
                    T0 = 1._dl
                    T1 = x
                    T2 = 2._dl*x**2 - 1._dl
                    T3 = 4._dl*x**3 - 3._dl*x
                    w_de = -(this%cheb_c0*T0 + &
                            this%cheb_c1*T1 + &
                            this%cheb_c2*T2 + &
                            this%cheb_c3*T3)
                else
                    ! Smooth transition to w=-1 beyond zmax 
                    u = log((1+z)/(1+this%cheb_zmax))
                    
                    B0 = 1._dl - (this%cheb_c0 + &
                                  this%cheb_c1 + &  
                                  this%cheb_c2 + &  
                                  this%cheb_c3)

                    B1 = -2._dl * (1+this%cheb_zmax)/this%cheb_zmax * &
                                  (this%cheb_c1 + & 
                                   4._dl*this%cheb_c2 + &
                                   9._dl*this%cheb_c3)
                    
                    w_de = -1._dl+(B0+B1*u)*exp(-u**2/this%cheb_delta**2)                    
                endif

            case default        
                stop "Invalid DEmodel"   
        end select
    end function TLateDE_w_de

    function TLateDE_grho_de(this, a) result(grho_de)
        ! Returns 8*pi*G * rho_de, no factor of a^4
        class(TLateDE) :: this
        real(dl), intent(in) :: a
        real(dl) :: grho_de, z
        ! Bin w
        real(dl) :: faci, temp
        integer  :: max_num_of_bins
        integer  :: i, j
        ! Flexknots
        real(dl) :: a_lower, a_upper, m_i, c_i, integral
        ! Chebyshev
        real(dl) :: alpha
        real(dl) :: q0, q1, q2, q3
        real(dl) :: Ipoly, Imax, Itrans
        real(dl) :: B0, B1, u

        grho_de = 0
        z = 1.0_dl/a - 1.0_dl

        select case (this%DEmodel)
            case(1)
                ! Constant w
                grho_de = grho_de_today * a**(-3 * (1 + this%w0))

            case(2)
                ! CPL parametrization w0wa
                grho_de = grho_de_today * a**(-3 * (1 + this%w0 + this%w1)) * exp(-3 * this%w1 * (1 - a))
            
            case(3)
                ! Piecewise-constant binned w
                max_num_of_bins = size(this%z_knot)
                do i = 1, max_num_of_bins
                    if ( i==1 ) then
                        faci = 1.0_dl
                    else
                        faci = 1.0_dl
                        do j = 1, i-1
                            temp = (1.0_dl + this%z_knot(j))**(3.0_dl * (this%w_knot(j) - this%w_knot(j+1)))
                            faci = faci * temp 
                        end do
                    end if
    
                    if (z < this%z_knot(i)) then
                        grho_de = grho_de_today * faci * a**(-3.0_dl * (1.0_dl + this%w_knot(i)))
                    exit
                    end if
                end do
                if ( z > this%z_knot(max_num_of_bins) ) then
                    faci = 1.0_dl
                    do i = 1, max_num_of_bins-1
                        temp = (1.0_dl+this%z_knot(i))**(3.0_dl * (this%w_knot(i) - this%w_knot(i+1)))
                        faci = faci * temp
                    end do
                    faci = faci * (1.0_dl + this%z_knot(max_num_of_bins))**(3.0_dl * (this%w_knot(max_num_of_bins) - (-1.0_dl)))
                    grho_de = grho_de_today * faci
                end if

            case(4)
                ! Flexknots
                ! Appendix A2 of Ormondroyd et al. 2025, 2503.08658v2
                if (size(this%a_flexknot) == 1) then
                    ! n=1 flexknot = constant-w model
                    grho_de = grho_de_today * &
                              a**(-3._dl*(1._dl + this%w_flexknot(1)))
                else
                    integral = 0._dl
                    do i = 1,size((this%a_flexknot))-1
                        a_lower = max(a,this%a_flexknot(i))
                        a_upper = this%a_flexknot(i+1)
                        
                        if (a_upper > a_lower) then
                            m_i = (this%w_flexknot(i+1) - this%w_flexknot(i)) / &
                                  (this%a_flexknot(i+1) - this%a_flexknot(i))
                            c_i = this%w_flexknot(i) - m_i*this%a_flexknot(i)

                            integral = integral + &
                                       (1._dl+c_i)*log(a_upper/a_lower) + &
                                       m_i*(a_upper-a_lower)
                        end if
                    end do    
                    grho_de = grho_de_today * exp(3._dl*integral)
                end if    

            case(5)
                ! Chebyshev
                ! Calderon et al. 2024, arXiv:2405.04216v1
                ! Eq. (2.4)
                alpha = 2._dl / this%cheb_zmax
                ! 1 + w(z) = q0 + q1*z + q2*z^2 + q3*z^3

                q0 = 1._dl - this%cheb_c0 + this%cheb_c1 &
                           - this%cheb_c2 + this%cheb_c3

                q1 = -alpha * ( &
                                this%cheb_c1 &
                                - 4._dl*this%cheb_c2 &
                                + 9._dl*this%cheb_c3 )

                q2 = -alpha**2 * ( &
                                2._dl*this%cheb_c2 &
                                - 12._dl*this%cheb_c3 )

                q3 = -4._dl*alpha**3*this%cheb_c3

                if (z <= this%cheb_zmax) then
                    Ipoly = &
                        q0*log(1._dl + z) + &
                        q1*(z - log(1._dl + z)) + &
                        q2*(0.5_dl*z**2 - z + log(1._dl + z)) + &
                        q3*(z**3/3._dl - 0.5_dl*z**2 + z - &
                            log(1._dl + z))
                    grho_de = grho_de_today * exp(3._dl*Ipoly)
                else    
                    ! Integral from z=0 to z=zmax
                    Imax = &
                        q0*log(1._dl + this%cheb_zmax) + &
                        q1*(this%cheb_zmax - &
                            log(1._dl + this%cheb_zmax)) + &
                        q2*(0.5_dl*this%cheb_zmax**2 - &
                            this%cheb_zmax + &
                            log(1._dl + this%cheb_zmax)) + &
                        q3*(this%cheb_zmax**3/3._dl - &
                            0.5_dl*this%cheb_zmax**2 + &
                            this%cheb_zmax - &
                            log(1._dl + this%cheb_zmax))

                    B0 = 1._dl - ( &
                        this%cheb_c0 + &
                        this%cheb_c1 + &
                        this%cheb_c2 + &
                        this%cheb_c3 )   

                    B1 = -2._dl*(1._dl + this%cheb_zmax) / &
                        this%cheb_zmax * &
                        (this%cheb_c1 + &
                        4._dl*this%cheb_c2 + &
                        9._dl*this%cheb_c3)     
                    
                    u = log((1._dl + z) / &
                        (1._dl + this%cheb_zmax))
            
                    Itrans = &
                        B0 * sqrt(acos(-1._dl)) * &
                        this%cheb_delta / 2._dl * &
                        erf(u/this%cheb_delta) + &
                        B1 * this%cheb_delta**2 / 2._dl * &
                        (1._dl - &
                         exp(-u**2/this%cheb_delta**2))
                         
                    grho_de = grho_de_today * &
                        exp(3._dl*(Imax + Itrans))
                endif    

            case default
                stop "Invalid DEmodel"
        end select
    end function TLateDE_grho_de

    subroutine TLateDE_Init(this, State)
        use classes
        use results
        class(TLateDE), intent(inout) :: this
        class(TCAMBdata), intent(in), target :: State

        select type (State)
            type is (CAMBdata)
            grho_de_today = State%grhov
        end select      
    end subroutine TLateDE_Init

    subroutine TLateDE_ReadParams(this, Ini)
        use IniObjects
        use FileUtils
        class(TLateDE) :: this
        class(TIniFile), intent(in) :: Ini
        this%DEmodel = Ini%Read_Int('DEmodel', 1)
        this%w0 = Ini%Read_Double('w0', 0.0_dl)
    end subroutine TLateDE_ReadParams
    
    subroutine TLateDE_PrintFeedback(this, FeedbackLevel)
        class(TLateDE) :: this
        integer, intent(in) :: FeedbackLevel

        if (FeedbackLevel >0) write(*,'("(w0, wa) = (", f8.5,", ", f8.5, ")")') !&
        ! &   this%w_lam, this%wa
    end subroutine TLateDE_PrintFeedback

    subroutine TLateDE_Effective_w_wa(this, w, wa)
        class(TLateDE), intent(inout) :: this
        real(dl), intent(out) :: w, wa

        select case (this%DEmodel)
            case(1) 
                ! constant w
                w = this%w0
                wa = 0._dl
            case(2)
                ! CPL    
                w = this%w0
                wa = this%w1
            case(3)
                ! No unique CPL equivalent for binned w(z)
                stop "[TLateDE_Effective_w_wa] Effective (w,wa) not defined for binned w(z)"
    
            case(4)
                ! No unique CPL equivalent for flexknots
                stop "[TLateDE_Effective_w_wa] Effective (w,wa) not defined for flexknots"
            
            case(5)
                stop "[TLateDE_Effective_w_wa] Effective (w,wa) not defined for Chebyshev w(z)"
        end select
    end subroutine TLateDE_Effective_w_wa

    subroutine TLateDE_SelfPointer(cptr,P)
        use iso_c_binding
        Type(c_ptr) :: cptr
        Type (TLateDE), pointer :: PType
        class (TPythonInterfacedClass), pointer :: P

        call c_f_pointer(cptr, PType)
        P => PType
    end subroutine TLateDE_SelfPointer

    subroutine TLateDE_density(this, grhov, a, grhov_t, w)
        ! Get grhov_t = 8*pi*G*rho_de*a**2 and (optionally) equation of state at scale factor a
        ! DHFS Note: translate the parameterization-specific 
        ! function into the quantities CAMB expects at a given scale factor
        class(TLateDE), intent(inout) :: this
        real(dl), intent(in) :: grhov, a
        real(dl), intent(out) :: grhov_t
        real(dl), optional, intent(out) :: w

        if (a > 1e-10) then
            grhov_t = this%grho_de(a) * a**2
        else
            grhov_t = 0
        end if
        if (present(w)) then
            w = this%w_de(a)
        end if
    end subroutine TLateDE_density

end module LateDE
