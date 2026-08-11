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
        real(dl) :: w0
        real(dl) :: w1
        real(dl), allocatable :: z_knot(:)
        real(dl), allocatable :: w_knot(:)
        contains
        procedure :: ReadParams => TLateDE_ReadParams
        procedure :: Init => TLateDE_Init
        procedure :: PrintFeedback => TLateDE_PrintFeedback
        procedure :: w_de => TLateDE_w_de
        procedure :: grho_de => TLateDE_grho_de
        procedure :: Effective_w_wa => TLateDE_Effective_w_wa   !VM: wont be called with CASARINI (our mod)         
        procedure, nopass :: SelfPointer => TLateDE_SelfPointer
        procedure :: BackgroundDensityAndPressure => TLateDE_density ! DHFS: Do I Need This ? If yes why, if not why
    end type TLateDE

    public TLateDE

    contains

    function TLateDE_w_de(this, a) result(w_de)
        class(TLateDE) :: this
        real(dl), intent(in) :: a    
        real(dl) :: w_de, z
        integer  :: i

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
            
            case default        
                stop "Invalid DEmodel"   
        end select
    end function TLateDE_w_de

    function TLateDE_grho_de(this, a) result(grho_de)
        ! Returns 8*pi*G * rho_de, no factor of a^4
        class(TLateDE) :: this
        real(dl), intent(in) :: a
        real(dl) :: grho_de, z
        real(dl) :: faci, temp
        integer  :: max_num_of_bins
        integer  :: i, j

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

        if (this%DEmodel == 2) then
            !'w0wa'
            w  = this%w0
            wa = this%w1
        else if (this%DEmodel /= 2) then
            w  = this%w0
            wa = 0
        else
            stop "[Late Fluid DE @TLateDE_Effective_w_wa] Invalid Dark Energy Model (TLateDE_Effective_w_wa)"
        endif
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
