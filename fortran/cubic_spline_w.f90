module CubicSplineW

    use constants, only: dl
    
    implicit none
    private
    
    public :: SplineWValue
    public :: SplineWIntegral
    
    contains
    
    
    !===============================================================================
    ! Cubic-spline second derivatives
    !
    ! Boundary conditions:
    !
    !   lower boundary:  w''(z_1) = 0
    !
    !   upper boundary:  w'(z_N)  = 0
    !
    ! The upper boundary is chosen so that the spline joins smoothly onto
    ! w = -1 for z > z_N, provided w(z_N) = -1.
    !===============================================================================
    
    subroutine SplineSecondDerivatives(znode, wnode, y2)
    
        real(dl), intent(in)  :: znode(:)
        real(dl), intent(in)  :: wnode(:)
        real(dl), intent(out) :: y2(:)
    
        real(dl), allocatable :: work(:)
    
        real(dl) :: sig, p
        real(dl) :: qn, un
        real(dl) :: h
    
        integer :: i, k, n
    
        n = size(znode)
    
        if (size(wnode) /= n .or. size(y2) /= n) then
            error stop "[CubicSplineW] inconsistent array sizes"
        end if
    
        if (n < 3) then
            error stop "[CubicSplineW] at least 3 spline nodes are required"
        end if
    
        do i = 1, n-1
            if (znode(i+1) <= znode(i)) then
                error stop "[CubicSplineW] z nodes must be strictly increasing"
            end if
        end do
    
        allocate(work(n))
    
        y2   = 0._dl
        work = 0._dl
    
    
        !===========================================================================
        ! Lower boundary:
        !
        ! w''(z_1) = 0
        !===========================================================================
    
        y2(1)   = 0._dl
        work(1) = 0._dl
    
    
        !===========================================================================
        ! Interior equations
        !===========================================================================
    
        do i = 2, n-1
    
            sig = (znode(i) - znode(i-1)) / &
                  (znode(i+1) - znode(i-1))
    
            p = sig*y2(i-1) + 2._dl
    
            y2(i) = (sig - 1._dl) / p
    
            work(i) = ( &
                6._dl * ( &
                    (wnode(i+1)-wnode(i)) / &
                    (znode(i+1)-znode(i)) - &
                    (wnode(i)-wnode(i-1)) / &
                    (znode(i)-znode(i-1)) &
                ) / &
                (znode(i+1)-znode(i-1)) - &
                sig*work(i-1) &
            ) / p
    
        end do
    
    
        !===========================================================================
        ! Upper clamped boundary:
        !
        ! w'(z_N) = 0
        !
        ! Standard cubic-spline clamped boundary equation.
        !===========================================================================
    
        h = znode(n) - znode(n-1)
    
        qn = 0.5_dl
    
        un = (3._dl/h) * ( &
             0._dl - &
             (wnode(n)-wnode(n-1))/h )
    
        y2(n) = (un - qn*work(n-1)) / &
                (qn*y2(n-1) + 1._dl)
    
    
        !===========================================================================
        ! Back substitution
        !===========================================================================
    
        do k = n-1, 1, -1
            y2(k) = y2(k)*y2(k+1) + work(k)
        end do
    
        deallocate(work)
    
    end subroutine SplineSecondDerivatives
    
    
    
    !===============================================================================
    ! Cubic coefficients on one spline interval
    !
    ! On:
    !
    !   z0 <= z <= z1
    !
    ! use:
    !
    !   w(z) =
    !       c0
    !     + c1*(z-z0)
    !     + c2*(z-z0)^2
    !     + c3*(z-z0)^3
    !===============================================================================
    
    subroutine SplineCoefficients( &
            z0, z1, w0, w1, M0, M1, &
            c0, c1, c2, c3)
    
        real(dl), intent(in) :: z0, z1
        real(dl), intent(in) :: w0, w1
        real(dl), intent(in) :: M0, M1
    
        real(dl), intent(out) :: c0, c1, c2, c3
    
        real(dl) :: h
    
        h = z1-z0
    
        c0 = w0
    
        c1 = (w1-w0)/h - &
             h*(2._dl*M0 + M1)/6._dl
    
        c2 = M0/2._dl
    
        c3 = (M1-M0)/(6._dl*h)
    
    end subroutine SplineCoefficients
    
    
    
    !===============================================================================
    ! Evaluate cubic-spline w(z)
    !===============================================================================
    
    function SplineWValue(z, znode, wnode) result(w)
    
        real(dl), intent(in) :: z
        real(dl), intent(in) :: znode(:)
        real(dl), intent(in) :: wnode(:)
    
        real(dl) :: w
    
        real(dl), allocatable :: y2(:)
    
        real(dl) :: c0, c1, c2, c3
        real(dl) :: dz
    
        integer :: i, n
    
        n = size(znode)
    
        if (size(wnode) /= n) then
            error stop "[CubicSplineW] znode and wnode sizes differ"
        end if
    
        allocate(y2(n))
    
        call SplineSecondDerivatives( &
            znode, &
            wnode, &
            y2)
    
    
        ! Lower endpoint
        if (z <= znode(1)) then
    
            w = wnode(1)
    
            deallocate(y2)
            return
    
        end if
    
    
        ! Upper endpoint
        if (z >= znode(n)) then
    
            w = wnode(n)
    
            deallocate(y2)
            return
    
        end if
    
    
        ! Find interval
        do i = 1, n-1
    
            if (z <= znode(i+1)) then
    
                call SplineCoefficients( &
                    znode(i), &
                    znode(i+1), &
                    wnode(i), &
                    wnode(i+1), &
                    y2(i), &
                    y2(i+1), &
                    c0, c1, c2, c3)
    
                dz = z-znode(i)
    
                w = c0 + &
                    c1*dz + &
                    c2*dz**2 + &
                    c3*dz**3
    
                deallocate(y2)
                return
    
            end if
    
        end do
    
        deallocate(y2)
    
        error stop "[CubicSplineW] spline interval not found"
    
    end function SplineWValue
    
    
    
    !===============================================================================
    ! Primitive required for:
    !
    !   integral [1+w(z)]/(1+z) dz
    !
    ! when:
    !
    !   1+w(z) = q0 + q1*z + q2*z^2 + q3*z^3
    !===============================================================================
    
    function CubicIntegralPrimitive( &
            z, q0, q1, q2, q3) result(F)
    
        real(dl), intent(in) :: z
        real(dl), intent(in) :: q0, q1, q2, q3
    
        real(dl) :: F
        real(dl) :: log1pz
    
        log1pz = log(1._dl+z)
    
        F = &
            q0*log1pz + &
            q1*(z-log1pz) + &
            q2*(0.5_dl*z**2-z+log1pz) + &
            q3*( &
                z**3/3._dl - &
                0.5_dl*z**2 + &
                z - &
                log1pz)
    
    end function CubicIntegralPrimitive
    
    
    
    !===============================================================================
    ! Dark-energy continuity integral within the spline region:
    !
    !   I(z) =
    !
    !   integral_0^z [1+w(z')]/[1+z'] dz'
    !
    ! The integration is analytic interval by interval.
    !===============================================================================
    
    function SplineWIntegral(z_upper, znode, wnode) result(integral)
    
        real(dl), intent(in) :: z_upper
        real(dl), intent(in) :: znode(:)
        real(dl), intent(in) :: wnode(:)
    
        real(dl) :: integral
    
        real(dl), allocatable :: y2(:)
    
        real(dl) :: c0, c1, c2, c3
        real(dl) :: q0, q1, q2, q3
    
        real(dl) :: zi
        real(dl) :: zlo, zhi
        real(dl) :: Flo, Fhi
    
        integer :: i, n
    
        integral = 0._dl
    
        if (z_upper <= 0._dl) return
    
        n = size(znode)
    
        if (size(wnode) /= n) then
            error stop "[CubicSplineW] znode and wnode sizes differ"
        end if
    
        allocate(y2(n))
    
        call SplineSecondDerivatives( &
            znode, &
            wnode, &
            y2)
    
    
        do i = 1, n-1
    
            zlo = znode(i)
    
            zhi = min( &
                z_upper, &
                znode(i+1))
    
            if (zhi <= zlo) cycle
    
    
            call SplineCoefficients( &
                znode(i), &
                znode(i+1), &
                wnode(i), &
                wnode(i+1), &
                y2(i), &
                y2(i+1), &
                c0, c1, c2, c3)
    
    
            zi = znode(i)
    
    
            !-----------------------------------------------------------------------
            ! Rewrite:
            !
            !   w(z) =
            !
            !     c0
            !   + c1*(z-zi)
            !   + c2*(z-zi)^2
            !   + c3*(z-zi)^3
            !
            ! as:
            !
            !   1+w(z) =
            !
            !     q0
            !   + q1*z
            !   + q2*z^2
            !   + q3*z^3
            !-----------------------------------------------------------------------
    
            q0 = 1._dl + &
                 c0 - &
                 c1*zi + &
                 c2*zi**2 - &
                 c3*zi**3
    
            q1 = &
                 c1 - &
                 2._dl*c2*zi + &
                 3._dl*c3*zi**2
    
            q2 = c2 - 3._dl*c3*zi
    
            q3 = c3
    
    
            Flo = CubicIntegralPrimitive( &
                zlo, q0, q1, q2, q3)
    
            Fhi = CubicIntegralPrimitive( &
                zhi, q0, q1, q2, q3)
    
            integral = integral + Fhi-Flo
    
    
            if (zhi >= z_upper) exit
    
        end do
    
        deallocate(y2)
    
    end function SplineWIntegral
    
    
    end module CubicSplineW