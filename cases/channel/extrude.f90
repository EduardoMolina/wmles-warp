!
!
! Module definitions
!
!

module kind_values
  !
  integer, parameter :: int_kind = kind(1)
  integer, parameter :: flt_kind = kind(1.0d0)
  !
end module kind_values

module global_constants
  !
  use kind_values
  !
  real(kind=flt_kind), parameter :: zero = 0.0_flt_kind, one = 1.0_flt_kind
  !
end module global_constants

program extrude
  !
  use kind_values
  use global_constants
  !
  integer(kind=int_kind) :: p3d_unit_num = 1
  integer(kind=int_kind) :: out_unit_num = 2
  !
  integer(kind=int_kind) :: nb
  !
  integer(kind=int_kind), allocatable, dimension(:) :: ni,nj,nk
  !
  real(kind=flt_kind), allocatable, dimension(:,:,:) :: x,y,z
  !
  character(len=132) :: p3d_grid_file
  character(len=132) :: out_grid_file
  !
  logical file_exists
  !
  ! Prompt for the 2-D PLOT3D grid file to be extruded
  !
  write(*,*)
  write(*,*) 'Select the 2-D (PLOT3D format) grid file to be extruded:'
  write(*,*)
  write(*,*) 'Channel grids for WMLES that invoke 1-D wall-models'
  write(*,*) '1)  65x33 LES channel grid for use with 1-D wall-models (dx/dz=2)'
  write(*,*) '2)  97x49 LES channel grid for use with 1-D wall-models (dx/dz=2)'
  write(*,*) '3) 129x65 LES channel grid for use with 1-D wall-models (dx/dz=2)'
  write(*,*) '4) 193x97 LES channel grid for use with 1-D wall-models (dx/dz=2)'
  write(*,*) '5) 129x33 LES channel grid for use with 1-D wall-models (dx/dz=1)'
  write(*,*) '6) 193x49 LES channel grid for use with 1-D wall-models (dx/dz=1)'
  write(*,*) '7) 257x65 LES channel grid for use with 1-D wall-models (dx/dz=1)'
  write(*,*) '8) 385x97 LES channel grid for use with 1-D wall-models (dx/dz=1)'
  write(*,*)
  write(*,*) 'Channel grids for WMLES that integrate to the wall (e.g. IDDES)'
  write(*,*) '9)   65x?? "complete" channel grid with y+ of 1.0 (dx/dz=2)'
  write(*,*) '10)  97x?? "complete" channel grid with y+ of 1.0 (dx/dz=2)'
  write(*,*) '11) 129x?? "complete" channel grid with y+ of 1.0 (dx/dz=2)'
  write(*,*) '12) 193x?? "complete" channel grid with y+ of 1.0 (dx/dz=2)'
  write(*,*) '13) 129x?? "complete" channel grid with y+ of 1.0 (dx/dz=1)'
  write(*,*) '14) 193x?? "complete" channel grid with y+ of 1.0 (dx/dz=1)'
  write(*,*) '15) 257x?? "complete" channel grid with y+ of 1.0 (dx/dz=1)'
  write(*,*) '16) 385x?? "complete" channel grid with y+ of 1.0 (dx/dz=1)'
  write(*,*)
  read (*,*) input_option
  !
  if (input_option < 1 .or. input_option > 16) then
    write(*,*)
    write(*,*) 'An invalid 2-D PLOT3D format grid file choice was selected'
    write(*,*)
    stop
  else if (input_option == 1) then
    p3d_grid_file = 'channel_ar2_1dwm_65x33.p3d'
  else if (input_option == 2) then
    p3d_grid_file = 'channel_ar2_1dwm_97x49.p3d'
  else if (input_option == 3) then
    p3d_grid_file = 'channel_ar2_1dwm_129x65.p3d'
  else if (input_option == 4) then
    p3d_grid_file = 'channel_ar2_1dwm_193x97.p3d'
  else if (input_option == 5) then
    p3d_grid_file = 'channel_ar1_1dwm_129x33.p3d'
  else if (input_option == 6) then
    p3d_grid_file = 'channel_ar1_1dwm_193x49.p3d'
  else if (input_option == 7) then
    p3d_grid_file = 'channel_ar1_1dwm_257x65.p3d'
  else if (input_option == 8) then
    p3d_grid_file = 'channel_ar1_1dwm_385x97.p3d'
  end if
  !
  ! Prompt for the name of the 3-D output grid file with logic to ensure that
  ! the number of characters in the file name does not exceed what is expected
  !
  write(*,*)
  write(*,*) 'Enter the file name of the extruded 3-D grid to be output'
  !
  len_str = 0
  do
    len_str = len_str + 1
    if (len_str <= len(out_grid_file)) then
      read (*,'(a1)',advance='no',iostat=ierr) out_grid_file(len_str:len_str)
      if (ierr /= 0) exit
    else
      read (*,'(a1)',advance='no',iostat=ierr) out_grid_file(1:1)
      if (ierr /= 0) then
        write(*,*)
        write(*,*) 'The grid file name is too long:'
        write(*,*) 'Allowable string length = ', len(out_grid_file)
        write(*,*) 'String length (trimmed) = ', len_str - 1
        write(*,*)
        stop
      end if
    end if
  end do
  !
  out_grid_file = adjustl(out_grid_file(1:max(len_str-1,1)))
  !
  if (len_trim(out_grid_file) == 0) then
    write(*,*)
    write(*,*) 'The grid file name appears to be a "null"'
    write(*,*)
    stop
  end if
  !
  ! Ensure that output file name is different than the 2-D PLOT3D format file
  !
  if (trim(out_grid_file) == trim(p3d_grid_file)) then
    write(*,*)
    write(*,*) 'The input and output grid files must have different names'
    write(*,*)
    stop
  end if
  !
  ! Check for the existence of the 2-D PLOT3D format grid file
  !
  inquire (file=p3d_grid_file,exist=file_exists)
  if (.not. file_exists) then
    write(*,*)
    write(*,*) 'The PLOT3D format grid file was not found'
    write(*,*) trim(p3d_grid_file)
    write(*,*)
    stop
  end if
  !
  ! Open the 2-D PLOT3D format grid file (64-bit unformatted file)
  !
  open (p3d_unit_num, file=p3d_grid_file, form='unformatted', status='old')
  !
  ! Read in the number of structured grid blocks
  !
  read(p3d_unit_num) nb
  !
  if (nb /= 1) then
    write(*,*)
    write(*,*) 'This utility expects a single block to be present'
    write(*,*) 'nb = ', nb
    write(*,*)
    stop
  end if
  !
  ! Allocate memory for the grid dimensions for each block
  !
  allocate (ni(1:nb),stat=istat)
  if (istat /= 0) then
    write(*,*)
    write(*,*) 'MEMORY UNAVAILABLE FOR ALLOCATION OF ni'
    write(*,*)
    stop
  end if
  allocate (nj(1:nb),stat=istat)
  if (istat /= 0) then
    write(*,*)
    write(*,*) 'MEMORY UNAVAILABLE FOR ALLOCATION OF nj'
    write(*,*)
    stop
  end if
  allocate (nk(1:nb),stat=istat)
  if (istat /= 0) then
    write(*,*)
    write(*,*) 'MEMORY UNAVAILABLE FOR ALLOCATION OF nk'
    write(*,*)
    stop
  end if
  !
  ! Read in the grid dimensions for each block
  !
  read(p3d_unit_num) (ni(n),nj(n),nk(n),n=1,nb)
  !
  if (nk(1) /= 1) then
    write(*,*)
    write(*,*) 'This utility expects a single 2-D plane of grid points'
    write(*,*) 'nk = ', nk(1)
    write(*,*)
    stop
  end if
  !
  ! Loop through the grid blocks and read in the x,y,z coordinates
  !
  do n = 1,nb
    !
    ! Define the grid size in the extruded Z-direction
    !
    if (index(p3d_grid_file,"channel_") /= 0) then
      !
      ! Define the extruded channel grid size when dx/dz = 1
      ! NOTE: length = 12.8 channel heights, depth = 4.8 channel heights
      !
      nk(n) = nint(real(ni(1)-1,flt_kind) * 4.8_flt_kind / 12.8_flt_kind) + 1
      !
    end if
    !
    ! Correct the grid size in the extruded Z-direction when dx/dz = 2
    !
    if (index(p3d_grid_file,"_ar2_") /= 0) then
      nk(n) = 2 * (nk(n) - 1) + 1
    end if
    !
    ! Allocate memory for the x,y,z coordinates of the current block
    !
    allocate (x(1:ni(n),1:nj(n),1:nk(n)),stat=istat)
    if (istat /= 0) then
      write(*,*)
      write(*,*) 'MEMORY UNAVAILABLE FOR ALLOCATION OF x'
      write(*,*)
      stop
    end if
    allocate (y(1:ni(n),1:nj(n),1:nk(n)),stat=istat)
    if (istat /= 0) then
      write(*,*)
      write(*,*) 'MEMORY UNAVAILABLE FOR ALLOCATION OF y'
      write(*,*)
      stop
    end if
    allocate (z(1:ni(n),1:nj(n),1:nk(n)),stat=istat)
    if (istat /= 0) then
      write(*,*)
      write(*,*) 'MEMORY UNAVAILABLE FOR ALLOCATION OF z'
      write(*,*)
      stop
    end if
    !
    ! Read in the 2-D x,y,z coordinates of the current block
    !
    read(p3d_unit_num) (((x(i,j,k),i=1,ni(n)),j=1,nj(n)),k=1,1), &
                       (((y(i,j,k),i=1,ni(n)),j=1,nj(n)),k=1,1), &
                       (((z(i,j,k),i=1,ni(n)),j=1,nj(n)),k=1,1)
    !
    if (index(p3d_grid_file,"channel_") /= 0) then
      !
      ! Check if the channel flow grid length is 12.8 channel heights (~ 4*pi)
      !
      if (abs(x(ni(n),1,1)-x(1,1,1))-12.8_flt_kind > sqrt(epsilon(zero))) then
        write(*,*)
        write(*,*) 'A channel length of 12.8 channel heights was expected'
        write(*,*) 'Channel length = ', abs(x(ni(n),1,1)-x(1,1,1))
        write(*,*)
        stop
      end if
      !
      ! Extrude the channel grid 4.8 channel heights (~ 1.5*pi)
      !
      dz = 4.8_flt_kind / real(nk(n)-1,flt_kind)
      !
    end if
    !
    ! Extrude the grid in the Z-direction
    !
    do k=2,nk(n)
      do j=1,nj(n)
        do i=1,ni(n)
          x(i,j,k) = x(i,j,1)
          y(i,j,k) = y(i,j,1)
          z(i,j,k) = z(i,j,1) + real(k-1,flt_kind) * dz
        end do
      end do
    end do
    !
    ! Output the x,y,z coordinates of the current block
    !
    call output_p3d (out_grid_file, out_unit_num, nb, n, ni, nj, nk, x, y, z)
    !
    deallocate (x,y,z)
    !
  end do
  !
  deallocate (ni,nj,nk)
  !
  close(p3d_unit_num)
  !
end program extrude

subroutine output_p3d (out_grid_file, out_unit_num, nb, n, ni, nj, nk, x, y, z)
  !
  use kind_values
  !
  character(len=*), intent(in) :: out_grid_file
  !
  integer(kind=int_kind), intent(in) :: out_unit_num
  !
  integer(kind=int_kind), intent(in) :: nb
  !
  integer(kind=int_kind), intent(in) :: n
  !
  integer(kind=int_kind), dimension(nb), intent(in) :: ni,nj,nk
  !
  real(kind=flt_kind), dimension(ni(n),nj(n),nk(n)) :: x,y,z
  !
  ! Open the output grid file and write the PLOT3D header information
  ! NOTE: This should only be performed when the 1st block number is processed
  !
  if (n == 1) then
    !
    open (out_unit_num, file=out_grid_file, form='unformatted')
    !
    write(out_unit_num) nb
    !
    write(out_unit_num) (ni(nn),nj(nn),nk(nn),nn=1,nb)
    !
  end if
  !
  ! Write out the x,y,z coordinates of the current block
  !
  write(out_unit_num) (((x(i,j,k),i=1,ni(n)),j=1,nj(n)),k=1,nk(n)), &
                      (((y(i,j,k),i=1,ni(n)),j=1,nj(n)),k=1,nk(n)), &
                      (((z(i,j,k),i=1,ni(n)),j=1,nj(n)),k=1,nk(n))
  !
  ! Close the output grid file when the last block number is processed
  !
  if (n == nb) then
    close(out_unit_num)
  end if
  !
end subroutine output_p3d
