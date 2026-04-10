c   SOLVE THE VISCOPLASTICITY RESPONSE OF A 2D POLYCRYSTAL USING THE FFT ALGORITHM
c   Written by Ricardo A. Lebensohn (lebenso@lanl.gov)  
c   For an explanaition of the method see:
c 	- Lebensohn (2001) Acta Mater. 2723-2737
c	- Lebensohn et al. (2008) Acta Mater. 
c 
c   This version has been adapted to works with ELLE platform (see www.materialsknowledge.org/elle).
c   version 9.01:
c   	- Compatible with non-orthogonal meshes of  real and frecuency space
c	- Calculates the dislocation densities field using the gradient of the strain rate field
c       - Output files:      
c   The code has been succesfull compiled with g77, gfortran or ifortran.   
c   input files: mineral.sx, ppc.in, ppc.dim and temp.out.
c   output files:all.out, err.out,conv.out,tex.out,unodeang.out,unodexyz.out,temp-FFT.out (last 4-files are required by ELLE)    
c   Additional information albert griera (agriera@agriera.info) 2009

      program fft
c
      INCLUDE 'ppc.dim'
c
      integer p,q
c      character*1 dum
c
      dimension data(2*npts1*npts2*npts3),nn(3),nn2(2)
      dimension xk(3),xk2(3),xk3(3)
      dimension a(4,4),g1(3,3,3,3)
c
      dimension delta(5,npts1,npts2,npts3)
      dimension deltaim(5,npts1,npts2,npts3)
      dimension aux5(5),aux33(3,3)
c      dimension auxd(3,3),auxs(3,3)
      dimension aux55(5,5),aux3333(3,3,3,3)
      dimension minv1(4),minv2(4)
cc      dimension chi(3,3)
c
      dimension d5(5),d5im(5),d33(3,3),d33im(3,3),fbar(3,3)
c
      dimension velmax(3),vel_alb(3)
cu
      dimension g2(3,3,3),velmacro(3,npts1,npts2,npts3)
      dimension velim(3,npts1,npts2,npts3)
      dimension vel(3,npts1,npts2,npts3)
      dimension vel_data(3,npts1,npts2,npts3)
      dimension tlocal(3,npts1,npts2,npts3)
cw      dimension tmacro1(3,npts1,npts2,npts3)
      dimension tmacro2(3,npts1,npts2,npts3)
cu
      dimension sg5(5),sgx(3,3)
      dimension dg5(5),dgx(3,3)
calb
      real length_alb(2),odelt(3),nn_d(2,2,2)
      real d_alb(2,npts1,npts2)
      real n_d(2,2,npts1,npts2)
      real nn_inv(npts1,npts2,npts3)
      real rot(3,3,npts1,npts2,npts3)
cw    non-orthogonal frequency space
      dimension yf1(2),yf2(2),yr1(2),yr2(2)
cw

c
      pi=4.*atan(1.)
c
cc      open(40,file='str_str.out',status='unknown')
cc      open(35,file='activ.out',status='unknown')
      open(21,file='err.out',status='unknown')
cc      open(20,file='ppc.out',status='unknown')
cc      open(22,file='taugam.out',status='unknown')
      open(23,file='all.out',status='unknown')
cc      open(24,file='tex.out',status='unknown')
      open(25,file='conv.out',status='unknown')
c
      write(21,*) 
     #              'IT    ERRD       ERRS       DVM     SVM    WD/DVM
     # EFFMU13   SINH    DINH    ACT ...'
c
cc      write(21,*) 
cc     #'IT    ERRD       ERRS      EQERR       SEQ     DEQ     EFFMU13  
cc     # EFFMU23   SINH    DINH    ACT ...'
cc
cc      write(21,*)'IT    ERRD       ERRS      EQERR    S13/2D13  SINH   
cc     #DINH    ACT...'

      nn(1)=npts1
      nn(2)=npts2
      nn(3)=npts3
c
      nn2(1)=npts1
      nn2(2)=npts2
c
      prodnn=float(nn(1)*nn(2)*nn(3))
      wgt=1./prodnn
c
cc      delt(1)=1.
cc      delt(2)=1.
ccc      delt(2)=sqrt(3.)/2.
cc      delt(3)=1.
cc      delt(3)=0.5
c
cc      deltvol3=(delt(1)*delt(2)*delt(3))**(1./3.)
c
      ur0=0
      open(ur0,file='ppc.in',status='old')
calb
      open(98,file='temp.out',status='old')
      UR1=1      ! FILECRYS
      UR2=2      ! FILETEXT
c
      call vpsc_input
c
c      tdot=0.1
c

      write(*,*) 'fshear = ',fshear
      dvm=0.
      do i=1,3
      do j=1,3
      dvm=dvm+dsim(i,j)**2
      enddo
      enddo
      dvm=sqrt(2./3.*dvm)
c
      do 777 i=1,npts1
      do 777 j=1,npts2
      do 777 k=1,npts3
c
c     INITIAL TLOCAL
c
calb Add the displacement of tlocal?? for non-ortho 
calb      tlocal(1,i,j,k)=(i-1)*delt(1)
      tlocal(1,i,j,k)=(i-1)*delt(1)+(j-1)*fshear
      tlocal(2,i,j,k)=(j-1)*delt(2)
      tlocal(3,i,j,k)=(k-1)*delt(3)
c
cw      tmacro1(1,i,j,k)=(i-1)*delt(1)
cw      tmacro1(2,i,j,k)=(j-1)*delt(2)
cw      tmacro1(3,i,j,k)=(k-1)*delt(3)
c
c     INITIALIZE D~
c
      do ii=1,5
      dtilde(ii,i,j,k)=0.
      enddo
c
777   continue
c
      do 3000 imicro=1,nsteps
c
      write(*,*) 'STEP = ',imicro
      if(nsteps.ne.1) write(21,*) 'STEP = ',imicro
      write(25,*) 'STEP = ',imicro
c
      call update_schmid
c
      IF(IMICRO.EQ.1) THEN
c
c     INITIAL GUESS FOR LOCAL FIELDS
c
      IF(IRECOVER.EQ.0) THEN
c
      write(*,*) 'CALCULATING INITIAL TAYLOR GUESS'
c
c     SACHS ESTIMATE OF INITIAL STRESS
c
      do 55 k=1,npts3
      do 55 j=1,npts2
      do 55 i=1,npts1
c
        TAUMAX=0.
        DO IS=1,NSYST(1)
          PROD=0.
          DO JW=1,5
            PROD=PROD+DBAR(JW)*SCH(JW,IS,i,j,k)
          ENDDO
          ISIGN=1
          IF(PROD.LT.0.) ISIGN=2
          PROD=PROD/CRSS(IS,ISIGN,i,j,k)
          IF(ABS(PROD).GT.TAUMAX) TAUMAX=ABS(PROD)
        ENDDO
        DO IW=1,5
          SG(IW,i,j,k)=DBAR(IW)/TAUMAX
        ENDDO
c
55    continue
c
      call visc_fft_initial
      call get_gamdot
c
      ELSE
c
      write(*,*) 'READING INITIAL GUESS'
C
      do i=1,5
      read(50) (xlsec(i,j),j=1,5)
      enddo
c
      do i=1,5
      do j=1,5
      xlsec(i,j)=xlfac1*xlsec(i,j)
      enddo
      enddo
c
      do kk=1,npts3
      do jj=1,npts2
      do ii=1,npts1
      read(50) (sg(i,ii,jj,kk),i=1,5)
      enddo
      enddo      
      enddo
c
      call get_gamdot
c
      ENDIF
c
      call chg_basis(aux5,aux33,xlsec,xlsec33,3)
c
      call second_order(0)
c
      ENDIF
C
cw
calb  define the non-orthogonal real space 

      yr1(1)=1.
      yr1(2)=0.

      yr2(1)=1.*fshear
      yr2(2)=1.

calb  Define the reciprocal space 

      yf1(1)=yr2(2)
      yf1(2)=-yr2(1)

      yf2(1)=-yr1(2)
      yf2(2)=yr1(1)

cw      yf1(1)=1.
cw      yf1(2)=-1.
cw      yf2(1)=0.
cw      yf2(2)=1.
cw

      iter=0
c
c      err=0.011
c      do while(iter.lt.itmax.and.err.gt.error)
c
      err2mod=2*error
      do while(iter.lt.itmax.and.err2mod.gt.error)
      iter=iter+1
c
cw      write(*,*)
      write(*,*)'ITER = ',iter
c
      write(25,*)'ITER = ',iter
c
cw      write(*,*)
cw      write(*,*) 'VP CALCULATION OF LOCAL STRESSES'
cw      write(*,*)
c
c      write(31,*)'ITER = ',iter
c      write(32,*)'ITER = ',iter
c      write(33,*)'ITER = ',iter

      write(*,*) 
     #'DIRECT FFT OF POLARIZATION AND LAGRANGE MULTIPLIER FIELDS'
cw      write(*,*)
c
      do 300 ii=1,5
c
      k1=0
      do 5 k=1,npts3
c
cw      write(*,'(1H+,a,i2,2(a,i4))') 
cw     #    'DELTA - COMPONENT',ii,'  -   Z = ',k,'   OUT OF ',npts
c
      do 5 j=1,npts2
      do 5 i=1,npts1
cc
cc      delta(ii,i,j,k)=sg(ii,i,j,k)-sbar(ii)
cc
      delta(ii,i,j,k)=sg(ii,i,j,k)
c
      do jj=1,5
      delta(ii,i,j,k)=delta(ii,i,j,k)-xlsec(ii,jj)*dtilde(jj,i,j,k)
      enddo
c
      k1=k1+1
      data(k1)=delta(ii,i,j,k)
      k1=k1+1
      data(k1)=0.
5     continue 
c
      if(npts3.gt.1) then
      call fourn(data,nn,3,1)
      else
      call fourn(data,nn2,2,1)
      endif
c
      k1=0
      do 6 kzz=1,npts3
      do 6 kyy=1,npts2
      do 6 kxx=1,npts1
c
      k1=k1+1
      delta(ii,kxx,kyy,kzz)=data(k1)
c
      k1=k1+1
      deltaim(ii,kxx,kyy,kzz)=data(k1)
c
6     continue
c
cc      k1=0
cc      do 58 k=1,npts3
ccc
cccw      write(*,'(1H+,a,i2,2(a,i4))') 
cccw     #    'SIGMA - COMPONENT',ii,'  -   Z = ',k,'   OUT OF ',npts
ccc
cc      do 58 j=1,npts2
cc      do 58 i=1,npts1
ccc
cc      k1=k1+1
cc      data(k1)=sg(ii,i,j,k)
cc      k1=k1+1
cc      data(k1)=0.
cc58     continue
ccc
cc      if(npts3.gt.1) then
cc      call fourn(data,nn,3,1)
cc      else
cc      call fourn(data,nn2,2,1)
cc      endif
ccc
cc      k1=0
cc      do 68 kzz=1,npts3
cc      do 68 kyy=1,npts2
cc      do 68 kxx=1,npts1
ccc
cc      k1=k1+1
cc      sgfft(ii,kxx,kyy,kzz)=data(k1)
cc      k1=k1+1
cc      sgfftim(ii,kxx,kyy,kzz)=data(k1)
ccc
cc68     continue
c
300   continue
c
cc      snormfft=0.
ccc      snormfftim=0.
cc      do i=1,5
cc      snormfft=snormfft+sgfft(i,1,1,1)**2
ccc      snormfftim=snormfftim+sgfftim(i,1,1,1)**2
cc      enddo
cc      snormfft=sqrt(snormfft)
ccc      snormfftim=sqrt(snormfftim)
ccc
ccc      write(*,*)
ccc      write(*,*)'REAL SNORM = ',snormfft
ccc      write(*,*)'IMAG SNORM = ',snormfftim
ccc      pause
c
      write(*,*) 'CALCULATING G^pq,ij : TAU^ij ...'
cw      write(*,*)
c
      do 1 kzz=1,npts3
c
c      if(kzz/1*1.eq.kzz) write(*,'(a,i4)') ' K= ',kzz
c 
cw      write(*,'(1H+,a,i4,a,i4)') ' Z = ',kzz,'   OUT OF ',npts
c
      do 1 kyy=1,npts2
      do 1 kxx=1,npts1
c
      do 46 i=1,5
      d5(i)=delta(i,kxx,kyy,kzz)
      d5im(i)=deltaim(i,kxx,kyy,kzz)
46    continue

      call chg_basis(d5,d33,aux55,aux3333,1)
      call chg_basis(d5im,d33im,aux55,aux3333,1)
c
      if(kxx.le.npts1/2) kx=kxx-1
      if(kxx.gt.npts1/2) kx=kxx-npts1-1
c
      if(kyy.le.npts2/2) ky=kyy-1
      if(kyy.gt.npts2/2) ky=kyy-npts2-1
c
      if(kzz.le.npts3/2) kz=kzz-1
      if(kzz.gt.npts3/2) kz=kzz-npts3-1
c
calb      xk3(1)=kx/(delt(1)*nn(1))
calb      xk3(2)=ky/(delt(2)*nn(2))
      xk(1)=(kx*yf1(1))/(delt(1)*nn(1))+
     #     ky*yf2(1)/(delt(2)*nn(2))
      xk(2)=kx*yf1(2)/(delt(1)*nn(1))+
     #    ky*yf2(2)/(delt(2)*nn(2))
calb      write(*,*) xk(1),xk(2),xk3(1),xk3(2)
c
      if(npts3.gt.1) then
      xk(3)=kz/(delt(3)*nn(3))
      else
      xk(3)=0.
      endif
c
      xknorm=sqrt(xk(1)**2+xk(2)**2+xk(3)**2)
c
      if (xknorm.ne.0.) then
      do i=1,3
cbug      xk2(i)=xk(i)/(xknorm*xknorm)
      xk2(i)=xk(i)/(xknorm*xknorm*2.*pi)
      xk(i)=xk(i)/xknorm
      enddo
      endif

calb      write(*,*) ' xk ',xk(1),xk(2),xk2(1),xk2(2)
c

      do 2 i=1,3
      do 2 k=1,3
      a(i,k)=0.
      do 2 j=1,3
      do 2 l=1,3
      a(i,k)=a(i,k)+xlsec33(i,j,k,l)*xk(j)*xk(l)
2     continue
c
      do i=1,3
      a(4,i)=xk(i)
      a(i,4)=xk(i)
      enddo
c
      a(4,4)=0.
c
      call minv(a,4,det,minv1,minv2)
c
c      if(det.eq.0) then
c      write(*,*) kx,ky,kz,'  --> SINGULAR SYSTEM'
c      stop
c      pause
c      endif
c
      do 3 p=1,3
      do 3 q=1,3
      do 3 i=1,3
      do 3 j=1,3
      g1(p,q,i,j)=-a(p,i)*xk(q)*xk(j)
3     continue
c
      do 33 i=1,3
      do 33 j=1,3
      do 33 k=1,3
cbug      g2(i,j,k)=-xk2(j)*a(k,i)
      g2(i,j,k)=-xk2(j)*a(k,i)
33     continue
c
cc      do i=1,3
cc      do j=1,3
cc      chi(i,j)=xk(j)*a(4,i)
cc      enddo
cc      enddo
ccc
cc      press(kxx,kyy,kzz)=0.
cc      pressim(kxx,kyy,kzz)=0.
ccc
cc      if(kx.eq.0.and.ky.eq.0.and.kz.eq.0.) goto 104
ccc
cc      do i=1,3
cc      do j=1,3
cc      press(kxx,kyy,kzz)=
cc     #    press(kxx,kyy,kzz)+chi(i,j)*d33(i,j)
cc      pressim(kxx,kyy,kzz)=
cc     #    pressim(kxx,kyy,kzz)+chi(i,j)*d33im(i,j)
cc      enddo
cc      enddo
ccc
104   do 4 i=1,3
      do 4 j=1,3
c
      velgrad(i,j,kxx,kyy,kzz)=0.
      velgradim(i,j,kxx,kyy,kzz)=0.
c
      if(kx.eq.0.and.ky.eq.0.and.kz.eq.0.) goto 4
c
      do k=1,3
      do l=1,3
      velgrad(i,j,kxx,kyy,kzz)=
     #    velgrad(i,j,kxx,kyy,kzz)+g1(i,j,k,l)*d33(k,l)
      velgradim(i,j,kxx,kyy,kzz)=
     #    velgradim(i,j,kxx,kyy,kzz)+g1(i,j,k,l)*d33im(k,l)
      enddo
      enddo
c
4     continue
c
      do k=1,3
      velim(k,kxx,kyy,kzz)=0.
      vel(k,kxx,kyy,kzz)=0.
      enddo
c
      if(kx.eq.0.and.ky.eq.0.and.kz.eq.0.) goto 1
c
      do 47 k=1,3
      do 47 i=1,3
      do 47 j=1,3
      velim(k,kxx,kyy,kzz)=
     #    velim(k,kxx,kyy,kzz)+g2(i,j,k)*d33(i,j)
      vel(k,kxx,kyy,kzz)=vel(k,kxx,kyy,kzz)-g2(i,j,k)*d33im(i,j)

47    continue
c
1     continue
c
cc      call equilibrium(snormfft,snormfftim,nn,err2mod)
c
      write(*,*) 'INVERSE FFT TO GET STRAIN-RATE FLUCTUATION FIELD'
c
      do 51 m=1,3
      do 51 n=1,3
c
      k1=0
c
      do 50 k=1,npts3
      do 50 j=1,npts2
      do 50 i=1,npts1
c
      k1=k1+1
      data(k1)=velgrad(m,n,i,j,k)
c
      k1=k1+1
      data(k1)=velgradim(m,n,i,j,k)
c
50     continue
c
      if(npts3.gt.1) then
      call fourn(data,nn,3,-1)
      else
      call fourn(data,nn2,2,-1)
      endif
c
      do i=1,2*npts1*npts2*npts3
      data(i)=data(i)/prodnn
      enddo
c
      k1=0
      do 16 kzz=1,npts3
      do 16 kyy=1,npts2
      do 16 kxx=1,npts1
      k1=k1+1
c      write(*,*) 'REAL PART =',kxx,kyy,kzz,data(k1)

      velgrad(m,n,kxx,kyy,kzz)=data(k1)
      k1=k1+1
c      write(*,*) 'IMAGINARY PART =',kxx,kyy,kzz,data(k1)
c      pause
16    continue
c
51    continue
c
c     DTILDE=SYM(VELGRAD)
c
cc      err=0.
c
      do 167 kzz=1,npts3
      do 167 kyy=1,npts2
      do 167 kxx=1,npts1
c 
      do ii=1,3
      do jj=1,3
      aux33(ii,jj)=(velgrad(ii,jj,kxx,kyy,kzz)+
     #              velgrad(jj,ii,kxx,kyy,kzz))/2.
      enddo
      enddo
c
      call chg_basis(aux5,aux33,aux55,aux3333,2)
c
      do m=1,5
cc      err=err+abs(dtilde(m,kxx,kyy,kzz)-aux5(m))*wgt
      dtilde(m,kxx,kyy,kzz)=aux5(m)
      enddo
c
167   continue
c
       write(*,*) 'UPDATE LAGRANGE MULTIPLIER FIELD'
       call visc_fft
       call second_order(0)
c
       write(*,*) 'STRESS FIELD ERROR =',errs/svm
       write(*,*) 'STRAIN-RATE FIELD ERROR =',errd/dvm
       write(*,*) 'EFF MU 13 =',effmu13
c
c     STATISTIC ON ACTIVITIES
c
      call statactiv
c
c      write(*,*) 'ITER = ',iter,' - ERR = ',err
c      write(20,*) 'ERR = ',err
c
cc      write(21,101) iter,errd,errs/svm,err2mod,dvm,svm,
cc     #            sinhom,dinhom,((gavmod(k1)),k1=1,nmodes(1))
cc
cc      write(21,101) iter,errd/dvm,errs/svm,err2mod,svm,dvm,
cc     # effmu13,effmu23,sinhom,dinhom,((gavmod(k1)),k1=1,nmodes(1))
c
      write(21,101) iter,errd/dvm,errs/svm,dvm,svm,
     # wdot/dvm,effmu13,sinhom,dinhom,((gavmod(k1)),k1=1,nmodes(1))
c
c101   format(i3,3(1x,e10.4),2(1x,f7.4),2(1x,f10.6),8(1x,F7.4))
101   format(i3,2(1x,e10.4),3(1x,f7.4),1x,f10.6,8(1x,F7.4))
c
c     ENDDO ... WHILE
c
      enddo
c
      IF(IUPDATE.EQ.1) THEN
c
c     VELMAX
c
      velmax(1)=dsim(1,1)*delt(1)*(npts1-1)
      velmax(2)=dsim(2,2)*delt(2)*(npts2-1)
      velmax(3)=dsim(3,3)*delt(3)*(npts3-1)
c
c    VELMACRO
c
      do 555 k=1,npts3
      do 555 j=1,npts2
      do 555 i=1,npts1
c
      velmacro(1,i,j,k)=(i-1)*velmax(1)/(npts1-1)
      velmacro(1,i,j,k)=velmacro(1,i,j,k)+(j-1)*udot(1,2)
      velmacro(2,i,j,k)=(j-1)*velmax(2)/(npts2-1)
      if(npts3.gt.1) then
      velmacro(3,i,j,k)=(k-1)*velmax(3)/(npts3-1)
      else
      velmacro(3,i,j,k)=0.
      endif
c
555   continue
c
c   LOCAL VELOCITIES
c
      do 510 m=1,3
c
      k1=0
c
      do 350 k=1,npts3
      do 350 j=1,npts2
      do 350 i=1,npts1
c
      k1=k1+1
      data(k1)=vel(m,i,j,k)
c
      k1=k1+1
      data(k1)=velim(m,i,j,k)
c
350   continue
c
      if(npts3.gt.1) then
      call fourn(data,nn,3,-1)
      else
      call fourn(data,nn2,2,-1)
      endif
calb      call fourn(data,nn,3,-1)
c
      do i=1,2*npts1*npts2*npts3
calb      data(i)=data(i)/prodnn
      data(i)=data(i)/prodnn
      enddo
c
      k1=0
      do 160 kzz=1,npts3
      do 160 kyy=1,npts2
      do 160 kxx=1,npts1
      k1=k1+1
      vel(m,kxx,kyy,kzz)=velmacro(m,kxx,kyy,kzz)+data(k1)
      k1=k1+1
c      write(*,*) 'IMAGINARY PART =',kxx,kyy,kzz,data(k1)
c      pause
160   continue
c
510   continue
c
c     UPDATE ORIENTATIONS
c
      call update_orient
c
c     UPDATE MICROSTRUCTURE 
c
      do 888 i=1,npts1
      do 888 j=1,npts2
      do 888 k=1,npts3
cu
      do m=1,3
cw      tmacro1(m,i,j,k)=tmacro1(m,i,j,k)+velmacro(m,i,j,k)*tdot
      tlocal(m,i,j,k)=tlocal(m,i,j,k)+vel(m,i,j,k)*tdot
      vel_data(m,i,j,k)=vel(m,i,j,k)-velmacro(m,i,j,k)
      enddo
cu
cu      t(1)=tlocal(1,i,j,k)
cu      t(2)=tlocal(2,i,j,k)
cu      t(3)=tlocal(3,i,j,k)
cuc
cuc      write(*,*) 'T=',t
cuc    
cu      call inter_velocity(t,delt,velint)
cuc
cuc      write(*,*) velint
cuc      pause
cuc
cu      do m=1,3
cu      tlocal(m,i,j,k)=tlocal(m,i,j,k)+velint(m)*tdot
cu      enddo
cuc
888   continue
c
c     UPDATE DELT
c
      odelt(1)=delt(1)
      odelt(2)=delt(2)
      odelt(3)=delt(3)

      delt(1)=(delt(1)*(npts1-1)+velmax(1)*tdot)/(npts1-1)
      delt(2)=(delt(2)*(npts2-1)+velmax(2)*tdot)/(npts2-1)
      if(npts3.gt.1) then
      delt(3)=(delt(3)*(npts3-1)+velmax(3)*tdot)/(npts3-1)
      endif
calb
      vel_alb(1)=velmax(1)*tdot/(npts1-1)
      vel_alb(2)=velmax(2)*tdot/(npts2-1)
      vel_alb(3)=0
cw
      do 737 i=1,npts1
      do 737 j=1,npts2
      do 737 k=1,npts3
      tmacro2(1,i,j,k)=(i-1)*odelt(1)+(j-1)*fshear
      tmacro2(2,i,j,k)=(j-1)*odelt(2)
      tmacro2(3,i,j,k)=(k-1)*odelt(3)
737   continue
cw
cuc
cuc     REASIGN GRAIN PROPERTIES TO GRID POINTS
cuc
cu      call reasign(delt,deltvol3)
c
      ENDIF
c
3000  CONTINUE
c
      if(iwriteg.eq.0) call second_order(1)
      if(iwriteg.eq.1) call second_order(2)
c
      IF(IUPDATE.EQ.1) THEN
c
c     TEX.OUT
c
      open(24,file='tex.out',status='unknown')
      open(94,file='unodeang.out',status='unknown')
      open(95,file='unodexyz.out',status='unknown')
calb      open(96,file='fpxyz.out',status='unknown')
calb generate a temp.out to store x and y length of model, and displacement 
      open(97,file='temp-FFT.out',status='unknown')
calb      open(99,file='vel.out',status='unknown')
calb      open(100,file='velmacro.out',status='unknown')
calb      open(101,file='veldata.out',status='unknown')
c
c By Florian Steinbach: Write strain rate tensors of Fourier points to file
c
c      open(103,file='edottensors.out',status='unknown')
c
      zero=0.
      one=1.
c
      do i=1,3
      do j=1,3
      fbar(i,j)=0.
      enddo
      enddo
c
      do i=1,3
      fbar(i,i)=delt(i)
      enddo
c
calb      write(24,*)
calb     #'TEX.OUT: generated w/PPC13, to be plotted with POLE or CORTE'
calb      write(24,*) npts1
calb      write(24,111)((fbar(i,j),j=1,3),i=1,3)
calb      write(24,'(a1,i10)') 'B',npts1*npts2*npts3
calb temporal file 
      write(97,112)(fbar(i,i),i=1,3) 
      write(97,112)(vel_alb(i),i=1,3) 
      write(97,112) udot(1,2)*tdot, udot(1,3)*tdot, udot(2,3)*tdot  
c
111   format(9f7.3) 
112   format(3f10.6) 

calb
calb Geometrical Necessary Dislocation Density using Strain Gradient Theory-- curvature of strain field 
calb Calculate a scalar value, assume similar burgers vector   
c
c  CALCULATE third order strain gradient tensor (only antyisymmetric part?)
c
calb 1
calb calculate dislacement vector 
      length_alb(1)=2*(odelt(1)/npts1)*scalealb
      length_alb(2)=2*(odelt(2)/npts2)*scalealb
calb 2
calb calculate antysimmetric part velgrad
      do k=1,npts3
      do j=1,npts2
      do i=1,npts1
c
c     LOCAL ROTATION RATE: ANTISYM(VELGRAD)
c
      do ii=1,3
      do jj=1,3
      rot(ii,jj,i,j,k)=(velgrad(ii,jj,i,j,k)-velgrad(jj,ii,i,j,k))/2.0
      enddo
      enddo
c 
      enddo
      enddo
      enddo
calb3
calb Calculate 2nd gradient of displacement 
calb
      do k=1,npts3
      do j=1,npts2
      do i=1,npts1
calb central difference, wrapping with boundary FP 
      ii=i+1
      iii=i-1
      jj=j+1
      jjj=j-1
calb
      if (iii.LT.1) iii=npts1
      if (ii.GT.npts1) ii=1
calb
      if (jjj.LT.1) jjj=npts2
      if (jj.GT.npts2) jj=1
calb
calb Effective plastic strain rate gradient as Brinckmann et al., 2006
calb substitute velgrad per dtilde (is the symmetric tensor part and in the crystal reference frame)
      nn_d(1,1,1)=(rot(1,1,ii,j,k)-rot(1,1,iii,j,k))/
     # length_alb(1)
      nn_d(1,1,2)=(rot(1,2,ii,j,k)-rot(1,2,iii,j,k))/
     # length_alb(1)
      nn_d(1,2,1)=(rot(2,1,ii,j,k)-rot(2,1,iii,j,k))/
     # length_alb(1)
      nn_d(1,2,2)=(rot(2,2,ii,j,k)-rot(2,2,iii,j,k))/
     # length_alb(1)
calb
      nn_d(2,1,1)=(rot(1,1,i,jj,k)-rot(1,1,i,jjj,k))/
     # length_alb(2) 
      nn_d(2,1,2)=(rot(1,2,i,jj,k)-rot(1,2,i,jjj,k))/
     # length_alb(2)
      nn_d(2,2,1)=(rot(2,1,i,jj,k)-rot(2,1,i,jjj,k))/
     # length_alb(2)
      nn_d(2,2,2)=(rot(2,2,i,jj,k)-rot(2,2,i,jjj,k))/
     # length_alb(2)
calb
calb Calculate invariant [1/L**2]
      nn_inv(i,j,k)=0

      do n=1,2
      do m=1,2
      do l=1,2

      nn_inv(i,j,k)= nn_inv(i,j,k)+nn_d(l,m,n)*nn_d(l,m,n)

      enddo
      enddo
      enddo
calb Scale Scale 
calb ice burger vectors 5e-10 m, scale factor model per nn_inv [1/L] 
      nn_inv(i,j,k)=((0.25*nn_inv(i,j,k))**0.5)/burgers_vect
      nn_inv(i,j,k)=tdot*nn_inv(i,j,k)

calb      write(*,*) 'nn_inv', nn_inv(i,j,k)

      enddo
      enddo
      enddo
calb end do of the 3rd 
calb End of Strain Gradient 
c
      ig=0
      do k=1,npts3
      do j=1,npts2
      do i=1,npts1
      ig=ig+1
c
      do ii=1,3
      do jj=1,3
      aux33(ii,jj)=ag(jj,ii,i,j,k)
      enddo
      enddo
c
      call euler(1,ph,th,om,aux33)
c
c      write(24,'(3f11.5,i6,i3)') 
c     # ph,th,om,jgrain(i,j,k),mark(jgrain(i,j,k))
c
c     together with the euler angles, 
c     write equivalent strain-rate ...
c
       do ii=1,5
       sg5(ii)=sg(ii,i,j,k)
       dg5(ii)=dbar(ii)+dtilde(ii,i,j,k)
       enddo

       call chg_basis(sg5,sgx,aux55,aux3333,1)
       call chg_basis(dg5,dgx,aux55,aux3333,1)
c
       deqx=0.
       seqx=0.
cc       wdotx=0.
       do ii=1,3
       do jj=1,3
       deqx=deqx+dgx(ii,jj)*dgx(ii,jj)
       seqx=seqx+sgx(ii,jj)*sgx(ii,jj)
cc       wdotx=wdotx+sgx(ii,jj)*dgx(ii,jj)
       enddo
       enddo
c
      deqx=sqrt(2./3.*deqx)
      seqx=sqrt(3./2.*seqx)
c
c     ... and activities of the 2 first active modes 
c         for each Fourier point
c
      gmodgr1=gmodgr(1,i,j,k)
      gmodgr2=gmodgr(2,i,j,k)

calb statiscally stored dislocation density, using equivalent strain; see Gao et al., 1999; Brinckmann et al., 2006
calb "scalealb/npts1" represents "the radius of dislocation loop". This value evolve with deformation (decrease) or annealing (increase)
calb but here I assume a constant value and it's the maximum travel distance..the length of one FP (?¿)
calb To revise .. .. ..  
      dis_stat=3*sqrt(1./4.)*tdot*deqx/(burgers_vect*scalealb/npts1)
c
cc      write(24,172) 
cc     # ph,th,om,one,zero,deqx,gmodgr1,gmodgr2,ig,jgrain(i,j,k)
c
calb modify tex-out to add dislocation density and all dislocation activities 

      write(24,172) 
     # ph,th,om,one,deqx/dvm,seqx,gmodgr1,gmodgr2,nn_inv(i,j,k),
     # dis_stat,ig,jgrain(i,j,k)
c
172   format(3f8.2,f6.2,4f12.8,e15.8,e15.8,2i10)

c
c By Florian: Write components of strain rate tensor to file:
c
c      write(103,'(7E17.9)') dgx(1,1),dgx(2,2),dgx(3,3),
c     # dgx(2,3),dgx(1,3),dgx(1,2)
c
c 173   format(6e15.8) 

      enddo
      enddo
      enddo
c
c     UNODE FILES
c
      ig=0
c
      do k=1,npts3
      do j=1,npts2
      do i=1,npts1
c
      ig=ig+1
c
      do ii=1,3
      do jj=1,3
      aux33(ii,jj)=ag(jj,ii,i,j,k)
      enddo
      enddo
c
      call euler(1,ph,th,om,aux33)
c
      write(94,272) ig-1,ph,th,om
cww      write(95,273) ig-1,tlocal(2,i,j,k)/npts1,
cww     #              tlocal(1,i,j,k)/npts2,tlocal(3,i,j,k)/npts3
cw      write(95,273) ig-1,tlocal(1,i,j,k),tlocal(2,i,j,k),tlocal(3,i,j,k)
      write(95,273) ig-1,tlocal(1,i,j,k)/npts1,tlocal(2,i,j,k)/npts2,
     #              tlocal(3,i,j,k)/npts3
cw      write(96,273) ig-1,tmacro1(2,i,j,k)/npts1,
cw     #              tmacro1(1,i,j,k)/npts2,tmacro1(3,i,j,k)/npts3
calb      write(96,273) ig-1,tmacro2(1,i,j,k)/npts1,
calb     #              tmacro2(2,i,j,k)/npts2,tmacro2(3,i,j,k)/npts3
c

calb      write(99,272) ig-1,vel(1,i,j,k)*tdot/npts1,
calb     #        vel(2,i,j,k)*tdot/npts2,vel(3,i,j,k)*tdot/npts3
calb      write(100,272) ig-1,velmacro(1,i,j,k)*tdot/npts1,
calb     #  velmacro(2,i,j,k)*tdot/npts2,velmacro(3,i,j,k)*tdot/npts3
calb      write(101,272) ig-1,vel_data(1,i,j,k)*tdot/npts1,
calb     #  vel_data(2,i,j,k)*tdot/npts2,vel_data(3,i,j,k)*tdot/npts3
272   format(i9,3(1x,e15.8))
273   format(i9,3(1x,f11.8)) 
cc 274   format(2(1x,f11.8),3(1x,e15.8))


c
      enddo
      enddo
      enddo
c
      ENDIF
c      
c      STR_STR.OUT
c
c      do k=1,npts3
c      do j=1,npts2
c      do i=1,npts1
c
c      do ii=1,5
c      aux5(ii)=dbar(ii)+dtilde(ii,i,j,k)
c      enddo
c
c      call chg_basis(aux5,auxd,aux55,aux3333,1)
c
c      do ii=1,5
c      aux5(ii)=sg(ii,i,j,k)
c      enddo
c
c      call chg_basis(aux5,auxs,aux55,aux3333,1)
c
c      write(40,'(4i5)') i,j,k,jgrain(i,j,k)
c      write(40,305)
c     #      auxd(1,1),auxd(2,2),auxd(3,3),
c     #      auxd(2,3),auxd(3,1),auxd(1,2)
c      write(40,305)
c     #      auxs(1,1),auxs(2,2),auxs(3,3),
c     #      auxs(2,3),auxs(3,1),auxs(1,2)
c305   format(6(f9.5,1x))
c
c      enddo
c      enddo
c      enddo
c
c      STRESS.OUT
c
      if(isave.eq.1) then
c
      open(40,file='stress.out',status='unknown',
     #     access='sequential',form='unformatted')
c
      do i=1,5
      write(40) (xlsec(i,j),j=1,5)
      enddo
c
      do kk=1,npts3
      do jj=1,npts2
      do ii=1,npts1
c
      write(40) (sg(i,ii,jj,kk),i=1,5)
c
      enddo
      enddo
      enddo
c
      endif
c
cx      write (20,'(a)') 
cx     # '*******************************************************'
cx      write (20,310) ' ERR s,d,eq, IT     =',
cx     # errs/svm,errd/dvm,err2mod,iter
cx      write (20,311) ' SVMis, DVMis       =',svm,dvm
cx      write (20,311)  ' S11,22,33,23,13,12 =',
cx     # sdeviat(1,1),sdeviat(2,2),sdeviat(3,3),
cx     # sdeviat(2,3),sdeviat(1,3),sdeviat(1,2)
cx      write (20,311)  ' D11,22,33,23,13,12 =',
cx     # dsim(1,1),dsim(2,2),dsim(3,3),
cx     # dsim(2,3),dsim(1,3),dsim(1,2)
cx      write (20,311) ' TAUs, TAUh         =',tau(1,1,1),tau(2,1,1)
cx      write (20,312) ' EfVisc 13,23       =',effmu13,effmu23
cx      write (20,311) ' MacroOmega S,D     =',sinhom,dinhom
cx      write (20,311) ' Activ per mode     =',
cx     # (gavmod(k1),k1=1,nmodes(1))
cx      write (20,'(a)')
cx     # '*******************************************************'
cx      write(20,*) 
cx     #'   XOR   MRK  SEQ2/SEQ   SEQ/SVM    DEQ2/DEQ   DEQ/DVM'
cx      do i=1,nor
cx      write(20,102) xor(i),mark(i),seq2(i)/seq(i),seq(i)/svm,
cx     # deq2(i)/deq(i),deq(i)/dvm
cx      enddo
cx      write (22,'(a)') 
cx     # '*******************************************************'
cx      write (22,310) ' ERR s,d,eq, IT     =',
cx     # errs/svm,errd/dvm,err2mod,iter
cx      write (22,311) ' SVMis, DVMis       =',svm,dvm
cx      write (22,311)  ' S11,22,33,23,13,12 =',
cx     # sdeviat(1,1),sdeviat(2,2),sdeviat(3,3),
cx     # sdeviat(2,3),sdeviat(1,3),sdeviat(1,2)
cx      write (22,311)  ' D11,22,33,23,13,12 =',
cx     # dsim(1,1),dsim(2,2),dsim(3,3),
cx     # dsim(2,3),dsim(1,3),dsim(1,2)
cx      write (22,311) ' TAUs, TAUh         =',tau(1,1,1),tau(2,1,1)
cx      write (22,312) ' EfVisc 13,23       =',effmu13,effmu23
cx      write (22,311) ' MacroOmega S,D     =',sinhom,dinhom
cx      write (22,311) ' Activ per mode     =',
cx     # (gavmod(k1),k1=1,nmodes(1))
cx      write (22,'(a)')
cx     # '*******************************************************'
cx      write(22,*)'   XOR   MRK TA12/TA1    TA1/SVM   TA22/TA2    TA2/SVM
cx     #   GD12/GA1    GD1/DVM   GD22/GA2    GD2/DVM'
cx      do i=1,nor
cx      write(22,102) xor(i),mark(i),
cx     # ta12(i)/ta1(i),ta1(i)/svm,
cx     # ta22(i)/ta2(i),ta2(i)/svm,
cx     # gd12(i)/gd1(i),gd1(i)/dvm,
cx     # gd22(i)/gd2(i),gd2(i)/dvm
cx102   format(f8.3,i4,8(f10.5,1x))
cx      enddo
c
      write (23,'(a)') 
     # '*******************************************************'
      write (23,310) ' ERR s,d,eq, IT     =',
     # errs/svm,errd/dvm,err2mod,iter
      write (23,3115) ' SVM, DVM, Wdot/DVM =',svm,dvm,wdot/dvm
      write (23,3115)  ' S11,22,33,23,13,12 =',
     # sdeviat(1,1),sdeviat(2,2),sdeviat(3,3),
     # sdeviat(2,3),sdeviat(1,3),sdeviat(1,2)
      write (23,3115)  ' D11,22,33,23,13,12 =',
     # dsim(1,1),dsim(2,2),dsim(3,3),
     # dsim(2,3),dsim(1,3),dsim(1,2)
      write (23,311) ' TAUs, TAUh         =',tau(1,1,1),tau(2,1,1)
      write (23,312) ' EfVisc 1,2         =',effmu13,effmu23
      write (23,311) ' MacroOmega S,D     =',sinhom,dinhom
      write (23,311) ' Activ per mode     =',
     # (gavmod(k1),k1=1,nmodes(1))
      write (23,'(a)')
     # '*******************************************************'
c
310   format(a,3e10.3,i4)
311   format(a,6F9.5)
3115  format(a,6E11.3)
312   format(a,2(F12.7,1x))
c
      write (23,'(a)') '       XOR       MRK'
      write (23,'(a)') 
     # ' SEQ_1         SEQ_2         DEQ_1         DEQ_2'
      write (23,'(a)') 
     # ' S23_1         S23_2         S13_1         S13_2'
      write (23,'(a)') 
     # ' TAs_1         TAs_2         TAh_1         TAh_2'
      write (23,'(a)') 
     # ' GDs_1         GDs_2         GDh_1         GDh_2'
      do i=1,nor      
      write(23,302) xor(i),mark(i)
      write(23,303) seq(i),seq2(i),deq(i),deq2(i)
      write(23,303) s23(i),s232(i),s13(i),s132(i)
      write(23,303) ta1(i),ta12(i),ta2(i),ta22(i)
      write(23,303) gd1(i),gd12(i),gd2(i),gd22(i)
302   format (F10.3,i10)
303   format (6(E13.6,1x))
      enddo
c
      end
c     
      SUBROUTINE fourn(data,nn,ndim,isign)
      INTEGER isign,ndim,nn(ndim)
      REAL data(*)
      INTEGER i1,i2,i2rev,i3,i3rev,ibit,idim,ifp1,ifp2,ip1,ip2,ip3,k1,
     *k2,n,nprev,nrem,ntot
      REAL tempi,tempr
      DOUBLE PRECISION theta,wi,wpi,wpr,wr,wtemp
      ntot=1
      do 11 idim=1,ndim
        ntot=ntot*nn(idim)
11    continue
      nprev=1
      do 18 idim=1,ndim
        n=nn(idim)
        nrem=ntot/(n*nprev)
        ip1=2*nprev
        ip2=ip1*n
        ip3=ip2*nrem
        i2rev=1
        do 14 i2=1,ip2,ip1
          if(i2.lt.i2rev)then

            do 13 i1=i2,i2+ip1-2,2
              do 12 i3=i1,ip3,ip2
                i3rev=i2rev+i3-i2
                tempr=data(i3)
                tempi=data(i3+1)
                data(i3)=data(i3rev)
                data(i3+1)=data(i3rev+1)
                data(i3rev)=tempr
                data(i3rev+1)=tempi
12            continue
13          continue
          endif
          ibit=ip2/2
1         if ((ibit.ge.ip1).and.(i2rev.gt.ibit)) then
            i2rev=i2rev-ibit
            ibit=ibit/2
          goto 1
          endif

          i2rev=i2rev+ibit
14      continue
        ifp1=ip1
2       if(ifp1.lt.ip2)then
          ifp2=2*ifp1
          theta=isign*6.28318530717959d0/(ifp2/ip1)
          wpr=-2.d0*sin(0.5d0*theta)**2
          wpi=sin(theta)
          wr=1.d0
          wi=0.d0
          do 17 i3=1,ifp1,ip1
            do 16 i1=i3,i3+ip1-2,2
              do 15 i2=i1,ip3,ifp2
                k1=i2
                k2=k1+ifp1
                tempr=sngl(wr)*data(k2)-sngl(wi)*data(k2+1)
                tempi=sngl(wr)*data(k2+1)+sngl(wi)*data(k2)
                data(k2)=data(k1)-tempr

                data(k2+1)=data(k1+1)-tempi
                data(k1)=data(k1)+tempr
                data(k1+1)=data(k1+1)+tempi
15            continue
16          continue
            wtemp=wr
            wr=wr*wpr-wi*wpi+wr
            wi=wi*wpr+wtemp*wpi+wi
17        continue
          ifp1=ifp2
        goto 2
        endif
        nprev=n*nprev
18    continue
      return
      END
c
      SUBROUTINE VOIGT(C2,C4,IOPT)
C
C *** TRANSFORMS SECOND ORDER MATRIX C2 INTO FOURTH ORDER TENSOR C4 IF
C *** IOPT=1 AND VICEVERSA IF IOPT=2. IF IOPT=3,TRANSFORMS WITH INV.FACT.
C *** IOPT=4 FOR GO FROM 6x6 TO 3x3x3x3 WITH Aijkl ANTISYMMETRY
C
      DIMENSION C2(6,6),C4(3,3,3,3),IJV(6,2),F(6,6)
      DATA ((IJV(N,M),M=1,2),N=1,6)/1,1,2,2,3,3,2,3,1,3,1,2/
C
      IF(IOPT.EQ.1) THEN
      DO 10 I=1,6
      I1=IJV(I,1)
      I2=IJV(I,2)
      DO 10 J=1,6
      J1=IJV(J,1)
      J2=IJV(J,2)
      C4(I1,I2,J1,J2)=C2(I,J)
      C4(I2,I1,J1,J2)=C2(I,J)
      C4(I1,I2,J2,J1)=C2(I,J)
   10 C4(I2,I1,J2,J1)=C2(I,J)
      ENDIF
C
      IF(IOPT.EQ.2) THEN
      DO 20 I=1,6
      I1=IJV(I,1)
      I2=IJV(I,2)
      DO 20 J=1,6
      J1=IJV(J,1)
      J2=IJV(J,2)
   20 C2(I,J)=C4(I1,I2,J1,J2)
      ENDIF
c      
      IF(IOPT.EQ.3) THEN
      DO 9 I=1,6
      DO 9 J=1,6
      F(I,J)=1.
      IF(I.GT.3) F(I,J)=0.5
      IF(J.GT.3) F(I,J)=0.5*F(I,J)
9     CONTINUE
C
      DO 101 I=1,6
      I1=IJV(I,1)
      I2=IJV(I,2)
      DO 101 J=1,6
      J1=IJV(J,1)
      J2=IJV(J,2)
      C4(I1,I2,J1,J2)=F(I,J)*C2(I,J)
      C4(I2,I1,J1,J2)=F(I,J)*C2(I,J)
      C4(I1,I2,J2,J1)=F(I,J)*C2(I,J)
101   C4(I2,I1,J2,J1)=F(I,J)*C2(I,J)
      ENDIF
C
      IF(IOPT.EQ.4) THEN
      DO 17 I=1,6
      I1=IJV(I,1)
      I2=IJV(I,2)
      DO 17 J=1,6
      J1=IJV(J,1)
      J2=IJV(J,2)
      IF(I.LE.3) THEN
      C4(I1,I2,J1,J2)=C2(I,J)
      C4(I2,I1,J1,J2)=C2(I,J)
      C4(I1,I2,J2,J1)=C2(I,J)
      C4(I2,I1,J2,J1)=C2(I,J)
      ELSE
      C4(I1,I2,J1,J2)=C2(I,J)
      C4(I2,I1,J1,J2)=-C2(I,J)
      C4(I1,I2,J2,J1)=C2(I,J)
      C4(I2,I1,J2,J1)=-C2(I,J)
      ENDIF
17    CONTINUE
      ENDIF
      RETURN
      END
c
      subroutine euler(iopt,ph,th,tm,a)
      dimension a(3,3)
      pi=4.*atan(1.d0)
c
c     CALCULATE THE EULER ANGLES ASSOCIATED WITH THE TRANSFORMATION
c     MATRIX A(I,J) IF IOPT=1 AND VICEVERSA IF IOPT=2
c     A(i,j) TRANSFORMS FROM SYSTEM sa TO SYSTEM ca.
c     ph,th,om ARE THE EULER ANGLES OF ca REFERRED TO sa.
c
      if(iopt.eq.1) then
        th=acos(a(3,3))
        if(abs(a(3,3)).ge.0.9999) then
          tm=0.
          ph=atan2(a(1,2),a(1,1))
        else
          sth=sin(th)
          tm=atan2(a(1,3)/sth,a(2,3)/sth)
          ph=atan2(a(3,1)/sth,-a(3,2)/sth)
        endif
        th=th*180./pi
        ph=ph*180./pi
        tm=tm*180./pi
      else if(iopt.eq.2) then
        sph=sin(ph)
        cph=cos(ph)
        sth=sin(th)
        cth=cos(th)
        stm=sin(tm)
        ctm=cos(tm)
        a(1,1)=ctm*cph-sph*stm*cth
        a(2,1)=-stm*cph-sph*ctm*cth
        a(3,1)=sph*sth
        a(1,2)=ctm*sph+cph*stm*cth
        a(2,2)=-sph*stm+cph*ctm*cth
        a(3,2)=-sth*cph
        a(1,3)=sth*stm
        a(2,3)=ctm*sth
        a(3,3)=cth
      endif

      return
      end
c
C ********************************************************************
C     SUBROUTINE VPSC_INPUT      --->      VERSION 31/jan/99
C
C     READS CHARACTERISTICS OF THE RUN: # OF PHASES, NAMES OF INPUT FILES,
C     DEFORMATION TO BE IMPOSED, CONVERGENCE PARAMETERS, ETC.
C     READS SINGLE CRYSTAL PROPERTIES: DEFORMATION MODES, CRSS, HARDENING
C     READS CRYSTAL AND MORPHOLOGIC TEXTURES.
C     INITIALIZES ARRAYS REQUIRED TO RUN VPSC.
C     OPENS AND CLOSES INPUT FILES.   OPENS OUTPUT FILES.
C
C     MODIFIED 21/07/98 by CNT:
C     INITIALIZATION RELATED TO 'ELEMENTS' IS DONE INSIDE A SINGLE BLOCK.
C *****************************************************************************
C
      SUBROUTINE VPSC_INPUT

      INCLUDE 'ppc.dim'

      dimension aux55(5,5),aux3333(3,3,3,3)
      DIMENSION IJV(6,2)
      DATA ((IJV(N,M),M=1,2),N=1,6)/1,1,2,2,3,3,2,3,1,3,1,2/

C *********   INITIALIZATION BLOCK   ***************************
C
      PI=4.*ATAN(1.)
C
      IREADSET=0      ! used as control in CUBCOMP to open unit
C
C     CALCULATES TENSORS OF THE SYMMETRIC BASIS 'B(3,3,6)'
      CALL CHG_BASIS(DUM1,DUM2,DUM3,DUM4,0)
C
C     SEED FOR RANDOM NUMBER GENERATOR (RAN2) (USED FOR TWINNING AND RX)
      JRAN=-1
C
C     READS # OF ELEMENTS, # OF PHASES AND PHASE RELATIVE VOLUME FROM
C     DRIVER3a.IN.
C     PHASE AND ELEMENT IDENTIFICATION ARE CONTROLLED BY THE SAME INDEX.
C     AS A CONSEQUENCE THE ONLY CASES POSSIBLE ARE EITHER A MULTIPHASE
C     AGGREGATE WITH ONE ELEMENT, OR A MULTIELEMENT CALCULATION WITH 1 PHASE
C
c      READ(UR0,*) nelem
      nelem=1
      READ(UR0,*) nph
      READ(UR0,*) (wph(i),i=1,nph)
c
C     THE FOLLOWING REQUIRED FOR SEVERAL ROUTINES WITH 'do iph=iphbot,iphtop'
      if(nelem.eq.1) then
        iphbot=1
        iphtop=nph
      endif
c
      if(nph.gt.nphmx) then
        write(*,'('' number of phases exceeds multiphase dimens !!'')')
        write(*,'('' --> increase parameter NPHMX to'',i5)') nph
        stop
      endif
c
      if(nelem.gt.nphmx) then
        write(*,'('' number of elements exceeds multielement dim !!'')')
        write(*,'('' --> increase parameter NPHMX to'',i5)') nelem
        stop
      endif
c
c     CHECKS PHASE FRACTIONS AND NORMALIZES
c
      if(nph.gt.1) then
        wphtot=0.
        do iph=1,nph
          wphtot=wphtot+wph(iph)
        enddo
        if(abs(wphtot-1.) .gt. 1.e-4) then
          write(*,'('' --> vol fraction of phases should add to 1 !'')')
          stop
        endif
        do iph=1,nph
          wph(iph)=wph(iph)/wphtot
        enddo
      endif
c
c ***************************************************************************
c     BIG LOOP OVER PHASES
c     FOR EACH PHASE READS CRYSTAL SYSTEMS, TEXTURE AND GRAIN SHAPES.
c ***************************************************************************
c
      NGR(0)=0
c
      DO IPH=1,NPH
c

      write(*,*) iph
      READ(UR0,'(a)') prosa
      write(*,*) prosa
      READ(UR0,*)     ngr(iph)
cc      READ(UR0,*)     npts1,npts2,npts3
cc      ngr(iph)=npts1*npts2*npts3
c
      ngr(iph)=ngr(iph-1)+ngr(iph)
c      if(ngr(iph).gt.ngrmx) then
c        write(*,'('' number of grains exceeds dimension !!'')')
c        write(*,'('' --> increase parameter NGRMX to'',i5)') ngr(iph)
c        stop
c      endif
c
      READ(UR0,'(a)') prosa
calb
      READ(UR0,*) filetext
      write(*,*) 'Reading input file: ',filetext
      READ(UR0,'(a)') prosa
      READ(UR0, '(a10)') filecrys
      write(*,*) 'Reading crystal file: ',filecrys 
calb  read fact per boundaries zones
      READ(UR0,*) FACT_BOUND
calb      write(*,*) fact_bound
c
c     READS SLIP AND TWINNING MODES FOR THE PHASE
c
      OPEN (unit=UR1,file=filecrys,status='old')
cw      OPEN (unit=UR1,file='ice3d.sx',status='old')
        call data_crystal(iph)
      CLOSE(unit=UR1)
c
c     READS INITIAL TEXTURE (BUNGE.ROE,KOCKS CONVENTIONS) FROM FILETEXT
c     READS INITIAL GRAIN AXES AND ORIENTATION FROM FILEAXES IF ISHAPE>2.
c
      OPEN(unit=UR2,file=filetext,status='old')
cw      OPEN(unit=UR2,file='make.out',status='old')
        call data_grain(iph)
      CLOSE(unit=UR2)

c
c     INITIALIZE HARDENING, ACCUM SHEAR AND BEARTEX ARRAYS FOR GRAINS
c
c      do KKK=ngr(iph-1)+1,ngr(iph)
CFEB
c        eqst(kkk)  =0.
c        work(kkk)  =0.
CFEE
c        gtotgr(kkk)=0.
c
        do ii=1,npts1
        do jj=1,npts2
        do kk=1,npts3
c
        do i=1,nsyst(iph)
c 
        jph=jphase(ii,jj,kk)
        IF(jph.eq.iph) THEN
c 
        fact=1.
        crss(i,1,ii,jj,kk)=fact*tau(I,1,iph)
        crss(i,2,ii,jj,kk)=fact*tau(I,2,iph)
        ENDIF
calb
c        if(jphase(ii,jj,kk).eq.2) then 
c	 fact=fact_bound
c take homogeneous CRSS in phase 2, reference first glide system
c        crss(i,1,ii,jj,kk)=fact*tau(1,1,iph)
c        crss(i,2,ii,jj,kk)=fact*tau(1,2,iph)
c        endif
calb
c        if(jphase(ii,jj,kk).eq.3) then
c	fact=fact_bound
c CRSS contrast in all glide systems
c        crss(i,1,ii,jj,kk)=fact_bound*tau(I,1,iph)
c        crss(i,2,ii,jj,kk)=fact_bound*tau(I,2,iph)
c        endif
calb
c

       enddo
      enddo
      enddo
      enddo
c
c     INITIALIZE TWINNING PARAMETERS FOR PHASE AND GRAINS
c
c      NREOR (iph)=0
c      DO ITM=1,NTWMOD(IPH)
c        EFTWFR(itm,iph)=0.
c        TWFRPH(itm,iph)=0.
c      ENDDO
c      DO KKK=NGR(iph-1)+1,NGR(iph)
c        KTWREO(KKK)=0
c
c        NTWEVENTS(KKK)=0
c
c        DO IS=1,NTWSYS(IPH)
c          TWFRSY(IS,KKK)=0.
c        ENDDO
c      ENDDO
c
      ENDDO     ! END OF DATA INPUT LOOP OVER ALL PHASES
c ***************************************************************************
c
c      write(*,'('' --> NUMBER OF GRAINS PER ELEMENT IS'',I6)') ngr(nph)
c      if(ngr(nph). gt. NGRPEL) then
c        write(*,'('' --> INCREASE PARAMETER NGRPEL IN ppc.dim'')')
c        stop
c      endif
c
c      write(*,'('' --> TOTAL NUMBER OF GRAINS IS'',I6)') nelem*ngr(nph)
c      if(nelem*ngr(nph). gt. NGRMX) then
c        write(*,'('' --> INCREASE PARAMETER NGRMX IN ppc.dim'')')
c        stop
c      endif
c
c     CHECKS IF RATE SENSITIVITY IS THE SAME FOR ALL SYSTEMS.
c     IF NUNIQUE=0 WILL GO THROUGH AN EXTRA LOOP OVER THE TANGENT COMPLIANCE.
c
      nunique=1
      ncomp=nrs(1,1)
      DO IPH=1,NPH
        do is=1,nsyst(iph)
          if(ncomp.ne.nrs(is,iph)) nunique=0
        enddo
      ENDDO
      IF(NUNIQUE.EQ.0)
     #      WRITE(*,'('' THIS RUN CONTAINS MIXED nrs EXPONENTS !!'')')
c
c     SEARCH FOR NRSMIN (NEEDED TO GET TAUMAX INSIDE NR SUBROUTINE)
c
      nrsmin=nrs(1,1)
      DO IPH=1,NPH
        do is=1,nsyst(iph)
          if(nrs(is,iph).lt.nrsmin) nrsmin=nrs(is,iph)
        enddo
      ENDDO
c
C ****************************************************************************
C     READ BOUNDARY CONDITIONS ON OVERALL STRESS AND STRAIN-RATE
C ****************************************************************************
C
      READ(UR0,'(A)') PROSA
c
c     RVE DIMENSIONS
c
      READ(UR0,*) DELT
calb
      READ(98,*) DELT
calb  Add lines to read finite shear angle
      READ(98,'(A)') PROSA
      READ(98,*) fshear
c
      READ(UR0,'(A)') PROSA
C
      do i=1,3
        READ(UR0,*) (iudot(i,j),j=1,3)
      enddo
c
      if(iudot(1,1)+iudot(2,2)+iudot(3,3).eq.2) then
        write(*,*) 'CHECK DIAGONAL BOUNDARY CONDITIONS IUDOT'
        write(*,*) 'CANNOT ENFORCE ONLY TWO DEVIATORIC COMPONENTS'
        stop
      endif
c
      do i=1,3
      do j=1,3
        if(i.ne.j.and.iudot(i,j)+iudot(j,i).eq.0) then
          write(*,*) 'CHECK OFF-DIAGONAL BOUNDARY CONDITIONS IUDOT'
          stop
        endif
      enddo
      enddo
c
      READ(UR0,*)
      DO I=1,3
        READ(UR0,*) (UDOT(I,J),J=1,3)
      ENDDO
C
c     SYMMETRIC STRAIN-RATE, ANTISYMMETRIC ROTATION-RATE TENSORS
c     AND INDICES OF IMPOSED COMPONENTS
c
      do i=1,3
      do j=1,3
        dsim(i,j)=(udot(i,j)+udot(j,i))/2.
        tomtot(i,j)=(udot(i,j)-udot(j,i))/2.
      enddo
      enddo
c
      do i=1,3
        idsim(i)=iudot(i,i)
      enddo
c
      idsim(4)=0
      if(iudot(2,3).eq.1.and.iudot(3,2).eq.1) idsim(4)=1
      idsim(5)=0
      if(iudot(1,3).eq.1.and.iudot(3,1).eq.1) idsim(5)=1
      idsim(6)=0
      if(iudot(1,2).eq.1.and.iudot(2,1).eq.1) idsim(6)=1
c
c     WRITES STRAIN RATE DSIM(I,J) IN b-BASIS AS A 5-DIM VECTOR DBAR(K)
c
      call chg_basis(dbar,dsim,aux55,aux3333,2)

      READ(UR0,*)
      READ(UR0,*) iscau(1),iscau(6),iscau(5)
      READ(UR0,*) iscau(2),iscau(4)
      READ(UR0,*) iscau(3)
c
      do i=1,6
        if(iscau(i)*idsim(i).ne.0.or.iscau(i)+idsim(i).ne.1) then
          WRITE(*,*) ' CHECK BOUNDARY CONDITS ON STRAIN-RATE AND STRESS'
          WRITE(*,'('' IDSIM = '',6I3)') IDSIM
          WRITE(*,'('' ISCAU = '',6I3)') ISCAU
          STOP
        endif
      enddo
c
      READ(UR0,*)
      READ(UR0,*) scauchy(1,1),scauchy(1,2),scauchy(1,3)
      READ(UR0,*) scauchy(2,2),scauchy(2,3)
      READ(UR0,*) scauchy(3,3)
c
      scauchy(3,2)=scauchy(2,3)
      scauchy(3,1)=scauchy(1,3)
      scauchy(2,1)=scauchy(1,2)
c
c     IF SOME OFF-DIAG COMPS OF SCAUCHY ARE IMPOSED
c            WRITE THE CORRESP COMP OF SBAR
c
      if(iscau(4).eq.1) sbar(3)=sqrt(2.d0)*scauchy(2,3)
      if(iscau(5).eq.1) sbar(4)=sqrt(2.d0)*scauchy(1,3)
      if(iscau(6).eq.1) sbar(5)=sqrt(2.d0)*scauchy(1,2)
c
      READ(UR0,*)
      READ(UR0,*) DUMMY
      READ(UR0,*) ICTRL
c
      if(ictrl.eq.-1) tdot   =dummy
      if(ictrl.eq.0)  evmincr=dummy
      if(ictrl.gt.0)  eijincr=dummy
c
      if(ictrl.gt.0) then
        if(.not.(idsim(ictrl).eq.1.and.
     #      dsim(ijv(ictrl,1),ijv(ictrl,2)).ne.0.)) then
         write(*,*) 'ICTRL        =',ictrl
         write(*,*) 'IDSIM(ICTRL) =',idsim(ictrl)
         write(*,*) 'DSIM(ICRTL)  =',dsim(ijv(ictrl,1),ijv(ictrl,2))
         write(*,*) 'ICTRL MUST BE -1 TO IMPOSE DSIM*TDOT'
         write(*,*) 'OR IT MUST BE 0 TO CONTROL VON MISES INCREMENT'
         write(*,*) 'OR IT MUST IDENTIFY A NON-ZERO STRAIN COMPONENT'
         stop
        endif
      endif
c
      READ(UR0,*)
      READ(UR0,*) NSTEPS
      READ(UR0,*) ERROR
      READ(UR0,*) ITMAX
      READ(UR0,*) IRECOVER
      read(UR0,*) xlfac0, xlfac1
c
      if(irecover.eq.1) open(50,file='stress.in',status='old',
     #     access='sequential',form='unformatted') 
c
      READ(UR0,*) ISAVE
      READ(UR0,*) IWRITEG
      READ(UR0,*) IUPDATE
c
calb new parameters
      READ(UR0,'(A)') PROSA 
      READ(UR0,*) scalealb
      READ(UR0,*) burgers_vect
c
      RETURN
      END
c
C *****************************************************************************
C     SUBROUTINE DATA_CRYSTAL        --->      VERSION 03/FEB/2000
C *****************************************************************************

      SUBROUTINE DATA_CRYSTAL(IPH)

      INCLUDE 'ppc.dim'

      DIMENSION ISN(12,4),ISB(12,4),SN(3),SB(3),CDIM(3)
      DIMENSION aux5(5),aux33(3,3),aux55(5,5),aux3333(3,3,3,3)
      DIMENSION HSELFX(10),HLATEX(10,10),MODE(10)
c
      READ(UR1,'(a)') prosa
      READ(UR1,'(a)') icryst(iph)
      READ(UR1,*)     (cdim(i),i=1,3)
      covera=cdim(3)/cdim(1)
      READ(UR1,*)     nmodesx
      READ(UR1,*)     nmodes(iph)
      READ(UR1,*)     (mode(i),i=1,nmodes(iph))
C
      IF(NMODES(IPH).GT.NMODMX) THEN
        WRITE(*,'('' NMODES IN PHASE'',I3,'' IS'',I3)') IPH,NMODES(IPH)
        WRITE(*,'('' CHANGE PARAMETER NMODMX IN ppc.dim'')')
        STOP
      ENDIF
c
      NTWMOD(iph)=0
      NSYST(iph) =0
      NTWSYS(iph)=0
      KOUNT=1
c
c     START READING DEFORMATION MODES FROM FILECRYS
c
      do 100 nm=1,nmodesx
c
        READ(UR1,'(a)') prosa
        READ(UR1,*)     modex,nsmx,nrsx,gamd0x,twshx,isectwx
        READ(UR1,*)     tau0xf,tau0xb,tau1x,thet0x,thet1x
        READ(UR1,*)     hselfx(nm),(hlatex(nm,jm),jm=1,nmodesx)
c
c     SKIPS nsmx LINES IF THE MODE IS NOT IN THE LIST.
        if(modex.ne.mode(kount)) then
          do iz=1,nsmx
            READ(UR1,*)
          enddo
          go to 100
        endif
C
        IF(THET0X.LT.THET1X) THEN
          WRITE(*,'('' INITIAL HARDENING LOWER THAN FINAL HARDENING FOR
     #      MODE'',I3,''  IN PHASE'',I3)') KOUNT,IPH
          STOP
        ENDIF
C
C     CASE TAU1=0 CORRESPONDS TO LINEAR HARDENING AND IS INDEPENDENT OF TAU0.
C     AVOID DIVISION BY ZERO
        IF(TAU1X.LE.1.E-6) THEN
          TAU1X=1.E-6
          THET0X=THET1X
        ENDIF
c
c     REORDER HARDENING COEFFICIENTS TO ACCOUNT ONLY FOR ACTIVE MODES
        hselfx(kount)=hselfx(nm)
        do i=1,nmodes(iph)
          hlatex(kount,i)=hlatex(nm,mode(i))
        enddo
c
c     SYSTEMS GIVEN IN FOUR INDEX NOTATION: HEXAGONALS AND TRIGONALS
c     SYSTEMS GIVEN IN THREE INDEX NOTATION: CUBIC AND ORTHORHOMBIC
c
      IF(icryst(iph).EQ.'HEX' .OR. icryst(iph).EQ.'TRI' .OR.
     #   icryst(iph).EQ.'hex' .OR. icryst(iph).EQ.'tri') THEN
        DO J=1,NSMX
          READ(UR1,*) (ISN(J,K),K=1,4),(ISB(J,K),K=1,4)
        ENDDO
      ELSE IF(icryst(iph).EQ.'CUB' .OR. icryst(iph).EQ.'ORT' .OR.
     #        icryst(iph).EQ.'cub' .OR. icryst(iph).EQ.'ort') THEN
        DO J=1,NSMX
          READ(UR1,*)(ISN(J,K),K=1,3),(ISB(J,K),K=1,3)
        ENDDO
      ELSE
        WRITE(*,'('' CANNOT IDENTIFY THE CRYSTAL SYMMETRY OF PHASE '',
     #            I3)') IPH
        STOP
      ENDIF
C
      NSM(kount,iph)=nsmx
      IF(TWSHX.NE.0) NTWMOD(iph)=NTWMOD(iph)+1
C
      IF(NTWMOD(IPH).GT.NTWMMX) THEN
        WRITE(*,'('' NTWMOD IN PHASE'',I3,'' IS'',I3)') IPH,NTWMOD(IPH)
        WRITE(*,'('' CHANGE PARAMETER NTWMMX IN VPSC.DIM'')')
        STOP
      ENDIF
C
      DO 205 JS=1,NSM(kount,iph)
C
      NSYST(iph)=NSYST(iph)+1
      NSYSX=NSYST(iph)
      IF(TWSHX.NE.0) NTWSYS(iph)=NTWSYS(iph)+1
C
      IF(NSYST(IPH).GT.NSYSMX) THEN
        WRITE(*,'('' NSYST IN PHASE'',I3,'' IS'',I3)') IPH,NSYST(IPH)
        WRITE(*,'('' CHANGE PARAMETER NSYSMX IN VPSC.DIM'')')
        STOP
      ENDIF
C
C   DEFINES RATE SENSITIVITY AND CRSS FOR EACH SYSTEM IN THE MODE
C
      GAMD0(NSYSX,iph) =GAMD0X
      NRS(NSYSX,iph)   =NRSX
      TWSH(NSYSX,iph)  =TWSHX
      TAU(NSYSX,1,iph) =TAU0XF
      TAU(NSYSX,2,iph) =TAU0XB
      TAU(NSYSX,3,iph) =TAU1X
      THET(NSYSX,0,iph)=THET0X
      THET(NSYSX,1,iph)=THET1X
c
      isectw(NSYSX,iph)=isectwx
C
      IF(icryst(iph).EQ.'HEX' .OR. icryst(iph).EQ.'TRI' .OR.
     #   icryst(iph).EQ.'hex' .OR. icryst(iph).EQ.'tri') THEN
        SN(1)= ISN(JS,1)
        SN(2)=(ISN(JS,1)+2.*ISN(JS,2))/SQRT(3.)
        SN(3)= ISN(JS,4)/COVERA
        SB(1)= 3./2.*ISB(JS,1)
        SB(2)=(ISB(JS,1)/2.+ISB(JS,2))*SQRT(3.)
        SB(3)= ISB(JS,4)*COVERA
      ELSE IF(icryst(iph).EQ.'CUB' .OR. icryst(iph).EQ.'ORT' .OR.
     #        icryst(iph).EQ.'cub' .OR. icryst(iph).EQ.'ort') THEN
        DO M=1,3
          SN(M)=ISN(JS,M)/CDIM(M)
          SB(M)=ISB(JS,M)*CDIM(M)
        ENDDO
      ENDIF
C
C *** NORMALIZES SYSTEM VECTORS AND CHECKS NORMALITY
C
      SNOR=SQRT(SN(1)*SN(1)+SN(2)*SN(2)+SN(3)*SN(3))
      QNOR=SQRT(SB(1)*SB(1)+SB(2)*SB(2)+SB(3)*SB(3))
      PROD=0.
      DO J=1,3
        DNCA(J,NSYSX,iph)=SN(J)/SNOR
        DBCA(J,NSYSX,iph)=SB(J)/QNOR
        IF(ABS(DNCA(J,NSYSX,iph)).LT.1.E-03) DNCA(J,NSYSX,iph)=0.
        IF(ABS(DBCA(J,NSYSX,iph)).LT.1.E-03) DBCA(J,NSYSX,iph)=0.
        PROD=PROD+DNCA(J,NSYSX,IPH)*DBCA(J,NSYSX,IPH)
      ENDDO
      IF(PROD.GE.1.E-3) THEN
        WRITE(*,'(''SYSTEM'',I4,''  IN MODE'',I4,'' IN PHASE'',I4,
     #       ''  IS NOT ORTHOGONAL !!'')') JS,NM,IPH
        STOP
      ENDIF
C
C   DEFINE SCHMID VECTOR IN CRYSTAL AXES FOR EACH SYSTEM
C
      DO I=1,3
      DO J=1,3
        AUX33(i,j)=(DNCA(i,NSYSX,IPH)*DBCA(j,NSYSX,IPH)+
     #              DNCA(j,NSYSX,IPH)*DBCA(i,NSYSX,IPH))/2.
      ENDDO
      ENDDO
c
      call chg_basis(aux5,aux33,aux55,aux3333,2)
c
      DO I=1,5
        SCHCA(I,NSYSX,IPH)=AUX5(I)
      ENDDO

  205 CONTINUE    ! END OF LOOP OVER A GIVEN DEFORMATION MODE

      kount=kount+1

  100 CONTINUE    ! END OF LOOP OVER ALL MODES IN A GIVEN PHASE

C     INITIALIZE SELF & LATENT HARDENING COEFS FOR EACH SYSTEM OF THE PHASE.
C     ABSOLUTE UNITS ARE ACCOUNTED FOR BY MODULATING FACTOR IN HARDENING LAW.

      I=0
      DO IM=1,NMODES(iph)
      DO IS=1,NSM(IM,iph)
        I=I+1
        J=0
        DO JM=1,NMODES(iph)
        DO JS=1,NSM(JM,iph)
          J=J+1
          HARD(I,J,IPH)=HLATEX(IM,JM)
        ENDDO
        ENDDO
        HARD(I,I,IPH)=HSELFX(IM)
      ENDDO
      ENDDO

C     LATENT HARDENING OF SLIP AND TWINNING BY TWINNING IS BASED ON THE
C     RELATIVE DIRECTIONS OF THE SHEAR DIRECTION AND THE TWIN PLANE
C     THIS APPROACH IS STILL BEING TESTED (30/4/99)

C     NSLSYS=NSYST(IPH)-NTWSYS(IPH)
C     DO IS=1,NSYST(IPH)
C       DO JT=NSLSYS+1,NSYST(IPH)
C         IF(IS.NE.JT) THEN
C           COSA=DBCA(1,IS,IPH)*DNCA(1,JT,IPH)+
C    #           DBCA(2,IS,IPH)*DNCA(2,JT,IPH)+
C    #           DBCA(3,IS,IPH)*DNCA(3,JT,IPH)
C           COSA=ABS(COSA)
C           HARD(IS,JT,IPH)=HARD(IS,JT,IPH)*(0.5+1.0*COSA)
C         ENDIF
C       ENDDO
C     ENDDO
C
C     WRITE(10,'(''  HARDENING MATRIX FOR PHASE'',I3)') IPH
C     DO I=1,NSYST(IPH)
C       WRITE(10,'(24F5.1)') (HARD(I,J,IPH),J=1,NSYST(IPH))
C     ENDDO

C     VERIFICATION OF TWINNING DATA TO BE SURE PROGRAM WILL RUN PROPERLY

      IF (NMODES(IPH). GT. 1) THEN
        DO I=2,NSYST(IPH)
          IF(TWSH(I,IPH).EQ.0. .AND. TWSH(I-1,IPH).NE.0.) THEN
            WRITE(*,*) ' WARNING! THE TWINNING MODES MUST FOLLOW THE'
            WRITE(*,*) ' SLIP MODES   -->   REORDER CRYSTAL FILE'
            STOP
          ENDIF
        ENDDO
      ENDIF
C
      RETURN
      END
C
C *****************************************************************************
C     SUBROUTINE DATA_GRAIN        --->      VERSION 31/mar/99
C *****************************************************************************

      SUBROUTINE DATA_GRAIN(IPH)

      INCLUDE 'ppc.dim'

      DIMENSION AA(3,3)
c
c      CHARACTER NOMEN*1
c
c      READ(UR2,'(a)') prosa
c      READ(UR2,'(a)') prosa
c      READ(UR2,'(a)') prosa
c      READ(UR2,'(a)') nomen
c
      READ(UR2,*) nor
      do ior=1,nor
c
      read(UR2,*) dum,dum,dum,xor(ior),mark(ior)
c
c      read(UR2,*) dum,dum,dum,tayfac
c
c      IF(icryst(iph).EQ.'HEX'.OR.icryst(iph).EQ.'hex') THEN
c
c       if(ph.le.90) then
c        xor(ior)=ph*pi/180.
c       else
c        xor(ior)=(180.-ph)*pi/180.
c       endif
c
c      ELSE
c
c      xor(ior)=tayfac
c
c      ENDIF
c
      enddo
c
calb      do kkk=ngr(iph-1)+1,ngr(iph)
      do kkk=1,npts1*npts2*npts3
c
c      READ(UR2,*) ph,th,om,tayf,dg2,ii,jj,kk,jgr,jph
      READ(UR2,*) ph,th,om,ii,jj,kk,jgr,jph
c      READ(UR2,*) ph,th,om,tayf,ii,jj,kk,jgr,jph
c
c      write(*,*) ii,jj,kk,jgr,jph,iph
c
c      dg2vpsc(ii,jj,kk)=dg2
c      tayfac(ii,jj,kk)=tayf

Calb  READ ALL DATA, but select only iph data
      IF(jph.eq.iph) THEN
c 
      jgrain(ii,jj,kk)=jgr
      jphase(ii,jj,kk)=jph
calb      write(*,*) jph, iph
C
C     CALCULATES THE TRANSFORMATION MATRIX AA WHICH TRANSFORMS FROM
C     SAMPLE TO CRYSTAL. STORES AG, WHICH TRANSFORMS FROM CRYSTAL TO SAMPLE.

        CALL EULER(2,ph*pi/180,th*pi/180,om*pi/180,aa)
        DO J=1,3
        DO K=1,3
          AG(J,K,ii,jj,kk)=AA(K,J)
        ENDDO
        ENDDO
      ENDIF
Calb
      ENDDO
C
      RETURN
      END
c
C ************************************************************************
C
C     SUBROUTINE CHG_BASIS_OLD    --->   VERSION 06/02/98
C
C     (modif. 10/JAN/97 - KDIM version - R.L.)
C     (modif. 06/FEB/98 - same convention as SELFPOLY - C.N.T.)
C     (modif. 16/JUN/99 - same convention as Maudlin  - C.N.T.)
C
C     DEFINES A BASIS OF 6 SECOND ORDER TENSORS B(N) IF IOPT=0.
C     CALCULATES THE SECOND ORDER TENSOR 'C2' AS AN EXPANSION IN
C     TERMS OF ITS COMPONENTS CE2(N) AND THE BASIS TENSORS B(N) IF
C     IOPT=1.
C     CALCULATES THE COMPONENTS OF C2 AS A VECTOR CE2(6) IF IOPT=2.
C     CALCULATES THE FOURTH ORDER TENSOR 'C4' AS AN EXPANSION IN TERMS
C     OF ITS COMPONENTS CE4(N,M) AND THE BASIS TENSORS B(N) IF IOPT=3.
C     CALCULATES THE COMPONENTS OF C4 AS A MATRIX CE4(6,6) IF IOPT=4.
C **************************************************************************
C
      SUBROUTINE CHG_BASIS(CE2,C2,CE4,C4,IOPT)
C
      PARAMETER (KDIM=5)
c      PARAMETER (SQR2=1.41421356237309   )
      PARAMETER (RSQ2=0.70710678118654744)
      PARAMETER (RSQ3=0.57735026918962584)
      PARAMETER (RSQ6=0.40824829046386304)
C
      DIMENSION CE2(KDIM),C2(3,3),CE4(KDIM,KDIM),C4(3,3,3,3)
C
C     DIMENSION B(3,3,6)
C     DATA B /RSQ6,0,   0,   0,   RSQ6,0,   0,   0,  -2*RSQ6,
C    #        RSQ2,0,   0,   0,  -RSQ2,0,   0,   0,   0,
C    #        0,   0,   0,   0,   0,   RSQ2,0,   RSQ2,0,
C    #        0,   0,   RSQ2,0,   0,   0,   RSQ2,0,   0,
C    #        0,   RSQ2,0,   RSQ2,0,   0,   0,   0,   0,
C    #        RSQ3,0,   0,   0,   RSQ3,0,   0,   0,   RSQ3/

      COMMON/BASIS/ B(3,3,6)

      IF(IOPT.EQ.0) THEN
C *** CALCULATES BASIS TENSORS B(N)

        DO I=1,3
          DO J=1,3
            DO N=1,6
              B(I,J,N)=0.0
            ENDDO
          ENDDO
        ENDDO

        B(1,1,2)=-RSQ6
        B(2,2,2)=-RSQ6
        B(3,3,2)= 2.D0*RSQ6

        B(1,1,1)=-RSQ2
        B(2,2,1)= RSQ2

        B(2,3,3)=RSQ2
        B(3,2,3)=RSQ2

        B(1,3,4)=RSQ2
        B(3,1,4)=RSQ2

        B(1,2,5)=RSQ2
        B(2,1,5)=RSQ2

        B(1,1,6)=RSQ3
        B(2,2,6)=RSQ3
        B(3,3,6)=RSQ3

      ENDIF
C
      IF(IOPT.EQ.1) THEN
C *** CALCULATES CARTESIAN SECOND ORDER TENSOR FROM b-COMPONENTS VECTOR.
      DO 40 I=1,3
      DO 40 J=1,3
      C2(I,J)=0.0
      DO 40 N=1,KDIM
   40 C2(I,J)=C2(I,J)+CE2(N)*B(I,J,N)
C
      ENDIF
C
      IF(IOPT.EQ.2) THEN
C *** CALCULATES KDIMx1 b-COMPONENTS VECTOR FROM SECOND ORDER TENSOR.
      DO 50 N=1,KDIM
      CE2(N)=0.0
      DO 50 I=1,3
      DO 50 J=1,3
   50 CE2(N)=CE2(N)+C2(I,J)*B(I,J,N)
C
      ENDIF
C
      IF(IOPT.EQ.3) THEN
C *** CALCULATES FOURTH ORDER TENSOR FROM b-COMPONENTS MATRIX.
      DO 20 I=1,3
      DO 20 J=1,3
      DO 20 K=1,3
      DO 20 L=1,3
      C4(I,J,K,L)=0.0
      DO 20 N=1,KDIM
      DO 20 M=1,KDIM
   20 C4(I,J,K,L)=C4(I,J,K,L)+CE4(N,M)*B(I,J,N)*B(K,L,M)
C
      ENDIF
C
      IF(IOPT.EQ.4) THEN
C *** CALCULATES KDIMxKDIM b-COMPONENTS MATRIX FROM FOURTH ORDER TENSOR.
      DO 30 N=1,KDIM
      DO 30 M=1,KDIM
      CE4(N,M)=0.0
      DO 30 I=1,3
      DO 30 J=1,3
      DO 30 K=1,3
      DO 30 L=1,3
   30 CE4(N,M)=CE4(N,M)+C4(I,J,K,L)*B(I,J,N)*B(K,L,M)
C
      ENDIF
C
      RETURN
      END
c
C **************************************************************************
C     SUBROUTINE UPDATE_SCHMID
C
C     ROTATES SCHMID TENSORS OF EACH GRAIN FROM CRYSTAL TO SAMPLE AXES
C **************************************************************************
      SUBROUTINE UPDATE_SCHMID
c
      INCLUDE 'ppc.dim'
c
      DIMENSION aux5(5),aux33(3,3),aux55(5,5),aux3333(3,3,3,3)
      dimension aux33r(3,3)
c
      do ii=1,npts1
      do jj=1,npts2
      do kk=1,npts3
c
      do 120 is=1,nsyst(1)
c
      do j=1,5
        aux5(j)=schca(j,is,1)
      enddo
c
      call chg_basis(aux5,aux33,aux55,aux3333,1)
c
      do 140 i=1,3
      do 140 j=1,3
        aux33r(i,j)=0.
      do 140 i1=1,3
      do 140 j1=1,3
        aux33r(i,j)=aux33r(i,j)+
     #      ag(i,i1,ii,jj,kk)*ag(j,j1,ii,jj,kk)*aux33(i1,j1)
140   continue
c
      call chg_basis(aux5,aux33r,aux55,aux3333,2)
c
      do j=1,5
        sch(j,is,ii,jj,kk)=aux5(j)
      enddo
c
120   CONTINUE
c
      ENDDO
      ENDDO
      ENDDO
c
      return
      end
c
      SUBROUTINE VISC_FFT_INITIAL

      INCLUDE 'ppc.dim'
c
c      dimension dtilde(5,npts,npts,npts)
c
cc      dimension rss(NSYSMX),rss1(NSYSMX),x(5)
      dimension x(5)

      COMMON/NLNR /sc(5,NSYSMX),taux(NSYSMX,2),nrsx(NSYSMX),
     #             ti(5),fp1(5,5),sc1(5,NSYSMX)
c
c     EMPIRIC ALGORITHM TO GET TAULIM FOR NR SUBROUTINE (RL: 01/FEB/00)
c
      taulim=2.*(tnorm(dbar,5,1)/gamd0(1,1))**(1./nrsmin)
      if(taulim.lt.2.) taulim=2.
c
       do kk=1,npts3
c
cw      write(*,'(1H+,a,i4,a,i4)') ' Z = ',kk,'   OUT OF ',npts
c
       do jj=1,npts2
       do ii=1,npts1
c 
       nph=jphase(ii,jj,kk) 
c       write (*,*) ii,jj,kk,nph
        DO I=1,5
          TI(I)=DBAR(I)
          DO J=1,5
            FP1(I,J)=0.
          ENDDO
        ENDDO
      
C     COPY CRITICAL STRESS, RATE SENSIT, SCHMID TENSOR INTO AUXILIAR ARRAYS
C
      DO is=1,nsyst(1)
calb        nrsx(is)  =nrsmin 
        nrsx(is)  =nrs(is,nph)
c        write(*,*) nrs(is,nph), nph
        taux(is,1)=crss(is,1,ii,jj,kk)
        taux(is,2)=crss(is,2,ii,jj,kk)
        DO j=1,5
          sc(j,is)=sch(j,is,ii,jj,kk)
        ENDDO
      ENDDO

C     USE STRY AS A STARTING VALUE TO SOLVE NONLINEAR SYSTEM

      DO i=1,5
        x(i)=sg(i,ii,jj,kk)
      ENDDO

C     MAKE SURE STRY IS NOT OUTSIDE THE YIELD SURFACE.
C     CORRECTS THE NORM IF IT EXCEEDS THE Y.S. BY MORE THAN 10%

  100 taumax=0.
      do ik=1,nsyst(1)
        htau=0.
        do jk=1,5
          htau=htau+x(jk)*sc(jk,ik)
        enddo
        isign=1
        if(htau.lt.0.) isign=2
        htau=htau/taux(ik,isign)
        if(abs(htau).gt.taumax) taumax=abs(htau)
      enddo

      if(taumax. lt. 1.e-10) then
        write(*,'('' taumax.lt.1e-10 inside subroutine VISC'')')
calb        stop
      endif
c
c     if(taumax .gt. taulim) then
c
c     BACK TO 1.1 (RL: 9/FEB/00)
c
      if(taumax .gt. 1.1) then
        tcoef=1./taumax
        do i=1,5
          x(i)=x(i)*tcoef
        enddo
      endif

C     CALLS Newton-Raphson SUBROUTINE TO CALCULATE GRAIN STRESS

      itmx=1000
      eps=5.e-04

      CALL SNLNR_INITIAL (x,itmx,eps,taulim,ii,jj,kk)

      do i=1,5
        sg(i,ii,jj,kk)=x(i)
      enddo

      ENDDO
      ENDDO
      ENDDO
c
      RETURN
      END
c
      SUBROUTINE GET_GAMDOT

      INCLUDE 'ppc.dim'
c
      dimension rss(NSYSMX),x(5)
      dimension rss1(NSYSMX),xlcsec(5,5)

      COMMON/NLNR /sc(5,NSYSMX),taux(NSYSMX,2),nrsx(NSYSMX),
     #             ti(5),fp1(5,5),sc1(5,NSYSMX)
c
      if(irecover.eq.0) then
      do i=1,5
      do j=1,5
      xlsec(i,j)=0.
      enddo
      enddo
      endif
c
       do kk=1,npts3
c
cw      write(*,'(1H+,a,i4,a,i4)') ' Z = ',kk,'   OUT OF ',npts
c
       do jj=1,npts2
       do ii=1,npts1
c 
       nph=jphase(ii,jj,kk)
      DO i=1,5
        x(i)=sg(i,ii,jj,kk)
      ENDDO
c
      DO is=1,nsyst(1)
calb        nrsx(is)  =nrsmin
        nrsx(is)  =nrs(is,nph)
        taux(is,1)=crss(is,1,ii,jj,kk)
        taux(is,2)=crss(is,2,ii,jj,kk)
        DO j=1,5
          sc(j,is)=sch(j,is,ii,jj,kk)
        ENDDO
      ENDDO

C     GET RESOLVED SHEAR STRESSES 'rss' AND SHEAR RATES 'gamdot'.
C     SIGN(GAMDOT)=SIGN(RSS).
C     NRS CAN BE EVEN OR ODD.
C     RSS1 IS ALWAYS > 0 AND IS USED TO CALCULATE VISCOUS COMPLIANCE.

      do i=1,nsyst(1)
        rss(i)=sc(1,i)*x(1)+sc(2,i)*x(2)+sc(3,i)*x(3)+sc(4,i)*x(4)+
     #         sc(5,i)*x(5)
        isign=1
        if(rss(i).lt.0.) isign=2
        rss(i)=rss(i)/taux(i,isign)
        rss1(i)=gamd0(i,1)*abs(rss(i)**(nrsx(i)-1))/taux(i,isign)
        gamdot(i,ii,jj,kk)=
     #          gamd0(i,1)*abs(rss(i)**nrsx(i))*sign(1.,rss(i))
      enddo
C
C     CALCULATE MACRO TANGENT STIFFNESS
C
      if(irecover.eq.0) then
c
      DO I=1,5
      DO J=1,5
      XLCSEC(I,J)=0.
        DO K1=1,NSYST(1)
          XLCSEC(I,J)=XLCSEC(I,J)+SC(I,K1)*SC(J,K1)*RSS1(K1)
        ENDDO
      ENDDO
      ENDDO

      CALL LU_INVERSE(XLCSEC,5)

      DO I=1,5
      DO J=1,5
          XLSEC(I,J)=XLSEC(I,J)+xlfac0*XLCSEC(I,J)*WGT
      ENDDO
      ENDDO
c
      endif
c
      ENDDO
      ENDDO
      ENDDO
c
      RETURN
      END

C ***************************************************************************
C
C     SUBROUTINE SNLNR      --->      VERSION 11/MAR/2000
C
C     GIVEN AN INPUT GRAIN STRESS 'X' SOLVES THE VISCOPLASTIC EQUATION
C     USING NEWTON-RAPHSON LINEARIZATION AND ITERATING UNTIL CONVERGENCE.
C     IPH: LABEL OF CRYSTAL PHASE AND NOT OF ELEMENT.
C     CORRECTIONS TO 'X' AND RELAXATION OF TOLERANCE ELIMINATED ON FEB/2000
C
C ***************************************************************************

      SUBROUTINE SNLNR_INITIAL (x,kmax,eps,taulim,iii,jjj,kkk)

      INCLUDE 'ppc.dim'

      dimension fp(5,5),f(5),x(5),xold(5),xori(5),workx(5)
      dimension rss(NSYSMX),gd(NSYSMX)
      common/nlnr/sc(5,NSYSMX),taux(NSYSMX,2),nrsx(NSYSMX),
     #            ti(5),fp1(5,5),sc1(5,NSYSMX)

      coef=0.2

      do i=1,5
        xori(i)=x(i)
      enddo

      DO 710 K=1,KMAX

C     RESOLVED SHEARS CALCULATION & OUTSIDE Y.S. ERROR MANAGING BLOCK
        DO II=1,NSYST(1)
          rss(ii)=sc(1,ii)*x(1)+sc(2,ii)*x(2)+sc(3,ii)*x(3)
     #           +sc(4,ii)*x(4)+sc(5,ii)*x(5)
          isign=1
          if(rss(ii).lt.0.) isign=2
          rss(ii)=rss(ii)/taux(ii,isign)
          if(abs(rss(ii)).lt.1.e-10) rss(ii)=0.

          if(abs(rss(ii)).gt.taulim) then
            do i=1,5
              x(i)=xold(i)+coef*(x(i)-xold(i))
            enddo
            go to 710
          endif
        ENDDO

C      NRS MAY BE EVEN OR ODD:
C      GD ALWAYS > 0 -> GD's ARE USED TO GET DERIVATIVES TO BUILD THE
C      COEFFICIENT MATRIX FOR N-R METHOD IN WORKX CALCULATION (INDEPENDENT
C      TERM FOR N-R METHOD)   RSS*GD=GAMDOT -> SIGN(RSS*GD)=SIGN(RSS)

        do iw=1,nsyst(1)
          if(nrsx(iw).ne.1) then
            gd(iw)=gamd0(iw,1)*abs(rss(iw)**(nrsx(iw)-1))
          else
            gd(iw)=gamd0(iw,1)
          endif
        enddo
c
      do 30 i=1,5
      WORKX(I)=-TI(I)
      DO 30 J=1,5
30    WORKX(I)=WORKX(I)+FP1(I,J)*X(J)
      do 40 i=1,5
      do 40 ii=1,nsyst(1)
40    workx(i)=workx(i)+sc(i,ii)*rss(ii)*gd(ii)
      do 35 j=1,5
      do 35 i=1,5
35    fp(i,j)=-FP1(I,J)
      do 45 ii=1,nsyst(1)
      isign=1
      if(rss(ii).lt.0) isign=2
      do 45 i=1,5
      fp(i,1)=fp(i,1)-nrsx(ii)*sc(i,ii)*sc(1,ii)*gd(ii)/taux(ii,isign)
      fp(i,2)=fp(i,2)-nrsx(ii)*sc(i,ii)*sc(2,ii)*gd(ii)/taux(ii,isign)
      fp(i,3)=fp(i,3)-nrsx(ii)*sc(i,ii)*sc(3,ii)*gd(ii)/taux(ii,isign)
      fp(i,4)=fp(i,4)-nrsx(ii)*sc(i,ii)*sc(4,ii)*gd(ii)/taux(ii,isign)
      fp(i,5)=fp(i,5)-nrsx(ii)*sc(i,ii)*sc(5,ii)*gd(ii)/taux(ii,isign)
45    continue
      do i=1,5
        f(i)=workx(i)
        xold(i)=x(i)
      enddo

C     SOLVE THE LINEAR SYSTEM

      CALL LU_EQSYSTEM(FP,F,5,IER)

      if(ier.eq.1) then
c
        write(*,*)
     #'SINGULAR SYST - NR ROUTINE CANNOT SOLVE POINT',iii,jjj,kkk
        write(*,*) 'THE INITIAL VALUE OF STRESS IS ASSUMED'
c
        write(25,*)
     #' SINGULAR SYST - NR ROUTINE CANNOT SOLVE POINT',iii,jjj,kkk
        write(25,*) ' THE INITIAL VALUE OF STRESS IS ASSUMED'
c
          do ih=1,5
            x(ih)=xori(ih)
          enddo
          return
      endif

      do i=1,5
        x(i)=xold(i)+f(i)
      enddo
      rerror=tmismatch(x,xold,5,1)
      if(rerror.lt.eps) return

710   continue      ! END OF MASTER DO
C ****************************************************************

       write(*,*)
     #' ITMAX REACHED - NR ROUTINE CANNOT SOLVE POINT',iii,jjj,kkk
       write(*,*) ' THE INITIAL VALUE OF STRESS IS ASSUMED'
c
       write(25,*)
     #' ITMAX REACHED - NR ROUTINE CANNOT SOLVE POINT',iii,jjj,kkk
       write(25,*) ' THE INITIAL VALUE OF STRESS IS ASSUMED'
c
        do ih=1,5
          x(ih)=xori(ih)
        enddo

      RETURN
      END
c
      SUBROUTINE VISC_FFT

      INCLUDE 'ppc.dim'
c
c      dimension dtilde(5,npts,npts,npts)
c
      dimension x(5),dg(5),edot(5),dsg(5),ddg(5)

      COMMON/NLNR /sc(5,NSYSMX),taux(NSYSMX,2),nrsx(NSYSMX),
     #             ti(5),fp1(5,5),sc1(5,NSYSMX)
c
C     EMPIRIC ALGORITHM TO GET TAULIM FOR NR SUBROUTINE (RL: 01/FEB/00)

      taulim=2.*(tnorm(dbar,5,1)/gamd0(1,1))**(1./nrsmin)
      if(taulim.lt.2.) taulim=2.
c
       ERRD=0.
       ERRS=0.
c
       do kk=1,npts3
c
cw       write(*,'(1H+,a,i4,a,i4)') ' Z = ',kk,'   OUT OF ',npts
c
       do jj=1,npts2
       do ii=1,npts1
c 
       nph=jphase(ii,jj,kk)
        DO I=1,5
        DG(I)=DBAR(I)+DTILDE(I,II,JJ,KK)
        ENDDO

        DO I=1,5
          TI(I)=SG(I,II,JJ,KK)
          DO J=1,5
            TI(I)=TI(I)+XLSEC(I,J)*DG(J)
            FP1(I,J)=(I/J)*(J/I)
          ENDDO
        ENDDO

C     COPY CRITICAL STRESS, RATE SENSIT, SCHMID TENSOR INTO AUXILIAR ARRAYS

      DO is=1,nsyst(1)
c
calb        nrsx(is)  =nrsmin
        nrsx(is)  =nrs(is,nph)
        taux(is,1)=crss(is,1,ii,jj,kk)
        taux(is,2)=crss(is,2,ii,jj,kk)
c
        DO I=1,5
          sc(i,is)=sch(i,is,ii,jj,kk)
        ENDDO
c
        DO I=1,5
          sc1(i,is)=0.
          do j=1,5
          sc1(i,is)=sc1(i,is)+xlsec(i,j)*sc(j,is)
          enddo
        ENDDO
c
      ENDDO

C     USE STRY AS A STARTING VALUE TO SOLVE NONLINEAR SYSTEM

      DO i=1,5
        x(i)=sg(i,ii,jj,kk)
      ENDDO

C     MAKE SURE STRY IS NOT OUTSIDE THE YIELD SURFACE.
C     CORRECTS THE NORM IF IT EXCEEDS THE Y.S. BY MORE THAN 10%

  100 taumax=0.
      do ik=1,nsyst(1)
        htau=0.
        do jk=1,5
          htau=htau+x(jk)*sc(jk,ik)
        enddo
        isign=1
        if(htau.lt.0.) isign=2
        htau=htau/taux(ik,isign)
        if(abs(htau).gt.taumax) taumax=abs(htau)
      enddo

      if(taumax. lt. 1.e-10) then
        write(*,'('' taumax.lt.1e-10 inside subroutine VISC'')')
calb        stop
      endif
c
c     if(taumax .gt. taulim) then
c
c     BACK TO 1.1 (RL: 9/FEB/00)
c
      if(taumax .gt. 1.1) then
        tcoef=1./taumax
        do i=1,5
          x(i)=x(i)*tcoef
        enddo
      endif

C     CALLS Newton-Raphson SUBROUTINE TO CALCULATE GRAIN STRESS

      itmx=1000
      eps=5.e-04

cc      itmx=10000
cc      eps=1.e-03

      CALL SNLNR (x,itmx,eps,taulim,ii,jj,kk)

cc      do i=1,5
cc        sg(i,ii,jj,kk)=x(i)
cc      enddo

C     GET RESOLVED SHEAR STRESSES 'rss' AND SHEAR RATES 'gamdot'.
C            SIGN(GAMDOT)=SIGN(RSS), NRS CAN BE EVEN OR ODD

      do i=1,5
       edot(i)=0.
      enddo
c
      DO IS=1,NSYST(1)
c
        rss=sc(1,is)*x(1)+sc(2,is)*x(2)+sc(3,is)*x(3)
     #         +sc(4,is)*x(4)+sc(5,is)*x(5)
        isign=1
        if(rss.lt.0.) isign=2
        rss=rss/taux(is,isign)
        gamdot(is,ii,jj,kk)=
     #          gamd0(is,1)*abs(rss**nrsx(is))*sign(1.,rss)
c
        do i=1,5
        edot(i)=edot(i)+sc(i,is)*gamdot(is,ii,jj,kk)
        enddo
c
      ENDDO
c
c     UPDATE SG (LAGRANGE MULTIPLIER)
c
      DO I=1,5
       DDG(I)=DG(I)-EDOT(I)
       DSG(I)=0.
       DO J=1,5
        DSG(I)=DSG(I)+XLSEC(I,J)*(DG(J)-EDOT(J))
       ENDDO
       SG(I,II,JJ,KK)=SG(I,II,JJ,KK)+DSG(I)
      ENDDO
c
      ERRD=ERRD+TNORM(DDG,5,1)*WGT
      ERRS=ERRS+TNORM(DSG,5,1)*WGT
c
      ENDDO
      ENDDO
      ENDDO
c
      RETURN
      END
c
C ***************************************************************************
C
C     SUBROUTINE SNLNR      --->      VERSION 11/MAR/2000
C
C     GIVEN AN INPUT GRAIN STRESS 'X' SOLVES THE VISCOPLASTIC EQUATION
C     USING NEWTON-RAPHSON LINEARIZATION AND ITERATING UNTIL CONVERGENCE.
C     IPH: LABEL OF CRYSTAL PHASE AND NOT OF ELEMENT.
C     CORRECTIONS TO 'X' AND RELAXATION OF TOLERANCE ELIMINATED ON FEB/2000
C
C ***************************************************************************

      SUBROUTINE SNLNR (x,kmax,eps,taulim,iii,jjj,kkk)

      INCLUDE 'ppc.dim'

      dimension fp(5,5),f(5),x(5),xold(5),xori(5),workx(5)
      dimension rss(NSYSMX),gd(NSYSMX)
      common/nlnr/sc(5,NSYSMX),taux(NSYSMX,2),nrsx(NSYSMX),
     #            ti(5),fp1(5,5),sc1(5,NSYSMX)

      coef=0.2

      do i=1,5
        xori(i)=x(i)
      enddo

      DO 710 K=1,KMAX

C     RESOLVED SHEARS CALCULATION & OUTSIDE Y.S. ERROR MANAGING BLOCK
        DO II=1,NSYST(1)
          rss(ii)=sc(1,ii)*x(1)+sc(2,ii)*x(2)+sc(3,ii)*x(3)
     #           +sc(4,ii)*x(4)+sc(5,ii)*x(5)
          isign=1
          if(rss(ii).lt.0.) isign=2
          rss(ii)=rss(ii)/taux(ii,isign)
          if(abs(rss(ii)).lt.1.e-10) rss(ii)=0.

          if(abs(rss(ii)).gt.taulim) then
            do i=1,5
              x(i)=xold(i)+coef*(x(i)-xold(i))
            enddo
            go to 710
          endif
        ENDDO

C      NRS MAY BE EVEN OR ODD:
C      GD ALWAYS > 0 -> GD's ARE USED TO GET DERIVATIVES TO BUILD THE
C      COEFFICIENT MATRIX FOR N-R METHOD IN WORKX CALCULATION (INDEPENDENT
C      TERM FOR N-R METHOD)   RSS*GD=GAMDOT -> SIGN(RSS*GD)=SIGN(RSS)

        do iw=1,nsyst(1)
          if(nrsx(iw).ne.1) then
            gd(iw)=gamd0(iw,1)*abs(rss(iw)**(nrsx(iw)-1))
          else
            gd(iw)=gamd0(iw,1)
          endif
        enddo
c
      do 30 i=1,5
30    WORKX(I)=X(I)-TI(I)

c
      do 40 i=1,5
      do 40 ii=1,nsyst(1)
40    workx(i)=workx(i)+sc1(i,ii)*rss(ii)*gd(ii)
c
      do 35 j=1,5
      do 35 i=1,5
35    fp(i,j)=-FP1(I,J)
c
      do 45 ii=1,nsyst(1)
      isign=1
      if(rss(ii).lt.0) isign=2
      do 45 i=1,5
      fp(i,1)=fp(i,1)-nrsx(ii)*sc1(i,ii)*sc(1,ii)*gd(ii)/taux(ii,isign)
      fp(i,2)=fp(i,2)-nrsx(ii)*sc1(i,ii)*sc(2,ii)*gd(ii)/taux(ii,isign)
      fp(i,3)=fp(i,3)-nrsx(ii)*sc1(i,ii)*sc(3,ii)*gd(ii)/taux(ii,isign)
      fp(i,4)=fp(i,4)-nrsx(ii)*sc1(i,ii)*sc(4,ii)*gd(ii)/taux(ii,isign)
      fp(i,5)=fp(i,5)-nrsx(ii)*sc1(i,ii)*sc(5,ii)*gd(ii)/taux(ii,isign)
45    continue
c
      do i=1,5
        f(i)=workx(i)
        xold(i)=x(i)
      enddo

C     SOLVE THE LINEAR SYSTEM

      CALL LU_EQSYSTEM(FP,F,5,IER)

      if(ier.eq.1) then
c
        write(25,*)
     #'SINGULAR SYST - NR ROUTINE CANNOT SOLVE POINT',iii,jjj,kkk
        write(25,*) 'THE INITIAL VALUE OF STRESS IS ASSUMED'
c
        write(25,*)
     #'SINGULAR SYST - NR ROUTINE CANNOT SOLVE POINT',iii,jjj,kkk
        write(25,*) 'THE INITIAL VALUE OF STRESS IS ASSUMED'
c
          do ih=1,5
            x(ih)=xori(ih)
          enddo
          return
      endif

      do i=1,5
        x(i)=xold(i)+f(i)
      enddo
      rerror=tmismatch(x,xold,5,1)
      if(rerror.lt.eps) return

710   continue      ! END OF MASTER DO
C ****************************************************************

c       write(*,*)
c     #'ITMAX REACHED - NR ROUTINE CANNOT SOLVE POINT',iii,jjj,kkk,eps
c       write(*,*) 'THE INITIAL VALUE OF STRESS IS ASSUMED'
c
c        do ih=1,5
c          x(ih)=xori(ih)
c        enddo

cc       write(*,*) 'ITMAX REACHED',iii,jjj,kkk,rerror
c
       write(25,*) ' ITMAX REACHED',iii,jjj,kkk,rerror
c
      RETURN
      END

C *************************************************************************
      SUBROUTINE LU_INVERSE(A,N)
c
c   INVERTS A MATRIX USING LU DECOMPOSITION
c
      DIMENSION A(N,N),Y(N,N),INDX(N)
c
      DO I=1,N
        DO J=1,N
          Y(I,J)=0.
        ENDDO
        Y(I,I)=1.
      ENDDO
c
      CALL LUDCMP(A,N,N,INDX,D,ISINGULAR)
c
      DO J=1,N
        CALL LUBKSB(A,N,N,INDX,Y(1,J))
      ENDDO
c
      DO I=1,N
      DO J=1,N
       A(I,J)=Y(I,J)
      ENDDO
      ENDDO
c
      RETURN
      END
c
c *************************************************************************
c
      SUBROUTINE LU_EQSYSTEM(A,B,N,ISINGULAR)
c
c     SOLVES A*X=B USING LU DECOMPOSITION
c
      DIMENSION A(N,N),B(N),INDX(N)
c
      CALL LUDCMP(A,N,N,INDX,D,ISINGULAR)
c
      IF(ISINGULAR.EQ.1) RETURN
c
      CALL LUBKSB(A,N,N,INDX,B)
c
      RETURN
      END
c
C *****************************************************************************
c
      SUBROUTINE ludcmp(a,n,np,indx,d,isingular)
      INTEGER n,np,indx(n),NMAX
c      REAL d,a(np,np),TINY
c      PARAMETER (NMAX=500,TINY=1.0e-20)
      REAL d,a(np,np)
      PARAMETER (NMAX=500)
      INTEGER i,imax,j,k,isingular
      REAL aamax,dum,sum,vv(NMAX)
      d=1.
      do 12 i=1,n
        aamax=0.
        do 11 j=1,n
          if (abs(a(i,j)).gt.aamax) aamax=abs(a(i,j))
11      continue
c
c        if (aamax.eq.0.) pause 'singular matrix in ludcmp'
c
        if(aamax.eq.0.) then
        isingular=1
        return
        endif
c
        vv(i)=1./aamax
12    continue
      do 19 j=1,n
        do 14 i=1,j-1
          sum=a(i,j)
          do 13 k=1,i-1
            sum=sum-a(i,k)*a(k,j)
13        continue
          a(i,j)=sum
14      continue
        aamax=0.

        do 16 i=j,n
          sum=a(i,j)
          do 15 k=1,j-1
            sum=sum-a(i,k)*a(k,j)
15        continue
          a(i,j)=sum
          dum=vv(i)*abs(sum)
          if (dum.ge.aamax) then
            imax=i
            aamax=dum
          endif
16      continue
        if (j.ne.imax)then
          do 17 k=1,n
            dum=a(imax,k)
            a(imax,k)=a(j,k)
            a(j,k)=dum
17        continue
          d=-d
          vv(imax)=vv(j)
        endif
        indx(j)=imax
c
c        if(a(j,j).eq.0.) a(j,j)=TINY
c
        if(a(j,j).eq.0.) then
        isingular=1
        return
        endif
c
        if(j.ne.n)then
          dum=1./a(j,j)
          do 18 i=j+1,n
            a(i,j)=a(i,j)*dum
18        continue
        endif
19    continue
c
      isingular=0
c
      return
      END
c
c *****************************************************************************
c
      SUBROUTINE lubksb(a,n,np,indx,b)
      INTEGER n,np,indx(n)
      REAL a(np,np),b(n)
      INTEGER i,ii,j,ll
      REAL sum
      ii=0
      do 12 i=1,n
        ll=indx(i)
        sum=b(ll)
        b(ll)=b(i)
        if (ii.ne.0)then
          do 11 j=ii,i-1
            sum=sum-a(i,j)*b(j)
11        continue
        else if (sum.ne.0.) then
          ii=i
        endif
        b(i)=sum
12    continue
      do 14 i=n,1,-1
        sum=b(i)
        do 13 j=i+1,n
          sum=sum-a(i,j)*b(j)
13      continue
        b(i)=sum/a(i,i)
14    continue
      return
      END
c
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
C     FUNCTION TMISMATCH   ---->   VERSION OF 27/DEC/98
C
C     CALCULATES RELATIVE DIFFERENCE BETWEEN TWO NROWSxNCOLS MATRICES
C     THE DIFFERENCE IS RELATIVE TO THE NORM OF THE ARITHMETIC AVERAGE
C     OF BOTH DATA.
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION tmismatch (v1,v2,nrows,ncols)
C
COLD      DIMENSION v1(nrows,ncols),v2(nrows,ncols)
COLD      DIMENSION v_dif(6,6),v_ave(6,6)
      DIMENSION v1(36),v2(36)
      DIMENSION v_dif(36),v_ave(36)
C
      do i=1,nrows*ncols
        v_dif(i)=v1(i)-v2(i)
        v_ave(i)=0.5d0*(v1(i)+v2(i))
      enddo
      tmismatch=tnorm(v_dif,nrows,ncols)/tnorm(v_ave,nrows,ncols)
C
      RETURN
      END

C
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
C     FUNCTION TNORM   ---->   VERSION OF 27/DEC/98
C
C     CALCULATES THE NORM OF A NROWSxNCOLS-MATRIX (NROWS,NCOLS =< 6)
C ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C
      FUNCTION tnorm(v,nrows,ncols)
C
COLD      DIMENSION v(nrows,ncols)
      DIMENSION v(36)
C
      tnorm=0.d0
      do i=1,nrows*ncols
        tnorm=tnorm+v(i)*v(i)
      enddo
      tnorm=sqrt(tnorm)
C
      RETURN
      END
c
      SUBROUTINE MINV (A,N,D,L,M)
c
      DIMENSION A(*),L(*),M(*)
C
C       SEARCH FOR LARGEST ELEMENT
C
      D=1.d0
      NK=-N
      DO 180 K=1,N
      NK=NK+N
      L(K)=K
      M(K)=K
      KK=NK+K
      BIGA=A(KK)
      DO 20 J=K,N
      IZ=N*(J-1)
      DO 20 I=K,N
      IJ=IZ+I
      IF (ABS(BIGA)-ABS(A(IJ))) 10,20,20
   10 BIGA=A(IJ)
      L(K)=I
      M(K)=J
   20 CONTINUE
C
C       INTERCHANGE ROWS
C
      J=L(K)
      IF (J-K) 50,50,30
   30 KI=K-N
      DO 40 I=1,N
      KI=KI+N
      HOLD=-A(KI)
      JI=KI-K+J
      A(KI)=A(JI)
   40 A(JI)=HOLD
C
C       INTERCHANGE COLUMNS
C
   50 I=M(K)
      IF (I-K) 80,80,60
   60 JP=N*(I-1)
      DO 70 J=1,N
      JK=NK+J
      JI=JP+J
      HOLD=-A(JK)
      A(JK)=A(JI)
   70 A(JI)=HOLD
C
C       DIVIDE COLUMN BY MINUS PIVOT (BIGA)
C
   80 IF (ABS(BIGA).LT.1.e-10) THEN
   90 D=0.0
      RETURN
      ENDIF
  100 DO 120 I=1,N
      IF (I-K) 110,120,110
  110 IK=NK+I
      A(IK)=A(IK)/(-BIGA)
  120 CONTINUE
C
C       REDUCE MATRIX
C
      DO 150 I=1,N
      IK=NK+I
      HOLD=A(IK)
      IJ=I-N
      DO 150 J=1,N
      IJ=IJ+N
      IF (I-K) 130,150,130
  130 IF (J-K) 140,150,140
  140 KJ=IJ-I+K
      A(IJ)=HOLD*A(KJ)+A(IJ)
  150 CONTINUE
C
C       DIVIDE ROW BY PIVOT
C
      KJ=K-N
      DO 170 J=1,N
      KJ=KJ+N
      IF (J-K) 160,170,160
  160 A(KJ)=A(KJ)/BIGA
  170 CONTINUE
C
C       PRODUCT OF PIVOTS
C
      D=D*BIGA
C
C       REPLACE PIVOT BY RECIPROCAL
C
      A(KK)=1.d0/BIGA
  180 CONTINUE
C
C       FINAL ROW AND COLUMN INTERCHANGE
C
      K=N
  190 K=(K-1)
      IF (K) 260,260,200
  200 I=L(K)
      IF (I-K) 230,230,210
  210 JQ=N*(K-1)
      JR=N*(I-1)
      DO 220 J=1,N
      JK=JQ+J
      HOLD=A(JK)
      JI=JR+J
      A(JK)=-A(JI)
  220 A(JI)=HOLD
  230 J=M(K)
      IF (J-K) 190,190,240
  240 KI=K-N
      DO 250 I=1,N
      KI=KI+N
      HOLD=A(KI)
      JI=KI-K+J
      A(KI)=-A(JI)
  250 A(JI)=HOLD
      GO TO 190
  260 RETURN
      END
c
cc      subroutine equilibrium(snormfft,snormfftim,nn,err2mod)
ccc
cc      INCLUDE 'ppc.dim'
ccc
cc      dimension aux5(5),aux55(5,5),aux3333(3,3,3,3)
cc      dimension sigma(3,3),xk(3),gamma(3)
cc      dimension nn(3)
ccc
cc      write(*,*) 'CHECKING EQUILIBRIUM ...'
cccw      write(*,*)
ccc
cc      err2=0.
cc      err2im=0.
ccc
cc      do 1 kzz=1,npts3
ccc
ccc      if(kzz/1*1.eq.kzz) write(*,'(a,i4)') ' K= ',kzz
ccc 
cccw      write(*,'(1H+,a,i4,a,i4)') ' Z = ',kzz,'   OUT OF ',npts
ccc
cc      do 1 kyy=1,npts2
cc      do 1 kxx=1,npts1
ccc
cc      if(kxx.le.npts1/2) kx=kxx-1
cc      if(kxx.gt.npts1/2) kx=kxx-npts1-1
ccc
cc      if(kyy.le.npts2/2) ky=kyy-1
cc      if(kyy.gt.npts2/2) ky=kyy-npts2-1
ccc
cc      if(kzz.le.npts3/2) kz=kzz-1
cc      if(kzz.gt.npts3/2) kz=kzz-npts3-1
ccc
cc      xk(1)=kx/(delt(1)*nn(1))
cc      xk(2)=ky/(delt(2)*nn(2))
ccc
cc      if(npts3.gt.1) then
cc      xk(3)=kz/(delt(3)*nn(3))
cc      else
cc      xk(3)=0.
cc      endif
ccc
ccc      xknorm=sqrt(xk(1)**2+xk(2)**2+xk(3)**2)
ccc
ccc      if (xknorm.ne.0.) then
ccc      do i=1,3
ccc      xk(i)=xk(i)/xknorm
ccc      enddo
ccc      endif
ccc
cc      do i=1,5
cc      aux5(i)=sgfft(i,kxx,kyy,kzz)
cc      enddo
ccc
cc      call chg_basis(aux5,sigma,aux55,aux3333,1)
ccc
cc      do i=1,3
cc      gamma(i)=-xk(i)*press(kxx,kyy,kzz)
cc      do j=1,3
cc      gamma(i)=gamma(i)+sigma(i,j)*xk(j)
cc      enddo
cc      enddo
ccc
cc      do i=1,3
cc      err2=err2+(gamma(i)**2)*wgt
cc      enddo
ccc
cc      do i=1,5
cc      aux5(i)=sgfftim(i,kxx,kyy,kzz)
cc      enddo
ccc
cc      call chg_basis(aux5,sigma,aux55,aux3333,1)
ccc
cc      do i=1,3
cc      gamma(i)=-xk(i)*pressim(kxx,kyy,kzz)
cc      do j=1,3
cc      gamma(i)=gamma(i)+sigma(i,j)*xk(j)
cc      enddo
cc      enddo
ccc
cc      do i=1,3
cc      err2im=err2im+(gamma(i)**2)*wgt
cc      enddo
ccc
cc1     continue
ccc
cc      err2=sqrt(err2)/snormfft
ccc      err2im=sqrt(err2im)/snormfftim
cc      err2im=sqrt(err2im)/snormfft
cc      err2mod=sqrt(err2**2+err2im**2)
ccc
ccc      write(*,*) 'REAL ERROR IN EQUILIBRIUM =',err2
ccc      write(*,*) 'IMAG ERROR IN EQUILIBRIUM =',err2im
ccc
cc      write(*,*) 'ERROR IN EQUILIBRIUM =',err2mod
ccc      write(20,*) 'ERROR IN EQUILIBRIUM =',err2
ccc
cc      return
cc      end
c
      subroutine statactiv

      include 'ppc.dim'
      dimension gtot(nmodmx,npts1,npts2,npts3)
      dimension gavgr(npts1,npts2,npts3)

      iph=1         !simplified case. to be changed in case of 2 phases

      do 4000 ii=1,npts1
      do 4000 jj=1,npts2
      do 4000 kk=1,npts3

      is=0
      gavgr(ii,jj,kk)=0.

      do 4001 k1=1,nmodes(iph)
        gtot(k1,ii,jj,kk)=0.
        do 4002 k2=1,nsm(k1,iph)
          is=is+1
          gabs=abs(gamdot(is,ii,jj,kk))
          gtot(k1,ii,jj,kk)=gtot(k1,ii,jj,kk)+gabs
4002    continue 
        gtot(k1,ii,jj,kk)=gtot(k1,ii,jj,kk)*tdot
        gavgr(ii,jj,kk)=gavgr(ii,jj,kk)+gtot(k1,ii,jj,kk)
4001  continue

4000  continue

      gavpc=0.
      do k1=1,nmodes(iph)
        gavmod(k1)=0.
      enddo


      do 4005 ii=1,npts1
      do 4005 jj=1,npts2
      do 4005 kk=1,npts3

      do 4006 k1=1,nmodes(iph)
      gmodgr(k1,ii,jj,kk)=gtot(k1,ii,jj,kk)/gavgr(ii,jj,kk)
4006  gavmod(k1)=gavmod(k1)+gtot(k1,ii,jj,kk)*wgt
      gavpc=gavpc+gavgr(ii,jj,kk)*wgt
      
4005  continue

      do k1=1,nmodes(iph)
        gavmod(k1)=gavmod(k1)/gavpc
      enddo

c      write(35,*) ((gavmod(k1)),k1=1,nmodes(iph))

      return
      end
c
      subroutine second_order(kw)
c
      include 'ppc.dim'
c
      dimension sgg(3,3)
      dimension dgg(3,3)
      dimension sg5(5),sgx(3,3)
      dimension dg5(5),dgx(3,3)
      dimension aux55(5,5),aux3333(3,3,3,3)
c
      IF(KW.EQ.2) THEN
c
      open(30,file='g0.out',status='unknown')
      open(31,file='g10.out',status='unknown')
      open(32,file='g20.out',status='unknown')
      open(33,file='g30.out',status='unknown')
      open(34,file='g40.out',status='unknown')
      open(35,file='g50.out',status='unknown')
      open(36,file='g60.out',status='unknown')
      open(37,file='g70.out',status='unknown')
      open(38,file='g80.out',status='unknown')
      open(39,file='g90.out',status='unknown')
c
      do ifil=30,39
      write(ifil,*) 
     # '     s11           s22         s33           s23          s13    
     #     s12'
      enddo
c
      ENDIF
c
c     OVERALL STRESS
c
      do ii=1,5
      sav(ii)=0.
      do k=1,npts3
      do j=1,npts2
      do i=1,npts1
      sav(ii)=sav(ii)+sg(ii,i,j,k)*wgt
      enddo
      enddo
      enddo
      enddo
c
      call chg_basis(sav,sdeviat,aux55,aux3333,1)
c
c      effective viscosity in 2D antiplane case
c
      if(dsim(1,3).ne.0.) then
      effmu13=sdeviat(1,3)/(2*dsim(1,3))
      effmu23=sdeviat(2,3)/(2*dsim(1,3))
c      else
c      effmu13=0.
      endif
c
      if(dsim(2,3).ne.0.) then
      effmu13=sdeviat(1,3)/(2*dsim(2,3))
      effmu23=sdeviat(2,3)/(2*dsim(2,3))
c      else
c      effmu23=0.
      endif
c
      svm=0.
      do i=1,3
      do j=1,3
      svm=svm+sdeviat(i,j)*sdeviat(i,j)
      enddo
      enddo
      svm=sqrt(3./2.*svm)
c
      sinhom=0.
      dinhom=0.
      wdot=0.

      DO IGR=1,NOR
c
       IF(KW.GE.1) THEN
c
       do ii=1,3
       do jj=1,3
        sgg(ii,jj)=0.
        dgg(ii,jj)=0. 
       enddo
       enddo
c
       s232(igr)=0.
       s132(igr)=0.
c
       seq2(igr)=0.
       deq2(igr)=0.
c
       gd1(igr)=0.
       gd12(igr)=0.
       gd2(igr)=0.
       gd22(igr)=0.
c
       ta1(igr)=0.
       ta12(igr)=0.
       ta2(igr)=0.
       ta22(igr)=0.
c
       ENDIF
c
       iw=0

       do i=1,npts1
       do j=1,npts2
       do k=1,npts3

        if(jgrain(i,j,k).eq.igr) then

         iw=iw+1

       do ii=1,5
       sg5(ii)=sg(ii,i,j,k)
       dg5(ii)=dbar(ii)+dtilde(ii,i,j,k)
       enddo

       call chg_basis(sg5,sgx,aux55,aux3333,1)
       call chg_basis(dg5,dgx,aux55,aux3333,1)
c
       do ii=1,3
       do jj=1,3
       wdot=wdot+sgx(ii,jj)*dgx(ii,jj)
       enddo
       enddo
c
       IF(KW.GE.1) THEN
c  
       s23x=sgx(2,3)
       s232(igr)=s232(igr)+s23x**2
c
       s13x=sgx(1,3)
       s132(igr)=s132(igr)+s13x**2
c
       gd1x=gamdot(1,i,j,k)
       gd2x=gamdot(2,i,j,k)
       gd1(igr)=gd1(igr)+gd1x
       gd2(igr)=gd2(igr)+gd2x
       gd12(igr)=gd12(igr)+gd1x**2
       gd22(igr)=gd22(igr)+gd2x**2
c
       ta1x=0.
       ta2x=0.
       do ii=1,5
       ta1x=ta1x+sch(ii,1,i,j,k)*sg5(ii)
       ta2x=ta2x+sch(ii,2,i,j,k)*sg5(ii)
       enddo
c
       ta1(igr)=ta1(igr)+ta1x
       ta2(igr)=ta2(igr)+ta2x
       ta12(igr)=ta12(igr)+ta1x**2
       ta22(igr)=ta22(igr)+ta2x**2
c
       ENDIF
c
         do ii=1,3
         do jj=1,3
c
          IF(KW.GE.1) THEN
c
          sgg(ii,jj)=sgg(ii,jj)+sgx(ii,jj)
          dgg(ii,jj)=dgg(ii,jj)+dgx(ii,jj)

          seq2(igr)=seq2(igr)+sgx(ii,jj)**2
          deq2(igr)=deq2(igr)+dgx(ii,jj)**2
c
          ENDIF
c
          sinhom=sinhom+sgx(ii,jj)**2
          dinhom=dinhom+dgx(ii,jj)**2
c
         enddo
         enddo
c
      IF(KW.EQ.2) THEN
c
      if(mark(igr).eq.0) write(30,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.1) write(31,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.2) write(32,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.3) write(33,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.4) write(34,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.5) write(35,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.6) write(36,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.7) write(37,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.8) write(38,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
      if(mark(igr).eq.9) write(39,201) sgx(1,1),sgx(2,2),sgx(3,3),
     #  sgx(2,3),sgx(1,3),sgx(1,2)
c
      ENDIF

201   format (6(E12.5,1x))

      endif

      enddo
      enddo
      enddo
c
      IF(KW.GE.1) THEN
c
      s23(igr)=sgg(2,3)/iw
      s13(igr)=sgg(1,3)/iw
c
      seq(igr)=0.
      deq(igr)=0.
      do ii=1,3
      do jj=1,3
      seq(igr)=seq(igr)+(sgg(ii,jj)/iw)**2
      deq(igr)=deq(igr)+(dgg(ii,jj)/iw)**2
      enddo
      enddo

      seq(igr)=sqrt(3./2.*seq(igr))
      deq(igr)=sqrt(2./3.*deq(igr))
c
      s232(igr)=sqrt(s232(igr)/iw)
      s132(igr)=sqrt(s132(igr)/iw)
c
      seq2(igr)=sqrt(3./2.*seq2(igr)/iw)
      deq2(igr)=sqrt(2./3.*deq2(igr)/iw)
c
      gd1(igr)=gd1(igr)/iw
      gd2(igr)=gd2(igr)/iw
      gd12(igr)=sqrt(gd12(igr)/iw)
      gd22(igr)=sqrt(gd22(igr)/iw)
c
      ta1(igr)=ta1(igr)/iw
      ta2(igr)=ta2(igr)/iw
      ta12(igr)=sqrt(ta12(igr)/iw)
      ta22(igr)=sqrt(ta22(igr)/iw)
c
      ENDIF
c
      ENDDO
c
      sinhom=sqrt(3./2.*sinhom/(npts1*npts2*npts3))/svm
      dinhom=sqrt(2./3.*dinhom/(npts1*npts2*npts3))/dvm
c
      wdot=wdot/(npts1*npts2*npts3)
c    
c      close(31)
c      close(32)
c      close(33)
c
      IF(KW.EQ.2) THEN
c 
      do ifil=30,39
      close(ifil)
      enddo
c
      ENDIF
c
      return

      end
c
      subroutine update_orient
c
      include 'ppc.dim'
c
      dimension aa(3,3),distor(3,3)
      dimension dnsa(3),dbsa(3)
      dimension rotslip(3,3),rotloc(3,3),rot(3,3)
c
      RSLBAR=0.
      RLCBAR=0.
c
C     MASTER DO
C
      do 2000 k=1,npts3
      do 2000 j=1,npts2
      do 2000 i=1,npts1
c
c     LOCAL ROTATION RATE: ANTISYM(VELGRAD)
c
      do 100 ii=1,3
      do 100 jj=1,3
      rotloc(ii,jj)=(velgrad(ii,jj,i,j,k)-velgrad(jj,ii,i,j,k))/2.
  100 continue
c
c     SLIP ROTATION RATE
c
      do ii=1,3
      do jj=1,3
        aa(ii,jj)=ag(ii,jj,i,j,k)
        distor(ii,jj)=0.
      enddo
      enddo
c
        do is=1,nsyst(1)
c
        do ii=1,3
          dnsa(ii)=0.
          dbsa(ii)=0.
          do jj=1,3
            dnsa(ii)=dnsa(ii)+aa(ii,jj)*dnca(jj,is,1)
            dbsa(ii)=dbsa(ii)+aa(ii,jj)*dbca(jj,is,1)
          enddo
        enddo
c
        do ii=1,3
        do jj=1,3
         distor(ii,jj)=distor(ii,jj)+dbsa(ii)*dnsa(jj)*gamdot(is,i,j,k)
        enddo
        enddo
c
      enddo
C
      do ii=1,3
      do jj=1,3
        rotslip(ii,jj)=(distor(ii,jj)-distor(jj,ii))/2.
      enddo
      enddo
C
C     AVERAGE ROTATION RATE
C
      rslbar=rslbar+sqrt(rotslip(3,2)**2+rotslip(1,3)**2+
     #       rotslip(2,1)**2)*wgt
      rlcbar=rlcbar+sqrt(rotloc(3,2)**2+rotloc(1,3)**2+
     #       rotloc(2,1)**2)*wgt
c
c    TOTAL ROTATION
c
      do ii=1,3
      do jj=1,3
        rot(ii,jj)=(tomtot(ii,jj)+rotloc(ii,jj)-rotslip(ii,jj))*tdot
      enddo
      enddo
c
c     REORIENTATION
c
      call orient(aa,rot)
c
c     UPDATE ORIENTATION MATRIX
c
      do ii=1,3
      do jj=1,3
        ag(ii,jj,i,j,k)=aa(ii,jj)
      enddo
      enddo
c
2000  continue  
c
      write(*,*)
      write(*,*) 'AVERAGE PLASTIC ROTATION =',rslbar
      write(*,*) 'AVERAGE LOCAL ROTATION =',rlcbar
      write(*,*)
c
      RETURN
      END
c
      subroutine orient(a,c)
      dimension a(3,3),c(3,3),th2(3,3),v(3),vbar(3)
      dimension th(3,3)
      dimension rot(3,3),anew(3,3)

c     BUILD ROTATION TENSOR BASED ON RODRIGUES FORMULA

      v(1)=c(3,2)
      v(2)=c(1,3)
      v(3)=c(2,1)
      snorm=sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3))
      snorm1=tan(snorm/2.)
      if(snorm.gt.1.e-06) go to 97
      snorm=1.
97    do 20 i=1,3
      vbar(i)=snorm1*v(i)/snorm
20    continue
      snorm=vbar(1)*vbar(1)+vbar(2)*vbar(2)+vbar(3)*vbar(3)
      th(3,2)=vbar(1)
      th(1,3)=vbar(2)
      th(2,1)=vbar(3)
      th(2,3)=-vbar(1)
      th(3,1)=-vbar(2)
      th(1,2)=-vbar(3)
      do 40 i=1,3
40    th(i,i)=0.
      do 30 i=1,3
      do 30 j=1,3
      th2(i,j)=0.
      do 50 k=1,3
50    th2(i,j)=th2(i,j)+th(i,k)*th(k,j)
30    continue
      do 60 i=1,3
      do 60 j=1,3
60    rot(i,j)=(i/j)*(j/i)+2.*(th(i,j)+th2(i,j))/(1.+snorm)
      do 70 i=1,3
      do 70 j=1,3
      anew(i,j)=0.
      do 80 k=1,3
80    anew(i,j)=anew(i,j)+rot(i,k)*a(k,j)
70    continue
      do 90 i=1,3
      do 90 j=1,3
90    a(i,j)=anew(i,j)
      return
      end
