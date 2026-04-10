C*--------------------------------------------------------------------
C*    Basil / Sybil:   strain.f  1.1  1 October 1998
C*
C*    Copyright (c) 1997 by G.A. Houseman, T.D. Barr, & L.A. Evans
C*    See README file for copying and redistribution conditions.
C*--------------------------------------------------------------------
C
C    strain.f includes STRAIN STRAIX STRAIL STRAIC BINTIP STCOMP
C                      PAVG MATPRT MITPRT
C
      SUBROUTINE STRAIN(JELL,IQU,AMESH,BMESH,CMESH,DMESH,EMESH,
     1                  SE,ARGAN,BRGAN,HLENSC,BDEPSC,BIG,EX,EY,VHB,
     2                  UVP,LEM,NOR,SSQ,IELFIX,NE,NUP,NUVP,NP3,IVV,
     3                  NCOMP,ICR)
C
C    Routine to calculate different components of the strain or
C    stress tensor (or derived quantities) from the velocity field
C    Note that option numbers are tied to names of physical quantities
C    by definitions in strain.h
C
      DIMENSION AMESH(NUP),BMESH(NUP),CMESH(NUP),DMESH(NUP)
      DIMENSION EMESH(NUP)
      DIMENSION EX(NUP),EY(NUP),VHB(8,NE),UVP(NUVP)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION SSQ(NUP),IELFIX(NUP)
      REAL BY(3),CX(3),DNDP(84)
      REAL PNI(7,6), PLI(7,3), XX(3), YY(3)
      DOUBLE PRECISION TRI,DPX2,DPX3,DPY2,DPY3

      EPS=1.0E-4
      PI=3.141592653
      CALL LNCORD(PNI,PLI)
      GPEZERO=-0.5*ARGAN/(HLENSC*HLENSC)
C
C    Zero the mesh first
C
      DO 10 I=1,NUP
      EMESH(I)=0.0
      AMESH(I)=0.0
      BMESH(I)=0.0
      CMESH(I)=0.0
   10 DMESH(I)=0.0

      VF=1.0
      SEXP=SE
C    bigv should be the same as that calculated in VISK (cginit.f)
      BIGV = BIG/10000.0
C
C    Look at each element in turn
C
      DO 50 N=1,NE
C
C     Calculate the geometrical coefficients
C
      TRI=0.0
      DO 20 K=1,3
      K2=MOD(K,3)+1
      K3=MOD(K+1,3)+1
      LK1=NOR(LEM(K,N))
      LK2=NOR(LEM(K2,N))
      LK3=NOR(LEM(K3,N))
      X2=EX(LK2)
      X3=EX(LK3)
      Y2=EY(LK2)
      Y3=EY(LK3)
      XX(K)=EX(LK1)
      YY(K)=EY(LK1)
      BY(K)=Y2 - Y3
      CX(K)=X3 - X2
      DPX2=X2
      DPX3=X3
      DPY2=Y2
      DPY3=Y3
      TRI=TRI + (DPX2*DPY3 - DPX3*DPY2)
   20 CONTINUE
C
C    TRI is twice the area of the triangle element
C    Get the derivatives of the interpolation function at the nodes
C
      CALL DNCOM(5,BY,CX,DNDP)
C
C    activate variable viscosity depending on the value of IVV
C
      IF(IVV.GE.3)SEXP=VHB(8,N)
      DO 50 K=1,7
        IF(IVV.GE.1)VF=VHB(K,N)
C
C     Find the internal angle made by the boundaries of the
C     triangle at this node
C
      IF(K.LE.6)THEN
        LK=LEM(K,N)
        IF(LK.LT.0)LK=-LK
        ANGL=0.5
      END IF
      IF(K.LE.3)THEN
        K2=MOD(K,3)+1
        K3=MOD(K+1,3)+1
        DB2= BY(K2)*BY(K2) + CX(K2)*CX(K2)
        DA2= BY(K3)*BY(K3) + CX(K3)*CX(K3)
        IF (DB2.EQ.0.0.OR.DA2.EQ.0.0) THEN
          SANGL=TRI/EPS
        ELSE
          SANGL=TRI/SQRT(DB2*DA2)
        END IF
        CSANGL=1.0-SANGL*SANGL
        IF(CSANGL.LT.0.)CSANGL=0.0
        CSANGL=SQRT(CSANGL)
        IF (SANGL.EQ.0.0.AND.CSANGL.EQ.0.0) THEN
          ANGL = 0.0
        ELSE
          ANGL=0.5*ATAN2(SANGL,CSANGL)/PI
        END IF
      END IF
C
C    Location of node K
C
      IF(K.LE.3)THEN
        ILK=NOR(LEM(K,N))
        XP=EX(ILK)
        YP=EY(ILK)
      ELSE IF((K.GE.4).AND.(K.LE.6))THEN
        KK1=MOD(K+1,3)+1
        KK2=MOD(K+2,3)+1
        ILK1=NOR(LEM(KK1,N))
        ILK2=NOR(LEM(KK2,N))
        XP=0.5*(EX(ILK1)+EX(ILK2))
        YP=0.5*(EY(ILK1)+EY(ILK2))
      ELSE
        ILK1=NOR(LEM(1,N))
        ILK2=NOR(LEM(2,N))
        ILK3=NOR(LEM(3,N))
        XP=(EX(ILK1)+EX(ILK2)+EX(ILK3))/3.0
        YP=(EY(ILK1)+EY(ILK2)+EY(ILK3))/3.0
      END IF
C
C   Calculate the velocity derivatives at node K
C
   25 DUDX=0.0
      DUDY=0.0
      DVDX=0.0
      DVDY=0.0
      UUI=0.0
      VVI=0.0
C
C    Sum the interpolation functions
C
      DO 55 I=1,6
        NI=LEM(I,N)
        IF(NI.LT.0)NI=-NI
        KIN=(I-1)*14 + (K-1)*2 + 1
        PNIKI=PNI(K,I)
        UI=UVP(NI)
        VI=UVP(NI+NUP)
C
C    du/dx, du/dy, dv/dx, dv/dy at node K
C    UUI is used if NCOMP = -1 or 2,  VV if NCOMP=-1
C
        DUDX=DUDX + UI*DNDP(KIN)
        DUDY=DUDY + UI*DNDP(KIN+1)
        DVDX=DVDX + VI*DNDP(KIN)
        DVDY=DVDY + VI*DNDP(KIN+1)
        UUI=UUI+UI*PNIKI
        VVI=VVI+VI*PNIKI
   55 CONTINUE
      DUDX=DUDX/TRI
      DVDX=DVDX/TRI
      DUDY=DUDY/TRI
      DVDY=DVDY/TRI
C
C    Following section calculates quanties:
C      EDXX...BRIT
C
      EDXX=DUDX
      EDYY=DVDY
      EDXY=0.5*(DUDY+DVDX)
      VORT=DVDX-DUDY
C
C    Corrections to the strain-rate expressions for the
C     thin spherical sheet
C
      IF(NCOMP.EQ.-1)THEN
        YLA=YY(1)*PLI(K,1)+YY(2)*PLI(K,2)+YY(3)*PLI(K,3)
        XLA=XX(1)*PLI(K,1)+XX(2)*PLI(K,2)+XX(3)*PLI(K,3)
        YCOTK=1.0/TAN(YLA)
        EDXX=EDXX+VVI*YCOTK
        EDYY=EDYY+XLA*YCOTK*DVDX
        EDXY=EDXY+0.5*(XLA*DUDX-UUI)*YCOTK
        VORT=VORT-(XLA*DUDX-UUI)*YCOTK
      END IF
      EDZZ=-EDXX-EDYY
C
C   strain-rates perpendicular to the computation plane are from the
C   hoop strain (theta-theta) in cylindrical axisymmetry
C
      IF(NCOMP.EQ.2)THEN
        XLA=XX(1)*PLI(K,1)+XX(2)*PLI(K,2)+XX(3)*PLI(K,3)
        EDZZ=UUI/XLA
      END IF
C
C    Second invariant of the strain-rate tensor
C
      ED2I=SQRT(EDXX*EDXX + EDYY*EDYY + EDZZ*EDZZ + 2.0*EDXY*EDXY)
C
C    Viscosity and thermal dissipation
C
      VISC = 1.0
      IF(IVV.GE.1) VISC = VF
      THDI=0.0
      IF(ED2I.GT.0.0)THDI = VISC*ED2I**(1.0/SEXP + 1.0)
      IF(IVV.GE.2)THEN
        IF(ED2I.GT.0.0)THEN
          VISC = VISC*ED2I**(1.0/SEXP - 1.0)
          IF(VISC.GT.BIGV) VISC = BIGV
        ELSE
         VISC = BIGV
        END IF
      END IF
C
C    principal strain rates and stresses
C      THETA and THETA1 are the angles of the principal stress axes
C
      IF ((DVDX+DUDY).EQ.0.0.AND.(DUDX-DVDY).EQ.0.0) THEN
        THETA = 0.0
      ELSE
        THETA=0.5*ATAN2((DVDX+DUDY),(DUDX-DVDY))
      END IF
      THETA1=THETA+PI*0.5
      TCOS=COS(THETA)
      TCOS1=COS(THETA1)
      TSIN=SIN(THETA)
      TSIN1=SIN(THETA1)
      S01=EDXX*TCOS*TCOS + EDYY*TSIN*TSIN + 2.0*EDXY*TSIN*TCOS
      S02=EDXX*TCOS1*TCOS1 + EDYY*TSIN1*TSIN1 + 2.0*EDXY*TSIN1*TCOS1
      IF(S01.GE.S02)THEN
         PSR1=S01
         PSR2=S02
         CANG=THETA
         TANG=THETA1
      ELSE
         PSR1=S02
         PSR2=S01
         CANG=THETA1
         TANG=THETA
      END IF
      PSRM=0.5*(PSR1-PSR2)
      SANG=0.5*(CANG+TANG)
      IF (IQU.EQ.24) VOTA=VORT/(PSR1-PSR2)
      TAXX=2.0*VISC*EDXX
      TAYY=2.0*VISC*EDYY
      TAZZ=2.0*VISC*EDZZ
      TAXY=2.0*VISC*EDXY
      TAU1=2.0*VISC*PSR1
      TAU2=2.0*VISC*PSR2
      TAUM=2.0*VISC*PSRM
C
C    Type of faulting : sum of two double couples :
C       0 to 1/4 : thrust + thrust ; 1/4 to 1/2 : thrust + strike slip ;
C       1/2 to 3/4 : normal + strike slip ; 3/4 to 1 : normal + normal
C       Note:(1/4 to 3/8 thrust dominates strike slip faulting
C             3/8 to 1/2 strike slip dominates thrust faulting
C             1/2 to 5/8 strike slip dominates normal faulting)
C             5/8 to 3/4 normal dominates strike slip faulting
C
      IF (PSR1.NE.0.0) DBLC=(0.75*PI+ATAN(PSR2/PSR1))/PI
      IF(PSR1.LT.0.0) DBLC=DBLC-1.0
C
C    Pressure below which is brittle failure (must subtract pressure term)
C      AMU is the internal coefficient of friction
C
      AMU=0.85
      SAVG=0.5*(PSR1+PSR2)
      APSS= -PSRM*SQRT(1.0+1.0/AMU/AMU)-SAVG
C Normal faulting for thin viscous sheet case
      TMAX=0.5*(PSR1-EDZZ)
      SAVG=0.5*(PSR1+EDZZ)
      APN= -TMAX*SQRT(1.0+1.0/AMU/AMU)-SAVG
C Thrust faulting for thin viscous sheet case
      TMAX=0.5*(EDZZ-PSR2)
      SAVG=0.5*(EDZZ+PSR2)
      APT= -TMAX*SQRT(1.0+1.0/AMU/AMU)-SAVG
C Take the minimum of APSS, APN and APT
        IF(APSS.LE.APN.AND.APSS.LE.APT) TI=APSS
        IF(APN.LE.APSS.AND.APN.LE.APT) TI=APN
        IF(APT.LE.APSS.AND.APT.LE.APN) TI=APT
      BRIT=2.0*VISC*TI
C
C Type of faulting (SS=0.5, N=-0.5, T=1.5)
C    (Contour with level=0.0 and interval=1.0)
C
       IF(APN.LE.APSS.AND.APN.LE.APT) TI= -0.5
       IF(APT.LE.APSS.AND.APT.LE.APN) TI=1.5
       IF(APSS.LE.APN.AND.APSS.LE.APT) TI=0.5
      BRI2=TI
C
C  Orientation of the intermediate deviatoric stress
C   Vertical=0.5, PSR1=-0.5, PSR2=1.5
C    (Contour with level=0.0 and interval=1.0)
C    
C     FOLT=0.5
C     IF((EDZZ.GT.PSR1).AND.(EDZZ.GT.PSR2)) FOLT=-0.5
C     IF((EDZZ.LT.PSR1).AND.(EDZZ.LT.PSR2)) FOLT=1.5
C
C   For thin sheet calculations, SIZZ is got from the gravitational load
C   For plane-strain calculations pressure requires a linear
C   interpolation instead of a quadratic one - therefore it is
C   subtracted after the NTRPLT step (refer stmesh.f)
C
      IF((NCOMP.LE.0).AND.(ICR.NE.0))THEN
        SFAC=-0.5*ARGAN
        IF(IELFIX(LK).NE.0)SFAC=-0.5*(ARGAN+BRGAN)
        SIZZ=SFAC*EXP(2.0*SSQ(LK)) - GPEZERO
        PRES=SIZZ-TAZZ
      ELSE
        PRES=0.0
        SIZZ=TAZZ
      END IF
      SIG1=TAU1+PRES
      SIG2=TAU2+PRES
      SIXX=TAXX+PRES
      SIYY=TAYY+PRES
      BRIT=BRIT-PRES
C
      IF(JELL.EQ.0.AND.K.LT.7) THEN
        SI=0.0
        IF(IQU.EQ.1)  SI=EDXX
        IF(IQU.EQ.2)  SI=EDYY
        IF(IQU.EQ.3)  SI=EDZZ
        IF(IQU.EQ.4)  SI=EDXY
        IF(IQU.EQ.5)  SI=PSR1
        IF(IQU.EQ.6)  SI=PSR2
        IF(IQU.EQ.7)  SI=PSRM
        IF(IQU.EQ.8)  SI=CANG
        IF(IQU.EQ.9)  SI=TANG
        IF(IQU.EQ.10) SI=SANG
        IF(IQU.EQ.11) SI=DBLC
        IF(IQU.EQ.12) SI=VORT
        IF(IQU.EQ.13) SI=ED2I
        IF(IQU.EQ.14) SI=THDI
        IF(IQU.EQ.15) SI=ALOG10(VISC)
        IF(IQU.EQ.16) SI=TAXX
        IF(IQU.EQ.17) SI=TAYY
        IF(IQU.EQ.18) SI=TAZZ
        IF(IQU.EQ.19) SI=TAXY
        IF(IQU.EQ.20) SI=TAU1
        IF(IQU.EQ.21) SI=TAU2
        IF(IQU.EQ.22) SI=TAUM
        IF(IQU.EQ.23) SI=0.0
        IF(IQU.EQ.24) SI=VOTA
        IF(IQU.EQ.25) SI=SIXX
        IF(IQU.EQ.26) SI=SIYY
        IF(IQU.EQ.27) SI=SIZZ
        IF(IQU.EQ.28) SI=SIG1
        IF(IQU.EQ.29) SI=SIG2
        IF(IQU.EQ.30) SI=PRES
        IF(IQU.EQ.31) SI=BRIT
        IF(IQU.EQ.32) SI=BRI2
C       IF(IQU.EQ.33) SI=FOLT
        AMESH(LK)=AMESH(LK) + SI*ANGL
        EMESH(LK)=EMESH(LK) + ANGL
C
      ELSE IF(JELL.EQ.1.AND.K.LT.7) THEN
C
        SI=0.0
C
C    Amplitude and direction of principal strain axes
C
        IF(IQU.EQ.1) THEN
          ARRO1=PSR1
          ARRO2=PSR2
          ARRANG=CANG
C         IF (ABS(ABS(PSR1)-ABS(PSR2)).LT.EPS) ARRANG=PI*0.25
        END IF
C
C     Direction only of principal strain axes
C
        IF(IQU.EQ.2) THEN
          ARRO1=1.0
          ARRO2=-1.0
          ARRANG=CANG
C         IF (ABS(ABS(PSR1)-ABS(PSR2)).LT.EPS) ARRANG=PI*0.25
        END IF
C
C     Amplitude and direction of principal stresses
C
        IF(IQU.EQ.3) THEN
          ARRO1=TAU1
          ARRO2=TAU2
          ARRANG=CANG
C         IF (ABS(ABS(PSR1)-ABS(PSR2)).LT.EPS) ARRANG=PI*0.25
        END IF
C
C     amplitude and direction of maximum shear strain rate
C
        IF(IQU.EQ.4) THEN
          ARRO1=PSRM
          ARRO2=-PSRM
          ARRANG=CANG+PI*0.25
C         IF (ABS(ABS(PSR1)-ABS(PSR2)).LT.EPS) ARRANG=PI*0.5
        END IF
C
C     Direction only of maximum shear stresses
C
        IF(IQU.EQ.5) THEN
          ARRO1=1.0
          ARRO2=-1.0
          ARRANG=CANG+PI*0.25
C         IF (ABS(ABS(PSR1)-ABS(PSR2)).LT.EPS) ARRANG=PI*0.5
        END IF
C
C     Direction and magnitude of maximum shear stress
C
        IF(IQU.EQ.22) THEN
          ARRO1=TAUM
          ARRO2=-TAUM
          ARRANG=CANG+PI*0.25
C         IF (ABS(ABS(PSR1)-ABS(PSR2)).LT.EPS) ARRANG=PI*0.5
        END IF
C
C     Direction of likely strike-slip faulting 
C
        IF(IQU.EQ.7) THEN
          ARRO1=1.0
          ARRO2=0.0
          ARRANG=CANG+PI/3.0
          CAT=COS(ARRANG)
          SAT=SIN(ARRANG)
          ROTRA1=(DVDY-DUDX)*SAT*CAT + DVDX*CAT*CAT - DUDY*SAT*SAT
          ARRANG=CANG+PI*2.0/3.0
          CAT=COS(ARRANG)
          SAT=SIN(ARRANG)
          ROTRA2=(DVDY-DUDX)*SAT*CAT + DVDX*CAT*CAT - DUDY*SAT*SAT
          IF(ABS(ROTRA1).LE.ABS(ROTRA2))THEN
             ARRANG=CANG+PI/3.0
          ELSE
             ARRANG=CANG+PI*2.0/3.0
          END IF
        END IF
C
C   check for aliasing of the angle by averaging sin(2TH) and cos(2TH)
C    sines and cosines are transferred to ARWTWO, where theta is extracted
C
        AMESH(LK)=AMESH(LK) + COS(2.0*ARRANG+PI*0.5)*ANGL
        BMESH(LK)=BMESH(LK) + SIN(2.0*ARRANG+PI*0.5)*ANGL
C       NLK=LK+NUPP
        EMESH(LK)=EMESH(LK) + ANGL
        CMESH(LK)=CMESH(LK) + ARRO1*ANGL
        DMESH(LK)=DMESH(LK) + ARRO2*ANGL
      END IF
   50 CONTINUE
C
C    Normalise by the total angle around the node
C     note that averaging may produce odd effect on angles
C     in neighbourhood of reflection symmetry axis
C
      IF(JELL.EQ.0)THEN
        DO 60 I=1,NUP
          TANGL=EMESH(I)
          IF (TANGL.EQ.0.0) TANGL=EPS
          AMESH(I)=AMESH(I)/TANGL
   60   CONTINUE
      ELSE IF(JELL.EQ.1)THEN
        DO 70 I=1,NUP
          TANGL=EMESH(I)
          IF (TANGL.EQ.0.0) TANGL=EPS
          AMESH(I)=AMESH(I)/TANGL
          BMESH(I)=BMESH(I)/TANGL
          CMESH(I)=CMESH(I)/TANGL
          DMESH(I)=DMESH(I)/TANGL
   70   CONTINUE
      END IF
C      CALL MATPRT(AMESH,NUP,1,NUP,LUW)
      RETURN
C 200 WRITE(LUW,10102) JELL
C0102 FORMAT('INVALID VALUE OF JELL =',I4)
C     STOP
      END
      SUBROUTINE STRAIX(JELL,IQU,NX3,NY3,NP3,RMESH,IHELP,XCMIN,YCMIN,
     1                  XCMAX,YCMAX,SE,ARGAN,BRGAN,HLENSC,BDEPSC,BIG,
     2                  EX,EY,VHB,UVP,SSQ,IELFIX,LEM,NOR,NE,NUP,NUVP,
     3                  NFP,IVV,NCOMP,ICR,TBXOFF,TBYOFF,VOFFSET,MESHNUM)
C
C    Routine to obtain interpolated values of the strain or
C    stress tensor (or derived quantities) from the velocity field
C    Note that option numbers are tied to names of physical quantities
C    by definitions in strain.h
C    STRAIX is intended to replace STRAIN, using a different more
C    accurate means of interpolating quantities based on the spatial
C    derivatives. STRAIX finds components on a rectangular array of 
C    interpolation points.
C
      DIMENSION RMESH(1),IHELP(NP3)
      DIMENSION SSQ(NUP),IELFIX(NUP)
      DIMENSION EX(NUP),EY(NUP),VHB(8,NE),UVP(NUVP)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION A0(3),BY(3),CX(3),DNDP(84),COOL(3)
      DIMENSION PNI(7,6),PLI(7,3),XX(3),YY(3)
      DIMENSION DLDX(3),DLDY(3)
C     DOUBLE PRECISION TRI,X1,X2,X3,Y1,Y2,Y3
C     DOUBLE PRECISION A0(3),BY(3),CX(3),COOL(3)
C
      NUP2=NUP*2
      EPS=1.0E-4
      CALL LNCORD(PNI,PLI)
      GPEZERO=-0.5*ARGAN/(HLENSC*HLENSC)
      HSX=(XCMAX-XCMIN)/FLOAT(NX3-3)
      HSY=(YCMAX-YCMIN)/FLOAT(NY3-3)
      EPSX=HSX*0.5
      EPSY=HSY*0.5
      UUI=0.0
      VVI=0.0
C
C   Zero the arrays
C   IHELP is an index array which records whether an interpolation
C   grid point is actually within the finite element mesh.  It contains
C   the relevant element number - or else zero.
C
      DO 10 I=1,NP3
      IHELP(I)=0
   10 RMESH(I)=0.0
      SXP=SE
C
C    BIGV should be the same as that calculated in VISK (cginit.f)
C
      BIGV=BIG/10000.0
C
C    Look at each element in turn
C
      DO 80 N=1,NE
      IF(IVV.GE.3)SXP=VHB(8,N)
      VF=1.0
C
C     Calculate the geometrical coefficients
C
   12 XMIN=999.
      YMIN=999.
      XMAX=-999.
      YMAX=-999.
      DO 20 K1=1,3
        K2=MOD(K1,3)+1
        K3=MOD(K1+1,3)+1
        LK1=NOR(LEM(K1,N))
        LK2=NOR(LEM(K2,N))
        LK3=NOR(LEM(K3,N))
        X1=EX(LK1)
        X2=EX(LK2)
        X3=EX(LK3)
        Y1=EY(LK1)
        Y2=EY(LK2)
        Y3=EY(LK3)
        XMIN=MIN(XMIN,X1)
        XMAX=MAX(XMAX,X1)
        YMIN=MIN(YMIN,Y1)
        YMAX=MAX(YMAX,Y1)
        A0(K1)=X2*Y3 - X3*Y2
        BY(K1)=Y2    - Y3
        CX(K1)=X3    - X2
   20 CONTINUE
C
C    TRI is twice the area of the triangle element
C    spatial gradients of the Li interpolation functions are constant
C
      TRI=X2*Y3-X3*Y2+X3*Y1-X1*Y3+X1*Y2-X2*Y1
      DO KP=1,3
        DLDX(KP)=BY(KP)/TRI
        DLDY(KP)=CX(KP)/TRI
      ENDDO
      XOFF=0.0
      YOFF=0.0
      VOFF=0.0
      DO 75 IDUP=1,MESHNUM
        IF (IDUP.GT.1) THEN
          XOFF=XOFF+TBXOFF
          YOFF=YOFF+TBYOFF
          XMIN = XMIN + XOFF
          XMAX = XMAX + XOFF
          YMIN = YMIN + YOFF
          YMAX = YMAX + YOFF
          VOFF = VOFF + VOFFSET
        END IF
C     print *,MESHNUM, XMIN
C
C     Now look for the points in the mesh that are inside the tri.
C     define a rectangular patch in RMESH that contains this element
C      IMIN to IMAX, JMIN to JMAX.  If element is outside defined
C      region covered by grid, go to the next element (80)
C
      JMAX=(YMAX+EPSY-YCMIN)/HSY +2
      IF (JMAX.LT.2) GO TO 75
      IF (JMAX.GT.NY3) JMAX=NY3
      JMIN=(YMIN-EPSY-YCMIN)/HSY +2
      IF (JMIN.LT.2) JMIN=2
      IF (JMIN.GT.NY3) GO TO 75
      IF ((JMAX-JMIN).LT.2) JMAX=JMAX+1
      IMAX=(XMAX+EPSX-XCMIN)/HSX +2
      IF (IMAX.LT.2) GO TO 75
      IF (IMAX.GT.NX3) IMAX=NX3
      IMIN=(XMIN-EPSX-XCMIN)/HSX +2
      IF (IMIN.LT.2) IMIN=2
      IF (IMIN.GT.NX3) GO TO 75
C
C    get the coordinates (XP,YP) of point to be interpolated
C
      YP=YCMIN + (JMIN-3)*HSY - YOFF
      DO 70 J=JMIN,JMAX
        YP=YP+HSY
        XP=XCMIN + (IMIN-3)*HSX - XOFF
        DO 50 I=IMIN,IMAX
          XP=XP+HSX
C
C     Calculate the natural coordinates of (XP,YP)
C
          DO 25 K=1,3
            CNL=(A0(K) + XP*BY(K) + YP*CX(K))/TRI
            IF((CNL.GT.1.0+EPS).OR.(CNL.LT.-EPS))GO TO 50
            COOL(K)=CNL
   25     CONTINUE
C
C     If we reach here the point is within the triangle, so
C         Interpolate all strain-rate components
C
          DUDX=0.0
          DUDY=0.0
          DVDX=0.0
          DVDY=0.0
          DO 35 KI=1,6
            LK=LEM(KI,N)
            UK=UVP(LK)
            VK=UVP(LK+NUP)
            IF(KI.LE.3)THEN
              DNDX=(4.0*COOL(KI)-1.0)*DLDX(KI)
              DNDY=(4.0*COOL(KI)-1.0)*DLDY(KI)
            ELSE
              KIF=KI-3
              KIB=MOD(KI+1,3)+1
              DNDX=4.0*(COOL(KIF)*DLDX(KIB)+COOL(KIB)*DLDX(KIF))
              DNDY=4.0*(COOL(KIF)*DLDY(KIB)+COOL(KIB)*DLDY(KIF))
            END IF
            DUDX=DUDX+DNDX*UK
            DUDY=DUDY+DNDY*UK
            DVDX=DVDX+DNDX*VK
            DVDY=DVDY+DNDY*VK
   35     CONTINUE
C
C   if required interpolate the velocity components
C
          IF((NCOMP.GE.2).OR.(NCOMP.LE.-1))THEN
            UUI=0.0
            VVI=0.0
            DO KI=1,6
              LK=LEM(KI,N)
              IF(KI.LE.3)THEN
                QLI=COOL(KI)*(2.0*COOL(KI)-1.0)
              ELSE
                KIF=KI-3
                KIB=MOD(KI+1,3)+1
                QLI=4.0*COOL(KIF)*COOL(KIB)
              END IF
              UUI=UUI+UVP(LK)*QLI
              VVI=VVI+UVP(LK+NUP)*QLI
            END DO
          END IF
C
C   if required, get the viscosity coefficient by interpolation
C
          IF(IVV.GE.1)CALL BINTIP(VHB(1,N),COOL,VF)
C
C   if required, interpolate the pressure field (NCOMP.GE.1)
C
          IF(IQU.GE.25)THEN
            SIZZ=0.0
            IF(NCOMP.GE.1)THEN
              PRES=0.0
              DO K=1,3
                LK=NUP2+NFP+NOR(LEM(K,N))
                PRES=PRES+UVP(LK)*COOL(K)
              END DO
            ELSE
C
C   obtain pressure from SIZZ for thin sheet calculations (NCOMP.LE.0)
C   (from interpolated crustal thickness distribution and Argand number)
C
              IF(ICR.NE.0)THEN
                SFAC=-0.5*ARGAN
                SSQIP=0.0
                ELF=0.0
                DO KI=1,6
                  LK=LEM(KI,N)
                  IF(KI.LE.3)THEN
                    QLI=COOL(KI)*(2.0*COOL(KI)-1.0)
                  ELSE
                    KIF=KI-3
                    KIB=MOD(KI+1,3)+1
                    QLI=4.0*COOL(KIF)*COOL(KIB)
                  END IF
                  SSQIP=SSQIP+SSQ(LK)*QLI
                  ELF=ELF+FLOAT(IELFIX(LK))*QLI
                END DO
                IF(ELF.GE.0.5)SFAC=-0.5*(ARGAN+BRGAN)
                SIZZ=SFAC*EXP(2.0*SSQIP) - GPEZERO
              END IF
            END IF
          END IF  
C
C   place the required quantity in the interpolation grid
C
          IJ=(J-1)*NX3 + I
          IHELP(IJ)=N
          CALL STCOMP(IQU,VALUE,DUDX,DUDY,DVDX,DVDY,PRES,SIZZ,
     :                NCOMP,UUI,VVI,XP,YP,IVV,SXP,VF,BIGV)
          RMESH(IJ)=VALUE
   50   CONTINUE
   70 CONTINUE
   75 CONTINUE
   80 CONTINUE
      RETURN
      END
      SUBROUTINE STRAIL(JELL,IQU,NL,RLINE,ILINE,XP1,YP1,XP2,YP2,
     1                  SE,ARGAN,BRGAN,HLENSC,BDEPSC,BIG,EX,EY,
     2                  VHB,UVP,SSQ,IELFIX,LEM,NOR,NE,NUP,NUVP,
     3                  NFP,IVV,NCOMP,ICR)
C
C    Routine to obtain interpolated values of the strain or
C    stress tensor (or derived quantities) from the velocity field
C    Note that option numbers are tied to names of physical quantities
C    by definitions in strain.h
C    Similar to STRAIX, STRAIL finds components on a trajectory
C    across the solution region
C
      DIMENSION RLINE(NL),ILINE(NL)
      DIMENSION SSQ(NUP),IELFIX(NUP)
      DIMENSION EX(NUP),EY(NUP),VHB(8,NE),UVP(NUVP)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION A0(3),BY(3),CX(3),DNDP(84),COOL(3)
      DIMENSION PNI(7,6),PLI(7,3),XX(3),YY(3)
      DIMENSION DLDX(3),DLDY(3)
C     DOUBLE PRECISION TRI,X1,X2,X3,Y1,Y2,Y3
C     DOUBLE PRECISION A0(3),BY(3),CX(3),COOL(3)
C
      NUP2=NUP*2
      EPS=1.0E-4
      CALL LNCORD(PNI,PLI)
      GPEZERO=-0.5*ARGAN/(HLENSC*HLENSC)
      UUI=0.0
      VVI=0.0
C
C   Zero the arrays
C   RLINE(NL) contains the intrpolated values on the line
C   ILINE(NL) contains a switch indicating a value in RLINE
C   Line runs between (XP1,YP1) and (XP2,YP2)
C
      DO 10 I=1,NL
      ILINE(I)=0
   10 RLINE(I)=0.0
      DXP=(XP2-XP1)/FLOAT(NL-1)
      DYP=(YP2-YP1)/FLOAT(NL-1)
C     EPSX=DXP*0.1
C     EPSY=DYP*0.1
      EPS=0.01*SQRT(DXP*DXP+DYP*DYP)
      SXP=SE
C
C    BIGV should be the same as that calculated in VISK (cginit.f)
C
      BIGV=BIG/10000.0
C
C    Look at each element in turn
C
      DO 80 N=1,NE
      IF(IVV.GE.3)SXP=VHB(8,N)
      VF=1.0
C
C     Calculate the geometrical coefficients
C
   12 XMIN=999.
      YMIN=999.
      XMAX=-999.
      YMAX=-999.
      DO 20 K1=1,3
        K2=MOD(K1,3)+1
        K3=MOD(K1+1,3)+1
        LK1=NOR(LEM(K1,N))
        LK2=NOR(LEM(K2,N))
        LK3=NOR(LEM(K3,N))
        X1=EX(LK1)
        X2=EX(LK2)
        X3=EX(LK3)
        Y1=EY(LK1)
        Y2=EY(LK2)
        Y3=EY(LK3)
        XMIN=MIN(XMIN,X1)
        XMAX=MAX(XMAX,X1)
        YMIN=MIN(YMIN,Y1)
        YMAX=MAX(YMAX,Y1)
        A0(K1)=X2*Y3 - X3*Y2
        BY(K1)=Y2    - Y3
        CX(K1)=X3    - X2
   20 CONTINUE
C
C    TRI is twice the area of the triangle element
C    spatial gradients of the Li interpolation functions are constant
C
      TRI=X2*Y3-X3*Y2+X3*Y1-X1*Y3+X1*Y2-X2*Y1
      DO KP=1,3
        DLDX(KP)=BY(KP)/TRI
        DLDY(KP)=CX(KP)/TRI
      ENDDO
C
C     Now look for the points on the line that are inside the tri.
C
      DO 70 J=1,NL
        XP=XP1+DXP*FLOAT(J-1)
        YP=YP1+DYP*FLOAT(J-1)
        IF((YP.GT.YMAX+EPS).OR.(YP.LT.YMIN-EPS).OR.
     1     (XP.GT.XMAX+EPS).OR.(XP.LT.XMIN-EPS))GO TO 70
C
C     Calculate the natural coordinates of (XP,YP)
C
          DO 25 K=1,3
            CNL=(A0(K) + XP*BY(K) + YP*CX(K))/TRI
            IF((CNL.GT.1.0+EPS).OR.(CNL.LT.-EPS))GO TO 70
            COOL(K)=CNL
   25     CONTINUE
C
C     If we reach here the point is within the triangle, so
C         Interpolate all strain-rate components
C
          DUDX=0.0
          DUDY=0.0
          DVDX=0.0
          DVDY=0.0
          DO 35 KI=1,6
            LK=LEM(KI,N)
            UK=UVP(LK)
            VK=UVP(LK+NUP)
            IF(KI.LE.3)THEN
              DNDX=(4.0*COOL(KI)-1.0)*DLDX(KI)
              DNDY=(4.0*COOL(KI)-1.0)*DLDY(KI)
            ELSE
              KIF=KI-3
              KIB=MOD(KI+1,3)+1
              DNDX=4.0*(COOL(KIF)*DLDX(KIB)+COOL(KIB)*DLDX(KIF))
              DNDY=4.0*(COOL(KIF)*DLDY(KIB)+COOL(KIB)*DLDY(KIF))
            END IF
            DUDX=DUDX+DNDX*UK
            DUDY=DUDY+DNDY*UK
            DVDX=DVDX+DNDX*VK
            DVDY=DVDY+DNDY*VK
   35     CONTINUE
C
C   if required interpolate the velocity components
C
          IF((NCOMP.GE.2).OR.(NCOMP.LE.-1))THEN
            UUI=0.0
            VVI=0.0
            DO KI=1,6
              LK=LEM(KI,N)
              IF(KI.LE.3)THEN
                QLI=COOL(KI)*(2.0*COOL(KI)-1.0)
              ELSE
                KIF=KI-3
                KIB=MOD(KI+1,3)+1
                QLI=4.0*COOL(KIF)*COOL(KIB)
              END IF
              UUI=UUI+UVP(LK)*QLI
              VVI=VVI+UVP(LK+NUP)*QLI
            END DO
          END IF
C
C   if required, get the viscosity coefficient by interpolation
C
          IF(IVV.GE.1)CALL BINTIP(VHB(1,N),COOL,VF)
C
C   if required, interpolate the pressure field (NCOMP.GE.1)
C
          IF(IQU.GE.25)THEN
            SIZZ=0.0
            IF(NCOMP.GE.1)THEN
              PRES=0.0
              DO K=1,3
                LK=NUP2+NFP+NOR(LEM(K,N))
                PRES=PRES+UVP(LK)*COOL(K)
              END DO
            ELSE
C
C   obtain pressure from SIZZ for thin sheet calculations (NCOMP.LE.0)
C   (from interpolated crustal thickness distribution and Argand number)
C
              IF(ICR.NE.0)THEN
                SFAC=-0.5*ARGAN
                SSQIP=0.0
                ELF=0.0
                DO KI=1,6
                  LK=LEM(KI,N)
                  IF(KI.LE.3)THEN
                    QLI=COOL(KI)*(2.0*COOL(KI)-1.0)
                  ELSE
                    KIF=KI-3
                    KIB=MOD(KI+1,3)+1
                    QLI=4.0*COOL(KIF)*COOL(KIB)
                  END IF
                  SSQIP=SSQIP+SSQ(LK)*QLI
                  ELF=ELF+FLOAT(IELFIX(LK))*QLI
                END DO
                IF(ELF.GE.0.5)SFAC=-0.5*(ARGAN+BRGAN)
                SIZZ=SFAC*EXP(2.0*SSQIP) - GPEZERO
              END IF
            END IF
          END IF  
C
C   place the required quantity in the interpolation grid
C
          ILINE(J)=N
          CALL STCOMP(IQU,VALUE,DUDX,DUDY,DVDX,DVDY,PRES,SIZZ,
     :                NCOMP,UUI,VVI,XP,YP,IVV,SXP,VF,BIGV)
          RLINE(J)=VALUE
   70   CONTINUE
   80 CONTINUE
      RETURN
      END
      SUBROUTINE STRAIC(JELL,IQU,CVALUE,AREA2,IVERBOSE,IERR,
     1                  SE,ARGAN,BRGAN,HLENSC,BDEPSC,BIG,EX,EY,
     2                  VHB,UVP,SSQ,IELFIX,LEM,NOR,NE,NUP,NUVP,
     3                  NFP,IVV,NCOMP,ICR)
C
C    Routine to obtain interpolated values of the strain or
C    stress tensor (or derived quantities) from the velocity field
C    Note that option numbers are tied to names of physical quantities
C    by definitions in strain.h
C    STRAIC finds component IQU at the centroid of element JELL
C    CVALUE is the component value, AREA2 is twice the element area
C
      DIMENSION SSQ(NUP),IELFIX(NUP)
      DIMENSION EX(NUP),EY(NUP),VHB(8,NE),UVP(NUVP)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION A0(3),BY(3),CX(3),DNDP(84),COOL(3)
      DIMENSION PNI(7,6),PLI(7,3),XX(3),YY(3)
      DIMENSION DLDX(3),DLDY(3)
C     DOUBLE PRECISION TRI,X1,X2,X3,Y1,Y2,Y3
C     DOUBLE PRECISION A0(3),BY(3),CX(3),COOL(3)
C
      NUP2=NUP*2
      EPS=1.0E-4
      CALL LNCORD(PNI,PLI)
      GPEZERO=-0.5*ARGAN/(HLENSC*HLENSC)
      UUI=0.0
      VVI=0.0
C
C   Zero the values
C
      CVALUE=0.0
      AREA2=0.0
      EPS=0.01*SQRT(DXP*DXP+DYP*DYP)
      SXP=SE
C
C    BIGV should be the same as that calculated in VISK (cginit.f)
C
      BIGV=BIG/10000.0
C
C    Look at each element in turn
C
      N=JELL
      IF(IVV.GE.3)SXP=VHB(8,N)
      VF=1.0
C
C     Calculate the geometrical coefficients
C
      DO 20 K1=1,3
        K2=MOD(K1,3)+1
        K3=MOD(K1+1,3)+1
        LK1=NOR(LEM(K1,N))
        LK2=NOR(LEM(K2,N))
        LK3=NOR(LEM(K3,N))
        X1=EX(LK1)
        X2=EX(LK2)
        X3=EX(LK3)
        Y1=EY(LK1)
        Y2=EY(LK2)
        Y3=EY(LK3)
        A0(K1)=X2*Y3 - X3*Y2
        BY(K1)=Y2    - Y3
        CX(K1)=X3    - X2
   20 CONTINUE
      XP = (X1+X2+X3)/3.0
      YP = (Y1+Y2+Y3)/3.0
C
C    TRI is twice the area of the triangle element
C    spatial gradients of the Li interpolation functions are constant
C
      TRI=X2*Y3-X3*Y2+X3*Y1-X1*Y3+X1*Y2-X2*Y1
      AREA2=TRI
      DO KP=1,3
        DLDX(KP)=BY(KP)/TRI
        DLDY(KP)=CX(KP)/TRI
      ENDDO
C
C     Calculate the natural coordinates of (XP,YP)
C
      DO 25 K=1,3
        CNL=(A0(K) + XP*BY(K) + YP*CX(K))/TRI
        IF((CNL.GT.1.0+EPS).OR.(CNL.LT.-EPS)) THEN
          IERR=1
          GO TO 70
        END IF
        COOL(K)=CNL
   25 CONTINUE
C
C     If we reach here the point is within the triangle, so
C         Interpolate all strain-rate components
C
          DUDX=0.0
          DUDY=0.0
          DVDX=0.0
          DVDY=0.0
          DO 35 KI=1,6
            LK=LEM(KI,N)
            UK=UVP(LK)
            VK=UVP(LK+NUP)
            IF(KI.LE.3)THEN
              DNDX=(4.0*COOL(KI)-1.0)*DLDX(KI)
              DNDY=(4.0*COOL(KI)-1.0)*DLDY(KI)
            ELSE
              KIF=KI-3
              KIB=MOD(KI+1,3)+1
              DNDX=4.0*(COOL(KIF)*DLDX(KIB)+COOL(KIB)*DLDX(KIF))
              DNDY=4.0*(COOL(KIF)*DLDY(KIB)+COOL(KIB)*DLDY(KIF))
            END IF
            DUDX=DUDX+DNDX*UK
            DUDY=DUDY+DNDY*UK
            DVDX=DVDX+DNDX*VK
            DVDY=DVDY+DNDY*VK
   35     CONTINUE
C
C   if required interpolate the velocity components
C
          IF((NCOMP.GE.2).OR.(NCOMP.LE.-1))THEN
            UUI=0.0
            VVI=0.0
            DO KI=1,6
              LK=LEM(KI,N)
              IF(KI.LE.3)THEN
                QLI=COOL(KI)*(2.0*COOL(KI)-1.0)
              ELSE
                KIF=KI-3
                KIB=MOD(KI+1,3)+1
                QLI=4.0*COOL(KIF)*COOL(KIB)
              END IF
              UUI=UUI+UVP(LK)*QLI
              VVI=VVI+UVP(LK+NUP)*QLI
            END DO
          END IF
C
C   if required, get the viscosity coefficient by interpolation
C
          IF(IVV.GE.1)CALL BINTIP(VHB(1,N),COOL,VF)
C
C   if required, interpolate the pressure field (NCOMP.GE.1)
C
          IF(IQU.GE.25)THEN
            SIZZ=0.0
            IF(NCOMP.GE.1)THEN
              PRES=0.0
              DO K=1,3
                LK=NUP2+NFP+NOR(LEM(K,N))
                PRES=PRES+UVP(LK)*COOL(K)
              END DO
            ELSE
C
C   obtain pressure from SIZZ for thin sheet calculations (NCOMP.LE.0)
C   (from interpolated crustal thickness distribution and Argand number)
C
              IF(ICR.NE.0)THEN
                SFAC=-0.5*ARGAN
                SSQIP=0.0
                ELF=0.0
                DO KI=1,6
                  LK=LEM(KI,N)
                  IF(KI.LE.3)THEN
                    QLI=COOL(KI)*(2.0*COOL(KI)-1.0)
                  ELSE
                    KIF=KI-3
                    KIB=MOD(KI+1,3)+1
                    QLI=4.0*COOL(KIF)*COOL(KIB)
                  END IF
                  SSQIP=SSQIP+SSQ(LK)*QLI
                  ELF=ELF+FLOAT(IELFIX(LK))*QLI
                END DO
                IF(ELF.GE.0.5)SFAC=-0.5*(ARGAN+BRGAN)
                SIZZ=SFAC*EXP(2.0*SSQIP) - GPEZERO
              END IF
            END IF
          END IF  
C
C   place the required quantity in the interpolation grid
C
          CALL STCOMP(IQU,VALUE,DUDX,DUDY,DVDX,DVDY,PRES,SIZZ,
     :                NCOMP,UUI,VVI,XP,YP,IVV,SXP,VF,BIGV)
          CVALUE=VALUE
   70 RETURN
      END
      SUBROUTINE BINTIP(VHB,COOL,VINT)
C
C    Interpolate a quantity defined on the 7 integration points
C    using a best fit quadratic interpolation function
C
      DIMENSION VHB(7),VNOD(6),ARR(7,6),COOL(3),QOOL(6)
C
C   the following matrix is the 7x6 matrix that projects from the
C   7 interpolation points to a best fit set of 6 nodal values
C
      SAVE ARR
      DATA ARR/1.9743924,0.1434449,0.1434449,-0.4126757,-0.4126757,
     : 0.2563768,-0.6923076,0.1434449,1.9743924,0.1434449,0.2563768,
     :-0.4126757,-0.4126757,-0.6923076,0.1434449,0.1434449,1.9743924,
     :-0.4126757,0.2563768,-0.4126757,-0.6923076,0.09812581,0.1562660,
     :0.09812581,1.0345624,-0.2822974,-0.2822974,0.1775148,0.09812581,
     :0.09812581,0.1562660,-0.2822974,1.0345624,-0.2822974,0.1775148,
     :0.1562660,0.09812581,0.09812581,-0.2822974,-0.2822974,1.0345624,
     :0.1775148/
C
C    obtain the best-fit equivalent nodal values
C
      DO KE=1,6
        VNOD(KE)=0.0
        DO KPP=1,7
          VNOD(KE)=VNOD(KE)+ARR(KPP,KE)*VHB(KPP)
        ENDDO
      ENDDO
C
C     compute the quadratic interpolation function values at
C     the given local coordinates.
C
      DO KE=1,3
        QOOL(KE)=COOL(KE)*(2.0*COOL(KE)-1.0)
      ENDDO
      DO KE=4,6
        KPF=KE-3
        KPB=MOD(KE+1,3)+1
        QOOL(KE)=4.0*COOL(KPF)*COOL(KPB)
      ENDDO
C
C   the interpolated function value then is inner product of VNOD and QOOL
C
      SUM=0.0
      DO KE=1,6
        SUM=SUM+QOOL(KE)*VNOD(KE)
      ENDDO
      VINT=SUM
      RETURN
      END
      SUBROUTINE STCOMP(IQU,VALUE,DUDX,DUDY,DVDX,DVDY,PRES,SIZZ,
     :                  NCOMP,UUI,VVI,XLA,YLA,IVV,SXP,VF,BIGV)
      SAVE PI
      DATA PI/3.141592653/
C
C    this routine assigns the relevant component of the stress or
C    strain-rate tensor for use by STRAIX
C
      EDXX=DUDX
      EDYY=DVDY
      EDXY=0.5*(DUDY+DVDX)
      VORT=DVDX-DUDY
C
C    Corrections to the strain-rate expressions for the
C     thin spherical sheet
C
      IF(NCOMP.EQ.-1)THEN
        YCOTK=1.0/TAN(YLA)
        EDXX=EDXX+VVI*YCOTK
        EDYY=EDYY+XLA*YCOTK*DVDX
        EDXY=EDXY+0.5*(XLA*DUDX-UUI)*YCOTK
        VORT=VORT-(XLA*DUDX-UUI)*YCOTK
      END IF
      EDZZ=-(EDXX+EDYY)
C
C   strain-rates perpendicular to the plane in cylindrical axisymmetry
C
      IF(NCOMP.EQ.2)THEN
        EDZZ=UUI/XLA
      END IF
C
      IF(IQU.EQ.1)THEN
        VALUE=EDXX
        RETURN
      ELSE IF(IQU.EQ.2)THEN
        VALUE=EDYY
        RETURN
      ELSE IF(IQU.EQ.3)THEN
        VALUE=EDZZ
        RETURN
      ELSE IF(IQU.EQ.4)THEN
        VALUE=EDXY
        RETURN
C
C    vorticity VORT
C
      ELSE IF(IQU.EQ.12)THEN
        VALUE=VORT
        RETURN
      END IF   ! end of IF BLOCK A
C
C    Second invariant of the strain-rate tensor
C
      IF(IQU.GE.13)THEN
        ED2I=SQRT(EDXX*EDXX + EDYY*EDYY + EDZZ*EDZZ + 2.0*EDXY*EDXY)
        IF(IQU.EQ.13)THEN
          VALUE=ED2I
          RETURN
C
C     thermal dissipation
C
        ELSE IF(IQU.EQ.14)THEN
          THDI=0.0
          IF(ED2I.GT.0.0)THDI=VF*ED2I**(1.0/SXP + 1.0)
          VALUE=THDI
          RETURN
        END IF
C
C     viscosity
C
        VISC=VF
        IF(IVV.GE.2)THEN
          VISC=BIGV
          IF(ED2I.GT.0.0)VISC=VF*ED2I**(1.0/SXP - 1.0)
          IF(VISC.GT.BIGV)VISC=BIGV
        END IF
      END IF    !end of IF BLOCK B
      IF(IQU.EQ.15)THEN
        VALUE=ALOG10(VISC)
        RETURN
C
C    TAXX, TAYY, TAZZ, TAXY
C
      ELSE IF(IQU.EQ.16)THEN
        VALUE=2.0*EDXX*VISC
        RETURN
      ELSE IF(IQU.EQ.17)THEN
        VALUE=2.0*EDYY*VISC
        RETURN
      ELSE IF(IQU.EQ.18)THEN
        VALUE=2.0*EDZZ*VISC
        RETURN
      ELSE IF(IQU.EQ.19)THEN
        VALUE=2.0*EDXY*VISC
        RETURN
      END IF    ! end of IF BLOCK C
C
C    Change pressure for thin sheet
C
      IF(NCOMP.LE.0)PRES=SIZZ-2.0*EDZZ*VISC
C
C    PRES, SIXX, SIYY, SIZZ
C
      IF(IQU.EQ.30)THEN
        VALUE=PRES
        RETURN
      ELSE IF(IQU.EQ.25)THEN
        VALUE=2.0*EDXX*VISC+PRES
        RETURN
      ELSE IF(IQU.EQ.26)THEN
        VALUE=2.0*EDYY*VISC+PRES
        RETURN
      ELSE IF(IQU.EQ.27)THEN
        VALUE=2.0*EDZZ*VISC+PRES
        RETURN
      END IF     ! end of IF BLOCK D
C
C    Diagonalise the strain-rate tensor if required
C    THETA and THETA1 are the angles of the principal axes
C
      IF((EDXY.EQ.0.0).AND.(EDXX-EDYY).EQ.0.0)THEN
        THETA = 0.0
      ELSE
        THETA=0.5*ATAN2(2.0*EDXY,(EDXX-EDYY))
      END IF
      THETA1=THETA+PI*0.5
      TCOS=COS(THETA)
      TCS1=COS(THETA1)
      TSIN=SIN(THETA)
      TSN1=SIN(THETA1)
      S01=EDXX*TCOS*TCOS + EDYY*TSIN*TSIN + 2.0*EDXY*TSIN*TCOS
      S02=EDXX*TCS1*TCS1 + EDYY*TSN1*TSN1 + 2.0*EDXY*TSN1*TCS1
      IF(S01.GE.S02)THEN
        PSR1=S01
        PSR2=S02
        CANG=THETA
        TANG=THETA1
      ELSE
        PSR1=S02
        PSR2=S01
        CANG=THETA1
        TANG=THETA
      END IF
      PSRM=0.5*(PSR1-PSR2)
      SANG=0.5*(CANG+TANG)
      IF(IQU.EQ.5)THEN
        VALUE=PSR1
        RETURN
      ELSE IF(IQU.EQ.6)THEN
        VALUE=PSR2
        RETURN
      ELSE IF(IQU.EQ.7)THEN
        VALUE=PSRM
        RETURN
      ELSE IF(IQU.EQ.8)THEN
        VALUE=CANG
        RETURN
      ELSE IF(IQU.EQ.9)THEN
        VALUE=TANG
        RETURN
      ELSE IF(IQU.EQ.10)THEN
        VALUE=SANG
        RETURN
C
C    Type of faulting : sum of two double couples :
C       0 to 1/4 : thrust + thrust ; 1/4 to 1/2 : thrust + strike slip ;
C       1/2 to 3/4 : normal + strike slip ; 3/4 to 1 : normal + normal
C       Note:(1/4 to 3/8 thrust dominates strike slip faulting
C             3/8 to 1/2 strike slip dominates thrust faulting
C             1/2 to 5/8 strike slip dominates normal faulting)
C             5/8 to 3/4 normal dominates strike slip faulting
C
      ELSE IF(IQU.EQ.11)THEN
        DBLC=0.5
        IF((PSR1.NE.0.0).OR.(PSR2.NE.0.0))
     :              DBLC=(0.75*PI+ATAN2(PSR2,PSR1))/PI
        VALUE=DBLC
        RETURN
C
C   vorticity to shear stress ratio: VOTA
C
      ELSE IF(IQU.EQ.24)THEN
        VALUE=0.0
        IF(ABS(PSR1-PSR2).GT.1.E-6)VALUE=VORT/(PSR1-PSR2)
        RETURN
      ELSE IF(IQU.EQ.20)THEN
        VALUE=2.0*VISC*PSR1
        RETURN
      ELSE IF(IQU.EQ.21)THEN
        VALUE=2.0*VISC*PSR2
        RETURN
      ELSE IF(IQU.EQ.22)THEN
        VALUE=2.0*VISC*PSRM
        RETURN
      ELSE IF(IQU.EQ.28)THEN
        VALUE=2.0*VISC*PSR1+PRES
        RETURN
      ELSE IF(IQU.EQ.29)THEN
        VALUE=2.0*VISC*PSR2+PRES
        RETURN
      END IF     ! end of IF BLOCK E
C
C    Type of brittle failure: AMU is the coefficient of friction
C    APSS for strike-slip, APN for normal, APT for thrust, latter
C    two only relevant for thin sheet
C
      AMU=0.85
      SAVG=0.5*(PSR1+PSR2)
      APSS= -PSRM*SQRT(1.0+1.0/AMU/AMU)-SAVG
      TMAX=0.5*(PSR1-EDZZ)
      SAVG=0.5*(PSR1+EDZZ)
      APN= -TMAX*SQRT(1.0+1.0/AMU/AMU)-SAVG
      TMAX=0.5*(EDZZ-PSR2)
      SAVG=0.5*(EDZZ+PSR2)
      APT= -TMAX*SQRT(1.0+1.0/AMU/AMU)-SAVG
C
C     Take the minimum of APSS, APN and APT
C
      IF(IQU.EQ.31)THEN
        IF(APSS.LE.APN.AND.APSS.LE.APT) TI=APSS
        IF(APN.LE.APSS.AND.APN.LE.APT) TI=APN
        IF(APT.LE.APSS.AND.APT.LE.APN) TI=APT
        VALUE=2.0*VISC*TI    ! BRIT
        RETURN
C
C Type of faulting (SS=0.5, N=-0.5, T=1.5)
C    (Contour with level=0.0 and interval=1.0)
C
      ELSE IF(IQU.EQ.32)THEN
        IF(APN.LE.APSS.AND.APN.LE.APT) TI= -0.5
        IF(APT.LE.APSS.AND.APT.LE.APN) TI=1.5
        IF(APSS.LE.APN.AND.APSS.LE.APT) TI=0.5
        VALUE=TI             ! BRI2
        RETURN
C
C  Orientation of the intermediate deviatoric stress
C   Vertical=0.5, PSR1=-0.5, PSR2=1.5
C    (Contour with level=0.0 and interval=1.0)
C    
      ELSE IF(IQU.EQ.33)THEN  !may not be active any more
C       FOLT=0.5
C       IF((EDZZ.GT.PSR1).AND.(EDZZ.GT.PSR2)) FOLT=-0.5
C       IF((EDZZ.LT.PSR1).AND.(EDZZ.LT.PSR2)) FOLT=1.5
        VALUE=0.0
        RETURN
      END IF
C
C   default value for any unrecognised value of IQU
C
      VALUE=0.0
      RETURN
      END
C
      SUBROUTINE PAVG(SUM,EX,EY,UVP,NN,NUP,NFP,NUVP)
C
C  Routine to calculate the mean of the pressure since the pressure
C  is indeterminate by a constant amount
C
      DIMENSION EX(NUP),EY(NUP),UVP(NUVP)
C     INCLUDE 'cg.parameters'
C     COMMON/B1/EX1(NUPP)
C     COMMON/C1/EY1(NUPP)
C     COMMON/AD/UVP(NROWSP)
C     COMMON/CONTRO/LUW,LRBAT,LRINT,LDIN,LWINT,LPLOT,IAUTO,ILABEL,ICOLO
      SUM=0.0
      ITOT=0
      DO 10 I=1,NN
        AX=EX(I)-1.0
        AY=EY(I)-1.0
        RR=SQRT(AX*AX+AY*AY)
        NI=I+NUP+NUP+NFP
        IF(RR.GT.0.05.AND.RR.LT.0.5) THEN
           SUM=SUM+UVP(NI)
           ITOT=ITOT+1
        END IF
   10 CONTINUE
      IF (ITOT.NE.0) SUM=SUM/ITOT
C      WRITE(LSC,10001)SUM
C10001 FORMAT("The mean pressure is ",G12.5)
      RETURN
      END
C
      SUBROUTINE  MATPRT(Z,NX,NY,NK,LUW)
C 
C    MATPRT and MITPRT are designed for outputting matrix data in
C    tabulated form, either for purpose of debugging, or for verbose
C    mode output. They are respectively for real and integer variables
C
      SAVE A
      DIMENSION Z(NK)
      CHARACTER*1 A(10)
      DIMENSION L(10)
      DATA A/10*' '/
C            FIND NUMBER OF BLOCKS
C
      JB=10
      NB=NX/JB+1
      MM=MOD(NX,JB)
      IF (MM.EQ.0) NB=NB-1
C
C            CYCLE OVER BLOCKS
C
      DO12 I=1,NB
      JA=(I-1)*JB + 1
      JC=I*JB
      IF (I.EQ.NB) JC=NX
      JJ=JC-JA+1
C
      DO 10 J=1,JJ
   10 L(J)=JA+J-1
      WRITE (LUW,101) (A(J),L(J),J=1,JJ)
  101 FORMAT (7X,10(A1,'J=',I5,3X))
C
C            CYCLE OVER ROWS
C
      DO 11 K1=1,NY
   11 WRITE (LUW,102) K1, (Z((K1-1)*NX+J),J=JA,JC)
  102 FORMAT (1X,'K=',I4,1X,10G11.5)
C
   12 WRITE(LUW,103)
  103 FORMAT(/)
C
      RETURN
      END
C
      SUBROUTINE  MITPRT(JZ,NX,NY,NK,LUW)
C
C    MATPRT and MITPRT are designed for outputting matrix data in
C    tabulated form, either for purpose of debugging, or for verbose
C    mode output. They are respectively for real and integer variables
C
      SAVE A
      DIMENSION JZ(NK)
      CHARACTER*1 A(10)
      DIMENSION L(10)
      DATA A/10*' '/
C            FIND NUMBER OF BLOCKS
C
      JB=10
      NB=NX/JB+1
      MM=MOD(NX,JB)
      IF (MM.EQ.0) NB=NB-1
C
C            CYCLE OVER BLOCKS
C
      DO12 I=1,NB
      JA=(I-1)*JB + 1
      JC=I*JB
      IF (I.EQ.NB) JC=NX
      JJ=JC-JA+1
C
      DO 10 J=1,JJ
   10 L(J)=JA+J-1
      WRITE (LUW,101) (A(J),L(J),J=1,JJ)
  101 FORMAT (10X,10(A1,'J=',I5,3X))
C
C            CYCLE OVER ROWS
C
      DO 11 K1=1,NY
   11 WRITE (LUW,102) K1, (JZ((K1-1)*NX+J),J=JA,JC)
  102 FORMAT (1X,'K=',I4,1X,10I11)
C
   12 WRITE(LUW,103)
  103 FORMAT(/)
C
      RETURN
      END

      SUBROUTINE NTRPLTPRES(AMESH,PRES,NOR,LEM,
     :                      NUP,NE,NN)
      DIMENSION AMESH(NUP),NOR(NUP),PRES(NN),LEM(6,NE)

      DO 10 N = 1,NE
        DO 20 K = 1,3
          LK = LEM(K,N)
          NLK = NOR(LK)
          AMESH(LK) = PRES(NLK)
          K4 = MOD(K,3) + 3
          LK4 = LEM(K4,N)
          K3 = MOD(K+1,3) + 1
          LK3 = LEM(K3,N)
          NLK3 = NOR(LK3)
          AMESH(LK4) = 0.5 * (PRES(NLK) + PRES(NLK3))
  20    CONTINUE
  10  CONTINUE
      RETURN
      END
