
      SUBROUTINE STRAINAVG(IM,EX,EY,EXREF,EYREF,WORK,
     :                     LEM,NOR,IPOLYNUM,
     :                     FAVG,AL,NE,NUP,IVERBOSE)
C
C     STRAINAVG calculates an area averaged deformation for
C     the polygonal region, specified by IM, using the
C     deformation for each mesh element.
C     Only valid for small strain increments.
C     (TRI is twice element area, ASUM is twice the polygon area)
C
C
C     Y1 CONTAINS COORDINATES OF UNDEFORMED ELEMENT
C
C     Y2 CONTAINS COORDINATES OF DEFORMED   ELEMENT
C
C     ORDER OF COORDINATES: X(PT1),Y(PT1), ....... ,X(PT3),Y(PT3)
C
C     NNN ALLOWS CHOICE OF RELATIVE POSITION VECTORS TO
C
C     BE PLACED IN X1 AND X2
C
C     NNN = 1:      (PT2 - PT1, PT3 - PT2) (PT2 is reference pt)
C
C     NNN = 2:      (PT2 - PT1, PT3 - PT1) (PT1 is reference pt)
C
C     NNN = 3:      (PT3 - PT2, PT3 - PT1) (PT3 is reference pt)
C
C
      DIMENSION EX(NUP),EY(NUP),EXREF(NUP),EYREF(NUP)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION FAVG(4),IPOLYNUM(NE)
      DIMENSION WORK(NUP)
C     COMMON/CONTRO/LUW,LRBAT,LRINT,LDIN,LWINT,LPLOT,IAUTO,ILABEL
C    1 ,ICOLO
      DIMENSION Y1(6),Y2(6),X1(4),X2(4),F(4),N1(4),N2(4)
      DOUBLE PRECISION TRI,DXP(3),DYP(3),ASUM
      DOUBLE PRECISION DFAVG(4), DFAVGSUM(4)
      DATA PI /3.14159/
C
      ASUM = 0.0
      AL1SUM = 0.0
      AL2SUM = 0.0
      DO 10 N=1,4
  10  DFAVGSUM(N) = 0.0

C
C    For each triangle in the polygon
C

      DO 60 N=1,NE
      IF (IPOLYNUM(N).NE.IM) GO TO 60
C
C    Put the element coordinates into the Y1 and Y2 matrices
C

      DO 20 K=1,3
      LK=NOR(LEM(K,N))
      JJ=2*(K-1)+1
      Y1(JJ)=EXREF(LK)
      Y1(JJ+1)=EYREF(LK)
      Y2(JJ)=EX(LK)
      Y2(JJ+1)=EY(LK)
      DXP(K)=EX(LK)
      DYP(K)=EY(LK)
   20 CONTINUE
C
C    TRI is twice the area of the triangle element
C
      TRI=DXP(2)*DYP(3)-DXP(3)*DYP(2)+DXP(3)*DYP(1)-DXP(1)*DYP(3)
     :                +DXP(1)*DYP(2)-DXP(2)*DYP(1)
      ASUM = ASUM + TRI

      NNN=1
      DO 1 I = 1,4
      N1(I) = I + 2
 1    N2(I) = I
      GOTO (2,3,4) NNN
 3    N2(3) = 1
      N2(4) = 2
      GOTO 2
 4    N1(1) = 5
      N1(2) = 6
 2    DO 5 I = 1,4
      X1(I) = Y1(N1(I)) - Y1(N2(I))
 5    X2(I) = Y2(N1(I)) - Y2(N2(I))
C
C     CALCULATE ELEMENTS OF STRAIN TENSOR
C
C       Calculated as:
C       F(1) = Exx = du/dx+1
C       F(2) = Exy = du/dy
C       F(3) = Eyx = dv/dx
C       F(4) = Eyy = dv/dy+1
C       If there is no deformation, strain ellipse is a circle with
C       radius of 1 and F(1) = F(4) = 1
      DET = X1(1)*X1(4) - X1(2)*X1(3)
C       update sum over the elements
      DFAVG(1) = (X2(1)*X1(4) - X1(2)*X2(3))/DET
      DFAVG(2) = (X1(1)*X2(3) - X2(1)*X1(3))/DET
      DFAVG(3) = (X2(2)*X1(4) - X1(2)*X2(4))/DET
      DFAVG(4) = (X1(1)*X2(4) - X2(2)*X1(3))/DET
      DFAVGSUM(1) = DFAVGSUM(1) + DFAVG(1) * TRI
      DFAVGSUM(2) = DFAVGSUM(2) + DFAVG(2) * TRI
      DFAVGSUM(3) = DFAVGSUM(3) + DFAVG(3) * TRI
      DFAVGSUM(4) = DFAVGSUM(4) + DFAVG(4) * TRI
 
C     CALCULATE ORIENTATION OF PRINCIPAL AXES
C
C       AL1 and AL2 are STRAIN "in plane" MIN and MAX
C       AL3 is STRAIN vertical
C
      BOT = DFAVG(1)*DFAVG(1) + DFAVG(3)*DFAVG(3)
     :       - DFAVG(2)*DFAVG(2) - DFAVG(4)*DFAVG(4)
      TOP = 2.*(DFAVG(1)*DFAVG(2) + DFAVG(3)*DFAVG(4))
      THETA1 = PI/2
      IF(ABS(BOT).GT.1.E-37) THETA1 = 0.5*ATAN(TOP/BOT)
      C = COS(THETA1)
      S = SIN(THETA1)
      AL1 = SQRT((DFAVG(1)*C + DFAVG(2)*S)**2
     :             + (DFAVG(3)*C + DFAVG(4)*S)**2)
C       sum over the elements
      AL1SUM = AL1SUM + AL1 * TRI
      TOP = DFAVG(3)*C + DFAVG(4)*S
      BOT = DFAVG(1)*C + DFAVG(2)*S
      THETA2 = THETA1 + PI/2
      C = COS(THETA2)
      S = SIN(THETA2)
      AL2 = SQRT((DFAVG(1)*C + DFAVG(2)*S)**2
     :             + (DFAVG(3)*C + DFAVG(4)*S)**2)
      AL2SUM = AL2SUM + AL2 * TRI
      AL3 = 1.0/(AL1*AL2)
C
C     Calculate orientation of principal axes after deformation
C
      TH1 = PI/2
      IF(ABS(BOT).GT.1.E-37) TH1 = ATAN(TOP/BOT)
      TH2 = TH1 + PI/2
      ROT = TH1 - THETA1
      IF(AL2.GT.AL1)DEF=ALOG10(AL2/AL1)
      IF(AL2.LE.AL1)DEF=ALOG10(AL1/AL2)
      NC=2*(N-1) + 1
      WORK(NC)=DEF
      WORK(NC+1)=ROT*180.0/PI
      IF(IM.GE.3)GO TO 60
      XO = (Y2(1) + Y2(3) + Y2(5))/3
      YO = (Y2(2) + Y2(4) + Y2(6))/3
      IF (IVERBOSE.NE.0) THEN
        TH1DEG=TH1*180.0/PI
        WRITE(6,10555)N,XO,YO,AL1,AL2,AL3,TH1DEG
10555   FORMAT(I6,6G13.5)
      END IF
   60 CONTINUE
      IF (ASUM.NE.0.0) THEN
C
C       Average over the polygon elements
C         If using the convention that no deformation is
C         indicated by a strain of zero, subtract 1
C         This will give values greater than zero if the
C         polygon has been deformed, otherwise zero
        AL1AVG = AL1SUM/ASUM - 1
        AL2AVG = AL2SUM/ASUM - 1
        AL = MAX(AL2AVG,AL1AVG)
        FAVG(1) = DFAVGSUM(1)/ASUM - 1
        FAVG(2) = DFAVGSUM(2)/ASUM
        FAVG(3) = DFAVGSUM(3)/ASUM
        FAVG(4) = DFAVGSUM(4)/ASUM - 1
      ENDIF
      RETURN
      END

      SUBROUTINE STRAINPNT(IM,XPT,YPT,EX,EY,EXREF,EYREF,
     :                     LEM,NOR,IPOLYNUM,
     :                     FAVG,AL,NE,NUP,IVERBOSE,IERR)
C
C     STRAINPNT calculates an area averaged deformation for
C     the mesh element which contains XPT,YPT.
C     Only valid for small strain increments.
C     (TRI is twice element area)
C
C
C     Y1 CONTAINS COORDINATES OF UNDEFORMED ELEMENT
C
C     Y2 CONTAINS COORDINATES OF DEFORMED   ELEMENT
C
C     ORDER OF COORDINATES: X(PT1),Y(PT1), ....... ,X(PT3),Y(PT3)
C
C     NNN ALLOWS CHOICE OF RELATIVE POSITION VECTORS TO
C
C     BE PLACED IN X1 AND X2
C
C     NNN = 1:      (PT2 - PT1, PT3 - PT2) (PT2 is reference pt)
C
C     NNN = 2:      (PT2 - PT1, PT3 - PT1) (PT1 is reference pt)
C
C     NNN = 3:      (PT3 - PT2, PT3 - PT1) (PT3 is reference pt)
C
C
      DIMENSION EX(NUP),EY(NUP),EXREF(NUP),EYREF(NUP)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION FAVG(4),IPOLYNUM(NE)
C     COMMON/CONTRO/LUW,LRBAT,LRINT,LDIN,LWINT,LPLOT,IAUTO,ILABEL
C    1 ,ICOLO
      DIMENSION X1(4),X2(4),F(4),N1(4),N2(4)
      DOUBLE PRECISION Y1(6),Y2(6),PNT(2),XPT,YPT
C     DIMENSION Y1(6),Y2(6),PNT(2)
C     DOUBLE PRECISION XPT,YPT
      DOUBLE PRECISION TRI,DXP(3),DYP(3),ASUM
      DOUBLE PRECISION DFAVG(4), DFAVGSUM(4)
      DATA PI /3.14159/
C
      IERR=0
      ASUM = 0.0
      AL1SUM = 0.0
      AL2SUM = 0.0
      DO 10 N=1,4
  10  DFAVGSUM(N) = 0.0
      DO 15 N=1,6
  15  Y1(N) = 0.0
      PNT(1) = XPT
      PNT(2) = YPT

C
C    For the triangle containing the point (first triangle,
C     if it lies on an element boundary
C

      DO 60 N=1,NE
C     IF (IPOLYNUM(N).NE.IM) GO TO 60
C
C    Put the element coordinates into the Y1 and Y2 matrices
C

      DO 20 K=1,3
      LK=NOR(LEM(K,N))
      JJ=2*(K-1)+1
      Y1(JJ)=EXREF(LK)
      Y1(JJ+1)=EYREF(LK)
      Y2(JJ)=EX(LK)
      Y2(JJ+1)=EY(LK)
      DXP(K)=EX(LK)
      DYP(K)=EY(LK)
   20 CONTINUE
      CALL CROSSINGSTEST(Y1,3,PNT,INSIDE)
      IF (INSIDE.EQ.0) GO TO 60
C        WRITE(6,10550)N,XPT,YPT
C10550   FORMAT('element',I4,' contains point ',2G13.5)
C
C    TRI is twice the area of the triangle element
C
      TRI=DXP(2)*DYP(3)-DXP(3)*DYP(2)+DXP(3)*DYP(1)-DXP(1)*DYP(3)
     :                +DXP(1)*DYP(2)-DXP(2)*DYP(1)
      ASUM = ASUM + TRI

      NNN=1
      DO 1 I = 1,4
      N1(I) = I + 2
 1    N2(I) = I
      GOTO (2,3,4) NNN
 3    N2(3) = 1
      N2(4) = 2
      GOTO 2
 4    N1(1) = 5
      N1(2) = 6
 2    DO 5 I = 1,4
      X1(I) = Y1(N1(I)) - Y1(N2(I))
 5    X2(I) = Y2(N1(I)) - Y2(N2(I))
C
C     CALCULATE ELEMENTS OF STRAIN TENSOR
C
C       Calculated as:
C       F(1) = Exx = du/dx+1
C       F(2) = Exy = du/dy
C       F(3) = Eyx = dv/dx
C       F(4) = Eyy = dv/dy+1
C       If there is no deformation, strain ellipse is a circle with
C       radius of 1 and F(1) = F(4) = 1
      DET = X1(1)*X1(4) - X1(2)*X1(3)
C       update sum over the elements
      DFAVG(1) = (X2(1)*X1(4) - X1(2)*X2(3))/DET
      DFAVG(2) = (X1(1)*X2(3) - X2(1)*X1(3))/DET
      DFAVG(3) = (X2(2)*X1(4) - X1(2)*X2(4))/DET
      DFAVG(4) = (X1(1)*X2(4) - X2(2)*X1(3))/DET
      DFAVGSUM(1) = DFAVGSUM(1) + DFAVG(1) * TRI
      DFAVGSUM(2) = DFAVGSUM(2) + DFAVG(2) * TRI
      DFAVGSUM(3) = DFAVGSUM(3) + DFAVG(3) * TRI
      DFAVGSUM(4) = DFAVGSUM(4) + DFAVG(4) * TRI
 
C     CALCULATE ORIENTATION OF PRINCIPAL AXES
C
C       AL1 and AL2 are STRAIN "in plane" MIN and MAX
C       AL3 is STRAIN vertical
C
      BOT = DFAVG(1)*DFAVG(1) + DFAVG(3)*DFAVG(3)
     :       - DFAVG(2)*DFAVG(2) - DFAVG(4)*DFAVG(4)
      TOP = 2.*(DFAVG(1)*DFAVG(2) + DFAVG(3)*DFAVG(4))
      THETA1 = PI/2
      IF(ABS(BOT).GT.1.E-37) THETA1 = 0.5*ATAN(TOP/BOT)
      C = COS(THETA1)
      S = SIN(THETA1)
      AL1 = SQRT((DFAVG(1)*C + DFAVG(2)*S)**2
     :             + (DFAVG(3)*C + DFAVG(4)*S)**2)
C       sum over the elements
      AL1SUM = AL1SUM + AL1 * TRI
      TOP = DFAVG(3)*C + DFAVG(4)*S
      BOT = DFAVG(1)*C + DFAVG(2)*S
      THETA2 = THETA1 + PI/2
      C = COS(THETA2)
      S = SIN(THETA2)
      AL2 = SQRT((DFAVG(1)*C + DFAVG(2)*S)**2
     :             + (DFAVG(3)*C + DFAVG(4)*S)**2)
      AL2SUM = AL2SUM + AL2 * TRI
      AL3 = 1.0/(AL1*AL2)
C
C     Calculate orientation of principal axes after deformation
C
      TH1 = PI/2
      IF(ABS(BOT).GT.1.E-37) TH1 = ATAN(TOP/BOT)
      TH2 = TH1 + PI/2
      ROT = TH1 - THETA1
      IF(AL2.GT.AL1)DEF=ALOG10(AL2/AL1)
      IF(AL2.LE.AL1)DEF=ALOG10(AL1/AL2)
      NC=2*(N-1) + 1
      XO = (Y2(1) + Y2(3) + Y2(5))/3
      YO = (Y2(2) + Y2(4) + Y2(6))/3
      IF (IVERBOSE.NE.0) THEN
        TH1DEG=TH1*180.0/PI
        WRITE(6,10555)N,XO,YO,AL1,AL2,AL3,TH1DEG
10555   FORMAT(I6,6G13.5)
      END IF
      GO TO 61
   60 CONTINUE
   61 IF (ASUM.NE.0.0) THEN
C
C       Average over the polygon elements
C         If using the convention that no deformation is
C         indicated by a strain of zero, subtract 1
C         This will give values greater than zero if the
C         polygon has been deformed, otherwise zero
        AL1AVG = AL1SUM/ASUM - 1
        AL2AVG = AL2SUM/ASUM - 1
        AL = MAX(AL2AVG,AL1AVG)
        FAVG(1) = DFAVGSUM(1)/ASUM - 1
        FAVG(2) = DFAVGSUM(2)/ASUM
        FAVG(3) = DFAVGSUM(3)/ASUM
        FAVG(4) = DFAVGSUM(4)/ASUM - 1
      ENDIF
      IF (INSIDE.EQ.0) THEN
C        WRITE(6,10556)XPT,YPT
C10556   FORMAT('No element contains point ',2G15.6)
        IERR=1
      END IF
      RETURN
      END

      SUBROUTINE MATCHMESHNODE(INDX,IM,XPT,YPT,
     :                     EXREF,EYREF,
     :                     LEM,NOR,IPOLYNUM,
     :                     NE,NUP,IERR)
C
C     MATCHMESHNODE finds the mesh node which is closest to XPT,YPT.
C     Only the elements with polynum equal to IM are used.
C
C
      DIMENSION EXREF(NUP),EYREF(NUP)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION IPOLYNUM(NE)
      DOUBLE PRECISION XPT,YPT
      DOUBLE PRECISION DPX,DPY
      DOUBLE PRECISION PNT(2),Y1(6)
      LOGICAL ISET
C
      IERR=0
      VOFF = 0.0
      INDX=-1

C
C    For the triangle containing the point (first, if it lies
C     on a boundary
C
      X=XPT
      Y=YPT
      PNT(1)=XPT
      PNT(2)=YPT
      EPS = 1E-6
      DIST=0
      CURRDIST=0
      ISET=.FALSE.
      DO 60 N=1,NE
        IF (IPOLYNUM(N).NE.IM) GO TO 60
C
C    This should only be done once if unode pt was an input to 
C    triangulation
C

        DO 10 I=1,3
          LK=NOR(LEM(I,N))
          JJ=2*(I-1)+1
          Y1(JJ)=EXREF(LK)
          Y1(JJ+1)=EYREF(LK)
C    Warning! this is assuming vertical boundaries are periodic and
C    width is 1!!!!!
          IF (DABS((Y1(JJ)-XPT)).GT.0.5)
     :    CALL ellepointplotxy(Y1(JJ),Y1(JJ+1),XPT,YPT)
  10    CONTINUE
        CALL CROSSINGS(Y1,3,XPT,YPT,INSIDE)
C       CALL CROSSINGSTEST(Y1,3,PNT,INSIDE)
        IF (INSIDE.EQ.0.AND. XPT.NE.0) GO TO 60
C    Warning! bug in CrossingsTest - counts 2 crossings if pt on x=0
C    ie x=0 segment and side crossed by +ve ray
C    Fix this (and check it does not break other cases!) or try Basil
C    CNL test
        IF (XPT.EQ.0) THEN
          JJ=1
          KK=JJ+2
          CALL POINTONSEGMENT(Y1(JJ),Y1(JJ+1),Y1(KK),Y1(KK+1),
     :                         XPT,YPT,INSIDE)
          IF (INSIDE.EQ.0) THEN
          JJ=3
          KK=JJ+2
          CALL POINTONSEGMENT(Y1(JJ),Y1(JJ+1),Y1(KK),Y1(KK+1),
     :                         XPT,YPT,INSIDE)
          END IF
          IF (INSIDE.EQ.0) THEN
          JJ=5
          KK=1
          CALL POINTONSEGMENT(Y1(JJ),Y1(JJ+1),Y1(KK),Y1(KK+1),
     :                         XPT,YPT,INSIDE)
          ENDIF
        ENDIF
        IF (INSIDE.EQ.0) GO TO 60
        DO 20 K=1,6
          LK=NOR(LEM(K,N))
          FPX=EXREF(LK)
          FPY=EYREF(LK)
          DPX = FPX
          DPY = FPY
          CALL ellepointplotxy(DPX,DPY,XPT,YPT)
          FPX = DPX
          FPY = DPY
          CURRDIST=SQRT((FPX-X)*(FPX-X) +
     :                 (FPY-Y)*(FPY-Y))
          IF (.NOT.ISET .OR. CURRDIST.LT.DIST) THEN
            DIST=CURRDIST
            INDX=LK
            ISET=.TRUE.
          ENDIF
   20   CONTINUE
        GOTO 80
   60 CONTINUE
C     if dist>? then unode is in centre of tri not one of the nodes
   80 IF (DIST.GT.9E-3) THEN
        PRINT *,IM,'No matching node ',XPT,YPT,DIST
C       IERR=1
      END IF
      IF (INDX.EQ.-1) THEN
C       PRINT *,'No element contains point ',IM,XPT,YPT
       
      END IF
      RETURN
      END

      SUBROUTINE NTRPLTPNT(VAL,S,IM,XPT,YPT,
     :                     EX,EY,EXREF,EYREF,
     :                     LEM,NOR,IPOLYNUM,
     :                     NE,NUP,IERR)
C
C     NTRPLTPNT finds the mesh element which contains XPT,YPT.
C     The value of S is interpolated for XPT, YPT and placed in VAL.
C     Only the elements with polynum equal to IM are used.
C     (TRI is twice element area)
C
C
      DIMENSION EX(NUP),EY(NUP),EXREF(NUP),EYREF(NUP),S(NUP)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION IPOLYNUM(NE)
      DIMENSION IPREVTRI(3)
      DOUBLE PRECISION XPT,YPT,PNT(2)
      DOUBLE PRECISION A0(3),BY(3),CX(3),COOL(3)
      DOUBLE PRECISION TRI,DPX(3),DPY(3),Y1(6),Y2(6)
C
      IERR=0
      VOFF = 0.0
      PNT(1) = XPT
      PNT(2) = YPT
      DO 15 N=1,6
  15  Y1(N) = 0.0

C
C    For the triangle containing the point (first, if it lies
C     on a boundary
C
      DO 16 N=1,3
  16  IPREVTRI(N) = 0

      ICMN=0
      DO 60 N=1,NE
      IF (IPOLYNUM(N).NE.IM) GO TO 60
C
C    Put the element coordinates into the Y1 matrix
C

      IF (ICMN.EQ.0) ICMN=LEM(1,N)
      DO 20 K=1,3
      LK=NOR(LEM(K,N))
      JJ=2*(K-1)+1
      Y1(JJ)=EXREF(LK)
      Y1(JJ+1)=EYREF(LK)
      DPX(K)=EX(LK)
      DPY(K)=EY(LK)
      A0(K)=DPX(2)*DPY(3) - DPX(3)*DPY(2)
      BY(K)=DPY(2)    - DPY(3)
      CX(K)=DPX(3)    - DPX(2)
      TRI=TRI+(DPX(2)*DPY(3)-DPX(3)*DPY(2))
   20 CONTINUE
      CALL CROSSINGSTEST(Y1,3,PNT,INSIDE)
      IF (INSIDE.EQ.0) GO TO 60
C        WRITE(6,10550)N,XPT,YPT
C10550   FORMAT('element',I4,' contains point ',2G13.5)
C
C    TRI is twice the area of the triangle element
C
      TRI=DPX(2)*DPY(3)-DPX(3)*DPY(2)+DPX(3)*DPY(1)-DPX(1)*DPY(3)
     :                +DPX(1)*DPY(2)-DPX(2)*DPY(1)
C
C     Calculate the natural coordinates
C
              DO 45 K=1,3
                CNL=(A0(K) + XP*BY(K) + YP*CX(K))/TRI
C               IF((CNL.GT.1.0+EPSA).OR.(CNL.LT.-EPSA))GO TO 50
   45         COOL(K)=CNL
C
C     If we reach here the point is within the triangle, so
C         interpolate
C
              VAL=0.0
              DO 35 K=1,6
                LK=LEM(K,N)
                IF(LK.LT.0)LK=-LK
                SK=S(LK)+VOFF
                IF(K.GE.4)GO TO 30
                K2=MOD(K,3)+1
                K3=MOD(K+1,3)+1
                CNK=COOL(K)*(COOL(K) - COOL(K2) - COOL(K3))
                GO TO 35
   30           K2=K-3
                K3=MOD(K+1,3)+1
                CNK=4.0*COOL(K2)*COOL(K3)
   35         VAL=VAL + SK*CNK
   60 CONTINUE
      IF (INSIDE.EQ.0) THEN
C       WRITE(6,105)XPT,YPT
C105    FORMAT('No element contains point ',2G15.6)
        IERR=1
      END IF
      RETURN
      END


      SUBROUTINE CROSSINGS(POLYG,NPTS,XP,YP,NCROSSINGS)

      DOUBLE PRECISION POLYG(2,NPTS),XP,YP
      DOUBLE PRECISION XP1,YP1,XP2,YP2
C
C    check the no. of polygon edge crossings made by a ray along
C    the +X axis. See Graphics Gems IV, 1.4, Ed. Paul S. Heckbert, 1994
            XP1=POLYG(1,NPTS)
            YP1=POLYG(2,NPTS)
            XP2=POLYG(1,1)
            YP2=POLYG(2,1)
            IYFLG0=0
            IF (YP1.GE.YP) IYFLG0=1
            NCROSSINGS=0
            DO 35 J=1,NPTS
C    check if segment endpts are on opposite sides of (XP,YP)
              IF (YP2.GE.YP) THEN
                IYFLG1=1
              ELSE
                IYFLG1=0
              END IF
              IF (IYFLG0.NE.IYFLG1) THEN
                IF (XP1.GE.XP) THEN
                  IXFLG0=1
                ELSE
                  IXFLG0=0
                END IF
C    check if segment endpts are on the same side of y=YP
                IF ((IXFLG0.EQ.0.AND.XP2.LT.XP).OR.
     1             (IXFLG0.EQ.1.AND.XP2.GE.XP)) THEN
                  IF (IXFLG0.EQ.1) THEN
                    IF (IYFLG0.EQ.1) NCROSSINGS = NCROSSINGS-1
                    IF (IYFLG0.EQ.0) NCROSSINGS = NCROSSINGS+1
                  END IF
                ELSE
C    compute intersection of segment and +X ray
C    if >= XP then ray hits
                  IF ((XP2-(YP2-YP)*(XP1-XP2)/(YP1-YP2)).GE.XP)
     1                                                   THEN
                    IF (IYFLG0.EQ.1) NCROSSINGS = NCROSSINGS-1
                    IF (IYFLG0.EQ.0) NCROSSINGS = NCROSSINGS+1
                  END IF
                END IF
              END IF
              IYFLG0 = IYFLG1
              XP1 = XP2
              YP1 = YP2
              IF (J.LE.NPTS) THEN
                XP2=POLYG(1,J+1)
                YP2=POLYG(2,J+1)
              END IF
   35       CONTINUE
      RETURN
      END

      SUBROUTINE STRESSAVG(IM,IQU,ASTRESS,
     :                  SE,ARGAN,BRGAN,HLENSC,BDEPSC,BIG,EX,EY,
     :                  VHB,UVP,SSQ,IELFIX,LEM,NOR,IPOLYNUM,
     :                  NE,NUP,NUVP,NFP,IVV,NCOMP,ICR,IVERBOSE)
      DIMENSION EX(NUP),EY(NUP),VHB(7,NE)
      DIMENSION LEM(6,NE),NOR(NUP)
      DIMENSION IPOLYNUM(NE)

C
C    For each triangle in the polygon
C

      ASTRESS = 0.0
      ASUM = 0.0
      SSUM = 0.0
      IERR = 0
      DO 60 N=1,NE
        IF (IPOLYNUM(N) .NE. IM) GOTO 60
        CALL STRAIC(N,IQU,CVALUE,AREA2,IVERBOSE,IERR,
     1                  SE,ARGAN,BRGAN,HLENSC,BDEPSC,BIG,EX,EY,
     2                  VHB,UVP,SSQ,IELFIX,LEM,NOR,NE,NUP,NUVP,
     3                  NFP,IVV,NCOMP,ICR)
        SSUM = SSUM + CVALUE*AREA2
        ASUM = ASUM + AREA2
   60 CONTINUE
      IF ((ASUM.NE.0.0) .AND. (IERR .EQ. 0)) THEN
        ASTRESS = SSUM/ASUM
      END IF
      RETURN
      END
