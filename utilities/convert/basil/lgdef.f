C*--------------------------------------------------------------------
C*    Basil / Sybil:   lgmesh.f  1.1  1 October 1998
C*
C*    Copyright (c) 1997 by G.A. Houseman, T.D. Barr, & L.A. Evans
C*    See README file for copying and redistribution conditions.
C*--------------------------------------------------------------------

      SUBROUTINE LGDEF(DT,UVP,UXLG,UYLG,UVELPX,UVELPY,
     :                 EX,EY,EXLG,EYLG,STELPX,STELPY,
     :                 LEM,NOR,
     :                 NUP,NE,NROWS,NUL,NNL,NML,NSM,NPM,
     :                 ILAG,IFLT)
C
C    Routine to calculate the position of the lagrangian nodes.
C    The lagrangian nodes are used to determine the strain or
C    total deformation.
C
C     DOUBLE PRECISION UVP
      DOUBLE PRECISION TRIA2,A0(3),DPEX2,DPEX3,DPEY2,DPEY3
      DIMENSION UVP(NROWS)
      DIMENSION UXLG(NUL),UYLG(NUL)
      DIMENSION EXLG(NUL),EYLG(NUL)
      DIMENSION UVELPX(NPM,NSM),UVELPY(NPM,NSM)
      DIMENSION STELPX(NPM,NSM),STELPY(NPM,NSM)
      DIMENSION LEM(6,NE)
      DIMENSION NOR(NUP)
      DIMENSION EX(NUP)
      DIMENSION EY(NUP)
      DIMENSION BY0(3),CX0(3),COOL(3)

      DIMENSION BBOX(2,4)
      DOUBLE PRECISION DBBOX(8), DPT(2)
C
C     EPS=1.E-8
      EPS=1.E-5
      DO 5 I=1,NUL
        UXLG(I)=-999
        UYLG(I)=-999
  5   CONTINUE

      CALL FINDBBOX(EX,EY,BBOX,NUP)
      XLEN = BBOX(1,2)-BBOX(1,1)
      XMINR = BBOX(1,2)
      IF (BBOX(1,3).LT.XMINR) THEN
        XMAXR = XMINR
        XMINR = BBOX(1,3)
      ELSE
        XMAXR = BBOX(1,3)
      ENDIF
      XMAXL = BBOX(1,1)
      IF (BBOX(1,4).LT.XMAXL) THEN
        XMINL = XMAXL
        XMAXL = BBOX(1,4)
      ELSE
        XMINL = BBOX(1,4)
      ENDIF
      DBBOX(1) = BBOX(1,1)
      DBBOX(2) = BBOX(2,1)
      DBBOX(3) = BBOX(1,2)
      DBBOX(4) = BBOX(2,2)
      DBBOX(5) = BBOX(1,3)
      DBBOX(6) = BBOX(2,3)
      DBBOX(7) = BBOX(1,4)
      DBBOX(8) = BBOX(2,4)
      NVERT = 4
C
C   update the velocity of each lagrangian point (uxlg and uylg)
C    first, search through every element in the finite element mesh
C
      DO 100 JEL=1,NE
C
C     First calculate the geometrical coefficients for JEL
C
         TRIA2=0.0
         DO 120 KK=1,3
            K2=MOD(KK,3)+1
            K3=MOD(KK+1,3)+1
            LK=NOR(LEM(KK,JEL))
            LK2=NOR(LEM(K2,JEL))
            LK3=NOR(LEM(K3,JEL))
            X1=EX(LK)
            X2=EX(LK2)
            X3=EX(LK3)
            Y1=EY(LK)
            Y2=EY(LK2)
            Y3=EY(LK3)
            DPEX2 = X2
            DPEX3 = X3
            DPEY2 = Y2
            DPEY3 = Y3
            A0(KK)=DPEX2*DPEY3-DPEX3*DPEY2
            BY0(KK)=Y2-Y3
            CX0(KK)=X3-X2
            TRIA2=TRIA2+A0(KK)
  120    CONTINUE
C
C     Now calculate the natural coordinates of Lagrange pt IL
C      CNL must be between 0 and 1 if NKJ is inside the element
C      We assume no points cross the fault - i.e. if JEL <= NEP/2,
C      we only look at the points <= NNLP/2 or NNLP<=points<=NNLP+NMLP/2
C      and if JEL > NEP/2, we look at the NNLP/2+1<=points<NNLP or
C      points > NNLP+NMLP/2
C
      IF((ILAG.EQ.1).OR.(ILAG.EQ.3))THEN
         DO 160 IL=1,NUL
            LFLT=0
C            these lines only apply to centred internal faults
            IF (IFLT.EQ.2) THEN
              IF(IL.GT.NNL/2.AND.IL.LE.NNL) LFLT=1
              IF(IL.GT.NNL+NML/2) LFLT=1
              IF(JEL.GT.NE/2.AND.LFLT.EQ.0) GOTO 160
              IF(JEL.LE.NE/2.AND.LFLT.EQ.1) GOTO 160
            END IF
            XP=EXLG(IL)
            YP=EYLG(IL)
C
C    these checks for elle runs using lg mesh to track finite strain
            IF (IFLT.EQ.1.AND.XP.GT.XMINR) THEN
C    use XMINL to stop endless loop if test failing
              DPT(1) = XP
              DPT(2) = YP
   10         CALL CROSSINGSTEST(DBBOX,NVERT,DPT,IRES)
              IF (IRES.EQ.0.AND.XP.GT.XMINL) THEN
                XP = XP-XLEN
                DPT(1) = XP
                GOTO  10
              ENDIF
            ELSE IF (IFLT.EQ.1.AND.XP.LT.XMAXL) THEN
              DPT(1) = XP
              DPT(2) = YP
   20         CALL CROSSINGSTEST(DBBOX,NVERT,DPT,IRES)
              IF (IRES.EQ.0.AND.XP.LT.XMAXR) THEN
                XP = XP+XLEN
                DPT(1) = XP
                GOTO  20
              ENDIF
            ENDIF

            DO 130 KK=1,3
               CNL=(A0(KK) + XP*BY0(KK) + YP*CX0(KK))/TRIA2
               IF((CNL.GT.1.0+EPS).OR.(CNL.LT.-EPS))GO TO 160
               COOL(KK)=CNL
  130       CONTINUE
C
C     If we reach here the point is within the triangle, so
C     get the interpolated values
C
            UXT=0.0
            UYT=0.0
            DO 135 KK=1,3
               LK=LEM(KK,JEL)
               UXV=UVP(LK)
               UYV=UVP(LK+NUP)
               K2=MOD(KK,3)+1
               K3=MOD(KK+1,3)+1
               CNK=COOL(KK)*(COOL(KK) - COOL(K2) - COOL(K3))
               UXT=UXT + UXV*CNK
               UYT=UYT + UYV*CNK
  135       CONTINUE
            DO 140 KK=4,6
               LK=IABS(LEM(KK,JEL))
               UXV=UVP(LK)
               UYV=UVP(LK+NUP)
               K2=KK-3
               K3=MOD(KK+1,3)+1
               CNK=4.0*COOL(K2)*COOL(K3)
               UXT=UXT + UXV*CNK
               UYT=UYT + UYV*CNK
  140       CONTINUE
            UXLG(IL)=UXT
            UYLG(IL)=UYT
  160    CONTINUE
       END IF
C
C  Now update the lagrangian strain markers
C
         IF (ILAG.EQ.1.OR.ILAG.EQ.2) THEN
           DO 170 JL=1,NSM
           DO 180 IL=1,NPM
              XP=STELPX(IL,JL)
              YP=STELPY(IL,JL)
              DO 185 KK=1,3
                 CNL=(A0(KK) + XP*BY0(KK) + YP*CX0(KK))/TRIA2
                 IF((CNL.GT.1.0+EPS).OR.(CNL.LT.-EPS))GO TO 180
                 COOL(KK)=CNL
  185         CONTINUE
C
C     If we reach here the point is within the triangle, so
C     get the interpolated values
C
              UXT=0.0
              UYT=0.0
              DO 190 KK=1,3
                 LK=LEM(KK,JEL)
                 UXV=UVP(LK)
                 UYV=UVP(LK+NUP)
                 K2=MOD(KK,3)+1
                 K3=MOD(KK+1,3)+1
                 CNK=COOL(KK)*(COOL(KK) - COOL(K2) - COOL(K3))
                 UXT=UXT + UXV*CNK
                 UYT=UYT + UYV*CNK
  190         CONTINUE
              DO 195 KK=4,6
                 LK=IABS(LEM(KK,JEL))
                 UXV=UVP(LK)
                 UYV=UVP(LK+NUP)
                 K2=KK-3
                 K3=MOD(KK+1,3)+1
                 CNK=4.0*COOL(K2)*COOL(K3)
                 UXT=UXT + UXV*CNK
                 UYT=UYT + UYV*CNK
  195         CONTINUE
              UVELPX(IL,JL)=UXT
              UVELPY(IL,JL)=UYT
  180      CONTINUE
  170      CONTINUE
         END IF
  100 CONTINUE
      DO 50 I=1,NUL
        IF (UXLG(I).EQ.-999) THEN
          print *,'index=',I,' not set'
          UXLG(I)=0
          UYLG(I)=0 
          IERR=1
        ENDIF
  50  CONTINUE
C
C   now update the position of each lagrangian mesh point (exlg and eylg)
C
      IF((ILAG.EQ.1).OR.(ILAG.EQ.3))THEN
      DO 200 I=1,NUL
         EXLG(I)=EXLG(I)+UXLG(I)*DT
         EYLG(I)=EYLG(I)+UYLG(I)*DT
  200 CONTINUE
      END IF
C
C   now update the position of each marker point (stelpx and stelpy)
C
      IF (ILAG.EQ.1.OR.ILAG.EQ.2) THEN
        DO 220 JL=1,NSM
           DO 210 IL=1,NPM
              STELPX(IL,JL)=STELPX(IL,JL)+UVELPX(IL,JL)*DT
              STELPY(IL,JL)=STELPY(IL,JL)+UVELPY(IL,JL)*DT
  210      CONTINUE
  220    CONTINUE
      END IF
C
      RETURN
      END



      SUBROUTINE FINDBBOX(EX,EY,BBOX,NUP)
      DIMENSION EX(NUP), EY(NUP), BBOX(2,4)

C    finds the corners of the bounding box (assuming y values
C    for top and bottom boundaries are constant). Stores the
C    coordinates in BBOX starting at the bottom left and in
C    anticlockwise order

      EPS = 1.0E-5
      BBOX(2,1) = EY(1)
      BBOX(2,2) = EY(1)
      BBOX(2,3) = EY(1)
      BBOX(2,4) = EY(1)
      BBOX(1,1) = EX(1)
      BBOX(1,2) = EX(1)
      BBOX(1,3) = EX(1)
      BBOX(1,4) = EX(1)
      DO 10 I=2,NUP
        IF (EY(I).GT.BBOX(2,3)) BBOX(2,3) = EY(I)
        IF (EY(I).LT.BBOX(2,2)) BBOX(2,2) = EY(I)
  10  CONTINUE
      BBOX(2,1) = BBOX(2,2)
      BBOX(2,4) = BBOX(2,3)
      YMIN = BBOX(2,2)+EPS
      YMAX = BBOX(2,3)-EPS
      ISETMIN1 = 0
      ISETMIN2 = 0
      ISETMAX1 = 0
      ISETMAX2 = 0
      DO 20 I=1,NUP
        IF (EY(I).LT.YMIN) THEN
          IF (ISETMIN1.EQ.0) THEN
            BBOX(1,1) = EX(I)
            ISETMIN1 = 1
          ELSE IF (EX(I).LT.BBOX(1,1)) THEN
            BBOX(1,1) = EX(I)
          ENDIF
          IF (ISETMAX1.EQ.0) THEN
            BBOX(1,2) = EX(I)
            ISETMAX1 = 1
          ELSE IF (EX(I).GT.BBOX(1,2)) THEN
            BBOX(1,2) = EX(I)
          ENDIF
        ELSE IF (EY(I).GT.YMAX) THEN
          IF (ISETMIN2.EQ.0) THEN
            BBOX(1,4) = EX(I)
            ISETMIN2 = 1
          ELSE IF (EX(I).LT.BBOX(1,4)) THEN
            BBOX(1,4) = EX(I)
          ENDIF
          IF (ISETMAX2.EQ.0) THEN
            BBOX(1,3) = EX(I)
            ISETMAX2 = 1
          ELSE IF (EX(I).GT.BBOX(1,3)) THEN
            BBOX(1,3) = EX(I)
          ENDIF
        END IF
  20  CONTINUE
      RETURN
      END
