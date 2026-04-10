C*--------------------------------------------------------------------
C*    Basil / Sybil:   profil.f  1.1  1 October 1998
C*
C*    Copyright (c) 1997 by G.A. Houseman, T.D. Barr, & L.A. Evans
C*    See README file for copying and redistribution conditions.
C*--------------------------------------------------------------------
C
      SUBROUTINE NTRPLN(S,RLINE,NL,SS0,XP1,YP1,XP2,YP2,
     :                   TBXOFF,TBYOFF,VOFFSET,
     :                   EX,EY,LEM,NOR,ILINE,NE,NUP,NP3,
     :                   IVERBOSE)
C
C
C    Routine to interpolate the nodal values from the finite
C     element mesh onto a line traversing the solution region
C     NTRPLN uses quadratic interpolation (e.g. velocity)
C     NTRPLN2 uses linear interpolation (e.g. pressure)
C
      DIMENSION EX(NUP),EY(NUP),LEM(6,NE),NOR(NUP)
      DIMENSION S(NUP),RLINE(NL),ILINE(NP3)
C     DIMENSION BY(3),CX(3),COOL(3)
      DOUBLE PRECISION BY(3),CX(3),COOL(3)
      DOUBLE PRECISION A0(3),TRI,DPX1,DPX2,DPX3,DPY1,DPY2,DPY3
      DATA EPS/1.0E-3/
      SAVE EPS 
C
C   Zero the arrays
C   RLINE(NL) contains the intrpolated values on the line
C   ILINE(NL) contains a switch indicating a value in RLINE
C   Line runs between (XP1,YP1) and (XP2,YP2)
C
      DO 10 I=1,NL
      ILINE(I)=0
   10 RLINE(I)=SS0
      DXP=(XP2-XP1)
      DYP=(YP2-YP1)
      IF (NL.GT.1) THEN
        DXP=(XP2-XP1)/FLOAT(NL-1)
        DYP=(YP2-YP1)/FLOAT(NL-1)
      END IF
      EPSX=ABS(DXP*0.1)
      EPSY=ABS(DYP*0.1)
C
C    Look at each element in turn
C
      DO 60 N=1,NE
C
C     Calculate the geometrical coefficients
C
   12 XMIN=999.
      YMIN=999.
      XMAX=-999.
      YMAX=-999.
      DO 20 K=1,3
      K2=MOD(K,3)+1
      K3=MOD(K+1,3)+1
      LK=NOR(LEM(K,N))
      LK2=NOR(LEM(K2,N))
      LK3=NOR(LEM(K3,N))
      X1=EX(LK)
      X2=EX(LK2)
      X3=EX(LK3)
      Y1=EY(LK)
      Y2=EY(LK2)
      Y3=EY(LK3)
      XMIN=MIN(XMIN,X1)
      XMAX=MAX(XMAX,X1)
      YMIN=MIN(YMIN,Y1)
      YMAX=MAX(YMAX,Y1)
      DPX1=X1
      DPX2=X2
      DPX3=X3
      DPY1=Y1
      DPY2=Y2
      DPY3=Y3
      A0(K)=DPX2*DPY3 - DPX3*DPY2
      BY(K)=Y2    - Y3
      CX(K)=X3    - X2
   20 CONTINUE
C
C    TRI is twice the area of the triangle element
C
      TRI=DPX2*DPY3-DPX3*DPY2+DPX3*DPY1-DPX1*DPY3+DPX1*DPY2-DPX2*DPY1
C
C     Now look for the points on the line that are inside the tri.
C
      XP = XP1-DXP
      YP = YP1-DYP
      XOFF=0.0
      YOFF=0.0
      VOFF=0.0
      DO 80 J=1,NL
      XP=XP+DXP-XOFF
      YP=YP+DYP-YOFF
      IF((YP.GT.YMAX+EPSY).OR.(YP.LT.YMIN-EPSY).OR.
     1   (XP.GT.XMAX+EPSX).OR.(XP.LT.XMIN-EPSX).OR.
     2   (ILINE(J).NE.0)) GO TO 80
C
C     Calculate the natural coordinates
C
      DO 45 K=1,3
      CNL=(A0(K) + XP*BY(K) + YP*CX(K))/TRI
      IF((CNL.GT.1.0+EPS).OR.(CNL.LT.-EPS))GO TO 80
   45 COOL(K)=CNL
C
C     If we reach here the point is within the triangle, so
C         Interpolate !
C
      SI=0.0
      DO 35 K=1,6
      LK=LEM(K,N)
      IF(LK.LT.0)LK=-LK
      SK=S(LK)
      IF(K.GE.4)GO TO 30
      K2=MOD(K,3)+1
      K3=MOD(K+1,3)+1
      CNK=COOL(K)*(COOL(K) - COOL(K2) - COOL(K3))
      GO TO 35
   30 K2=K-3
      K3=MOD(K+1,3)+1
      CNK=4.0*COOL(K2)*COOL(K3)
   35 SI=SI + SK*CNK
      RLINE(J)=SI
      ILINE(J)=N
   80 CONTINUE
   60 CONTINUE
C
      IF (IVERBOSE.NE.0) THEN
      WRITE(6,*)' data values from NTRPLN, between '
      WRITE(6,*)'    (',XP1,',',YP1,') and (',XP2,',',YP2,')'
      END IF
C
      ICNT = 0
      DO 100 I=1,NL
        IF (ILINE(I).NE.0) ICNT = ICNT+1
 100  CONTINUE
      IF (ICNT.NE.NL) print *, ICNT
      RETURN
      END

      SUBROUTINE NTRPLN2(S,RLINE,NL,SS0,XP1,YP1,XP2,YP2,
     :                   TBXOFF,TBYOFF,VOFFSET,
     1                   EX,EY,LEM,NOR,NE,NUP,NN,NP3,IVERBOSE)
C
C
C    Routine to interpolate the nodal values from the finite
C     element mesh onto a line traversing the solution region
C     NTRPLN uses quadratic interpolation (e.g. velocity)
C     NTRPLN2 uses linear interpolation (e.g. pressure)
C
      DIMENSION EX(NUP),EY(NUP),LEM(6,NE),NOR(NUP)
      DIMENSION S(NN),RLINE(NL)
C     DIMENSION BY(3),CX(3),COOL(3)
      DOUBLE PRECISION BY(3),CX(3),COOL(3)
      DOUBLE PRECISION A0(3),TRI,DPX1,DPX2,DPX3,DPY1,DPY2,DPY3
      DATA EPS/1.0E-3/
      SAVE EPS 
C
C   Zero the arrays
C   RLINE(NL) contains the intrpolated values on the line
C   ILINE(NL) not used in this routine
C   Line runs between (XP1,YP1) and (XP2,YP2)
C
      DO 10 I=1,NL
   10 RLINE(I)=SS0
      DXP=(XP2-XP1)/FLOAT(NL-1)
      DYP=(YP2-YP1)/FLOAT(NL-1)
      EPSX=ABS(DXP*0.1)
      EPSY=ABS(DYP*0.1)
C
C    Look at each element in turn
C
      DO 60 N=1,NE
C
C     Calculate the geometrical coefficients
C
   12 XMIN=999.
      YMIN=999.
      XMAX=-999.
      YMAX=-999.
      DO 20 K=1,3
      K2=MOD(K,3)+1
      K3=MOD(K+1,3)+1
      LK=NOR(LEM(K,N))
      LK2=NOR(LEM(K2,N))
      LK3=NOR(LEM(K3,N))
      X1=EX(LK)
      X2=EX(LK2)
      X3=EX(LK3)
      Y1=EY(LK)
      Y2=EY(LK2)
      Y3=EY(LK3)
      XMIN=MIN(XMIN,X1)
      XMAX=MAX(XMAX,X1)
      YMIN=MIN(YMIN,Y1)
      YMAX=MAX(YMAX,Y1)
      DPX1=X1
      DPX2=X2
      DPX3=X3
      DPY1=Y1
      DPY2=Y2
      DPY3=Y3
      A0(K)=DPX2*DPY3 - DPX3*DPY2
      BY(K)=Y2    - Y3
      CX(K)=X3    - X2
   20 CONTINUE
C
C    TRI is twice the area of the triangle element
C
      TRI=DPX2*DPY3-DPX3*DPY2+DPX3*DPY1-DPX1*DPY3+DPX1*DPY2-DPX2*DPY1
C
C     Now look for the points on the line that are inside the tri.
C
C
      DO 80 J=1,NL
      XP=XP1+DXP*FLOAT(J-1)
      YP=YP1+DYP*FLOAT(J-1)
      IF((YP.GT.YMAX+EPSY).OR.(YP.LT.YMIN-EPSY).OR.
     1   (XP.GT.XMAX+EPSX).OR.(XP.LT.XMIN-EPSX))GO TO 80
C
C     Calculate the natural coordinates
C
      DO 45 K=1,3
      CNL=(A0(K) + XP*BY(K) + YP*CX(K))/TRI
      IF((CNL.GT.1.0+EPS).OR.(CNL.LT.-EPS))GO TO 80
   45 COOL(K)=CNL
C
C     If we reach here the point is within the triangle, so
C         Interpolate !
C
      SI=0.0
C     DO 35 K=1,6
      DO 35 K=1,3
      LK=NOR(LEM(K,N))
      SK=S(LK)
   35 SI=SI + SK*COOL(K)
      RLINE(J)=SI
   80 CONTINUE
   60 CONTINUE
      IF (IVERBOSE.NE.0) THEN
      WRITE(6,*)' data values from NTRPLN2, between '
      WRITE(6,*)'    (',XP1,',',YP1,') and (',XP2,',',YP2,')'
      END IF
      RETURN
      END

      SUBROUTINE NTRPLND(DENS,RLINE,NL,BGD,XP1,YP1,XP2,YP2,
     :                   TBXOFF,TBYOFF,VOFFSET,
     :                   EX,EY,LEM,NOR,ILINE,NE,NUP,NP3,
     :                   IVERBOSE)
C
C
C    Routine to interpolate the nodal values from the finite
C     element mesh onto a line traversing the solution region
C     NTRPLN uses quadratic interpolation (e.g. velocity)
C     NTRPLN2 uses linear interpolation (e.g. pressure)
C
      DIMENSION EX(NUP),EY(NUP),LEM(6,NE),NOR(NUP)
      DIMENSION DENS(7,NE),RLINE(NL),ILINE(NP3),BNODEV(6)
C     DIMENSION BY(3),CX(3),COOL(3)
      DOUBLE PRECISION BY(3),CX(3),COOL(3)
      DOUBLE PRECISION A0(3),TRI,DPX1,DPX2,DPX3,DPY1,DPY2,DPY3
      DATA EPS/1.0E-3/
      DATA EPS1/1.5E-3/
      SAVE EPS 
      SAVE EPS1
C
C   Zero the arrays
C   RLINE(NL) contains the intrpolated values on the line
C   ILINE(NL) contains a switch indicating a value in RLINE
C   Line runs between (XP1,YP1) and (XP2,YP2)
C
      DO 10 I=1,NL
      ILINE(I)=0
   10 RLINE(I)=BGD
      DXP=(XP2-XP1)/FLOAT(NL-1)
      DYP=(YP2-YP1)/FLOAT(NL-1)
C
C    Look at each element in turn
C
      DO 60 N=1,NE
C
C     Calculate the geometrical coefficients
C
   12 XMIN=999.
      YMIN=999.
      XMAX=-999.
      YMAX=-999.
      DO 20 K=1,3
      K2=MOD(K,3)+1
      K3=MOD(K+1,3)+1
      LK=NOR(LEM(K,N))
      LK2=NOR(LEM(K2,N))
      LK3=NOR(LEM(K3,N))
      X1=EX(LK)
      X2=EX(LK2)
      X3=EX(LK3)
      Y1=EY(LK)
      Y2=EY(LK2)
      Y3=EY(LK3)
      XMIN=MIN(XMIN,X1)
      XMAX=MAX(XMAX,X1)
      YMIN=MIN(YMIN,Y1)
      YMAX=MAX(YMAX,Y1)
      DPX1=X1
      DPX2=X2
      DPX3=X3
      DPY1=Y1
      DPY2=Y2
      DPY3=Y3
      A0(K)=DPX2*DPY3 - DPX3*DPY2
      BY(K)=Y2    - Y3
      CX(K)=X3    - X2
   20 CONTINUE
C
C    TRI is twice the area of the triangle element
C
      TRI=DPX2*DPY3-DPX3*DPY2+DPX3*DPY1-DPX1*DPY3+DPX1*DPY2-DPX2*DPY1
C
C     Now look for the points on the line that are inside the tri.
C
      XP = XP1-DXP
      YP = YP1-DYP
      XOFF=0.0
      YOFF=0.0
      VOFF=0.0
      DO 80 J=1,NL
      XP=XP+DXP-XOFF
      YP=YP+DYP-YOFF
      IF((YP.GT.YMAX+EPSY).OR.(YP.LT.YMIN-EPSY).OR.
     1   (XP.GT.XMAX+EPSX).OR.(XP.LT.XMIN-EPSX))GO TO 80
C
C     Calculate the natural coordinates
C
      DO 45 K=1,3
      CNL=(A0(K) + XP*BY(K) + YP*CX(K))/TRI
      IF((CNL.GT.1.0+EPS).OR.(CNL.LT.-EPS))GO TO 80
   45 COOL(K)=CNL
C
C     If we reach here the point is within the triangle, so
C         Interpolate !
C
              DO 25 K=1,3
                KM=MOD(K+1,3)+4
                BNODEV(K)=1.83095*DENS(K,N)-1.5*DENS(7,N)+
     1                                       0.669052*DENS(KM,N)
                BNODEV(KM)=0.0581402*DENS(K,N)-0.375*DENS(7,N)+
     1                                       1.31686*DENS(KM,N)
   25         CONTINUE
              SI=0.0
              DO 35 K=1,6
                SK=BNODEV(K)+VOFF
                IF(K.LE.3)THEN
                  K2=MOD(K,3)+1
                  K3=MOD(K+1,3)+1
                  CNK=COOL(K)*(COOL(K) - COOL(K2) - COOL(K3))
                ELSE
                  K2=K-3
                  K3=MOD(K+1,3)+1
                  CNK=4.0*COOL(K2)*COOL(K3)
                END IF
   35         SI=SI + SK*CNK
C     DO 35 K=1,6
C     LK=LEM(K,N)
C     IF(LK.LT.0)LK=-LK
C     SK=S(LK)
C     IF(K.GE.4)GO TO 30
C     K2=MOD(K,3)+1
C     K3=MOD(K+1,3)+1
C     CNK=COOL(K)*(COOL(K) - COOL(K2) - COOL(K3))
C     GO TO 35
C  30 K2=K-3
C     K3=MOD(K+1,3)+1
C     CNK=4.0*COOL(K2)*COOL(K3)
C  35 SI=SI + SK*CNK
      RLINE(J)=SI
      ILINE(J)=N
   80 CONTINUE
   60 CONTINUE
C
      IF (IVERBOSE.NE.0) THEN
      WRITE(6,*)' data values from NTRPLND, between '
      WRITE(6,*)'    (',XP1,',',YP1,') and (',XP2,',',YP2,')'
      END IF
C
      ICNT = 0
      DO 100 I=1,NL
        IF (ILINE(I).NE.0) ICNT = ICNT+1
 100  CONTINUE
      IF (ICNT.NE.NL) print *, ICNT
      RETURN
      END

      SUBROUTINE EDGES(EDGEPTS,YLWR,YUPR,EX,EY,PTS,ILINE,IBNGH,IBC,
     1                                      NOR,NUP,NE,NBP,NL)
      DIMENSION EDGEPTS(4,NL)
      DIMENSION EX(NUP),EY(NUP)
      DIMENSION NOR(NUP),IBC(NBP),IBNGH(NBP*2)
      DIMENSION ILINE(NL)
      DIMENSION PTS(NBP)
      PARAMETER (IX1INDX=1,IY1INDX=2,IX2INDX=3,IY2INDX=4)

      EPS = 1.0E-5
      DY = (YUPR-YLWR)/FLOAT(NL-1)
      YVAL = YLWR

      DO 10 J = 1,NL
        ILINE(J) = 0
        NUMPTS = 0
        DO 20 I = 1,NBP
C    FIX - check for IBCTYPE of 11 or 13 (internal fault nodes)
C         IF (IBCTYP(J).EQ.11.OR.IBCTYP(J).EQ.13) GOTO 20
          NODE = NOR(IBC(I))
          NODENEXT = NOR(IBNGH(I+NBP))
          Y = EY(NODE)
          YNEXT = EY(NODENEXT)
C    the line passes through the node or
C    between the node  and its neighbour
          IF ((Y.GE.YVAL.AND.YNEXT.LE.YVAL).OR.
     1                  (Y.LE.YVAL.AND.YNEXT.GE.YVAL)) THEN
            X = EX(NODE)
            XNEXT = EX(NODENEXT)
            IF (YNEXT-Y.NE.0.0) X = X+(XNEXT-X)*(YVAL-Y)/(YNEXT-Y)
            NUMPTS = NUMPTS+1
            PTS(NUMPTS) = X
          END IF
  20    CONTINUE
C
C    find pts with max and min x vals
C
        IF (NUMPTS.GT.0) THEN
          EDGEPTS(IX1INDX,J) = PTS(1)
          EDGEPTS(IX2INDX,J) = PTS(1)
          DO 30 K=2,NUMPTS
            IF (PTS(K).LT.EDGEPTS(IX1INDX,J)) EDGEPTS(IX1INDX,J)=PTS(K)
            IF (PTS(K).GT.EDGEPTS(IX2INDX,J)) EDGEPTS(IX2INDX,J)=PTS(K)
            PTS(K) = 0.0
            EDGEPTS(IY1INDX,J) = YVAL
            EDGEPTS(IY2INDX,J) = YVAL
            ILINE(J) = 1
  30      CONTINUE
        ENDIF
        YVAL = YVAL+DY
  10  CONTINUE
      RETURN
      END

      SUBROUTINE INTEGRAT(RLINE,ILINE,NL,HLEN,AREA,LABEL,LEN)
      CHARACTER*10 LABEL,LLBL
      DIMENSION RLINE(NL),ILINE(NL)
C
      HINC=HLEN/(NL-1)
      DO 50 K=1,NL
        IF(ILINE(K).NE.0)THEN
          KSTART=K
          GO TO 60
        END IF
   50 CONTINUE
   60 CONTINUE
      KEND=KSTART
      DO 70 K=KSTART,NL
        IF(ILINE(K).NE.0)THEN
          KEND=K
        ELSE
          GO TO 80
        END IF
   70 CONTINUE
   80 CONTINUE

      NUMSEG = KEND-KSTART
      AREA=0.0
      AREA2=0.0
      IF (NUMSEG.GE.3) THEN
        IF (MOD(NUMSEG,2).NE.0) THEN
          AREA1 = 3.0/8.0*HINC*(RLINE(KEND-3) + 3.0*RLINE(KEND-2)
     1              + 3.0*RLINE(KEND-1) + RLINE(KEND))
          KEND = KEND-3
        ELSE  
          AREA1 = 0.0
        ENDIF
        IF (NUMSEG.NE.3) THEN
          DO 90 K=KSTART,KEND-2,2
            AREA2 = AREA2 + 1.0/3.0*HINC*(RLINE(K) +
     1                   4.0*RLINE(K+1) + RLINE(K+2))
  90      CONTINUE
        ENDIF
        AREA = AREA1 + AREA2
      ELSE
C       WRITE(LUW,*)' only ',NUMSEG,' data values in profile'
        WRITE(6,*)' only ',NUMSEG,' data values in profile'
        RETURN
      ENDIF
      LLBL(1:10) = '          '
      IF (LEN.GT.0) THEN
        IF (LEN.GT.10) LEN = 10
        LLBL(1:LEN) = LABEL(1:LEN)
C       WRITE(LUW,10001)LLBL,AREA
        WRITE(6,10001)LLBL,AREA
      ENDIF
10001 FORMAT(A10,' Integral under profile ',F8.4)
      RETURN
      END

      SUBROUTINE INTEGRATTRPZ(RLINE,ILINE,NL,HLEN,AREA,LABEL,LEN)
      CHARACTER*10 LABEL,LLBL
      DIMENSION RLINE(NL),ILINE(NL)
C
      HINC=HLEN/(NL-1)
      DO 50 K=1,NL
        IF(ILINE(K).NE.0)THEN
          KSTART=K
          GO TO 60
        END IF
   50 CONTINUE
   60 CONTINUE
      KEND=KSTART
      DO 70 K=KSTART,NL
        IF(ILINE(K).NE.0)THEN
          KEND=K
        ELSE
          GO TO 80
        END IF
   70 CONTINUE
   80 CONTINUE

      AREA=0.0
      IF ((KEND-KSTART).GT.0) THEN
        AREA = AREA + RLINE(KSTART)*HINC/2.0
        DO 90 K=KSTART+1,KEND-1
          AREA = AREA + RLINE(K)*HINC
  90    CONTINUE
        AREA = AREA + RLINE(KEND)*HINC/2.0
      ELSE
C       WRITE(LUW,*)' less than 2 values in profile'
        WRITE(6,*)' less than 2 values in profile'
        RETURN
      ENDIF
      LLBL(1:10) = '          '
      IF (LEN.GT.0) THEN
        IF (LEN.GT.10) LEN = 10
        LLBL(1:LEN) = LABEL(1:LEN)
C       WRITE(LUW,10001)LLBL,AREA
        WRITE(6,10001)LLBL,AREA
      ENDIF
10001 FORMAT(A10,' Integral under profile ',F8.4)
      RETURN
      
      END
