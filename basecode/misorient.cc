/*
 *----------------------------------------------------
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 *----------------------------------------------------
 */

static const char rcsid[] =
       "$Id$";
/*
 *----------------------------------------------------
 */
#include <stdio.h>
#include <math.h>
#include "nodes.h"
#include "convert.h"
#include "interface.h"
#include "mat.h"
#include "file.h"
#include "unodes.h"
#include "misorient.h"
#include "log.h"
#include "error.h"


void FindBndAttributeRange(int type, double *min, double *max)
{
    int nmax, i, j;
    int nbnodes[3];
    int set = 0, err=0;
    double val;

    CalculateBoundaryAttribute(type);
    nmax = ElleMaxNodes();
    for (j=0;j<nmax;j++) {
        if (ElleNodeIsActive(j)) {
            ElleNeighbourNodes(j,nbnodes);
            for (i=0;i<3;i++) {
                if (nbnodes[i]!=NO_NB && nbnodes[i]<j ) {
                    if ((val = ElleGetBoundaryAttribute(j,nbnodes[i]))
                                   > 0) {
                        if (!set) {
                            *min = *max = val;
                            set = 1;
                        }
                        else {
                            if (val < *min) *min = val;
                            if (val > *max) *max = val;
                        }
                    }
                }
            }
        }
    }
}

void CalculateBoundaryAttribute(int type)
{
    int max, i, j;
    int rgn1,rgn2;
    int nbnodes[3];
    double orient;

    max = ElleMaxNodes();
    for (j=0;j<max;j++) {
        if (ElleNodeIsActive(j)) {
            ElleNeighbourNodes(j,nbnodes);
            for (i=0;i<3;i++) {
                if (nbnodes[i]!=NO_NB && nbnodes[i]<j && 
                      !EllePhaseBoundary(nbnodes[i],j)) {
                    ElleNeighbourRegion(j,nbnodes[i],&rgn1);
                    ElleNeighbourRegion(nbnodes[i],j,&rgn2);
                    CalcMisorient(rgn1,rgn2,&orient);
                    ElleSetBoundaryAttribute(j,i,orient);
                }
            }
        }
    }
}

void CalcMisorient(int rgn1, int rgn2, double *orient)
{   
    double curra[3];
    double currb[3];

                //retrieve 3 Euler angles
    ElleGetFlynnEuler3(rgn1, &curra[0], &curra[1], &curra[2]);
    ElleGetFlynnEuler3(rgn2, &currb[0], &currb[1], &currb[2]);
    
    *orient=Misorient(curra,currb);
}


void CalcMisorientUnodes(int unode1, int unode2, double *orient)
{
    double curra[3];
    double currb[3];

                //retrieve 3 Euler angles
    ElleGetUnodeAttribute(unode1,&curra[0],&curra[1],&curra[2],EULER_3);
    ElleGetUnodeAttribute(unode2,&currb[0],&currb[1],&currb[2],EULER_3);

    *orient=Misorient(curra,currb);

}

/*
 * the angle from this calculation is the same as misorientation from
 * orilib but not the same as disori or the misorientation used in
 * CSLFactor.
 * this should only do the calculation inverse x matrix once.
 * To get the smallest misorientation (disorientation), the crystal 
 * symmetry operators (24 for cubic) should be applied in turn to the
 * product matrix and the smallest angle returned 
 */
double Misorient(double curra[3], double currb[3])
{   
    int i;
    double tmpA, tmpB, angle;
    double val;
    double eps=1e-6;
    double loca[3], locb[3];

    double rmap1[3][3];
    double rmap2[3][3];
    double rmap3[3][3];
    double rmapA[3][3];
    double rmapB[3][3];
    double a11, a22,a33;
    

    for (i=0;i<3;i++) {
        loca[i] = curra[i] * DTOR;
        locb[i] = currb[i] * DTOR;
    }

    eulerZXZ(rmap1,loca[0],loca[1],loca[2]);// gives rotation matrix
    eulerZXZ(rmap2,locb[0],locb[1],locb[2]);// gives rotation matrix
    
    /*calculation of tmpA where the inverse of rmap1 is taken for calculation*/
    matinverse(rmap1,rmap3);
    matmult(rmap2,rmap3,rmapA);
    
    a11=rmapA[0][0];
    a22=rmapA[1][1];
    a33=rmapA[2][2];
    
    val = (a11+a22+a33-1)/2;
/*
    if (val > 1.0+eps || val < -1.0-eps ) {
        sprintf(logbuf,"CalcMisorient - adjusting val for acos from %lf for bnd %d, %d\n",val,rgn1,rgn2);
        Log( 1,logbuf );
    }
*/
    if (val>1.0) val = 1.0;
    else if (val<-1.0) val = -1.0;
    tmpA=acos(val);

    /*calculation of tmpB where the inverse of rmap1 is taken for calculation*/
    matinverse(rmap2,rmap3);
    matmult(rmap1,rmap3,rmapB);

    
    a11=rmapB[0][0];
    a22=rmapB[1][1];
    a33=rmapB[2][2];
    
    val = (a11+a22+a33-1)/2;
/*
    if (val > 1.0+eps || val < -1.0-eps ) {
        sprintf(logbuf,"CalcMisorient - adjusting val for acos from %lf for bnd %d, %d\n",val,rgn1,rgn2);
        Log( 1,logbuf );
    }
*/
    if (val>1.0) val = 1.0;
    else if (val<-1.0) val = -1.0;
    tmpB=acos(val);

    if (tmpA<tmpB) angle=tmpA;
    else angle=tmpB;
    angle *= RTOD;
    return(angle);
}

/*
 * Function Misorient
 * To get the smallest misorientation (disorientation), the crystal 
 * symmetry operators (symm_ops) are applied in turn to the
 * product matrix and the smallest angle (degrees) returned 
 */
/*!
	\brief		Use crystal symmetry to find smallest misorientation

	\param		ang1 - first Euler angle (degrees)
	                ang2 - second Euler angle (degrees)
                        symm - symmetry matrix 24x3x3
                        symm_ops - number of symmetry entries (use symm_opsx3x3)

	\return		\a minang - smallest misorientation for symmetry (degrees)

	\par		Description:
                        To calculate the smallest misorientation (disorientation),
                        the crystal symmetry operators (symm_ops) are applied in turn
                        to the product matrix and the smallest angle returned

	\exception
                        Checks that symm_ops is in range 0 -> 24 

	\par		Example:

\verbatim
                        int symm_ops, mineral, symmetry;
                        double symm[24][3][3];
                        double ang1[3], ang2[3];
                        double minang = 0;

                        ElleGetFlynnMineral(flynn1,&mineral);
                        symmetry = (int)GetMineralAttribute(mineral,SYMMETRY);
                        symm_ops = symmetry_set(symm, (const int)symmetry);
                        
                        ElleGetFlynnEuler3(flynn1,&ang1[0],&ang1[1],&ang1[2]);
                        ElleGetFlynnEuler3(flynn2,&ang2[0],&ang2[1],&ang2[2]);

                        minang = Misorientation(ang1, ang2, symm, (const int)symm_ops);


\endverbatim
*/

double Misorientation(double ang1[3],double ang2[3], double symm[24][3][3],
                      const int symm_ops)

{
    double val[4];
    double eps=1e-6;

    double rmap1[3][3];
    double rmap2[3][3];
    double rmapA[3][3],rmapAA[3][3];
    double a11, a22,a33;
    int n,i,j;
    double misang,minang;
    double rmap1A[3][3],rmap2A[3][3];
    double aux_symm[3][3];
    double curra1,currb1,currc1;
    double curra2,currb2,currc2;

    curra1 = ang1[0]*DTOR; currb1 = ang1[1]*DTOR; currc1 = ang1[2]*DTOR;
    curra2 = ang2[0]*DTOR; currb2 = ang2[1]*DTOR; currc2 = ang2[2]*DTOR;

    orientmatZXZ(rmap1, (double)curra1,(double)currb1,(double)currc1);// gives rotation matrix
    orientmatZXZ(rmap2, (double)curra2,(double)currb2,(double)currc2);

        // Ini symmetry operators...
    minang=1000;

    if (symm_ops>24 || symm_ops<0)
        OnError("Misorientation - invalid symm_ops",0);
    for(n=0;n<symm_ops;n++) {
        // auxiliar symmetry matrix [3][3]
        for (i=0;i<3;i++) {
            for (j=0;j<3;j++) {
                aux_symm[i][j] = symm[n][i][j];
            }
        }

                // 1st
        //calculation of rmapA where the inverse of rmap2 is taken for calculation
        matinverse(rmap2,rmap2A);
        matmult(rmap1,rmap2A,rmap1A);

        // symmetry operators
        matmult(aux_symm,rmap1A,rmapA);

        // 2nd
        //calculation of rmapAA where the inverse of rmap1 is taken for calculation
        matinverse(rmap1,rmap1A);
        matmult(rmap2,rmap1A,rmap2A);

        // symmetry operators
        matmult(aux_symm,rmap2A,rmapAA);

        // Take trace.. and misorientation angle
        a11=rmapA[0][0];
        a22=rmapA[1][1];
        a33=rmapA[2][2];

        val[0] = (a11+a22+a33-1)/2.0;

        a11=rmapAA[0][0];
        a22=rmapAA[1][1];
        a33=rmapAA[2][2];

        val[1] = (a11+a22+a33-1)/2.0;

        for(i=0;i<=1;i++){
            if (val[i]>1.0) val[i] = 1.0;
            else if (val[i]<-1.0) val[i] = -1.0;

            misang=(acos(val[i]));
            misang *= RTOD;

            if (misang < minang) {
                minang=misang;
            }
        }
   }
   return(minang);
}


#if XY
void CalcMisorient(int rgn1, int rgn2, float *orient)
{   
    float curra1, currb1, currc1;
    float curra2, currb2, currc2;
    double tmpA, tmpB, angle;
    double val;

    double rmap1[3][3];
    double rmap2[3][3];
    double rmap3[3][3];
    double rmapA[3][3];
    double rmapB[3][3];
    double a11, a22,a33;
    

    ElleGetFlynnEuler3(rgn1, &curra1, &currb1, &currc1);
    ElleGetFlynnEuler3(rgn2, &curra2, &currb2, &currc2);
                //retrieve 3 Euler angles
    

    curra1 *= DTOR; currb1 *= DTOR; currc1 *= DTOR;
    curra2 *= DTOR; currb2 *= DTOR; currc2 *= DTOR;
    orientmat(rmap1,(double)curra1,(double)currb1,(double)currc1);// gives rotation matrix
    orientmat(rmap2,(double)curra2,(double)currb2,(double)currc2);// gives rotation matrix
    
    /*calculation of tmpA where the inverse of rmap1 is taken for calculation*/
    matinverse(rmap1,rmap3);
    matmult(rmap2,rmap3,rmapA);
    
    a11=rmapA[0][0];
    a22=rmapA[1][1];
    a33=rmapA[2][2];
    
    val = (a11+a22+a33-1)/2;
    if (val > 1.0 || val < -1.0 )
        sprintf(logbuf,"CalcMisorient - adjusting val for acos from %lf for bnd %d, %d\n",val,rgn1,rgn2);
    Log( 1,logbuf );
    if (val>1.0) val = 1.0;
    else if (val<-1.0) val = -1.0;
    tmpA=acos(val);

    /*calculation of tmpB where the inverse of rmap1 is taken for calculation*/
    matinverse(rmap2,rmap3);
    matmult(rmap1,rmap3,rmapB);
    
    a11=rmapB[0][0];
    a22=rmapB[1][1];
    a33=rmapB[2][2];
    
    val = (a11+a22+a33-1)/2;
    if (val > 1.0 || val < -1.0 )
        sprintf(logbuf,"CalcMisorient - adjusting val for acos from %lf for bnd %d, %d\n",val,rgn1,rgn2);
    Log( 1,logbuf );
    if (val>1.0) val = 1.0;
    else if (val<-1.0) val = -1.0;
    tmpB=acos(val);

    if (tmpB<tmpA) angle=tmpB;
    if (tmpA<tmpB) angle=tmpA;
    else angle=tmpA;
    angle *= RTOD;
    *orient=(float)angle;
   
}
#endif

