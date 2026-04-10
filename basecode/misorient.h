 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef E_misorient_h
#define E_misorient_h

void CalculateBoundaryAttribute(int type);
void CalcMisorient(int rgn1, int rgn2, double *orient);
void CalcMisorient(int rgn1, int rgn2, float *orient);
void CalcMisorientUnodes(int unode1, int unode2, double *orient);
double Misorient(double*, double*);
double Misorientation(double ang1[3],double ang2[3], double symm[24][3][3],
                      const int symm_ops);
void FindBndAttributeRange(int type, double *min, double *max);

#endif
