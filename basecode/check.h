 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_check_h
#define _E_check_h

#ifndef _E_attrib_h
#include "attrib.h"
#endif
void ElleCheckUnit(double *xpts,double *ypts,int num,int *xflags,
                   int *yflags,Coords *bl,Coords *tr);
bool ElleSameRegions(int node1, int node2);

#ifdef __cplusplus
extern "C" {
#endif
int ElleCheckDoubleJ(int node);
int ElleCheckTripleJ(int node);
void ElleCheckUnit(float *xpts,float *ypts,int num,int *xflags,
                   int *yflags,Coords *bl,Coords *tr);
#ifdef __cplusplus
}
#endif
#endif
