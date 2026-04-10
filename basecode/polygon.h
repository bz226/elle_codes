 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <vector>

#ifndef _Elle_polygon_h
#define _Elle_polygon_h
int EllePtInRegion(Coords *bndpts,int num,Coords *pt);
void RegionExpanding(int n,int nb,int rgn_exp[3][2],Coords *newxy);
double polyArea(std::vector<Coords> &pts);
#ifdef __cplusplus
extern "C" {
#endif
int EllePtInRegion(ERegion poly,Coords *pt);
int EllePtInRect(Coords *bndpts,int num,Coords *pt);
double ElleRegionArea(ERegion poly);
int ElleRegionCentroid(ERegion poly,Coords *centre);
double polyArea(double x[], double y[], int n);
double pointSeparation(Coords *xy1, Coords *xy2);
#ifdef __cplusplus
}
#endif
#endif
