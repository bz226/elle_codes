 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_crossings_h
#define _E_crossings_h

#include <set>

#define MAX_I_CNT 10

typedef struct {
    int node1a, node2a, node1b, node2b;
    Coords xingpt;
    ERegion chk_rgn;
    int pivot;
} Intersection;

typedef struct {
    int pivot;
    int nb;
    int node;
    double prev_angle;
} Angle_data;

typedef std::set<int,std::less<int> > set_int;

int ElleRegionIsSimple(ERegion poly,Intersection *intersect);
int ElleReplaceIntersection( Intersection *intersect, int *rgn_new );
int ElleFlynnTopologyCheck();
int ElleNodeTopologyCheck(int node);
#ifdef __cplusplus
extern "C" {
#endif
int ElleCrossingsCheck(int node, Coords *pos_incr);
int ElleFindIntersection(int node1, int nb, ERegion rgn,Coords *strt_pos,
                         Coords *pos_incr, Intersection *intersect);
void ElleCheckAngles(int node, Coords *newxy, Angle_data *ang, int *count);
void ElleMoveLink(Intersection *intersect, Angle_data *ang,
                  Coords *newxy);
void InitIntersection(Intersection *intersect);
#ifdef __cplusplus
}
#endif
#endif
