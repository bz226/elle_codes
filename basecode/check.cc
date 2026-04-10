 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nodes.h"
#include "errnum.h"
#include "file.h"
#include "interface.h"
#include "runopts.h"
#include "check.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
int RandomiseSequence( int order[], int size );

int ElleCheckDoubleJ(int node)
{
    int deleted=0, err=0;
    int j,next,nn,nbnodes[3];

    if (ElleNodeIsActive(node) && ElleNodeIsDouble(node)) {
        ElleNeighbourNodes(node,nbnodes);
        for (j=0;j<3;j++) {
            next = nbnodes[j];
            if (next!=NO_NB && ElleGapTooLarge(node,next)) 
                ElleInsertDoubleJ(node,next,&nn,0.5);
        }
        for (j=0;j<3 && !deleted;j++) {
            next = nbnodes[j];
            if (next!=NO_NB && ElleNodesCoincident(node,next)) {
                            /*&& (Elle2NodesSameMineral(node,next)==1))  {*/
                if (ElleDeleteDoubleJNoCheck(node)==0)
                    deleted = 1;
            }
            else if (next!=NO_NB && ElleGapTooSmall(node,next)) {
                            /*&& (Elle2NodesSameMineral(node,next)==1))
 * {*/
                if (ElleDeleteDoubleJ(node)==0)
                    deleted = 1;
            }
        }
    }
    return(err);
}

int ElleCheckTripleJ(int node)
{
    int err=0;
    int j,next,nn,numTnb,idTnb[3],nbnodes[3];
    double minsep, separation[3];

    if (!ElleNodeIsActive(node) || !ElleNodeIsTriple(node)) return(err);
    for (j=0;j<3;j++) idTnb[j] = NO_NB;
    ElleNeighbourNodes(node,nbnodes);
    for (j=0,numTnb=0;j<3;j++) {
        next = nbnodes[j];
        if (ElleGapTooLarge(node,next)) {
            ElleInsertDoubleJ(node,next,&nn,0.5);
        }
        else if (ElleNodeIsDouble(next) && ElleGapTooSmall(node,next)) {
            ElleDeleteDoubleJ(next);
            if (!ElleNodeIsTriple(node)) return(err);
        }
        else if (ElleNodeIsTriple(next) && ElleSwitchGap(node,next)) {
            idTnb[numTnb] = next;
            separation[numTnb] = ElleNodeSeparation(node,next);
            numTnb++;
        }
    }
        
    switch(numTnb) {
    case 3:
    case 2: next = idTnb[0];
            minsep = separation[0];
            for (j=1;j<numTnb;j++) {
                if (separation[j]< minsep) {
                    next = idTnb[j];
                    minsep = separation[j];
                }
            }
            /*if(Elle2NodesSameMineral(node,next) == 1) mwj 12/12/99*/
            if(!EllePhaseBoundary(node,next)) /*le 14/02/03*/
                ElleSwitchTripleNodes(node,next);
            else if (ElleSmallSwitchGap(node,next))
                    ElleSwitchTripleNodesForced(node,next);
// NB stopping triple switching
#if XY
#endif
            break;
    case 1:
            /* if(Elle2NodesSameMineral(node,idTnb[0])==1) {mwj 12/12/99*/
            if(!EllePhaseBoundary(node,idTnb[0])) {/*le 14/02/03*/
                    ElleSwitchTripleNodes(node,idTnb[0]);
                }
                else if (ElleSmallSwitchGap(node,idTnb[0]))
                {
                    ElleSwitchTripleNodesForced(node,idTnb[0]);
                }
// NB stopping triple switching
#if XY
#endif
            break;
    case 0: break;
    default: break;
    }
    return(err);
}

/*
 * checks to see if a node has the same mineral on all 2 or 3 sides
 * and if so which one
 */
void ElleNodeSameMineral(int node, int *same, int *samemin)
{
    int i,j, themin=NO_NB,mintype[3];
    ERegion rgn[3];


    *same=1;
    *samemin=NO_NB;
    if (ElleFlynnAttributeActive(MINERAL)) {
        ElleRegions(node,rgn);

        for (i=0;i<3;i++) {
            mintype[i]=NO_NB;
            if (rgn[i]!=NO_NB) {
                ElleGetFlynnIntAttribute(rgn[i], &mintype[i], MINERAL);
                themin=mintype[i];
            }
        }

        for(i=0;i<3;i++)
            if(mintype[i]!=NO_NB && mintype[i]!=themin) *same=0;
/*
        for(i=0;i<3;i++)
            if(mintype[i]!=NO_NB)
                for(j=0;j<3;j++)
                if(j != i && mintype[j] != NO_NB)
                    if(mintype[i]!=mintype[j])
                        *same=0;
*/

        *samemin=themin;
    }
    else *samemin=QUARTZ;
}

/*
 * checks to see if 2 nodes have the same minerals on all 2, 3 or 4 sides
 */
int Elle2NodesSameMineral(int node1, int node2)
{
    int same = 0;
    int min1, min2, same1, same2;

    if (!ElleFlynnAttributeActive(MINERAL))
        same = 1;
    else {
        ElleNodeSameMineral(node1, &same1, &min1);
        ElleNodeSameMineral(node2, &same2, &min2);
   
        if(same1==1 && same2==1 && min1==min2) same = 1;
    }
    return(same);
}

/*
 * checks to see if boundary has the same mineral on both sides
 */
unsigned char EllePhaseBoundary(int node1, int node2)
{
    int rgn[2], mintype[2];

    mintype[0] = mintype[1] = NO_NB;
    if (ElleFlynnAttributeActive(MINERAL)) {
        ElleNeighbourRegion(node1,node2,&rgn[0]);
        ElleNeighbourRegion(node2,node1,&rgn[1]);
        ElleGetFlynnIntAttribute(rgn[0], &mintype[0], MINERAL);
        ElleGetFlynnIntAttribute(rgn[1], &mintype[1], MINERAL);
    }
    return(mintype[0]!=mintype[1]);
}

/*
 * checks to see if tjs have same regions (switch will cause wrap)
 */
bool ElleSameRegions(int node1, int node2)
{
	bool match=true;
	int i;
	int rgns1[3], rgns2[3];

	ElleRegions(node1,rgns1);
	ElleRegions(node2,rgns2);
	i=0;
	while (match && i<3) {
		if ((rgns1[i]!=rgns2[0]) && (rgns1[i]!=rgns2[1]) &&
			(rgns1[i]!=rgns2[2]) ) match=false;
        i++;
	}
    return(match);
}

void ElleCheckUnit(float *xpts,float *ypts,int num,int *xflags,int *yflags,
                   Coords *bl,Coords *tr)
{
    int i;
    float minx,maxx,miny,maxy,xp,gamma;
    float eps;
    float xoffset;
    CellData unitcell;

    eps = 1.5e-6;
    eps = 5e-5;
    ElleCellBBox(&unitcell);
    maxy = (float)unitcell.cellBBox[TOPLEFT].y;
    miny = (float)unitcell.cellBBox[BASELEFT].y;
    maxx = (float)unitcell.cellBBox[BASERIGHT].x;
    minx = (float)unitcell.cellBBox[BASELEFT].x;
    xoffset = (float)(unitcell.cellBBox[TOPLEFT].x-
                      unitcell.cellBBox[BASELEFT].x);
	if ((unitcell.cum_xoffset<-eps || unitcell.cum_xoffset>eps) &&
			xoffset>unitcell.xlength-eps && xoffset<unitcell.xlength+eps)
		xoffset=unitcell.xlength;
    gamma = xoffset * (float)unitcell.ylength;
    xflags[0] = xflags[1] = 0;
    yflags[0] = yflags[1] = 0;
    bl->x = tr->x = xpts[0];
    bl->y = tr->y = ypts[0];
    for (i=0;i<num;i++) {
        if (ypts[i]<miny) yflags[0]=1;
        if (ypts[i]>maxy) yflags[1]=1;
        xp = (ypts[i] - miny)*gamma + unitcell.cellBBox[0].x;
        if (xpts[i]<(xp-eps)) xflags[0]=1;
        xp += unitcell.xlength;
        if (xpts[i]>(xp+eps)) xflags[1]=1;
        if (ypts[i]<bl->y) bl->y = ypts[i];
        if (ypts[i]>tr->y) tr->y = ypts[i];
        if (xpts[i]<bl->x) bl->x = xpts[i];
        if (xpts[i]>tr->x) tr->x = xpts[i];
    }
    /* this hasn't been checked for -ve xoffset */
    while (bl->y<(miny-unitcell.ylength*yflags[0])) yflags[0]++;
    while (tr->y>(maxy+unitcell.ylength*yflags[1])) yflags[1]++;
    while (bl->x<(minx-(unitcell.xlength-xoffset)*xflags[0]))
        xflags[0]++;
    while (tr->x>(maxx+(unitcell.xlength-xoffset)*xflags[1]))
        xflags[1]++;
}

void ElleCheckUnit(double *xpts,double *ypts,int num,int *xflags,int *yflags,
                   Coords *bl,Coords *tr)
{
    int i;
    double minx,maxx,miny,maxy,xp,gamma;
    double eps;
    double xoffset;
    CellData unitcell;

    eps = 1.5e-6;
    ElleCellBBox(&unitcell);
    maxy = unitcell.cellBBox[TOPLEFT].y;
    miny = unitcell.cellBBox[BASELEFT].y;
    maxx = unitcell.cellBBox[BASERIGHT].x;
    minx = unitcell.cellBBox[BASELEFT].x;
    xoffset = unitcell.cellBBox[TOPLEFT].x-unitcell.cellBBox[BASELEFT].x;
    gamma = xoffset * unitcell.ylength;
    xflags[0] = xflags[1] = 0;
    yflags[0] = yflags[1] = 0;
    bl->x = tr->x = xpts[0];
    bl->y = tr->y = ypts[0];
    for (i=0;i<num;i++) {
        if (ypts[i]<miny) yflags[0]=1;
        if (ypts[i]>maxy) yflags[1]=1;
        xp = (ypts[i] - miny)*gamma + unitcell.cellBBox[0].x;
        if (yflags[0]==1) xp -= unitcell.xoffset;
        else if (yflags[1]==1) xp +=unitcell.xoffset;
        if (xpts[i]<(xp-eps)) xflags[0]=1;
        xp += unitcell.xlength;
        if (xpts[i]>(xp+eps)) xflags[1]=1;
        if (ypts[i]<bl->y) bl->y = ypts[i];
        if (ypts[i]>tr->y) tr->y = ypts[i];
        if (xpts[i]<bl->x) bl->x = xpts[i];
        if (xpts[i]>tr->x) tr->x = xpts[i];
    }
    /* this hasn't been checked for -ve xoffset */
    while (bl->y<(miny-unitcell.ylength*yflags[0])) yflags[0]++;
    while (tr->y>(maxy+unitcell.ylength*yflags[1])) yflags[1]++;
    /*while (bl->x<(minx-(unitcell.xlength-unitcell.xoffset)*xflags[0]))*/
    while (bl->x<(minx-unitcell.xlength*xflags[0]))
        xflags[0]++;
    /*while (tr->x>(maxx+(unitcell.xlength-unitcell.xoffset)*xflags[1]))*/
    while (tr->x>(maxx+unitcell.xlength*xflags[1]))
        xflags[1]++;
}
