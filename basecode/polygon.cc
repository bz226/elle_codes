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
#include <malloc.h>
#include "attrib.h"
#include "bflynns.h"
#include "nodes.h"
#include "error.h"
#include "polygon.h"
#include "general.h"
#include "interface.h"
#include "crossings.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

int EllePtInRegion(ERegion poly,Coords *pt)
{
    int j, k, *id=0, in=0, num_nodes, start=0;
    double *coords, *ptr, pnt[2];
    double dist, min;
    Coords xy,prev;

    ElleFlynnNodes(poly,&id,&num_nodes);
    if (num_nodes>2) {
        if ((coords = (double *)malloc(num_nodes*sizeof(double)*2))
                                == 0) OnError("EllePtInRegion",MALLOC_ERR);
    /*
     * find the flynn node that is closest to the test point
     * otherwise flynn may be located on other side of unit cell
     */
        ElleNodePosition(id[0],&prev);
        min = (prev.x-pt->x)*(prev.x-pt->x) + (prev.y-pt->y)*(prev.y-pt->y);
        for (j=1,ptr=coords;j<num_nodes;j++) {
            ElleNodePosition(id[j],&prev);
            dist = (prev.x-pt->x)*(prev.x-pt->x) +
                          (prev.y-pt->y)*(prev.y-pt->y);
            if (dist<min) {
                min = dist;
                start = j;
            }
        }
        //Closest flynn node may be on other side of cell
        ElleNodePlotXY(id[start],&prev,pt);
        for (k=0,j=start,ptr=coords;k<num_nodes;k++,j++) {
            if (j==num_nodes) j=0;
            ElleNodePlotXY(id[j],&xy,&prev);
            *ptr = (xy.x); ptr++;
            *ptr = (xy.y); ptr += 1;
            prev = xy;
        }
    /*
     * add eps - have had trouble with pnm2elle files not
	 * finding a flynn for unodes on x=0. Not sure why
	 * but this fixed it. All flynns were checked so it must 
	 * be in CrossingsTest??
     */
        pnt[0] = pt->x + 1e-40;
        pnt[1] = pt->y + 1e-40;
        in = CrossingsTest(coords,num_nodes,pnt);
        free(coords);
    }
    if (id) free(id);
    return(in);
}

int EllePtInRegion(Coords *bndpts,int num,Coords *pt)
{
    int k, in=0;
    double *coords=0, *ptr, pnt[2];
    Coords xy,prev;

    if ((coords = (double *)malloc(num*sizeof(double)*2))
                                == 0) OnError("EllePtInRegion",MALLOC_ERR);
    for (k=0,ptr=coords;k<num;k++) {
        *ptr = bndpts[k].x; ptr++;
        *ptr = bndpts[k].y; ptr += 1;
    }
    pnt[0] = pt->x + 1e-40;
    pnt[1] = pt->y + 1e-40;
    in = CrossingsTest(coords,num,pnt);
    free(coords);
    return(in);
}

int EllePtInRect(Coords *bndpts,int num,Coords *pt)
{
    int in=0;
    if (num==4 &&
        pt->x >= bndpts[BASELEFT].x && pt->x <= bndpts[TOPRIGHT].x &&
        pt->y >= bndpts[BASELEFT].y && pt->y <= bndpts[TOPRIGHT].y)
        in = 1;
    return(in);
    
}

double pointSeparation(Coords *xy1, Coords *xy2)
{
    double tmp = (xy1->x - xy2->x) * (xy1->x-xy2->x) +
                  (xy1->y - xy2->y) * (xy1->y - xy2->y);
    return(sqrt(tmp));
}

double ElleRegionArea(ERegion poly)
{
    int j, *id=0, num_nodes;
    double area, *coordsx=0, *coordsy=0, *ptrx, *ptry;
    Coords xy,prev;

    ElleFlynnNodes(poly,&id,&num_nodes);
    if ((coordsx = (double *)malloc(num_nodes*sizeof(double)))
                                == 0) OnError("ElleRegionArea",MALLOC_ERR);
    if ((coordsy = (double *)malloc(num_nodes*sizeof(double)))
                                == 0) OnError("ElleRegionArea",MALLOC_ERR);
    ElleNodePosition(id[0],&prev);
    for (j=0,ptrx=coordsx,ptry=coordsy;j<num_nodes;j++) {
        ElleNodePlotXY(id[j],&xy,&prev);
        *ptrx = xy.x; ptrx++;
        *ptry = xy.y; ptry++;
        prev = xy;
    }
    area = polyArea(coordsx,coordsy,num_nodes);
    free(coordsx);
    free(coordsy);
    if (id) free(id);
    return(area);
}

int ElleRegionCentroid(ERegion poly,Coords *centre)
{
    int j, *id=0, num_nodes, err=0;
    double cx, cy;
    double area;
    double *coordsx=0, *coordsy=0, *ptrx, *ptry;
    Coords xy,prev;

    ElleFlynnNodes(poly,&id,&num_nodes);
    if ((coordsx = (double *)malloc(num_nodes*sizeof(double)))
                                == 0) OnError("ElleRegionArea",MALLOC_ERR);
    if ((coordsy = (double *)malloc(num_nodes*sizeof(double)))
                                == 0) OnError("ElleRegionArea",MALLOC_ERR);
    ElleNodePosition(id[0],&prev);
    for (j=0,ptrx=coordsx,ptry=coordsy;j<num_nodes;j++) {
        ElleNodePlotXY(id[j],&xy,&prev);
        *ptrx = (xy.x); ptrx++;
        *ptry = (xy.y); ptry++;
        prev = xy;
    }
    cx = coordsx[0]; cy = coordsy[0];
    err = polyCentroid(coordsx,coordsy,num_nodes,&cx,&cy,&area);
    centre->x = cx;
    centre->y = cy;
    ElleNodeUnitXY(centre);
    if (coordsx) free(coordsx);
    if (coordsy) free(coordsy);
    if (id) free(id);
    return(err);
}

/*
 * split area calculation out of polyCentroid (general.c)
 */
double polyArea(double x[], double y[], int n)
{
    register int i, j;
    double ai, area = 0;
    if (n < 3) return 0;
    for (i = n-1, j = 0; j < n; i = j, j++) {
        ai = x[i] * y[j] - x[j] * y[i]; /*+ve counterclockwise*/
        /*ai = x[j] * y[i] - x[i] * y[j];*/
        area += ai;
    }
    return (area/2);
}

/*
 * split area calculation out of polyCentroid (general.c)
 */
double polyArea(std::vector<Coords> &pts)
{
    register int i, j;
    double ai, area = 0;
    int n = pts.size();
    if (n < 3) return 0;
    for (i = n-1, j = 0; j < n; i = j, j++) {
        ai = pts[i].x * pts[j].y - pts[j].x * pts[i].y; /*+ve counterclockwise*/
        /*ai = pts[j].x * pts[i].y - pts[i].x * pts[j].y;*/
        area += ai;
    }
    return (area/2);
}

/*!
 \brief	Which of the polygons expands if the node is moved
 \par	Description:
 		If node n is moved to the position newxy, this function 
		determines which of the neighbouring regions
		will increase in size as each boundary moves.
		rgn_exp will contain the neighbouring regions (rgn_exp[n][0])
		and expand flag 0/1 (rgn_exp[n][1], n=1..3.
 */
void RegionExpanding(int n,int nb,int rgn_exp[2][2],Coords *newxy)
{
	int i;
	double area1;
	double xpts[3],ypts[3],tmp;
	Coords xy,xy1;

	for(i=0;i<2;i++) {
		rgn_exp[i][0]=NO_NB;
		rgn_exp[i][1]=0;
	}
	ElleNeighbourRegion(n,nb,&rgn_exp[0][0]);
	ElleNeighbourRegion(nb,n,&rgn_exp[1][0]);
	ElleNodePosition(n,&xy);
	xpts[0]=xy.x; ypts[0]=xy.y;
	xpts[1]=newxy->x; ypts[1]=newxy->y;
	ElleNodePlotXY(nb,&xy1,&xy);
	xpts[2]=xy1.x; ypts[2]=xy1.y;
	area1 = polyArea(xpts,ypts,3);
	rgn_exp[0][1]=((area1>0.0) ? 1 : 0);
	if (rgn_exp[0][1]==0) rgn_exp[1][1]=1;
}
