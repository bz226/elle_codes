/******************************************************
 * Spring Code Mike 2.0
 *
 * Functions for lattice class in lattice.cc
 *
 * Basic Spring Code for fracturing 
 *
 *
 *
 *
 * Daniel Koehn and Jochen Arnold feb. 2002 to feb. 2003
 * Oslo, Mainz 
 *
 * Daniel Koehn dec. 2003
 *
 * We thank Anders Malthe-S�renssen for his enormous help
 * and introduction to these codes
 *
 * Daniel Koehn and Till Sachau 2004/2005
 ******************************************************/

// ------------------------------------
// system headers
// ------------------------------------

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ------------------------------------
// elle headers
// ------------------------------------

#include "graingrowth.h"
#include "unodes.h"		// include unode funct. plus undodesP.h
				// for c++
#include "attrib.h"		// enth�lt struct coord (hier: xy)
#include "attribute.h"
#include "attribarray.h"
#include "nodes.h"
#include "nodesP.h"
#include "interface.h"
#include "file.h"
#include "error.h"
#include "general.h"
#include "runopts.h"
#include "polygon.h"
#include "tripoly.h"
#include "display.h"
#include "check.h"
#include "convert.h"
#include "update.h"

// have to define these in the new Elle version

using std::cout;
using std::endl;
using std::vector;

// CONSTRUCTOR
/*******************************************************
 * Constructor for lattice class 
 * calls the MakeLattice function and defines some variables 
 * at the moment. The Constructor then builds a lattice once
 * the lattice class is called (i.e. by the user in the "elle" 
 * function). 
 *
 * Daniel spring 2002
 *
 * add security stop for pictdumps
 *
 * Daniel march 2003
 ********************************************************/

// ---------------------------------------------------------------
// Constructor of Lattice class
// ---------------------------------------------------------------

GrainGrowth::GrainGrowth ()

    // -----------------------------------------------------------------------------
    // default values for some variables. Variables are declared in
    // lattice header
    // -----------------------------------------------------------------------------


{
	TotalTime=0;
	
}


int GrainGrowth::DoGrowth(int step)
{
    int i,j, k;
    int interval=0,st_interval=0,err=0,max;
    int *seq;
    char fname[32];
    FILE *fp;
	
	i = step;

        max = ElleMaxNodes();
        if (i%2) {
        for (j=0;j<max;j++) {
            if (ElleNodeIsActive(j)) {
                if (ElleNodeIsDouble(j)) {
                    MoveDoubleJ(j);
                    ElleCheckDoubleJ(j);
					
                }
                else if (ElleNodeIsTriple(j)) {
                    MoveTripleJ(j);
                    ElleCheckTripleJ(j);
                }
            }
        }
        }
        else {
        for (j=max-1;j>=0;j--) {
            if (ElleNodeIsActive(j)) {
                if (ElleNodeIsDouble(j)) {
                    MoveDoubleJ(j);
                    ElleCheckDoubleJ(j);
                }
                else if (ElleNodeIsTriple(j)) {
                    MoveTripleJ(j);
                    ElleCheckTripleJ(j);
                }
            }
        }
        }
        ElleUpdate();
  
}

int GrainGrowth::MoveDoubleJ(int node1)
{
    int i, nghbr[2], nbnodes[3], err;
    double maxV,gb_energy,ray,deltaT,vlen;
    double switchDist, speedUp;
    Coords xy1, movedist;

    switchDist = ElleSwitchdistance();
    speedUp = ElleSpeedup() * switchDist * switchDist * 0.02;
    maxV = ElleSwitchdistance()/5.0;
    /*
     * allows speedUp to be 1 in input file
     */
    gb_energy = speedUp;
    deltaT = 0.0;
    /*
     * find the node numbers of the neighbours
     */
    if (err=ElleNeighbourNodes(node1,nbnodes))
        OnError("MoveDoubleJ",err);
    i=0;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nghbr[0] = nbnodes[i]; i++;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nghbr[1] = nbnodes[i];

    GetRay(node1,nghbr[0],nghbr[1],&ray,&movedist);
    if (ray > 0.0) {
    /*if (ray > ElleSwitchdistance()/100.0) {*/
        vlen = gb_energy/ray;
        if (vlen > maxV) {
            vlen = maxV;
            deltaT = 1.0;
        }
        if (vlen>0.0) {
            movedist.x *= vlen;
            movedist.y *= vlen;
        }
        else {
            movedist.x = 0.0;
            movedist.y = 0.0;
        }
        TotalTime += deltaT;
        ElleUpdatePosition(node1,&movedist);
    }
    else {
        vlen = 0.0;
    }
}

int GrainGrowth::MoveTripleJ(int node1)
{
    int i, nghbr[3], finished=0, err=0;
    double maxV,gb_energy[3],ray[3],deltaT,vlen[3],vlenTriple;
    double switchDist, speedUp;
    Coords xy1, movedist[3], movedistTriple;

    switchDist = ElleSwitchdistance();
    /*
     * allows speedUp to be 1 in input file
     */
    speedUp = ElleSpeedup() * switchDist * switchDist * 0.02;
    maxV = switchDist/5.0;
    for (i=0;i<3;i++) gb_energy[i] = speedUp;
    deltaT = 0.0;
    /*
     * find the node numbers of the neighbours
     */
    if (err=ElleNeighbourNodes(node1,nghbr))
        OnError("MoveTripleJ",err);

    GetRay(node1,nghbr[0],nghbr[1],&ray[0],&movedist[0]);
    GetRay(node1,nghbr[1],nghbr[2],&ray[1],&movedist[1]);
    GetRay(node1,nghbr[2],nghbr[0],&ray[2],&movedist[2]);
    for(i=0;i<3;i++) {
        if (ray[i] > 0.0) {
        /*if (ray[i] > switchDist/100.0) {*/
            vlen[i] = gb_energy[i]/ray[i];
            /*if (vlen[i] > maxV) vlen[i] = maxV;*/
        }
        else {
            vlen[i] = 0.0;
            finished = 1;
        }
    }
    if (!finished) {
        for(i=0;i<3;i++) {
            if (vlen[i] < maxV) {
                movedist[i].x *= vlen[i];
                movedist[i].y *= vlen[i];
            }
            else {
                movedist[i].x *= maxV;
                movedist[i].y *= maxV;
            }
        }
        movedistTriple.x = movedist[0].x+movedist[1].x+movedist[2].x;
        movedistTriple.y = movedist[0].y+movedist[1].y+movedist[2].y;
        vlenTriple = sqrt(movedistTriple.x*movedistTriple.x + 
                          movedistTriple.y*movedistTriple.y);
        if (vlenTriple > maxV) {
            vlenTriple = maxV/vlenTriple;
            movedistTriple.x *= vlenTriple;
            movedistTriple.y *= vlenTriple;
            deltaT = 1.0;
        }
        if (vlenTriple <= 0.0) movedistTriple.x = movedistTriple.y = 0.0;
          
        TotalTime += deltaT;
    }
    else {
        ElleNodePosition(node1,&xy1);
        ElleNodePlotXY(nghbr[0],&movedist[0],&xy1);
        ElleNodePlotXY(nghbr[1],&movedist[1],&xy1);
        ElleNodePlotXY(nghbr[2],&movedist[2],&xy1);
        for(i=0;i<3;i++) {
            movedist[i].x = movedist[i].x - xy1.x;
            movedist[i].y = movedist[i].y - xy1.y;
        }
        movedistTriple.x = (movedist[0].x+movedist[1].x+movedist[2].x)/2.0;
        movedistTriple.y = (movedist[0].y+movedist[1].y+movedist[2].y)/2.0;
#if XY
        vlenTriple = sqrt(movedistTriple.x*movedistTriple.x + 
                          movedistTriple.y*movedistTriple.y);
        if (vlenTriple > maxV) {
            vlenTriple = maxV/vlenTriple;
            movedistTriple.x *= vlenTriple;
            movedistTriple.y *= vlenTriple;
            deltaT = 1.0;
        }
#endif
    }
    ElleUpdatePosition(node1,&movedistTriple);
}

void GrainGrowth::GetRay(int node1,int node2,int node3,double *ray,Coords *movedist)
{
    double dx2,dy2,dx3,dy3,tmpx,tmpy;
    double k, x0,y0;
    double switchDist;
    double eps = 1e-8;
    double r;
    Coords xy1, xy2, xy3;

    switchDist = ElleSwitchdistance();
    ElleNodePosition(node1,&xy1);
    ElleNodePlotXY(node2,&xy2,&xy1);
    ElleNodePlotXY(node3,&xy3,&xy1);
    dx2 = xy2.x - xy1.x;
    dy2 = xy2.y - xy1.y;
    dx3 = xy3.x - xy1.x;
    dy3 = xy3.y - xy1.y;
    if (dx2==0.0) {
        tmpx = dx2;
        tmpy = dy2;
        dx2 = dx3;
        dy2 = dy3;
        dx3 = tmpx;
        dy3 = tmpy;
    }
    *ray = 0.0;
    movedist->x = movedist->y = 0.0;

    if (dx2>eps || dx2<-eps) {
        k = 2.0 * dx3 * dy2/dx2 - 2.0*dy3;
        if (k!=0.0) {
            y0 = ((dx3/dx2)*(dx2*dx2 + dy2*dy2)-dx3*dx3-dy3*dy3)/k;
            x0 = (dx2*dx2+dy2*dy2 - 2.0*y0*dy2)/(2.0*dx2);
            r = sqrt((double)(x0*x0+y0*y0));
            if (r!=0.0) {
                *ray = r;
                movedist->x = x0/ *ray;
                movedist->y = y0/ *ray;
                if (*ray < switchDist/3.0) *ray = switchDist/3.0;
            }
        }
    }
}                

int GrainGrowth::IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff)
{
    Coords xynew;
    /*
     * find the average of the 3 points
     * move 0.1 of the distance towards the average point
     */
    xynew.x = (xy->x + xy1->x + xy2->x) / 3;
    xynew.y = (xy->y + xy1->y + xy2->y) / 3;
    diff->x = xynew.x - xy->x;
    diff->y = xynew.y - xy->y;
    diff->x *= 0.1;
    diff->y *= 0.1;
}

/* #define MINANG 0.087  approx 5deg */
#define MINANG 0.14  /* approx 8deg */
void GrainGrowth::CheckAngles()
{
    int moved=1,i,j,k,max;
    int nbnodes[3];
    double currang;
    double ang;
    Coords xy[3], movedist;

    max = ElleMaxNodes();
    while (moved)  {
    for (k=0,moved=0;k<max;k++) {
        if (ElleNodeIsActive(k)) {
            ElleNodePosition(k,&xy[0]);
            ElleNeighbourNodes(k,nbnodes);
            if (ElleNodeIsDouble(k)) {
                j=0; i=1;
                while (j<3) {
                    if (nbnodes[j]!=NO_NB)
                        ElleNodePlotXY(nbnodes[j],&xy[i++],&xy[0]);
                    j++;
                }
                if (angle( xy[0].x,xy[0].y,xy[1].x,xy[1].y,xy[2].x,xy[2].y,
                                                             &currang))
                        OnError("angle error",0);
                ang = fabs(currang);
                if (ang<MINANG /*|| ang<(-M_PI+MINANG)*/) {
                    do {
                        IncreaseAngle(&xy[0],&xy[1],&xy[2],&movedist);
                        xy[0].x += movedist.x;
                        xy[0].y += movedist.y;
                        if (angle( xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                                            xy[2].x,xy[2].y,&currang))
                            OnError("angle error",0);
                        ang = fabs(currang);
                    } while (ang<MINANG /*|| ang>(M_PI-MINANG)*/);
                    ElleSetPosition(k,&xy[0]);
                    ElleCheckDoubleJ(k);
                    moved = 1;
                }
            }
            else if (ElleNodeIsTriple(k)) {
                for (j=0;j<3 ;j++) {
                    i = (j+1)%3;
                    ElleNodePlotXY(nbnodes[j],&xy[1],&xy[0]); 
                    ElleNodePlotXY(nbnodes[i],&xy[2],&xy[0]); 
                    if (angle( xy[0].x,xy[0].y,xy[1].x,xy[1].y,xy[2].x,xy[2].y,
                                                             &currang))
                        OnError("angle error",0);
                    ang = fabs(currang);
                    if (ang<MINANG /*|| ang>(M_PI-MINANG)*/) {
                        do {
                            IncreaseAngle(&xy[0],&xy[1],&xy[2],&movedist);
                            xy[0].x += movedist.x;
                            xy[0].y += movedist.y;
                            if (angle( xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                                            xy[2].x,xy[2].y,&currang))
                                OnError("angle error",0);
                            ang = fabs(currang);
                        } while (ang<MINANG/* || ang>(M_PI-MINANG)*/);
                        ElleSetPosition(k,&xy[0]);
                        ElleCheckTripleJ(k);
                        moved = 1;
                        j=3;
                    }
                }
            }
        }
    }
    }
}
