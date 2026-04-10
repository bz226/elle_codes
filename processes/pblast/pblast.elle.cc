#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "nodes.h"
#include "bflynns.h"
#include "display.h"
#include "errnum.h"
#include "error.h"
#include "runopts.h"
#include "polygon.h"
#include "file.h"
#include "init.h"
#include "check.h"
#include "crossings.h"
#include "general.h"
#include "convert.h"
#include "interface.h"
#include "update.h"

using std::cout;
using std::endl;

#define UNIT_FILL_COL  55

double TotalTime=0.0;

extern void GetRay(int node1,int node2,int node3,double *ray,Coords *movedist);
int MoveDoubleJOut(int node1, Coords *movedist, ERegion full_id);
int MoveTripleJOut(int node1, Coords *movedist, ERegion full_id);
int MovingNode(int node, ERegion *rgn);
void RotatePosition(int node,Coords *curr,int nb1, int nb2);
void CheckAngles();
int IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff);
int ExpandGrain(), InitExpand();

int InitExpand()
{
    char *infile;
    int err=0;
    int max;

    ElleReinit();
    ElleSetRunFunction(ExpandGrain);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError("",err);
        /*
         * check for any necessary attributes
         */
        if (!ElleFlynnAttributeActive(EXPAND)) 
            ElleAttributeNotInFile(infile,EXPAND);
        ElleAddDoubles();
    }
    else {
        ElleInitFlynnAttribute(COLOUR);
        ElleInitFlynnAttribute(EXPAND);
        ElleSetFile("/usr/local/elle/examples/init/10x0.1_input.elle");
        if (err=ElleReadData("/usr/local/elle/examples/init/10x0.1_input.elle"))
            OnError("",err);
        ElleSwitchTriples(8);
        ElleSwitchTriples(6);
        ElleSetFlynnIntAttribute(46,UNIT_FILL_COL,COLOUR);
        ElleSetFlynnIntAttribute(55,UNIT_FILL_COL,COLOUR);
        ElleSetFlynnIntAttribute(46,1,EXPAND);
        ElleSetFlynnIntAttribute(55,1,EXPAND);
    }
    if (ElleDisplay()) EllePlotRegions(ElleCount());
}

int ExpandGrain()
{
    int i, j, k;
    int interval=0,err=0,max;
    int *seq;
    Coords incr;
    ERegion rgn;

	ElleCheckFiles();
    if (ElleDisplay()) EllePlotRegions(ElleCount());
    for (i=0;i<EllemaxStages();i++) {
        max = ElleMaxNodes();
        if (i%2) {
            for (j=0;j<max;j++) {
                if (ElleNodeIsActive(j)) {
                    if (ElleNodeIsDouble(j)) {
                        if (MovingNode(j,&rgn)) {
                            MoveDoubleJOut(j,&incr,rgn);
                            ElleCrossingsCheck(j,&incr);
                        }
                        else
                            ElleCheckDoubleJ(j);
                    }
                    else if (ElleNodeIsTriple(j)) {
                        if (MovingNode(j,&rgn)) {
                            MoveTripleJOut(j,&incr,rgn);
                            ElleCrossingsCheck(j,&incr);
                        }
                        else ElleCheckTripleJ(j);
                    }
                    if (ElleNodeIsActive(j)) {
                       if (ElleNodeIsDouble(j))
                           ElleCheckDoubleJ(j);
                       else if (ElleNodeIsTriple(j))
                           ElleCheckTripleJ(j);
                    }
                }
            }
        }
        else {
            for (j=max-1;j>=0;j--) {
                if (ElleNodeIsActive(j)) {
                    if (ElleNodeIsDouble(j)) {
                        if (MovingNode(j,&rgn)) {
                            MoveDoubleJOut(j,&incr,rgn);
                            ElleCrossingsCheck(j,&incr);
                        }
                    }
                    else if (ElleNodeIsTriple(j)) {
                        if (MovingNode(j,&rgn)) {
                            MoveTripleJOut(j,&incr,rgn);
                            ElleCrossingsCheck(j,&incr);
                        }
                    }
                    if (ElleNodeIsActive(j)) {
                       if (ElleNodeIsDouble(j))
                           ElleCheckDoubleJ(j);
                       else if (ElleNodeIsTriple(j))
                           ElleCheckTripleJ(j);
                    }
                }
            }
        }
        max = ElleMaxNodes();
        for (j=0;j<max;j++) {
            if (ElleNodeIsActive(j)) {
                if (ElleNodeIsDouble(j))
                  ElleCheckDoubleJ(j);
            }
        }
		ElleUpdate();
    }
}

/* #define MINANG 0.087  approx 5deg */
#define MINANG 0.175  /* approx 10deg */
void CheckAngles()
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
                if (angle( xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                           xy[2].x,xy[2].y,&currang))
                        OnError("angle error",0);
                ang = fabs(currang);
                if (ang<MINANG) {
                    do {
                        IncreaseAngle(&xy[0],&xy[1],&xy[2],&movedist);
                        xy[0].x += movedist.x;
                        xy[0].y += movedist.y;
                        if (angle( xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                                   xy[2].x,xy[2].y,&currang))
                            OnError("angle error",0);
                        ang = fabs(currang);
                    } while (ang<MINANG);
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
                    if (angle(xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                              xy[2].x,xy[2].y,&currang))
                        OnError("angle error",0);
                    ang = fabs(currang);
                    if (ang<MINANG) {
							cout << ang << "node " << k << endl;
                        do {
                            IncreaseAngle(&xy[0],&xy[1],&xy[2],&movedist);
                            xy[0].x += movedist.x;
                            xy[0].y += movedist.y;
                            if (angle( xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                                       xy[2].x,xy[2].y,&currang))
                                OnError("angle error",0);
                            ang = fabs(currang);
                        } while (ang<MINANG);
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

int IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff)
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
    return(0);
}

/*
 * Allow node to move if it is on the boundary of
 * an expanding grain but not on the boundary
 * between two expanding grains
 */
int MovingNode(int node, ERegion *rgn)
{
    int i,nbnodes[3],move=0,cnt=0,val=0;
    ERegion tmp;

    ElleNeighbourNodes(node,nbnodes);
    for (i=0;i<3;i++) {
        if (nbnodes[i]!=NO_NB) {
            ElleNeighbourRegion(node,nbnodes[i],&tmp);
            ElleGetFlynnIntAttribute(tmp,&val,EXPAND);
            if (val) {
                if (!move) *rgn = tmp;
                move++;
            }
            cnt++;
        }
    }
    if (move==cnt) move=0;
    return(move);
}

int MoveDoubleJOut(int node1, Coords *movedist, ERegion full_id)
{
    int i, j, nghbr[2], nbnodes[3], err;
    double maxV,gb_energy,ray,deltaT,vlen;
    Coords xy, nodexy;
    double x, y, ang, step=PI;

    maxV = ElleSwitchdistance()/50.0;
    gb_energy = ElleSpeedup();
    deltaT = 0.0;
    /*
     * find the node numbers of the neighbours
     */
    if (err=ElleNeighbourNodes(node1,nbnodes))
        OnError("MoveDoubleJOut",err);
    i=0;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nghbr[0] = nbnodes[i]; i++;
    while (i<3 && nbnodes[i]==NO_NB) i++;
    nghbr[1] = nbnodes[i];

    GetRay(node1,nghbr[0],nghbr[1],&ray,movedist);
    if (ray > ElleSwitchdistance()/100.0) {
        vlen = maxV;
/*
        vlen = gb_energy/ray;
        if (vlen > maxV) {
            vlen = maxV;
            deltaT = 1.0;
        }
*/
        /*
         * Check whether the node is moving into the grain
         * ie this part of the polygon is convex
         * if so, move it in the opposite direction
         */
        movedist->x *= vlen;
        movedist->y *= vlen;
        ElleNodePosition(node1,&nodexy);
        xy = nodexy;
        xy.x += movedist->x;
        xy.y += movedist->y;

        if (EllePtInRegion(full_id,&xy)) {
            movedist->x *= -1;
            movedist->y *= -1;
            xy.x = nodexy.x + movedist->x;
            xy.y = nodexy.y + movedist->y;
        }
        if (EllePtInRegion(full_id,&xy)) {
            i = ElleFindBndNb(node1,full_id);
            if (i!=nghbr[1]) {
                i=nghbr[1]; nghbr[1]=nghbr[0];nghbr[0]=i;
            }
            RotatePosition(node1,&xy,nghbr[0],nghbr[1]);
            movedist->x = xy.x-nodexy.x;
            movedist->y = xy.y-nodexy.y;
        }
        TotalTime += deltaT;
    }
    else {
        vlen = 0.0;
    }
}

int MoveTripleJOut(int node1, Coords *movedistTriple, ERegion full_id)
{
    int i, nghbr[3], finished=0, err=0;
    double maxV,gb_energy[3],ray[3],deltaT,vlen[3],vlenTriple;
    double switchDist, speedUp;
    Coords xy, xy1, nodexy, movedist[3];

    switchDist = ElleSwitchdistance();
    speedUp = ElleSpeedup();
    maxV = switchDist/50.0;
maxV *= 1.5;
    for (i=0;i<3;i++) gb_energy[i] = speedUp;
    deltaT = 0.0;
    /*
     * find the node numbers of the neighbours
     */
    if (err=ElleNeighbourNodes(node1,nghbr))
        OnError("MoveTripleJOut",err);

    GetRay(node1,nghbr[0],nghbr[1],&ray[0],&movedist[0]);
    GetRay(node1,nghbr[1],nghbr[2],&ray[1],&movedist[1]);
    GetRay(node1,nghbr[2],nghbr[0],&ray[2],&movedist[2]);
    for(i=0;i<3;i++) {
        if (ray[i] > switchDist/100.0) {
/*
            vlen[i] = gb_energy[i]/ray[i];
*/
            vlen[i] = maxV;
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
        movedistTriple->x = movedist[0].x+movedist[1].x+movedist[2].x;
        movedistTriple->y = movedist[0].y+movedist[1].y+movedist[2].y;
        vlenTriple = sqrt(movedistTriple->x*movedistTriple->x +
                          movedistTriple->y*movedistTriple->y);
        if (vlenTriple > maxV) {
            vlenTriple = maxV/vlenTriple;
            movedistTriple->x *= vlenTriple;
            movedistTriple->y *= vlenTriple;
            deltaT = 1.0;
        }
        if (vlenTriple <= 0.0) movedistTriple->x = movedistTriple->y = 0.0;

        TotalTime += deltaT;
    }
    else {
        ElleNodePosition(node1,&xy1);
        ElleNodePlotXY(nghbr[0],&movedist[0],&xy1);
        ElleNodePlotXY(nghbr[1],&movedist[1],&xy1);
        ElleNodePlotXY(nghbr[2],&movedist[2],&xy1);
        for(i=0;i<3;i++) {
            movedist[i].x = xy1.x - movedist[i].x;
            movedist[i].y = xy1.y - movedist[i].y;
        }
        movedistTriple->x = (movedist[0].x+movedist[1].x+movedist[2].x)/2.0;
        movedistTriple->y = (movedist[0].y+movedist[1].y+movedist[2].y)/2.0;
    }
    ElleNodePosition(node1,&nodexy);
    xy = nodexy;
    xy.x += movedistTriple->x;
    xy.y += movedistTriple->y;

    if (EllePtInRegion(full_id,&xy)) {
        movedistTriple->x *= -1;
        movedistTriple->y *= -1;
        xy.x = nodexy.x + movedistTriple->x;
        xy.y = nodexy.y + movedistTriple->y;
    }
    if (EllePtInRegion(full_id,&xy)) {
        i = ElleFindBndNb(node1,full_id);
        if (i!=nghbr[1]) {
            i=nghbr[1]; nghbr[1]=nghbr[0];nghbr[0]=i;
        }
        i = ElleFindBndNb(nghbr[2],full_id);
        if (i==node1) {
            i=nghbr[0]; nghbr[0]=nghbr[2];nghbr[2]=i;
        }
        RotatePosition(node1,&xy,nghbr[0],nghbr[1]);
        movedistTriple->x = xy.x-nodexy.x;
        movedistTriple->y = xy.y-nodexy.y;
    }

    /*ElleUpdatePosition(node1,&movedistTriple);*/
}

/*
 * rotate to lie on the line which bisects the angle
 * nb1-node-nb2
 */
void RotatePosition(int node,Coords *curr,int nb1, int nb2)
{
    Coords xy,xy_nb1,xy_nb2;
    double ang1,ang2;
    double ang, xnew, ynew;

    ElleNodePosition(node,&xy);
    ElleNodePlotXY(nb1,&xy_nb1,&xy);
    ElleNodePlotXY(nb2,&xy_nb2,&xy);
    angle(xy.x,xy.y,
          xy_nb2.x,xy_nb2.y,
          curr->x,curr->y,
          &ang2);
    angle(xy.x,xy.y,
          xy_nb1.x,xy_nb1.y,
          curr->x,curr->y,
          &ang1);
    /*
     * angles returned are -PI -> PI
     */
    if (ang1 < 0) ang1 += PI*2;
    if (ang2 < 0) ang2 += PI*2;
    ang = (ang1+ang2)*0.5;
    rotate_coords(curr->x,curr->y,xy.x,xy.y,
              &xnew, &ynew, ang);
    curr->x = xnew;  curr->y = ynew;
}
#if XY
int MoveDoubleJ(int node1)
{
    int i, nghbr[2], nbnodes[3], err;
    float maxV,gb_energy,ray,deltaT,vlen;
    Coords xy1, movedist;

    maxV = ElleSwitchdistance()/50.0;
    gb_energy = ElleSpeedup();
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
    if (ray > ElleSwitchdistance()/100.0) {
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
#endif
