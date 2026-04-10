#include <vector>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "attrib.h"
#include "nodes.h"
#include "file.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "init.h"
#include "general.h"
#include "stats.h"
#include "update.h"

using std::vector;

int GrainGrowth();
int InitGrowth();
int MoveDoubleJ(int node1);
int MoveTripleJ(int node1);
extern void GetRay(int node1,int node2,int node3,double *ray,Coords *movedist);
void CheckAngles();

double TotalTime;

int InitGrowth()
{
    int err=0;
    int max;
    char *infile;

    ElleReinit();
    ElleSetRunFunction(GrainGrowth);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        ElleAddDoubles();
    }
    /*
    else {
        if (err=ElleReadData("../init/10x0.1_input"))
            OnError("10x0.1_input",err);
        ElleSwitchTriples(8);
        ElleSwitchTriples(6);
    }
    */

    TotalTime=0;
}

int GrainGrowth()
{
    int i, j, k, n;
    int interval=0,st_interval=0,err=0,max;
    vector<int> seq;
    char fname[32];
    FILE *fp;

    if (ElleCount()==0) ElleAddDoubles();
    if (ElleDisplay()) EllePlotRegions(ElleCount());
	ElleCheckFiles();

    for (i=0;i<EllemaxStages();i++) {
        max = ElleMaxNodes();
        seq.clear();
        for (j=0;j<max;j++) if (ElleNodeIsActive(j)) seq.push_back(j);
        random_shuffle(seq.begin(),seq.end());
        max = seq.size();
        for (n=0;n<max;n++) {
            j=seq[n];
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
        ElleUpdate();
    }
    /*CheckAngles();*/
}

int MoveDoubleJ(int node1)
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

int MoveTripleJ(int node1)
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
}

/* #define MINANG 0.087  approx 5deg */
#define MINANG 0.14  /* approx 8deg */
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
