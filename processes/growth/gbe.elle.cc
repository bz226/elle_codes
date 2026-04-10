#include <stdio.h>
#include <math.h>
#include <string.h>
#include "convert.h"
#include "attrib.h"
#include "nodes.h"
#include "file.h"
#include "display.h"
#include "check.h"
#include "errnum.h"
#include "error.h"
#include "runopts.h"
#include "init.h"
#include "general.h"
#include "interface.h"
#include "lut.h"
#include "stats.h"
#include "update.h"
#include "mat.h"
#include <stdlib.h>
#include "polygon.h"
int GBE_GrainGrowth();
int InitGBE_Growth();

double  GetAngleEnergy(double angle);
int GBE_MoveNode(int n, Coords *movedir);
int GetNodeEnergy(int n,Coords *xy,double *energy);

double    Incroffset=0.002; /* this is reassigned in GBE_GrainGrowth */

int InitGBE_Growth()
{
    int err=0;
    int max;
    char *infile;

    ElleReinit();
    ElleSetRunFunction(GBE_GrainGrowth);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes
         */
        if (!ElleFlynnAttributeActive(CAXIS) &&
            !ElleFlynnAttributeActive(EULER_3))
            ElleAttributeNotInFile(infile,EULER_3);
        ElleAddDoubles();
    }
    else /* warn that there is no file open */;

}

int GBE_GrainGrowth()
{
    int i, j, k;
    int interval=0,st_interval=0,err=0,max;
    char fname[FILENAME_MAX];
    FILE *fp;
    Coords incr;

    if (ElleCount()==0) ElleAddDoubles();
    if (ElleDisplay()) EllePlotRegions(ElleCount());
    interval = EllesaveInterval();
        max = ElleMaxFlynns();
            for (k=0;k<max;k++) {
                if (ElleFlynnIsActive(k) && ElleRegionArea(k)<0.0)
                   printf ("backwards %d \n",k);
            }

    Incroffset = ElleSwitchdistance() * 0.01;

    st_interval = ES_statsInterval();
    if (st_interval>0) ES_WriteStatistics(0);

    for (i=0;i<EllemaxStages();i++) {
        max = ElleMaxNodes();
        if (i%2) {
            for (k=0;k<max;k++) {
                if (ElleNodeIsActive(k)) {
                    if (GBE_MoveNode(k,&incr))
                        ElleUpdatePosition(k,&incr);
                    if (ElleNodeIsDouble(k))
                        ElleCheckDoubleJ(k);
                    else if (ElleNodeIsTriple(k))
                        ElleCheckTripleJ(k);
                }
            }
        }
        else {
            for (k=max-1;k>=0;k--) {
                if (ElleNodeIsActive(k)) {
                    if (GBE_MoveNode(k,&incr))
                        ElleUpdatePosition(k,&incr);
                    if (ElleNodeIsDouble(k))
                        ElleCheckDoubleJ(k);
                    else if (ElleNodeIsTriple(k))
                        ElleCheckTripleJ(k);
                }
            }
        }
        ElleUpdate();
    }
}

/*
 * changed May '01 to set a move flag if it finds a lower energy
 * position
 */
int GBE_MoveNode(int n, Coords *movedir)
{
    register int    i,j;
    int             moveflag=0;
    double          startenergy,tryenergy,newenergy,dx,dy,dE,mobility;
    double          dx2, dy2;
    double          randangle, angle;
    Coords          xy, newxy, prev;

    ElleNodePosition(n,&xy);
    ElleNodePrevPosition(n,&prev);
    newxy = xy;
    GetNodeEnergy(n,&newxy,&startenergy);
    newenergy = startenergy;
    randangle=((double)(rand()%1000))/2000.0*M_PI;
    dx=cos(randangle)*Incroffset;
    dy=sin(randangle)*Incroffset;
    newxy.x = xy.x + dx;
    newxy.y = xy.y + dy;
    GetNodeEnergy(n,&newxy,&newenergy);
    movedir->x=dx;
    movedir->y=dy;

    for (i=1;i<4;i++) {
        angle=i*M_PI/2;
        rotate_coords(dx, dy, 0.0, 0.0, &dx2, &dy2, angle);
        newxy.x = xy.x + dx2;
        newxy.y = xy.y + dy2;
        GetNodeEnergy(n,&newxy,&tryenergy);
        if (tryenergy<newenergy) {
            newenergy=tryenergy;
            movedir->x=dx2;
            movedir->y=dy2;
        }
    }
    dE=startenergy-newenergy;
    if (dE>0.0)
    {
        /*mobility=dE*SpeedUp/Incroffset;
        if (mobility>1000.0)    mobility=1000.0;*/
        /*mobility=dE*ElleSpeedup(); speedup of 500 in input file */
        /*
         * allow speedup of 1 in input file
         */
        mobility=dE*ElleSpeedup()/
                    ElleSwitchdistance()/ElleSwitchdistance()/10;
        if (mobility>1.0)    mobility=1.0;
        movedir->x *= mobility;
        movedir->y *= mobility;
        moveflag=1;
    }
    /*
     * added May '01
     */
    else
    {
        movedir->y = 0.0;
        movedir->y = 0.0;
    }
    return(moveflag);
}

double GetAngleEnergy(double angle)
{
    int angdeg;
    double    C,energy,sinangle;

    angdeg = (int)(angle*RTOD + 0.5);
    energy = ElleEnergyLUTValue(angdeg);
    return(energy);
}

int GetNodeEnergy(int n,Coords *xy,double *energy)
{
    int        i,nb[3];
    Coords  xynb;
    double    cosalpha,alpha,E;
    double  len;
    double val[3];
    int        G0,G1;
    ERegion  rgn[3],rgn2;
    Coords_3D xyz[3];

    ElleRegions(n,rgn);
    ElleNeighbourNodes(n,nb);
    E = 0.0;
    for (i=0;i<3;i++) {
        if (nb[i]!=NO_NB) {
            ElleRelPosition(xy,nb[i],&xynb,&len);
            ElleNeighbourRegion(nb[i],n,&rgn2);

            if (ElleFlynnAttributeActive(EULER_3))
            {                                                                        
                ElleGetFlynnEuler3(rgn[i],&val[0],&val[1],&val[2]);
                eulerZXZ2cax(val[0],val[1],val[2],&xyz[0].x);

                ElleGetFlynnEuler3(rgn2,&val[0],&val[1],&val[2]);
                eulerZXZ2cax(val[0],val[1],val[2],&xyz[1].x);
            }
            else if (ElleFlynnAttributeActive(CAXIS))
            {
                ElleGetFlynnCAxis(rgn[i],&xyz[0]);
                ElleGetFlynnCAxis(rgn2,&xyz[1]);
            }

            if (len==0.0) len = 1.0;
            cosalpha = (xynb.x * xyz[0].x + xynb.y * xyz[0].y)/len;
            /*
             * make sure cosalpha is in range -1,1 for acos
             */
            if (cosalpha > 1.0) cosalpha = 1.0;
            if (cosalpha < -1.0) cosalpha = -1.0;
            if (cosalpha >= 0.0) alpha = acos(cosalpha);
            else                 alpha = acos(-cosalpha);
            E += len*GetAngleEnergy(alpha);
            cosalpha = (xynb.x * xyz[1].x + xynb.y * xyz[1].y)/len;
            /*
             * make sure cosalpha is in range -1,1 for acos
             */
            if (cosalpha > 1.0) cosalpha = 1.0;
            if (cosalpha < -1.0) cosalpha = -1.0;
            if (cosalpha >= 0.0) alpha = acos(cosalpha);
            else                 alpha = acos(-cosalpha);

            E += len*GetAngleEnergy(alpha);
        }
    }
    *energy=E;
    return(0);
}
