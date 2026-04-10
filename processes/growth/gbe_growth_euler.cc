#include <stdio.h>
#include <math.h>
#include <string.h>
#include "convert.h"
#include "attrib.h"
#include "nodes.h"
#include "file.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "init.h"
#include "general.h"
#include "interface.h"
#include "lut.h"
#include "stats.h"

int GBE_GrainGrowth();
int InitGBE_Growth();

int Count;
double  GetAngleEnergy(double angle);
int GBE_MoveNode(int n, Coords *movedir);
int GetNodeEnergy(int n,Coords *xy,float *energy);
int GetCSLFactor(Coords_3D xyz[3], float *factor);

float	incroffset=0.002; /* this is reassigned in GBE_GrainGrowth */

int InitGBE_Growth()
{
    int err=0;
    int max;
    char *infile;
    extern int Count;

    ElleReinit();
    ElleSetRunFunction(GBE_GrainGrowth);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes
         */
        if (!ElleFlynnAttributeActive(CAXIS))
            OnError(infile,INVALIDF_ERR);
    }
    else /* warn that there is no file open */;

    Count = 0;
}

int GBE_GrainGrowth()
{
    int i, j, k;
    int interval=0,st_interval=0,err=0,max;
    char fname[FILENAME_MAX];
    FILE *fp;
    Coords incr;
    extern int Count;

    if (Count==0) ElleAddDoubles();
    if (ElleDisplay()) EllePlotRegions(Count);
    interval = EllesaveInterval();

    incroffset = ElleSwitchdistance() * 0.01;

    st_interval = ES_statsInterval();
    if (st_interval>0) ES_WriteStatistics(0);
 
    for (i=0;i<EllemaxStages();i++) {
        max = ElleMaxNodes();
        if (i%2) {
            for (k=0;k<max;k++) {
                if (ElleNodeIsActive(k)) {
                    GBE_MoveNode(k,&incr);
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
                    GBE_MoveNode(k,&incr);
		    ElleUpdatePosition(k,&incr);
                    if (ElleNodeIsDouble(k)) 
                        ElleCheckDoubleJ(k);
                    else if (ElleNodeIsTriple(k)) 
                        ElleCheckTripleJ(k);
        	}
            }
        }
        Count++;
        if (ElleDisplay()) {
            EllePlotRegions( Count );
            ElleShowStages( Count );
        }
        if (interval>0) {
            if (Count>0 && Count%interval==0) {
                ElleAutoWriteFile(Count);
            }
        }
        if (st_interval>0) {
            if (Count%st_interval==0) {
                ES_WriteStatistics(Count);
            }
        }
    }
}

int GBE_MoveNode(int n, Coords *movedir)
{
	register int	i,j;
	float			startenergy,tryenergy,newenergy,dx,dy,dE,mobility;
	double			randangle;
    Coords          xy, newxy;
    
    ElleNodePosition(n,&xy);
    newxy = xy;
	GetNodeEnergy(n,&newxy,&startenergy);
	randangle=((double)(rand()%1000))/2000.0*M_PI;
	dx=(float)cos(randangle)*incroffset;
	dy=(float)sin(randangle)*incroffset;
	movedir->x=dx;
	movedir->y=dy;	
	
    newxy.x = xy.x + dx;
    newxy.y = xy.y + dy;
	GetNodeEnergy(n,&newxy,&newenergy);
    newxy.x = xy.x - dx;
    newxy.y = xy.y + dy;
	GetNodeEnergy(n,&newxy,&tryenergy);
	if (tryenergy<newenergy)
	{
		newenergy=tryenergy;
		movedir->x=-dx;
		movedir->y=dy;
	}
    newxy.x = xy.x + dx;
    newxy.y = xy.y - dy;
	GetNodeEnergy(n,&newxy,&tryenergy);
	if (tryenergy<newenergy)
	{
		newenergy=tryenergy;
		movedir->x=dx;
		movedir->y=-dy;
	}
    newxy.x = xy.x - dx;
    newxy.y = xy.y - dy;
	GetNodeEnergy(n,&newxy,&tryenergy);
	if (tryenergy<newenergy)
	{
		newenergy=tryenergy;
		movedir->x=-dx;
		movedir->y=-dy;
	}
	dE=startenergy-newenergy;
	//printf("node=%d\tdE=%f\n",n,dE);
	if (dE>0.0)
	{
		/*mobility=dE*(float)SpeedUp/incroffset;
		if (mobility>1000.0)	mobility=1000.0;*/
		mobility=dE*ElleSpeedup();
		if (mobility>1.0)	mobility=1.0;
        movedir->x *= mobility;
        movedir->y *= mobility;
	}
}

double GetAngleEnergy(double angle)
{
    int angdeg;
	double	C,energy,sinangle;

    angdeg = (int)(angle*RTOD + 0.5);
    energy = ElleEnergyLUTValue(angdeg);
	return(energy);
}

int GetNodeEnergy(int n,Coords *xy,float *energy)
{
    int		i,nb[3];
    Coords  xynb;
    double	cosalpha,alpha,E;
    double  len;
    int		G0,G1;
    ERegion  rgn[3], rgn2;
    Coords_3D xyz[3];
    float csl_factor;


    ElleRegions(n,rgn);
    ElleNeighbourNodes(n,nb);
    E = 0.0;

    for (i=0;i<3;i++) {
        if (nb[i]!=NO_NB) {
            ElleRelPosition(xy,nb[i],&xynb,&len);
            ElleNeighbourRegion(nb[i],n,&rgn2);
            
            //ElleGetFlynnCAxis(rgn[i],&xyz[0]);
	    
            //ElleGetFlynnCAxis(rgn2,&xyz[1]);
            
	    ElleGetFlynnEulerCAxis(rgn[i],&xyz[0]);
	    
            ElleGetFlynnEulerCAxis(rgn2,&xyz[1]);
	    
            GetCSLFactor(xyz,&csl_factor);
	    //printf("%d\t%f\n",n,csl_factor);

            if (len==0.0) len = 1.0;
            cosalpha = ((double)xynb.x * xyz[0].x + xynb.y * xyz[0].y)/len;
            /*
             * make sure cosalpha is in range -1,1 for acos
             */
            if (cosalpha > 1.0) cosalpha = 1.0;
            if (cosalpha < -1.0) cosalpha = -1.0;
            if (cosalpha >= 0.0) alpha = acos(cosalpha);
            else                 alpha = acos(-cosalpha);
            E += len*GetAngleEnergy(alpha)*csl_factor;
            cosalpha = ((double)xynb.x * xyz[1].x + xynb.y * xyz[1].y)/len;
            /*
             * make sure cosalpha is in range -1,1 for acos
             */
            if (cosalpha > 1.0) cosalpha = 1.0;
            if (cosalpha < -1.0) cosalpha = -1.0;
            if (cosalpha >= 0.0) alpha = acos(cosalpha);
            else                 alpha = acos(-cosalpha);

            E += len*GetAngleEnergy(alpha)*csl_factor;

        }
    }
    
    *energy=(float)E;

    return(0);
}

int GetCSLFactor(Coords_3D xyz[3], float *factor)
{
    double misorientation;
    double tmp;
    double dotprod;
    
    dotprod=(xyz[0].x*xyz[1].x)+(xyz[0].y*xyz[1].y)+(xyz[0].z*xyz[1].z);

    if(fabs(dotprod-1.0) < 0.0001)
	misorientation=0.0;
    else
	misorientation = fabs(acos(dotprod)*RTOD);
    
    tmp=1.0/((misorientation/90.0)+0.1);

    tmp=pow(tmp,4.0);

    tmp=(10000.0-tmp)/10000.0;

    *factor=tmp;
     
    // CSL function produces 0% at 0 degrees, 95% at 10 degrees 
    // and 99% at 20 degrees c axis misorientation 
}


#if XY
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos);
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos)
{
    float alpha,beta,gamma;

    ElleGetFlynnEuler3(flynn_no,&alpha,&beta,&gamma);
	
	alpha=alpha*DTOR;
	beta=beta*DTOR;

    dircos->x=sin((double)beta)*cos((double)alpha);
    dircos->y=-sin((double)beta)*sin((double)alpha);
    dircos->z=cos((double)beta);

    //converts Euler angles into direction cosines of c-axis
}
#endif

	    
            
