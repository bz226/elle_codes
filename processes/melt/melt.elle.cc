#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "errnum.h"
#include "error.h"
#include "general.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "crossings.h"
#include "convert.h"
#include "lut.h"
#include "timefn.h"
#include "mineraldb.h"
#include "update.h"
#include "melt.h"

/********************physical constants*********************************/
double energyofdislocations;        // in Jm-1    (from mineraldb)
double energyofsurface;         // in Jm-2    (from mineraldb)
double mobilityofboundary;        // in m2s-1J-1    (aka fudge factor)
double dislocationdensityscaling=10e13; // in m-2    (ie 1.5 in elle file = 1.5e13 in real world)



//this is not necessarily the right time step!
double truetimestep = 3.1536e10;    // in s     (1000yrs here, not including leap years)
//it should be whatever is set in the input file!
//truetimestep=ElleTimestep();



double lengthscale = 1e-3;        // in m     (sides of box)
double R = 8.314472;    // in Jmol-1K-1    (Gas constant)
double Qgbm = 200;        // in Jmole-1    (Activation energy for GBM  (made up number))
/***********************************************************************/

using std::cout;
using std::endl;

//#define DEBUG

int GBM();
int InitGBM();

const int MeltId=F_ATTRIB_A; // melt phase

double  GetAngleEnergy(double angle);
int GBM_MoveNode(int n, Coords *movedir, int *same,double arealiq,
                 double factor, double fudge,double areaequil,
                 double energyxlxl, double energyliqxl,
                 double energyliqliq);
int GetBodyNodeEnergy(int n, double *total_energy);
int GetCSLFactor(Coords_3D xyz[3], float *factor);
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos);
int GetSurfaceNodeEnergy(int n,int meltmineral,Coords *xy,double *energy,
                         double arealiq,double fudge, double areaequil,
                         double energyxlxl, double energyliqxl,
                         double energyliqliq);
double CalcAreaLiq(int meltmineral);
void ElleNodeSameAttrib(int node, int *same, int *samemin, int attrib_id);
extern double ElleSwitchLength();

double    incroffset=0.002; /* this is reassigned in GBE_GrainGrowth */

int InitGBM()
{
    int err=0;
    int max;
    char *infile;
    extern UserData CurrData;

long tmp;
tmp = currenttime();
#ifdef DEBUG
tmp = 1028731772;
#endif
cout << "seed= " << tmp << endl;
    srand(tmp);

    ElleReinit();
    ElleSetRunFunction(GBM);

    infile = ElleFile();
    if (strlen(infile)>0)
    {
        if (err=ElleReadData(infile)) OnError(infile,err);


    }
    else /* warn that there is no file open */;
}

int GBM()
{
    int i, j, k, same;
    int interval=0,st_interval=0,err=0,max;
    char fname[FILENAME_MAX];
    FILE *fp;
    Coords incr;
    UserData CurrData;
    double old_speedup, new_speedup;
    double arealiq=0;
	  double upper=0.005; // upper bound melt variation
	  double lower=0.001; // lower bound melt variation
    
    if (ElleCount()==0){
        ElleAddDoubles();
        ElleUserData(CurrData);
        cout << "mob= " << CurrData[Mobility] << endl;
        cout << "magic= " << CurrData[Magic] << endl;
        cout << "percent= " << CurrData[EquilMeltFrac] << endl;
        cout << "xl-xl= " << CurrData[XlXlEnergy] << endl;
        cout << "liq-xl= " << CurrData[LiqXlEnergy] << endl;
        cout << "liq-liq= " << CurrData[LiqLiqEnergy] << endl;
        cout << "inc change= " << CurrData[IncMeltfraction] << endl;
    	if(CurrData[EquilMeltFrac]<0) {
            // calculate global melt fraction for input file
	    	CurrData[EquilMeltFrac]=CalcAreaLiq(MeltId);
            /*
             *  uncomment if it should be saved in the user options
            ElleSetUserData(CurrData);
             */

        cout << "areaequil= " << CurrData[EquilMeltFrac] << endl;
        }
    }
    if (ElleDisplay()) EllePlotRegions(ElleCount());

    incroffset = ElleSwitchdistance() * 0.01;

	ElleCheckFiles();

    max = ElleMaxNodes();
    for (k=0;k<max;k++) {
         if (ElleNodeIsActive(k)) {
            if (ElleNodeIsDouble(k))
               ElleCheckDoubleJ(k);
            else if (ElleNodeIsTriple(k))
               ElleCheckTripleJ(k);
         }
    }
//printf("%10.8lf %10.8lf\n", ElleminNodeSep(),ElleSwitchLength());
    for (i=0;i<EllemaxStages();i++) {
        arealiq = CalcAreaLiq(MeltId);

		CurrData[EquilMeltFrac]+= CurrData[IncMeltfraction];

		if((fabs(arealiq-CurrData[EquilMeltFrac]))/CurrData[EquilMeltFrac]>upper) //.00003 orig
			CurrData[Magic]=CurrData[Magic]*1.0075;
		else if((fabs(arealiq-CurrData[EquilMeltFrac]))/CurrData[EquilMeltFrac]<lower)
			CurrData[Magic]=CurrData[Magic]/1.0025;

		if((fabs(arealiq-CurrData[EquilMeltFrac]))/CurrData[EquilMeltFrac]>upper) 
        
        cout << "i arealiq %diseq fudge equil:  " << i << " " << arealiq << " " <<100*(fabs(arealiq-CurrData[EquilMeltFrac])/CurrData[EquilMeltFrac]) << " " <<CurrData[Magic] << " " <<CurrData[EquilMeltFrac] << endl;
 
        max = ElleMaxNodes();
        if (i%2) {
            for (k=0;k<max;k++) {
                if (ElleNodeIsActive(k)) {
                    GBM_MoveNode(k,&incr,&same,arealiq,
                                 CurrData[Mobility],CurrData[Magic],
                                 CurrData[EquilMeltFrac],
                                 CurrData[XlXlEnergy],
                                 CurrData[LiqXlEnergy],
                                 CurrData[LiqLiqEnergy]);
                   // if(same && !(incr.x==0 && incr.y==0))
                    /*{*/
                         ElleCrossingsCheck(k,&incr);
                    /*if (ElleNodeIsDouble(k))
                            ElleCheckDoubleJ(k);
                    else if (ElleNodeIsTriple(k))
                            ElleCheckTripleJ(k);
            }*/
                }
            }
        }
        else {
            for (k=max-1;k>=0;k--) {
                if (ElleNodeIsActive(k)) {
                    GBM_MoveNode(k,&incr,&same,arealiq,
                                 CurrData[Mobility],CurrData[Magic],
                                 CurrData[EquilMeltFrac],
                                 CurrData[XlXlEnergy],
                                 CurrData[LiqXlEnergy],
                                 CurrData[LiqLiqEnergy]);
                   // if(same && !(incr.x==0 && incr.y==0))
                    /*{*/
                        ElleCrossingsCheck(k,&incr);
                    /*if (ElleNodeIsDouble(k))
                            ElleCheckDoubleJ(k);
                    else if (ElleNodeIsTriple(k))
                            ElleCheckTripleJ(k);
            }*/
            }
            }
        }
        max = ElleMaxNodes();
        for (j=0;j<max;j++) {
            if (ElleNodeIsActive(j)){
                if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
                else if (ElleNodeIsTriple(j)) ElleCheckTripleJ(j);

            }
        }
		ElleUpdate();
        if (ElleNodeCONCactive() && (ElleCount()%20==0))
            /*printf("mass= %lf \n",ElleSumNodeMass(CONC_A))*/;
    }
    //ElleSetSpeedup(old_speedup);
}

int GBM_MoveNode(int n, Coords *movedir, int *same, double arealiq,
                 double factor, double fudge,double areaequil,
                 double energyxlxl, double energyliqxl,
                 double energyliqliq)
{
    register int    i,l;
    double     tryenergy,newenergy,dx,dy,dx2,dy2,dEdX,mobility,startenergy;
    double     temp,velocity,distance;
    double   randangle,angle;
    Coords   xy, newxy,incr,prev,new_incr;
    ERegion  rgn[3];
    double   new_area[3],area[3],old_area[3] ;
    double   old_density[3], new_density[3], density[3], energy[3] ;
    double   surfaceenergy=0,bodyenergy=0;
    double   startsurfaceenergy=0,startbodyenergy=0;
    int      themin,mintype[3];
    double   minmobfactor;


	// so far there are no body energies in this model, so these values are always zero

    ElleNodePosition(n,&xy);
    newxy = xy;

    ElleRegions(n,rgn);

    mobilityofboundary=factor; // user data 1

    randangle=(rand()/(double)RAND_MAX/2.0)*M_PI;
    dx=cos(randangle)*incroffset;
    dy=sin(randangle)*incroffset;
    //printf("rand=%lf\tdx=%le\ty=%le\n",randangle*180.0/M_PI,dx,dy);

	double dblarealiq = CalcAreaLiq(MeltId);
    GetSurfaceNodeEnergy(n,MeltId,&newxy,&startsurfaceenergy,dblarealiq,
                         fudge,areaequil,energyxlxl,energyliqxl,
                         energyliqliq);
    //printf("startbody=%le\tstartsurface=%le\n",startbodyenergy,startsurfaceenergy);
    startenergy=startbodyenergy+startsurfaceenergy;
    movedir->x=0;
    movedir->y=0;
    newenergy = 0;

    ElleNodePosition(n,&xy);
    ElleNodePrevPosition(n,&prev);
    for(i=0;i<4;i++)
    {
        angle=i*M_PI/2;
        rotate_coords(dx, dy, 0.0, 0.0, &dx2, &dy2, angle);

        newxy.x = xy.x+dx2;
        newxy.y = xy.y+dy2;
        ElleSetPosition(n,&newxy);

	    dblarealiq = CalcAreaLiq(MeltId);
        GetSurfaceNodeEnergy(n,MeltId,&newxy,&surfaceenergy,dblarealiq,
                             fudge,areaequil,energyxlxl,energyliqxl,
                             energyliqliq);

        tryenergy=(bodyenergy-startbodyenergy)+(surfaceenergy-startsurfaceenergy);

        if (tryenergy<newenergy)
        {
            newenergy=tryenergy;
            new_incr.x=movedir->x=dx2;
            new_incr.y=movedir->y=dy2;
        }
    }
    ElleSetPosition(n,&xy);
    ElleSetPrevPosition(n,&prev);

    dEdX=-newenergy;

    //printf("dE=%le\n",dEdX);
    if (dEdX>0.0)
    {
        temp=ElleTemperature();
        mobility=mobilityofboundary*ElleSpeedup();
        velocity=mobility*dEdX;
        distance=velocity*truetimestep;

        if (distance>1.0)
        {
            distance=1.0;
            printf("distance=%le\n",distance);
        }
        movedir->x *= distance/(incroffset*lengthscale);
           movedir->y *= distance/(incroffset*lengthscale);
        new_incr.x *=distance/(incroffset*lengthscale);
        new_incr.y *=distance/(incroffset*lengthscale);
        //printf("io=%le temp=%le dE=%le mob=%le velocity=%le distance=%le x=%le y=%le\n",
            //incroffset,temp,dEdX,mobility,velocity,distance,new_incr.x,new_incr.y);
        newxy.x = xy.x+new_incr.x;
        newxy.y = xy.y+new_incr.y;
        ElleSetPosition(n,&newxy);
        if(ElleFlynnAttributeActive(DISLOCDEN))
        {
            for (l=0;l<3;l++) {
                if (rgn[l]!=NO_NB) {
                    new_area[l] = fabs((double)ElleRegionArea (rgn[l]));
                    if (new_area[l]>old_area[l]){
								new_density[l] = (old_area[l]/new_area[l])*old_density[l];
                         //printf("area ratio= %e\n",old_area[l]/new_area[l]);
                        ElleSetFlynnRealAttribute(rgn[l],new_density[l],DISLOCDEN);
                        //^****************************EEEEEEEEEKKKKKKK!!!!!!!!!
                    }
                }
            }
        }
        ElleSetPosition(n,&xy);
        ElleSetPrevPosition(n,&prev);
    }
    else
    {
        movedir->x = 0;
        movedir->y = 0;
    }
}

int GetBodyNodeEnergy(int n, double *total_energy)
{
    int        i,nb[3];
    ERegion rgn[3];
    double  sum_energy;
    double   area[3],energy[3];
    double density[3];
    UserData udata;
    ElleRegions(n,rgn);

    sum_energy =0;

    for (i=0;i<3;i++) {
        if (rgn[i]!=NO_NB) {
            ElleGetFlynnRealAttribute(rgn[i],&density[i],DISLOCDEN);
            area[i] = fabs(ElleRegionArea (rgn[i]))*(lengthscale*lengthscale);
            energy[i] = energyofdislocations*area[i]*density[i]*dislocationdensityscaling;
            sum_energy += energy[i];
            //printf("area=%le\tdensity=%le\tenergy=%le\n",area[i],density[i],energy[i]);

        }
    }

    *total_energy=sum_energy;
    //*total_energy=0.0; //  <*****************************EEEEEEEEEEEEEEKKKKKK

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

    if(misorientation>90.0)
    	misorientation=180-misorientation;

    tmp=1.0/((misorientation/90.0)+0.1);

    tmp=(10.0-tmp)/10.0;

    //*factor=(float)tmp;  // gbm version (supresses sub-gbm)
    //*factor=1.0-tmp; // sub gbm version (supresses gbm)


    // CSL function produces 0% at 0 degrees, 95% at 10 degrees
    // and 99% at 20 degrees c axis misorientation

	*factor=1.0; // normal mode, no CSL
}


#if XY
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos);
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos)
{
    double alpha,beta,gamma;

    ElleGetFlynnEuler3(flynn_no,&alpha,&beta,&gamma);

	alpha=alpha*DTOR;
	beta=beta*DTOR;

    dircos->x=sin(beta)*cos(alpha);
    dircos->y=-sin(beta)*sin(alpha);
    dircos->z=cos(beta);

    //converts Euler angles into direction cosines of c-axis
}
#endif

int GetSurfaceNodeEnergy(int n,int meltmineral,Coords *xy,double *energy,
                        double arealiq, double fudge, double areaequil,
                        double energyxlxl, double energyliqxl,
                        double energyliqliq)
{
    int        i,nb[3];
    Coords  xynb;
    double    cosalpha,alpha,E;
    double  len;
    int        G0,G1;
    ERegion  rgn[3], rgn2,rgn3;
    Coords_3D xyz[3];
    float csl_factor;
    double val2,val3;
    double energyofsurface;

    for (i=0;i<3;i++)
    	xyz[i].x = xyz[i].y = xyz[i].z = 0;

    ElleRegions(n,rgn);
    ElleNeighbourNodes(n,nb);
    E = 0.0;

    for (i=0;i<3;i++) {
        if (nb[i]!=NO_NB) {
            ElleRelPosition(xy,nb[i],&xynb,&len);
            ElleNeighbourRegion(nb[i],n,&rgn2); // get one neighbouring flynn
            ElleNeighbourRegion(n,nb[i],&rgn3); // get other neighbouring flynn


	    	// meltmineral=1, melt
	    	// meltmineral=0, xl

           // get code of one neighbouring flynn
	    	ElleGetFlynnRealAttribute(rgn2, &val2, meltmineral);
           // get code of other neighbouring flynn
	    	ElleGetFlynnRealAttribute(rgn3, &val3, meltmineral);

    	    energyofsurface=energyliqxl;      // interface energy of xl-melt

	    	if(val2==val3)
	    	{
        		if ( val2==0)
					energyofsurface=energyxlxl; // interface energy of xl-xl
        		else
					energyofsurface=energyliqliq; //interface energy of melt-melt
    	    }

            if(ElleFlynnHasAttribute(rgn2,CAXIS) && ElleFlynnHasAttribute(rgn3,CAXIS) )
            {
        		ElleGetFlynnCAxis(rgn3,&xyz[0]);
        		ElleGetFlynnCAxis(rgn2,&xyz[1]);
        		GetCSLFactor(xyz,&csl_factor);
            }
            else if(ElleFlynnHasAttribute(rgn2,CAXIS))
            {
        		ElleGetFlynnCAxis(rgn2,&xyz[0]);
        		csl_factor=1.0;
            }
            else if(ElleFlynnHasAttribute(rgn3,CAXIS) )
            {
        		ElleGetFlynnCAxis(rgn3,&xyz[0]);
        		csl_factor=1.0;
            }
            else if(ElleFlynnHasAttribute(rgn2,EULER_3)&& ElleFlynnHasAttribute(rgn3,EULER_3))
            {
        		ElleGetFlynnEulerCAxis(rgn3,&xyz[0]);
        		ElleGetFlynnEulerCAxis(rgn2,&xyz[1]);
        		GetCSLFactor(xyz,&csl_factor);
            }
            else if(ElleFlynnHasAttribute(rgn2,EULER_3))
            {
        		ElleGetFlynnEulerCAxis(rgn2,&xyz[0]);
        		csl_factor=1.0;
            }
            else if(ElleFlynnHasAttribute(rgn3,EULER_3) )
            {
        		ElleGetFlynnEulerCAxis(rgn3,&xyz[0]);
        		csl_factor=1.0;
            }
		    else
        		csl_factor=1.0;


            csl_factor=1.0; //  <**************************EEEEEEEEEEEEEEKKKKKK

            if (len==0.0) len = 1.0;

            cosalpha = ((double)xynb.x * xyz[0].x + xynb.y * xyz[0].y)/len;

            /*
             * make sure cosalpha is in range -1,1 for acos
             */
            if (cosalpha > 1.0) cosalpha = 1.0;
            if (cosalpha < -1.0) cosalpha = -1.0;
            if (cosalpha >= 0.0) alpha = acos(cosalpha);
            else                 alpha = acos(-cosalpha);
            E += len*GetAngleEnergy(alpha)*csl_factor*energyofsurface*lengthscale;

	    	//if(val2 != MeltId || val3 != MeltId)
			//	printf("alpha xyz[0].x xyz[0].y :%lf %lf %lf\n",alpha*180/3.1415927,xyz[0].x,xyz[0].y );

            cosalpha = ((double)xynb.x * xyz[1].x + xynb.y * xyz[1].y)/len;

        	/*
             * make sure cosalpha is in range -1,1 for acos
             */
            if (cosalpha > 1.0) cosalpha = 1.0;
            if (cosalpha < -1.0) cosalpha = -1.0;
            if (cosalpha >= 0.0) alpha = acos(cosalpha);
            else                 alpha = acos(-cosalpha);

            E += len*GetAngleEnergy(alpha)*csl_factor*energyofsurface*lengthscale;

        }
    }

	E += fabs((arealiq - areaequil)/areaequil) * fudge; // area balancing fudge to ensure global melt fraction is roughly stable


    *energy=(float)E;

    return(0);
}

double GetAngleEnergy(double angle)
{
    int angdeg;
    double    C,energy,sinangle;

    angdeg = (int)(angle*RTOD + 0.5);
    energy = ElleEnergyLUTValue(angdeg);
    return(energy);
}

double CalcAreaLiq(int meltmineral)
{
    int i;
    int max = ElleMaxFlynns();
    double val;
	double area=0;

    if (ElleFlynnAttributeActive(meltmineral)) {
        for (i=0;i<max;i++)  {
            if (ElleFlynnIsActive(i)) {
                ElleGetFlynnRealAttribute(i,&val,meltmineral);
                if (val>0) area+=ElleFlynnArea(i);
            }
        }
    }
    return(area);
}

void ElleNodeSameAttrib(int node, int *same, double *sameval, int attrib_id)
{
    int i,j;
    double theval=0;
    double dval[3];
    ERegion rgn[3];


    *same=1;
    *sameval=0;
    if (ElleFlynnAttributeActive(attrib_id))
	{
        ElleRegions(node,rgn);

        for (i=0;i<3;i++)
		{
            dval[i]=0;
            if (rgn[i]!=NO_NB)
			{
                ElleGetFlynnRealAttribute(rgn[i], &dval[i], attrib_id);
                theval=dval[i];
            }
        }

        for(i=0;i<3;i++)
            if(rgn[i]!=NO_NB && dval[i]!=theval) *same=0;

        if (*same==1) *sameval=theval;
    }
}
