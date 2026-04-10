#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "convert.h"
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "string_utils.h"
#include "runopts.h"
#include "lut.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "splitting.h" 
#include "mineraldb.h"
#include "mat.h"
#include "timefn.h"
#include "split.h"
#include "update.h"


int Split();
bool SplitFlynn1(int flid, float dx, float dy);
void SplitFlynn0(int flid, int mineral);
void CalcSplitAngle(double *cax,float *gox, float *goy); 
int InitThisProcess();
 


/********************physical constants*********************************/
double energyofdislocations;		// in Jm-1	(from mineraldb)
double dislocationdensityscaling=10e13; // in m-2	(ie 1.5 in elle file =
                                                       //1.5e13

double lengthscale = 1e-3;		// in m 	(sides of box)
/***********************************************************************/
 
#define FLUID_PRESENCE 1
//minNodeSep?
double flynn_area[2]  = {0.0025*0.05*10.0, 0.0025*0.05*20.0} ;
double TopoMinArea;

static int Count;
int test=0;

int InitThisProcess()
{
    int err=0;
    int max;
    char *infile;
    
    ElleReinit();
    ElleSetRunFunction(Split);
    TopoMinArea = ElleminNodeSep()*ElleminNodeSep()*SIN60*0.5;

	UserData userdata;
    int splitMode;
 	ElleUserData(userdata);
   	splitMode=(int)userdata[SMode]; // default calculation mode

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes
         */
        if (!ElleFlynnAttributeActive(MINERAL))
            ElleAttributeNotInFile(infile,MINERAL);
        if (splitMode==0) {
//          if (!ElleFlynnAttributeActive(EULER_3))
 //             ElleAttributeNotInFile(infile,EULER_3);
          if (!ElleFlynnAttributeActive(DISLOCDEN)) 
              ElleAttributeNotInFile(infile,DISLOCDEN);
        }
        if (!ElleFlynnAttributeActive(SPLIT)){
           /*Split Attribute is set in if necessary*/
	    ElleInitFlynnAttribute(SPLIT);
	    printf("put in FlynnAttribute Split");
	    }
    }
    else{ /* warn that there is no file open */;
    }
    
    Count = 0;
    //return(err);
}
#ifdef __cplusplus 
extern "C" { 
#endif 
    int tbh_(); 
#ifdef __cplusplus 
} 
#endif

/* main splitting routine*/
 
int Split()
{
    int i, j, k, l, m, min_type, type,  split;
    float min_prob, max_prob;
    int interval=0,st_interval=0,err=0,max;
    
    
    double dislocden;
    double fluid_factor;
    double test, critical_energy, energy;
    char fname[32];
    FILE *fp;
    extern int Count;
    double attrib;
    float try_x,try_y; 
    int noevents=0, no_splits=0,split_test,mintype;
    double ran_ang;
    double e;
	  UserData userdata;
    int splitMode;
 
 	  ElleUserData(userdata);
   	splitMode=(int)userdata[SMode]; // Change default calculation mode
   
    /*srand(currenttime());*/
    NodeAttributeStats attr_stats;
    /*  values used for probability calculation*/
    min_prob=0.0;  // allowed minimum value
    max_prob=0.5;  // allowed maximum value
    
    //clean-up topology
    if (Count==0) {
        ElleAddDoubles();
    }
    if (ElleDisplay()) EllePlotRegions(Count);
    interval = EllesaveInterval();
    st_interval = ES_statsInterval();
    if (st_interval>0) ES_WriteStatistics(0);
    
    
    for (i=0;i<EllemaxStages();i++) {
        max = ElleMaxFlynns();
        for (k=0;k<max;k++) {
	      	if(splitMode==0)
  			{
	  				// should this process init EULER_3 if not active
						// or flag attribute error??
            if (ElleFlynnIsActive(k)&& !isParent(k) /*&& !hasParent(k)*/) {
                ElleGetFlynnRealAttribute(k, &dislocden, DISLOCDEN);
                ElleGetFlynnIntAttribute(k, &type, MINERAL);
                if (QUARTZ == type) min_type = 0;    
                if (FELDSPAR == type) min_type = 1; 

                /* split_threshold with no temperature dependence */

                //split_threshold[0]=split_ratio[0];
                //split_threshold[1]=split_ratio[1];
               
                energyofdislocations=GetMineralAttribute(type,DD_ENERGY); 
                //critical_energy=
                //(1000/(temp+1000))*GetMineralAttribute(type,CRITICAL_SG_ENERGY);
                
                critical_energy=
                GetMineralAttribute(type,CRITICAL_SG_ENERGY);
                
                energy =
                dislocden*energyofdislocations*dislocationdensityscaling;
                

                /* probabilistic pick of splitting*/  
                //split_threshold[min_type]=split_threshold[min_type]/energy;
                critical_energy= critical_energy/energy;
								test=ElleRandomD();	
                
                //printf(" \n test = %f\n", test);
		
                if(test>critical_energy)
                {

                    /*ElleGetFlynnIntAttribute(k,&split, SPLIT);*/
                    /*if (!split) {*/
                         
                         SplitFlynn0 (k, min_type);
                         noevents++;
                    /*}*/
                }
          }
    		}
    	  else if(splitMode==1)
    	  	{
    	  		if (ElleFlynnIsActive(k)) 
    	  		{
    	    		/* random values for split direction vector*/
    	    		ElleGetFlynnIntAttribute(k, &mintype, MINERAL);
    	    		if(mintype==QUARTZ || mintype==CALCITE)
    	    			{
    		    			/*ran_ang=((double)rand()/RAND_MAX)*2.0*acos(0.0);*/
    		    			ran_ang=ElleRandomD()*2.0*acos(0.0);
    
    		    			try_x = cos(ran_ang);
    		    			try_y = sin(ran_ang);    	  
    
    		    			/*test=((double)rand()/RAND_MAX);	*/
								test=ElleRandomD();	
    
    		    			ElleGetFlynnIntAttribute(k,&split_test,SPLIT);
    
    		    			/* probabilistic pick of splitting*/  
				//ElleGetFlynnRealAttribute(k, &e, F_BULK_S);
				ElleGetFlynnRealAttribute(k, &e, BULK_S);
    
    		    if((min_prob<test && test<(max_prob*e)) && (split_test==0)) 
    		    {
                 	if (SplitFlynn1 (k,try_x, try_y)) no_splits++;
						noevents++;
    			}
			}
        }
      }
    }
    printf("*****\nnumber of splitting events = %d splits = %d\n******\n",noevents, no_splits);
    if(splitMode==1)
    {
   		max = ElleMaxFlynns();
    	for (k=0;k<max;k++) 
    	{
			if (ElleFlynnIsActive(k)) 
			{
        		ElleSetFlynnIntAttribute(k,0,SPLIT);
			}
    	}
	} 		
	ElleUpdate();
  }

    return(err);
}

bool SplitFlynn1(int flid, float dx, float dy)
{
    bool split_flynn=false;
    int maxnd;  
    int interval=0,st_interval=0; 
    int child1, child2;
    double fl_area;  
   
    /* Get Area of Flynn that should be split*/
    fl_area = fabs(ElleRegionArea (flid));
    
    /* set minimum area for allowed new child of flynn*/
    ElleSetMinFlynnArea(fl_area*0.4);
    
    /* Split Flynn*/
   
    if(ElleSplitFlynn(1,flid,&child1,&child2,dx,dy)==0)
    {
        EllePromoteFlynn(child1);
        EllePromoteFlynn(child2);


        ElleSetFlynnIntAttribute(child1,1,SPLIT);
        ElleSetFlynnIntAttribute(child2,1,SPLIT); 
        ElleSetFlynnRealAttribute(child1,0.0,AGE);
        ElleSetFlynnRealAttribute(child2,0.0,AGE); 
        ElleSetFlynnRealAttribute(child1,0.0,BULK_S);
        ElleSetFlynnRealAttribute(child2,0.0,BULK_S); 
        split_flynn=true;
    }
   
    return(split_flynn);
}

void SplitFlynn0(int flid, int mineral)
{
    int maxnd; 
    double curra, currb, currc; 
    int interval=0,st_interval=0; 
    int child1, child2;
    float gox,goy; 
    double rmap[3][3]; 
    double cax[3];
    double distance,orgdistance;
    double new_area[3], old_area[3]; 
    double new_dislocden[3], old_dislocden[3]; 
    
    
    orgdistance =ElleSwitchdistance();

    if(mineral==0) 
		{ 
		    /* sets Minimum flynn area for Qtz */

            ElleSetMinFlynnArea(flynn_area[0]*0.4);

            ElleGetFlynnEuler3(flid, &curra, &currb, &currc);
                    //retrieve 3 Euler angles   
		    Euler2cax(curra, currb, currc, cax); // convert to c-axis dir cosines 
 
		    CalcSplitAngle(cax,&gox,&goy); 
 
		    //gox=0.5; 
		    //goy=1.000; 
                    /*distance =0.0025;
                    ElleSetSwitchdistance(distance);
                    ElleAddDoubles(); */
                    
                    
            // maintain only two levels
            if (!hasParent(flid)) {
		        if (ElleSplitFlynn(1,flid,&child1,&child2,gox,goy)==0) {
		          ElleSplitFlynnOnArea(1,child1,flynn_area[mineral],gox,goy);
		          ElleSplitFlynnOnArea(1,child2,flynn_area[mineral],gox,goy);
                }
            }
            else
		        ElleSplitFlynnOnArea(1,flid,flynn_area[mineral],gox,goy);
                    
                  } 

     if(mineral==1) 
		  { 
		    /* sets Minimum flynn area for Fsp */
            ElleSetMinFlynnArea(flynn_area[1]*0.4);
            // maintain only two levels
            if (!hasParent(flid)) {
		        if (ElleSplitFlynn(1,flid,&child1,&child2,0,0)==0) {
		          ElleSplitFlynnOnArea(1,child1,flynn_area[mineral],0,0);
		          ElleSplitFlynnOnArea(1,child2,flynn_area[mineral],0,0);
                }
            }
            else

                ElleSplitFlynnOnArea(1,flid,flynn_area[mineral],0,0);
            } 
}
 
void CalcSplitAngle(double *cax, float *gox, float *goy) 
{ 
    double x,y,sx,sy,x2,y2; 
    double rot_angle; 
 
    x=(((double)rand()/RAND_MAX)*2.0)-1; 
    y=(((double)rand()/RAND_MAX)*2.0)-1; 
    x=ElleRandomD()*2.0-1; 
    y=ElleRandomD()*2.0-1; 
 
    if(x<0.0) 
	sx=-1.0; 
    else 
	sx=1.0; 
 
    if(y<0.0) 
	sy=-1.0; 
    else 
	sy=1.0; 
 
 
    x=pow(fabs(x),1+(6.0*(1-cax[2])))*sx; 
    y=pow(fabs(y),1+(6.0*(1-cax[2])))*sy; 
 
 
    if(fabs(cax[0]-1) < 0.000001) {
	rot_angle=0; 
    }
    else if(fabs(cax[0]) < 0.000001) {
    rot_angle=PI/2;
    if (cax[0] < 0) rot_angle*= -1.0;
    }
    else {
	rot_angle=atan(cax[1]/cax[0]); 
    }
     
    x2=(x*cos(rot_angle)) - (y*sin(rot_angle)); 
    y2=(x*sin(rot_angle)) + (y*cos(rot_angle)); 
 
     
    *gox=(float)x2; 
    *goy=(float)y2; 
}  
