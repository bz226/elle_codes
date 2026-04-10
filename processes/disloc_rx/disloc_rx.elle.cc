#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "display.h"
#include "errnum.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "stats.h"
#include "init.h"
#include "mineraldb.h"
#include "mat.h"
#include "misorient.h"
#include "update.h"

int InitThisProcess(), ProcessFunction();
/* for checking of random orientation of newly grown grain*/
void CheckMisorientEuler(int flynn, int *eulertest);

int DoSomethingToFlynn(int flynn);

/********************physical constants*********************************/
double energyofdislocations;    	// in Jm-1	(from mineraldb)
double dislocationdensityscaling=10e13; // in m-2    (ie 1.5 in elle file =1e13 in real worlds=

//double lengthscale = 1e-3;    	// in m 	(sides of box)
double lengthscale = ElleUnitLength();   	// in m 	(sides of box)
/***********************************************************************/


#define FLUID_PRESENCE 1
/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */


    ElleReinit();

    ElleSetRunFunction(ProcessFunction);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes
         */
        if (!ElleFlynnAttributeActive(MINERAL))
            ElleAttributeNotInFile(infile,MINERAL);
        if (!ElleFlynnAttributeActive(DISLOCDEN)) 
            ElleAttributeNotInFile(infile,DISLOCDEN);

    }
}

#ifdef __cplusplus
extern "C" {
#endif
    int tbh_();
    long currenttime(void);
#ifdef __cplusplus
}
#endif
int ProcessFunction()
{
    int i, j, l,max, mintype,type,split,testE;
    double energy,critical_energy,temp, dislocden,age;
    double cycle;
    int interval=0,st_interval=0;
    double test;
    int noevents=0;
    double fluid_factor;
    /* for random orientation of newly grown grain */
    double curra, currb, currc;
    double olda, oldb, oldc;
    double rmap[3][3];
    int nb[3],nn;

    srand(currenttime());
    temp = ElleTemperature();
    interval = EllesaveInterval();

	ElleCheckFiles();
    if (ElleDisplay()) EllePlotRegions(ElleCount());
    if (FLUID_PRESENCE==1) fluid_factor=1;
    if (FLUID_PRESENCE==0) fluid_factor=1.5;
    for (i=0;i<EllemaxStages();i++) 
    {
        
        max = ElleMaxFlynns(); 
        for (j=0;j<max;j++)
        {
            
           if (ElleFlynnIsActive(j) && !isParent(j))
           {
               
    	        ElleGetFlynnIntAttribute(j, &mintype, MINERAL);

              //retrieve critical energy for recrystallization QUARTZ
                if (QUARTZ==mintype)
                {
    	          critical_energy=
                      (GetMineralAttribute(mintype,CRITICAL_RX_ENERGY)
                      * (1000/(temp+1000)) * fluid_factor);
                  energyofdislocations=
                      GetMineralAttribute(mintype,DD_ENERGY); 
                }
 
              //to make critical energy for Fsp a lot higher than for Quartz
            	if (FELDSPAR==mintype)
                {
                  critical_energy=
                      (GetMineralAttribute(mintype,CRITICAL_RX_ENERGY)
                      *(1000/(temp+1000))*fluid_factor);
                  energyofdislocations=
                      GetMineralAttribute(mintype,DD_ENERGY); 
    	        }

              //retrieve dislocation  density 
                ElleGetFlynnRealAttribute(j, &dislocden, DISLOCDEN);
                
                /* calculation of present energy in grain due to disloc*/            
                energy =
                   dislocden*energyofdislocations*dislocationdensityscaling;
                
                /* probabilistic pick of recrystallizing*/
                critical_energy= critical_energy/energy;
                test=(double)rand()/(double)RAND_MAX;    
               

                /* another kind of test!- not used here
                diff = dislocden-threshold;
                max_diff =max-threshold;
                val = diff/max_diff;
                if (test<val)*/

             /* tests if dislocden high enough and give it a probability*/
       	        if (test>critical_energy)
                {
    	     /* for random orientation of newly grown grain*/
                     
                    ElleGetFlynnEuler3(j,&olda, &oldb, &oldc);
                    l=0; 
                    do {
                        orientZXZ(rmap);
                        uneulerZXZ(rmap,&curra,&currb,&currc);
                        ElleSetFlynnEuler3(j, curra, currb, currc);
                        CheckMisorientEuler(j, &testE);
                        l++;
                    }while ((testE==0) && (l<11));
                    if (testE==1){   
                        dislocden=0.001;
     	                noevents++;
            	        ElleSetFlynnRealAttribute(j,dislocden, DISLOCDEN);
                        //printf(" \n flynn %d\n",j);
                    } 
                    else ElleSetFlynnEuler3(j,olda, oldb, oldc);
    	        }
            }
        }
    printf("***********\nno disloc_rx events = %d\n**********\n",noevents);
        /*
         * update
         */
        ElleUpdate();
    }
    /*ElleAutoWriteFile(Count);*/
    
} 

/* for checking of random orientation of newly grown grain*/
void CheckMisorientEuler(int flynn, int *euler)
{
    int mintype[2],test;
    double critical_angle, misorient;
	std::list<int>::iterator it;

    test=1;

    ElleGetFlynnIntAttribute(flynn, &mintype[0], MINERAL);
    critical_angle=GetMineralAttribute(mintype[0],CRITICAL_MISORIENT) ;
	std::list<int> nbflynns;
    ElleFlynnNbRegions(flynn,nbflynns);
    for (it=nbflynns.begin();it!=nbflynns.end() && test==1;it++)
    {
      ElleGetFlynnIntAttribute(*it,&mintype[1],MINERAL);
      if (mintype[1]==mintype[0]){
        CalcMisorient(flynn, *it, &misorient);
        if (misorient<critical_angle) test=0;
      }
    }
    *euler=test;

}
