#include <stdio.h>
#include <math.h>
#include <string.h>
#include "flynnarray.h"
#include "attrib.h"
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "errnum.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "timefn.h"
#include "update.h"
#include "mineraldb.h"
#include "unodes.h"

int InitThisProcess(), ProcessFunction();

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
            if (!ElleUnodesActive() || !ElleUnodeAttributeActive(U_DISLOCDEN))
                ElleAttributeNotInFile(infile,DISLOCDEN);
    }
    
}

int ProcessFunction()
{
    int i, j, max, mintype, type;
    double dislocden,recovery_rate,temp=400, base_visc, viscosity;
    double fluid_factor;
    int interval=0,st_interval=0;

    interval = EllesaveInterval();
    temp=ElleTemperature();

    ElleCheckFiles();
   
    if (ElleDisplay()) EllePlotRegions(ElleCount());
    if (FLUID_PRESENCE==1) fluid_factor=1;
    if (FLUID_PRESENCE==0) fluid_factor=1.1;//???
    for (i=0;i<EllemaxStages();i++) 
    {
        if (ElleUnodesActive())
        {
        max = ElleMaxUnodes();
        for (j=0;j<max;j++)
        {
             /*ElleGetUnodeAttribute(j, &mintype, MINERAL);    */
            mintype = QUARTZ;
             base_visc=GetMineralAttribute(mintype,VISCOSITY_BASE);
             /*
              * viscosity should be based on orientation of xl?
              */
             ElleGetUnodeAttribute(j, &base_visc, U_VISCOSITY);
                                                                                
             /* set viscosity of Fsp in contrast to qtz*/
             if (mintype==FELDSPAR) base_visc=base_visc*2;
                                                                                
             //retrieve dislocation density
             ElleGetUnodeAttribute(j, &dislocden, U_DISLOCDEN);
             viscosity=base_visc+fluid_factor*sqrt(dislocden); //?????
                //printf("\nviscosity = %f\n",viscosity);
             ElleSetUnodeAttribute(j,viscosity,U_VISCOSITY);
        }
        }
		else 
		{
        max = ElleMaxFlynns();
        for (j=0;j<max;j++)
        {
            if (ElleFlynnIsActive(j) && !isParent(j))
            {
            
                ElleGetFlynnIntAttribute(j, &mintype, MINERAL);
                recovery_rate=GetMineralAttribute(mintype,RECOVERY_RATE);
                //printf(" recovery_rate= %f\n",recovery_rate); 
                recovery_rate=fluid_factor*
                              (0.97*recovery_rate+0.03*recovery_rate*
                              (1000/(temp+1000)));//?????
                //retrieve dislocdensity
                ElleGetFlynnRealAttribute(j, &dislocden, DISLOCDEN);
         //printf(" recovery_rate= %f\n",recovery_rate);    
                dislocden=dislocden*recovery_rate;

                ElleSetFlynnRealAttribute(j,dislocden, DISLOCDEN);
        
            }
        }
        }
		ElleUpdate();
    }
} 
