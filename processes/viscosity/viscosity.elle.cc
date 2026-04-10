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
#include "mineraldb.h"
#include "unodes.h"
#include "update.h"
#include "viscosity.h"

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
    
    UserData userdata;
    int viscosityMode;
    ElleUserData(userdata);
    viscosityMode=(int)userdata[VMode]; // default calculation mode

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
        if (viscosityMode==0) {
          if (!ElleFlynnAttributeActive(DISLOCDEN)) 
              ElleAttributeNotInFile(infile,DISLOCDEN);
          if (!ElleFlynnAttributeActive(AGE))
              ElleInitFlynnAttribute(AGE);
        }
        if (!ElleFlynnAttributeActive(VISCOSITY))
            ElleInitFlynnAttribute(VISCOSITY);
    }
}

int ProcessFunction()
{
    int i, j, max, mintype,type;
    double dislocden,viscosity,base_visc,temp,age,e;
    int interval=0,st_interval=0;
    int fluid_factor;
    UserData userdata;
    int viscosityMode;
    
    
    ElleUserData(userdata);
       viscosityMode=(int)userdata[VMode]; // Change default calculation mode

    //temp=ElleTemperature();
    ElleCheckFiles();
                                                                                
    if (ElleDisplay()) EllePlotRegions(ElleCount());
    if (FLUID_PRESENCE==1) fluid_factor=1;
    if (FLUID_PRESENCE==0) fluid_factor=2;
    for (i=0;i<EllemaxStages();i++) 
    {
        if (ElleFlynnAttributeActive(VISCOSITY)) {
        max = ElleMaxFlynns();
        for (j=0;j<max;j++)
        {

          if (ElleFlynnIsActive(j) && !isParent(j))
          {
              if(viscosityMode==0)
              { 
                   ElleGetFlynnIntAttribute(j, &mintype, MINERAL);    
                   base_visc=GetMineralAttribute(mintype,VISCOSITY_BASE);

                   /* set viscosity of Fsp in contrast to qtz*/
                   if (mintype==FELDSPAR) base_visc=base_visc*2;

                   //retrieve dislocation density 
                   ElleGetFlynnRealAttribute(j, &dislocden, DISLOCDEN);
                   viscosity=base_visc+fluid_factor*sqrt(dislocden); //?????
                   //printf("\nviscosity = %f\n",viscosity);
                   ElleSetFlynnRealAttribute(j,viscosity,VISCOSITY); 
                   ElleGetFlynnRealAttribute(j, &age, AGE);
                   age=age+1;
                   ElleSetFlynnRealAttribute(j, age, AGE);
              }
              else if(viscosityMode==1)
              {
                   //ElleGetFlynnRealAttribute(j, &e, F_BULK_S);
                   ElleGetFlynnRealAttribute(j, &e, BULK_S);

                   e=sqrt(e); 

                   printf("j,e,area:%d\t%lf\t%lf\t%lf\n",j,e,ElleRegionArea(j),e*ElleRegionArea(j));
         
                   viscosity = .25+(100*ElleRegionArea(j)*e);
            
                   ElleSetFlynnRealAttribute(j,viscosity,VISCOSITY); 
              }

          }
        }
        }
        if (ElleUnodesActive() &&
            ElleUnodeAttributeActive(U_VISCOSITY) &&
            ElleUnodeAttributeActive(U_DISLOCDEN)) {
          max = ElleMaxUnodes();
          for (j=0;j<max;j++)
          {

             /*ElleGetUnodeAttribute(j, &mintype, MINERAL);    */
            mintype = QUARTZ;
            if(viscosityMode==0)
            { 
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
        }
        ElleUpdate();
    }

} 
