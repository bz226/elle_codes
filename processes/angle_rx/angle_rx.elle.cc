#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "stats.h"
#include "init.h"
#include "mineraldb.h"
#include "mat.h"
#include "misorient.h"
#include "update.h"

static int Count; /* how many iterations have been completed */
                  /* in the ProcessFunction (relate to time) */

int InitThisProcess(), ProcessFunction();

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
void CheckRecrystEvent(int flynn, int *check);

#define pi 3.1415927


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
         *  MINERAL, AGE, CYCLE
         */
        /*if (!ElleFlynnAttributeActive(SPLIT))*/
            /*ElleAttributeNotInFile(infile,SPLIT);*/
        if (!ElleFlynnAttributeActive(MINERAL))
            ElleAttributeNotInFile(infile,MINERAL);
        if (!ElleFlynnAttributeActive(AGE))
            ElleAttributeNotInFile(infile,AGE);
        if (!ElleFlynnAttributeActive(CYCLE)) 
            ElleAttributeNotInFile(infile,CYCLE);
    }

}

int ProcessFunction()
{
    int i, j, fmax, maxno, mintype[2], split;
    int check;
    double cycle;
    int interval=0,st_interval=0;
    double misorient, critical_angle;
    interval = EllesaveInterval();
    int new_grain;
    int noevents=0;
	std::list<int>::iterator it;
    
    if (ElleCount()==0) ElleAddDoubles();
    ElleCheckFiles();
    for (j=0;j<EllemaxStages();j++) 
    {
        fmax = ElleMaxFlynns();
        for (i=0;i<fmax;i++){
            // only look at subgrains
            if (ElleFlynnIsActive(i) && !isParent(i) && hasParent(i)) {
                new_grain=1;
                ElleGetFlynnIntAttribute(i,&mintype[0],MINERAL);
                critical_angle=GetMineralAttribute(mintype[0],CRITICAL_MISORIENT);
				std::list<int> nbflynns;
                ElleFlynnNbRegions(i,nbflynns);
                for (it=nbflynns.begin();it!=nbflynns.end() && new_grain==1;
                                                                    it++) {
                    ElleGetFlynnIntAttribute(*it,&mintype[1],MINERAL);
                    if (mintype[1]==mintype[0]){
                        CalcMisorient(i, *it, &misorient);
                        if (misorient<critical_angle) new_grain=0;
                    }
                }
                if (new_grain==1){
                    while (hasParent(i)) EllePromoteFlynn(i);
                    /*ElleSetFlynnIntAttribute(i,0,SPLIT);*/
                    ElleGetFlynnRealAttribute(i,&cycle,CYCLE);
                    cycle = cycle +1;
                    ElleSetFlynnRealAttribute (i,cycle,CYCLE);
   	 	            ElleSetFlynnRealAttribute (i,0.0,AGE);
                    noevents++;
                    printf(" \n flynn %d\n",i);
                }
            }
        }
        printf("***********\nno angle_rx events = %d\n**********\n",noevents);  
        /*
         * update the display and write files if necessary
         */
        ElleUpdate();
    }
    //ElleAutoWriteFile(Count);
    
} 

void CheckRecrystEvent(int flynn, int *check)
{ 
  int max, j, grain1, grain2, checktest;
  
  checktest=0;
  ElleGetFlynnIntAttribute(flynn,&grain1,GRAIN);
  max = ElleMaxFlynns();
  for (j=0;j<max;j++)
  {
      if (ElleFlynnIsActive(j))
      { 
         ElleGetFlynnIntAttribute(j,&grain2,GRAIN);
         if (grain1==grain2 && flynn!=j)
         {
             checktest=1;
         }
      }
   }          

   *check=checktest;
}
