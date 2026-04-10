#include <stdio.h> 
#include <math.h> 
#include <string.h> 
#include <list> 
#include <set> 
#include <vector> 
#include "convert.h" 
#include "nodes.h" 
#include "file.h" 
#include "display.h" 
#include "check.h" 
#include "error.h" 
#include "runopts.h" 
#include "init.h" 
#include "general.h" 
#include "interface.h" 
#include "crossings.h"
#include "stats.h" 
#include "flynnarray.h" 
#include "polygon.h" 
#include "mineraldb.h"
#include "update.h"
#include "merge.h"

static int Count; /* how many iterations have been completed */
                  /* in the ProcessFunction (relate to time) */

int InitThisProcess(), ProcessFunction();


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
        if (!ElleFlynnAttributeActive(EULER_3)) 
            ElleAttributeNotInFile(infile,EULER_3);
    }

    Count = 0;
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
    int i, j, k, max, mintype,type, parent,err=0;
    int interval=0,st_interval=0, merge_rgn, merge,min;
    double min_area_threshold = 0.0001;
    double min_areaF_threshold = 0.0002;
    double  old_density[3] ;
    double area, alpha, beta, gamma; 
    UserData userdata;
    int removeflynn=NO_VAL, keepflynn=NO_VAL;

    ElleUserData(userdata);
    removeflynn=(int)userdata[RemoveFlynn];
    keepflynn=(int)userdata[KeepFlynn];

    ElleCheckFiles();
    if (ElleDisplay()) EllePlotRegions(ElleCount());
    
    if (removeflynn!=NO_VAL && keepflynn!=NO_VAL) {
        parent = getParent(keepflynn);
        if (getParent(removeflynn)!=parent)
            OnError("Parent flynns differ",0);
        if ((err=ElleMergeFlynns(keepflynn,removeflynn))!=0) 
            OnError("Merge error",err);
        /*
         * update
         */
		ElleUpdate();
    }
    else {
    for (i=0;i<EllemaxStages();i++) 
    {
        int still_merging = 1;
        while (still_merging) {
        still_merging = 0;
        max = ElleMaxFlynns();
        for (j=0;j<max;j++)
        {
          
            if (ElleFlynnIsActive(j) && !isParent(j) && hasParent(j))
            {  
                  
              parent = getParent(j);
              ElleGetFlynnMineral(j,&min);
              ElleGetFlynnEuler3(j, &alpha, &beta, &gamma);
			  std::list<int> rgn;
              ElleFlynnNbRegions(j,rgn); 
			  std::vector<int> rgn_split(rgn.size(),0);
			  std::vector<double> rgnalpha(rgn.size(),0);
			  std::vector<double> rgnbeta(rgn.size(),0);
			  std::vector<double> rgngamma(rgn.size(),0);
			  std::vector<double> rgn_area(rgn.size(),0);
              if (min==QUARTZ){                
                if ( (area=fabs(ElleRegionArea(j)))<min_area_threshold){
                    //printf("\n flynn_area_org = %f\n",area);
                    merge=1;
                    
                    
					std::list<int>::iterator its=rgn.begin();
                    for (k=0;its!=rgn.end() && merge;k++,its++) { 

                       ElleGetFlynnEuler3(*its, &rgnalpha[k], &rgnbeta[k],
                                          &rgngamma[k]); 
                       if (alpha==rgnalpha[k]&& beta==rgnbeta[k] &&
                           gamma==rgngamma[k]){           
                             if (getParent(*its)==parent &&
                                 ElleMergeFlynns(*its,j)==0) { 
                                    merge=0;                        
                                    still_merging = 1; 
                             }                        
                        }                     
                    }
                  } 
              }
              if (min==FELDSPAR){ 
                if ( (area=fabs(ElleRegionArea(j)))<min_areaF_threshold){
                    //printf("\n flynn_area_org = %f\n",area);
                   merge=1;
                    
                    
				   std::list<int>::iterator its=rgn.begin();
                   for (k=0;its!=rgn.end() && merge;k++,its++) { 

                       ElleGetFlynnEuler3(*its, &rgnalpha[k], &rgnbeta[k],
                                          &rgngamma[k]); 
                       if (alpha==rgnalpha[k]&& beta==rgnbeta[k] &&
                           gamma==rgngamma[k]){           
                           if (getParent(*its)==parent &&
                               ElleMergeFlynns(*its,j)==0) { 
                                  merge=0;                        
                                  still_merging = 1; 
                           }                        
                       }                     
                   }
                 } 
               }
          }
        }
        }
        
        /*
         * update
         */
		ElleUpdate();
    }
    ElleAutoWriteFile(Count);
    }
}
