#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "errnum.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "polygon.h"
#include "unodes.h"
#include "stats.h"
#include "init.h"
#include "convert.h"
#include "update.h"
#include "mat.h"
#include "timefn.h"

int DoSomethingToFlynn(int flynn);
int InitThisProcess(), ProcessFunction();
int CheckRange(double *val,double min,double max,double tol);

#define XLINFO_EXT "xl"
#define CRSS_EXT "crss"
#define TBH_ONLY 0 //change to 1 if you want TBH without BASIL !! otherwise 0

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
    srand(currenttime());


    ElleReinit();

    ElleSetRunFunction(ProcessFunction);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes which may
         * not have been in the elle file
         * EULER_3, DISLOCDEN, MINERAL
         */
        
        if (!ElleFlynnAttributeActive(MINERAL))
            ElleAttributeNotInFile(infile, MINERAL); 
        if (!ElleFlynnAttributeActive(EULER_3))
            if (!ElleUnodesActive() || !ElleUnodeAttributeActive(EULER_3))
                ElleAttributeNotInFile(infile, EULER_3); 
        if (!ElleFlynnAttributeActive(DISLOCDEN)&& !ElleUnodesActive())
            ElleAttributeNotInFile(infile, DISLOCDEN); 
    }
    
}

#ifdef __cplusplus
extern "C" {
#endif
    int tbh_(double *,double *,double *,int *,char *,char *);
#ifdef __cplusplus
}
#endif
int ProcessFunction()
{
    int i, j, k, l,xlinfo_flag=0;
    int interval=0,st_interval=0,err=0,max,mintype;
	char *user_prefix;
    char fnxlinfo[32]="quartz.xl",fncrss[2][32]={"quartz.crss","fsp.crss"};
    double curra, currb, currc;
    double etensor[4],ermap[9],ework,disdensity;
    double rmap[3][3],rigid[3][3];
    double sym,rigid_rot;
	double lengthscale=ElleUnitLength();

    FILE *xl_out;

    if(xlinfo_flag==1)
    xl_out=fopen("xlinfo.out","a");

	user_prefix = ElleExtraFile();
	if (strlen(user_prefix)>0) {
		i = 32-strlen(XLINFO_EXT)-1-1;//max prefix for filename
		strncpy(fnxlinfo,user_prefix,i);
		strcat(fnxlinfo,".");
		strcat(fnxlinfo,XLINFO_EXT);
		i = 32-strlen(CRSS_EXT)-1-1;//max prefix for filename
		strncpy(fncrss[0],user_prefix,i);
		strcat(fncrss[0],".");
		strcat(fncrss[0],CRSS_EXT);
	}

    ElleCheckFiles();
   
    if (ElleDisplay()) EllePlotRegions(ElleCount());
    
    for (i=0;i<EllemaxStages();i++) {
      if (!ElleUnodesActive()) {
        max = ElleMaxFlynns();
        for (j=0;j<max;j++) {

        err=1;
        if (ElleFlynnIsActive(j) /*&&
                    ElleFlynnHasAttribute(j,EULER_3)&&
    	                   ElleFlynnHasAttribute(j,E_XX)*/)
        {
            
            ElleGetFlynnIntAttribute(j, &mintype, MINERAL);

            if(mintype==QUARTZ || mintype==FELDSPAR)
            {
            	ElleGetFlynnEuler3(j, &curra, &currb, &currc); //retrieve 3 Euler angles
            
            	curra=curra*PI/180.0;
                currb=currb*PI/180.0;
                currc=currc*PI/180.0;
            
                euler(rmap,curra,currb,currc); //calculate equivalent rotation tensor
         
                ermap[0]=  rmap[0][0];
            	ermap[1]=  rmap[0][1];
                ermap[2]=  rmap[0][2];
            	ermap[3]=  rmap[1][0];
            	ermap[4]=  rmap[1][1];
            	ermap[5]=  rmap[1][2];
            	ermap[6]=  rmap[2][0];
            	ermap[7]=  rmap[2][1];
            	ermap[8]=  rmap[2][2];
    	

            	if(TBH_ONLY)
            	{
            	    etensor[0]=1.0; // use fixed strain for TBH only calculations
            	    etensor[1]=0.05;
            	    etensor[2]=0.0;
            	    etensor[3]=1.0;
            	}
            	else
            	{
            	    ElleGetFlynnRealAttribute(j, &(etensor[0]), E_XX); //retrieve averaged strain for Flynn
                    ElleGetFlynnRealAttribute(j, &(etensor[1]), E_XY); 
    	            ElleGetFlynnRealAttribute(j, &(etensor[2]), E_YX); 
            	    ElleGetFlynnRealAttribute(j, &(etensor[3]), E_YY);

            	    etensor[3]=-etensor[0]; //forces no area change

            	    etensor[0]=etensor[0]+1;
                    etensor[3]=etensor[3]+1;
    	        }

                /*printf("from BASIL: %d\t%.3e\t%.3e\t%.3e\t%.3e\n",
    		j,etensor[0],etensor[1],etensor[2],etensor[3]);*/
    	
                if(xlinfo_flag==1 && i == 0)
            	{
            	    for(k=0;k<9;k++)
                		fprintf(xl_out,"%lf\t",ermap[k]);
    	            fprintf(xl_out,"\n");
        	    } 

            	if(mintype==QUARTZ )
            	    tbh_(etensor,ermap,&ework,&err,fncrss[0],fnxlinfo); //calculate quartz lattice rotations
                else
        	        tbh_(etensor,ermap,&ework,&err,fncrss[1],fnxlinfo); //calculate fsp lattice rotations

            	if(err!=0)
            	{
            	    printf("tbh output error\n");
            	    exit(1);
            	}		    

                rmap[0][0]=ermap[0];
            	rmap[0][1]=ermap[1];
                rmap[0][2]=ermap[2];
            	rmap[1][0]=ermap[3];
            	rmap[1][1]=ermap[4];
            	rmap[1][2]=ermap[5];
            	rmap[2][0]=ermap[6];
            	rmap[2][1]=ermap[7];
            	rmap[2][2]=ermap[8];

             	ElleGetFlynnRealAttribute(j, &disdensity, DISLOCDEN);
                //printf("%f\t%f\t\n",disdensity,ework);
                //disdensity += 0.8*ework; //related to strain rate?
                disdensity=(0.7*disdensity) + (0.4*ework);
                //disdensity=(0.8*disdensity) + (0.2*ework);
                ElleSetFlynnRealAttribute(j, disdensity, DISLOCDEN);

                uneuler(rmap,&curra,&currb,&currc);
        	//printf("%f\t%f\t%f\t%f\n",curra,currb,currc,ework);

               	ElleSetFlynnEuler3(j, curra, currb, currc); // store new Euler angles
            }
        }
        }
	  }
	  else {
		max = ElleMaxUnodes();
		if (!ElleUnodeAttributeActive(U_DISLOCDEN))
			 ElleInitUnodeAttribute(U_DISLOCDEN);
        for (j=0;j<max;j++) {

        err=1;
            	ElleGetUnodeAttribute(j, &curra, &currb, &currc, EULER_3); //retrieve 3 Euler angles
            
            	curra=curra*PI/180.0;
                currb=currb*PI/180.0;
                currc=currc*PI/180.0;
            
                euler(rmap,curra,currb,currc); //calculate equivalent rotation tensor
         
                ermap[0]=  rmap[0][0];
            	ermap[1]=  rmap[0][1];
                ermap[2]=  rmap[0][2];
            	ermap[3]=  rmap[1][0];
            	ermap[4]=  rmap[1][1];
            	ermap[5]=  rmap[1][2];
            	ermap[6]=  rmap[2][0];
            	ermap[7]=  rmap[2][1];
            	ermap[8]=  rmap[2][2];
    	

            	if(TBH_ONLY)
            	{
            	    etensor[0]=1.0; // use fixed strain for TBH only calculations
            	    etensor[1]=0.05;
            	    etensor[2]=0.0;
            	    etensor[3]=1.0;
            	}
            	else
            	{
            	    ElleGetUnodeAttribute(j, &(etensor[0]), E_XX); //retrieve averaged strain for Flynn
                    ElleGetUnodeAttribute(j, &(etensor[1]), E_XY); 
    	            ElleGetUnodeAttribute(j, &(etensor[2]), E_YX); 
            	    ElleGetUnodeAttribute(j, &(etensor[3]), E_YY);

            	    etensor[3]=-etensor[0]; //forces no area change

            	    etensor[0]=etensor[0]+1;
                    etensor[3]=etensor[3]+1;
    	        }

                /*printf("from BASIL: %d\t%.3e\t%.3e\t%.3e\t%.3e\n",
    		j,etensor[0],etensor[1],etensor[2],etensor[3]);*/
    	
                if(xlinfo_flag==1 && i == 0)
            	{
            	    for(k=0;k<9;k++)
                		fprintf(xl_out,"%lf\t",ermap[k]);
    	            fprintf(xl_out,"\n");
        	    } 

            	/*if(mintype==QUARTZ )*/
            	    tbh_(etensor,ermap,&ework,&err,fncrss[0],fnxlinfo); //calculate quartz lattice rotations

            	if(err!=0)
            	{
            	    printf("tbh output error\n");
            	    exit(1);
            	}		    

                rmap[0][0]=ermap[0];
            	rmap[0][1]=ermap[1];
                rmap[0][2]=ermap[2];
            	rmap[1][0]=ermap[3];
            	rmap[1][1]=ermap[4];
            	rmap[1][2]=ermap[5];
            	rmap[2][0]=ermap[6];
            	rmap[2][1]=ermap[7];
            	rmap[2][2]=ermap[8];

             	ElleGetUnodeAttribute(j, &disdensity, U_DISLOCDEN);
                //printf("%f\t%f\t\n",disdensity,ework);
                //disdensity += 0.8*ework; //related to strain rate?
                disdensity=(0.7*disdensity) + (0.4*ework);
                //disdensity=(0.8*disdensity) + (0.2*ework);
                ElleSetUnodeAttribute(j, disdensity, U_DISLOCDEN);

				/*!
				 * rmap[2][2],rmap[1][0] should be in range -1,1
				 */
			  if ( CheckRange(&rmap[2][2],-1.0,1.0,1e-5))
				OnError(" rmap[2][2] not in valid range",0);
			  if ( CheckRange(&rmap[1][0],-1.0,1.0,1e-5))
				OnError(" rmap[1][0] not in valid range",0);
                uneuler(rmap,&curra,&currb,&currc);
        	//printf("%f\t%f\t%f\t%f\n",curra,currb,currc,ework);

               	ElleSetUnodeAttribute(j, curra, currb, currc,EULER_3); // store new Euler angles
        }
	  }
		ElleUpdate();
    }
    
    if(xlinfo_flag==1)
    {
    fprintf(xl_out,"2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0\n"); 
    fclose(xl_out);
    }

} 

int CheckRange(double *val,double min,double max,double tol)
{
	int err=0;
	double diff=0.0;

	diff = *val-max;
	if (diff>0.0) {
		if (diff<tol) *val=max;
		else err = 1;
	}
	diff = min - *val;
	if (diff>0.0) {
		if (diff<tol) *val=min;
		else err = 1;
	}
	return(err);
}
