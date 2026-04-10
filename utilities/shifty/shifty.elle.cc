#include <stdio.h>
#include <math.h>
#include <string.h>
#include <vector>
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "string_utils.h"
#include "runopts.h"
#include "file.h"
#include "init.h"
#include "unodes.h"
#include "interface.h"
#include "update.h"
#include "stats.h"
#include "shifty.h"
#include "erand.h"
#include "time.h"

using std::vector;

int InitShift();
int Do_Shift();

static Erand angran;

int InitShift()
{
    int err=0;
    int max;
    char *infile;
    time_t time_seed;
    
    time(&time_seed);
    angran.seed((unsigned long) time_seed);
    	
    ElleReinit();
    ElleSetRunFunction(Do_Shift);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
	return(err);
}


int Do_Shift()
{
    int i, j, k, n,max,err=0;
    Coords incr,xy,curr;
    CellData        unitcell;
    FILE *shift_state=0,*shift_hist=0;
	double dy=0.0,cum_offset,offset,tmp,last_cum_offset,dx=0.0;
	double last_ylength=0.0;
    double eps = 1e-5;
		
    ElleCellBBox(&unitcell);
    cum_offset = ElleCumSSOffset();

	shift_state=fopen("shift_state","r"); // temp file that shows previous offset state
	shift_hist=fopen("shift_hist","a");  // file that records all displacements
    
    if(!shift_state) // no tmp file, so first time encountered in experiment, so apply random shift
    {
    	shift_state=fopen("shift_state","w");
		do
		{
    	dy=angran.randouble() * unitcell.ylength;
		} while(dy==0.0);
    	fprintf(shift_state,"%12.8lf %12.8lf %12.8lf",
								dy,cum_offset,unitcell.ylength);
    	fclose(shift_state);
    }
    else // need to check if in shifted state or not
 	{
 		if ((n=fscanf(shift_state,"%lf %lf %lf",
					&dy,&last_cum_offset,&last_ylength)<3))
			OnError("shift_state",READ_ERR);
	 				
	 	if(dy >0.0) // shifted state, so unshift, and remove horizontal offset too
	 	{
    		fprintf(shift_hist,"%12.8lf ",dy);
/*
            dx = -(cum_offset-last_cum_offset)*(last_ylength/dy);
            dy = -(last_ylength-unitcell.ylength)*(last_ylength/dy);
*/
			dy *= (unitcell.ylength/last_ylength);
	 		dx=-dy/last_ylength*(cum_offset-last_cum_offset);
	 		dy=-dy;
	 		fclose(shift_state);
	 		shift_state=fopen("shift_state","w");
    		fprintf(shift_state,"%12.8lf %12.8lf %12.8lf",
								0.0,cum_offset,unitcell.ylength);
    		fclose(shift_state);
    		fprintf(shift_hist,"%12.8lf\n",-dy);
    		fclose(shift_hist);

    	}
    	else // non shifted state, so calc random shift
    	{
 	 		fclose(shift_state);
	 		shift_state=fopen("shift_state","w");
    		dy=angran.randouble() * unitcell.ylength;
    		fprintf(shift_state,"%12.8lf %12.8lf %12.8lf",
								dy,cum_offset,unitcell.ylength);
    		fclose(shift_state);
    	} 		
  	}
    	
      /*
       * move nodes
       */
	max = ElleMaxNodes();
	for (k=0;k<max;k++) {
		if (ElleNodeIsActive(k)) {
			ElleNodePosition(k,&xy);
			curr.x = xy.x +dx;
			curr.y = xy.y +dy;
			ElleCopyToPosition(k,&curr);
		}
	}
	  /*
	   * move unodes
	   */
		
	if (ElleUnodesActive()) {
		max = ElleMaxUnodes();
		for (k=0;k<max;k++) {
			ElleGetUnodePosition(k,&xy);
			incr.x = dx;
			incr.y = dy;
			if (ElleUnodeAttributeActive(U_STRAIN) &&
                  ElleUnodeAttributeActive(CURR_S_X)) {
				ElleGetUnodeAttribute(k,&curr.x,CURR_S_X);
				ElleGetUnodeAttribute(k,&curr.y,CURR_S_Y);
				ElleSetUnodeAttribute(k,curr.x,PREV_S_X);
				ElleSetUnodeAttribute(k,curr.y,PREV_S_Y);
				curr.x += incr.x;
				curr.y += incr.y;
				ElleSetUnodeAttribute(k,curr.x,CURR_S_X);
				ElleSetUnodeAttribute(k,curr.y,CURR_S_Y);
			}
			xy.x += incr.x;
			xy.y += incr.y;
			ElleSetUnodePosition(k,&xy);
		}
	}

	  /*
	   * move bounding box
	   */

	for (j=0;j<4;j++)
	{
		unitcell.cellBBox[j].x += dx;
		unitcell.cellBBox[j].y += dy;
	}	  
	  
	ElleSetCellBBox(&unitcell.cellBBox[BASELEFT],
					  &unitcell.cellBBox[BASERIGHT],
					  &unitcell.cellBBox[TOPRIGHT],
					  &unitcell.cellBBox[TOPLEFT]);
	ElleCellBBox(&unitcell);
	//LE1212 ElleSetCumSSOffset(cum_offset+unitcell.xoffset);
      /* if no pure shear component */
    if (fabs(unitcell.ylength-1.0)<eps) {
	  offset = modf(ElleCumSSOffset(),&tmp);
    }
    else {
      offset = ElleCumSSOffset();
      if (unitcell.xlength<offset) offset = offset-unitcell.xlength;
    }
	//LE1212 ElleSetSSOffset(offset);
	  
	ElleAutoWriteFile(1);
	
	return(err);
}
