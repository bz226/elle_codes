#include <stdio.h>
#include <math.h>
#include "attrib.h"
#include "nodes.h"
#include "check.h"
#include "update.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "tidy.h"
#include "mat.h"
#include "regions.h"
#include "general.h"
#include "unodes.h"
#include "convert.h"

int DoSomethingToFlynn(int flynn);
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
         * check for any necessary attributes which may
         * not have been in the elle file
         */
        //if (!ElleFlynnAttributeActive(ValidAtt))
            //ElleAttributeNotInFile(infile,ValidAtt);
    }
}
/*!
 \brief	Utility for transforming existing elle files
 \par	Description:
		Code for tidying up elle files, the following activities may be performed:

		1) Change bnode spacing
		2) Randomise EULER_3 angles for flynns with MINERAL attribute set to QUARTZ 
		3) Convert flynns with specific F_ATTRIB_A to be mineral MICA
		4) Set MICA grains to have c0axis normal to long axis of grain
		5) Strip unodes from file
		6) Add unodes to file in one of 4 patterns (hex, square, random, or random within regular cells)

		The format of the command line function is:

		tidy -i filename.elle -u A B C D E F G -s 1 -f 1 -n

		where 

		A = bnode spacing [default = 0 leaves them unchanged]

		B = 1 for randomising orientations [default = 0 leaves them unchanged]

		C = F_ATTRIB_A for flynns to be set to MICA  [default = 0 leaves them unchanged]

		D = 1 for calculating MICA crystal axis orientations [default = 0 leaves them unchanged]

		E = -1 for stripping nodes from a file [default = 0 leaves them unchanged]
		E= no of unodes per row for adding unodes (strips previous unodes first) [default = 0 leaves them unchanged]

		F = code for pattern of unodes to add if E>1 [default = 0 HEX, 1 = SQUARE, 2 = RANDOM, 3 = SEMI RANDOM, where E x E cells are defined, and then G unodes are created per cell]

		G = number of unodes per cell if F = 3 [default = 0]
		
		H = rotation of bnodes (not unodes!)[default = 0] only works for multiples of 90°!!!
		
		examples: 
		
		to add unodes to a file with a square pattern, 50 unodes per row, type
		
			tidy -i filename.elle -u 0 0 0 0 50 1
		
		to change bnode spacing to 0.001
		
			tidy -i filename.elle -u 0.001		
 */

int ProcessFunction()
{
    int i,err;
	UserData userdata;
	double bnodespacing;
	int ranorient;
	double makemica;
	int orientmica;
	int addunodes;
	int unodepattern;
	int unodecells;
	double rotation;
	
	ElleUserData(userdata);
   
   
  bnodespacing=userdata[BnodeSpacing]; // Change default bnode spacing
 	ranorient=(int)userdata[Ranorient];   // randomise EULER angles
 	makemica=userdata[MakeMica];        // set flynns woith this F_ATTRIB_A to MICA else QUARTZ 
 	orientmica=(int)userdata[OrientMica]; 	// set MICA axes wrt to grain long axis
 	addunodes=(int)userdata[AddUnodes];   // add unodes to file
 	unodepattern=(int)userdata[UnodePattern];   // type of unode distribution 
 	unodecells=(int)userdata[UnodeCells];   // number of subcells for semi random patterns of unodes 
 	rotation=(int)userdata[Rotation];   // rotation of whole model in degrees  (clockwise is +ve)

    ElleCheckFiles();

    for (i=0;i<EllemaxStages();i++) // loop for number of stages
	{
        if(bnodespacing>0)
		{
			ReSpaceBnodes(bnodespacing);
		}
		else if(fabs(bnodespacing) < 1e-10 )
			ReSpaceBnodes(0.0);

 		if(ranorient!=0)
		{
   		 	if (!ElleFlynnAttributeActive(EULER_3))
			{
				ElleInitFlynnAttribute(EULER_3);
				ElleSetDefaultFlynnEuler3(0.0,0.0,0.0);
			}
			RanorientQuartz();
		}
		
		if(makemica>0)
		{
    		//ElleInitFlynnAttribute(MINERAL);
			//ElleSetDefaultFlynnMineral(QUARTZ);
			MicaQuartzFlynn(makemica);
		}
		
        if(orientmica!=0)
		{
   		 	//ElleInitFlynnAttribute(EULER_3);
			ElleSetDefaultFlynnEuler3(0.0,0.0,0.0);
			OrientMicaAxis();
		}
        
		if(addunodes>0)
		{
			UnodesClean(); 
			if(unodepattern==HEX_GRID || unodepattern==SQ_GRID)
				ElleInitUnodes(addunodes, unodepattern);
			else if(unodepattern==RAN_GRID || (unodepattern==SEMI_RAN_GRID && unodecells==0) )
				ElleInitRanUnodes(1, addunodes , RAN_GRID);
			else if(unodepattern==SEMI_RAN_GRID && unodecells>0)
				ElleInitRanUnodes(unodecells, addunodes, unodepattern);
		}
		else if(addunodes<0)
		{
				UnodesClean(); 
		}
		
		if(fabs(rotation)>0)
		{
			Rotate();
		}
				
		ElleUpdate();  
    }
    return(err);
} 

int ReSpaceBnodes(double spacing)
{
	int i,j,l;
	int max=0,newmax=0;
    int numtrp=0, numdbl=0, newnumdbl=0;
	
	if(spacing > 0.0)
		ElleSetSwitchdistance(spacing);
	
	/*for(i=5;i>0;i--)*/
    ElleNumberOfNodes(&newnumdbl,&numtrp);
    while (newnumdbl!=numdbl)
	{
		numdbl = newnumdbl;
        max = ElleMaxNodes();
        	for (j=0,l=0;j<max;j++) 
			{
        	    if (ElleNodeIsActive(j))
				/*{*/
	    			/*l++;*/
					/*if(l%i == 0)*/
                		if (ElleNodeIsDouble(j)) 
							ElleCheckDoubleJ(j);
						/*else */
							/*ElleCheckTripleJ(j);*/

        	    /*}*/
        	}
        ElleNumberOfNodes(&newnumdbl,&numtrp);
	}
    ElleAddDoubles();

}
             
int RanorientQuartz()
{
    int max,flynn;
    double curra, currb, currc,dflt1,dflt2,dflt3;
    double rmap[3][3];
    double eps = 1e-5;
    int mintype;
	
	max = ElleMaxFlynns();		// index of maximum flynn used in model
    
	for (flynn=0;flynn<max;flynn++) 		// loop though all flynns
	{
        if (ElleFlynnIsActive(flynn)) // process flynn if it is active
		{
           // std::cout << "blah" << std::endl;
			ElleGetFlynnIntAttribute(flynn, &mintype, MINERAL);
            //std::cout << "blah2" << std::endl;

            if(mintype==QUARTZ || mintype==ICE) // "|| mintype==ICE" added 12th August 2014 FSteinbach
            {
                ElleDefaultFlynnEuler3(&dflt1,&dflt2,&dflt3);
                ElleGetFlynnEuler3(flynn,&curra,&currb,&currc);
                /*printf("curr %.8e %.8e %.8e\ndflt %.8e %.8e %.8e\n\n",
                    curra,currb,currc,
                    dflt1,dflt2,dflt3);*/
            
                /*if (curra==dflt1 && currb==dflt2 && currc==dflt3 ) */
                if (fabs(curra-dflt1) < eps &&
                    fabs(currb-dflt2) < eps &&
                    fabs(currc-dflt3) < eps ) 
                {
                    orientZXZ(rmap);
                    uneulerZXZ(rmap,&curra,&currb,&currc);

                    ElleSetFlynnEuler3(flynn, curra,currb,currc);
                }

            }
       }
    }

}
int MicaQuartzFlynn(double mid)
{
    int max,flynn;
	double id;
    
	max = ElleMaxFlynns();		// index of maximum flynn used in model
    
	for (flynn=0;flynn<max;flynn++) 		// loop though all flynns
	{
        if (ElleFlynnIsActive(flynn)) // process flynn if it is active
		{
			ElleGetFlynnRealAttribute(flynn, &id, F_ATTRIB_A); // get attribute for this flynn
    		if(id == mid)
				ElleSetFlynnIntAttribute(flynn, MICA,MINERAL); // set attribute for this flynn
			else
				ElleSetFlynnIntAttribute(flynn, QUARTZ,MINERAL); // set attribute for this flynn
        }
    }

}
int OrientMicaAxis()
{
    int max,flynn;
    int mintype;
	double angle;
	
	max = ElleMaxFlynns();		// index of maximum flynn used in model
    
	for (flynn=0;flynn<max;flynn++) 		// loop though all flynns
	{
        if (ElleFlynnIsActive(flynn)) // process flynn if it is active
		{
            ElleGetFlynnMineral(flynn, &mintype);

            if(mintype==MICA)
            {
	    	 	angle=GetLongAxis(flynn);
                 //should be in the range -180<=angle<180
                if (angle>180.0) angle=M_PI*2-angle;
                if (angle<=-180) angle=M_PI*2+angle;    
                ElleSetFlynnEuler3(flynn, 90.0, angle, 0.0);
			}
        }
    }

}

double GetLongAxis(int flynn)
{
    int max, i, j, k, m, nbnodes[3];
    double len, totlen, dx, dy;
    double rosedata[181], pabs, p,alfa, cosalfa;
    double minL, maxL, minA, maxA;
    Coords xy, xynb;
    int *ids;
    
    totlen = 0.0;
    for (i=0; i<181; i++) rosedata[i] = 0.0;
    
    ElleFlynnNodes(flynn,  &ids, &max);
    
    for (m=0; m<max; m++) {
    	i=ids[m];
        if (ElleNodeIsActive(i)) {
            ElleNeighbourNodes(i,nbnodes);
            for (j=0;j<3;j++) {
                if (nbnodes[j]!=NO_NB && nbnodes[j]>i) {
                    ElleNodePosition(i,&xy);
                    ElleNodePlotXY(nbnodes[j],&xynb,&xy);
                    dx = (xy.x - xynb.x);
                    dy = (xy.y - xynb.y);
                    len = sqrt(dx*dx + dy*dy);
                    if (len > 0.0) {
                        totlen += len;
                        cosalfa = dy/len;
                        alfa = acos(cosalfa);
                        if (dx < 0.0) alfa = M_PI-alfa;
                        for (k=0;k<180;k++) {
                            pabs = cos(alfa - (double)k/180.0*M_PI);
                            p = len * fabs(pabs);
                            rosedata[k] += p;
                        }
                    }
                    else
                        printf("PanozzoAnalysis-len 0",0);
                }
            }
        }
    }
    minL = maxL = rosedata[0];
    minA = maxA = 0;
    for (i=0; i<180; i++) {
        if (rosedata[i] < minL) {
            minL = rosedata[i];
            minA = (double)i;
        }
        if (rosedata[i] > maxL) {
            maxL = rosedata[i];
            maxA = (double)i;
        }
    }
    
    return (maxA);
}

int Rotate()
{
    int i, j, k;
    int max;
	int nodeid;
	int count, grain_count, index;
    double angle = 0;
    Coords xy, newxy, centre;
	CellData unitcell;
    UserData udata;
    
/*
 * ASSUMES 1x1 UNIT CELL
 */
    ElleCheckFiles();
	ElleUserData(udata);

    angle = (double)udata[Rotation]*DTOR;
	centre.x = centre.y = 0.5;

	max = ElleMaxNodes();
	for (k=0;k<max;k++) {
        if (ElleNodeIsActive(k)) {
			ElleNodePosition(k,&xy);
		    rotate_coords_ac(xy.x,xy.y,
                  centre.x,centre.y,
                  &newxy.x, &newxy.y, angle);
			ElleCopyToPosition(k,&newxy);
		}
    }
} 
