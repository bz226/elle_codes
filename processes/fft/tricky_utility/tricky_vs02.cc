#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "parseopts.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "setup.h"
#include "triattrib.h"
#include "unodes.h"
#include "polygon.h"
#include "check.h"

//using std::ios;
//using std::cout;
//using std::cerr;
//using std::endl;
//using std::ifstream;
//using std::string;
using std::vector;
//using std::cin;

// funcions definides
int Input_tex(), Init_input();
void check_dist();

main(int argc, char **argv)
{
    int err=0;
    UserData udata;

    /*
     * initialise
     */
    ElleInit();

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(Init_input);

	ElleUserData(udata);
	ElleSetUserData(udata);
	
    if (err=ParseOptions(argc,argv))
        OnError("",err);
    ElleSetSaveFrequency(1);

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("tricky");

    /*
     * run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 


/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int Init_input()   //InitMoveBnodes()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(Input_tex);

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

    }
}

/*!
 */


int Input_tex()
{
    int err=0, i;

    /*
     * set the x,y  to the values from the fft step
     */

    /*
     * set the euler angles to the values from the fft step
     */

    check_dist(); //file is tex.out
    if (err) OnError("error",err);

    /*
     * write the updated Elle file
     */
     if (err=ElleWriteData("tricky.elle"))
         OnError("",err);
}

void check_dist()
{
	int i,j,k; 
	int *ids,num, max;
	Coords xy1, xy2, newxy1, newxy2;
	double switchd, dist, increm, dist1, dist2;


	max = ElleMaxNodes();
	switchd = ElleSwitchdistance()/5.0;
	
	increm= switchd;
	
	for (i=0;i<ElleMaxFlynns();i++) 
   	{
      if (ElleFlynnIsActive(i)) 
	  {  
	    ElleFlynnNodes(i, &ids, &num);
		  
        for ( j = 0; j<num-1; j++ ){
			
			for (k=j+1; k<num; k++){
				// printf("j k %i %i\n", ids[j],ids[k]);
            if ( ElleNodeIsActive( ids[j] ) && ElleNodeIsActive( ids[k] )) {
				
                ElleNodePosition( ids[j], & xy1 );
				ElleNodePosition( ids[k], & xy2 );

                dist = pointSeparation(&xy1,&xy2);
				// printf("dist %lf\n", dist);
				
				if (dist< switchd) {
				newxy1.x=newxy1.y=0.0;					
                    printf("node1 node2 %i %i\n", ids[j], ids[k]);
					printf(" %lf %lf\n", xy1.x, xy1.y );
					printf(" %lf %lf\n", xy2.x, xy2.y );
					// + 
			xy1.x += increm;
			xy1.y += increm;
			dist1 = pointSeparation(&xy1,&xy2);		
					
					// - 
			xy1.x -= 2*increm;
			xy1.y -= 2*increm;
			dist2 = pointSeparation(&xy1,&xy2);					

					
			if ( dist1 < dist2) {
					newxy1.x -=increm ;
					newxy1.y -=increm;
			} 
			else {				
					newxy1.x +=increm ;
					newxy1.y +=increm;					
			}				
					printf(" %lf %lf\n", newxy1.x,newxy1.y );
					ElleUpdatePosition( ids[j], & newxy1 );
					
	                    if (ElleNodeIsDouble( ids[j] ))
                            ElleCheckDoubleJ( ids[j] );
						// else if (ElleNodeIsTriple( ids[j] ))
                          //  ElleCheckTripleJ( ids[j] );

					}
				}
			}
		}
	free(ids); 		
	}
	}

	// add un check of bnodes and position 
}
