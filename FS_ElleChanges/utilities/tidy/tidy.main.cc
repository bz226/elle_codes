
/*
 *  main.cc
 */

#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "stats.h"
#include "setup.h"
#include "unodes.h"
#include "update.h"
#include "tidy.h"

main(int argc, char **argv)
{
    int err=0;
    extern int InitThisProcess(void);
    UserData userdata;

    /*
     * initialise
     */
    ElleInit();
 	
    ElleUserData(userdata);
	userdata[BnodeSpacing]=0; // Change default bnode spacing
 	userdata[Ranorient]=0;   // randomise EULER angles
 	userdata[MakeMica]=0;        // set flynns woith this F_ATTRIB_A to MICA else QUARTZ 
 	userdata[OrientMica]=0; 	// set MICA axes wrt to grain long axis
 	userdata[AddUnodes]=0;   // add unodes to file
 	userdata[UnodePattern]=HEX_GRID;   // type of unode distribution 
 	userdata[UnodeCells]=0;   // number of subcells for semi random patterns of unodes 
 	userdata[Rotation]=0;   // rotation of whole model in degrees  (clockwise is +ve) 
    
	ElleSetUserData(userdata);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);
    ElleSetOptNames("BnodeSpacing","RandomOrient","MakeMica","OrientMica","AddUnodes","UnodePattern","NumSubcells","Rotation","unused");

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the interval for writing to the stats file
    ES_SetstatsInterval(100);
     */
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("tidy");

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 
