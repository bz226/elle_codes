
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
#include "melt.h"

main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    extern int InitGBM(void);
 
    /*
     * initialise
     */
    ElleInit();
    ElleUserData(userdata);
    userdata[Mobility] = 1e-11;    // mobility
    userdata[Magic] = 0.01;        // magic number (area fudge factor)
    userdata[EquilMeltFrac] = 0.1; // equilibrium melt fraction (not currently used)
	userdata[XlXlEnergy] = 1;      // xl-xl interface energy
	userdata[LiqXlEnergy] = 1;     // liq-xl interface energy
	userdata[LiqLiqEnergy] = 0;    // liq-liq interface energy
	userdata[IncMeltfraction] = 0; // incremental melt fraction change
	
    ElleSetUserData(userdata);
    ElleSetOptNames("Mobility","Area Factor","EquiMeltFrac","xl-xl","liq-xl","liq-liq","IncMeltFrac","unused","unused");
    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitGBM);

    /*
     * set the interval for writing to the stats file
    ES_SetstatsInterval(100);
     */

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("melt");

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run your initialisation function and start the application
     */
    StartApp();
    
     return(0);
} 
