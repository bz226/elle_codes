
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
#include "metamorphism.h"

main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    extern int InitMet(void);
 
    /*
     * initialise
     */
    ElleInit();
		
    ElleUserData(userdata);
    userdata[Mobility] = 1e-11;    // mobility
    userdata[Magic] = 0.0001;        // magic number (area fudge factor)
	userdata[IncT] = 0.1;		   // incremental Temperature change
	userdata[IncP] = 0.0;		   // incremental Pressure change
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv))
        OnError("",err);

    ElleSetOptNames("Mobility","Area Factor","Temp Incr","Pres Incr","unused","unused","unused","unused","unused");
    /*
     * set the function to the one in your process file
     */
    ElleUserData(userdata);
	printf("mob\t\tmagic\t\tinc T\t\tinc P\n%le\t%lf\t%lf\t%lf\n",
	userdata[Mobility],userdata[Magic],userdata[IncT],userdata[IncP]);

    ElleSetInitFunction(InitMet);

    /*
     * set the interval for writing to the stats file
    ES_SetstatsInterval(100);
     */

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("metamorphism");

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
