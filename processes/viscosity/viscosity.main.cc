
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
#include "viscosity.h"

int main(int argc, char **argv)
{
    int err=0;
    extern int InitThisProcess(void);
    UserData userdata;
 
    /*
     * initialise
     */
    ElleInit();
	  
    ElleUserData(userdata);
    userdata[VMode]=0; // Change default claculation mode
    ElleSetUserData(userdata);
    ElleSetOptNames("VMode","unused","unused","unused","unused","unused","unused","unused","unused");
    
    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);


    /*
     * set the interval for writing to the stats file
    ES_SetstatsInterval(100);
     */

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("viscosity");

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
