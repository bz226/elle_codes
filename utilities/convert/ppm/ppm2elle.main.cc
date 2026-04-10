
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

main(int argc, char **argv)
{
    int err=0;
    UserData udata;
    extern int InitThisProcess(void);

    /*
     * initialise
     */
    ElleInit();

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);

    ElleSetOptNames("decimate","Save unodes",
                    "unused","unused",
                    "unused",
                    "unused","unused","unused","unused");

    ElleUserData(udata);
    udata[0] = 1;
    udata[1] = 0;
    ElleSetUserData(udata);

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the interval for writing to the stats file
    ES_SetstatsInterval(100);
     */

    /*
     * set the base for naming statistics and elle files
     * image file name is used in this process
     */
    ElleSetSaveFileRoot("ppm2elle");

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
