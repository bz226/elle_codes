
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
    extern int InitThisProcess(void);

    /*
     * initialise
     */
    ElleInit();

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the interval for writing data file
	 * this utility should only run once
     */
    ElleSetSaveFrequency(1);
    /*
     * set the interval for writing to the stats file
    ES_SetstatsInterval(100);
     */

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("repos");

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
