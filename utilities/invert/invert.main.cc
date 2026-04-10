
/*
 *  main.cc
 */

#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "setup.h"

main(int argc, char **argv)
{
    int err=0;
    extern int InitThisProcess(void);
 
    /*
     * initialise
     */
    ElleInit();
    
    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("invert");

    ElleSetDisplay(0);
    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);
    /*
     * run your initialisation function
     */
    StartApp();
    
     return(0);
} 
