
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
    extern int InitExpand(void);

    ElleInit();

    ElleSetInitFunction(InitExpand);

    /*ElleSetDisplay(0);*/
    /*ElleSetSaveFrequency(10);*/

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    ElleSetSaveFileRoot("pblast");

    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run init and run functions
     */
    StartApp();

    return(0);
} 
