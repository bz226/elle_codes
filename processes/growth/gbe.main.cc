
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
    extern int InitGBE_Growth(void);

    ElleInit();

    ElleSetInitFunction(InitGBE_Growth);

    if (err=ParseOptions(argc,argv))
        OnError("",err);
    //ES_SetstatsInterval(100);

    ElleSetSaveFileRoot("gbegrowth");

    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run init and run functions
     */
    StartApp();

    return(0);
} 
