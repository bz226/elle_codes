
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
    extern int InitGrowth(void);

    ElleInit();

    ElleSetInitFunction(InitGrowth);

    if (err=ParseOptions(argc,argv))
        OnError("",err);
    //ES_SetstatsInterval(50);

    ElleSetSaveFileRoot("growth");

    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run init and run functions
     */
    StartApp();

    return(0);
} 
