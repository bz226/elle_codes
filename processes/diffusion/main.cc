
/*
 *  main.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "parseopts.h"
#include "error.h"
#include "runopts.h"
#include "diffusion.h"
#include "init.h"
#include "setup.h"

main(int argc, char **argv)
{
    int err=0;
    extern int InitDiffusion(),RunDiffusion(),CleanDiffusion();

    ElleInit();

    ElleSetInitFunction(InitDiffusion);
    ElleSetRunFunction(RunDiffusion);
    ElleSetExitFunction(CleanDiffusion);

    if (argc>1) {
        if (err=ParseOptions(argc,argv))
            OnError("",err);
    }
    else ElleSetStages(100);

    if (strlen(ElleFile())==0) ElleSetFile("testsw1.poly");

    if (ElleDisplay()) SetupApp(argc,argv);

    StartApp();

    return(0);
} 
