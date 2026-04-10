
/*
 *  main.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runopts.h"
#include "init.h"
#include "error.h"
#include "file.h"
#include "parseopts.h"
#include "setup.h"
#include "display.h"

main(int argc, char **argv)
{
    int err=0;
    char *infile;

    ElleInit();

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    SetupApp(argc,argv);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError("",err);
        /*
         * initialize any necessary flynn attributes which
         * are not in the input file
         */
    }

    Run_App(0);
} 

int SetStage(int stage)
{

}

void ClearBg()
{
}

