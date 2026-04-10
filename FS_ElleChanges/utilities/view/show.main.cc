
/*
 *  main.c
 */

#include <stdio.h>
#include <stdlib.h>

#include "runopts.h"
#include "init.h"
#include "error.h"
#include "parseopts.h"
#include "setup.h"
#include "plotaxes.h"

main(int argc, char **argv)
{
    int err=0;
    UserData udata;
    ElleRunFunc init;
    extern int InitShow(void);

    ElleInit();

    ElleSetInitFunction(InitShow);

    // options only apply to stereonet output
    ElleUserData(udata);
    udata[REVERSE_ORDER]=0; // Output row order, 1 for ebsd top down
    udata[SAMPLE_STEP]=1000; // sampling frequency (for unodes)
    udata[CAXIS_OUT]=1; // generate c-Axis data file (for unodes)
    ElleSetUserData(udata);

    ElleSetOptNames("RowOrder","Step","CAxisData","unused",
                    "unused","unused","unused","unused","unused");

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    if (ElleDisplay()) SetupApp(argc,argv);

    StartApp();
} 
