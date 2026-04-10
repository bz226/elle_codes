
/*
 *  main.cc
 */

#include <stdio.h>
#include <stdlib.h>

#include "runopts.h"
#include "init.h"
#include "error.h"
#include "setup.h"
#include "parseopts.h"
#include "plotaxes.h"

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

    ElleUserData(udata);
    udata[REVERSE_ORDER]=0; // Output row order, 1 for ebsd top down
    udata[SAMPLE_STEP]=1000; // sampling frequency (for unodes)
    udata[CAXIS_OUT]=1; // generate caxis output (for unodes)
    ElleSetUserData(udata);

    ElleSetOptNames("RowOrder","Step","CAxisData","unused",
					"unused","unused","unused","unused","unused");

    if (argc>1) {
        if (err=ParseOptions(argc,argv))
            OnError("",err);
    }

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
