
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
#include "plot_ugrid.h"

main(int argc, char **argv)
{

    int err=0;
    extern int InitThisProcess(void);
    UserData userdata;

	
    /*
     * initialise
     */
    ElleInit();

    ElleUserData(userdata);
    userdata[UGridSize]=256; // Change default grid size
    userdata[1]=5; // FS: By default the crit. misorientation angle is 5°
    userdata[2]=2; // FS: By default exclude phase with id=2 (usually air in my models) --> Set to 0 to not to use this option
    userdata[3]=1; // FS: default neigbour order = 1 --> 8 neighbours
    ElleSetUserData(userdata);
    ElleSetOptNames("UGridSize","HAGB","ExcludePhase","NbOrder","unused","unused","unused","unused","unused");
    
    if (argc>1) {
        if (err=ParseOptions(argc,argv))
            OnError("",err);
    }

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);

    /*
     * run your initialisation function and start the application
     */
    StartApp();

     return(0);
}
