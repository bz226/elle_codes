
/*
 *  main.cc
 */

#include <stdio.h>
#include <stdlib.h>

#include "attrib.h"
#include "errnum.h"
#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "file.h"
#include "setup.h"

extern const int KAPPA_INDEX = 0;
extern const int NODE_ID = 1;
extern const int ATTRIB_VAL = 2;
extern const int DIFFUSION_ONLY = 3;
extern const int TEMP_INCR = 4;
extern const double KAPPA_DFLT = 0.3090528;

main(int argc, char **argv)
{
    int err=0;
    UserData udata;
    extern int InitSetUnodes(void);

    /*
     * initialise
     */
    ElleInit();

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitSetUnodes);

    ElleUserData(udata);
    udata[KAPPA_INDEX] = KAPPA_DFLT;
    udata[NODE_ID] = NO_NB;
    udata[DIFFUSION_ONLY] = 0;
    udata[TEMP_INCR] = 0;
    ElleSetUserData(udata);
    ElleSetOptNames("Kappa","Bnode Id","Attribute","Diff Only","Temp Incr","unused","unused","unused","unused");
    
    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("exchange");

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 
