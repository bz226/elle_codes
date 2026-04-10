
/*
 *  main.cc
 */

#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "file.h"
#include "setup.h"

main(int argc, char **argv)
{
    int err=0;
    extern int InitSetUnodes(void);
    UserData userdata;

    /*
     * initialise
     */
    ElleInit();

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitSetUnodes);

    ElleUserData(userdata);
    userdata[1]=1; // Change default calculation mode
    ElleSetUserData(userdata);
                                                                                
    ElleSetOptNames("Max Angle","Slope","unused","unused","unused","unused","unused","unused","unused");

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("sub_gg");

    /*
     * run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 
