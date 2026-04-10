#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "parseopts.h"
#include "init.h"
#include "runopts.h"
#include "stats.h"
#include "setup.h"

int main(int argc, char **argv)
{
    int err=0;
    
    // Declare initialisation function
    extern int InitGrowth(void);
    
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("diagonal","afactor","logscreen","Start Timestep","ExcludePhase","unused","unused","unused","unused");
    /* 
     * Give default values for userdata (all = 0 by default):
     */
    ElleUserData(userdata);
    for (int i=0 ; i<9 ; i++)
    {
        userdata[i]=0;
    }
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) 
        OnError("",err);
    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitGrowth);	
	/*
     * Set the default output name for elle files
     */
    char cFileroot[] = "gbm_pp_fft";
    ElleSetSaveFileRoot(cFileroot);
    /*
     * Set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);
    /*
     * Run initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 
