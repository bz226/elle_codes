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
    UserData userdata;
    
    extern int InitThisProcess(void); 
    
    /*
     * initialise
     */
    ElleInit();
    ElleUserData(userdata); 
    
    /*
     * FS (!) Set default values for userdata
     */
    userdata[0] = 0; // High angle GB angle, default: 15
    userdata[1] = 0; // Set ID of phase which should be excluded from recovery (Default 0 means: No phase)
    userdata[2] = 0; // Switch off random picking of flynns, unodes etc. by setting this to 1
    userdata[3] = 0; // Set initial rotation mobility, 0 is default but will lead to NO rotation
    userdata[4] = 0; // Set initial start step, 0 is default: usually start from 1st step
    
    /* Input the rotation matrix for material being
     * examined --> need to setup this userdata with Lynn
     * --> also the temperature information
     */     
    ElleSetUserData(userdata);
    
    ElleSetOptNames("HAGB-angle","ExcludePhase","NoRandomisation","RotationMobility","StartStep","unused","unused","unused","unused");
   
   /* Rotation will be the matrix with material slip
    * system info --> temperature will determine how large
    * the neighbourhood examined is
    */   
    if (err=ParseOptions(argc,argv)) 
    
    /* Checks to see if you have done any command line 
     * options eg. ./elle.verity -s etc.
     */
    OnError("",err);

    /*
     * set the function to the one in your process file
     */ 
    ElleSetInitFunction(InitThisProcess);

    /*
     * Set the interval for writing to the stats file: Eill spit out stats 
     * data - at the mo every 100 steps
     */
    ES_SetstatsInterval(100); 
    
    /*
     * set the base for naming statistics and elle files
     */   
    //ElleSetSaveFileRoot("SGG");
    //ElleSetSaveFrequency(1);
   
    //char cFileroot[] = "FS_recovery";
    //ElleSetSaveFileRoot(cFileroot);

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



//int main(int argc, char **argv)
//{
    //int err=0;
    //UserData userdata;
    ///*
     //* Initialise and set option names
     //*/
    //ElleInit();
    //ElleSetOptNames("unused","unused","unused","unused","unused","unused","unused","unused","unused");
    ///* 
     //* Give default values for userdata:
     //*/
    //ElleUserData(userdata);
    //userdata[UInput1]=0; // Default: 0
    //userdata[UInput2]=0; // Default: 0
    //userdata[UInput3]=0; // Default: 0
    //userdata[UInput4]=0; // Default: 0
    //userdata[UInput5]=0; // Default: 0
    //userdata[UInput6]=0; // Default: 0
    //userdata[UInput7]=0; // Default: 0
    //userdata[UInput8]=0; // Default: 0
    //userdata[UInput9]=0; // Default: 0
    //ElleSetUserData(userdata);
    
    //if (err=ParseOptions(argc,argv)) OnError("",err);
    ///*
     //* Eventually set the interval for writing file, stages etc. if e.g.
	 //* this utility should only run once etc.
     //*/
    ////ElleSetSaveFrequency(1);
    ////ElleSetStages(1);
    ///*
     //* Set the function to the one in your process file
     //*/
    //ElleSetInitFunction(InitThisProcess);	
	///*
     //* Set the default output name for elle files
     //*/
    //char cFileroot[] = "FS_raw";
    //ElleSetSaveFileRoot(cFileroot);
    ///*
     //* Set up the X window
     //*/
    //if (ElleDisplay()) SetupApp(argc,argv);
    ///*
     //* Run your initialisation function and start the application
     //*/
    //StartApp();

    //CleanUp();

    //return(0);
//} 
