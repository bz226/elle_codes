#include "FS_topocheck.h"

int main(int argc, char **argv)
{
    int err=0;
    //UserData userdata;
    
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("unused","unused","unused","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    //UserData userdata;    
    //ElleUserData(userdata);
    //userdata[0]=0; // Default: 0
    //ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    /*
     * Set the interval for writing file and stages: Topology checks only need 
     * to be performed once 
     */
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);	
	/*
     * Set the default output name for elle files
     */
    char cFileroot[] = "FS_topocheck";
    ElleSetSaveFileRoot(cFileroot);
    /*
     * Set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);
    /*
     * Run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 
