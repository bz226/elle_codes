#include "FS_statistics.h"
using namespace std;

main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * initialise
     */
    ElleInit();
    ElleSetOptNames("uMode","uOption","unused","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    userdata[uMode]=0; // Default: 0, used for flynn area statistics
    userdata[uOption]=0; // Default: 0, change for further options (see readme)
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);	
	/*
     * set the nameing for outputs and default stages (1)
     */
    char fileroot[] = "FS_statistics";
    ElleSetSaveFileRoot(fileroot);
    ElleSetStages(1);
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
