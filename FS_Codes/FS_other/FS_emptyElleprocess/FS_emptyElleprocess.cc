#include "FS_emptyElleprocess.h"
using namespace std;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("unused","unused","unused","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    for (int i=0;i<9;i++) 
        userdata[i]=0; // By default, set all user inputs to 0
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    /*
     * Eventually set the interval for writing file, stages etc. if e.g.
	 * this utility should only run once etc.
     */
    //ElleSetSaveFrequency(1);
    //ElleSetStages(1);
    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);	
	/*
     * Set the default output name for elle files
     */
    char cFileroot[] = "FS_emptyElleprocess.elle";  
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


/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;
    /*
     * Clear the data structures
     */
    ElleSetRunFunction(ProcessFunction);
    /*
     * Read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) 
    {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }     
}

/* 
 * Anything can now be in the ProcessFunction itself:
 */ 
int ProcessFunction()
{
    int err=0;
    /*
     * Read the input data and store it in the array of type "Userdata" called
     * "userdata":
     */
    UserData userdata;              // Initialize the "userdata" array
    ElleUserData(userdata);         // Load the input data
    int iInput1 = (int)userdata[0];  // E.g. read the first input
    
    /*!
     * ADD ANY CODE YOU LIKE HERE
     */


    /*
     * This just saves axactly the name specified in "ElleSaveFileRoot()", if 
     * you use it, make sure to add ".elle" to cFileroot in main function:
     */
    //err=ElleWriteData(ElleSaveFileRoot());
    //if(err) OnError("",err);
    
    /* This may be better and saves it with 3 digit stage number:*/
    //err=ElleUpdate();
    //if(err) OnError("",err);
    
    return 0;
}
