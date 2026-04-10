#include "FS_splitflynn.h"
using namespace std;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("BnodeID1","BnodeID2","FlynnID","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    for (int i=0;i<9;i++) userdata[i]=0; // All defaults: 0
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
    char cFileroot[] = "splitted_flynn.elle";
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
    /*
     * Now that the Elle file is loaded, the user code could potentially check 
     * if e.g. unodes are in the file in case they are necessary or check for
     * attributes in flynns, bnodes or unodes that are needed in this code
     */
    /*! EXAMPLE:
    if(!ElleUnodesActive()) 
    {
        printf("Error: No unodes in file\n\n");  
        return 0;
    } 
    */
     
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
    int newflynn=0;
    int splitnode1 = (int)userdata[0];
    int splitnode2 = (int)userdata[1];
    int oldlynn = (int)userdata[2];
    ElleNewGrain(splitnode1,splitnode2,oldlynn,&newflynn);
    printf("New flynn: %u created by splitting\n",newflynn);

    /* Now check unodes in this flynn and re-assign to U_ATTRIB_C */
    // for the new flynn: Loop through all unodes and check if they are inside
    Coords cUnodePos;
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        ElleGetUnodePosition(i,&cUnodePos);
        if (EllePtInRegion(newflynn,&cUnodePos))
        {
            ElleAddUnodeToFlynn(newflynn,i);
            ElleSetUnodeAttribute(i,(double)newflynn,U_ATTRIB_C);
        }        
    }

    err=ElleWriteData(ElleSaveFileRoot());
    if(err) OnError("",err);
    
    return 0;
}
