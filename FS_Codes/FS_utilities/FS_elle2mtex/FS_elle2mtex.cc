#include "FS_elle2mtex.h"
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
    char cFileroot[] = "FS_elle2mtex";
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
    
    if (!ElleUnodesActive())
    {
        printf("ERROR (elle2mtex): Terminating: No unodes in file.\n");
        err = 1;
        return err;
    }   
    
    int iMaxUnodes = ElleMaxUnodes();
    int iFlynnID = 0;
    double dUnodeVisc = 0.0;
    double dEulerAlpha = 0.0;
    double dEulerBeta = 0.0;
    double dEulerGamma = 0.0;
    Coords cUnodeCoord;
    char cFileName[] = "elle2mtex_output.txt";
    /*
     * IDEA:
     * Loop through all unodes and save their content in a space-delimited 
     * textfile afterwards in the following format:
     * COLUMN1  COLUMN2     COLUMN3     COLUMN4     COLUMN5     COLUMN6:
     * X        Y           euler_alpha euler_beta  euler_gamma phase (U_VISC.)
     */
    if (FileExists(cFileName))
    {
         printf("ATTENTION:\n\tFile %s already exists!\n",cFileName);
         printf("\tYOU HAVE 10 SECONDS TO KILL THAT SCRIPT (CTRL+C)!!\n");
         printf("\tOtherwise it will be overwritten!!\n");
         
         Wait(10);
         
         printf("\n\tSorry, now it's too late: Continuing computation :-P\n");
    }
    fstream fOutFile;
    fOutFile.open (cFileName, fstream::out | fstream::trunc);
     
    /*
     * Loop through all unodes to get the data:
     */
    for (int i=0;i<iMaxUnodes;i++)
    {
        /*
         * Get unode's coordinate
         */
        ElleGetUnodePosition(i,&cUnodeCoord);
        
        /*
         * Get VISCOSITY (phase indicator)
         */        
        if (ElleUnodeAttributeActive(U_VISCOSITY))
        {
            ElleGetUnodeAttribute(i,&dUnodeVisc,U_VISCOSITY);
        }
        else // attribute is not in unodes, take it from flynns if possible
        {
            if (ElleNodeAttributeActive(VISCOSITY))
            {
                iFlynnID=ElleUnodeFlynn(i);
                if (ElleFlynnIsActive(iFlynnID))
                    ElleGetFlynnRealAttribute(iFlynnID,&dUnodeVisc,VISCOSITY);
            }
            else
            {
                /* attribute is neither in unodes, nor in flynns, set viscosity 
                 * to 0 -->to make sure people realise that is incorrect 
                 * data 
                 */
                dUnodeVisc = 0.0;
            }
        }
        /*
         * Get euler angles
         */      
        if (ElleUnodeAttributeActive(EULER_3))
        {
            ElleGetUnodeAttribute(i,
                &dEulerAlpha,
                &dEulerBeta,
                &dEulerGamma,
                EULER_3);
        }
        else // attribute is not in unodes, take it from flynns if possible
        {
            if (ElleNodeAttributeActive(EULER_3))
            {
                iFlynnID=ElleUnodeFlynn(i);
                if (ElleFlynnIsActive(iFlynnID))
                    ElleGetFlynnEuler3(iFlynnID,
                        &dEulerAlpha,
                        &dEulerBeta,
                        &dEulerGamma);
            }
            else 
            {
                /* attribute is neither in unodes, nor in flynns, set Euler 
                 * angles ALL to 1e3 -->to make sure people realise that is 
                 * incorrect data 
                 */
                dEulerAlpha = 1e3;
                dEulerBeta = 1e3;
                dEulerGamma = 1e3;
            }
        }          
        
        /*
         * Write data to file:
         */
        // tab delimited
        //fOutFile << cUnodeCoord.x << "\t" << cUnodeCoord.y << "\t"
                 //<< dEulerAlpha << "\t" << dEulerBeta << "\t" << dEulerGamma
                  //<< "\t" << dUnodeVisc << endl;
        // space delimited:
        fOutFile << cUnodeCoord.x << " " << cUnodeCoord.y << " "
                 << dEulerAlpha << " " << dEulerBeta << " " << dEulerGamma
                  << " " << dUnodeVisc << endl;
    }
    
    fOutFile.close();
    
    return 0;
}

bool FileExists(const char *filename)
{
  ifstream ifile(filename);
  return ifile.good();
}

void Wait(int iSeconds)
{
  clock_t endwait;
  endwait = clock () + iSeconds * CLOCKS_PER_SEC ;
  while (clock() < endwait) {}
}
