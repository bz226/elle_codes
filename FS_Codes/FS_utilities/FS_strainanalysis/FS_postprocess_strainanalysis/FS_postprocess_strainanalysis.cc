#include "FS_postprocess_strainanalysis.h"
using namespace std;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata; 
    
    ElleInit();
    ElleSetOptNames("Starstep","Endstep","WriteTxtFile","WriteIncrFiles","unused","unused","unused","unused","unused");
    
    ElleUserData(userdata);
    for (int i=0;i<9;i++) 
        userdata[i]=0; // By default, set all user inputs to 0
    userdata[0] = 1; // By default, first step to process is real 1st simulation step
    userdata[1] = 2; // By default, final step to take data from is 2nd step os simulation
    //userdata[2] = 0; // By default, no text file with finite tensor data is written
    //userdata[3] = 0; // If set to !=0, an elle file for each step containing incremental data is written
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    
    ElleSetInitFunction(InitThisProcess);	
    
    char cFileroot[] = "FS_postprocess_strainanalysis.elle";  
    ElleSetSaveFileRoot(cFileroot);
    
    if (ElleDisplay()) SetupApp(argc,argv);
    
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
    
    // Clear the data structures
    ElleSetRunFunction(ProcessFunction);
    
    // Read the data
    infile = ElleFile();
    if (strlen(infile)>0) 
        if (err=ElleReadData(infile)) 
            OnError(infile,err); 
}

/* 
 * Anything can now be in the ProcessFunction itself:
 */ 
int ProcessFunction()
{
    UserData userdata;             
    ElleUserData(userdata); 
    int iStart = (int)userdata[0];
    int iEnd = (int)userdata[1];
    int iWrite2File = (int)userdata[2];
    int iWriteIncrFile = (int)userdata[3];
    const char *cIncrElleFilePath = "";// you can change "" to the new path where Elle files are stored! 
            // make sure the path ends with "/"  
    const char *cInputPath = ""; // path to tensor files, end with "/"
    const char *cInputRoot = "PosGradTensor"; // without .txt
    char cInroot[500];
    sprintf(cInroot,"%s%s",cInputPath,cInputRoot);
    char cFilename[500]; // used later
    
    /* Checking and preparing stuff:
     * 
     * Checking if all files are present
     * Loading initial incremental position gradient data in unode attribs
     * Preparing U_FINITE_STRAIN (if not active anyway)
     * */
    if (PrepareProcess(cInroot))
    {
        printf("Insufficient data, script terminated\n");
        return 1;
    }
    
    printf("Processing ... please wait\n");
    
    /* In case user switched on this option: Also output incremental Elle
     * file from the 1st step*/
    if (iWriteIncrFile!=0)
    {
        sprintf(cFilename,"%s%03u.txt",cInroot,iStart);
        printf("Writing incremental data of step %u to Elle file\n",iStart);
        WriteData2IncrElleFile(iStart,cFilename,cIncrElleFilePath);           
    }
    
    /* 
     * Loop through all steps and (step by step) calculate finite position
     * gradient tensor (PGT) from the one of the previous step. 
     * 
     * It is enough to 
     * start from start+1, as initial data is already stored in unodes (see 
     * PrepareProcess function):
     */
    double dPGTtmp[6]; //1-4 is F11,F12,F21,F22 and 5,6 is dx,dy
    double dPGTprev[6]; //1-4 is F11,F12,F21,F22 and 5,6 is dx,dy
    double dPGTnew[6]; //1-4 is F11,F12,F21,F22 and 5,6 is dx,dy
    double dDummyVals[3]; // only dummy variable for the 3 remaining columns in text-files
    for (int step=iStart+1;step<=iEnd;step++) 
    {        
        sprintf(cFilename,"%s%03u.txt",cInroot,step);
        
        int i=0;
        ifstream datafile(cFilename);
        
        /* In case user switched on this option: Also output incremental Elle
         * file from this step*/
        if (iWriteIncrFile!=0)
        {
            printf("Writing incremental data of step %u to Elle file\n",step);
            WriteData2IncrElleFile(step,cFilename,cIncrElleFilePath);           
        }
        
        // Going through all unodes:
        while (datafile && i<ElleMaxUnodes()) 
        {        
            // Load data
            datafile >>dDummyVals[0]>>dDummyVals[1]>>dDummyVals[2]
                     >>dPGTtmp[4]>>dPGTtmp[5]  // dx,dy
                     >>dPGTtmp[0]>>dPGTtmp[1]  // f11,f12
                     >>dPGTtmp[2]>>dPGTtmp[3]; // f21,f22
                     
                     
                     
            //// Load previous finite tensor in U_FINITE_STRAIN
            ElleGetUnodeAttribute(i,&dPGTprev[0],U_ATTRIB_A);
            ElleGetUnodeAttribute(i,&dPGTprev[1],U_ATTRIB_B);
            ElleGetUnodeAttribute(i,&dPGTprev[2],U_ATTRIB_C);
            ElleGetUnodeAttribute(i,&dPGTprev[3],U_ATTRIB_D);
            ElleGetUnodeAttribute(i,&dPGTprev[4],U_ATTRIB_E);
            ElleGetUnodeAttribute(i,&dPGTprev[5],U_ATTRIB_F);
            
            
            // Calculate finite tensor from previous to this step
            /* Explanation:
             * 
             * Fnew = Fincr_thisstep * Fprev_step (tensor multiplication)
             * dxnew = dxincr_thisstep+( dxprev_step*Fincr_thisstep )
             * dynew = see dxnew
             * 
             */
            // f11,f12,f21,f22:
            dPGTnew[0] = dPGTtmp[0]*dPGTprev[0] + dPGTtmp[1]*dPGTprev[2]; 
            dPGTnew[1] = dPGTtmp[0]*dPGTprev[1] + dPGTtmp[1]*dPGTprev[3]; 
            dPGTnew[2] = dPGTtmp[2]*dPGTprev[0] + dPGTtmp[3]*dPGTprev[2];  
            dPGTnew[3] = dPGTtmp[2]*dPGTprev[1] + dPGTtmp[3]*dPGTprev[3];
            //dx,dy:
            dPGTnew[4] = dPGTtmp[4] + ( dPGTtmp[0]*dPGTprev[4]+ dPGTtmp[1]*dPGTprev[5]);
            dPGTnew[5] = dPGTtmp[5] + ( dPGTtmp[2]*dPGTprev[4]+ dPGTtmp[3]*dPGTprev[5]);
            
            // Reset finite data in unode attributes
            ElleSetUnodeAttribute(i,dPGTnew[0],U_ATTRIB_A);
            ElleSetUnodeAttribute(i,dPGTnew[1],U_ATTRIB_B);
            ElleSetUnodeAttribute(i,dPGTnew[2],U_ATTRIB_C);
            ElleSetUnodeAttribute(i,dPGTnew[3],U_ATTRIB_D);
            ElleSetUnodeAttribute(i,dPGTnew[4],U_ATTRIB_E);
            ElleSetUnodeAttribute(i,dPGTnew[5],U_ATTRIB_F);
                                    
            i++;
        }
        datafile.close();             
    }
    
    // test something:
    //double dummy = 0.0,tmp=0.0;
    //for (int i=0;i<ElleMaxUnodes();i++)
    //{
        //ElleGetUnodeAttribute(i,&tmp,U_ATTRIB_B);
        //dummy+=tmp;
    //}
    //printf("Mean finite f12 = %f\n",dummy/(double)ElleMaxUnodes());
    
    if (iWrite2File!=0)
    {
        printf("Writing data to text-file\n");
        WriteData2TxtFile();        
    }
        
    if( ElleWriteData(ElleSaveFileRoot()) ) OnError("",1);
    return 0;
}

int PrepareProcess(const char *cInroot)
{
    UserData userdata;             
    ElleUserData(userdata); 
    int iStart = (int)userdata[0];
    int iEnd = (int)userdata[1];  
    char cFname[50];
    
    if (!ElleUnodesActive)
    {
        printf("ERROR: No unodes in file\n");
        return 1;
    }
    
    if (iEnd<iStart)
    {
        printf("ERROR: First processing step should be <= last one\n");
        return 1;
    }
    
    // Check if all input files exist
    bool bAllFilesExist = true;
    for (int i=iStart;i<=iEnd;i++)
    {
        sprintf(cFname,"%s%03u.txt",cInroot,i);
        if (!fileExists(cFname))
        {
            printf("ERROR: File %s does not exist\n",cFname);
            bAllFilesExist = false;
        }        
    }
    if (!bAllFilesExist)
    {
        printf("INSUFFICIENT INPUT FILES!\n");
        return 1;
    }
    
    if (!ElleUnodeAttributeActive(U_FINITE_STRAIN))
    {
        printf("\nWARNING: The attribute U_FINITE_STRAIN does not exist:\n");
        printf("This is fine for this script but might cause trouble while ");
        printf("plotting\n\n");
        
        printf(">Loading positions in final incremental position\n");
        printf(">gradient file to U_FINITE_STRAIN\n\n");
        ElleInitUnodeAttribute(U_FINITE_STRAIN);
        
        // Load and add data like indicated in "printf" above:
        sprintf(cFname,"%s%03u.txt",cInroot,iEnd);
        ifstream datafile(cFname); 
        int i=0;
        double dVals[9];
        while (datafile && i<ElleMaxUnodes()) 
        {            
            // Load position gradient tensor from 
            datafile >>dVals[0]>>dVals[1]>>dVals[2]>>dVals[3]>>dVals[4]>>dVals[5]
                     >>dVals[6]>>dVals[7]>>dVals[8];
            
            // Store in U_FINITE_STRAIN
            ElleSetUnodeAttribute(i,0.0,START_S_X);
            ElleSetUnodeAttribute(i,0.0,START_S_Y);
            ElleSetUnodeAttribute(i,0.0,PREV_S_X);
            ElleSetUnodeAttribute(i,0.0,PREV_S_Y);
            ElleSetUnodeAttribute((int)dVals[0],dVals[1],CURR_S_X);
            ElleSetUnodeAttribute((int)dVals[0],dVals[2],CURR_S_Y);
            
            i++;
        }
        datafile.close();      
    }
    
    /* Deleting existing attributes (U_ATTRIB_A-F) and adding new ones*/
    vector<int> vAttribs;
    vAttribs.push_back(U_ATTRIB_A);
    vAttribs.push_back(U_ATTRIB_B);
    vAttribs.push_back(U_ATTRIB_C);
    vAttribs.push_back(U_ATTRIB_D);
    vAttribs.push_back(U_ATTRIB_E);
    vAttribs.push_back(U_ATTRIB_F);
    vAttribs.push_back(E3_ALPHA);
    vAttribs.push_back(E3_BETA);
    vAttribs.push_back(E3_GAMMA);
    vAttribs.push_back(U_DISLOCDEN);
    
    // cleaning the file:
    ElleRemoveUnodeAttributes(vAttribs); 
    // add clean attributes (only A-F, the 1st six ones in vector):
    for (int i=0;i<6;i++) ElleInitUnodeAttribute(vAttribs[i]);
    // store initial data (from 1st step to process) in unode attribs:
    sprintf(cFname,"%s%03u.txt",cInroot,iStart);
    StoreIniData(cFname);
            
    return 0;
}

int StoreIniData(const char *cFilename)
{
    /* Reads position gradient data from one of the output text-files. Data is
     * stored in unode attribs A-F. A-D are tensor elements f11,f12,f21,f22 and
     * E,F are rigid body translations dx and dy */
    double dVals[9];
    
    int i=0;
    ifstream datafile(cFilename);
    while (datafile && i<ElleMaxUnodes()) 
    {        
        // Load data
        datafile >>dVals[0]>>dVals[1]>>dVals[2]>>dVals[3]>>dVals[4]>>dVals[5]
                 >>dVals[6]>>dVals[7]>>dVals[8];
        
        // Store in U_FINITE_STRAIN
        ElleSetUnodeAttribute(i,dVals[5],U_ATTRIB_A);
        ElleSetUnodeAttribute(i,dVals[6],U_ATTRIB_B);
        ElleSetUnodeAttribute(i,dVals[7],U_ATTRIB_C);
        ElleSetUnodeAttribute(i,dVals[8],U_ATTRIB_D);
        ElleSetUnodeAttribute(i,dVals[3],U_ATTRIB_E);
        ElleSetUnodeAttribute(i,dVals[4],U_ATTRIB_F);
        
        i++;
    }
    datafile.close();
    
    return 0;
}

void WriteData2TxtFile()
{
    double dTensor[4];
    double dX,dY;

    fstream fOutFile;
    fOutFile.open ( "FinitePosGradTensor.txt", fstream::out | fstream::trunc); 
    fOutFile << "unode dx dy f11 f12 f21 f22"<<endl;
    
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        ElleGetUnodeAttribute(i,&dTensor[0],U_ATTRIB_A); //f11
        ElleGetUnodeAttribute(i,&dTensor[1],U_ATTRIB_B); //f12
        ElleGetUnodeAttribute(i,&dTensor[2],U_ATTRIB_C); //f21
        ElleGetUnodeAttribute(i,&dTensor[3],U_ATTRIB_D); //f22
        
        ElleGetUnodeAttribute(i,&dX,U_ATTRIB_E); // dx
        ElleGetUnodeAttribute(i,&dY,U_ATTRIB_F); // dy
        
        fOutFile << i << " ";
        fOutFile << dX << " " << dY << " ";
        fOutFile << dTensor[0] << " ";
        fOutFile << dTensor[1] << " ";
        fOutFile << dTensor[2] << " ";
        fOutFile << dTensor[3] << endl;

    }
    fOutFile.close();
}

void WriteData2IncrElleFile(int iStep,const char *cFilename,const char *cWritePath)
{
    /* Writing all relevant data in incremental Elle file of this step
     * 
     * This requires us to write a file without flynns and bnodes, as this 
     * information is currently unknown. For later plotting, it would have to
     * be added from the actual Elle file from this step
     */
    int i=0;
    ifstream datafile(cFilename);
    vector<double> vXpassmarker,vYpassmarker,vDX,vDY,vF11,vF12,vF21,vF22;
    double dVals[9];
    
    // To write the Elle file "manually":
    char cEllefileName[500];
    sprintf(cEllefileName,"%sFS_postprocess_strainanalysis_incremental_step%03u.elle",cWritePath,iStep);
    fstream fEllefile;
    fEllefile.open( cEllefileName, fstream::out | fstream::trunc);
    fEllefile << "UNODES" << endl;
    
    // Going through all unodes and reading data:
    while (datafile && i<ElleMaxUnodes()) 
    {        
        // Load data
        datafile >>dVals[0]
                 >>dVals[1]>>dVals[2] // x,y position of passive marker point
                 >>dVals[3]>>dVals[4]  // dx,dy
                 >>dVals[5]>>dVals[6]  // f11,f12
                 >>dVals[7]>>dVals[8]; // f21,f22    
                 
        // Store data:
        vXpassmarker.push_back(dVals[1]);
        vYpassmarker.push_back(dVals[2]);
        vDX.push_back(dVals[3]);
        vDY.push_back(dVals[4]);
        vF11.push_back(dVals[5]);
        vF12.push_back(dVals[6]);
        vF21.push_back(dVals[7]);
        vF22.push_back(dVals[8]);
        
        // write dummy unode x,y position: 
        // not used in FS_plot_strainanalysis anyway
        fEllefile << i << " 0 0" << endl;
        i++;
    }
    datafile.close(); 
    
    // Test if number of unodes is alright
    if (vDX.size()!=ElleMaxUnodes())
    {
        printf("ERROR: Error writing incremental data to Elle file\n");
        fEllefile.close();
    
        vXpassmarker.clear();
        vYpassmarker.clear();
        vDX.clear();
        vDY.clear();
        vF11.clear();
        vF12.clear();
        vF21.clear();
        vF22.clear();  
        return;
    }
    
    // Write remaining data to Elle file
    
    // U_ATTRIB_A
    fEllefile << "U_ATTRIB_A" << endl;
    for (int i=0;i<ElleMaxUnodes();i++) 
        fEllefile << i << " " << vF11[i] << endl;
    // U_ATTRIB_B
    fEllefile << "U_ATTRIB_B" << endl;
    for (int i=0;i<ElleMaxUnodes();i++) 
        fEllefile << i << " " << vF12[i] << endl;
    // U_ATTRIB_C
    fEllefile << "U_ATTRIB_C" << endl;
    for (int i=0;i<ElleMaxUnodes();i++) 
        fEllefile << i << " " << vF21[i] << endl;
    // U_ATTRIB_D
    fEllefile << "U_ATTRIB_D" << endl;
    for (int i=0;i<ElleMaxUnodes();i++) 
        fEllefile << i << " " << vF22[i] << endl;
    // U_ATTRIB_E
    fEllefile << "U_ATTRIB_E" << endl;
    for (int i=0;i<ElleMaxUnodes();i++) 
        fEllefile << i << " " << vDX[i] << endl;
    // U_ATTRIB_F
    fEllefile << "U_ATTRIB_F" << endl;
    for (int i=0;i<ElleMaxUnodes();i++) 
        fEllefile << i << " " << vDY[i] << endl;
    // U_FINITE_STRAIN
    fEllefile << "U_FINITE_STRAIN START_S_X START_S_Y PREV_S_X PREV_S_Y CURR_S_X CURR_S_Y" << endl;
    for (int i=0;i<ElleMaxUnodes();i++) 
    {
        // first 4 values are irrelevant:
        fEllefile << i << " 0 0 0 0 ";
        // last 2 ones are the actual passive marker position
        fEllefile << vXpassmarker[i] << " " << vYpassmarker[i] << endl;
    }
    
    
    fEllefile.close();
    
    vXpassmarker.clear();
    vYpassmarker.clear();
    vDX.clear();
    vDY.clear();
    vF11.clear();
    vF12.clear();
    vF21.clear();
    vF22.clear();      
}

bool fileExists(const char *cFilename)
{
  ifstream ifile(cFilename);
  return ifile.good();
}
