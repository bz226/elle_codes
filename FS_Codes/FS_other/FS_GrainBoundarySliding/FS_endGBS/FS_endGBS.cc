#include "FS_endGBS.h"
using namespace std;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("GB-phase","ReadTmpProps","unused","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    for (int i=0;i<9;i++) 
        userdata[i]=0; // By default, set all user inputs to 0
    userdata[0] = 2; // Be default, the phase ID for boundary phase is 2
    userdata[1] = 0; // Be default, temporarily stored props are not used for GB-unodes
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    
    ElleSetSaveFrequency(1);
    ElleSetStages(1);
    
    ElleSetInitFunction(InitThisProcess);	
    
    char cFileroot[] = "FS_endGBS.elle";  
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

int ProcessFunction()
{
    int err=0;
    UserData userdata;              
    ElleUserData(userdata); 
    int iReadTmpProps = (int)userdata[1];
    
    printf("\nResetting boundary-phase unodes\n");
    
    /* Check if file is suitable */
    if (!CheckFile()) return 1;
    
    if (iReadTmpProps==0) 
    {
        printf("Using nearest unode option\n\n");
        UseNearestUnodes();
    }
    else 
    {
        if (fileExists("TempGBunodesProps.txt"))
        {
            printf("Using temporarily stored previous attributes\n\n");
            UsePrevAttribs();
        }
        else
        {
            printf("File ""TempGBunodesProps.txt"" does not exist\n");
            printf("Using nearest unode option instead\n\n");
            UseNearestUnodes();
        }
    }
    
    if (ElleWriteData(ElleSaveFileRoot())) OnError("",err);
    
    return 0;
}

void UsePrevAttribs()
{
    /* Read properties from the temporary file that stored previous props
     * ATTENTION: Some unodes may have changed flynns during FFT, therefore,
     * it is important to run FS_topocheck after this script to correct for this
     */
    char fname[]="TempGBunodesProps.txt";
    ifstream datafile(fname);
    
    int iUnodeID, iAttrib;
    double dVisc[2]; // First element indicates if attrib was active, 2nd 
                     // the previous value
    double dDD[2];
    double dA[2],dB[2],dC[2],dD[2],dE[2],dF[2];
    double dE3[4]; // should always have been active, anyway, 1st element 
                   // inidicates if it was active or not
    /* Whether or not an attribute was active before is not so important any
     * more, more important is if it is active right now */
    
    while (datafile) 
    {
        datafile >> iUnodeID >> dVisc[0] >> dVisc[1] >> dDD[0] >> dDD[1]
                 >> dE3[0] >> dE3[1] >> dE3[2] >> dE3[3] >> dA[0] >> dA[1]
                 >> dB[0] >> dB[1] >> dC[0] >> dC[1] >> dD[0] >> dD[1]
                 >> dE[0] >> dE[1] >> dF[0] >> dF[1];     
        
        iAttrib = U_VISCOSITY;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dVisc[1],iAttrib);
        
        iAttrib = U_DISLOCDEN;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dDD[1],iAttrib);
        
        iAttrib = EULER_3;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dE3[1],dE3[2],dE3[3],iAttrib);
        
        iAttrib = U_ATTRIB_A;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dA[1],iAttrib);
        
        iAttrib = U_ATTRIB_B;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dB[1],iAttrib);
        
        iAttrib = U_ATTRIB_C;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dC[1],iAttrib);
        
        iAttrib = U_ATTRIB_D;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dD[1],iAttrib);
        
        iAttrib = U_ATTRIB_E;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dE[1],iAttrib);
        
        iAttrib = U_ATTRIB_F;
        if (ElleUnodeAttributeActive(iAttrib))
            ElleSetUnodeAttribute(iUnodeID,dF[1],iAttrib);
    }
    datafile.close();
    
}

void UseNearestUnodes()
{    
    /* Go through all GB-phase unodes and assign its properties to nearest non 
     * GB-phase unode */  
    UserData udata;
    int iGBphase = (int)udata[0];
    int iFlynn = 0, iNbUnode = 0;
    vector<int> vFUnodes;
      
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        vFUnodes.clear();
        iNbUnode = -1;
        if(IsGBPhase(i,iGBphase))
        {
            /* Get flynn of this unode*/
            iFlynn = ElleUnodeFlynn(i);
            if (ElleFlynnIsActive(iFlynn))
            {
                ElleGetFlynnUnodeList(iFlynn,vFUnodes);     
                /* Find nearest unode to the unode of interest 
                 * (that are not GB phase) */
                iNbUnode = NearestUnode(i,vFUnodes);
                vFUnodes.clear();
                
                if (iNbUnode<0)
                {
                    // no nearest unode was found, keep old properties, only
                    // reset phase ID
                    printf("No nearest unode found for unode ");
                    printf("%u in flynn %u\n",i,iFlynn);
                    printf("Keeping old properties\n");
                }
                else
                {
                    // we found a neighbour unode, reset to these properties
                    ResetUnodeProps(i,iNbUnode);
                }                                   
            }
            else
            {
                // flynn detected for this unode is inactive...should actually
                // never happen...
                printf("WARNING: Flynn %u detected for unode ",iFlynn);
                printf("%u is inactive\n",i);
                printf("Keeping old properties\n");
            }
        }
    }
}

bool CheckFile()
{
    UserData userdata;              
    ElleUserData(userdata); 
    int iGBphase = (int)userdata[0];
    /*
     * Returning true if file is okay, false if there is something wrong
     */
    if(!ElleUnodesActive()) 
    {
        printf("Error: No unodes in file\n\n");  
        return (false);
    }  
    if(!ElleUnodeAttributeActive(iUnodePhaseAttrib)) 
    {
        printf("Error: Phase unode-attribute is inactive\n\n");  
        return (false);
    }    
    // If phase attribute is inactive, initialse and set all unodes to 1 by
    // default
    if(!ElleUnodeAttributeActive(iUnodePhaseAttrib)) 
    {
        ElleInitUnodeAttribute(iUnodePhaseAttrib);
        if (iGBphase!=1)
            ElleSetDefaultUnodeAttribute(1.0,iUnodePhaseAttrib);        
        else
            ElleSetDefaultUnodeAttribute(2.0,iUnodePhaseAttrib);                
    }       
    
    return (true);
}

bool IsGBPhase(int iUnodeID,int iGBphase)
{
    /* returns true if unode is at grain boundary phase, false if not */
    double dPhase = 0.0;
    ElleGetUnodeAttribute(iUnodeID,&dPhase,iUnodePhaseAttrib);
    
    if (iGBphase!=(int)dPhase) return false;
    
    return true;
}

int NearestUnode(int iUnodeID,vector<int> vNbUnodes)
{
    /* Find the nearest unode to iUnodeID in the vector vNbUnodes.
     * Output is nearest unode ID, NOT the element in vector*/
    UserData udata;
    int iGBphase = (int)udata[0]; 
    
    double dMinDist = 1e3, dSep = 0.0;
    int iNearestUnode = -1; // initialise to -1 to later detect if no nearest unode was found
    Coords cPos,cNbPos;
    ElleGetUnodePosition(iUnodeID,&cPos);
    for (int i=0;i<vNbUnodes.size();i++)
    {
        if (iUnodeID!=vNbUnodes[i])
        {
            if (!IsGBPhase(vNbUnodes[i],iGBphase)) // only if this unode is not at boundary-phase as well
            {
                ElleGetUnodePosition(vNbUnodes[i],&cNbPos);
                ElleCoordsPlotXY(&cNbPos,&cPos);
                dSep = pointSeparation(&cNbPos,&cPos);
                
                if (dSep<dMinDist)
                {
                    iNearestUnode = vNbUnodes[i];
                    dMinDist = dSep;       
                }
            }
        }
    }
    
    return (iNearestUnode);
}


void ResetUnodeProps(int iReceive,int iDonate)
{
    /* Resetting unode property of iReceive (receiving new properties) to the 
     * properties "donated" from unode iDonate */
    /* ATTENTION: Phase ID is not resetted...this should be done with 
     * FS_flynn2unodeattribute process later during the simulation */
     
    //double dPhase = 0.0;
    double dDislocden = 0.0, dAttrib = 0.0;
    double dEuler[3];
    
    // phase
    //ElleGetUnodeAttribute(iDonate,&dPhase,iUnodePhaseAttrib);
    //ElleSetUnodeAttribute(iReceive,dPhase,iUnodePhaseAttrib);
    
    // dislocation density
    if (ElleUnodeAttributeActive(U_DISLOCDEN))
    {
        ElleGetUnodeAttribute(iDonate,&dDislocden,U_DISLOCDEN);
        ElleSetUnodeAttribute(iReceive,dDislocden,U_DISLOCDEN);        
    }
    
    // U_ATTRIB_A-F
    int iAttrib = U_ATTRIB_A;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iDonate,&dAttrib,iAttrib);
        ElleSetUnodeAttribute(iReceive,dAttrib,iAttrib);        
    }
    iAttrib = U_ATTRIB_B;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iDonate,&dAttrib,iAttrib);
        ElleSetUnodeAttribute(iReceive,dAttrib,iAttrib);        
    }
    iAttrib = U_ATTRIB_C;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iDonate,&dAttrib,iAttrib);
        ElleSetUnodeAttribute(iReceive,dAttrib,iAttrib);        
    }
    iAttrib = U_ATTRIB_D;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iDonate,&dAttrib,iAttrib);
        ElleSetUnodeAttribute(iReceive,dAttrib,iAttrib);        
    }
    iAttrib = U_ATTRIB_E;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iDonate,&dAttrib,iAttrib);
        ElleSetUnodeAttribute(iReceive,dAttrib,iAttrib);        
    }
    iAttrib = U_ATTRIB_F;
    if (ElleUnodeAttributeActive(iAttrib))
    {
        ElleGetUnodeAttribute(iDonate,&dAttrib,iAttrib);
        ElleSetUnodeAttribute(iReceive,dAttrib,iAttrib);        
    }
    
    // euler angles
    if (ElleUnodeAttributeActive(EULER_3))
    {
        ElleGetUnodeAttribute(iDonate,&dEuler[0],&dEuler[1],&dEuler[2],EULER_3);
        ElleSetUnodeAttribute(iReceive,dEuler[0],dEuler[1],dEuler[2],EULER_3);            
    }
}



bool fileExists(const char *cFilename)
{
  ifstream ifile(cFilename);
  return ifile;
}

