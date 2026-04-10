#include "FS_gg_potts.h"

/*
 * Some global variables, used very often
 */
int iDim = 0; // will be sqrt(ElleMaxUnodes())
int iNumbPhases = 1;
int StateAttrib = U_ATTRIB_A; // Attribute storing the state of the unode
int PhaseAttrib = U_VISCOSITY; // Attribute storing the phase of the unode
int GrainAttrib = U_ATTRIB_B; // Attribute storing grain ID (all grains of the same state), 1st checked after 1st step
int iGrainID = -1; // Starting grain ID, default value, 1st grain will have ID = iGrainID+1 = 0

/*
 * Parameters for probability function
 */


int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("Pzero","C","RandomShuffleOff","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    for (int i=0;i<9;i++) userdata[i]=0; // All default values: 0
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);

    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitGGPolyphase);	
	/*
     * Set the default output name for elle files
     */
    char cFileroot[] = "gg_polyphase";
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
int InitGGPolyphase()
{
    char *infile;
    int err=0;

    //ElleReinit();
    ElleSetRunFunction(GGPolyphase);

    infile = ElleFile();
    if (strlen(infile)>0) 
    {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
    
    /*
     * Check if grain attribute is active, if not initialise it and set dummy 
     * default value (will be overwritten later anyway):
     */
    if (!ElleUnodeAttributeActive(GrainAttrib))
    {
        ElleInitUnodeAttribute(GrainAttrib);
        ElleSetDefaultUnodeAttribute((-1.0),GrainAttrib);
    }

    if(!ElleUnodesActive()) OnError("Error: No unodes in file",1);  
}

/* 
 * Start Polyphase grain growth using a potts model
 */ 
int GGPolyphase()
{
    int err=0;
    UserData userdata;
    ElleUserData(userdata);
    int iRandomShuffleOff = (int)userdata[uRandomShuffleOff];
    int iCountStage = 0;
    int iCountSth = 0;
    int iNumbUnodes = 0;
    
    vector<int> vRanUnode;
    int iUnode = 0;
    double dUnodeEnergy = 0.0;
    vector<int> vNeighbours;
    int iSwitchNb = 0;
    double dNbEnergy = 0.0;
    double dEnergyChange = 0.0;
    double dPswitch = 0.0;
    double dQ_switch = 0.0;
        
    /* Prepare*/
    iDim = sqrt(ElleMaxUnodes());
    srand (time(NULL));
    
    /*
     * Assign surface energies for two phases (maybe do it outside the code 
     * later)
     */
    printf("check line %u\n",__LINE__);
    double dEsurf;
    // Assign more flexible in the future:
    dEsurf = 0.065; // e.g. ice ice boundary
    //dEsurf[1] = 0.53; // e.g. ice air boundary
    //dEsurf[2] = 0.0032; // e.g. air air boundary
    
    ///*
     //* Create array to random access all unodes
     //*/    
    //for (int i=0; i<ElleMaxUnodes(); i++) vRanUnode.push_back(i);   
    //if(!iRandomShuffleOff) random_shuffle(vRanUnode.begin(), vRanUnode.end());
    //iNumbUnodes = vRanUnode.size()
    
    /*
     * MAIN LOOP: GO THROUGH ALL UNODES FOR EACH STEP AND PERFORM SWITCHES
     */
    for (int stage = 0; stage < EllemaxStages(); stage++ )
	{
        printf("\nPolyphase GG Potts Model - Stage: %d\n\n", iCountStage);
            
            iCountSth = 0;
        
        /*
         * Create array to random access all unodes
         */    
        vRanUnode.clear();
        for (int i=0; i<ElleMaxUnodes(); i++) vRanUnode.push_back(i);   
        if(!iRandomShuffleOff) random_shuffle(vRanUnode.begin(), vRanUnode.end());
        iNumbUnodes = vRanUnode.size();
        
        for (int j=0; j<iNumbUnodes; j++)
        {
            iUnode = vRanUnode.at(j);
            
            // Get unode energy
            dUnodeEnergy = GetEnergyState(iUnode,dEsurf);
            
            // Find random neighbour for switch and determine its energy
            FindUnodeNbs(iUnode,&vNeighbours);
            random_shuffle(vNeighbours.begin(), vNeighbours.end()); 
            iSwitchNb = vNeighbours.at(0);
            dNbEnergy = GetEnergyState(iSwitchNb,dEsurf);
            
            // Determine probability for switch
            dEnergyChange = dNbEnergy-dUnodeEnergy;
            //if (dEnergyChange>0) printf("energy change: %f\n",dEnergyChange);
            dPswitch = SwitchProbability(dEnergyChange);
            
            // See if change will happen using a random number Q between 0-1
            // --> Switch will happen if Q <= dPflip
            dQ_switch = rand() % 10 + 1; // still between 
            dQ_switch /= 10;
            if (dQ_switch <= dPswitch) 
            {
                // For my own interest: Check for energetically unfavorable switches:
                double dState =0, dStateNb = 0;
                ElleGetUnodeAttribute(iUnode,StateAttrib,&dState);
                ElleGetUnodeAttribute(iSwitchNb,StateAttrib,&dStateNb);
                if (dState == dStateNb) iCountSth++;                
                
                //printf("Loop %u: Switch will happen\n",iCountSth);
                PerformSwitch(iUnode,iSwitchNb);
            }
            vNeighbours.clear();
        }
        //printf("Total number of energetically unfavorable switches: %u\n",iCountSth);
        err=ElleUpdate();
        if(err) OnError("",err); 
        iCountStage++;     
    }   
    
    return 0;
}

/*
 * Function performs switch by switching the state of a unode under 
 * investigation to the state of one of the neighbour nodes (switch probability
 * has previously been determined etc.)
 * Additionally the grain ID stored in the unodes is updated since a grain is 
 * growing with that switch
 */ 
void PerformSwitch(int iUnode, int iNb)
{
    double dStateNb = 0.0;       
    double dGrainIDNb = 0;
    
    ElleGetUnodeAttribute(iNb,StateAttrib,&dStateNb);
    ElleSetUnodeAttribute(iUnode,StateAttrib,dStateNb);
    
    /* 
     * Also update the grain ID (assign new grain ID if necessary)
     */
    ElleGetUnodeAttribute(iNb,GrainAttrib,&dGrainIDNb);
    
    if ((int)dGrainIDNb==-1)
    {
        iGrainID++; // grain ID +1, because the next unused grain ID is wanted
        ElleSetUnodeAttribute(iNb,GrainAttrib,(double)iGrainID);  
        dGrainIDNb = (double)iGrainID;    
    } 
    
    ElleSetUnodeAttribute(iUnode,GrainAttrib,dGrainIDNb);
}

/*
 * Calculate the switch probability using:
 * Pflip = P0                        if EnergyChange <= 0
 * Pflip = P0*exp(-cEnergyChange/kT) if EnergyChange  > 0
 */
double SwitchProbability(double dEnergyChange)
{
    /*
     * Set some parameters and factors -> think about switching this to user
     * input
     */
    double T  = ElleTemperature() + 273.15; // Temperature in Kelvin 	
    double k  = 1.3806488e-23; // Boltzmann constant in m2 kg s-2 K-1
    
    UserData userdata;
    ElleUserData(userdata);
    
    double P0 = userdata[uPzero]; //cout <<P0<< endl;
    double c  = userdata[uC];; //cout <<c<< endl; 
    
    /*
     * Check if default values are loaded, if yes: Set the default values for 
     * P0 and C:
     */
    if (P0==0) P0 = 0.1;
    if (c==0)  c  = k;
        
    if (dEnergyChange>0) return ( P0*exp((-c*dEnergyChange)/(k*T)) );
    else return ( P0 );
}

/*
 * Function determines the energy state of a unode with respect to its 1st order
 * neighbours (i.e. diagonal and direct)
 */
double GetEnergyState(int iUnodeID,double dEsurf)
{
    vector<int> vNbUnodes;
    double dEnergy = 0.0;
    double dState = 0.0, dStateNb = 0; // The state of the unode
    int iNbUnode = 0;
    int iH = 0; // Will be 0 if neighbour states are the same, 1 if not
    
    // Find phase (or "state") of unode and determine its neighbours
    ElleGetUnodeAttribute(iUnodeID,StateAttrib,&dState);
    FindUnodeNbs(iUnodeID,&vNbUnodes);
    
    /*
     * Loop through all neighbours to determine energy state
     */
    for (int i=0;i<vNbUnodes.size();i++)
    {
        iNbUnode = vNbUnodes.at(i);
        ElleGetUnodeAttribute(iNbUnode,StateAttrib,&dStateNb);
        if (i<4)
        {
            // direct neighbours
            if (dState==dStateNb) iH = 0;
            else iH = 1;
            dEnergy += dEsurf*(double)iH;             
        }        
        else
        {
            // diagonal neighbours
            if (dState==dStateNb) iH = 0;
            else iH = 1;
            dEnergy += (1/sqrt(2)) * dEsurf*(double)iH;
        }
        //printf("\tUnode %u (state: %f) Nb %u (state: %f) iH = %u\n",iUnodeID,dState,iNbUnode,dStateNb,iH);
    }    
    vNbUnodes.clear();
    
    return (dEnergy);
}

/*
 * Function searches 1st order unode neighbours in a regular square grid with 
 * periodic boundaries. Output will be stored as Nb_Ids vector (8 entries).
 * --> Only 1st neighbour unodes limits to direct and diagonal neighbours
 */
void FindUnodeNbs(int iUnodeID, vector<int>* vUnodeNbs)
{
    /*
     * Use row and column of unode in grid to determine the neighbours
     */
    // Find row and column of current unode
    int iCol = fmod(iUnodeID,iDim);
    int iRow = (iUnodeID-iCol)/iDim;
    // Used later:
    int iColNew = iCol,iRowNew = iRow;
    int iIDNew = 0;
    
    /*
     * Go through 8 potential neighbours
     */
    for (int i_nb=0; i_nb<8; i_nb++)
    {
        int iColNew = iCol,iRowNew = iRow;
        switch ( i_nb )
        {
            // 1st the direct neighbours
            case 0:
                iRowNew = PushIntoBox(iRowNew+1);
                iColNew = PushIntoBox(iColNew);
                
                // Find unode ID again from row and col:
                iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                vUnodeNbs->push_back(iIDNew);
                break;
            case 1:
                iRowNew = PushIntoBox(iRowNew);
                iColNew = PushIntoBox(iColNew+1);
                
                // Find unode ID again from row and col:
                iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                vUnodeNbs->push_back(iIDNew);
                break;
            case 2:
                iRowNew = PushIntoBox(iRowNew-1);
                iColNew = PushIntoBox(iColNew);
                
                // Find unode ID again from row and col:
                iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                vUnodeNbs->push_back(iIDNew);
                break;
            case 3:
                iRowNew = PushIntoBox(iRowNew);
                iColNew = PushIntoBox(iColNew-1);
                
                // Find unode ID again from row and col:
                iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                vUnodeNbs->push_back(iIDNew);
                break;
                
            // Now the diagonal neighbours
            case 4:
                iRowNew = PushIntoBox(iRowNew+1);
                iColNew = PushIntoBox(iColNew-1);
                
                // Find unode ID again from row and col:
                iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                vUnodeNbs->push_back(iIDNew);
                break;
            case 5:
                iRowNew = PushIntoBox(iRowNew+1);
                iColNew = PushIntoBox(iColNew+1);
                
                // Find unode ID again from row and col:
                iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                vUnodeNbs->push_back(iIDNew);
                break;
            case 6:
                iRowNew = PushIntoBox(iRowNew-1);
                iColNew = PushIntoBox(iColNew-1);
                
                // Find unode ID again from row and col:
                iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                vUnodeNbs->push_back(iIDNew);
                break;
            case 7:
                iRowNew = PushIntoBox(iRowNew-1);
                iColNew = PushIntoBox(iColNew+1);
                
                // Find unode ID again from row and col:
                iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                vUnodeNbs->push_back(iIDNew);
                break;
          }
    }
    //TEST:
        //if (iUnode==3)
        //{
            //printf("unode %u has %lu neighbours:\n",iUnode,vNbUnodes.size());
            //printf("%u\t%u\t%u\n%u\t\t%u\n%u\t%u\t%u\n\n",vNbUnodes.at(0),vNbUnodes.at(1),vNbUnodes.at(2),vNbUnodes.at(3),vNbUnodes.at(4),vNbUnodes.at(5),vNbUnodes.at(6),vNbUnodes.at(7));
        //} 
}

/*
 * Function pushes any unode row OR column back into the periodic box assuming 
 * a regular square grid
 */
int PushIntoBox(int iRowCol)
{
    if (iRowCol < 0) return (iRowCol+iDim);
    else if (iRowCol >= iDim) return(iRowCol-iDim);
     
    return iRowCol;
}
