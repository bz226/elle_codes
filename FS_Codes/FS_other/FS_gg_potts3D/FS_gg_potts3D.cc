#include "FS_gg_potts3D.h"

/*
 * Some global variables, used very often
 */

/* Constants */
const double dBoltzmann = 1.38064852e-23;

/* User input: First and second double is for probability function*/
const char *cFilename;
double dPZero = 0.1; 
double dC = dBoltzmann;
double dT_Celsius = -30; // temperature in °C
int iStages = 0; // Number of stages to perform
int iDim = 0; // Number of nodes in each direction
int iRandomShuffleOff = 0; // Set to 1 to switch of random shuffeling of nodes
int iStartStage = 0; // Stages already perform in input file only used for naming

/* Frequently used stuff written in capital letters*/
vector<int> vNODESTATES; // Storing all the states of the nodes. 
                         // Position of state in vector will be node ID

int main(int argn, char *argv[])
{
    /* Check for input arguments*/
    printf("\n# # # # # # # # # # # # # # # # # # # # #\n");
    printf("#      WELCOME TO 3D GRAIN GROWTH       #\n");
    printf("# # # # # # # # # # # # # # # # # # # # #\n\n");
    
    /* 
     * Load input arguments if they have been set by user, if not use default 
     * values as defined in the header of this code
     */
    if (argn>uFilename) // argn == 1 if no additional input argument was set
    {
        cFilename = argv[uFilename];
        /*
         * Maybe add a bit of code that runs, when cFilename (1st input) is 
         * equal to "help" or so... In that case give information about input
         */
    }
    else
    {
        printf("ERROR: NO DATAFILE PROVIDED!\n...COMPUTATION STOPPED\n\n");
        return (0);        
    }
       
    if (argn>uPzero)   dPZero = atof(argv[uPzero]);
    if (dPZero == 0) dPZero = 0.1;
    if (argn>uC)       dC = atof(argv[uC]);
    if (dC == 0) dC = dBoltzmann;
    if (argn>uT)       dT_Celsius = atof(argv[uT]);
    if (argn>uDim)     iDim = atoi(argv[uDim]);
    if (argn>uStages)  iStages = atoi(argv[uStages]);
    if (argn>uRandomShuffleOff) 
                        iRandomShuffleOff = atoi(argv[uRandomShuffleOff]);
    if (argn>uStartStep) 
                        iStartStage = atoi(argv[uStartStep]);
       
       
	printf("# # # # # # Input Information # # # # # #\n#\n"); 
    printf("# Input file name: %s\n",cFilename); 
    printf("#\n");  
	printf("# Probability Function P0 = %e\n",dPZero);  
	printf("# Probability Function C  = %e\n",dC); 
	printf("# Temperature  = %f °C\n",dT_Celsius); 
    printf("#\n");
    printf("# Run for %u stages\n",iStages);
    printf("#\n");    
    printf("# Number of nodes in each direction: %u\n",iDim);
    printf("#\n");    
	printf("# # # # # # # # # # # # # # # # # # # # #\n");
    
    if (iRandomShuffleOff) 
        printf("\nATTENTION: Random shuffeling of points is switched off\n");
    
    printf("\n~~~~~~~ STARTING 3D GRAIN GROWTH ~~~~~~~~\n\n");

    GG3DPotts();
    
    return(0);
} 

/* 
 * Start 3D grain growth using a potts model
 */ 
int GG3DPotts()
{
    int err=0;
    
    int iCountStage = 1;
    int iNumbNodes = 0;
    
    vector<int> vRanNode;
    int iNode = 0;
    double dNodeEnergy = 0.0;
    vector<int> vNeighbours;
    vector<int> vNeighboursOfNeighbour;
    int iSwitchNb = 0;
    
    double dEsurf;
    double dNbEnergy = 0.0;
    double dEnergyChange = 0.0;
    double dPswitch = 0.0;
    double dQ_switch = 0.0;
        
    /* Prepare and read input file*/
    ReadInputFile();
    srand (time(NULL));
    
    /* To look at all the data in the input file uncomment the following part:*/
    /*!
    printf("NODES:\n");
    for (int i=0;i<(iDim*iDim*iDim);i++)
    {
        printf("%u %u\n",i,vNODESTATES.at(i));
    }
    */
    
    
    /*
     * Assign surface energies ( maybe better to do it here instead of outside 
     * the code if we want to change the code for polyphase)for two phases
     * HOWEVER: Change this to user input somehow!
     */
    dEsurf = 0.065; // e.g. ice ice boundary
    
    /*
     * Create array to random access all unodes
     */ 
    for (int i=0;i<pow(iDim,3);i++) vRanNode.push_back(i);
    iNumbNodes = vRanNode.size();
    
    /*
     * MAIN LOOP: GO THROUGH ALL UNODES FOR EACH STEP AND PERFORM SWITCHES
     */
    for (int stage = 0; stage < iStages; stage++ )
	{
        printf("3D GG Potts Model - Stage: %d\n\n", iCountStage+iStartStage);
        
        /* Randomisation if not switched off */
        if(!iRandomShuffleOff) random_shuffle(vRanNode.begin(), vRanNode.end());
        
        for (int j=0; j<iNumbNodes; j++)
        {
            iNode = vRanNode.at(j);
            
            /* Find 1st order neighbours of node and node energy */
            FindNodeNbs(iNode,&vNeighbours);
            dNodeEnergy = GetEnergyState(iNode,vNeighbours,dEsurf);
            
            /* Find random neighbour for switch and determine its energy */
            random_shuffle(vNeighbours.begin(), vNeighbours.end()); 
            iSwitchNb = vNeighbours.at(0);
            FindNodeNbs(iNode,&vNeighboursOfNeighbour);
            dNbEnergy = GetEnergyState(iSwitchNb,vNeighboursOfNeighbour,dEsurf);
            
            /* Determine probability for switch */
            dEnergyChange = dNbEnergy-dNodeEnergy;
            //if (dEnergyChange>0) printf("energy change: %f\n",dEnergyChange);
            dPswitch = SwitchProbability(dEnergyChange);
            
            /* See if change will happen using a random number Q between 0-1
             * --> Switch will happen if Q <= dPflip */
            dQ_switch = rand() % 10 + 1; // still between 0 and 10
            dQ_switch /= 10; // now between 0 and 1
            if (dQ_switch <= dPswitch) 
            {
                /* --> For my own interest I could now check for energetically 
                 *     unfavorable switches */               
                
                //printf("Loop %u: Switch will happen\n",iCountSth);
                PerformSwitch(iNode,iSwitchNb);
            }
            vNeighbours.clear();
            vNeighboursOfNeighbour.clear();
        }
        
        /*
         * Write updated output file
         */
        Write3DPottsOutput(iCountStage+iStartStage);
        
        iCountStage++;     
    }   
    vRanNode.clear();
    
    vNODESTATES.clear();
    
    return 0;
}

/*
 * Reads the node information from the input file
 */
void ReadInputFile()
{
    // Dummy integer to temporarily store node ID and state
    int dummy = 0;
    int iNodeStateTmp;

    int iCounter = 0;
    
    ifstream datafile(cFilename);
    if (!datafile)
    {
        printf("ERROR: FILE %s NOT FOUND\n",cFilename);
        return;
    }
    
    while (datafile) 
    {
        datafile >> dummy >> iNodeStateTmp;
        vNODESTATES.push_back(iNodeStateTmp);
        iCounter++;
    }
    
    /* Loading the data like this will duplicate the last node, therefore
     * delete last element in the dataset:
     */
    vNODESTATES.pop_back(); 
    
    datafile.close();
}

/*
 * Function determines the energy state of a node with respect to its 1st order
 * neighbours (i.e. diagonal and direct)
 */
double GetEnergyState(int iNodeID,vector<int> vNbNodes,double dEsurf)
{
    double dEnergy = 0.0;
    int iState = 0.0, iStateNb = 0; // The state of the unode
    int iNbNode = 0;
    int iH = 0; // Will be 0 if neighbour states are the same, 1 if not
    
    // Find phase (or "state") of the relevatnt node and determine its neighbours
    iState = vNODESTATES.at(iNodeID);
    //FindNodeNbs(iNodeID,&vNbNodes); // is now an input of this function
    
    /*
     * Loop through all neighbours to determine energy state
     */
    /* Energy needs different weightings according to distances of 
     * neighbours to center. If direct neighbours hace distance = 1, than 
     * diagonal neighbours in the same 2D plane have distance = sqrt(2) and
     * diagonal neighbours in 3rd dimension have distance = sqrt(3)
     * 
     * Weightings are summarized in the following array
     */
    int iNbWeightSquared[26] = // Needs to be used with sqrt(..) later
        {3,2,3,2,1,2,3,2,3,2,1,2,1,1,2,1,2,3,2,3,2,1,2,3,2,3};
        
    for (int i=0;i<vNbNodes.size();i++)
    {
        iNbNode = vNbNodes.at(i);
        iStateNb = vNODESTATES.at(iNbNode);        

        // Get Hamiltonian:
        if (iState==iStateNb) iH = 0;
        else iH = 1;
        // Compute with neighbour weighting:
        dEnergy += (1/sqrt((double)iNbWeightSquared[i])) * dEsurf*(double)iH;
    }    
    vNbNodes.clear();
    
    return (dEnergy);
}

/*
 * Function searches 1st order node neighbours in a regular square grid with 
 * periodic boundaries in 3D. Output will be stored as Nb_Ids vector (26 entries).
 * --> Only 1st neighbour unodes limits to direct and diagonal neighbours
 * 
 * If we image all neighbours arranged in a cude around the node of interest, 
 * the neighbours are counter from 1 to 26 from the bottom left corner at the 
 * front of the cube. The node neighbours are found using their positions at
 * (row,column,slice) or "RCS coordinate". RCS coords are counted from 0-iDim
 */
void FindNodeNbs(int iNodeID, vector<int>* vNodeNbs)
{
    /*
     * Use row, column and slice of node in grid to determine the neighbours
     * (row,col,slice) = RCS coordinate
     */
    int iRCS_node[3];
    int iRCS_neighbour[3];
    int iNbNodeID = -1;
    
    ID2RCS(iNodeID,iRCS_node);   
    /*
     * Get neighbours from bottom left front corner of cube formed by all 
     * neighbours to top right back corner using modifiers
     * 
     * How these modifiers work:
     * From RCS of node of interest there has to be a subtraction addition or 
     * nothing to come to the RCS of a repsective neighbour, these operations
     * are stored in the size arrays defined here for rows, cols for a 2D case 
     * and will be added with either a subtraction, nothing or addition of 
     * slices
     */
    int iModRow[9] = {-1,-1,-1,0,0,0,1,1,1};
    int iModCol[9] = {-1,0,1,-1,0,1,-1,0,1};
    int iModSlc = 0; // Will be adjusted in the loop below
     
    /* Loop has to go to 27 since the node of interest is also regarded a 
     * neighbour but of course not stored in output array */    
    for (int i=0;i<27;i++)
    {
        /* Find the correct modifier for slice */
        if (i>=0 && i<9) iModSlc = (-1);
        if (i>=9 && i<18) iModSlc = 0;
        if (i>=18 && i<26) iModSlc = 1;   
        
        iRCS_neighbour[0] = iRCS_node[0] + iModRow[i%9];  
        iRCS_neighbour[1] = iRCS_node[1] + iModCol[i%9];  
        iRCS_neighbour[2] = iRCS_node[2] + iModSlc;     
        
        PushRCSIntoBox(iRCS_neighbour);      
        
        /* Get neighbour node ID from RCS */
        iNbNodeID = RCS2ID(iRCS_neighbour);
        
        /* This method will also detect the node of interest itself as a 
         * neighbour, but we do not want to store it in the neighbours array of 
         * course */
        if (iNbNodeID!=iNodeID) vNodeNbs->push_back(iNbNodeID); 
    }
    
    /* Look at results */
    /*!
    if (iNodeID==12)
    {
        printf("All %u neighbours of node %u are:\n",vNodeNbs->size(),iNodeID);
        for (int i=0;i<26;i++)  printf("%u\n",vNodeNbs->at(i));
    } 
    */
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
    double dT_Kelvin  = dT_Celsius + 273.15; // Temperature in Kelvin 	
    // dBoltzmann will be Boltzmann constant in m2 kg s-2 K-1
        
    if (dEnergyChange>0) 
        return ( dPZero*exp((-dC*dEnergyChange)/(dBoltzmann*dT_Kelvin)) );
    else 
        return ( dPZero );
}

/*
 * Function performs switch by switching the state of a node under 
 * investigation to the state of one of the neighbour nodes (switch probability
 * has previously been determined etc.)
 */ 
void PerformSwitch(int iNode, int iNb)
{    
    /* Get neighbour state and set state of node of interest no this state */
    int iStateNb = vNODESTATES.at(iNb);
    
    vNODESTATES.at(iNode) = iStateNb; 
}

/*
 * Function pushes any node RCS coordinate back into the periodic box assuming 
 * a regular square grid in 3D
 */
void PushRCSIntoBox(int iRCS[3])
{
    // Check row:
    if (iRCS[0] < 0) iRCS[0]+=iDim;
    else if (iRCS[0] >= iDim) iRCS[0]-=iDim;
    
    // Check column:
    if (iRCS[1] < 0) iRCS[1]+=iDim;
    else if (iRCS[1] >= iDim) iRCS[1]-=iDim;
    
    // Check slice:
    if (iRCS[2] < 0) iRCS[2]+=iDim;
    else if (iRCS[2] >= iDim) iRCS[2]-=iDim;
     
}

/*
 * Get the position of a node in a 3D regular square grid in row, column and 
 * slice (or in short RCS coordinate) from its ID.
 * Both ID and RCS values are counted from lower left front to upper right back
 * of the cube defined by all nodes starting from 0
 */
void ID2RCS(int iNodeID,int iRCS[3])
{
    int iDim2 = pow(iDim,2);
    iRCS[2] = floor(iNodeID/iDim2);
    iRCS[1] = fmod(iNodeID-(iRCS[2]*iDim2),iDim);
    iRCS[0] = (iNodeID-(iRCS[2]*iDim2)-iRCS[1])/iDim;
}

/*
 * Does the opposite of ID2RCS, check infos there...returns node ID
 */
int RCS2ID(int iRCS[3])
{
    return(iRCS[2]*pow(iDim,2) + iRCS[0]*iDim + iRCS[1]);
}

/*
 * Write output file with the node stages that are currently stored in 
 * vNODESTATES vector. 
 * File will have two columns: 1st is node ID, 2nd is node state, both are 
 * integers. The number of rows will be the number of nodes
 * Delimiter is space: " "
 * 
 * iStage is used to write the number of stages already performed in the ouput 
 * filename
 */
void Write3DPottsOutput(int iStage)
{
    fstream fPottsFile;
    char cOutPutFilename [50];
    sprintf(cOutPutFilename,"3DPotts_Dim%03ustage%04u.txt",iDim,iStage); 
    fPottsFile.open ( cOutPutFilename, fstream::out | fstream::trunc); 
    
    for (int i=0;i<vNODESTATES.size();i++)
    {
        fPottsFile << i << " " << vNODESTATES.at(i) << endl;
    } 
    
    fPottsFile.close();
}


/*
 * OLD FUNCTIONS FROM 2D MODEL
 */

///*
 //* Function performs switch by switching the state of a unode under 
 //* investigation to the state of one of the neighbour nodes (switch probability
 //* has previously been determined etc.)
 //* Additionally the grain ID stored in the unodes is updated since a grain is 
 //* growing with that switch
 //*/ 
//void PerformSwitch(int iUnode, int iNb)
//{
    //double dStateNb = 0.0;       
    //double dGrainIDNb = 0;
    
    //ElleGetUnodeAttribute(iNb,StateAttrib,&dStateNb);
    //ElleSetUnodeAttribute(iUnode,StateAttrib,dStateNb);
    
    ///* 
     //* Also update the grain ID (assign new grain ID if necessary)
     //*/
    //ElleGetUnodeAttribute(iNb,GrainAttrib,&dGrainIDNb);
    
    //if ((int)dGrainIDNb==-1)
    //{
        //iGrainID++; // grain ID +1, because the next unused grain ID is wanted
        //ElleSetUnodeAttribute(iNb,GrainAttrib,(double)iGrainID);  
        //dGrainIDNb = (double)iGrainID;    
    //} 
    
    //ElleSetUnodeAttribute(iUnode,GrainAttrib,dGrainIDNb);
//}

///*
 //* Calculate the switch probability using:
 //* Pflip = P0                        if EnergyChange <= 0
 //* Pflip = P0*exp(-cEnergyChange/kT) if EnergyChange  > 0
 //*/
//double SwitchProbability(double dEnergyChange)
//{
    ///*
     //* Set some parameters and factors -> think about switching this to user
     //* input
     //*/
    //double T  = ElleTemperature() + 273.15; // Temperature in Kelvin 	
    //double k  = 1.3806488e-23; // Boltzmann constant in m2 kg s-2 K-1
    
    //UserData userdata;
    //ElleUserData(userdata);
    
    //double P0 = userdata[uPzero]; //cout <<P0<< endl;
    //double c  = userdata[uC];; //cout <<c<< endl; 
    
    ///*
     //* Check if default values are loaded, if yes: Set the default values for 
     //* P0 and C:
     //*/
    //if (P0==0) P0 = 0.1;
    //if (c==0)  c  = k;
        
    //if (dEnergyChange>0) return ( P0*exp((-c*dEnergyChange)/(k*T)) );
    //else return ( P0 );
//}

///*
 //* Function determines the energy state of a unode with respect to its 1st order
 //* neighbours (i.e. diagonal and direct)
 //*/
//double GetEnergyState(int iUnodeID,double dEsurf)
//{
    //vector<int> vNbUnodes;
    //double dEnergy = 0.0;
    //double dState = 0.0, dStateNb = 0; // The state of the unode
    //int iNbUnode = 0;
    //int iH = 0; // Will be 0 if neighbour states are the same, 1 if not
    
    //// Find phase (or "state") of unode and determine its neighbours
    //ElleGetUnodeAttribute(iUnodeID,StateAttrib,&dState);
    //FindUnodeNbs(iUnodeID,&vNbUnodes);
    
    ///*
     //* Loop through all neighbours to determine energy state
     //*/
    //for (int i=0;i<vNbUnodes.size();i++)
    //{
        //iNbUnode = vNbUnodes.at(i);
        //ElleGetUnodeAttribute(iNbUnode,StateAttrib,&dStateNb);
        //if (i<4)
        //{
            //// direct neighbours
            //if (dState==dStateNb) iH = 0;
            //else iH = 1;
            //dEnergy += dEsurf*(double)iH;             
        //}        
        //else
        //{
            //// diagonal neighbours
            //if (dState==dStateNb) iH = 0;
            //else iH = 1;
            //dEnergy += (1/sqrt(2)) * dEsurf*(double)iH;
        //}
        ////printf("\tUnode %u (state: %f) Nb %u (state: %f) iH = %u\n",iUnodeID,dState,iNbUnode,dStateNb,iH);
    //}    
    //vNbUnodes.clear();
    
    //return (dEnergy);
//}

///*
 //* Function searches 1st order unode neighbours in a regular square grid with 
 //* periodic boundaries. Output will be stored as Nb_Ids vector (8 entries).
 //* --> Only 1st neighbour unodes limits to direct and diagonal neighbours
 //*/
//void FindUnodeNbs(int iUnodeID, vector<int>* vUnodeNbs)
//{
    ///*
     //* Use row and column of unode in grid to determine the neighbours
     //*/
    //// Find row and column of current unode
    //int iCol = fmod(iUnodeID,iDim);
    //int iRow = (iUnodeID-iCol)/iDim;
    //// Used later:
    //int iColNew = iCol,iRowNew = iRow;
    //int iIDNew = 0;
    
    ///*
     //* Go through 8 potential neighbours
     //*/
    //for (int i_nb=0; i_nb<8; i_nb++)
    //{
        //int iColNew = iCol,iRowNew = iRow;
        //switch ( i_nb )
        //{
            //// 1st the direct neighbours
            //case 0:
                //iRowNew = PushIntoBox(iRowNew+1);
                //iColNew = PushIntoBox(iColNew);
                
                //// Find unode ID again from row and col:
                //iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                //vUnodeNbs->push_back(iIDNew);
                //break;
            //case 1:
                //iRowNew = PushIntoBox(iRowNew);
                //iColNew = PushIntoBox(iColNew+1);
                
                //// Find unode ID again from row and col:
                //iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                //vUnodeNbs->push_back(iIDNew);
                //break;
            //case 2:
                //iRowNew = PushIntoBox(iRowNew-1);
                //iColNew = PushIntoBox(iColNew);
                
                //// Find unode ID again from row and col:
                //iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                //vUnodeNbs->push_back(iIDNew);
                //break;
            //case 3:
                //iRowNew = PushIntoBox(iRowNew);
                //iColNew = PushIntoBox(iColNew-1);
                
                //// Find unode ID again from row and col:
                //iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                //vUnodeNbs->push_back(iIDNew);
                //break;
                
            //// Now the diagonal neighbours
            //case 4:
                //iRowNew = PushIntoBox(iRowNew+1);
                //iColNew = PushIntoBox(iColNew-1);
                
                //// Find unode ID again from row and col:
                //iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                //vUnodeNbs->push_back(iIDNew);
                //break;
            //case 5:
                //iRowNew = PushIntoBox(iRowNew+1);
                //iColNew = PushIntoBox(iColNew+1);
                
                //// Find unode ID again from row and col:
                //iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                //vUnodeNbs->push_back(iIDNew);
                //break;
            //case 6:
                //iRowNew = PushIntoBox(iRowNew-1);
                //iColNew = PushIntoBox(iColNew-1);
                
                //// Find unode ID again from row and col:
                //iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                //vUnodeNbs->push_back(iIDNew);
                //break;
            //case 7:
                //iRowNew = PushIntoBox(iRowNew-1);
                //iColNew = PushIntoBox(iColNew+1);
                
                //// Find unode ID again from row and col:
                //iIDNew = fmod(iColNew,iDim) + fmod(iRowNew,iDim)*iDim;
                //vUnodeNbs->push_back(iIDNew);
                //break;
          //}
    //}
    ////TEST:
        ////if (iUnode==3)
        ////{
            ////printf("unode %u has %lu neighbours:\n",iUnode,vNbUnodes.size());
            ////printf("%u\t%u\t%u\n%u\t\t%u\n%u\t%u\t%u\n\n",vNbUnodes.at(0),vNbUnodes.at(1),vNbUnodes.at(2),vNbUnodes.at(3),vNbUnodes.at(4),vNbUnodes.at(5),vNbUnodes.at(6),vNbUnodes.at(7));
        ////} 
//}

///*
 //* Function pushes any unode row OR column back into the periodic box assuming 
 //* a regular square grid
 //*/
//int PushIntoBox(int iRowCol)
//{
    //if (iRowCol < 0) return (iRowCol+iDim);
    //else if (iRowCol >= iDim) return(iRowCol-iDim);
     
    //return iRowCol;
//}
