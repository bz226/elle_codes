#include "FS_header_DDEnergyRecovery.h"

double theta=1e-5; // trial rotation angle in degrees
double rot_mobL=1e21; // actually this is now called rotation viscosity
// FS (!) DEBUGGING 
int iFSDEBUGUNODE=0;
double dBurgersVec = 4.5e-10; // ice default value (in meter)
double dDDLineEnergy = 3.6e10; // ice default attrib (in J/m)
double dHAGB = 5.0; // ice default attrib (in degrees)

int symm_op;
int valf[2000];
extern runtime_opts Settings_run; // FS (!) To have the option to set start step to s.th. else than zero

int InitThisProcess()
{
    char *infile; // input file
    int err=0;

	/* 
	 *clear data structures
	 */
    ElleReinit(); 
    ElleSetRunFunction(FS_doannealing);

    /*
     * read the data
     */    
    infile = ElleFile();
    if (strlen(infile)>0) 
    {
        if (err=ElleReadData(infile)) 
            OnError(infile,err);         
    }
}

int FS_RecoveryTesting()
{
    /* This is just for testing and is switched off when running this 
     * recovery code */  
     
    /* Perform all possible rotations and save output files of the 
     * respective result to see which rotations are on basal, nonbasal planes
     * 
     * userdata[0] specifies which rotation to choose in FS_rot_matrix function
     * userdata[1] specifies the angle in degrees
     */
    UserData userdata;
	ElleUserData(userdata);
    
    int iMode = (int)userdata[0];
    double dRotAnge = userdata[1];
        
    int iError = 0;
    double dEulersIni[3];
    double dEulersOut[3];
    double dRmap[3][3];    
    
    for (int i=0;i<ElleMaxUnodes();i++)
    {
        ElleGetUnodeAttribute(i,&dEulersIni[0],&dEulersIni[1],&dEulersIni[2],EULER_3);
        for (int j=0;j<3;j++) dEulersIni[j]*=D2R;
        
        /* Create rotation matrix for the specific rotation */
        iError=FS_rot_matrix(iMode,dRmap, dRotAnge);

        /* error if rotating didn't work */	  	
        if (iError!=0)
        OnError("rotating failed",0);	                            
        /* 
         * Multiply the current orientation with the rotation matrix then put 
         * into new orientation, convert back to degrees
         */
        matvecmultalb(dRmap,dEulersIni,dEulersOut);
        for (int j=0;j<3;j++) dEulersOut[j]*=R2D;
        
        /* Set new unode orientation */
        ElleSetUnodeAttribute(i,dEulersOut[0],dEulersOut[1],dEulersOut[2],EULER_3);
    }
    
    char outfile[50];
    sprintf(outfile,"rotated_mode%u.elle",iMode);
    
    ElleWriteData(outfile);
    
    return 0;
}

int FS_doannealing()
{
    int 	iStages,iUnodeID;
	int		iError=0;
	int  	iNumberOfTrials=6;
	int		iNumNbs=0;
    
	double  dEdisloc, dMinE;
    double  dEulerTmp[3],dEulerNew[3];
	double  dSymm[24][3][3];
	double	dMisOri[8];
    double  dTotMisOri, dFinalMisOri, dMisoriBefore, dIniMisori;
    double  dOriChange; // the change in orientation of the unode after a trial rotation (this is NOT misorientation to neighbours!)
    double  dTotDist;
	double  dCurrOri[3], dNewOri[3];
	double	**dNbOri=0;
	double	dRmap[3][3];
	Coords  cUnodeXY;
    
    UserData userdata;
	ElleUserData(userdata);

    // albert 
    int iFlynnID, iNumUnodes, iUnodesRotatedTotal;
	double dAvRot;
    
    // Florian (!)
        int iCountTotalRotations=0; // Counting the total number of rotations to determine average rotation later
        int iExcludedPhase = (int)userdata[1]; 
        int iNoRandomisation = (int)userdata[2];   
        double dFlynnPhase = 0;
        int iPhaseAttrib = VISCOSITY;
        int iMineralType = 0.0;
        int iUserInStartStep = (int)userdata[4];
        if ( iUserInStartStep > 1 )
            Settings_run.Count = iUserInStartStep;

    dHAGB=userdata[0]; 
    
	rot_mobL=userdata[3]; // FS (!) calculate that internally using a function or so
	
    /* Initial user messages */
    printf("\n# # # # # # # DDEnergy Recovery # # # # # # #\n");
    printf("#                                           #\n");
    printf("#  Input parameters:                        #\n");
    printf("#                                           #\n");
    printf("#  Excluded phase ID:      %u                #\n",iExcludedPhase);
    printf("#  Trial rotation angle:   %.2e°        #\n",theta);
    printf("#  High angle boundary at: %.2f°            #\n",dHAGB);
    printf("#  Rotation viscosity:     %.2e Pas     #\n",rot_mobL);
    printf("#  Timestep:               %.2e s       #\n",ElleTimestep());
    printf("#                                           #\n");
    printf("# # # # # # # # # # # # # # # # # # # # # # #\n\n");

    /* Check and init flynns and unodes attributes */
    if (!ElleUnodeAttributeActive(U_ATTRIB_F))
    {
        ElleInitUnodeAttribute(U_ATTRIB_F);
        ElleSetDefaultUnodeAttribute(0.0,U_ATTRIB_F);
    }
    if (!ElleUnodeAttributeActive(U_DISLOCDEN))
    {
        ElleInitUnodeAttribute(U_DISLOCDEN);
        ElleSetDefaultUnodeAttribute(0.0,U_DISLOCDEN);
    }
    bool FSDEBUG=false;
    // FS (!) DEBUGGING 
    if (FSDEBUG) 
    {
        printf("\nDEBGGUNG:\nCheck line %u and following lines for debugging stuff\n",__LINE__);
        printf("Search for \"FS (!) DEBUGGING\" to find where debugging is active\n");

        if (!ElleUnodeAttributeActive(U_ATTRIB_A))
            ElleInitUnodeAttribute(U_ATTRIB_A);
        ElleSetDefaultUnodeAttribute(0.0,U_ATTRIB_A);
        if (!ElleUnodeAttributeActive(U_ATTRIB_B)) 
            ElleInitUnodeAttribute(U_ATTRIB_B);
        ElleSetDefaultUnodeAttribute(0.0,U_ATTRIB_B);
        if (!ElleUnodeAttributeActive(U_ATTRIB_C)) 
            ElleInitUnodeAttribute(U_ATTRIB_C);
        ElleSetDefaultUnodeAttribute(0.0,U_ATTRIB_C);
    }
	
    vector <int> vFlynnRan, vUnodeRan;
    vector <int> vNbUnodes;
    
    vFlynnRan.clear();
    
    symmetry(dSymm);
   
    /*
     * FS (!): Store all IDs of active flynns in a vector. Furthermore:
     * Only use flynns that are not of the phase that was excluded by the user
     */
	for (int i=0;i<ElleMaxFlynns();i++) 
    {
        if (ElleFlynnIsActive(i)) 
        {
            if (ElleFlynnAttributeActive(iPhaseAttrib))
            {
                ElleGetFlynnRealAttribute(i,&dFlynnPhase,iPhaseAttrib);  
                if (iExcludedPhase!=(int)dFlynnPhase) 
                {
                    /*
                     * If excluded phase is not the phase of this flynn: 
                     * Use it (and its unodes) for calculations:   
                     */
                    vFlynnRan.push_back(i);
                }     
            }
            else
            {
                vFlynnRan.push_back(i);
            }
        }
	}
    
    /*
     * FS (!): Give user message if randomisation has been switched off:
     */
    if (iNoRandomisation)
        printf("ATTENTION: Random picking of flynns and "
                "unodes has been switched off!\n");    


    /*
     * FS (!): 
     * IMPORTANT: NOW START LOOPING THROUGH ALL STEPS:
     * "iStages" contains the step number
     */    
    for(iStages=0;iStages<EllemaxStages();iStages++)
    {
		iUnodesRotatedTotal=0;
		dAvRot=0;		
        iCountTotalRotations=0;
		printf("\nPolyphase Recovery by Subgrain Rotation - Stage: %d\n\n",iStages+iUserInStartStep);
 		
        /*
         * FS (!): Randomize all flynn IDs if user did not switch off this 
         * option by setting userdata[2] to 1 (0 is default):
         */
        if (!iNoRandomisation)
            random_shuffle(vFlynnRan.begin(), vFlynnRan.end());
    
        /* setup an array of random unodes - vFlynnRan[i] */	
        int iCountTooFast=0; // track how many unodes rotate too fast
    		
        /*
         * FS (!): Loop through all flynns in "vFlynnRan" and only use unodes in 
         * the relevant flynn for calculations
         */
    	for (int i=0;i<vFlynnRan.size();i++) // FS (!) closes at the end of the ElleStages loop
    	{
            vUnodeRan.clear();
            iFlynnID = vFlynnRan[i];

            /*
             * Find this flynn's material properties
             */
            ElleGetFlynnIntAttribute(iFlynnID,&iMineralType,MINERAL);
            dBurgersVec = GetMineralAttribute(iMineralType,BURGERS_VECTOR);
            dDDLineEnergy = GetMineralAttribute(iMineralType,DD_ENERGY); 
            //dHAGB = userdata[0]; // should be material property in mindatabase
			
            /*
             * FS (!): Tirangulate unodes: Neccessary to later find the 
             * neighbours of the unode 
             */
            ElleClearTriAttributes();
            TriangulateUnodes(iFlynnID,MeshData.tri);

            /* FS (!): Find all unodes in this flynn: */
            vUnodeRan.clear();
            ElleGetFlynnUnodeList(iFlynnID,vUnodeRan);
            iNumUnodes=vUnodeRan.size();

            /* Special case: no unodes in flynn, avoid problems:*/
            if (iNumUnodes <= 0 ) 
            {                     
                iNumUnodes=0;
            }
            else
            {
                /*
                 * FS (!): Only randomly shuffle unodes if user did not switch
                 * this option off by setting userdata[2] to 1 (which is 0 by 
                 * default)
                 */
                if (!iNoRandomisation) 
                    random_shuffle(vUnodeRan.begin(), vUnodeRan.end());	
                //printf("Flynn %u\n",iFlynnID);
            }

            /* FS (!): Loop through all unodes in the relevant flynn "i" */
            for (int unode=0; unode<iNumUnodes; unode++) // FS (!) closes at the end of the ElleStages loop
            {
                iUnodeID = vUnodeRan[unode];
                vector<int> vBndflag;	
                vNbUnodes.clear();
                dIniMisori = 0.0; // just for testing purposes
                dTotDist = 0.0;              
                
                /* get attributes for and position of each unode */
                ElleGetUnodeAttribute(iUnodeID,
                    &dEulerTmp[0],&dEulerTmp[1],&dEulerTmp[2],EULER_3);


                ElleGetUnodePosition(iUnodeID,&cUnodeXY);
            
                /* current orientation array - convert to radians */
                dCurrOri[0]=dEulerTmp[0]*D2R;
                dCurrOri[1]=dEulerTmp[1]*D2R;
                dCurrOri[2]=dEulerTmp[2]*D2R;		
                
                /* Store orientation in case no recovery needs to take place
                 * Will be rotated back to degrees when setting new euler angles
                 */                
                dEulerNew[0]=dCurrOri[0];
                dEulerNew[1]=dCurrOri[1];
                dEulerNew[2]=dCurrOri[2];

                /* use unodes triangulation to find neighbours unodes*/
                //get the  list of neighbours for a unode --> 0 is to not include unodes on the boundary --> check basecode	
                ElleGetTriPtNeighbours(iUnodeID,vNbUnodes,vBndflag,0); 
                // vBndflag is hardly unused, only necessary for ElleGetTriPtNeighbours
                iNumNbs = vNbUnodes.size();
                                
                if(iNumNbs!=0) // If the number of neighbours is > 0
                {			
                    vector <Coords> cNbUnodeXY(iNumNbs);
                    
                    /* setup a matrix to hold the euler angle info plus num of nbs*/   			
                    if (dNbOri==0) dNbOri=dmatrix(0,iNumNbs,0,5);

                    /*
                    * FS (!): dNbOri stores orientation of the "k-th" 
                    * neighbour. Number of neighbours is thereofre = k
                    * i.e.: k = nbnodes.size();
                    * Neighbour-id [0]: Euler-alpha [1]: Euler-beta [2]: Euler-gamma  [3] Normalised Distance and [4] Actual distance to center unode (iUnodeID)
                    *          dNbOri[k][0]   dNbOri[][1] dNbOri[][2] dNbOri[][3]  dNbOri[][4]
                    *      0       53          50          176         0.9        4e-4 
                    *      1       ..          ..          ..          ..          ..
                    *      2       ..          ..          ..          ..          ..
                    *      ..      ..          ..          ..          ..          ..
                    *      k-1     ..          ..          ..          ..          ..
                    *      k       ..          ..          ..          ..          ..
                    */

                    /* get info for each of nb nodes and put into matrix*/    		
                    for (int k=0; k<iNumNbs;k++)
                    {
                        ElleGetUnodeAttribute(vNbUnodes[k],
                            &dEulerTmp[0],&dEulerTmp[1],&dEulerTmp[2],EULER_3);

                        ElleGetUnodePosition(vNbUnodes[k],&cNbUnodeXY[k]);

                        dNbOri[k][0]=dEulerTmp[0]*D2R;
                        dNbOri[k][1]=dEulerTmp[1]*D2R;
                        dNbOri[k][2]=dEulerTmp[2]*D2R;
                    }
                    vNbUnodes.clear(); // not needed afterwards 
                    /* Get separation to neighbour unodes to normalise later: 
                     *
                     * FS (!): 
                     * cUnodeXY: Coords of unode of interest
                     * cNbUnodeXY: Coords of all neighbours
                     * dNbOri: Matrix containing orientations for all neighbours
                     * iNumNbs: Number of neighbours
                     *
                     * dNbOri[k][0 until 2] contains 3 euler angles
                     * dNbOri[k][3] will after normsep contain the correct 
                     * distance to center unode
                     */
                    iError=FS_norm_sep(cUnodeXY,cNbUnodeXY,dNbOri,iNumNbs);

                    /*
                    * FS (!):
                    * dNbOri[k][3] will afterwards contain distance from 
                    * neighbour to center unode NORMALIZED TO NEAREST
                    * NEIGHBOUR:
                    * IF dNbOri[k][3] = 1 --> nearest (direct) neighbour
                    * IF dNbOri[k][3] = .71 -> diagonal neighbour
                    * Every value between 0 and 1: Some intermediate 
                    * distance: Possible if the unode grid is slightly 
                    * distored due to FFT code --> hence distance can also 
                    * get smaller than 0.71
                    * 
                    * Also dNbOri[k][4] will store the absolute distance in
                    * Elle units, this will not change during the trial 
                    * rotations, so we can calculate it right now:
                    * We also use a weighting with NORMALIZED nearest neighbour 
                    * distances:
                    */  
                    for (int k=0;k<iNumNbs;k++)
                        dTotDist += dNbOri[k][4]*dNbOri[k][3];                      

                    /* 
                    * Put the current orientation into new orientation for now,
                    * dNewOri will only stay like this if there is no rotation
                    * at all.
                    */                        
                    dNewOri[0]=dCurrOri[0];
                    dNewOri[1]=dCurrOri[1];
                    dNewOri[2]=dCurrOri[2];                

                    /* go through rotation trials to find lowest E config */  
                    int iCountRotations = 0; // counts how many unodes have seen rotation
                    int iCounter = 0; // helps counting how many unodes rotate too fast	
                    dMisoriBefore=0.0; // needs to be set to zero here  
                    for (int iTrial=0;iTrial<iNumberOfTrials+1;iTrial++)
                    {
                        dTotMisOri=0.0;

                        /* FS (!): if iTrial == 0 use dCurrOri as dNewOri,
                        * otherwise perform all the trial rotations with 
                        * the trial angle theta 
                        */
                        if (iTrial>0)
                        {
                            /* FS (!) Create rotation matrix for the 
                            * specific rotation
                            */
                            iError=FS_rot_matrix(iTrial-1,dRmap, theta);

                            /* error if rotating didn't work */	  	
                            if (iError!=0)
                            OnError("rotating failed",0);	                            
                            /* 
                            * multiply the current ori with the rotation matrix 
                            * then put into dNewOri
                            */
                            matvecmultalb(dRmap,dCurrOri,dNewOri);
                        }    

                        for (int k=0; k<iNumNbs;k++)
                        {
                                            
                            /* 
                             * Get misorientations for this unode: 
                             * CME_hex outputs angle in degree, but NbOri[0-2]
                             * and dNewOri[0-2] need to be in radians 
                             */
                            dMisOri[k]=CME_hex(dNbOri[k][0],dNbOri[k][1],dNbOri[k][2],
                            dNewOri[0],dNewOri[1],dNewOri[2], dSymm);
                            if (dMisOri[k] >= dHAGB) dMisOri[k] =dHAGB;	
                            
                            /* 
                             * Add this misorientation and create kernel average
                             * misorientation by normalising to 
                             * unode<->neighbour distance:
                             * --> The closest neighbour has 100% contribution
                             * to average misorientation, other have less
                             * contribution
                             * --> The same principle applies to summing up
                             * the distances, but they have been loaded prior to
                             * this part, since the won't change with trial
                             * rotations
                             */				
                            dTotMisOri += dMisOri[k]*dNbOri[k][3];
                        }
                        
                        /*
                         * Get dislocation energy (J/m3) from kernel average 
                         * misorientation at this state:
                         * --> Function aneeds access to elle mineral database
                         */
                        dEdisloc = FS_GetDDEnergy(dTotMisOri,dTotDist,iNumNbs);

                        if (iTrial==0) 
                        {
                            /*
                            * if iTrial == 0, no rotation is performed,
                            * we are only saving the initial state
                            */                                
                            dMinE=dEdisloc;
                            dMisoriBefore = dTotMisOri/(double)iNumNbs;
                            dIniMisori = dMisoriBefore;
                            dOriChange = 0.0;
                        }
                        else 
                        {
                            /* 
                             * Check if the new calculated energy of the area 
                             * is less than the original - if so then replace 
                             * the orientation with the euler angles of the 
                             * trial orientation - this will go through
                             * each of the trials and the final ori will be the 
                             * lowest energy configuration
                             */
                            if (dEdisloc<dMinE && dTotMisOri/(double)iNumNbs < dHAGB)
                            {				                    
                                double dRotationRate = 0.0, dRotation = 0.0;
                                double dDrivingStress = 0.0;
                                double dStressExpN = 3.0;
                                double dTimestep = ElleTimestep();
                                double dAnisoRotVisc = // adjust to slip system..at the moment doesn't change the rot. visc.
                                    FS_MakeAniso(iTrial-1,rot_mobL); 
                                    
                                /* 
                                 * Get the mismatch between the current and the 
                                 * previous orientation of THIS unode (this is not
                                 * the misorientation with respect to neighbours,
                                 * but the change in unode orientation for
                                 * which we determined the change in energy)
                                 * 
                                 * -> Don't be confused: dCurrOri is at this stage
                                 * actually the previous orientation of the unode
                                 */
                                dOriChange = CME_hex(dCurrOri[0],dCurrOri[1],dCurrOri[2],
                                    dNewOri[0],dNewOri[1],dNewOri[2], dSymm);
                                dOriChange*=D2R;
                                            
                                /*
                                * Calculation of rotation rate assuming the 
                                * rotation of the small crystallite can be
                                * described by simple shearing it by a 
                                * non-linear viscous flow law:
                                * 
                                * strain_rate = (1/VISC) * stress^(n)
                                * 
                                * where n is 3 in this code
                                */             
                                double ddMis = fabs(dMisoriBefore-dTotMisOri/(double)iNumNbs);
                                ddMis*=D2R;    // before used ddMis instead of dOriChange, but that is actually not correct
                                                                                                          
                                dDrivingStress = 
                                    ((dMinE-dEdisloc)/dOriChange);
                                
                                if (dAnisoRotVisc!=0) 
                                    dAnisoRotVisc = 1.0/dAnisoRotVisc;
                                
                                /* Calculate rotation rate*/
                                dRotationRate = (dAnisoRotVisc)*
                                        pow(dDrivingStress,dStressExpN);                                  
                                    
                                dRotation = dRotationRate*dTimestep;                                
                                
                                /* Convert back to degrees */
                                dRotation*=R2D;
                                
                                /*
                                * FS (!): Limit the rotation to NOT MORE 
                                * THAN:
                                * --> average misorientation between Nbs
                                * --> definetely not more than 1°
                                */
                                // FS (!) DEBUGGING 
                                if (FSDEBUG) ElleSetUnodeAttribute(iUnodeID,1.0,U_ATTRIB_A);

                                if (dRotation > dTotMisOri/(double)iNumNbs)
                                {
                                    dRotation= dTotMisOri/(double)iNumNbs;
                                    iCounter++;
                                    // FS (!) DEBUGGING 
                                    if (FSDEBUG) ElleSetUnodeAttribute(iUnodeID,2.0,U_ATTRIB_A);
                                }
                                if (dRotation > 2.0) 
                                {
                                    dRotation= 2.0;
                                    iCounter++;
                                    // FS (!) DEBUGGING 
                                    if (FSDEBUG) ElleSetUnodeAttribute(iUnodeID,3.0,U_ATTRIB_A);
                                }
                                // FS (!) DEBUGGING 
                                if (FSDEBUG)
                                {
                                    if (iUnodeID==iFSDEBUGUNODE)
                                    {
                                        printf("Information on unode %u rotation mode (trial-1) %u:\n\n",iUnodeID,iTrial-1);
                                        printf("Energy mismatch:          %e J/m3\n",dMinE-dEdisloc);
                                        printf("Misorientation now:       %e °\n",dTotMisOri/(double)iNumNbs);
                                        printf("Misorientation before:    %e °\n",dMisoriBefore);
                                        printf("Misorientation initial:   %e °\n",dIniMisori);
                                        printf("Orientation mismatch:     %e°\n",dOriChange*R2D);
                                        printf("Rotation rate:            %e °/s\n",dRotationRate*R2D);
                                        printf("Rotation:                 %e °\n",dRotation);
                                        // --> Actually we are already inputting an effective viscosity!?
                                        //printf("Effective rotation visc.: %e Pas\n",1.0/(dAnisoRotVisc * pow( dRotationRate, (1.0/dStressExpN)-1 ) ) );
                                        //printf("Effective rotation visc.: %e Pas\n",(1.0/dAnisoRotVisc) * pow( dRotationRate, (1.0/dStressExpN)-1 ) );
                                    }
                                }

                                iError=FS_rot_matrix(iTrial-1,dRmap,dRotation);

                                /* error if rotating didn't work */	  	
                                if (iError!=0)
                                OnError("rotating failed",0);	
                                /* 
                                * multiply the current ori with the rotation matrix then put
                                * into dNewOri
                                */
                                matvecmultalb(dRmap,dCurrOri,dNewOri);

                                // Recalculate misori
                                dTotMisOri=0.0;					
                                for (int k=0; k<iNumNbs;k++)
                                {
                                    dMisOri[k]=CME_hex(dNbOri[k][0],dNbOri[k][1],dNbOri[k][2],
                                    dNewOri[0],dNewOri[1],dNewOri[2], dSymm);
                                    if (dMisOri[k] >= dHAGB) 
                                        dMisOri[k] =dHAGB;
                                    dTotMisOri+= dMisOri[k]*dNbOri[k][4];
                                }
                                
                                /*
                                 * Recalculate energy
                                 * Get dislocation energy (J/m3) from kernel 
                                 * average misorientation at this state:
                                 * --> Function aneeds access to elle mineral 
                                 *  database
                                 */
                                double dEdislocTest=0.0;
                                dEdislocTest = FS_GetDDEnergy(dTotMisOri,dTotDist,iNumNbs);	
                                
                                /* Double check if new energy is still 
                                 * lower than the previous one: it was when
                                 * rotating around the trial rotation, but we 
                                 * need to make sure that we are not "shooting"
                                 * over the minimum energy possible with the
                                 * calculated rotation. This might happen if 
                                 * Elle timestep is too fast */		
                                if (dEdisloc<dMinE)
                                {
                                    /* Okay, the rotation really resulted in a 
                                     * energy decrease, set new orientations
                                     * and new min energy*/
                                    // Set new energies
                                    dEdisloc=dEdislocTest;
                                    dMinE=dEdisloc;
                                    dMisoriBefore = dTotMisOri/(double)iNumNbs;
                                    
                                    // Set new orientations
                                    dCurrOri[0]=dNewOri[0];
                                    dCurrOri[1]=dNewOri[1];
                                    dCurrOri[2]=dNewOri[2];

                                    dEulerNew[0]=dNewOri[0];
                                    dEulerNew[1]=dNewOri[1];
                                    dEulerNew[2]=dNewOri[2];
                                    
                                    // Logging
                                    dAvRot += dRotation;
                                    iCountTotalRotations++;
                                    iCountRotations ++;
                                }
                                else
                                {
                                    /* Ooops, the rotation was actually too high
                                     * "shooting" over the minimum energy and
                                     * actually INcreasing the energy again,
                                     * this was due to the node moving too fast
                                     * --> For the moment: Do not set new
                                     * mean energy and euler angles
                                     * i.e.: Do not do the rotation
                                     * --> actually this is a little incorrect
                                     * since we have here a rather fast rotation
                                     * and the node wants to quickly move to the
                                     * minimum...this only does not matter, if 
                                     * we try to avoid this case as good as
                                     * possible by using a suitable Elle 
                                     * timestep and/or rotation viscosity
                                     */
                                    
                                }
                                
                                /* Set new values for energy and misori*/ 
                                
                            } // end of if (dEdisloc<dMinE && dTotMisOri/(double)iNumNbs < dHAGB)                            
                        } // end of "else-part" of: if(Trial==) {} else { }                        
                    } // End of for loop going through trial rotations
                    // FS (!) DEBUGGING 
                    if (FSDEBUG) ElleSetUnodeAttribute(iUnodeID,(double)iCountRotations,U_ATTRIB_C);
                    if (iCounter>0)  iCountTooFast++;                    
                    if (iCountRotations>0) iUnodesRotatedTotal++;
                    
                    /* Set the new orientation attributes and convert back to 
                     * degrees */
                    ElleSetUnodeAttribute(iUnodeID,dEulerNew[0]*R2D,E3_ALPHA);
                    ElleSetUnodeAttribute(iUnodeID,dEulerNew[1]*R2D,E3_BETA);
                    ElleSetUnodeAttribute(iUnodeID,dEulerNew[2]*R2D,E3_GAMMA);

                    dFinalMisOri=0.0;
                    for (int k=0; k<iNumNbs;k++)
                    {				
                        dMisOri[k]=CME_hex(dNbOri[k][0],dNbOri[k][1],dNbOri[k][2],dEulerNew[0],dEulerNew[1],dEulerNew[2], dSymm);
                        if (dMisOri[k] >= dHAGB) 
                            dMisOri[k] =dHAGB;

                        dFinalMisOri +=dMisOri[k]*dNbOri[k][3];					
                    }

                    /*
                     * Set new misorientation: It shouldn't be higher than the 
                     * HAGB angle 
                     */
                    dFinalMisOri /= (double)iNumNbs;
                    if (dFinalMisOri >= dHAGB) dFinalMisOri=dHAGB; 
                    ElleSetUnodeAttribute(iUnodeID, dFinalMisOri, U_ATTRIB_F);
                    
                    /* 
                     * From new misorientation set new dislocation density
                     */
                    double dNewDD = FS_Misori2DD(dFinalMisOri,dTotDist/(double)iNumNbs);
                    ElleSetUnodeAttribute(iUnodeID,dNewDD,U_DISLOCDEN);
                    
                    cNbUnodeXY.clear();                   
                } // end of if number of neighbous >0 (iNumNbs!=0)
                else
                {
                    // FS (!) DEBUGGING 
                    if (FSDEBUG) ElleSetUnodeAttribute(iUnodeID,0.0,U_ATTRIB_C);
                }
                
                // FS (!) DEBUGGING 
                if (FSDEBUG) ElleSetUnodeAttribute(iUnodeID,(double)iNumNbs,U_ATTRIB_B);
                    
                /* free the matrix */
                if (dNbOri!=0) 	{free_dmatrix(dNbOri,0,iNumNbs,0,5);dNbOri=0;}	
            } // end of looping through unodes in flynn
        } // end of looping through flynns
        
        dAvRot /= (double)iCountTotalRotations;
        double dAvRotRate = dAvRot/ElleTimestep();
        
		printf("\n# # # # # # # # # # # # # # # RESULT OVERVIEW # # # # # # # # # # # # # # # #\n\n");
        printf("Total number of unodes:                 %u\n",ElleMaxUnodes());
        printf("Number (percentage) of unodes rotated:  ");
        printf("%u (%f %%)\n",iUnodesRotatedTotal,((double)iUnodesRotatedTotal/(double)ElleMaxUnodes())*100.0);
        printf("Average rotation (rate):                %e ° (%e °/s)\n",dAvRot,dAvRotRate);
        printf("Number (percentage) of too fast unodes: ");
        printf("%u (%f %%)\n",iCountTooFast,((double)iCountTooFast/(double)iUnodesRotatedTotal)*100.0);
		printf("\n# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #\n\n");
        
  		ElleUpdate();	// checks whether it needs to write out a file or stats for this stage
    } // End of looping through all Elle stages
	 
} // END OF "FS_doannealing" function

int FS_norm_sep(Coords jxy, vector<Coords> &nbxy, double **nbori, int numnbs)
{
    /* 
     * FS: Nearly the same function than in old recovery code, but nbori has 1 more
     * column here, that stores the actual unode separation, not the normalised
     * one
     */
	
	/* 
     * This function checks the separation of the nbnodes to the central unode -
	 * makes sure that only nearest nbs are selected and factors in a diagonals
	 */
	 
	Coords  temp;
	int		i;
	int		error=0;
	double	mindist=0;
	
	for (i=0;i<numnbs;i++)
	{
	
		temp=nbxy[i];
	
		ElleCoordsPlotXY(&temp,&jxy);
	
		/* gets the xy positions for the unodes of interest */
	
		nbori[i][4]=pointSeparation(&temp,&jxy);
		
		/* an array of the separation between each points */
				
		if (i==0)
			mindist=nbori[i][4];
		
		else if (nbori[i][4]<mindist)
			mindist = nbori[i][4];
	}	
	
	if (mindist==0) 
    {
        error = 1;
        OnError("norm_sep=0",0);
    }
	
	/* error if the separation is equal to zero*/
	
	for (i=0;i<numnbs;i++) nbori[i][3]=mindist/nbori[i][4];			
	
	return(error);
} 

double FS_GetDDEnergy(double dSumMisoris,double dSumDists,int iNumNbs)
{
    double dStoredE = 0.0;
    double dKAM = 0.0;
    double dMeanDist = 0.0;
    
    /* 
     * Determine kernel average misorientation and distance
     * --> dSumMisoris and dSumDists are actually not only the sum of
     * misorientations and distances to all neighbours, but orientation and 
     * distances are weighted to the unode<->neighbour distance. Closest
     * neighbour will get weighting factor of 1, all other < 1
     */
    if (iNumNbs!=0) // should always be the case, has been checked earlier...
    {
        dKAM = dSumMisoris/(double)iNumNbs;
        dMeanDist = dSumDists/(double)iNumNbs;
    }
    else
    {
        /* This should actually never happen since the number of neighbours was
         * checked before, this is just to prevent the code from crashing:
         */
        printf("Error (line %u): No neighbour unodes\n",__LINE__);
        dMeanDist = 1.0;                        
    }
    
    double dDislocden = FS_Misori2DD(dKAM,dMeanDist);
        
    dStoredE = dDDLineEnergy*dDislocden; 

    return (dStoredE);
}

double FS_Misori2DD(double dKAM, double dMeanDist)//, double dBurgersVec)
{
    /* FS:  
     * Using the kernel average misorientation (KAM, in radians!!) to determine the 
     * corresponding dislocation density using:
     * 
     * --> IDEA: Ashby (1970), Borthwick et al. (2013): 
     * dd=KAM/(b*x)
     * 
     * where b is the length of the BUrgers vector and x is the mean distance of
     * all unodes in the kernel to unode of interest. 
     * Actually this is then depending on resolution.
     * 
     * INPUT: 
     * The kernel average misorientation (mean of weighted sum)
     * Mean of weighted sum of all distances from neighbours to this unode
     * The length of the Burgers vector of the material of interest
     */
    
    double dDislocden = 0.0;
    dMeanDist *= ElleUnitLength();
    dKAM*=D2R;

    dDislocden = dKAM/(dBurgersVec*dMeanDist);
    
    return (dDislocden);    
}

int FS_rot_matrix_axis(double dRotAngle, double rmap[3][3], double axis[3])
{
    /* 
     * FS: changed naming to avoid errors and confusion:
     * theta is a global variable in recovery code, the naming within this
     * function should therefore be different to avoid such confusion and
     * errors!!
     */

//	double dRotAngle = 0.1*D2R; //radians
	int error=0, i;
    double norm=0;


// axis is a unit vector 	

	for (i=0;i<3;i++) norm += axis[i]*axis[i];
	norm=sqrt(norm);
	
	for (i=0;i<3;i++) axis[i] = axis[i]/norm;
	
// rotation matrix, counterclockwise sense
	
	rmap[0][0]=axis[0]*axis[0]+(1-axis[0]*axis[0])*cos(dRotAngle);
	rmap[0][1]=axis[0]*axis[1]*(1-cos(dRotAngle))-axis[2]*sin(dRotAngle);
	rmap[0][2]=axis[0]*axis[2]*(1-cos(dRotAngle))+axis[1]*sin(dRotAngle);
	rmap[1][0]=axis[1]*axis[0]*(1-cos(dRotAngle))+axis[2]*sin(dRotAngle);
	rmap[1][1]=axis[1]*axis[1]+(1-axis[1]*axis[1])*cos(dRotAngle);
	rmap[1][2]=axis[1]*axis[2]*(1-cos(dRotAngle))-axis[0]*sin(dRotAngle);
	rmap[2][0]=axis[2]*axis[0]*(1-cos(dRotAngle))-axis[1]*sin(dRotAngle);
	rmap[2][1]=axis[2]*axis[1]*(1-cos(dRotAngle))+axis[0]*sin(dRotAngle);
	rmap[2][2]=axis[2]*axis[2]+(1-axis[2]*axis[2])*cos(dRotAngle);
	
	return (error);
	
}

int FS_rot_matrix(int t, double rmap[3][3], double dRotAngle)
{
    /* 
     * FS: changed naming to avoid errors and confusion:
     * theta is a global variable in recovery code, the naming within this
     * function should therefore be different to avoid such confusion and
     * errors!!
     */

	int error=0, i,j;	
	double rmap1[3][3], axis[3];	
    dRotAngle = dRotAngle*D2R; //radians
	
	switch(t)
	{
	/* dRotAngle is the amount of rotation rather than put misori in -
	 * so - need to calc amount wrt misori etc.	
	 */
	 
	// dRotAngle = f(disloc type, T) - need to define properly
	 
	// clockwise around (100)
	
	case 4: rmap[0][0]=1;
			rmap[0][1]=0;
			rmap[0][2]=0;
			rmap[1][0]=0;
			rmap[1][1]=cos(dRotAngle);
			rmap[1][2]=-sin(dRotAngle);
			rmap[2][0]=0;
			rmap[2][1]=sin(dRotAngle);
			rmap[2][2]=cos(dRotAngle);
			
	break;
			
	// anti-clockwise around (100)
			
	case 5: dRotAngle=dRotAngle*-1;
			rmap[0][0]=1;
			rmap[0][1]=0;
			rmap[0][2]=0;
			rmap[1][0]=0;
			rmap[1][1]=cos(dRotAngle);
			rmap[1][2]=-sin(dRotAngle);
			rmap[2][0]=0;
			rmap[2][1]=sin(dRotAngle);
			rmap[2][2]=cos(dRotAngle);
	
	break;
	
	//clockwise about (010)
			
	case 0: rmap[0][0]=cos(dRotAngle);
			rmap[0][1]=0;
			rmap[0][2]=sin(dRotAngle);
			rmap[1][0]=0;
			rmap[1][1]=1;
			rmap[1][2]=0;
			rmap[2][0]=-sin(dRotAngle);
			rmap[2][1]=0;
			rmap[2][2]=cos(dRotAngle);
	
	break;
	
	//anti-clockwise about (010)
			
	case 1: dRotAngle=dRotAngle*-1;
		    rmap[0][0]=cos(dRotAngle);
			rmap[0][1]=0;
			rmap[0][2]=sin(dRotAngle);
			rmap[1][0]=0;
			rmap[1][1]=1;
			rmap[1][2]=0;
			rmap[2][0]=-sin(dRotAngle);
			rmap[2][1]=0;
			rmap[2][2]=cos(dRotAngle);

	break;

//clockwise about (001) // FS: around c-axis in ice
			
	case 2: rmap[0][0]=cos(dRotAngle);
			rmap[0][1]=sin(dRotAngle);
			rmap[0][2]=0;
			rmap[1][0]=-sin(dRotAngle);
			rmap[1][1]=cos(dRotAngle);
			rmap[1][2]=0;
			rmap[2][0]=0;
			rmap[2][1]=0;
			rmap[2][2]=1;
	
	break;
	
	//anti-clockwise about (001) // FS: around c-axis in ice
			
	case 3: dRotAngle=dRotAngle*-1;
			rmap[0][0]=cos(dRotAngle);
			rmap[0][1]=sin(dRotAngle);
			rmap[0][2]=0;
			rmap[1][0]=-sin(dRotAngle);
			rmap[1][1]=cos(dRotAngle);
			rmap[1][2]=0;
			rmap[2][0]=0;
			rmap[2][1]=0;
			rmap[2][2]=1;

	break;

	//clockwise about (111) etc.	
	case 6:
		axis[0]=1;
		axis[1]=1;
		axis[2]=1;	
		error=FS_rot_matrix_axis( dRotAngle, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;
			
	case 7:dRotAngle=dRotAngle*-1;
		axis[0]=1;
		axis[1]=1;
		axis[2]=1;	
		error=FS_rot_matrix_axis( dRotAngle, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;

	case 8:
		axis[0]=-1;
		axis[1]=1;
		axis[2]=1;	
		error=FS_rot_matrix_axis( dRotAngle, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;
			
	case 9:dRotAngle=dRotAngle*-1;
		axis[0]=-1;
		axis[1]=1;
		axis[2]=1;	
		error=FS_rot_matrix_axis( dRotAngle, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;

	case 10:
		axis[0]=1;
		axis[1]=-1;
		axis[2]=1;	
		error=FS_rot_matrix_axis( dRotAngle, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;
			
	case 11:dRotAngle=dRotAngle*-1;
		axis[0]=1;
		axis[1]=-1;
		axis[2]=1;	
		error=FS_rot_matrix_axis( dRotAngle, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;

	case 12:
		axis[0]=-1;
		axis[1]=-1;
		axis[2]=1;	
		error=FS_rot_matrix_axis( dRotAngle, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;
			
	case 13:dRotAngle=dRotAngle*-1;
		axis[0]=-1;
		axis[1]=-1;
		axis[2]=1;	
		error=FS_rot_matrix_axis( dRotAngle, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;			
	
	default: error=1;
	
	break;
	}
	
	return (error);
}

double FS_MakeAniso(int iRotationMode,double dInVisc)
{
    /* 
     * This is a tool to adjust the rotation viscosity as a function of the
     * axis of rotation (cf. mode of rotation in FS_rot_matrix function)
     * 
     * ATTENTION: Instead of this function we actually need a more general 
     * description of how rotation viscosity related to different active slip 
     * sysmtes in different materials
     */
     
    /* THIS IS FOR ICE: If rotation is around any other axis than the c-axis,
     * rotation viscosity will be lowered to approximate anisotropy
     */
    double dFACTOR=20.0;
    double dSlipSystRotVisc = dInVisc;
    if (iRotationMode!=2 && iRotationMode!=3) dSlipSystRotVisc*=dFACTOR;
    
    return(dSlipSystRotVisc);    
}

void symmetry(double symm[24][3][3])
{

    int i,n; 
	FILE *f=fopen("symmetry.symm","r");	
	double dum1,dum2,dum3;
    int ifscanf_return = 0; // FS (!) added this to remove warnings
	
	// first line of with the information of number of symmetry operation 
	ifscanf_return = fscanf(f,"%d",&symm_op);

	// store symmetry operators
     for(n=0;n<symm_op;n++) {
		 
	 	for (i=0;i<3;i++) {
			ifscanf_return = fscanf(f,"%lf %lf %lf",&dum1,&dum2,&dum3);	
			symm[n][i][0]=dum1;
			symm[n][i][1]=dum2;		
			symm[n][i][2]=dum3;
					
	 	}	
	}	

     fclose(f);		
}

double CME_hex(double curra1,double currb1,double currc1,double curra2,double currb2,double currc2, double symm[8][3][3])
{   
    double tmpA, tmpB, angle;
    double val, val2;
    double eps=1e-6;

    double rmap1[3][3];
    double rmap2[3][3];
    double rmapA[3][3],rmapAA[3][3];
    double a11, a22,a33;

	int n,i,j;
	double misang,minang, misang2;	
	double rmap1A[3][3],rmap2A[3][3],rmap1AA[3][3],rmap2AA[3][3];
	double aux_symm[3][3];
	
    //curra1 *= D2R; currb1 *= D2R; currc1 *= D2R;
    //curra2 *= D2R; currb2 *= D2R; currc2 *= D2R;

	orientmat2(rmap1, (double)curra1,(double)currb1,(double)currc1);
	orientmat2(rmap2, (double)curra2,(double)currb2,(double)currc2);
    // euler(rmap1,(double)curra1,(double)currb1,(double)currc1);// gives rotation matrix
    // euler(rmap2,(double)curra2,(double)currb2,(double)currc2);// gives rotation matrix

	// Ini symmetry operators... 
	minang=1000;
	
    for(n=0;n<symm_op;n++) {

		
		// auxiliar symmetry matrix [3][3] 
		for (i=0;i<3;i++) {
			for (j=0;j<3;j++) {
			
			aux_symm[i][j] = symm[n][i][j]; 
  		 		}				
			}		
	
// 1st	
    //calculation of tmpA where the inverse of rmap2 is taken for calculation
    matinverse(rmap2,rmap2A); 		
    matmult(rmap1,rmap2A,rmap1A);

	// symmetry operators			
	matmult(aux_symm,rmap1A,rmapA);	

// 2nd	
    //calculation of tmpAA where the inverse of rmap1 is taken for calculation
    matinverse(rmap1,rmap1A); 		
    matmult(rmap2,rmap1A,rmap2A);

	// symmetry operators			
	matmult(aux_symm,rmap2A,rmapAA);	

	// Take trace.. and misorientation angle	 
    a11=rmapA[0][0];
    a22=rmapA[1][1];
    a33=rmapA[2][2];
   
    val = (a11+a22+a33-1)/2;

    a11=rmapAA[0][0];
    a22=rmapAA[1][1];
    a33=rmapAA[2][2];

    val2 = (a11+a22+a33-1)/2;	
		
    if (val>1.0) val = 1.0;
    else if (val<-1.0) val = -1.0;
    if (val2>1.0) val2 = 1.0;
    else if (val2<-1.0) val2 = -1.0;
		
    misang=(acos(val));
	misang *= R2D;
    misang2=(acos(val2));
	misang2 *= R2D;

	if (misang2 < misang) {
		misang=misang2;	
		}
	
	if (misang < minang) {
		minang=misang;	
		}
	
 }
 
 // FS: I CHANGED THE FOLLOWING IF LOOP: IT SAID minang > 120 BEFORE!!!!)
 if (minang > 180 ) minang=180-minang; 

return(minang);
   
}


void orientmat2(double a[3][3], double phi, double rho, double g)
{
    //double a[3][3];
    int i,j;

    a[0][0]=cos(phi)*cos(g)-sin(phi)*sin(g)*cos(rho);
    a[0][1]=sin(phi)*cos(g)+cos(phi)*sin(g)*cos(rho);
    a[0][2]=sin(g)*sin(rho);
    a[1][0]=-cos(phi)*sin(g)-sin(phi)*cos(g)*cos(rho);
    // a[1][1]=sin(phi)*sin(g)+cos(phi)*cos(g)*cos(rho);
    a[1][1]=-sin(phi)*sin(g)+cos(phi)*cos(g)*cos(rho);
    a[1][2]=cos(g)*sin(rho);
    a[2][0]=sin(phi)*sin(rho);
    a[2][1]=-cos(phi)*sin(rho);
    a[2][2]=cos(rho);
}

void matvecmultalb (double rmap[3][3], double currentori[3], double newori[3]) 
{
	
	double rmapnew1[3][3];
	double rmapnew2[3][3];			
	double eps=1.0e-10;
	double signPhi=-1.0;
	
	orientmat2(rmapnew1, currentori[0],currentori[1],currentori[2]);
	matmult(rmap,rmapnew1,rmapnew2);
					
	if (rmapnew2[2][1]<0)
		signPhi=1.0;

	/*
	 * acos returns 0 -> PI
	 * asin, atan return -PI/2 -> PI/2
	 * atan2 returns -PI -> PI
	 */
    newori[1]= acos(rmapnew2[2][2])*signPhi;
	if (newori[1] <0 ) newori[1] *= -1;
	
    if (fabs(newori[1]) > eps) {
        newori[0]= -1*atan2(rmapnew2[2][0], rmapnew2[2][1]);
    	newori[2]= atan(rmapnew2[0][2]/ rmapnew2[1][2]);
    }
    else {
	// AVOID DIV BY 0
    // *Phi is 0 or PI and cos(*Phi)=1
		// cannot calc *phi2
		// can only calculate *a if we assume *g==0
      newori[0] = 0.0;
      newori[2] = asin(rmapnew2[1][0]);
    }
	
	// compare with current orientation 


	if (fabs(newori[0]-currentori[0]) > 5*D2R){ 	
 		newori[0] += 180*D2R;
		if (newori[0] > PI) newori[0] -=360*D2R;    
		}
				
	if (fabs(newori[1]-currentori[1]) > 5*D2R){ 	
 		newori[1] += 180*D2R;
		if (newori[1] > PI) newori[1] -=360*D2R; 
		if (newori[1] < 0)  newori[1] -=360*D2R;    
		}

	if (fabs(newori[2]-currentori[2]) > 15*D2R){ 	
 		newori[2] += 180*D2R;
		if (newori[2] > PI) newori[2] -=360*D2R;    
		}

}
