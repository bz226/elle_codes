#include "FS_header_recovery.h"

double theta=0.1; // FS (!): For trial rotations ??
double rot_mobL=500;//1483.275412;	 // M*l/A¿?, rotation mobility * boundary length / area cristallite ¿?--
double HAGB;
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
    ElleSetRunFunction(doannealing);

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

int doannealing()
{

	int 	loops,i,ii,iUnodeID,k,l,iTrial;
	int 	algorithm;
	int 	err;
	int		nb_nodes[8], max_unodes;	
	int		type, T;
	int		error=0;
	int  	iNumberOfTrials=6;
	int		numnbs=0;
	int		bndcount;
	//double	burger=0; // FS (!): Commented this, it wasn't used 
	double  newE, dislocE, minE;
	double  e1,e2,e3,k1,k2,k3,m1,m2,m3,t1,t2,t3;
	double  symm[24][3][3];
	double	misori[8], totmisori, bndmisori, avmisori, albmisori;
	double  currentori[3], newori[3];
	double	**nbori=0;
	double	rmap[3][3];
	Coords  jxy;
	//double theta2; // FS (!): Commented this, it wasn't used 
    
    UserData userdata;
	ElleUserData(userdata);

    // albert 
    int flynnid, count,kk, max_flynns, total, flynnok;
	double averotation;
    char psfile[20], outfile[10];
    
    // Florian (!)
        int iCountTotalRotations = 0; // counting total number of rotations to determine average rotation in the end
        bool FSDEBUG = false;
        if(FSDEBUG) printf("line: %u Debugging switched on in doannealing\n",__LINE__);
        if (theta!=0.1) printf("ATTENTION: theta is %f != 0.1\n",theta);
        int iExcludedPhase = (int)userdata[1]; 
        int iNoRandomisation = (int)userdata[2];   
        double dFlynnPhase = 0;
        int iPhaseAttrib = VISCOSITY;
        int iUserInStartStep = (int)userdata[4];
        if ( iUserInStartStep > 1 )
            Settings_run.Count = iUserInStartStep;
        
        int iDEBUG_Unode = -15083;
        
        // FS (!) For my debugging: Timing a bit of code:
            /*
            #include <ctime>
            clock_t begin = clock();
            doSomething();
            clock_t end = clock();
            double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );
            */
        clock_t timebegin,timeend;
        double dTimeSec = 0;

    HAGB=userdata[0]; 
	flynnok=1; //int(userdata[1]); // FS (!) WHAT IS "flynnok" ?????
    
	rot_mobL=userdata[3]; // FS (!) calculate that internally using a function or so
	
	//sprintf(psfile,"%d.flynn",flynnok); // FS (!) commeted that out to have new naming convention
    sprintf(psfile,"recovery"); // FS (!) new naming convention, atm more helpful for me
	sprintf(outfile,"%d.unodes",flynnok);

    ElleSetSaveFileRoot(psfile);
    /* 
     *choose which algorithm to be set to
     */
    //algorithm = (int)userdata[uALG];
   	
   	/*
   	 *either commandline or GUI - be careful of zip files!
   	 */

    /* AVOID TO USE ElleVoronoiUnodes() in sparse models! 
        err=ElleVoronoiUnodes();
        if (!ElleVoronoiActive() || err)
        OnError("Voronoi failed", err);
    */	

    // Init flynns and unodes attributes 

    if (!ElleUnodeAttributeActive(ATTRIB_F))
        ElleInitUnodeAttribute(U_ATTRIB_F);
	
    vector <int> ran, unoderan;
    vector <int> nbnodes;
    
    ran.clear();
 
    max_unodes=ElleMaxUnodes();
    max_flynns = ElleMaxFlynns();
    
    symmetry(symm);
   
    /*
     * FS (!): Store all IDs of active flynns in a vector. Furthermore:
     * Only use flynns that are not of the phase that was excluded by the user
     */
	for (i=0;i<max_flynns;i++) 
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
                    ran.push_back(i);
                }     
            }
            else
            {
                ran.push_back(i);
            }
        }
	}
    
    /*
     * FS (!): Give user message if randomisation has been switched off:
     */
    if (iNoRandomisation)
        printf("ATTENTION: Random picking of flynns and "
                "unodes has been switched off!\n");    
    
    // FS write logfile:
    //fstream fRecoveryLogfile;
    //fRecoveryLogfile.open("Logfile_Rotation.txt",fstream::out | fstream::app);
    //printf("Writing a Logfile for Rotation Rates\n");
    //fRecoveryLogfile << "Stage, Total number of unodes rotated, average rotation, average rotation rate, unode rotating too fast" << endl;

    /*
     * FS (!): 
     * IMPORTANT: NOW START LOOPING THROUGH ALL STEPS:
     * "loop" contains the step number
     */    
    for(loops=0;loops<EllemaxStages();loops++)
    {
		total=0;
		averotation=0;		
        iCountTotalRotations = 0;
		//printf("Stage %d %d \n", loops, flynnok);
		printf("\nPolyphase Recovery by Subgrain Rotation - Stage: %d\n\n",loops+iUserInStartStep);
 		
        /*
         * FS (!): Randomize all flynn IDs if user did not switch off this 
         * option by setting userdata[2] to 1 (0 is default):
         */
        if (!iNoRandomisation)
            random_shuffle(ran.begin(), ran.end());
    
        /* setup an array of random unodes - ran[i] */	
        int iCountTooFast=0; // track how many unodes rotate too fast
    		
        /*
         * FS (!): Loop through all flynns in "ran" and only use unodes in the 
         * relevant flynn for calculations
         */
    	for (i=0; i<ran.size(); i++) // FS (!) closes at the end of the ElleStages loop
    	{
            unoderan.clear();
            flynnid = ran[i];
			
            /*
             * FS (!): Tirangulate unodes: Neccessary to later find the 
             * neighbours of the unode 
             */
            ElleClearTriAttributes();
            TriangulateUnodes(flynnid,MeshData.tri);

            /* FS (!): Find all unodes in this flynn: */
            ElleGetFlynnUnodeList(flynnid,unoderan);
            count=unoderan.size();
			
            if (count > 0 ) // FS (!) closes ~ at the end of the ElleStages loop 
            {                           
                /*
                 * FS (!): Only randomly shuffle unodes if user did not switch
                 * this option off by setting userdata[2] to 1 (which is 0 by 
                 * default)
                 */
                if (!iNoRandomisation) 
                    random_shuffle(unoderan.begin(), unoderan.end());	
	
                /* FS (!): Loop through all unodes in the relevant flynn "i" */
                for (kk=0; kk<count; kk++) // FS (!) closes at the end of the ElleStages loop
                {
                    iUnodeID = unoderan[kk];	
                    vector<int> nbnodes,bndflag;				
                    nbnodes.clear();
    		
                    /* get attributes for and position of each unode */
                    ElleGetUnodeAttribute(iUnodeID, &e1, E3_ALPHA);
                    ElleGetUnodeAttribute(iUnodeID, &e2, E3_BETA);
                    ElleGetUnodeAttribute(iUnodeID, &e3, E3_GAMMA);

                    ElleGetUnodePosition(iUnodeID,&jxy);
				
                    /* current orientation array - convert to radians */
                    currentori[0]=e1*DTOR;
                    currentori[1]=e2*DTOR;
                    currentori[2]=e3*DTOR;		

                    /* use unodes triangulation to find neighbours unodes*/	
                    ElleGetTriPtNeighbours(iUnodeID,nbnodes,bndflag,0); //get the  list of neighbours for a unode --> 0 is to not include unodes on the boundary --> check basecode
                    numnbs = nbnodes.size();

                    if(numnbs!=0) // FS (!): If the number of neighbours is > 0
                    {			
                        vector <Coords> nbxy(numnbs);
                        
                        /* setup a matrix to hold the euler angle info plus num of nbs*/   			
                        if (nbori==0) nbori=dmatrix(0,numnbs,0,4);
                        
                        /*
                         * FS (!): nbori stores orientation of the "k-th" 
                         * neighbour. Number of neighbours is thereofre = k
                         * i.e.: k = nbnodes.size();
                         * Neighbour-id Euler-alpha Euler-beta Euler-gamma  Distance to center unode (iUnodeID)
                         *              nbori[k][0]   nbori[][1] nbori[][2] nbori[][3]
                         *      0           53          50          176         0.9
                         *      1           ..          ..          ..          ..
                         *      2           ..          ..          ..          ..
                         *      ..          ..          ..          ..          ..
                         *      k-1         ..          ..          ..          ..
                         *      k           ..          ..          ..          ..
                         */
                
                        /* get info for each of nb nodes and put into matrix*/    		
                        for (k=0; k<nbnodes.size();k++)
                        {	
                            ElleGetUnodeAttribute(nbnodes[k], &k1, E3_ALPHA);
                            ElleGetUnodeAttribute(nbnodes[k], &k2, E3_BETA);
                            ElleGetUnodeAttribute(nbnodes[k], &k3, E3_GAMMA);
                            
                            ElleGetUnodePosition(nbnodes[k],&nbxy[k]);
                            
                            nbori[k][0]=k1*DTOR;
                            nbori[k][1]=k2*DTOR;
                            nbori[k][2]=k3*DTOR;
                        }
                
                        /* 
                         * check the separation is not too high - err if nbnode 
                         * is too far away
                         * 
                         * FS (!): 
                         * jxy: Coords of unode of interest
                         * nbxy: Coords of all neighbours
                         * nbori: Matrix containing orientations for all neighbours
                         * numnbs: Number of neighbours
                         *
                         * nbori[k][0 until 2] contains 3 euler angles
                         * nbori[k][3] will after normsep contain the correct 
                         * distance to center unode
                         */
                        err=norm_sep(jxy,nbxy,nbori,numnbs);
                         
                        /*
                        * FS (!):
                        * nbori[k][3] will afterwards contain distance from 
                        * neighbour to center unode NORMALIZED TO NEAREST
                        * NEIGHBOUR:
                        * IF nbori[k][n] = 1 --> nearest (direct) neighbour
                        * IF nbori[k][n] = .71 -> diagonal neighbour
                        * Every value between 0 and 1: Some intermediate 
                        * distance: Possible if the unode grid is slightly 
                        * distored due to FFT code --> hence distance can also 
                        * get smaller than 0.71
                        */                        
                        
                        /* 
                         * put currentori into newori in case there is no 
                         * change - ori stays the same
                         * 
                         * FS (!): newori and currentori are defined as:
                         * double newori[3], currentori[3]
                         */
                        t1=currentori[0];
                        t2=currentori[1];
                        t3=currentori[2];
                        
                        newori[0]=currentori[0];
                        newori[1]=currentori[1];
                        newori[2]=currentori[2];                

                        /* go through rotation trials to find lowest E config */  
                        int iCounterRotations = 0; // helps counting how many unodes are rotated at all
                        int iCounter = 0; // helps counting how many unodes rotate too fast		 
                        for (iTrial=0; iTrial<iNumberOfTrials+1; iTrial++)
                        {
                            totmisori=0;
                            avmisori=0;
                            bndcount=0;
                            albmisori=0;
    
                            /* FS (!): if iTrial == 0 use currentori as newori,
                             * otherwise perform all the trial rotations with 
                             * the trial angle theta 
                             */
                            if (iTrial>0)
                            {
                                /* FS (!) Create rotation matrix for the 
                                 * specific rotation
                                 */
                                error=rot_matrix(iTrial-1,rmap, theta);
                                
                                /* error if rotating didn't work */	  	
                                if (error!=0)
                                    OnError("rotating failed",0);	                            
                                /* 
                                 * multiply the current ori with the rotation matrix then put
                                 * into newori
                                 */
                                matvecmultalb(rmap,currentori,newori);
                            }    
                
                            for (k=0; k<nbnodes.size();k++)
                            {
                                /* 
                                 * use Albert's misorient code to find the misorientation btw
                                 * central unode and nb unodes
                                 */
                            
                                // why give a value higher than fundamental orientation zone ?¿ 
                            
                                misori[k]=CME_hex(nbori[k][0],nbori[k][1],nbori[k][2],
                                newori[0],newori[1],newori[2], symm);
                                if (misori[k] >= HAGB)	misori[k] =HAGB;					
                                totmisori += misori[k];		
                            
                                // printf("%d %d %lf %lf %lf\n", iUnodeID,t, nbori[k][0]*RTOD,nbori[k][1]*RTOD,nbori[k][2]*RTOD);	
                                
                                /* add up all the misorientations in the area to use for the
                                 * temporary dummy calculation
                                 */
             
                                /* check if there are any boundaries more than 0.5degs - if
                                 * so add this to the boundary count to show that these need
                                 * to be treated differently - a boundary energy needs to
                                 * be removed from the energy calculation
                                 */
                                
                                /* !! Torque calculation (Albert's suggestion) - At this point 
                                 *  we could check the lengths of the boundaries - we already 
                                 *  have the misori angle
                                 * Could also work out the rotational mobility at this point --> 
                                 * need C and also to decide if lattice diffusion or GB diffusion
                                 */
                                 
                                /* FS (!): Commeted the following if-loop:
                                 * That does actually never happen, since it is 
                                 * checked before
                                 *
                                if (misori[k]>HAGB) 
                                {
                                    bndcount++;
                                    bndmisori += misori[k];
                                    printf("NOW WE'RE IN LINE: %u\n",__LINE__);
                                }   
                                */ 	
                            } 
                    
                            /* call dislocation density calculation to return an energy
                             * !! If we want to add in the torque calculation it will be in
                               dldense!!	
                             */
                    
                            dislocE = dldense(newori, nbori, numnbs, totmisori);	  			
                            //burger=0.5; //temp burgers vector - get from JW == projection of Nye's tensor as a vector in the plane 
                            
                            // FS(!): "burger" will only be used in boundE, what I commeted out below :-)
                            
                            /* FS (!): Commeted the following if-loop:
                                 * That does actually never happen, since it is 
                                 * checked before if misori[k]>HAGB, so bndcount 
                                 * will always be == 0
                                 * --> SO NOW THE FUNCTION boundE IS NOT NEEDED
                                 *     ANY MORE
                            if (bndcount>0)
                            {			
                                avmisori=totmisori/bndcount;
                                dislocE = dislocE - boundE(burger, avmisori);
                                // FS (!): with the settings, at the moment boundE will always return 0
                                
                                //if there are boundaries - calc boundE and substract from
                                //tot energy - later will link to frontracking model to 
                                //move boundaries
                            }
                            */
                            
                            if (iTrial==0) 
                            {
                                /*
                                 * FS (!): 
                                 * For first trial run set minimum energy 
                                 * (which is the energy without rotation):
                                 */                                
                                minE=dislocE;
                                
                            }
                            else 
                            {
                                /* check if the new calculated energy of the area is less than
                                 * the original - if so then replace the orientation with the
                                 * euler angles of the trial orientation - this will go through
                                 * each of the trials and the final ori will be the lowest energy
                                 * configuration
                                 */
                                if (dislocE<minE && totmisori/numnbs < HAGB)
                                {
                                    //printf("iUnodeID %d iTrial %d minE %lf dislocE %lf\n", iUnodeID,iTrial,minE, dislocE); 								
                                    /* rotation_mobility, constant boundary length
                                     * value calculated from maximum differential misorientation 
                                     * per step 0.5 when reduction from 0.5 to 0 
                                     * ARTIFITIAL..to improve */
                            
                                    // double rot_mobL=5.0;	 // M*l/A¿?, rotation mobility * boundary length / area cristallite ¿?-- 					                    
                                    double rotation, dummy;
                                    
                                    /* FS (!): Print some values to understand what 
                                     * is happening here:
                                     */
                                    if(FSDEBUG && iUnodeID==iDEBUG_Unode) 
                                    {
                                        printf("DATA FOR UNODE: %u\n",iUnodeID);
                                        
                                        printf("minE: %f\ndislocE: %f\ntheta: %f\n",minE,dislocE,theta);
                                        printf("--> dummy: %f\n",(minE-dislocE)*theta);
                                        printf("--> rot_mobL:\t%f\n",rot_mobL);
                                        printf("--> rotation:\t%f\n\n",((minE-dislocE)*theta)*rot_mobL);
                                    }
                                    
                                    /*
                                     * FS (!): 
                                     * Calculation of the rotation in degrees
                                     * using dummy, net-node energy (minE-
                                     * dislocE) and theta (in degrees).
                                     */
                                    
                                    dummy=(minE-dislocE)*theta;
                                    rotation = rot_mobL * dummy; 

                                    /*
                                     * FS (!): Limit the rotation to NOT MORE 
                                     * THAN:
                                     * --> average misorientation between Nbs
                                     * --> definetely not more than 2°
                                     */
                                    if (rotation > totmisori/nbnodes.size())
                                    {
                                        if (FSDEBUG) printf("Limiting the rotation to %f°\n",totmisori/nbnodes.size());
                                        rotation= totmisori/nbnodes.size();
                                        iCounter++;
                                    }
                                    if (rotation > 20*theta) 
                                    {
                                        if (FSDEBUG) printf("Limiting the rotation to %.0f°\n",20*theta);
                                        rotation= 20*theta;
                                        //rotation= totmisori/nbnodes.size();
                                        iCounter++;
                                    }
                                                                
                                    averotation += 	rotation;
                                    iCounterRotations++;
                                    
                                    error=rot_matrix(iTrial-1,rmap, rotation);
                            
                                    /* error if rotating didn't work */	  	
                                        if (error!=0)
                                            OnError("rotating failed",0);	
                                    /* 
                                     * multiply the current ori with the rotation matrix then put
                                     * into newori
                                     */
                            
                                     matvecmultalb(rmap,currentori,newori);

                                    /* 
                                     * FS (!): WHY CHANGE currentori HERE?? 
                                     * DOESN'T IT NEED TO STAY IN INITIAL 
                                     * ORIENTATION FOR FOLLOWING TRIAL 
                                     * ROTATIONS???
                                     * 
                                     * ANSWER:
                                     * NO, chaning currentori here is correct:
                                     * With this trial rotation, a rotation 
                                     * around a specific crystalligraphic axis 
                                     * was performed, the remaining trials only 
                                     * check if there is another rotation 
                                     * around a different crystallographic axis 
                                     * that lowers the energy even more.
                                     * --> For that purpose after each 
                                     * "incremental" rotation around one axis,
                                     * the resulting orientation of the unode 
                                     * between two "increments" has to be  
                                     * stored somehow...
                                     */
                                    currentori[0]=newori[0];
                                    currentori[1]=newori[1];
                                    currentori[2]=newori[2];
                                       
                                    t1=newori[0];
                                    t2=newori[1];
                                    t3=newori[2];

                                    // recalculate misori and energy!!
                                    albmisori=0.0;					
                                    for (k=0; k<nbnodes.size();k++)
                                    {
                                        misori[k]=CME_hex(nbori[k][0],nbori[k][1],nbori[k][2],
                                        newori[0],newori[1],newori[2], symm);
                                        if (misori[k] >= HAGB)	misori[k] =HAGB;
                                        albmisori+= misori[k];
                                    }
                        
                                    albmisori=albmisori/nbnodes.size();
                                    dislocE = dldense(newori, nbori, numnbs, albmisori);
                        
                                    // set new values 					
                                    minE=dislocE;
                                }
                            }
                        } // FS (!): End of for loop going through trial rotations
                        if (iCounter>0) 
                        {
                            iCountTooFast++;
                            //ElleSetUnodeAttribute(iUnodeID,1.0,U_ATTRIB_A);
                        }
                        //else ElleSetUnodeAttribute(iUnodeID,0.0,U_ATTRIB_A);
                        if(iCounterRotations>0) 
                        {
                            total++;
                            //ElleSetUnodeAttribute(iUnodeID,1.0,U_ATTRIB_B);
                        }
                        iCountTotalRotations+=iCounterRotations;
                        //else ElleSetUnodeAttribute(iUnodeID,0.0,U_ATTRIB_B);
                            
                        /* how much to rotate by?*/
                
                        // need a glide eqn then at higher T climb
                
                        /* this will be a function of the temp which has been
                         * put in with userdata (so prolly need to have something
                         * in there about amount of rotation) 
                         * also dislocation type - from dominant b vector+rot matrix
                         * time step here or earlier?
                         */
                        
                        /* next code is here to test particular rotations - check if working
                         * properly or not
                         */
                        /*			
                            if (iUnodeID==0)
                            {
                            
                            rot_matrix(0,rmap);
                            
                            matvecmult(rmap, currentori, newori);
                            
                            t1=newori[0];
                            t2=newori[1];
                            t3=newori[2];
                            
                            }
                        */			
                        //ends here

                        /* set the new orientation attributes and convert back to degs */
            
                        ElleSetUnodeAttribute(iUnodeID, t1*RTOD, E3_ALPHA);
                        ElleSetUnodeAttribute(iUnodeID, t2*RTOD, E3_BETA);
                        ElleSetUnodeAttribute(iUnodeID, t3*RTOD, E3_GAMMA);

                        albmisori=0;
                        for (k=0; k<nbnodes.size();k++)
                        {				
                            misori[k]=CME_hex(nbori[k][0],nbori[k][1],nbori[k][2],t1,t2,t3, symm);
                            if (misori[k] >= HAGB)	misori[k] =HAGB;
                            
                            albmisori +=misori[k];					
                        }
                                
                        albmisori=albmisori/nbnodes.size();
                        if (albmisori >= HAGB) albmisori=HAGB; // FS (!): i.e.: Misorientation shouldn't be higher than the critical HAGB
                        double dummy, density;
                
                        if (loops != 0) // FS (!): Do not do that in first model step?!
                        {

                            ElleGetUnodeAttribute(iUnodeID,U_ATTRIB_F,&dummy);
                            if(dummy >= HAGB) dummy=HAGB; 
                            ElleSetUnodeAttribute(iUnodeID, albmisori, U_ATTRIB_F);
                                                
                            ElleGetUnodeAttribute(iUnodeID,U_DISLOCDEN,&density);			
                            if (dummy <= 1e-1) dummy =1e-1;
                              
                            density =fabs(density*albmisori/dummy); // reduction of DDs proportional to reduction of <misori>
                        
                            ElleSetUnodeAttribute(iUnodeID, density, U_DISLOCDEN);					
                        } 
                        else // FS (!): In every other model step than the first one
                        {
                            ElleSetUnodeAttribute(iUnodeID, albmisori, U_ATTRIB_F);	
                        }	
                
                        /* set the new orientation attributes and convert back to degs */

                    }		
                    //else
                    //{
                        ///* If unode has no neighbours, the unode network or the 
                         //* flynn boundary itselfis probably quite distorted 
                         //* indicating a high amount of strain energy:
                         //* 
                         //* Reduce dislocation density by dummy value
                         //*/
                        //double dDis=0.0;
                        //double dDummyDDReductionFact = 0.1; // DD is multiplied with this number to achieve reduction
                        //ElleGetUnodeAttribute(iUnodeID,U_DISLOCDEN,&dDis);
                        //dDis=dDis-(dDis*dDummyDDReductionFact);
                        //ElleSetUnodeAttribute(iUnodeID,dDis,U_DISLOCDEN);
                    //}
                    if (nbori!=0) 	{free_dmatrix(nbori,0,numnbs,0,4);nbori=0;}	
    		
                    /* free the matrix */
                }
            }
 // } // end if flynnid
        }
        
        averotation /= (double)iCountTotalRotations;
        double averotation_rate = averotation/ElleTimestep();
        
		printf("\n# # # RESULT OVERVIEW: # # #\n");
        printf("Total number of unodes:\n%u\n",ElleMaxUnodes());
        printf("Number (percentage) of unodes rotated:\n");
        printf("%u (%f %%)\n",total,((double)total/(double)ElleMaxUnodes())*100.0);
        printf("Average rotation (rate):\n%e ° (%e °/s)\n",averotation,averotation_rate);
        printf("Number (percentage) of unodes rotating too fast:\n");
        printf("%u (%f %%)\n",iCountTooFast,((double)iCountTooFast/(double)total)*100.0);
        
        //fRecoveryLogfile << loops+iUserInStartStep << " " << total << " " << averotation << " " << averotation_rate << " " << iCountTooFast << endl;	
        
  		ElleUpdate();	// checks whether it needs to write out a file or stats for this stage
    } // FS (!): End of looping through all Elle stages
	//fRecoveryLogfile.close();
///*
  //ofstream outf(outfile);
  //vector<int> unodelist;	
//for (i=0; i<ran.size(); i++) 
    	//{
	//unodelist.clear();

        //ElleGetFlynnUnodeList(ran[i],unodelist);
        //count=unodelist.size();
						
    		//for (j=0; j<count; j++){	

    			//ElleGetUnodeAttribute(unodelist[j], &k1, E3_ALPHA);
    			//ElleGetUnodeAttribute(unodelist[j], &k2, E3_BETA);
    			//ElleGetUnodeAttribute(unodelist[j], &k3, E3_GAMMA);
      			//ElleGetUnodeAttribute(unodelist[j], &albmisori, U_ATTRIB_F);  			

				//outf << unodelist[j] << '\t' 
					  //<< k1 << '\t' 
					  //<< k2 << '\t' 
					  //<< k3 << '\t' 
					  //<< albmisori <<endl;
			//}	
	//}
    //outf.close();	
//*/	
 
} // FS (!) END OF "doannealing" function
	  
int norm_sep(Coords jxy, vector<Coords> &nbxy, double **nbori, int numnbs)
{
	
	/* this function checks the separation of the nbnodes to the central unode -
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
	
		nbori[i][3]=pointSeparation(&temp,&jxy);
		
		/* an array of the separation between each points */
				
		if (i==0)
			mindist=nbori[i][3];
		
		else if (nbori[i][3]<mindist)
			mindist = nbori[i][3];
	}	
	
	if (mindist==0) 
    {
        error = 1;
        OnError("norm_sep=0",0);
    }
	
	/* error if the separation is equal to zero*/
	
	for (i=0;i<numnbs;i++)
	{
		nbori[i][3]=mindist/nbori[i][3];			
	
		/* factors in amount for diagonals - values should be either 1 for near
		 * nbs or 0.71 for diagonals
		 */
	
		// printf("%lf ", nbori[i][3]);	
	}	
	
	// printf("\n");
	
	return(error);

}
double dldense(double currentori[3], double **nbori, int numnbs, double totmisori)
{
    double disdens = 0.0; // FS (!) unused
    double energy = 0.0; 
    //double HAGB=0.5; // FS (!): Commented that out --> why not use HAGB as defined by user??
    
    //double dHAGB_energy = 0.065; // FS (!): Why not implement the HAGB energy for the material, here: ICE???
	totmisori /= numnbs;
    
    /*
     * FS (!): This looked a little hard to understand, the if loops below do 
     * the same, but are arranged so I will understand them more easily:
     */
    //if (totmisori == 0 ) energy=0.0;
    //else if ( totmisori <= HAGB) energy=(totmisori/HAGB)*(1-log(totmisori/HAGB)); 
	//else if ( totmisori > HAGB) energy=1.0;
    
    /*
     * FS (!): This code is only valid if HAGB energy = 1, in any other case an 
     * additional multiplication of "energy" with the HAGB energy is needed 
     * (cf. Read Shockley equation e.g. in Borthwick et al. 2013)
     * --> cf. the code of this if-loops:
     */        
    if (totmisori == 0 ) 
    {
        energy=0.0;
    }
    else 
    {
        if ( totmisori <= HAGB) 
        {
            //energy=(totmisori/HAGB)*(1-log(totmisori/HAGB)); // FS (!): Read Shockley equation, but this is for surface energies ?!
            energy=(totmisori/HAGB)/(1-log(totmisori/HAGB)); // FS (!): According to Borthwick et al. 2013 the equation for energy=1 should look like this!!
        }
        else 
        {
            if ( totmisori > HAGB) 
            {
                energy=1.0;
            }
        }
    }
    
    /* FS (!): Mutiply with HAGB energy for a correct value*/    
    double dHAGB_energy = 1;//0.065; // FS (!): Why not implement the HAGB energy for the material, here: ICE=0.065 J/m²???
    energy *= dHAGB_energy;
	
	// disdens=totmisori;
	
	
	/* here need to work out dislocation density using Albert's code - at the moment 
	 * the code uses misorientation as a
	 * vague approx of dislocation density and therefore energy
	 */
	 
	/* !!If going to use the torque calculation for dummy rotations could do it here
	 * --> each of the trial rotations will call to this function to calculate the
	 * energy and then check which new orientation has the largest energy reduction
	 * --> so already the framework is here, we would just need to use a different
	 * calculation!! 
	 */	
	// will return density or Energy - if err in calc return 0 or call onerror to get out
	return(energy);
}

double boundE(double b, double theta)
{
	double mu=0; //surface energy
	double d=0; // need to have input of Burger's vectors (Wheeler)
	double b0=0; // how get this?
	double K=0; // gas constant?? find out
	double bndE=0;

	 // make dummy values for now 
	b0=0.3;
	d=1;
	mu=10; //how get SurfE?
	K=5.0; //how get constant?
	//area=need to ask Lynn some way to calc based on no. unodes in dldense 
	//d=initE/area;

	// Read-Shockley Equation
	// bndE = (((mu * b)/(4*M_PI*K))*theta) * (log(b)/b0 - log(theta));
	return(bndE);
	
	/* need to make an average dislocation spacing 
	 * - prolly calc over the area of choice and then
	 * - take an average
	 */   	
	 
	/* in order to get b - pick nearest dominant rotation
	 * axis and pick dom. b vector at that pt
	 */
	  
	/* also will need to calc surfaceE - tilt/twist
	 * need to check if we can get that from Albert's 
	 * code
	 */
}

int rot_matrix_axis(double theta, double rmap[3][3], double axis[3])
{

//	double theta = 0.1*DTOR; //radians
	int error=0, i;
    double norm=0;


// axis is a unit vector 	

	for (i=0;i<3;i++) norm += axis[i]*axis[i];
	norm=sqrt(norm);
	
	for (i=0;i<3;i++) axis[i] = axis[i]/norm;
	
// rotation matrix, counterclockwise sense
	
	rmap[0][0]=axis[0]*axis[0]+(1-axis[0]*axis[0])*cos(theta);
	rmap[0][1]=axis[0]*axis[1]*(1-cos(theta))-axis[2]*sin(theta);
	rmap[0][2]=axis[0]*axis[2]*(1-cos(theta))+axis[1]*sin(theta);
	rmap[1][0]=axis[1]*axis[0]*(1-cos(theta))+axis[2]*sin(theta);
	rmap[1][1]=axis[1]*axis[1]+(1-axis[1]*axis[1])*cos(theta);
	rmap[1][2]=axis[1]*axis[2]*(1-cos(theta))-axis[0]*sin(theta);
	rmap[2][0]=axis[2]*axis[0]*(1-cos(theta))-axis[1]*sin(theta);
	rmap[2][1]=axis[2]*axis[1]*(1-cos(theta))+axis[0]*sin(theta);
	rmap[2][2]=axis[2]*axis[2]+(1-axis[2]*axis[2])*cos(theta);
	
	return (error);
	
}

int rot_matrix(int t, double rmap[3][3], double theta)
{


	int error=0, i,j;	
	double rmap1[3][3], axis[3];	
    theta = theta*DTOR; //radians
	
	switch(t)
	{
	/* theta is the amount of rotation rather than put misori in -
	 * so - need to calc amount wrt misori etc.	
	 */
	 
	// theta = f(disloc type, T) - need to define properly
	 
	// clockwise around (100)
	
	case 4: rmap[0][0]=1;
			rmap[0][1]=0;
			rmap[0][2]=0;
			rmap[1][0]=0;
			rmap[1][1]=cos(theta);
			rmap[1][2]=-sin(theta);
			rmap[2][0]=0;
			rmap[2][1]=sin(theta);
			rmap[2][2]=cos(theta);
			
	break;
			
	// anti-clockwise around (100)
			
	case 5: theta=theta*-1;
			rmap[0][0]=1;
			rmap[0][1]=0;
			rmap[0][2]=0;
			rmap[1][0]=0;
			rmap[1][1]=cos(theta);
			rmap[1][2]=-sin(theta);
			rmap[2][0]=0;
			rmap[2][1]=sin(theta);
			rmap[2][2]=cos(theta);
	
	break;
	
	//clockwise about (010)
			
	case 0: rmap[0][0]=cos(theta);
			rmap[0][1]=0;
			rmap[0][2]=sin(theta);
			rmap[1][0]=0;
			rmap[1][1]=1;
			rmap[1][2]=0;
			rmap[2][0]=-sin(theta);
			rmap[2][1]=0;
			rmap[2][2]=cos(theta);
	
	break;
	
	//anti-clockwise about (010)
			
	case 1: theta=theta*-1;
		    rmap[0][0]=cos(theta);
			rmap[0][1]=0;
			rmap[0][2]=sin(theta);
			rmap[1][0]=0;
			rmap[1][1]=1;
			rmap[1][2]=0;
			rmap[2][0]=-sin(theta);
			rmap[2][1]=0;
			rmap[2][2]=cos(theta);

	break;

//clockwise about (001)
			
	case 2: rmap[0][0]=cos(theta);
			rmap[0][1]=sin(theta);
			rmap[0][2]=0;
			rmap[1][0]=-sin(theta);
			rmap[1][1]=cos(theta);
			rmap[1][2]=0;
			rmap[2][0]=0;
			rmap[2][1]=0;
			rmap[2][2]=1;
	
	break;
	
	//anti-clockwise about (001)
			
	case 3: theta=theta*-1;
			rmap[0][0]=cos(theta);
			rmap[0][1]=sin(theta);
			rmap[0][2]=0;
			rmap[1][0]=-sin(theta);
			rmap[1][1]=cos(theta);
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
		error=rot_matrix_axis( theta, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;
			
	case 7:theta=theta*-1;
		axis[0]=1;
		axis[1]=1;
		axis[2]=1;	
		error=rot_matrix_axis( theta, rmap1, axis);
	
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
		error=rot_matrix_axis( theta, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;
			
	case 9:theta=theta*-1;
		axis[0]=-1;
		axis[1]=1;
		axis[2]=1;	
		error=rot_matrix_axis( theta, rmap1, axis);
	
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
		error=rot_matrix_axis( theta, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;
			
	case 11:theta=theta*-1;
		axis[0]=1;
		axis[1]=-1;
		axis[2]=1;	
		error=rot_matrix_axis( theta, rmap1, axis);
	
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
		error=rot_matrix_axis( theta, rmap1, axis);
	
		for (j=0;j<3;j++){
				for (i=0;i<3;i++){
					rmap[i][j] = rmap1[i][j];
				}
			}
	break;
			
	case 13:theta=theta*-1;
		axis[0]=-1;
		axis[1]=-1;
		axis[2]=1;	
		error=rot_matrix_axis( theta, rmap1, axis);
	
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
	
    //curra1 *= DTOR; currb1 *= DTOR; currc1 *= DTOR;
    //curra2 *= DTOR; currb2 *= DTOR; currc2 *= DTOR;

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
	misang *= RTOD;
    misang2=(acos(val2));
	misang2 *= RTOD;

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


	if (fabs(newori[0]-currentori[0]) > 5*DTOR){ 	
 		newori[0] += 180*DTOR;
		if (newori[0] > PI) newori[0] -=360*DTOR;    
		}
				
	if (fabs(newori[1]-currentori[1]) > 5*DTOR){ 	
 		newori[1] += 180*DTOR;
		if (newori[1] > PI) newori[1] -=360*DTOR; 
		if (newori[1] < 0)  newori[1] -=360*DTOR;    
		}

	if (fabs(newori[2]-currentori[2]) > 15*DTOR){ 	
 		newori[2] += 180*DTOR;
		if (newori[2] > PI) newori[2] -=360*DTOR;    
		}

}
