#include "FS_header_gbm_pp_fft.elle.h"
#include "FS_topocheck.h"

/*
 * Declaration of shared variables of gbm_pp_fft.elle.cc and 
 * gbm_pp_fft.clusters.cc:
 */
AllPhases phases;
UserData userdata;
extern runtime_opts Settings_run;

int iFlynnPhase       = VISCOSITY;  // Storage of phases
int iFlynnCluster     = F_ATTRIB_B;
int iFlynnNumber      = F_ATTRIB_C; // Storage of FFT unode update variables
int iUnodeFlynnNumber = U_ATTRIB_C;
bool bClusterCheck    = 1;          // Are clusters present?
///JR: Alberts variables
int iUserInDiagonal=0;  // 0 use 4 dummy nodes <default>, 1 use diagonal positions 
double dUserInAfactor=0; // 0 non redistribution of DD inside flynn <default>, 1 full-automatic redistribution 

/* FS: For ice, you can scale line energies to non-basal activity 
 * (basal activity needs to be stored in U_ATTRIB_D). Non basal dislocation have
 * up to 3.65 times higher energies, to take this into account, set the 
 * following variable to "true"*/
bool bScaleSIBMToNonBasalAct=true;

/* The decreasing factor of intrinsic mobility in the middle third of the
 * model box*/
double dModRedFact = 0.0;

double dDummyDD = 2e12; // FS: Used if no unode is closer than ROI, assuming a very distored grain

// FS: For my changes in cluster tracking:
double dBoxArea = 0.0; 
/* FS: To count how many nodes are too fast and what would have been the mean 
 * highest possible timestep: */
double dTimeStepHighest = 0.0;
int iNodeTooFast = 0;
fstream fDataFile;
	
list<int> lFicksDiffPhases;

int InitGrowth()
{
    int err = 0;
    int i, jj;
    char *infile;
    char *dbfile;
    
    ElleReinit();
    ElleSetRunFunction( GBMGrowth );
    
    infile = ElleFile();
    if ( strlen( infile ) > 0 )
    {
        if ( err = ElleReadData( infile ) )
            OnError( infile, err );
        ElleAddDoubles();
    }

// Initialize flynns and unode attributes

///JR used by Albert's part of the code for internal energy things.
	if (!ElleUnodeAttributeActive(U_DISLOCDEN))
    ElleInitUnodeAttribute(U_DISLOCDEN);
	if (!ElleUnodeAttributeActive(iUnodeFlynnNumber))
    ElleInitUnodeAttribute(iUnodeFlynnNumber);
	if (!ElleFlynnAttributeActive(iFlynnNumber))
	ElleInitFlynnAttribute(iFlynnNumber);
///JR necessary for the cluster tracking + phase number.	
	if (!ElleFlynnAttributeActive(iFlynnPhase))
	ElleInitFlynnAttribute(iFlynnPhase); 
	if (!ElleFlynnAttributeActive(iFlynnCluster))
	ElleInitFlynnAttribute(iFlynnCluster); 
			
    /* FS: Old stuff assigning flynn ID to unode: But this is incorrect 
     * sometimes if it has not been checked by the process that was performed
     * before. Therefore I created a new function doing the job called
     * FS_check_unodes_in_flynn */
    //for (i=0;i<ElleMaxUnodes();i++)
    //{
	    //jj=ElleUnodeFlynn(i);
	    //ElleSetUnodeAttribute(i,iUnodeFlynnNumber, double(jj));
    //}

    //for (i=0;i<ElleMaxFlynns();i++) 
	//{
    	//if (ElleFlynnIsActive(i)) 
		//{
			//ElleSetFlynnRealAttribute(i,double(i),iFlynnNumber);
        //}
	//}
    FS_check_unodes_in_flynn(); // may be redundant with unode checks embedded in topology checks

///JR initialise userdata
	ElleUserData(userdata);
	
	// If there is an error with the database file
    if(!Read2PhaseDb(dbfile, &phases)) //dbfileed
    { 
    	printf("Seg fault from OnError function???\n");
    	OnError( "dbfile", 0 );
    }
}

int Read2PhaseDb(char *dbfile, AllPhases *phases)
{
	fstream file;
	string line;
	stringstream linestr;
	int no_phases = 0, first_phase = 0, comb, count, p1, p2, cluster, diff_times, x, p_en, p_track, disscale;
	double mob, en, dif, kappa, dGbActEn, stored, disbondscale;
	int input=0;
    int iMinTjs=0; // FS!: Not used any more
	bool bCheckVersion = 0;
	int plot=1; // enable plotting to command line.
	char c;
	// This functions reads the config file to storage...

	file.open ("phase_db.txt", fstream::in );

	if (file.is_open()) 
    {
		while (file.good()) 
        {
			getline (file,line);
			if (line.length() > 0) 
            {
				// check for file version
				// if you change some thing to the pharsing here, also change the file version. Otherwise 
				// all the different db file versions might geht mixed up and pharsing breaks...
				if ( line.find("v4.1") == string::npos  && bCheckVersion == 0) 
                {
					cout << "DB FILE DOES NOT HAVE THE RIGHT VERSION!!!" << endl;
					return (0);
				} else 
                {
					bCheckVersion = 1;
				}
				// check for keywords
				if ( line.find("PHASE PROPERTIES") != string::npos )
					input = 1;
				else if ( line.find("PHASE BOUNDARY PROPERTIES") != string::npos )
					input = 2;
				else if ( line.find("MELT TRACKING") != string::npos )
					input = 3;
				else if ( line.find("VERBOSE STUFF") != string::npos )
					input = 4;
				else if ( line.find("CLUSTER_TRACKING") != string::npos )
					input = 5;
				else if ( line.find("TROUBLESHOOTING") != string::npos )
					input = 6;
				c = line.at(0);
				if (c!='#' && c!=' ') 
                {
					// Read number of phases
					if ( input == 0 ) 
                    {
						linestr << line;
						linestr >> first_phase >> no_phases;
						linestr.clear();
						if (no_phases > MAX_PHASES) 
                        {
							cerr << "More phases than this program can handle" << endl;
							return 0;
						}
						phases->first_phase = first_phase;
						phases->no_phases = no_phases;
					}
					// Read Phase properties
					else if ( input == 1 ) 
                    {
						linestr << line;
						linestr >> p1 >> cluster >> diff_times >> kappa >> stored >> disscale >> disbondscale;
						linestr.clear();

						phases->phasep[p1].cluster_diff=cluster;
						phases->phasep[p1].diffusion_times=diff_times;
						phases->phasep[p1].kappa=kappa;
						phases->phasep[p1].stored=stored;
						phases->phasep[p1].disscale=disscale;
						phases->phasep[p1].disbondscale=disbondscale;
					}
					// Read Phase Boundary properties
					else if ( input == 2 ) 
                    {
						linestr << line;
						linestr >> p1 >> p2 >> mob >> en >> dGbActEn;
						linestr.clear();

						phases->pairs[p1][p2].mobility=mob;
						phases->pairs[p2][p1].mobility=mob;
						phases->pairs[p1][p2].b_energy=en;
						phases->pairs[p2][p1].b_energy=en;
						phases->pairs[p1][p2].dGbActEn=dGbActEn;
						phases->pairs[p2][p1].dGbActEn=dGbActEn;
					}
					// Read Phase Tracking
					else if ( input == 3 ) 
                    {
						linestr << line;
						linestr >> p_track;
						linestr.clear();
						if ( p_track > no_phases - 1 + first_phase ) // wenn größer als max phases -> kein tracking (kleiner spielt keine Rolle)
							p_track = -1;
						phases->p_track=p_track;
					}
					// Read Verbouse
					else if ( input == 4 ) 
                    {
						linestr << line;
						linestr >> p_en;
						linestr.clear();

						phases->p_en=p_en;
					}
					// read Clustertracking
					// Read Troubleshooting
					else if ( input == 6 ) 
                    {
						linestr << line;
						linestr >> iMinTjs;
						linestr.clear();
					}
				}
			}
		}
		// Plot the whole stuff into command line window.
		if (plot == 1) 
        {
			cout << endl << "=================== INPUT FILE ===================" << endl
				<< "First Phase: " << phases->first_phase << endl
				<< "Number of phases: " << phases->no_phases << endl
				<< "==================== TIMESTEP ====================" << endl
				<< ElleTimestep() << " sec  <=>  " << ElleTimestep()/(60*60) << " h  <=>  "<< ElleTimestep()/(365*24*60*60) << " years" << endl
				<< "================== LENGTH SCALE ==================" << endl
				<< ElleUnitLength() << " m  <=>  " << ElleUnitLength()*1000 << " mm" << endl //"  <=>  "<< ElleTimestep()/(365*24*60*60) << " years" << endl
				<< "===================== PHASES =====================" << endl
				<< "Phase\tCluster\tOstwalt\tKappa\tDisloc-EN\tDisDenScal\tDisBondScale" << endl;
			for ( int i = first_phase; i < (first_phase + no_phases); i++ ) 
            {
				cout << i << "\t"
					<< phases->phasep[i].cluster_diff << "\t"
					<< phases->phasep[i].diffusion_times << "\t"
					<< phases->phasep[i].kappa << "\t"
					<< phases->phasep[i].stored << "\t\t"
					<< phases->phasep[i].disscale << "\t\t"
					<< phases->phasep[i].disbondscale
					<< endl;
			}
			cout << "================ PHASE BOUNDARIES ================" << endl
				<< "Phase1\tPhase2\tMobility\tB-energy\tGB-ActivEn" << endl;
			for ( int i = first_phase; i < (first_phase + no_phases); i++ ) 
            {
				for ( int j = i; j < (first_phase + no_phases); j++ ) 
                {
					cout << i << "\t"
						<< j << "\t"
						<< phases->pairs[i][j].mobility << "\t\t"
						<< phases->pairs[i][j].b_energy << "\t\t"
						<< phases->pairs[i][j].dGbActEn
						<< endl;
				}
			}
			if ( phases->p_track >= 0 ) 
            {
				cout
					<< "================= PHASE TRACKING =================" << endl
					<< "Track phase: " << phases->p_track << " in Unode layer"
					<< endl;
			}
			if ( phases->p_en > 0 ) 
            {
				cout
					<< "================ VERBOSE STUFF ================" << endl
					<< "Print energies for node: " << phases->p_en
					<< endl;
			}
			if ( iMinTjs > 2 ) 
            {
				cout
					<< "================ TROUBLESHOOTING ================" << endl
					<< "Min Tjs: " << iMinTjs << " for Triple switching"
					<< endl;
			}
		}
		file.close();
		if ( fileExists("initial_stuff.txt") ) 
        {
			cout << endl << "# # # # # #   !!! WARNING !!!   # # # # # #" << endl << "--> Initial file present!" << endl << "--> If you start at step 0 delete this file first!" << endl;
		}
		return (1);
	} 
    else
		return (0);
}

bool fileExists(const char *filename)
{
  ifstream ifile(filename);
  return ifile.good();
}

int GBMGrowth()
{
    bool bFSLogBnodes=false; // ONLY true FOR DEBUG, see what it does in code below...
	int iMaxNodes = ElleMaxNodes();
	int iUserInLogscreen = 0;
    int iUserInStartStep = 0;
	Coords newxy;
	vector <int> ran;
    
    iUserInDiagonal  = (int)userdata[0];
	dUserInAfactor   = userdata[1];
	iUserInLogscreen = (int)userdata[2];
    iUserInStartStep = (int)userdata[3];
    dModRedFact = userdata[5];
    
    /*
     * Before starting: Perform a full topology check:
     */
    TopologyChecks();
    
    /* FS: I changed the way the user is informed about nodes too fast:*/
    printf("ATTENTION:\nMessages about nodes moving too fast switched off, instead check the summary!\n");
	
	/*
     * The following is necessary for correct naming of output files in case 
     * simulation was started from a previous GBM-result that already modeled 
     * "iUserInStartStep"-steps
     */
	if ( iUserInStartStep > 1 )
		Settings_run.Count = iUserInStartStep;	
	
	if ( ElleCount() == 0 )
		ElleAddDoubles(); // to be sure
	if ( ElleDisplay() )
		EllePlotRegions( ElleCount() );
	
    /* FS:
     * ElleCheckFiles() can cause an Elle file being created with the step 
     * number == startstep --> overwriting of files can happen!!
     * --> ElleCheckFiles writes an Elle file if startstep!= 0 AND 
     * startstep % save interval==0
     * --> This is avoided by temporarily setting the step to 0 (i.e. count to 0)
     * --> However actually then there is no point in using ElleCheckFiles?!
     * --> FS SWITCHED IT OFF
     */ 
    //int iCountBeforeElleCheckFiles = ElleCount(); // store real number of steps
    //ElleSetCount(0); // set number of steps to 0 to avoid ElleCheckFiles writing Ellefiles
    //ElleCheckFiles(); 
    //// Important: Reset the correct number of steps:
    //ElleSetCount(iCountBeforeElleCheckFiles); 
    
    /*
     * Choose the correct trail position usage depending on user input. Either 
     * trial positions only in x and y directions from the node 
     * (iUserInDiagonal=0) or additionally diagonal from it (iUserInDiagonal=1)
     * can be used
     */
	if (iUserInDiagonal != 0) 
		printf("8-node version, afactor= %lf\n", dUserInAfactor);
	else
		printf("4-node version, afactor= %lf\n", dUserInAfactor);
        
    /*
     * Initialize the clusterTracking class by Jens 
     */
    /* FS: I did some changes: First we need to know the total box area:*/
    dBoxArea = FS_GetTotalBoxArea();
	clusterTracking clusters;
	if ( clusters.writeInitialData("initial_stuff.txt") ) 
    {
		clusters.setClusterAreas();
		clusters.checkDoubleClusterAreaLoop();
	}
        
    /*
     * 
     *  # # # # MAIN LOOP # # # #
     * 
     * Main- and most important loop going through all steps of GBM:
     */
    for (int i = 0; i < EllemaxStages(); i++ )
	{
		printf("\nPolyphase GBM - Stage: %d\n\n", Settings_run.Count);
		/*
		 * Write a backup output file to store in case of an error:
		 */		 
		//ElleWriteData("gbm_pp_fft_Backup.elle");

		//if ( !(i % 10) && bClusterCheck )
		if ( bClusterCheck )
			clusters.writeData( "PhaseAreaHistory.txt", i );
		
		iMaxNodes = ElleMaxNodes();
		
		/*
         * To prevent moving a single node always at the same time, we shuffel 
         * them randomly at each step
         */
		ran.clear();
		for (int j = 0; j < iMaxNodes; j++ )
			if ( ElleNodeIsActive( j ) )
				ran.push_back( j );

		/*
         * Alberts addition:
         * Shuffle deactivated with certain settings:
         */
		if (iUserInLogscreen != 1) std::random_shuffle( ran.begin(), ran.end() );
        else printf("WARNING: Random node shuffeling is switched off with logscreen set to %d\n",iUserInLogscreen);
        
        /* FS: ONLY USE FOR DEBUGGING
         * The following code is only used for debugging where we need to 
         * use a random order of bnodes:
         * It tracks the order of bnodes and stores it in a logfile. Whenever 
         * this logfile is present in a future step, this order of bnodes is
         * used
         */
        if(bFSLogBnodes)
        {
            char fBnodeOrderName[]="GBMBnodeOrder.txt";
            /* Use a "bnode" file that dictates the order of bnodes, if it is
             * existing */
            if (fileExists(fBnodeOrderName))
            {
                printf("Reading order of bnodes from file\n");
                ran.clear();
                ifstream datafile(fBnodeOrderName);
                if (!datafile) return(OPEN_ERR);

                int iBnodeValue;
                
                while (datafile) 
                {
                    datafile >> iBnodeValue;
                    ran.push_back(iBnodeValue);
                }
                datafile.close();
            }            
            else
            {
                /* If not, create such a file */
                printf("Writing order of bnodes to file\n");
                fstream fBnodeOrder;
                fBnodeOrder.open(fBnodeOrderName,fstream::out | fstream::trunc);
                for (int i=0;i<ran.size();i++)
                    fBnodeOrder << ran.at(i) << endl;
            }
        }
        
        /*
         * Loop through al bnodes (randomly if iUserInLogscreen == 0 )
         */
		for (int j = 0; j < ran.size(); j++ )
		{
            /*
             * FS: added this topology check which checks and solves problems
             * with coinciding nodes:
             */
            //ElleSetNodeChange(0);
            //FS_InitialTopoCheck(ran.at( j ));
            //if (ElleNodeChange() != 0) 
                //clusters.updateClusters(); 
            //ElleSetNodeChange(0);            
                                    
            /*
             * FS: Check if node is still active (it may have been deleted by
             * initial topology check:
             */
			if ( ElleNodeIsActive( ran.at( j ) ) )
			{
                //printf("Using bnode %u\n",ran.at(j));
                /*
                 * GGMoveNode_all(...) returns 0 or 1 depending on if the node 
                 * is moved at all or not. If yes, the new position is stored 
                 * in newxy. The variable "clusters" is used to determine area 
                 * energies.
                 */
                if (GGMoveNode_all( ran.at(j), &newxy, &clusters ) ) // i.e. if node will move
                {
                    /*
                     * Check if movement is not much too large (Shouldn't be 
                     * since this is already check in MoveD- and MoveTNode)
                     * FS!: Therefore, maybe delete this check
                     */
                    if (sqrt((newxy.x*newxy.x)+(newxy.y*newxy.y)) > ElleSwitchdistance() * ElleTimestep() * ElleSpeedup() / 1.5 )
                    {
						cout << "PROBLEM: Movement is very large... Reduce mobility, energy or time step settings..." << endl;
                    }
                                    
                    /*
                     * Finally really move the node to the new position and do
                     * topology checks always checking if node is really still
                     * active after calculations. Also update clusters after every
                     * possible change in node arrangements:
                     */
                    
                    // The movement and basic topology checks:
                    if (ElleNodeIsActive(ran.at(j))) // FS!: maybe not neccessary?
                    {
                        ElleSetNodeChange(0);
                        //ElleUpdatePosition( ran.at( j ), & newxy ); // caused errors, especially with ElleNodeTopologyCheck afterwards
                        FS_CrossingsCheck(ran[j],&newxy);
                        if (ElleNodeIsActive(ran.at(j)))
                            ElleDeleteSingleJ(ran.at(j));
                        if (ElleNodeChange() != 0)
                            clusters.updateClusters();                    
                    }
                    
                    if (ElleNodeIsActive(ran.at(j))) // FS!: maybe not neccessary?
                    {
                        if ( ElleNodeIsDouble( ran.at( j ) ) )
                        {
                            ElleSetNodeChange(0);
                            ElleCheckDoubleJ( ran.at( j ) );
                            /*
                             * FS: Probably still neccessary since adding or 
                             * removing dJs next to the node can also slightly 
                             * change areas:
                             */
                            if (ElleNodeChange() != 0)
                                clusters.updateClusters();
                        }
                        else if ( ElleNodeIsTriple( ran.at( j ) ) )
                        {
                            ElleSetNodeChange(0);
                            ElleCheckTripleJ( ran.at( j ) );
                            if (ElleNodeChange() != 0)
                                clusters.updateClusters();
                        }
                        else 
                        {
                            //FS: Will probably never happen:
                            cout << "ERROR: No known node type(2)... (Node: " << ran.at( j ) << ")" << endl;
                        }
                    }
                    
                    /*
                     * Do only a part of all the topo checks:
                     */
                    //printf("TopoChecks - Node %u of %u\n",j,(int)ran.size());
                    //ElleGGTopoChecks();
                    //CheckNodesCoincide();
                    //clusters.updateClusters();
                }

			}
		}
        
        /*
         * FS: FS_update_dislocden is my new version fixing one bug with 
         * seperated, but not deleted grains: Only neccessary if unodes are in 
         * the file
         * 
         * ATTENTION: NOT NECESSARY ANY MORE SINCE IT IS EMBEDDED IN TOPOCHECKS
         */
        //if (ElleUnodesActive()) 
        //{
            //FS_update_dislocden2();
            //clusters.updateClusters();
        //}
        
        /*
         * Afterwards again perform a full topology check:
         */
        TopologyChecks();
        clusters.updateClusters();
        
        // ATTENTION: NOT NECESSARY ANY MORE SINCE IT IS EMBEDDED IN TOPOCHECKS
        //if (ElleUnodesActive()) 
        //{
            //FS_update_dislocden2();
            //clusters.updateClusters();
        //}
    
        /*
         * Just to be sure...add double junctions if the gaps are too large:
         */
		ElleAddDoubles();

		ElleUpdate(); // store new elle file
        
        /* Final user messages and logfile about nodes being too fast */
        printf("Nodes too fast: %u of initially %u bnodes (%f %%)\n",iNodeTooFast,(int)ran.size(),( (double)iNodeTooFast/(double)ran.size() )*100.0);
        if (iNodeTooFast>0) printf("Average max. possible timestep: %e\n",dTimeStepHighest/(double)iNodeTooFast);
        else printf("Average max. possible timestep: --\n");
        fDataFile.open ( "Logfile_MoveNode.txt", fstream::out | fstream::app);
        fDataFile << "Stage " << Settings_run.Count << ": " << endl;
        fDataFile<< "Nodes too fast: " << iNodeTooFast << " of initially " << ran.size() << " bnodes" << endl;
        if (iNodeTooFast>0) fDataFile<< scientific << "Average max. possible timestep: " << dTimeStepHighest/(double)iNodeTooFast << endl;
        else fDataFile<< "Average max. possible timestep: --" << endl;
        fDataFile.close();
        // reset
        iNodeTooFast=0;
        dTimeStepHighest = 0.0;        
        
	} // MARKS THE END OF THE LOOP OF ONE STAGE OF GBM
}

double ReturnArea(ERegion poly, int node, Coords *pos)
{
     int j, *id=0, num_nodes;
     double area, *coordsx=0, *coordsy=0, *ptrx, *ptry;
     Coords xy,prev;
     list<int> nodes;
     //printf("x:%f\ty:%f\n", pos->x, pos->y);

     ElleFlynnNodes(poly,&id,&num_nodes);
     if ((coordsx = (double *)malloc(num_nodes*sizeof(double)))== 0) OnError("ElleRegionArea",MALLOC_ERR);
     if ((coordsy = (double *)malloc(num_nodes*sizeof(double)))== 0) OnError("ElleRegionArea",MALLOC_ERR);

     for (j=0;j<num_nodes;j++)
    	 nodes.push_back(id[j]);

     if (num_nodes != nodes.size())
    	 printf("ERROR: ReturnArea: sizes don't match\n");

     if (id) free(id);

     j=0;

     //just reordeer the list until the current node is at the front
     while (j==0) {
    	 if(nodes.front()==node)
    		 j=1;
    	 else {
    		 nodes.push_back(nodes.front());
    		 nodes.pop_front();
    	 }
     }

     //do the same stuff as ElleRegionArea except for the first "node"
     prev.x = pos->x;
     prev.y = pos->y;

     ptrx=coordsx;
     ptry=coordsy;
     while (nodes.size()>0) {
    	 if (nodes.front()==node){
    		 xy=*pos;
    		 ElleCoordsPlotXY(&xy, &prev);
			 *ptrx = xy.x; ptrx++;
			 *ptry = xy.y; ptry++;
			 nodes.pop_front();
    	 } else {
			 ElleNodePlotXY(nodes.front(),&xy,&prev);
			 *ptrx = xy.x; ptrx++;
			 *ptry = xy.y; ptry++;
			 prev = xy;
			 nodes.pop_front();
    	 }
     }
     area = polyArea(coordsx,coordsy,num_nodes);
     free(coordsx);
     free(coordsy);
     return(area);
}

/* 
 * This is called to when it is time to move a node (and also to check if a 
 * node has to be moved) 
 */
int GGMoveNode_all( int node, Coords *xy, clusterTracking *clusterData)
{
    double e[8], es[8], ca[8], switchd = ElleSwitchdistance()/100, switchdd; //added a for the area
    /*
     * FS!: switchdd is used for pivot points diagonal from the node
     * Think about adding its factor to ElleSwitchdistance (here 1/100) as a 
     * user input ???
     */
    Coords oldxy, newxy;
    Coords pvec, pvec_strain, pvec_surf, pvec_strain2, pvec_surf2;
    
    /*
     * Explanation:
     * a is the Node energy -> changed from normal GBM to use surface energy settings from the config file
     * es is the Node stored energy from fft -> Alberts function
     * ca is the cluster areay energy from gbm_pp -> Jenss function
     */    

    /*
     * Explanation of accessing the trial positions if iUserInDiagonal == 0:
     *              (pos3)
     *                -
     *                -
     *                -
     *                -
     * (pos2)- - - - NODE - - - -(pos1)
     *                -
     *                -
     *                -
     *                -
     *              (pos4)
     */
     
    ElleNodePosition( node, &oldxy );
    // Position 1
    newxy.x = oldxy.x + switchd;
    newxy.y = oldxy.y;
    e[0] = GetNodeEnergy( node, &newxy );
    es[0] = GetNodeStoredEnergy( node, & newxy );
    ca[0] = clusterData->returnClusterAreaEnergy( node, &newxy );
    // Position 2
    newxy.x = oldxy.x - switchd;
    newxy.y = oldxy.y;
    e[1] = GetNodeEnergy( node, &newxy );
    es[1] = GetNodeStoredEnergy( node, & newxy );
    ca[1] = clusterData->returnClusterAreaEnergy( node, &newxy );
    // Position 3
    newxy.x = oldxy.x;
    newxy.y = oldxy.y + switchd;
    e[2] = GetNodeEnergy( node, &newxy );
    es[2] = GetNodeStoredEnergy( node, & newxy );
    ca[2] = clusterData->returnClusterAreaEnergy( node, &newxy );
    // Position 4
    newxy.x = oldxy.x;
    newxy.y = oldxy.y - switchd;
    e[3] = GetNodeEnergy( node, &newxy );
    es[3] = GetNodeStoredEnergy( node, & newxy );
    ca[3] = clusterData->returnClusterAreaEnergy( node, &newxy );
    
    //int testnode= 1;
    //if (node==testnode) 
    //{
        //printf("area energies node %u: %e %e %e %e\n",testnode,ca[0],ca[1],ca[2],ca[3]);
        //printf("surf energies node %u: %e %e %e %e\n",testnode,e[0],e[1],e[2],e[3]);
    //}
    
    /*
     * Bringing the energies together:
     */
    pvec_surf2.x = (e[0] + ca[0]) - (e[1] + ca[1]); // cf. Becker et al 2008 equation 5
	pvec_surf2.y = (e[2] + ca[2]) - (e[3] + ca[3]);
	pvec_strain2.x = es[0] - es[1];
	pvec_strain2.y = es[2] - es[3];
    
    if ( node < 0 ) 
    {
        printf("Step: %d - Node %d energies at trial positions:\n",Settings_run.Count,node);
        printf("E-surf\t\tE-disloc\tE-area\n");
        for ( int i = 0; i < 4; i++ )
        {
            printf("%e\t%e\t%e\n",e[i],es[i],ca[i]);
        }
    }

// 04/12/2008
/*
 * Default version is with 4-pivot points, but the possibility to use diagonal 
 * positions to increase stability of triple nodes was added
 */
/*
 * 4-pivot points + small time steps seems that TNs are normally stable but some 
 * drift if strain stored energy >>> boundary energy 
 */
	
    /*
     * Explanation of accessing the trial positions if iUserInDiagonal == 1:
     *   (pos8)     (pos3)       (pos5)
     *      -         -         -
     *        -       -       -
     *          -     -     -
     *            -   -   -
     * (pos2)- - - - NODE - - - -(pos1)
     *            -   -   -
     *          -     -     -
     *        -       -       -
     *      -         -         -
     *    (pos6)    (pos4)      (pos7)
     */
	
	if (iUserInDiagonal==1) 
    {
	
		switchdd=switchd/sqrt(2.0);
        // Position 5
    	newxy.x = oldxy.x + switchdd;
		newxy.y = oldxy.y + switchdd;
		e[4] = GetNodeEnergy( node, &newxy );
		es[4] = GetNodeStoredEnergy( node, & newxy );
		ca[4] = clusterData->returnClusterAreaEnergy( node, &newxy );
        // Position 6
    	newxy.x = oldxy.x - switchdd;
		newxy.y = oldxy.y - switchdd;	
		e[5] = GetNodeEnergy( node, &newxy );
		es[5] = GetNodeStoredEnergy( node, & newxy );
		ca[5] = clusterData->returnClusterAreaEnergy( node, &newxy );
        // Position 7
		newxy.x = oldxy.x + switchdd;
		newxy.y = oldxy.y - switchdd;
		e[6] = GetNodeEnergy( node, &newxy );
		es[6] = GetNodeStoredEnergy( node, & newxy );
		ca[6] = clusterData->returnClusterAreaEnergy( node, &newxy );
        // Position 8
		newxy.x = oldxy.x - switchdd;
		newxy.y = oldxy.y + switchdd;
		e[7] = GetNodeEnergy( node, &newxy );
		es[7] = GetNodeStoredEnergy( node, & newxy );
		ca[7] = clusterData->returnClusterAreaEnergy( node, &newxy );
        
	    /*
         * Bringing the energies together:
         */
		pvec_surf2.x += ( ( (e[4] + ca[4]) - (e[5] + ca[5]) + (e[6] + ca[6]) - (e[7] + ca[7]) ) / sqrt(2.0) );
		pvec_surf2.y += ( ( (e[4] + ca[4]) - (e[5] + ca[5]) + (e[6] + ca[6]) - (e[7] + ca[7]) ) / sqrt(2.0) );
		pvec_strain2.x += ( (es[4] - es[5] + es[6] - es[7]) / sqrt(2.0) );
		pvec_strain2.y += ( (es[4] - es[5] + es[6] - es[7]) / sqrt(2.0) );

		pvec_surf2.x /= 2.0;
		pvec_surf2.y /= 2.0;
		pvec_strain2.x /= 2.0;
		pvec_strain2.y /= 2.0;
	}
	
    //Get the vector of the direction with highest decrease of surface energy
   	pvec_surf.x = -1 * pvec_surf2.x / ( 2 * switchd ); // cf. Becker et al., 2008, pp. 204 equation 5
    pvec_surf.y = -1 * pvec_surf2.y / ( 2 * switchd );

    // Get the vector of direction with "highest" decrease of strain stored energy
    // call GGMoveNode_test( int node, es1, es2, es3, es4, trialdist); //FS: Not working any more
	
	pvec_strain.x = 2 * pvec_strain2.x / ( 2 * switchd ); // added factor 2
	pvec_strain.y = 2 * pvec_strain2.y / ( 2 * switchd ); 
    
    /* 
     * FS: Speciality: Store the change in surface and stored strain energy
     * in bnode attribute B and C respectively by calculating the vector 
     * lengths of dE/dx and dE/dy.
     * 
     * For later usage: Normalise to surface energy: Then the surface energy 
     * will always be 1 and we will see the relative intensity of strain energy
     * Highest value will be infinity, because at some nodes the surface energy
     * might be zero: Take this into account for later plotting
     */
    // printf("Changes in surface and strain energy for node %u:\n",node);
    // printf("%e,%e\n\n",GetVectorLength(pvec_surf),GetVectorLength(pvec_strain));

    if (!ElleNodeAttributeActive(N_ATTRIB_B)) ElleInitNodeAttribute(N_ATTRIB_B);  
    if (!ElleNodeAttributeActive(N_ATTRIB_C)) ElleInitNodeAttribute(N_ATTRIB_C); 

    ElleSetNodeAttribute(node,GetVectorLength(pvec_surf),N_ATTRIB_B);
    ElleSetNodeAttribute(node,GetVectorLength(pvec_strain),N_ATTRIB_C);
    
	// Get resolved "maximum" direction 
	pvec.x = pvec_surf.x + pvec_strain.x;
	pvec.y = pvec_surf.y + pvec_strain.y;
	return GetMoveDir( node, pvec, xy ,switchd,&iNodeTooFast,&dTimeStepHighest,dModRedFact);
}

double GetNodeEnergy( int node, Coords * xy )
{
    int err, n, node2, node1, node3, nbnode[3], mineral, rgn[3];
    Coords n1, n2, n3, v1, v2, v3;
    double l1, l2, l3, E, en = 0;
    double bodyenergy=0, energyofsurface=0;
    //Get the neighbouring nodes
    if ( err = ElleNeighbourNodes( node, nbnode ) )
        OnError( "MoveNode", err );
    n = 0;
    //and put them into variables. In case of a double node, one is NO_NB and we don't want to use
    //that
    while ( n < 3 && nbnode[n] == NO_NB )
        n++;
    node1 = nbnode[n];
    n++;
    while ( n < 3 && nbnode[n] == NO_NB )
        n++;
    node2 = nbnode[n];
    n = 0;
    //see if the neighbouring nodes are active. Do we need that? Don't think so...
    if ( ElleNodeIsActive( node1 ) )
        n++;
    if ( ElleNodeIsActive( node2 ) )
        n++;
    //Get positions of neighbouring nodes
    ElleNodePlotXY( node1, & n1, xy );
    ElleNodePlotXY( node2, & n2, xy );
    //we don't really need the positions, we just need the length of the segments
    v1.x = n1.x - xy->x;
    v1.y = n1.y - xy->y;
    v2.x = n2.x - xy->x;
    v2.y = n2.y - xy->y;
    l1 = GetVectorLength( v1 );
    l2 = GetVectorLength( v2 );
    //if the node is a triple, we have to get the third node and the length of the segment
    if ( ElleNodeIsTriple( node ) )
    {
        node3 = nbnode[2];
        ElleNodePlotXY( node3, & n3, xy );
        v3.x = n3.x - xy->x;
        v3.y = n3.y - xy->y;
        l3 = GetVectorLength( v3 );
    }
	// Check which combination of phases is present at the current bounary segment and return the energy of that.
	E = l1*CheckPair(node, node1, 1);
	E += l2*CheckPair(node, node2, 1);
	if (ElleNodeIsTriple(node)) {
		E += l3*CheckPair(node, node3, 1);
		//E /= 3; // FS: E/=3 is wrong!!! Had to be commented! WE USE DIFFERENT ENERGIES IN THIS POLYPHASE CODE (and not 3*one energy) SO NO DIVISION IS NEEDED!!!! 
	} else {
		// E /= 2; // FS: E/=2 is wrong!!! Had to be commented! WE USE DIFFERENT ENERGIES IN THIS POLYPHASE CODE (and not 2*one energy) SO NO DIVISION IS NEEDED!!!! 
	}
	E *= ElleUnitLength();
    
    return E;
}

/*
 * This returns the value for mob/en/any other stuff defined by type for a 
 * segment between node 1 and node 2
 * type = 0 == mobility, 1==b_energy, 2==Activation Energy
 */
double CheckPair(int node1, int node2, int type)
{
	int rgn[2], int_a[2], i;
	double type_a[2];

	ElleNeighbourRegion(node1,node2,&rgn[0]);
	ElleNeighbourRegion(node2,node1,&rgn[1]);
	ElleGetFlynnRealAttribute(rgn[0], &type_a[0], iFlynnPhase);
	ElleGetFlynnRealAttribute(rgn[1], &type_a[1], iFlynnPhase);
    
	int_a[0] = (int)type_a[0];
	int_a[1] = (int)type_a[1];

	if (type == 0)
		return phases.pairs[int_a[0]][int_a[1]].mobility;
	else if (type == 1)
		return phases.pairs[int_a[0]][int_a[1]].b_energy;
	else if (type == 2)
		return phases.pairs[int_a[0]][int_a[1]].dGbActEn;
	else
		return 0;
}

/*
 * FS!: Returns strain induced energy. Think about commenting and formatting it 
 * to make the code easier to read!!
 */
double GetNodeStoredEnergy( int node, Coords * xy )
{
	///JR There has to be a change... Because it is not single Phase anymore
	// First check if the node in question is surrounded by Flynns of the same phase. 
	// If that is true, get the parameters for that phase and continue as normal.
	// if not it gets more complicated...
    
	fstream fDataFile;
	int iNodeType = 3, iCheck = 1;
	int iNeighbours[3], iFlynns[3], iFlynnPhases[3];
	double dFlynnPhaseCheck;
	double area, density=0, density_energy;
    
    fDataFile.open ( "Logfile_GetNodeStoredEnergy.txt", fstream::out | fstream::app);
	
	ElleNeighbourNodes( node, iNeighbours );
	
	for ( int i = 0, j = 0; i < 3; i++ ) {
		if ( iNeighbours[i] != NO_NB ) {
			ElleNeighbourRegion( node, iNeighbours[i], &iFlynns[j] );
			ElleGetFlynnRealAttribute( iFlynns[j], &dFlynnPhaseCheck, iFlynnPhase );
			iFlynnPhases[j] = (int) dFlynnPhaseCheck;
			j++;
		}
		else
			iNodeType = 2;
	}
	
	for ( int i = 0; i < iNodeType-1; i++ ) {
		if ( iFlynnPhases[i] != iFlynnPhases[i+1] )
			iCheck = 0;
	}

	if ( iCheck == 1 ) {

		ElleNodeUnitXY(xy);

		for ( int i = 0; i < iNodeType; i++ ) 
        {
			if (EllePtInRegion( iFlynns[i], xy) ) 
            {
				if ( phases.phasep[ iFlynnPhases[i] ].disscale == 0 )
					return (0.0); // if a phase gets overgrown which is not regarded in the dislocations energies - return 0 and skip the rest of the calculation.
				else 
                {
					if (ElleUnodeAttributeActive(U_DISLOCDEN)) 
                    {
						density = FS_density_unodes( iFlynns[i], xy, node ); 
                    }
					else if (ElleFlynnAttributeActive(DISLOCDEN))
						ElleGetFlynnRealAttribute( iFlynns[i], &density, DISLOCDEN );
					else 
                    {
						//cout << "ERROR: (GetNodeStoredEnergy-Internal) -> No means for density calculation found" << endl;
						return (0.0);
					}
					
					area = area_swept_gpclip(node, xy); //ElleRegionArea( iFlynns[i] ) - ReturnArea( iFlynns[i], node, xy );
					
					density_energy = phases.phasep[ iFlynnPhases[i] ].stored;
					
					return ( density * area * density_energy );
				}
			}
		}
        fDataFile << "Stage: " << Settings_run.Count << " ERROR: (Phase internal) - "
            " Trial Position not in neighbour Flynn for node (type "
            << iNodeType << "): " << node << "\nNeighbour Flynns: " 
            << iFlynns[0] << " " << iFlynns[1] << " (" << iFlynns[2] << ")\n" 
            << "Neighbour nodes: " << iNeighbours[0] << " " << iNeighbours[1] 
            << " (" << iNeighbours[2] << ")\n" << "Trial position at (x,y) = "
            << xy->x << ", " << xy->y << endl;
	}
	///JR Now it gets complicated... The node has neigbouring Flynns which don't belong to the same phase...
	else {
				
		ElleNodeUnitXY(xy);
		
		for ( int i = 0; i < iNodeType; i++ ) 
        {
			if (EllePtInRegion( iFlynns[i], xy) ) 
            {
				if ( phases.phasep[ iFlynnPhases[i] ].disbondscale == 0 )
					return (0.0); // if a phase gets overgrown which is not regarded in the dislocations energies - return 0 and skip the rest of the calculation.
				else 
                {
					if (ElleUnodeAttributeActive(U_DISLOCDEN)) 
					{
						density = FS_density_unodes( iFlynns[i], xy, node ); 	
					}
					else if (ElleFlynnAttributeActive(DISLOCDEN))
					{
						ElleGetFlynnRealAttribute( iFlynns[i], &density, DISLOCDEN );
					}
					else 
                    {
						// cout << "ERROR: (GetNodeStoredEnergy-Boundary) -> No means for density calculation found" << endl;
						return (0.0);
					}
					
					area = area_swept_gpclip(node, xy); //ElleRegionArea( iFlynns[i] ) - ReturnArea( iFlynns[i], node, xy );
					
					density_energy = phases.phasep[ iFlynnPhases[i] ].stored;
					
					return ( density * area * density_energy * phases.phasep[ iFlynnPhases[i] ].disbondscale ); // phases.phasep[ iFlynnPhases[i] ].disbondscale between 0 and 1, to scale how strong the strain induced movement for the boundary type is
				}
			}
		}
        fDataFile << "Stage: " << Settings_run.Count << " ERROR: (Phase boundary) - "
            " Trial Position not in neighbour Flynn for node (type "
            << iNodeType << "): " << node << "\nNeighbour Flynns: " 
            << iFlynns[0] << " " << iFlynns[1] << " (" << iFlynns[2] << ")\n" 
            << "Neighbour nodes: " << iNeighbours[0] << " " << iNeighbours[1] 
            << " (" << iNeighbours[2] << ")\n" << "Trial position at (x,y) = "
            << xy->x << ", " << xy->y << endl;
        //cout << "ERROR: (GetNodeStoredEnergy-Boundary) - Trial Pos..." << endl << "Node(" << iNodeType << "): " << node << ", Flynns: " << iFlynns[0] << " " << iFlynns[1] << " " << iFlynns[2] << endl
        //<< "Neighbours: " << iNeighbours[0] << " " << iNeighbours[1] << " " << iNeighbours[2] << " -- Pos: x" << xy->x << " y" << xy->y << endl;
	}
	
	/* FS: This does ONLY happen if the trial position was in NONE of the
	 * neighbouring flynns: That would indicate a very distorted flynn, that
	 * is actually too close to the bnode of interest since the trial position
	 * is more far away then the next flynn boundary
	 * This is solved here:
	 */
	/* Find the highest dislcoden core energy of the materials involved and
	 * use disbondscale = 1 (use the complete energy) */
	density_energy = 0.0;
	for (int i = 0; i < iNodeType; i++)
	{
		if (density_energy < phases.phasep[ iFlynnPhases[i] ].stored)
			density_energy = phases.phasep[ iFlynnPhases[i] ].stored;
	}
	
	area = area_swept_gpclip(node, xy);
	//double dDummyDD = 2e12;
	printf("WARNING (GetNodeStoredEnergy): Need to set dummy dislocation density -> Check Logfile\n");
	fDataFile << scientific << "Stage: " << Settings_run.Count << 
				 " WARNING (GetNodeStoredEnergy): Trial position not found in "
				 "ANY of the neighbour flynns, set energy for this point to "
				 << dDummyDD * area * density_energy << " using dummy "
				 "dislocden = " << dDummyDD << endl;
	
	return (dDummyDD * area * density_energy);
}

/*
 * Determines the dislocation density at a given coordinate using dislocden 
 * stored in unodes. "int node" is just used for error messages
 * iFlynn is the flynn ID of the flynn containing trial position @ coordinate 
 * XY. "node" is the bnode ID of the relevant bnode
 */
double density_unodes_3( int iFlynn, Coords * xy, int node ) 
{
    vector<int> vUnodeList;
	vector<UnodeDist> vDist;
	double dDensity, dEnergy, dDistTotal, dDist, roi;
	int i, j, max_unodes;
	Coords ref;
	UnodeDist uDist;
	fstream fDataFile;
	
	//double dDummyDD = 2e12; // FS: Used if no unode is closer than ROI, assuming a very distored grain
	
	fDataFile.open ( "Logfile_DensityUnodes3.txt", fstream::out | fstream::app);
	
	max_unodes = ElleMaxUnodes();
	//roi = sqrt( 1.0 / (double) max_unodes / 3.142 ) * 3;	// aprox. limit at 2nd neighbours
    roi = FS_GetROI(3);

	ElleGetFlynnUnodeList( iFlynn, vUnodeList);
	
	dEnergy = dDistTotal = 0;
	
    if ( vUnodeList.size() > 0 ) 
    {
		
		j = vUnodeList.size();
		
		// if ( j > iUnodeNeighbours ) j = iUnodeNeighbours;
    
        for ( i = 0; i < vUnodeList.size(); i++ )
        {
            ElleGetUnodePosition( vUnodeList.at( i ) , &ref );
            ElleCoordsPlotXY( &ref, xy );			  
            dDist = pointSeparation( &ref, xy );
            
            uDist.iNode = vUnodeList.at( i );
            uDist.dDist = dDist;
            
            vDist.push_back( uDist );
        }
        
        if ( vDist.size() > 1 )
			sort( vDist.begin(), vDist.end() );
        
        i = 0;
        
        while ( vDist[ i ].dDist < roi && i < vDist.size() )
        {
            if (bScaleSIBMToNonBasalAct)
                dDensity = FS_ScaleDDForBasalActivity(vDist[ i ].iNode);
            else
                ElleGetUnodeAttribute( vDist[ i ].iNode, U_DISLOCDEN, &dDensity);
			
			dEnergy += dDensity * ( roi - vDist[ i ].dDist );
			dDistTotal += ( roi - vDist[ i ].dDist );
			i++;
		}
			
        //for ( i = 0 && j > 1; i < j; i++ )
        //{
                
            //ElleGetUnodeAttribute( vDist[ i ].iNode, U_DISLOCDEN, &dDensity);
            
            //dEnergy += dDensity * ( vDist[ j ].dDist - vDist[ i ].dDist );
            //dDistTotal += ( vDist[ j ].dDist - vDist[ i ].dDist );
        //}
        
        if ( i == 0 ) 
        {
			cout      << "Stage: " << Settings_run.Count << " WARNING (density_unodes_3): No Unodes closer than roi in Flynn (" 
                << iFlynn << ") for node (" << node << ")" << " Setting dislocden = " << dDummyDD << " m-2" <<endl;
            fDataFile << "Stage: " << Settings_run.Count << " WARNING (density_unodes_3): No Unodes closer than roi in Flynn (" 
                << iFlynn << ") for node (" << node << ")" << " Setting dislocden = " << dDummyDD << " m-2" <<endl;
			
            // Use closest unode outside of roi, but in same flynn instead:
            /* FS: Doing this might not be the best thing: Better just set a 
             * very high dislocation density because in this situation we can 
             * assume a very distorted, i.e. highly strained local microstruct.
             *
            ElleGetUnodeAttribute( vDist[ 0 ].iNode, U_DISLOCDEN, &dDensity);
            dEnergy += dDensity * vDist[ 0 ].dDist;
            dDistTotal += vDist[ 0 ].dDist;
             *
             * Set a very high dislocden in this situation because grain must be
             * very distorted:
             */
			dEnergy = dDummyDD; // That means setting output to dDummyDD (e.g. 2e12)
			dDistTotal = 1;
        }
	}
	else 
    {
		// if vUnodeList.size() == 0, i.e. no unodes in flynn
		/* FS: should not happen with topochecks, but keep this bit of code to
		 * be on the save side
		 * 
		 * This part is searching for a unode in ROI in ANY of the neighbour
		 * flynns, not only in the one where the trial position is in:
		 * Might be better to do it like this before setting a very high DD, 
		 * because the flynn of interest it very small and we want to close it
		 * slowly
		 */
		cout      << "Stage: " << Settings_run.Count << " WARNING (density_unodes_3): No Unodes found in Flynn (" << iFlynn << ") for node (" << node << ")" << endl 
                  << "--> Average across all unodes < roi regardless of Flynns..." << endl;
        fDataFile << "Stage: " << Settings_run.Count << " WARNING (density_unodes_3): No Unodes found in Flynn (" << iFlynn << ") for node (" << node << ")" << endl 
                  << "--> Average across all unodes < roi regardless of Flynns..." << endl;
		
		j = 0;
		
		for ( i = 0; i < max_unodes; i++ )
		{
            ElleGetUnodePosition( i , &ref );
            ElleCoordsPlotXY( &ref, xy );			  
            dDist = pointSeparation( &ref, xy );
            
            if ( dDist < roi )
            {
                if (bScaleSIBMToNonBasalAct)
                    dDensity = FS_ScaleDDForBasalActivity( i );
                else
                    ElleGetUnodeAttribute( i, U_DISLOCDEN, &dDensity);
                    
				dEnergy += dDensity * ( roi - dDist );
				dDistTotal += ( roi - dDist );
				j++;
			}
		}
		
		if ( j == 0 ) // i.e. even if ALL flynns around the relevant trial position are regarded, NO unodes are within the ROI
        {
			cout      << "Stage: " << Settings_run.Count << " WARNING (density_unodes_3): no unodes found < roi --> Total number of Unodes: " << max_unodes << endl;
			cout      << "Setting dislocation densitiy = " << dDummyDD << " m-2" << endl;
            fDataFile << "Stage: " << Settings_run.Count << " WARNING (density_unodes_3): no unodes found < roi --> Total number of Unodes: " << max_unodes << endl;
			fDataFile << "\tSetting dislocation densitiy = " << dDummyDD << " m-2" << endl;
			dEnergy = dDummyDD; // That means setting output to dDummyDD (e.g. 2e12)
			dDistTotal = 1;
        }
	}
	
	if ( dDistTotal <= 0 ) // FS: Can this still happen here?? Only if before there was no unode in none of the flynns, then better set a very high energy!
    {
		cout      << "Stage: " << " WARNING (density_unodes_3): dDistTotal <= 0 --> Number of unodes in Flynn " << iFlynn << ": " << vUnodeList.size() << endl;
		cout      << "Setting dislocation densitiy = " << dDummyDD << " m-2" << endl;
        fDataFile << "Stage: " << " WARNING (density_unodes_3): dDistTotal <= 0 --> Number of unodes in Flynn " << iFlynn << ": " << vUnodeList.size() << endl;
		fDataFile << "\tSetting dislocation densitiy = " << dDummyDD << " m-2" << endl;
		dEnergy = dDummyDD; // That means setting output to dDummyDD (e.g. 2e12)
		dDistTotal = 1;
	}
    
    vUnodeList.clear();
    return ( dEnergy / dDistTotal );
}

/*
 * Returns the area swept by a boundary moving due to the movement of one node
 */
double area_swept_gpclip( int node, Coords * xy )
{
    int n, node1, node2, node3, nbnode[3], err;
    Coords n1, n2, n3; 

	//new parameters
	double e[3], val[2], roi=0.01;
	Coords xy_unode, ref;
	int max_unodes, max_nbs, n_res;
	double range_gamma=2, dist, dist_total; // range of gamma search  
    int i, j, k;	
	double area, area_1, area_2, area_3=0.0, area_full;
	double area_swept;
	
	Coords nn[3], dum;
	vector <Coords> pv1, pv2, pv3, pvaux;
	vector< vector<Coords> > res, res2;	
	
	// First PART: Calculates the swept area 
	//Get the neighbouring nodes
    if ( err = ElleNeighbourNodes( node, nbnode ) )
        OnError( "MoveNode_unode", err );
    n = 0;
    
	//and put them into variables. In case of a double node, one is NO_NB and we don't want to use
	
    while ( n < 3 && nbnode[n] == NO_NB )
        n++;
    node1 = nbnode[n];
    n++;
    while ( n < 3 && nbnode[n] == NO_NB )
        n++;
    node2 = nbnode[n];
    n = 0;
    
	//see if the neighbouring nodes are active. Do we need that? Don't think so...
    if ( ElleNodeIsActive( node1 ) )
        n++;
    if ( ElleNodeIsActive( node2 ) )
        n++;
	
    //Get positions of neighbouring nodes
    ElleNodePlotXY( node1, & n1, xy ); // xy is the node 
    ElleNodePlotXY( node2, & n2, xy );  
	
    // Get "old" position of pivote node 	
    ElleNodePlotXY(node, &nn[2], xy ); 	
    nn[0].x=xy->x; 
	nn[0].y=xy->y;

	// Swept_triangles 
	// Coords triangle_1
    	nn[1].x=n1.x;
		nn[1].y=n1.y;
	    area_1= area_triangle(nn[0], nn[1], nn[2]);	
		// bnodes in counterclockwise ?  	
        if (area_1<0) {     
			dum.x=nn[0].x;
			dum.y=nn[0].y;			
			nn[0].x=nn[2].x;
			nn[0].y=nn[2].y;
			nn[2].x=dum.x;
			nn[2].y=dum.y;				
		}	

		// vector swept area 1
		for (i=0;i<3;i++) pv1.push_back(nn[i]); 
		
	// Coords triangle_2
    	nn[1].x=n2.x;
		nn[1].y=n2.y;
	    area_2= area_triangle(nn[0], nn[1], nn[2]);
		// counterclockwise  	
        if (area_2<0) {
			dum.x=nn[0].x;
			dum.y=nn[0].y;			
			nn[0].x=nn[2].x;
			nn[0].y=nn[2].y;
			nn[2].x=dum.x;
			nn[2].y=dum.y;			
		}
		// vector swept area 2
		for (i=0;i<3;i++) pv2.push_back(nn[i]); 
			
	//if the node is a triple, we have to get the third node and the position
		
    if ( ElleNodeIsTriple( node )  ) {
        node3 = nbnode[2];
        ElleNodePlotXY( node3, & n3, xy );
	// Coords triangle_3
   		nn[1].x=n3.x;
		nn[1].y=n3.y;			
	    area_3= area_triangle(nn[0], nn[1], nn[2]);		
		// counterclockwise  	
        if (area_3<0) {     
			dum.x=nn[0].x;
			dum.y=nn[0].y;			
			nn[0].x=nn[2].x;
			nn[0].y=nn[2].y;
			nn[2].x=dum.x;
			nn[2].y=dum.y;				
		}
				// vector swept area 2
		for (i=0;i<3;i++) pv3.push_back(nn[i]); 	
		
		}

	// Coords 'res' holds the resulting clip region
	// in theory always one region, but...

 		if (area_1 != 0 || area_2 != 0) {	
	       gpcclip(pv1,pv2,res,GPC_UNION);	
		   area=0.0;			
		  // n_res=res[0].size();

	 	  // in theory never, but .. 
           if (res.size()>1) {			  
              double max_area = 0.0;
              for (j=0; j<res.size(); j++) {
                area = fabs(polyArea(res[j]));
                if (area > max_area) max_area=area;
			    }
		     }
		  else  area = fabs(polyArea(res[0]));

		// if TNode	  
  		if ( ElleNodeIsTriple( node ) ) {
     		if (area_3 != 0) {	  
			gpcclip(pv3,res[0],res2,GPC_UNION);
			area = fabs(polyArea(res2[0]));
	 		}
		}		
	 }	
	   
		else {
		area=0.0;  
		if (ElleNodeIsTriple (node)) area=fabs(area_3); 
		}	 
	  
     area_full=fabs(area_1)+fabs(area_2)+fabs(area_3);
	 area_swept=area;
	area_swept *= ElleUnitLength()*ElleUnitLength(); // UnitLength squared because it's an area in m^2
	
	return (area_swept);

}

/*
 * Calculates the area of a triangle defined by the 3 coordinates
 */
double area_triangle (Coords n1, Coords n2, Coords n3)
{
	return (((n2.x-n1.x)*(n3.y-n1.y)-(n3.x-n1.x)*(n2.y-n1.y))/2.0);
}

/*
 * FS: Updates dislocation densities in unodes: If a unode changes flynn (i.e. 
 * is swept by a moving or -recrystallising- boundary) its dislocation density 
 * is set to zero
 * FS: Needed some adjustments because there was a tiny bug when a grain is 
 * seperated and the new part gets a new ID...
 */
int FS_update_dislocden(void)
{    
	fstream fDataFile;
    int iMaxUnodes = 0, iMaxFlynns = 0, iNumBnodes = 0;
    int iFlynnId = 0, iFlynnIdOld = 0, iFlynnIdOldUnode = 0, iUnodeID = 0;
    int iCount = 0;
    double dPhase1 = 0.0, dPhase2 = 0.0;
    double dDensity = 0.0, dDensityNew = 0.0, dDensityMin = 0.0; // dDensityMin: implicit in the GBM scheme	
    double dFlynnIdOld = 0.0, dFlynnIdOldUnode = 0.0;
    double dRoi = 0.0;
    double dValEuler[3], dDistTotal, dMinDist, dDist = 0.0;
    vector<int> vUnodeList;
    vector<Coords> vBnodesXY;
    Coords xy,refxy;
	
	fDataFile.open ( "Logfile_FS-UpdateDislocden.txt", fstream::out | fstream::app);
	fDataFile << "# # # # #   " << Settings_run.Count << "   # # # # #" << endl;
    
    /* FS: The following is for having a textfile which stores old and new unode
     * orientation after a unode has been swept by a moving grain boundary */
    fstream fUnodeOri;
	fUnodeOri.open ( "UnodeOriChangeGBM.txt", fstream::out | fstream::app);
	fUnodeOri << "# # # # #   " << Settings_run.Count << "   # # # # #" << endl;
    fUnodeOri << "id x y old_phase new_phase e1_before e2_before e3_before e1_after e2_after e3_after" << endl;
    
    iMaxUnodes = ElleMaxUnodes();
    iMaxFlynns = ElleMaxFlynns();
	//dRoi = sqrt(1.0/(double)iMaxUnodes/3.142)*5; // used for Euler angle reassignment
    dRoi = FS_GetROI(5);
    
    // STEP 1: Check and Update for new flynns and update flynn ID in F_ATTRIB_C
    // 1.1 Check and update for flynns that are now inactive:
    for (int i=0;i<iMaxUnodes;i++)
    {
        iFlynnId = ElleUnodeFlynn(i);
        
        if (!ElleFlynnIsActive(iFlynnId)) // if the flynn is now inactive, i.e. does not exist any more
        {
            ElleGetUnodePosition(i,&xy); // get the unode's position
            
            for (int j=0;j<iMaxFlynns;j++) // cycle through all flynns to find the one this unode sits in
            {
                if (ElleFlynnIsActive(j))
                {
                    if (EllePtInRegion(j,&xy)) //i.e. if the unode with position xy is in flynn j
                    {
                        ElleAddUnodeToFlynn(j,i); // refer the unode "i" to the flynn "j" in which it actually sits now
                        fDataFile << "INFO (UpdateDislocden): Flynn "<<iFlynnId<<" is inactive now, switching unode "
                            <<i<<" to flynn "<<j<<endl;
                        break;
                    }
                }
            }            
        }
    }
    
    // 1.2 Check for flynns that are still active, but may have been seperated into an old and new part with new ID:
    // FS: This is what was not working completely correct before:
    for (int k=0;k<iMaxFlynns;k++)
    {
        if (ElleFlynnIsActive(k))
        {
            ElleGetFlynnRealAttribute(k,&dFlynnIdOld,iFlynnNumber); 
            iFlynnIdOld = (int)dFlynnIdOld; // must be an integer
            // This value should be identical with the flynn ID as it is right now, i.e.:
            if (iFlynnIdOld != k)
            {
                // i.e.: There is a need for an update:
                fDataFile << "INFO (UpdateDislocden): Flynn " << iFlynnIdOld << " (partially) renumbered to new flynn " << k << endl;
                
                ElleGetFlynnUnodeList(k,vUnodeList); 
                
                for (int i=0;i<vUnodeList.size();i++)
                {
                    iUnodeID = vUnodeList.at(i);
                    ElleGetUnodeAttribute(iUnodeID,iUnodeFlynnNumber,&dFlynnIdOldUnode);
                    iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
                    if (iFlynnIdOldUnode==iFlynnIdOld) // Only update the ones that were in the old grain, not the ones that may be inside just by a moving boundary
                    {
                        ElleSetUnodeAttribute(iUnodeID,iUnodeFlynnNumber,double(k));
                        ElleAddUnodeToFlynn(k,iUnodeID); // k is flynn ID iUnodeID is unode ID
                    }
                }
                vUnodeList.clear();   
                
                // This will not be sufficient, if the old grain ID is still
                // present (i.e. not inactive) due to seperation of one grain
                // into two grains. Therefore there is a need for an additional
                // check for unodes in the part of the seperated grains with the
                // old grain ID:
                
                ElleGetFlynnUnodeList(iFlynnIdOld,vUnodeList); 
                
                for (int i=0;i<vUnodeList.size();i++)
                {
                    iUnodeID = vUnodeList.at(i);
                    ElleGetUnodePosition(iUnodeID,&xy);
                    ElleGetUnodeAttribute(iUnodeID,iUnodeFlynnNumber,&dFlynnIdOldUnode);
                    iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
                    if (EllePtInRegion(k,&xy) && iFlynnIdOldUnode == iFlynnIdOld) // Only update the ones that were in the old grain, not the ones that may be inside just by a moving boundary
                    {
                        ElleSetUnodeAttribute(iUnodeID,iUnodeFlynnNumber,double(k));
                        ElleAddUnodeToFlynn(k,iUnodeID); // k is flynn ID iUnodeID is unode ID                        
                    }
                }
                vUnodeList.clear();  
                           
            }
            ElleSetFlynnRealAttribute(k,double(k),iFlynnNumber);  // can also be in the "if (iFlynnIdOld != k)", maybe saves some nanoseconds :-)
        }
    }
    
    // STEP 2: Update dislocation densities and euler angles using nearest unodes
	if (!ElleUnodeAttributeActive(U_DISLOCDEN))
		fDataFile << "ERROR (update_dislocden): unode attrib *U_DISLOCDEN* not active!!!" << endl;
        
    for (int i=0;i<iMaxUnodes;i++)
    {
        ElleGetUnodeAttribute(i,iUnodeFlynnNumber,&dFlynnIdOldUnode);
        iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
        iFlynnId = ElleUnodeFlynn(i);
        
        if (iFlynnIdOldUnode != iFlynnId) // That means the unode has changed its flynn by a migrating boundary
        {
            //by JR: check if the unode changed phase...
			ElleGetFlynnRealAttribute( iFlynnIdOldUnode, &dPhase1, iFlynnPhase );
			ElleGetFlynnRealAttribute( iFlynnId, &dPhase2, iFlynnPhase );
			if ( int(dPhase1) != int(dPhase2) )
			{
				fDataFile << "WARNING: Unode " << i << " changed Phase... Set New DislocDen to 0" << endl; //Shifted Dislocden back." << endl;
                //ShiftDislocdenUnode( i, iFlynnIdOldUnode, iFlynnId );
			}
			
			fDataFile << "old new flynn " << iFlynnIdOldUnode << " " << iFlynnId << endl;
			
			// Update euler angles. Nearest unode, use quaternion for interpolate the Euler orientation 	
			ElleGetFlynnUnodeList(iFlynnId,vUnodeList);
			ElleGetUnodePosition(i,&refxy);
            
            dDistTotal = 0.0;
			dMinDist = 1;
			iCount = 0;
            
            for (int j=0;j<vUnodeList.size();j++)
            {
                ElleGetUnodeAttribute(vUnodeList[j],iUnodeFlynnNumber,&dFlynnIdOldUnode);	
				iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
                
                if (iFlynnIdOldUnode == iFlynnId)
                {

					ElleGetUnodePosition(vUnodeList[j],&xy);	
					ElleCoordsPlotXY (&refxy, &xy);			  
					dDist = pointSeparation(&refxy,&xy);
							
					/* FS: Find closest unode in new flynn and get its euler_3
                     * --> Should probably better check if there is a unoe in 
                     * roi and use a roi average
                     */
                    if (dDist<dRoi) 
					{
						iCount++;
						if ( dDist < dMinDist && ElleUnodeAttributeActive(EULER_3)) 
						{
								ElleGetUnodeAttribute(vUnodeList[j], &dValEuler[0],E3_ALPHA);
								ElleGetUnodeAttribute(vUnodeList[j], &dValEuler[1],E3_BETA);
								ElleGetUnodeAttribute(vUnodeList[j], &dValEuler[2],E3_GAMMA);	
								dMinDist = dDist;					
						}
					}                    
                }                
            }
			
            if (ElleUnodeAttributeActive(EULER_3))
            {
                /*
                 * FS: New way to check which orientations get eaten up by 
                 * moving grain boundaries and which ones are "growing":
                 * Store old and new unode orientation of the swept unode
                 * in a separate textfile called: UnodeOriChangeGBM.txt
                 */
                double dEulerOld[3];
                Coords cUnodepos;
                ElleGetUnodeAttribute(i,&dEulerOld[0],E3_ALPHA);
                ElleGetUnodeAttribute(i,&dEulerOld[1],E3_BETA);
                ElleGetUnodeAttribute(i,&dEulerOld[2],E3_GAMMA);
                ElleGetUnodePosition(i,&cUnodepos);
                
                fUnodeOri << i << " " << cUnodepos.x << " " << cUnodepos.y
                          << " " << (int)dPhase1 << " " << (int)dPhase2 
                          << " " << dEulerOld[0] << " " << dEulerOld[1] << " "
                          << dEulerOld[2] << " " << dValEuler[0] << " " 
                          << dValEuler[1] << " " << dValEuler[2] << endl;
                          
                          
                if ( (dValEuler[0] >= -180) &&  (dValEuler[0]<=180) && (dValEuler[1] >= -180) &&  (dValEuler[1]<=180) && (dValEuler[2] >= -180) &&  (dValEuler[2]<=180))
                {	
                    //printf(" unodes count %i\n", iCount);
                    fDataFile << "unodes count " << iCount << endl;
                    // set new information 

                    ElleSetUnodeAttribute(i,E3_ALPHA, dValEuler[0]);
                    ElleSetUnodeAttribute(i,E3_BETA, dValEuler[1]);
                    ElleSetUnodeAttribute(i,E3_GAMMA, dValEuler[2]); 			

                }
            }
			ElleSetUnodeAttribute(i,U_DISLOCDEN,dDensityMin);
			ElleSetUnodeAttribute(i,iUnodeFlynnNumber, double(iFlynnId));
			vUnodeList.clear();             
        }
    }
    
    // 2nd part: simulate internal restructuration of grains  
	// Not active with FFT simulations
    // To compare gbm with bnodes simulations: automatic readjust at average value 
    // Only scalar dislocation density, not effect on euler orientation of unodes 	
	double dAvdensity; //a_factor=0.0 redistribution factor; 1.0 full while 0.0 non redistribution
	int a_factor = 0;
    
    UserData userdata;
    ElleUserData(userdata);
    a_factor= (int)userdata[1];
    
	///JR since nothing changes if a_factor is 0, don't even do it then...
	if ( a_factor != 0 )
	{
		for (int j=0;j<iMaxFlynns;j++) 
		{
			if (ElleFlynnIsActive(j)) 
			{
				vUnodeList.clear();
				dAvdensity=0.0;
				ElleGetFlynnUnodeList(j,vUnodeList); // get the list of unodes for a flynn
				
				for (int i=0;i<vUnodeList.size();i++) 
                {
					ElleGetUnodeAttribute(vUnodeList[i],U_DISLOCDEN,&dDensity);					
					dAvdensity += dDensity; 			
                }
					
				dAvdensity /= vUnodeList.size();
				
				for (int i=0;i<vUnodeList.size();i++) 
                {		
					ElleGetUnodeAttribute(vUnodeList[i],U_DISLOCDEN,&dDensity);
					dDensityNew = dDensity-(dDensity-dAvdensity)*a_factor; 
					ElleSetUnodeAttribute(vUnodeList[i],U_DISLOCDEN, dDensityNew);			
				}
		
			}
		}
	}
	
	fDataFile.close();
    fUnodeOri.close();
	return 0;   
}
/*
 * FS: Updates dislocation densities in unodes: If a unode changes flynn (i.e. 
 * is swept by a moving or -recrystallising- boundary) its dislocation density 
 * is set to zero
 * FS: Needed some adjustments because there was a tiny bug when a grain is 
 * seperated and the new part gets a new ID...
 */
int FS_update_dislocden2(void)
{    
	fstream fDataFile;
    int iMaxUnodes = 0, iMaxFlynns = 0, iNumBnodes = 0;
    int iFlynnId = 0, iFlynnIdOld = 0, iFlynnIdOldUnode = 0, iUnodeID = 0;
    int iCount = 0;
    double dPhase1 = 0.0, dPhase2 = 0.0;
    double dDensity = 0.0, dDensityNew = 0.0, dDensityMin = 0.0; // dDensityMin: implicit in the GBM scheme	
    double dFlynnIdOld = 0.0, dFlynnIdOldUnode = 0.0;
    double dRoi = 0.0;
    double dValEuler[3], dDistTotal, dMinDist, dDist = 0.0;
    vector<int> vUnodeList;
    vector<Coords> vBnodesXY;
    Coords xy,refxy;
	
	fDataFile.open ( "Logfile_FS_UpdateDislocden.txt", fstream::out | fstream::app);
	fDataFile << "# # # # #   " << Settings_run.Count << "   # # # # #" << endl;
    
    /* FS: The following is for having a textfile which stores old and new unode
     * orientation after a unode has been swept by a moving grain boundary */
    fstream fUnodeOri;
	fUnodeOri.open ( "UnodeOriChangeGBM.txt", fstream::out | fstream::app);
	fUnodeOri << "# # # # #   " << Settings_run.Count << "   # # # # #" << endl;
    fUnodeOri << "id x y old_phase new_phase e1_before e2_before e3_before e1_after e2_after e3_after" << endl;
    
    iMaxUnodes = ElleMaxUnodes();
    iMaxFlynns = ElleMaxFlynns();
	//dRoi = sqrt(1.0/(double)iMaxUnodes/3.142)*5; // used for Euler angle reassignment
    dRoi = FS_GetROI(8);
    
    // STEP 1: Check and Update for new flynns and update flynn ID in F_ATTRIB_C
    // 1.1 Check and update for flynns that are now inactive:
    for (int i=0;i<iMaxUnodes;i++)
    {
        iFlynnId = ElleUnodeFlynn(i);
        
        if (!ElleFlynnIsActive(iFlynnId)) // if the flynn is now inactive, i.e. does not exist any more
        {
            ElleGetUnodePosition(i,&xy); // get the unode's position
            
            for (int j=0;j<iMaxFlynns;j++) // cycle through all flynns to find the one this unode sits in
            {
                if (ElleFlynnIsActive(j))
                {
                    if (EllePtInRegion(j,&xy)) //i.e. if the unode with position xy is in flynn j
                    {
                        ElleAddUnodeToFlynn(j,i); // refer the unode "i" to the flynn "j" in which it actually sits now
                        //ElleSetUnodeAttribute(i,iUnodeFlynnNumber,double(j)); // should not do this here, otherwise the unode cannot be detected later
                        fDataFile << "INFO (UpdateDislocden): Flynn "<<iFlynnId<<" is inactive now, switching unode "
                            <<i<<" to flynn "<<j<<endl;
                        break;
                    }
                }
            }            
        }
    }
    
    // 1.2 Check for flynns that are still active, but may have been seperated into an old and new part with new ID:
    // FS: This is what was not working completely correct before:
    for (int k=0;k<iMaxFlynns;k++)
    {
        if (ElleFlynnIsActive(k))
        {
            ElleGetFlynnRealAttribute(k,&dFlynnIdOld,iFlynnNumber); 
            iFlynnIdOld = (int)dFlynnIdOld; // must be an integer
            // This value should be identical with the flynn ID as it is right now, i.e.:
            if (iFlynnIdOld != k)
            {
                // i.e.: There is a need for an update:
                fDataFile << "INFO (UpdateDislocden): Flynn " << iFlynnIdOld << " (partially) renumbered to new flynn " << k << endl;
                
                ElleGetFlynnUnodeList(k,vUnodeList); 
                
                for (int i=0;i<vUnodeList.size();i++)
                {
                    iUnodeID = vUnodeList.at(i);
                    ElleGetUnodeAttribute(iUnodeID,iUnodeFlynnNumber,&dFlynnIdOldUnode);
                    iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
                    if (iFlynnIdOldUnode==iFlynnIdOld) // Only update the ones that were in the old grain, not the ones that may be inside just by a moving boundary
                    {
                        ElleSetUnodeAttribute(iUnodeID,iUnodeFlynnNumber,double(k));
                        ElleAddUnodeToFlynn(k,iUnodeID); // k is flynn ID iUnodeID is unode ID
                    }
                }
                vUnodeList.clear();   
                
                // This will not be sufficient, if the old grain ID is still
                // present (i.e. not inactive) due to seperation of one grain
                // into two grains. Therefore there is a need for an additional
                // check for unodes in the part of the seperated grains with the
                // old grain ID:
                if (ElleFlynnIsActive(iFlynnIdOld))
                {
                    ElleGetFlynnUnodeList(iFlynnIdOld,vUnodeList); 
                    
                    for (int i=0;i<vUnodeList.size();i++)
                    {
                        iUnodeID = vUnodeList.at(i);
                        ElleGetUnodePosition(iUnodeID,&xy);
                        ElleGetUnodeAttribute(iUnodeID,iUnodeFlynnNumber,&dFlynnIdOldUnode);
                        iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
                        if (EllePtInRegion(k,&xy) && iFlynnIdOldUnode == iFlynnIdOld) // Only update the ones that were in the old grain, not the ones that may be inside just by a moving boundary
                        {
                            ElleSetUnodeAttribute(iUnodeID,iUnodeFlynnNumber,double(k));
                            ElleAddUnodeToFlynn(k,iUnodeID); // k is flynn ID iUnodeID is unode ID                        
                        }
                    }
                    vUnodeList.clear();  
                }
                           
            }
            ElleSetFlynnRealAttribute(k,double(k),iFlynnNumber);  // can also be in the "if (iFlynnIdOld != k)", maybe saves some nanoseconds :-)
        }
    }
    
    // 1.3 We still need to check for unodes that changed their flynn by
    // "simply" sweeping boundaries
    // Maybe some of this stuff is redundant and there is a quicker way, but
    // with this, I wanted to be on the safe side
    for (int unode=0;unode<iMaxUnodes;unode++)
    {
        bool bFound=false; // will be true once correct host flynn is found
        iFlynnId = ElleUnodeFlynn(unode);
        ElleGetUnodePosition(unode,&xy);
        if (ElleFlynnIsActive(iFlynnId))
            if (EllePtInRegion(iFlynnId,&xy)) bFound=true;
        
        if (!bFound)
        {
            /* Need to search for the correct host flynn*/
            for (int flynn=0;flynn<iMaxFlynns;flynn++)
            {
                if (ElleFlynnIsActive(flynn))
                {
                    if (EllePtInRegion(flynn,&xy)) 
                    {
                        ElleAddUnodeToFlynn(flynn,unode);
                        //ElleSetUnodeAttribute(i,iUnodeFlynnNumber,double(j)); // should not do this here, otherwise the unode cannot be detected later
                        bFound=true;
                        break;
                    }
                }
            }
        }
    }
    
    // STEP 2: Update dislocation densities and euler angles using nearest unodes
	if (!ElleUnodeAttributeActive(U_DISLOCDEN))
		fDataFile << "ERROR (update_dislocden): unode attrib *U_DISLOCDEN* not active!!!" << endl;
        
    for (int i=0;i<iMaxUnodes;i++)
    {
        ElleGetUnodeAttribute(i,iUnodeFlynnNumber,&dFlynnIdOldUnode);
        iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
        iFlynnId = ElleUnodeFlynn(i);
        
        if (iFlynnIdOldUnode != iFlynnId) // That means the unode has changed its flynn by a migrating boundary
        {
            //by JR: check if the unode changed phase...
			ElleGetFlynnRealAttribute( iFlynnIdOldUnode, &dPhase1, iFlynnPhase );
			ElleGetFlynnRealAttribute( iFlynnId, &dPhase2, iFlynnPhase );
			if ( int(dPhase1) != int(dPhase2) )
			{
				fDataFile << "WARNING: Unode " << i << " changed Phase... Set New DislocDen to 0" << endl; //Shifted Dislocden back." << endl;
                //ShiftDislocdenUnode( i, iFlynnIdOldUnode, iFlynnId );
			}
			
			//fDataFile << "Unode " <<i<< " old new flynn " << iFlynnIdOldUnode << " " << iFlynnId << endl;
			
			// Update euler angles, use orientation of nearest unode 
            vUnodeList.clear();	
			ElleGetFlynnUnodeList(iFlynnId,vUnodeList);
			ElleGetUnodePosition(i,&refxy);
            
            dDistTotal = 0.0;
			dMinDist = 1; 
			iCount = 0;
            double dNewEuler[3];
            for (int ii=0;ii<3;ii++) dNewEuler[ii]=0.0;
            
            /* Determine new euler_3 from unodes of the same flynn in roi*/
            if (ElleUnodeAttributeActive(EULER_3))
            {          
                /* Get and store old orientation for potential later use or 
                 * logfile*/
                double dEulerOld[3];
                Coords cUnodepos;
                ElleGetUnodeAttribute(i,&dEulerOld[0],E3_ALPHA);
                ElleGetUnodeAttribute(i,&dEulerOld[1],E3_BETA);
                ElleGetUnodeAttribute(i,&dEulerOld[2],E3_GAMMA);
                ElleGetUnodePosition(i,&cUnodepos);
                
                for (int j=0;j<vUnodeList.size();j++)
                {
                    ElleGetUnodeAttribute(vUnodeList[j],iUnodeFlynnNumber,&dFlynnIdOldUnode);	
                    iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
                    
                    if (iFlynnIdOldUnode == iFlynnId)
                    {

                        ElleGetUnodePosition(vUnodeList[j],&xy);	
                        ElleCoordsPlotXY (&refxy, &xy);			  
                        dDist = pointSeparation(&refxy,&xy);
                        
                        if (dDist<=dRoi && dDist<dMinDist)
                        {
                            iCount++;                            
                            ElleGetUnodeAttribute(vUnodeList[j],&dValEuler[0],E3_ALPHA);
                            ElleGetUnodeAttribute(vUnodeList[j],&dValEuler[1],E3_BETA);
                            ElleGetUnodeAttribute(vUnodeList[j],&dValEuler[2],E3_GAMMA);
                            /*dNewEuler[0] += (dValEuler[0]*dDist);  
                            dNewEuler[1] += (dValEuler[1]*dDist);
                            dNewEuler[2] += (dValEuler[2]*dDist); 
                            dDistTotal += dDist; */
                            dMinDist = dDist;
                        }                    
                    }                
                }
                
                //if (iCount>0)
                //{
                    //dValEuler[0] = dNewEuler[0]/dDistTotal;
                    //dValEuler[1] = dNewEuler[1]/dDistTotal;
                    //dValEuler[2] = dNewEuler[2]/dDistTotal;
                //}
                if (iCount<=0)
                {
                    /* No unodes found in roi, use mean value of the whole flynn
                     * Only if there are no more unodes in flynn (meaning that
                     * vUnodeList.size()==0) keep old orientation */
                    if (vUnodeList.size()==0) // unlikely, but may be possible
                    {
                        for (int ii=0;ii<3;ii++) 
                            dValEuler[ii] = dEulerOld[ii];
                        fDataFile << "WARNING (FS_update_dislocden): Setting ";
                        fDataFile << "new orientation of swept unode "<<i;
                        fDataFile << " to old value" << endl;
                        printf("WARNING (FS_update_dislocden): Setting new orientation of swept unode %u to old value\n",i);
                    }
                    else
                    {
                        dDistTotal=0.0;
                        dDist=0.0;
                        for (int j=0;j<vUnodeList.size();j++)
                        {
                            ElleGetUnodeAttribute(vUnodeList[j],iUnodeFlynnNumber,&dFlynnIdOldUnode);	
                            iFlynnIdOldUnode = (int)dFlynnIdOldUnode;
                            
                            if (iFlynnIdOldUnode == iFlynnId)
                            {

                                ElleGetUnodePosition(vUnodeList[j],&xy);	
                                ElleCoordsPlotXY (&refxy, &xy);			  
                                dDist = pointSeparation(&refxy,&xy);
                                
                                ElleGetUnodeAttribute(vUnodeList[j],&dValEuler[0],E3_ALPHA);
                                ElleGetUnodeAttribute(vUnodeList[j],&dValEuler[1],E3_BETA);
                                ElleGetUnodeAttribute(vUnodeList[j],&dValEuler[2],E3_GAMMA);
                                dNewEuler[0] += (dValEuler[0]*dDist);  
                                dNewEuler[1] += (dValEuler[1]*dDist);
                                dNewEuler[2] += (dValEuler[2]*dDist); 
                                dDistTotal += dDist;                     
                            }                
                        }
                        // Next if is to be on the save side and avoid NaNs
                        // at least this avoid crashing of the model
                        if (dDistTotal<=0.0) dDistTotal=1.0;
                        dValEuler[0] = dNewEuler[0]/dDistTotal;
                        dValEuler[1] = dNewEuler[1]/dDistTotal;
                        dValEuler[2] = dNewEuler[2]/dDistTotal;
                        fDataFile << "WARNING (FS_update_dislocden): Setting ";
                        fDataFile << "new orientation of swept unode "<<i;
                        fDataFile << " to flynn mean value" << endl;
                        printf("WARNING (FS_update_dislocden): Setting new orientation of swept unode %u to flynn mean value\n",i);
                    }
                }
                /*
                 * FS: New way to check which orientations get eaten up by 
                 * moving grain boundaries and which ones are "growing":
                 * Store old and new unode orientation of the swept unode
                 * in a separate textfile called: UnodeOriChangeGBM.txt
                 */                
                fUnodeOri << i << " " << cUnodepos.x << " " << cUnodepos.y
                          << " " << (int)dPhase1 << " " << (int)dPhase2 
                          << " " << dEulerOld[0] << " " << dEulerOld[1] << " "
                          << dEulerOld[2] << " " << dValEuler[0] << " " 
                          << dValEuler[1] << " " << dValEuler[2] << endl;                          
                          
                if ( (dValEuler[0] >= -180) &&  (dValEuler[0]<=180) && (dValEuler[1] >= -180) &&  (dValEuler[1]<=180) && (dValEuler[2] >= -180) &&  (dValEuler[2]<=180))
                {	
                    //printf(" unodes count %i\n", iCount);
                    fDataFile << "number of unodes in flynn: " << iCount << endl;
                    // set new information 

                    ElleSetUnodeAttribute(i,E3_ALPHA, dValEuler[0]);
                    ElleSetUnodeAttribute(i,E3_BETA, dValEuler[1]);
                    ElleSetUnodeAttribute(i,E3_GAMMA, dValEuler[2]); 			

                }
            }
			ElleSetUnodeAttribute(i,U_DISLOCDEN,dDensityMin);
			ElleSetUnodeAttribute(i,iUnodeFlynnNumber, double(iFlynnId));
			vUnodeList.clear();             
        }
    }
    
    // 2nd part: simulate internal restructuration of grains  
	// Not active with FFT simulations
    // To compare gbm with bnodes simulations: automatic readjust at average value 
    // Only scalar dislocation density, not effect on euler orientation of unodes 	
	double dAvdensity; //a_factor=0.0 redistribution factor; 1.0 full while 0.0 non redistribution
	int a_factor = 0;
    
    UserData userdata;
    ElleUserData(userdata);
    a_factor= (int)userdata[1];
    
	///JR since nothing changes if a_factor is 0, don't even do it then...
	if ( a_factor != 0 )
	{
		for (int j=0;j<iMaxFlynns;j++) 
		{
			if (ElleFlynnIsActive(j)) 
			{
				vUnodeList.clear();
				dAvdensity=0.0;
				ElleGetFlynnUnodeList(j,vUnodeList); // get the list of unodes for a flynn
				
				for (int i=0;i<vUnodeList.size();i++) 
                {
					ElleGetUnodeAttribute(vUnodeList[i],U_DISLOCDEN,&dDensity);					
					dAvdensity += dDensity; 			
                }
					
				dAvdensity /= vUnodeList.size();
				
				for (int i=0;i<vUnodeList.size();i++) 
                {		
					ElleGetUnodeAttribute(vUnodeList[i],U_DISLOCDEN,&dDensity);
					dDensityNew = dDensity-(dDensity-dAvdensity)*a_factor; 
					ElleSetUnodeAttribute(vUnodeList[i],U_DISLOCDEN, dDensityNew);			
				}
		
			}
		}
	}
	
	fDataFile.close();
    fUnodeOri.close();
	return 0;   
}


/*
 * FS: In the beginning of the process, this function checks if each unode is
 * assigned to the correct flynn -> i.e. if flynn ID is correctly stored in
 * U_ATTRIB_C (or I should say in: iUnodeFlynnNumber), also the flynn ID should 
 * be correctly stored in a flynn attribute, this is also checked.
 */
void FS_check_unodes_in_flynn()
{
    int iFlynnID;
    Coords cUnodeXY;
    
    for (int unode=0;unode<ElleMaxUnodes();unode++)
    {
        ElleGetUnodePosition(unode,&cUnodeXY);
        iFlynnID = ElleUnodeFlynn(unode);
        
        /* Check if unode is really in this flynn: 
         * If yes, update unode and flynn attribute storing the correct
         * flynn id, if not find the flynn, the unode is in at the moment
         * and update everything afterwards*/
        if (EllePtInRegion(iFlynnID,&cUnodeXY))
        {
            /* unode is assigned to the correct flynn */
            ElleSetUnodeAttribute(unode,iUnodeFlynnNumber,(double)iFlynnID);
            ElleSetFlynnRealAttribute(iFlynnID,(double)iFlynnID,iFlynnNumber);            
        }
        else
        {
            /* unode is NOT assigned to the correct flynn, now we will have to 
             * loop through all flynn to find the correct one */
            bool bFound = false;
            int flynn = 0;
            while (flynn<ElleMaxFlynns() && !bFound)
            {
                if (ElleFlynnIsActive(flynn))
                {
                    if (EllePtInRegion(flynn,&cUnodeXY))
                    {
                        /* THIS is the correct flynn, update values now: */
                        ElleAddUnodeToFlynn(flynn,unode);
                        ElleSetUnodeAttribute(unode,iUnodeFlynnNumber,(double)flynn);
                        ElleSetFlynnRealAttribute(flynn,(double)flynn,iFlynnNumber);  
                        bFound = true;                      
                    }                   
                }
                flynn++;
            } // end of while loop going through all flynns
        } // end of if if (EllePtInRegion(iFlynnID,&cUnodeXY))
        
    }  // end of looping through all unodes
}

clusterTracking::clusterTracking( void )
{

	// find the phases --> temporary since it has to stay compatible to other functions...
	// At the end of the function the Phase numbers are included in the lists they belong to.
	lFicksDiffPhases.clear();

	for ( int i = phases.first_phase; i < (phases.first_phase + phases.no_phases); i++ ) {
		if ( phases.phasep[ i ].cluster_diff == 1 ) {
			vClusterPhases.push_back( i );
		}
		else {
			lFicksDiffPhases.push_back( i );
		}
	}
	
	// if there is no Phase present which uses Cluster Tracking, set the global variable to 0 and break.
	if ( vClusterPhases.size() == 0 ) {
		bClusterCheck = 0;
		return;
	}
	
	
	// read the Cluster Tracking stuff from the config file
	
	bool bFound = false;
	fstream file;
	string line;
	stringstream linestr;
	char c;
	// This functions reads the config file to storage...

	file.open ("phase_db.txt", fstream::in );

	if (file.is_open()) {
		while (file.good()) {
			getline (file,line);
			//cout << line << endl;
			if ( bFound == false ) {
				if ( line.find("CLUSTER_TRACKING") != string::npos )
					bFound = true;
			}
			else {
				if (line.length() > 0) {
					c = line.at(0);
					if (c!='#' && c!=' ') {
						linestr.clear();
						linestr << line;
						linestr >> dMultiplierA >> dMultiplierB >> dMultiplierC >> dMultiplierD;
						linestr.clear();
						bFound = false;
					}
				}
			}
		}
	}
	file.close();

	// set the max Flynns parameter.
	dAreaShift = 1e-10;
	clusterTracking::findClusters();
}

clusterTracking::~clusterTracking()
{
    
}

void clusterTracking::updateClusters( void )
{
	if ( !bClusterCheck )
		return;
	clusterTracking::findClusters();
	clusterTracking::findSplit();
	clusterTracking::findMerge();
	clusterTracking::checkDoubleClusterAreaLoop();
}

void clusterTracking::getPhaseAreas( void )
{
	double dPhaseArea;

	vPhaseAreas.clear();
	// get the complete area for each phase
	for ( int i = phases.first_phase; i < (phases.first_phase + phases.no_phases); i++ ) 
    {
		dPhaseArea = 0; // set the start to 0
		for ( int j = 0; j < iMaxFlynns; j++ ) 
        {
			if ( vFlynnPhase.at ( j ) == i ) 
            {
				dPhaseArea += ElleRegionArea( vFlynns.at ( j ) );
			}
		}
		vPhaseAreas.push_back( dPhaseArea );
	}
}

bool clusterTracking::writeInitialData( const char *filename )
{
	fstream fInitial;

	if (!fileExists( filename )) 
    {

		fInitial.open ( filename, fstream::out | fstream::trunc);

		clusterTracking::getPhaseAreas();
        
		if (fInitial.is_open()) 
        {
			fInitial << scientific << vPhaseAreas[0];
			for ( int i = phases.first_phase + 1; i < (phases.first_phase + phases.no_phases); i++ )
            {
				fInitial << " " << vPhaseAreas[i];
            }
			fInitial << endl;
		}
		fInitial.close();

		return true;
	}
	else {
		// cout << "WARNING: initial file present!" << endl << "If you start at step 0 delete this file first!" << endl;
		return false;
	}
}

bool clusterTracking::writeData( const char *filename, int step)
{
	fstream fDataFile;

	fDataFile.open ( filename, fstream::out | fstream::app);

	clusterTracking::getPhaseAreas();

	if (fDataFile.is_open()) {
		fDataFile << step << " " << scientific << vPhaseAreas[0];
		for ( int i = phases.first_phase + 1; i < (phases.first_phase + phases.no_phases); i++ )
			fDataFile << " " << vPhaseAreas[i];
		fDataFile << endl;
	}
	fDataFile.close();

	return true;
}


bool clusterTracking::getClusters( void )
{
	int temp_int, iOnList;
	double temp_double;

	vector<int> vCluster;
	vector<vector<int> > vClusters;
	list<int> lOriginal, lNeighbour;

	for ( int z = 0; z < vClusterPhases.size(); z++ ) 
    {
		lOriginal.clear();
		// get all flynns with phase i
		for ( int j = 0; j < iMaxFlynns; j++ ) 
        {
			if ( vFlynnPhase.at ( j ) == vClusterPhases.at( z ) )
				lOriginal.push_back( vFlynns.at ( j ) );
		}
		// just to be sure that every Flynn is only once in the Vector.
		if ( lOriginal.size() == 0 ) 
        {
			return (0);
		}
		lOriginal.sort();
		lOriginal.unique();
		

		// find flynns that are clustered together
		// as long as there are any flynns in the phase list
		while ( lOriginal.size() > 0 ) 
        {
			vCluster.clear();
			vCluster.push_back( lOriginal.front() ); // put the first flynn into the cluster list
			lOriginal.pop_front(); // delete that element from the list

			for ( int n = 0; n < vCluster.size(); n++ ) 
            {
				lNeighbour.clear(); // clear the neigbour list
				ElleFlynnNbRegions( vCluster.at(n), lNeighbour ); //find neighbours for the current flynn (n) in the cluster list

				// check whether any flynn in the neighbour list matches the current phase (i)
				// as long as there are entries in the neighbours list do the following
				while ( lNeighbour.size() > 0 ) 
                {
					ElleGetFlynnRealAttribute( lNeighbour.front(), &temp_double, iFlynnPhase ); // get phase from flynn
					temp_int = (int) temp_double; // convert to int

					//compare to current phase
					if ( temp_int == vClusterPhases.at( z ) )  // if Flynn has the same phase
                    { 		
						// look whether the Flynn is already on the cluster list
						int iOnList = 0;
						for ( int i = 0; i < vCluster.size() && iOnList == 0; i++ )
							if ( lNeighbour.front() == vCluster.at(i) )
								iOnList = 1;
						if ( iOnList == 0 ) {						// if NOT
							vCluster.push_back( lNeighbour.front() ); // add flynn to cluster list
							lOriginal.remove( lNeighbour.front() );	// remove that flynn from the original phase list
							lNeighbour.pop_front();					// remove it from the neighbours list
						}
						else										// if it is
							lNeighbour.pop_front();					// just remove it from the neighbours list
					}
					else										// if Flynn has not the same phase
						lNeighbour.pop_front();					// just remove it from the neighbours list
				}
			}
			vClusters.push_back(vCluster);	//vector which contains all the flynns beloning to a cluster is put in another vector
		}
		vPhasesClusters.push_back( vClusters );
	}
	return (1);
}

vector<double> clusterTracking::returnMultiplier ( vector<double> vAreaPercentage )
{
	//cout << dMultiplierA << " " << dMultiplierB << " " << dMultiplierC << " " << dMultiplierD << endl;
	vector<double> vMultiplier;
	
	for ( int i = 0; i < vAreaPercentage.size(); i++ ) {
		vMultiplier.push_back ( ( dMultiplierA * pow( vAreaPercentage[i], dMultiplierD ) ) + ( dMultiplierB * vAreaPercentage[i] ) + dMultiplierC );
	}
	return vMultiplier;
}

double clusterTracking::returnFlynnAreaChange ( int iFlynn, int iNode, Coords * xyNewPos )
{
	//if (iNode == phases.p_en ) {
		//fstream file;
		//file.open ("2.txt", fstream::out | fstream::app );
		//file << ReturnArea( iFlynn, iNode, xyNewPos ) << " - " << ElleRegionArea( iFlynn ) << " = " << (ReturnArea( iFlynn, iNode, xyNewPos ) - ElleRegionArea( iFlynn ) ) << endl;
		//file.close();
	//}
	return ( ReturnArea( iFlynn, iNode, xyNewPos ) - ElleRegionArea( iFlynn ) );
}

vector<double> clusterTracking::returnClusterAreaChange ( vector<vector<int> > vPhaseFlynns, int iNode, Coords * xyLoc )
{
    double dNewMultiplierA = 0.0; // FS: To scale A to the actual bubble size
    bool bFound;
	double dClusterArea, dClusterAreaCheck;
	vector<double> vClusterAreaPercentageChange, vClusterAreaMultiplier, vClusterAreaEnergy, vCurrentArea, vClusterArea;// just get the area for all clusters which touch that node and store it there.
	vector<double> vPhaseClusterAreaChange; // get the area change for all clusters which touch the node with the given movement.
	
	for ( int i = 0; i < vPhaseFlynns.size(); i++ ) {
		vPhaseClusterAreaChange.push_back ( 0 );
		ElleGetFlynnRealAttribute( vPhaseFlynns[i][0], &dClusterArea, iFlynnCluster );
		for ( int j = 0; j < vPhaseFlynns[i].size(); j++ ) {
			vPhaseClusterAreaChange.at ( i ) +=  returnFlynnAreaChange ( vPhaseFlynns[i][j], iNode, xyLoc );
			if ( j > 0 ) {
				ElleGetFlynnRealAttribute( vPhaseFlynns[i][j], &dClusterAreaCheck, iFlynnCluster );
				if ( dClusterArea != dClusterAreaCheck ) {
					cout << "WARNING: Stored Clusterareas in the Flynns that belong to the same Cluster are not the same!!" << endl;
					//cout << dClusterArea << "=" << dClusterAreaCheck << endl;
					//cout << *vPhaseFlynns[i][0] << " - " << *vPhaseFlynns[i][j] << endl;
					clusterTracking::findClusters();
					clusterTracking::findSplit();
					clusterTracking::findMerge();
					clusterTracking::checkDoubleClusterAreaLoop();
					ElleGetFlynnRealAttribute( vPhaseFlynns[i][0], &dClusterArea, iFlynnCluster );
				}
			}
		}
		
		vClusterArea.push_back( dClusterArea );
		
		
		// get the current area
		for ( int z = 0, bFound = false; z < vPhasesClusters.size() && bFound == false; z++ ) {
			for ( int j = 0; j < vPhasesClusters[z].size() && bFound == false; j++ ) {
				for ( int k = 0; k < vPhasesClusters[z][j].size() && bFound == false; k++ ) {
					if ( vPhasesClusters[z][j][k] == vPhaseFlynns[i][0] ) {
						vCurrentArea.push_back( vPhasesClusterAreas[z][j] );
						bFound = true;
					}
				}
			}
		}
		//vClusterAreaPercentageChange.push_back ( fabs( ( ( vCurrentArea.at( i ) + vPhaseClusterAreaChange.at( i ) ) / vClusterArea.at( i ) ) - 1 ) );
	}
	//vClusterAreaMultiplier = clusterTracking::returnMultiplier ( vClusterAreaPercentageChange );
	if ( vPhaseClusterAreaChange.size() != vCurrentArea.size() )
		cout << "ERROR in returnClusterAReaChange --> Vector size (PhaseCluster:CurrentArea) Not the same!! " 
			<< vPhaseClusterAreaChange.size() << ":" << vCurrentArea.size() 
			<< " " << vPhaseClusterAreaChange.front() << " " << vPhaseFlynns[0][0] << endl;
	
	if ( vPhaseClusterAreaChange.size() != vClusterArea.size() )
		cout << "ERROR in returnClusterAReaChange --> Vector size (PhaseCluster:ClusterArea) Not the same!!" 
			<< vPhaseClusterAreaChange.size() << ":" << vClusterArea.size() << endl;
            
    /*
     * FS: To scale area energy to actual cluster size, we need to know the 
     * minimum cluster area that is possible, i.e. the minimum area allowed in
     * the model (cf. topology checks):
     */   
    double dMinArea = 0.0;
    if (!ElleUnodesActive()) dMinArea = sqrt(3)*pow(ElleSwitchdistance(),2)/4;
    else dMinArea = 1/(double)ElleMaxUnodes(); // Which is the minimum area between four neighbouring unodes in a square grid

    /* FS: Or set the min area manually: */
    //dMinArea = 1.0/pow(256.0,2);
	
	for ( int i = 0; i < vPhaseClusterAreaChange.size(); i++ ) 
    {
		/*
         * FS: CHANING THE CLUSTER AREA ENERGY FOR CURRENT AREA:
         * The higher the current area, the higher the energy (multiplier A)
         * should be, therefore A is set input file as the highest value that is 
         * possible (for a bubble with minimum area that is allowed in the 
         * model (cf. topochecks). Hence, it is just a theoretical value. This 
         * value A is then multiplied by the normalised area of the bubble 
         * (currentarea/(dMinArea^-1)), to be in the end scaled to the 
         * bubble size
         */
        //dNewMultiplierA = dMultiplierA * (dMinArea/vCurrentArea.at( i ));
        //printf("Min area     = %e\n",dMinArea); 
        //printf("Cluster area = %e\n",vCurrentArea.at( i )); 
        //printf("A            = %f\n",dMultiplierA);
        //printf("A-scaled     = %f\n\n",dNewMultiplierA);
        
        /* FS: To avoid errors with timesteps being much to high for very small
         * flynns: Change area energy?!*/
        
        // FS: OLD CODE BY JENS:
         vClusterAreaEnergy.push_back ( dMultiplierA * pow ( fabs( ( ( vCurrentArea.at( i ) + ( vPhaseClusterAreaChange.at( i ) ) ) - vClusterArea.at( i ) ) / vClusterArea.at( i ) ), dMultiplierD ) );
        // FS: New code with scaled MultiplierA
        //vClusterAreaEnergy.push_back ( dNewMultiplierA * pow ( fabs( ( ( vCurrentArea.at( i ) + ( vPhaseClusterAreaChange.at( i ) ) ) - vClusterArea.at( i ) ) / vClusterArea.at( i ) ), dMultiplierD ) );
	}
	
	
	if (iNode == phases.p_en ) {
		fstream file;
		file.open ("1.txt", fstream::out | fstream::app );
		for ( int i = 0; i < vPhaseClusterAreaChange.size(); i++ ) {
			file << "TripleNodeArea " << i << " " << xyLoc->x << " " << xyLoc->y << " " << vPhaseClusterAreaChange.at ( i ) << " " << vClusterArea.at( i ) << " " << vCurrentArea.at( i ) << " " << vClusterAreaEnergy.at ( i ) << endl;
			//file << "TripleNodeArea " << i << " " << xyLoc->x << " " << xyLoc->y << " " << vPhaseClusterAreaChange.at ( i ) << " " << vClusterArea.at( i ) << " " << vCurrentArea.at( i ) << " " << vClusterAreaPercentageChange.at ( i ) << " " << vClusterAreaMultiplier.at ( i ) << " " << vClusterAreaEnergy.at ( i ) << endl;
		}
		file.close();
	}
	
	return vClusterAreaEnergy;
}

double clusterTracking::returnClusterAreaEnergy ( int iNode , Coords * xyLoc )
{
	int iNodeType = 3;
	int iNeighbours[3], iFlynns[3];
	
	ElleNeighbourNodes(iNode,iNeighbours);
	
	//cout << "Node: " << iNode << ", Neighbours:";
	//for ( int i = 0; i < 3; i++ )
		//cout << " " << iNeighbours[i];
	//cout << endl;
	
	for ( int i = 0, j = 0; i < 3; i++ ) {
		if ( iNeighbours[i] != NO_NB ) {
			ElleNeighbourRegion(iNode,iNeighbours[i],&iFlynns[j]);
			j++;
		}
		else
			iNodeType = 2;
	}
	//cout << "NodeType: " << iNodeType << ", NeighbourFlynns:";
	//for ( int i = 0; i < iNodeType; i++ )
		//cout << " " << iFlynns[i];
	//cout << endl;
	
	
	
	int iFlynnPhaseCheck;
	double dFlynnPhaseCheck;
	vector<int> vClusterPhaseFlynns;
	vector<vector<int> > vPhaseClusterFlynns;
	
	for ( int i = 0; i < vClusterPhases.size(); i++ ) {
		vClusterPhaseFlynns.clear();
		for ( int j = 0; j < iNodeType; j++ ) {
			ElleGetFlynnRealAttribute( iFlynns[j], &dFlynnPhaseCheck, iFlynnPhase );
			iFlynnPhaseCheck = (int) dFlynnPhaseCheck;
			if ( iFlynnPhaseCheck == vClusterPhases[ i ] ) {
				vClusterPhaseFlynns.push_back( iFlynns[j] );
			}
		}
		if ( vClusterPhaseFlynns.size() > 0 )
			vPhaseClusterFlynns.push_back( vClusterPhaseFlynns );
	}
	// if no Flynn with a Cluster Phase belongs to the node -> return 0
	if ( vPhaseClusterFlynns.size() == 0 )
		return 0.0;
	
	double dClusterAreaEnergy = 0;
	vector<double> vClusterAreaEnergy = clusterTracking::returnClusterAreaChange ( vPhaseClusterFlynns, iNode, xyLoc );
	
	for ( int i = 0; i < vClusterAreaEnergy.size(); i++ )
		dClusterAreaEnergy += vClusterAreaEnergy.at ( i );
	
	return dClusterAreaEnergy;
}


void clusterTracking::getClusterAreas( void )
{
	// Calculate the Area of the Cluster
	double dClusterArea = 0.0;

	vector<double> vClusterArea;

	vPhasesClusterAreas.clear();

	for ( int z = 0; z < vPhasesClusters.size(); z++ ) {
		vClusterArea.clear();
		for ( int i = 0; i < vPhasesClusters[z].size(); i++ ) {
			dClusterArea = 0.0;
			for ( int j = 0; j < vPhasesClusters[z][i].size(); j++ ) {
				dClusterArea += ElleRegionArea( vPhasesClusters[z][i][j] );
			}
			vClusterArea.push_back( dClusterArea );
		}
		vPhasesClusterAreas.push_back( vClusterArea );
	}
}

void clusterTracking::setClusterAreas( void )
{
	for ( int z = 0; z < vPhasesClusters.size(); z++ ) {
		for ( int i = 0; i < vPhasesClusters[z].size(); i++ ) {
			for ( int j = 0; j < vPhasesClusters[z][i].size(); j++ ) {
				ElleSetFlynnRealAttribute( vPhasesClusters[z][i][j], vPhasesClusterAreas[z][i], iFlynnCluster );
			}
		}
	}
}

void clusterTracking::findSplit( void )
{
	double dArea, dAreaCheck;

	vector<int> vSplitClusterFlynns;
	vector<vector<int> > vSplitClusters;

	for ( int z = 0; z < vPhasesClusters.size(); z++ ) {
		for ( int i = 0; i < vPhasesClusters[z].size(); i++ ) {
			ElleGetFlynnRealAttribute( vPhasesClusters[z][i][0], &dArea, iFlynnCluster );

			vSplitClusters.clear();

			for ( int j = 0; j < vPhasesClusters[z].size(); j++ ) {
				vSplitClusterFlynns.clear();
				for ( int k = 0; k < vPhasesClusters[z][j].size(); k++ ) {
					ElleGetFlynnRealAttribute( vPhasesClusters[z][j][k], &dAreaCheck, iFlynnCluster );
					if ( dArea == dAreaCheck ) {
						vSplitClusterFlynns.push_back( vPhasesClusters[z][j][k] );
					}
				}
				if ( vSplitClusterFlynns.size() > 0 ) {
					vSplitClusters.push_back( vSplitClusterFlynns );
				}
			}
			// Wenn mehr als ein Cluster mit Flynns mit der gleichen Fläche gefunden wurde --> Der Cluster hat sich geteilt --> Flächen neu verteilen.
			// (Ein Cluster bedeutet der Cluster selbst wurde gefunden)
			if ( vSplitClusters.size() > 1 ) {
				//cout << "Cluster with same areanumber detected.... (SPLIT)" << endl;
				//for ( int g = 0; g < vSplitClusters.size(); g++ ) {
					//for ( int h = 0; h < vSplitClusters[g].size(); h++ ) {
						//cout << vSplitClusters[g][h] << " ";
					//}
					//cout << endl;
				//}
				clusterTracking::resolveSplit( vSplitClusters );
			}
		}
	}
}

void clusterTracking::findClusters( void )
{
	vPhasesClusterAreas.clear();
	vPhasesClusters.clear();
	vFlynns.clear();
	vFlynnPhase.clear();
	
	int temp_int;
	double temp_double;

	vector<int> ran;

	iMaxFlynns = ElleMaxFlynns();
	ran.clear();
	for ( int i = 0; i < iMaxFlynns; i++ )
		if ( ElleFlynnIsActive( i ) )
			ran.push_back( i );
			
	iMaxFlynns = ran.size();

	for ( int i = 0; i < iMaxFlynns; i++ ) {
		vFlynns.push_back( ran.at( i ) );
		ElleGetFlynnRealAttribute( vFlynns.back(), &temp_double, iFlynnPhase );
		temp_int = (int)temp_double;
		vFlynnPhase.push_back( temp_int );
	}
	// now we've got two vectors. 
	// the first one includes the Flynn numbers. The second one the phase number of that flynn. 
	// The numbers that belong together are stored in the same position of the vectors.
	if ( vFlynnPhase.size() != vFlynns.size() )
		cout << "ERROR: findClusters --> FlynnPhase and Flynn Vector don't have the same size!!!" << endl;

	if ( !clusterTracking::getClusters() ) {
		bClusterCheck = 0;
		return;
	}		
	clusterTracking::getClusterAreas();
}

void clusterTracking::findMerge( void )
{
	double dArea, dAreaCheck;

	list<double> lNotMatchingAreas;

	for ( int z = 0; z < vPhasesClusters.size(); z++ ) {
		for ( int i = 0; i < vPhasesClusters[z].size(); i++ ) {
			lNotMatchingAreas.clear();
			ElleGetFlynnRealAttribute( vPhasesClusters[z][i][0], &dArea, iFlynnCluster );
			for ( int j = 1; j < vPhasesClusters[z][i].size(); j++ ) {
				ElleGetFlynnRealAttribute( vPhasesClusters[z][i][j], &dAreaCheck, iFlynnCluster );
				// if the Cluster Areas of the Flynn ain't match the first one... --> Merge?
				if ( dArea != dAreaCheck ) {
					lNotMatchingAreas.push_back(dAreaCheck);
				}
			}
			// all double Entries have to be deleted... --> problem if two of the merged clusters had the same areas...
			lNotMatchingAreas.sort();
			lNotMatchingAreas.unique();

			if ( lNotMatchingAreas.size() > 0 ) {
				lNotMatchingAreas.push_back(dArea);
				clusterTracking::resolveMerge( z, i, lNotMatchingAreas );
			}
		}
	}
}

void clusterTracking::checkDoubleClusterAreaLoop ( void )
{
	bool bChange;

	do {
		bChange = false;
		for ( int z = 0; z < vPhasesClusters.size() && !bChange; z++ )
			for ( int i = 0; i < vPhasesClusters[z].size() - 1 && !bChange; i++ )
				bChange = clusterTracking::checkDoubleClusterArea ( z, i, vPhasesClusters[z].size() - 1 );
	} while ( bChange == true );
}

bool clusterTracking::checkDoubleClusterArea ( int z, int i, int iMax )
{
	bool bChanged = false;

	if ( i < iMax) {
		double dCheckA, dCheckB;
		ElleGetFlynnRealAttribute( vPhasesClusters[z][i][0], &dCheckA, iFlynnCluster );
		ElleGetFlynnRealAttribute( vPhasesClusters[z][iMax][0], &dCheckB, iFlynnCluster );
		if ( dCheckA == dCheckB ) {
			cout << "DOUBLE AREA DETECTED!!!" << endl;
			cout << dCheckA << " | Flynn: " << vPhasesClusters[z][i][0] << " (" << i << ") || " << dCheckB << " | Flynn: " << vPhasesClusters[z][iMax][0] << " (" << iMax << ")" << endl;
			bChanged = clusterTracking::resolveDoubleClusterArea ( z, i, iMax, dCheckA );
		}
		if ( clusterTracking::checkDoubleClusterArea ( z, i, iMax-1 ) )
			bChanged = true;
	}
	return bChanged;
}

bool clusterTracking::resolveDoubleClusterArea ( int z, int i, int iMax, double dCheck )
{
	for ( int j = 0; j < vPhasesClusters[z][i].size(); j++ )
		ElleSetFlynnRealAttribute( vPhasesClusters[z][i][j], dCheck-dAreaShift, iFlynnCluster );
	for ( int j = 0; j < vPhasesClusters[z][iMax].size(); j++ )
		ElleSetFlynnRealAttribute( vPhasesClusters[z][iMax][j], dCheck+dAreaShift, iFlynnCluster );

	return true;
}

void clusterTracking::resolveMerge ( int z, int i, list<double> lNotMatchingAreas )
{
	double dMergedArea = 0.0;
	// calculate new cluster area (just add the old areas together)
	
	cout << "Cluster has different Clusterflynns... (MERGE) ";
	while ( lNotMatchingAreas.size() > 0) {
		cout << lNotMatchingAreas.back() << " ";
		dMergedArea += lNotMatchingAreas.back();
		lNotMatchingAreas.pop_back();
	}
	cout << ":: " << dMergedArea << endl;
	// set new area for ALL flynns in that cluster!
	for ( int j = 0; j < vPhasesClusters[z][i].size(); j++ ) {
		ElleSetFlynnRealAttribute( vPhasesClusters[z][i][j], dMergedArea, iFlynnCluster );
	}
}

void clusterTracking::resolveSplit( vector<vector<int> > vSplitClusters )
{
	double dSplitClusterAreas[vSplitClusters.size()], dSplitClusterRatio, dSplitClusterAreaComplete, dSplitClusterNewArea;

	ElleGetFlynnRealAttribute( vSplitClusters[0][0], &dSplitClusterNewArea, iFlynnCluster );
	cout << "Cluster with same areanumber detected.... (SPLIT) " << dSplitClusterNewArea << " |";
	dSplitClusterAreaComplete = 0;
	for ( int j = 0; j < vSplitClusters.size(); j++ ) {
		dSplitClusterAreas[j] = 0;
		cout << "| ";
		for ( int k = 0; k < vSplitClusters[j].size(); k++ ) {
			dSplitClusterAreas[ j ] += ElleRegionArea( vSplitClusters[j][k] );
			cout << vSplitClusters[j][k] << " ";
		}
		dSplitClusterAreaComplete += dSplitClusterAreas[ j ];
	}
	cout << "|" << endl;
	for ( int j = 0; j < vSplitClusters.size(); j++ ) {
		// Calculate Ratio for that part of the split Cluster (Split Part / Current Complete Area) --> For the Ratio calculation the old Area is not used.
		dSplitClusterRatio = dSplitClusterAreas[ j ] / dSplitClusterAreaComplete;
		// Calculate New Area with the OLD Area and the calculated Ratio
		ElleGetFlynnRealAttribute( vSplitClusters[j][0], &dSplitClusterNewArea, iFlynnCluster );
		dSplitClusterNewArea *= dSplitClusterRatio;
		// Write new Area in that part of the Flynn.
		for ( int k = 0; k < vSplitClusters[j].size(); k++ ) {
			ElleSetFlynnRealAttribute( vSplitClusters[j][k], dSplitClusterNewArea, iFlynnCluster );
		}
	}
}

clusters::clusters ( vector<int> vPushedFLynns, double dPushedArea )
{
	//use swap pointers if possible put the pushed flynns in class storage.
}

clusters::~clusters ()
{
	
}

/* FS: NEW CODE FOR SURFACE ENERGY DETERMINATION AT TRIAL POSITION
 * --> NOT WORKING CORRECTLY AT THE MOMENT
 * "GetNodeEnergy" calculates the (Surface) Energy of a node:
 * 
 * This really only calculates the surface energy of the node, nothing else. 
 * General equation is E=en(l1+l2+l3)*lengthscale with E=energy, en=surface 
 * energy and l1/l2/l3 the length of the segments next to the node adjusted to 
 * the lengthscale
 */
double ZZ_FS_GetNodeEnergy( int node, Coords * xy )
{
    int err = 0, n = 0;
    Coords cNodeTempXY;
    Coords cVecTemp;
    int iNbNode[3];
    
    double dEnergyTrialPosition = 0.0;
    double dEsurfBoundary[3];
    double dLength[3];
    
    // Pre-alocate a view variables:
    for (int i=0; i<3; i++)
    {
        dLength[i] = 0.0;
        dEsurfBoundary[i] = 0.0;
    }

    //Get the neighbouring nodes
    if ( err = ElleNeighbourNodes( node, iNbNode ) )
        OnError( "MoveNode", err );
        
    /*
     * Loop through all neighbour nodes and get their distances to trial 
     * position (iLength1, iLength2 and if triple node also iLength3)
     */
    while (n<3 && iNbNode[n] != NO_NB)
    {
        //printf("Neighbour: %u, n= %u\n",iNbNode[n],n);
        
        if (ElleNodeIsActive(iNbNode[n]))
        {
            // Determine distance to trial position at "xy":
            ElleNodePosition(iNbNode[n],&cNodeTempXY);
            cVecTemp.x = cNodeTempXY.x - xy->x;   
            cVecTemp.y = cNodeTempXY.y - xy->y;  
            
            dLength[n] = GetVectorLength(cVecTemp);   
            
            // Determine surface energy of specific boundary:
            dEsurfBoundary[n] = CheckPair(node,iNbNode[n],1);         
        }
        else
        {
            printf("FS_GetNodeEnergy line: %u - Error: Node %u not active\n",__LINE__,iNbNode[n]);
            return 0;
        }
        
        n++;
    }
    
    // Check if node type is correct, if not give an error message
    if (n!=2 && n!=3)
    {
        /* 
         * If n is neither 2 nor 3: (Neither double nor triple node)
         * There is something wrong, give an error message
         */
        printf("FS_GetNodeEnergy line: %u - Error: Unknown node type, n= %u\n",__LINE__,n);
        return 0;
    }
    
    for (int j=0; j<n; j++)
    {
        dEnergyTrialPosition += dEsurfBoundary[j]*dLength[j];
    }
    
    dEnergyTrialPosition *= ElleUnitLength();
    
    return dEnergyTrialPosition;
}

double FS_GetROI(int iFact)
{
    /*
     * FS: The product of boxwidth + boxheight will not remain constant in a 
     * pure shear simulation and unode distances change. Hence, a more accurate 
     * determination of ROI is used here using not sqrt(1.0 / (...), but 
     * sqrt(width*height / (...)
     * --> FOR THIS APPROACH THE BOX SHOULD NOT BE A PARALLELOGRAM ETC
     */
    CellData unitcell;
    ElleCellBBox(&unitcell); 
    int iMaxUnodes = ElleMaxUnodes(); 
    double dRoi = 0.0;
    double dBoxHeight = 0.0;
    double dBoxWidth = 0.0;
    double dBoxArea = 0.0;
    dBoxHeight = unitcell.cellBBox[TOPLEFT].y - unitcell.cellBBox[BASELEFT].y;
    dBoxWidth = unitcell.cellBBox[BASERIGHT].x - unitcell.cellBBox[BASELEFT].x;
    
    if (dBoxWidth>dBoxHeight)
        dBoxArea = dBoxWidth*dBoxWidth;
    else
        dBoxArea = dBoxHeight*dBoxHeight;
        
    //dBoxArea = dBoxHeight*dBoxWidth;    
    
	dRoi = sqrt( dBoxArea/ (double) iMaxUnodes / 3.142 ) * (double)iFact;	// aprox. limit at 2nd neighbours by using iFact = 3
    
    if (dRoi == 0)
    {
        printf("ERROR (FS_GetROI): Roi is zero!\n");
        return (0);
    }
    
    return (dRoi);
}

/*FS: 
 * Added a part to exclude air-air boundaries from error message of 
 * movements being too large: Move a air-air bnode always by the maximum
 * distance possible in that case
 * This function checks if a bnode is on an excluded boundary (i.e. if it is a 
 * phase-phase bounadry of the excluded phase (userdata[4])
 * 
 * Returns 1 if the boundary is excluded, 0 if not
 */
int FS_NodeOnExcludeBoundary(int iNode)
{
    UserData userdata;
    ElleUserData(userdata);
    
    int iExclPhase = (int)userdata[4];
    int iRgns[3]; 
    double dPhase[3];
    
    ElleRegions(iNode,iRgns);
    
    for (int i=0;i<3;i++)
    {
        if (iRgns[i]!=NO_NB)
        {
            ElleGetFlynnRealAttribute(iRgns[i],&dPhase[i],iFlynnPhase); 
            if (dPhase[i] != iExclPhase) return (0);
        }                
    }
    
    return (1);
}

/*
 * FS: A function to get the total box area, which is useful for the cluster 
 * tracking changes I did
 */
double FS_GetTotalBoxArea()
{
    CellData unitcell;
    ElleCellBBox(&unitcell);
    double dBoxArea = 0.0;
    
    Coords box_xy[4];
    
    box_xy[0].x = unitcell.cellBBox[BASELEFT].x;
    box_xy[0].y = unitcell.cellBBox[BASELEFT].y;
    box_xy[1].x = unitcell.cellBBox[BASERIGHT].x;
    box_xy[1].y = unitcell.cellBBox[BASERIGHT].y;
    box_xy[2].x = unitcell.cellBBox[TOPRIGHT].x;
    box_xy[2].y = unitcell.cellBBox[TOPRIGHT].y;
    box_xy[3].x = unitcell.cellBBox[TOPLEFT].x;
    box_xy[3].y = unitcell.cellBBox[TOPLEFT].y;
    
    // Calculate with gaussian euqation for polygon areas with n corners:
    // 2*Area = Σ(i=1 to n)  (y(i)+y(i+1))*(x(i)-x(i+1))
    // if i+1>n use i=1 again (or here in C++ i=0)
    int i2 = 0;
    for (int i=0;i<4;i++)    
    {
        i2 = fmod(i,4)+1;
        dBoxArea += ( (box_xy[i].y+box_xy[i2].y)*(box_xy[i].x-box_xy[i2].x) )/2;   
    }    
    return(dBoxArea);
}

double FS_ScaleDDForBasalActivity(int iUnode)
{
    /* FS:
     * This function loads dislocation density from an unode. It also checks the
     * basal activity for that unode (should be stored in U_ATTRIB_D) and 
     * scales the dislocation density to the amount of non-basal activity.
     * Non-basal dislocation have a 3.65 times higher line energy than basal 
     * ones due to their higher Burgers vector lengths.
     * 
     * According to how high the non basal activity is (0-100%), this function 
     * will increase the dislocation density (and i.e. the resulting line 
     * energy) by a factor of 1 (0% non-basal) to 3.65 (100% non-basal)
     *
     * THIS FUNCTION IS OPTIMIZED FOR ICE Ih!!!!!
     */
    double dDislocden = 0.0;
    double dBasalAct = 0.0, dNonBasalAct = 0.0;
    double dDDincreaseFact = 1.0;
    double dRatioLineEnergyNonBasalBasal = 3.65;
    
    ElleGetUnodeAttribute(iUnode,U_DISLOCDEN,&dDislocden);
    if (!ElleUnodeAttributeActive(U_ATTRIB_D)) return(dDislocden);
    
    ElleGetUnodeAttribute(iUnode,U_ATTRIB_D,&dBasalAct);
    dNonBasalAct = 1.0-dBasalAct;
    
    dDDincreaseFact = 1.0+( (dRatioLineEnergyNonBasalBasal-1.0)*dNonBasalAct);
    
    return(dDislocden*dDDincreaseFact);   
}

double FS_density_unodes(int iFlynn,Coords * cTrialXY,int iBnode)
{
    /* Returns the mean dislocation densitiy within a region of interest 
     * (ROI) around a trial position that was set to determine the GBM
     * driving forces for iBnode*/
	fstream fDataFile;
    fDataFile.open ( "Logfile_DensityUnodes.txt", fstream::out | fstream::app);
    
    double dRoi = FS_GetROI(3); // limit roi to approx. 1st neighbours
    double dDist = 0.0;
    double dDistTotal=0.0;
    double dTmpDD = 0.0;
    double dDislocden = 0.0;
    
    int iCounter = 0;    
    vector<int> vUnodeList;
    
    vUnodeList.clear();
	ElleGetFlynnUnodeList(iFlynn,vUnodeList);
    
    if (vUnodeList.size()>0)
    {
        /* Flynn has unodes, find the unodes that are within ROI */
        Coords cRefXY;
        dDist = dDistTotal = 0.0;
        iCounter = 0;
        for (int i;i<vUnodeList.size();i++)
        {
            ElleGetUnodePosition(vUnodeList[i],&cRefXY);
            ElleCoordsPlotXY(&cRefXY,cTrialXY);			  
            dDist = pointSeparation(&cRefXY,cTrialXY);
            
            if (dDist < dRoi)
            {
                if (bScaleSIBMToNonBasalAct) 
                    dTmpDD = FS_ScaleDDForBasalActivity(vUnodeList[i]);
                else
                    ElleGetUnodeAttribute(vUnodeList[i],U_DISLOCDEN,&dTmpDD);
                    
				dDislocden += dTmpDD*(dRoi-dDist);
				dDistTotal += (dRoi-dDist);
				iCounter++;
			}                        
        }
        
        if (iCounter==0)
        {
            /* Found no unodes in ROI, use dummy dislocden */
            printf("Stage: %u WARNING (FS_density_unodes): ",Settings_run.Count);
            printf("No unode in roi in flynn %u for bnode %u: ",iFlynn,iBnode);
            printf("Using dummy dislocation density of %e m-2\n",dDummyDD);
            fDataFile << "Stage: "<<Settings_run.Count<< " ";
            fDataFile << "WARNING (FS_density_unodes): No unode in roi in flynn";
            fDataFile << iFlynn << " for bnode "<<iBnode<<": Using dummy ";
            fDataFile << "dislocation density of "<<dDummyDD<<" m-2"<<endl; 
            dDislocden = dDummyDD;
            dDistTotal = 1.0;             
        }       
    }
    else
    {
        /* Flynn has NO unodes:
         * Use mean dislocation density of the whole box
         * This should actually never happen, but keep this bit inside the code
         * to be on the safe side */
		printf("Stage: %u WARNING (FS_density_unodes): ",Settings_run.Count);
        printf("No Unodes found in Flynn (%u) for bode (%u)\n",iFlynn,iBnode);
        fDataFile << "Stage: " << Settings_run.Count;
        fDataFile << " WARNING (density_unodes_4): No Unodes found ";
        fDataFile << "in Flynn (" << iFlynn << ") for node (";
        fDataFile << iBnode << ")" << endl; 
        fDataFile << "--> Average across all unodes < roi regardless ";
        fDataFile << "of Flynns..." << endl;
        
        Coords cRefXY;
        dDist = dDistTotal = 0.0;
        iCounter = 0;
        for (int unode;unode<ElleMaxUnodes();unode++)
        {
            ElleGetUnodePosition(unode,&cRefXY);
            ElleCoordsPlotXY(&cRefXY,cTrialXY);			  
            dDist = pointSeparation(&cRefXY,cTrialXY);
            
            if (dDist < dRoi)
            {
                if (bScaleSIBMToNonBasalAct) 
                    dTmpDD = FS_ScaleDDForBasalActivity(unode);
                else
                    ElleGetUnodeAttribute(unode,U_DISLOCDEN,&dTmpDD);
                    
				dDislocden += dTmpDD*(dRoi-dDist);
				dDistTotal += (dRoi-dDist);
				iCounter++;
			}                        
        }
        
        if (iCounter==0)
        {
            /* STILL no unodes in roi, use dummy dislocation density */
            printf("Stage: %u WARNING (FS_density_unodes): ",Settings_run.Count);
            printf("No unode in roi regardless of flynns for bnode %u: ",iBnode);
            printf("Using dummy dislocation density of %e m-2\n",dDummyDD);
            fDataFile << "Stage: "<<Settings_run.Count<< " ";
            fDataFile << "WARNING (FS_density_unodes): No unode in roi ";
            fDataFile << "regardles for flynns for bnode "<<iBnode<<": Using dummy ";
            fDataFile << "dislocation density of "<<dDummyDD<<" m-2"<<endl; 
            dDislocden = dDummyDD;
            dDistTotal = 1.0;           
        }        
    }
    
    fDataFile.close();
    
    /* Finally take mean dislocation density*/
    dDislocden = dDislocden/dDistTotal;
    
    return(dDislocden);   
}
