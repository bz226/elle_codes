#include "FS_topocheck.h"
using namespace std;

#define PI 3.141592654 
// for some reason must be declared here, I do not know why it was not possible in header file:
int FindNoNeighbourBnodes(int iBnode, int iFlynn, vector<int> vFlynnBnodes, vector<int> &vBnodesNoNb);

/*
 * This function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
//int InitThisProcess()
//{
    //char *infile;
    //int err=0;
    ///*
     //* Clear the data structures
     //*/
    //ElleSetRunFunction(TopologyChecks);
    ///*
     //* Read the data
     //*/
    //infile = ElleFile();
    //if (strlen(infile)>0) 
    //{
        //if (err=ElleReadData(infile)) OnError(infile,err);
    //}     
//}

int TopologyChecks()
{
    /* First of all check for errors that we cannot resolve with this 
     * topology check: Island flynn (flynn inside another flynn) */
    if(CheckforIslandFlynns())
    {
        printf("Correct this issue(s) manually within the Elle file\n");        
        return 1;
    }
    
    int iError=0;
    bool bPrintUserInfo = false;
    
    /* Decide if unode attribute update is needed, only makes sense if 
     * U_ATTRIB_C is storing the ID of the flynn that contains the unode,
     * F_ATTRIB_C stores the flynn ID and -of course- unodes are active*/
    bool bNeedUnodeUpdate=true; 
    if (!ElleUnodeAttributeActive(U_ATTRIB_C)) bNeedUnodeUpdate=false;
    if (!ElleUnodesActive()) bNeedUnodeUpdate=false;
    if (!ElleFlynnAttributeActive(F_ATTRIB_C)) bNeedUnodeUpdate=false;
    
    /* Decide if a logfile should be created storing unode orientation before 
     * and after being swept by a boundary: Recommend to only switch it on in 
     * GBM code */
    bool bWriteUnodeOri2File=false; 
    
    if(bPrintUserInfo) 
        printf("# # # Topology Checks # # #\n");
    
    if (bNeedUnodeUpdate)
    {
        if(bPrintUserInfo) printf("Checking unodes ...\n");
        UpdateUnodesAndAttribs(bWriteUnodeOri2File);
    }
    
   /*
     * Work the way through all topology check functions (see the functions for 
     * further descriptions):
     * THE ORDER OF THE FUNCTION SHOULD MAYBE CHANGE?
     * OLD ORDER:
     * Delete small flynns
     * Add doubles
     * elle gg topochecks
     * check nodes coincide
     * checkangle
     * check if flynn needs split
     */
    
    if(bPrintUserInfo) printf("Detecting and deleting small flynns ...\n");
    DeleteSmallFlynns();  
    
    // This is already done in ElleGGTopoChecks:
    //printf("Adding double nodes where necessary ...\n");
    ElleAddDoubles();
    
	if(bPrintUserInfo) printf("Checking for coinciding boundary nodes ...\n");
    iError = CheckNodesCoincide2(); 
    //if (iError) OnError("Error in CheckNodesCoincide",iError);
    
    if(bPrintUserInfo) printf("Performing standard elle_gg topology checks ...\n");
    ElleGGTopoChecks();
    
    //printf("Checking for coinciding boundary nodes ...\n");
    //iError = CheckNodesCoincide(); 
    //if (iError) OnError("Error in CheckNodesCoincide",iError);
    
    double dMinAngle = 20; // Type the angle in degrees, it is re-calculated afterwards
    if(bPrintUserInfo) printf("Checkangle with %f or %f° ...\n",dMinAngle/(180/PI),dMinAngle);
    CheckAngle(dMinAngle/(180/PI));
    
    if(bPrintUserInfo) printf ("Checking if flynn needs split due to boundaries being too close ...\n");
    CheckIfFlynnsNeedSplit();
    
    /*
     * Topology Checks finished
     */
    if (bNeedUnodeUpdate)
    {
        if(bPrintUserInfo) printf("Checking unodes ...\n");
        UpdateUnodesAndAttribs(bWriteUnodeOri2File);
    }
    
    //iError=ElleWriteData("FS_topocheck.elle");
    //if(iError) OnError("Error while saving the new Elle file",iError);
    
    if(bPrintUserInfo)
        printf("Topology checks finished ...\n");
    
    return 0;
}


/*
 * Function performing all elle_gg topology checks, those are:
 * 
 * + ElleNodeTopologyCheck (which is not in elle_gg I assume)
 * + ElleDeleteSingleJ (also not in elle_gg I assume)
 * 
 */
void ElleGGTopoChecksOLD()
{
    bool FSDEBUG = false;
    /*
     * Add double bnodes, where the gap is too large:
     */
    ElleAddDoubles();

    /*
     * Check every bnode (double or triple junction checks):
     */
    for (int node=0; node<ElleMaxNodes(); node++)
    {
        /*
         * After every step we need to check if the node is still active, it is
         * possible that some topo checks remove it
         */        
        if (ElleNodeIsActive(node))
            ElleDeleteSingleJ(node);     
               
        if (ElleNodeIsActive(node)) 
        {
            if (FSDEBUG) printf("line: %u, node: %u\n",__LINE__,node);
            ElleNodeTopologyCheck(node);
            if (FSDEBUG) printf("line: %u, node: %u\n",__LINE__,node);
        }
        
        if (ElleNodeIsActive(node)) 
        {
            if (ElleNodeIsDouble(node))
            {
                if (FSDEBUG) printf("line: %u, node: %u\n",__LINE__,node);
                ElleCheckDoubleJ(node);
                if (FSDEBUG) printf("line: %u, node: %u\n",__LINE__,node);
            }
            else if (ElleNodeIsTriple(node)) 
            {
                if (FSDEBUG) printf("line: %u, node: %u\n",__LINE__,node);
                ElleCheckTripleJ(node);
                if (FSDEBUG) printf("line: %u, node: %u\n",__LINE__,node);
            }
        }
    }
}

/*
 * CheckNodesCoincide:
 * Checks for all nodes if there is a node overlying one of its neighbour 
 * nodes (if they conincide). If "yes", the function gives an user message 
 * and repeats ElleGGTopoChecks()
 */
int CheckNodesCoincide()
{
    int iError = 0;
    int iNbNodes[3];
    
    for (int node=0; node<ElleMaxNodes(); node++)
    {
        if (ElleNodeIsActive(node))
        {
            iError = ElleNeighbourNodes(node,iNbNodes); // find neighbour nodes
            
            for (int j=0; j<3; j++)
            {
                /* 
                 * Do something if there is a j-th neighbour, this neighbour 
                 * node is active AND MOST IMPORTANTLY: If it conincides with
                 * the node of interest!
                 */                
                if ( iNbNodes[j]!=NO_NB && 
                     ElleNodeIsActive(iNbNodes[j]) &&
                     ElleNodeIsActive(node) &&
                     ElleNodesCoincident(node,iNbNodes[j]))
                {
                    printf("Node %u and node %u are conincident",node,iNbNodes[j]);
                    ElleGGTopoChecks();
                    
                    /*
                     * Check if after topocheck the nodes are still 
                     * conindicent: If one of them is not active any more, the
                     * function assumes that they are not conincident as well
                     * and outputs "solved".
                     */
                    if ( ElleNodeIsActive(node) && 
                         ElleNodeIsActive(iNbNodes[j]) )
                    {
                        if (ElleNodesCoincident(node,iNbNodes[j])) 
                            printf(" - Not solved\n");
                        else
                            printf(" - Solved!\n");
                    }
                    else
                        printf(" - Solved!\n");                       
                }
            }
        }
    }
    
    return (iError);
}

/*
 * CheckNodesCoincide2: Based on tricky_vs02 by Albert
 * In general has the same purpose than CheckNodesCoincide (number 1): If two
 * bnodes are at exactly the same position: Resolve this problem. However the
 * way to solve it is different: If two bnodes are overlapping, the program 
 * shifts them now by a small increment. Even though they are then still too 
 * close to each others, an ElleGG Topocheck wil resolve this afterwards.
 * --> Before this it has to be checked if the moved node is outside the unit 
 * cell
 */
int CheckNodesCoincide2()
{
	int *ids,num;
	Coords initialxy, xy1, xy2, newxy, incrxy;
	double switchd, dist, increm, dist1, dist2;
	switchd = ElleSwitchdistance()/5.0;
	
	increm= switchd;
	
	for (int i=0;i<ElleMaxFlynns();i++) 
   	{
		if (ElleFlynnIsActive(i)) 	  
		{  
			ElleFlynnNodes(i, &ids, &num);
			
			for (int j = 0; j<num-1; j++ )
			{
				// The "j" bnode is the one that will be potentially moved by a small increment
				for (int k=j+1; k<num; k++)
				{
					// printf("j k %i %i\n", ids[j],ids[k]);
					if ( ElleNodeIsActive( ids[j] ) && ElleNodeIsActive( ids[k] )) 
					{
						ElleNodePosition( ids[j], & xy1 );
						ElleNodePosition( ids[k], & xy2 );
						
						// store initial position since xy1 will be changed later
						initialxy.x = xy1.x;
						initialxy.y = xy1.y;

						dist = pointSeparation(&xy1,&xy2);
						// printf("dist %lf\n", dist);

						if (dist==0.0)//if (dist< switchd) 
						{
							printf("Solving problem with overlapping or coinciding bnodes %u and %u\n",ids[j],ids[k]);
							incrxy.x=incrxy.y=0.0;					
							//printf("node1 node2 %i %i\n", ids[j], ids[k]);
							//printf(" %lf %lf\n", xy1.x, xy1.y );
							//printf(" %lf %lf\n", xy2.x, xy2.y );
							// + 
							xy1.x += increm;
							xy1.y += increm;
							dist1 = pointSeparation(&xy1,&xy2);		

							// - 
							xy1.x -= 2*increm;
							xy1.y -= 2*increm;
							dist2 = pointSeparation(&xy1,&xy2);					


							if ( dist1 < dist2) 
							{
								incrxy.x -=increm;
								incrxy.y -=increm;
							} 
							else 
							{				
								incrxy.x +=increm;
								incrxy.y +=increm;					
							}				
							//printf(" %lf %lf\n", incrxy.x,incrxy.y );							
							//ElleUpdatePosition( ids[j], & incrxy );
							
							/* FS: New: Do not shift the node by an increment,
							 * but directly determine its new position that has
							 * to be in the unit cell (i.e. making use of 
							 * ElleNodeUnitXY */
							newxy.x = initialxy.x + incrxy.x;
							newxy.y = initialxy.y + incrxy.y;
							ElleNodeUnitXY(&newxy); // check that new position is inside the unit cell
							ElleNodePosition(ids[j],&newxy);

							if (ElleNodeIsDouble( ids[j] ))
								ElleCheckDoubleJ( ids[j] );
							else if (ElleNodeIsTriple( ids[j] ))
								ElleCheckTripleJ( ids[j] );
						}	
					}
				}			
			}
			free(ids); 		
		}
	}

	// add un check of bnodes and position 
}

/*
 * FS: Created by Jens: Detects very tiny flynns with the following criteria:
 * 1.: 
 * If a flynn has not enough bnodes (depending on SwitchDinstance and number of 
 * unodes) and additionally less unodes than a certain value (here zero) inside
 * -OR:-
 * 2.:
 * Area is too small (essentially smaller than the minimum area between four 
 * neighbouring unodes) 
 * 
 * ATTENTION:
 * If there are no unodes in the file, all flynns containing 3 or less (it is
 * actually not possible to have less than 3 bnodes) bnodes are removed
 */
int DeleteSmallFlynns( void )
{
	int iFlynnPhase = VISCOSITY;
    bool bDeleteFlynn = false;
    bool bMergeDifferentPhases = true; // Merge flynn with neighbour flynn of different phase, if there are no neighbours of the same phase
    int iRemoved=0, *id=0, num_nodes=5, iCheck, iFound, iMaxNodes;
    int iSizeUnodeList = 0;
	double dPhase[2], dMinArea = 0.0, dFlynnArea=0.0;
	vector<int> vUnodeList;
	list<int> lNbFlynns;
	fstream fDataFile;
    
	fDataFile.open ( "LogfileTopoChecks_DeletedFlynns.txt", fstream::out | fstream::app);
	
	fDataFile << "# # # Topology-Check Logfile # # #\n" << endl;
	
    /*
     * Determine minimum number of bnodes for a flynn before deletion:
     * --> Flynns with number of bnodes <= this number will be deleted, if not 
     * containing unodes
     * --> The minimum number is calculated differently, when no unodes are in 
     * the file 
     */
    if (!ElleUnodesActive())
    {
        /*
         * If no unodes are there use the value for a file with 256 unodes and
         * a switch distance of 2.5e-3, which is 8 bnodes or use 3 because
         * it is not possible to have a flynn with less than 3 bnodes
         */
        iMaxNodes = 3;
    }
    else
    {
        iMaxNodes = (int) round( 2.5 / ElleSwitchdistance() * ( 1 / sqrt ( ElleMaxUnodes() ) ) );
    }
    
    /*
     * Determine minimum area of flynn, all flynns smaller than this will be 
     * deleted
     */
    if (!ElleUnodesActive())
    {
        /*
         * If there are no unodes: Set the minimum area to the area of an
         * equilateral triangle where the length of one side a = switchdist
         *
         * Area = sqrt(3) * (switchdist^2) /4 
         */
        dMinArea = sqrt(3)*pow(ElleSwitchdistance(),2)/4;
        //cout << "dMinArea = " << dMinArea << endl;
    }
    else
    {
        dMinArea = 1/(double)ElleMaxUnodes(); // Which is the minimum area between four neighbouring unodes in a square grid
    }	
	
	for ( int i = 0; i < ElleMaxFlynns(); i++ ) 
   	{
		iCheck = 0; 
        if (ElleFlynnIsActive(i)) 
		{
			iFound = 0;
			vUnodeList.clear();
            dFlynnArea = ElleRegionArea(i);
			
			ElleFlynnNodes( i, &id, &num_nodes );
                     
            ElleGetFlynnUnodeList( i, vUnodeList );
            iSizeUnodeList=vUnodeList.size(); 
            
            /*
             * Now check if the flynns needs to be deleted following the 
             * criterions below:
             * Criterions are different for files with and without unodes 
             * because the number of unodes will ALWAYS be zero if there are no
             * unodes in the file, which of course then does not necesserily 
             * require a flynn deletion
             */
            bDeleteFlynn = false;
            if (!ElleUnodesActive())
            {
                if (num_nodes <= iMaxNodes || dFlynnArea < dMinArea)
                    bDeleteFlynn = true;
            }
            else
            {
                if ( (num_nodes <= iMaxNodes || iSizeUnodeList == 0) || dFlynnArea < dMinArea) // FS: changed that, first "||" was "&&" before!
                    bDeleteFlynn = true;
            }
            
            /* New: Check for dangling flynn that has the following 
             * characteristics:
             * (1) Has only 2 neighbour flynns
             * (2) Has only 2 triple nodes
             * (3) These triple nodes are neighbours
             *
             * and delete it */
            bool bIsDangling=false;
            if (FS_FlynnIsDangling(i)) 
            {
                printf("Deleting dangling flynn %u\n",i);
                bDeleteFlynn=true;
                bIsDangling=true;
            }
            
			if (bDeleteFlynn) 
			{
                lNbFlynns.clear();
				
				ElleGetFlynnRealAttribute( i, &dPhase[0], iFlynnPhase );
				ElleFlynnNbRegions( i, lNbFlynns );
                
                int iFirstNb = lNbFlynns.front();
                
				//fDataFile << iCheck << " " << i << " " << lNbFlynns.front() << endl;
				while ( lNbFlynns.size() > 0 && iFound == 0 ) 
				{
					ElleGetFlynnRealAttribute( lNbFlynns.front(), &dPhase[1], iFlynnPhase );
					
					if ( (int) dPhase[ 0 ] == (int) dPhase[ 1 ] )
					{
						iFound = 1;
						
						iCheck = MergeSmallFlynn( lNbFlynns.front(), i );
												
						if ( iCheck == 1 ) 
                        {
							fDataFile << "INFO (DeleteSmallFlynns): Small Flynn " << i << " merged with " << lNbFlynns.front() << endl; 
                            if (bIsDangling)
                                fDataFile << "Flynn " << i << " is dangling flynn\n --> Only 2 neighbour flynns + the only 2 triple nodes are neighbours" << endl;
                            else
                                fDataFile << "Flynn " << i << " has " << num_nodes << " Bnodes and " << iSizeUnodeList << " or less Unodes" << endl;
                            fDataFile << "MergeSmallFlynn returned " << iCheck << endl; 
							iRemoved++;
						}
						else
                        {
							//fDataFile << iCheck << " Flynn: " << i << " Nodes: " << num_nodes << " Unodes: " << iSizeUnodeList << " or less" << endl;
                            fDataFile << "INFO (DeleteSmallFlynns): Small Flynn " << i << " merged with " << lNbFlynns.front() << endl; 
                            if (bIsDangling)
                                fDataFile << "Flynn " << i << " is dangling flynn\n --> Only 2 neighbour flynns + the only 2 triple nodes are neighbours" << endl;
                            else
                                fDataFile << "Flynn " << i << " has " << num_nodes << " Bnodes and " << iSizeUnodeList << " or less Unodes" << endl;
                            fDataFile << "MergeSmallFlynn returned " << iCheck << endl; 
							iRemoved++;
                        }
							
						if ( iCheck == 4 ) 
                        {
							ElleWriteData("DeleteSmallFlynns_Error.elle");
							
							fDataFile << "ERROR (4): Flynn " << i << " not merged with Flynn " << lNbFlynns.front() << ". Check returned not matching Flynns on both sides of the boundary." << endl;
						}
					}
					
					lNbFlynns.pop_front();
				}
				if ( iFound == 0 ) // NO neighbour flynn has the same phase
                {
                    /* No neighbour flynn has the same phase: 
                     * 
                     * Either do not merge the small flynn with any neighbour:
                     */
                    fDataFile << "WARNING (DeleteSmallFlynns): Small Flynn " << i << " has no neighbour with the same phase..." << endl;     
                    /*
                     * Or merge anyway with the first neighbour flynn:
                     */
                    if (bMergeDifferentPhases)
                    {
                        iCheck = MergeSmallFlynn( iFirstNb, i );
                                                    
                        if ( iCheck == 1 ) 
                        {
                            fDataFile << "INFO (DeleteSmallFlynns): Small Flynn " << i << " merged with " << iFirstNb << " regardless of phases" << endl; 
                            if (bIsDangling)
                                fDataFile << "Flynn " << i << " is dangling flynn\n --> Only 2 neighbour flynns + the only 2 triple nodes are neighbours" << endl;
                            else
                                fDataFile << "Flynn " << i << " has " << num_nodes << " Bnodes and " << iSizeUnodeList << " or less Unodes" << endl;
                            fDataFile << "MergeSmallFlynn returned " << iCheck << endl;
                            iRemoved++;
                        }
                        else
                        {
                            //fDataFile << iCheck << " Flynn: " << i << " Nodes: " << num_nodes << " Unodes: " << iSizeUnodeList << " or less" << endl;
                            fDataFile << "INFO (DeleteSmallFlynns): Small Flynn " << i << " merged with " << iFirstNb << " regardless of phases" << endl; 
                            if (bIsDangling)
                                fDataFile << "Flynn " << i << " is dangling flynn\n --> Only 2 neighbour flynns + the only 2 triple nodes are neighbours" << endl;
                            else
                                fDataFile << "Flynn " << i << " has " << num_nodes << " Bnodes and " << iSizeUnodeList << " or less Unodes" << endl;
                            fDataFile << "MergeSmallFlynn returned " << iCheck << endl;
							iRemoved++;
                        }
                            
                        if ( iCheck == 4 ) 
                        {
                            ElleWriteData("DeleteSmallFlynns_Error.elle");
                            
                            fDataFile << "ERROR (4): Flynn " << i << " not merged with Flynn " << iFirstNb << ". Check returned not matching Flynns on both sides of the boundary." << endl;
                        }
                    }
                }
			}
			if ( id ) free ( id );
		}
        if (iCheck !=0) fDataFile << endl; // break line before next flynn
	}
	fDataFile << iRemoved << " Flynns removed. (MaxNodes: " << iMaxNodes << ", MinArea: " << scientific << dMinArea << ")" << endl;
	fDataFile.close();
	
	return ( iRemoved );
}

/*
 * FS: Created by Jens: If a small flynn in DeleteSmallFlynns is detected, it 
 * needs to be removed by merging it with another, larger flynn. This is what 
 * MergeSmallFlyn does.
 */
int MergeSmallFlynn( int iFlynn1, int iFlynn2 )
{
	// Flynn2 gets deleted and merged into Flynn1
	
	int *iNodes1=0, *iNodes2=0;
	int iNum1, iNum2, rmindx2, rmindx1, iTriple1, iTriple2;
	ERegion rCheck1, rCheck2;
	vector<int> vBoundaryNodes, vTriples;
	fstream fDataFile;
	
	ElleFlynnNodes( iFlynn1, &iNodes1, &iNum1 );
	ElleFlynnNodes( iFlynn2, &iNodes2, &iNum2 );
	
	// find all Nodes along the common boundary and put them in the vector.

	vBoundaryNodes.clear();
	
	for ( int i = 0; i < iNum1; i++ )
	{
		for ( int j = 0; j < iNum2; j++ )
		{
			if ( iNodes1[i] == iNodes2[j] )
				vBoundaryNodes.push_back( iNodes1[i] );
		}	
	}
	
	// delete all double nodes and put the triples in a new vector
	
	vTriples.clear();

	while ( vBoundaryNodes.size() > 0 )
	{
		if ( ElleNodeIsDouble( vBoundaryNodes.back() ) )
		{
			if (ElleDeleteDoubleJNoCheck( vBoundaryNodes.back() ) != 0)
            {
                fDataFile.open ( "LogfileTopoChecks_MergeFlynn.txt", fstream::out | fstream::app);
				fDataFile << "ERROR (MergeSmallFlynn): Double Node not deleted... " << vBoundaryNodes.back() << endl;
                fDataFile.close();
            }
			vBoundaryNodes.pop_back();
		}
		else if ( ElleNodeIsTriple( vBoundaryNodes.back() ) )
		{
			vTriples.push_back( vBoundaryNodes.back() );
			vBoundaryNodes.pop_back();
		}
		else
		{
			fDataFile.open ( "LogfileTopoChecks_MergeFlynn.txt", fstream::out | fstream::app);
            fDataFile << "ERROR (MergeSmallFlynn): Unkown node Type... " << vBoundaryNodes.back() << endl;
            fDataFile.close();
			vBoundaryNodes.pop_back();
		}
	}
	
	// if there are more than 2 triple nodes in the vector return an error...

	if ( vTriples.size() != 2 ) 
	{
		if ( iNodes1 ) free( iNodes1 );
		if ( iNodes2 ) free( iNodes2 );
		fDataFile.close();
		return 2;
	}
		
	iTriple1 = vTriples.front();
	iTriple2 = vTriples.back();
	
	
	ElleNeighbourRegion( iTriple1, iTriple2, &rCheck1 );
	ElleNeighbourRegion( iTriple2, iTriple1, &rCheck2 );

	if ( rCheck1 == iFlynn2 )
	{
		if ( rCheck2 == iFlynn1 )
			;// alles ok
		else
		{
			if ( iNodes1 ) free( iNodes1 );
			if ( iNodes2 ) free( iNodes2 );
			fDataFile.close();
			return 3;
		}
	}
	else if ( rCheck1 == iFlynn1 )
	{
		if ( rCheck2 == iFlynn2 )
		{
			// switch triples...
			iTriple2 = vTriples.front();
			iTriple1 = vTriples.back();
		}
		else
		{
			if ( iNodes1 ) free( iNodes1 );
			if ( iNodes2 ) free( iNodes2 );
			fDataFile.close();
			return 3;
		}
	}
	else
	{
		if ( iNodes1 ) free( iNodes1 );
		if ( iNodes2 ) free( iNodes2 );
		fDataFile.close();
		return 4;
	}
		
	rmindx1 = ElleFindBndIndex( iTriple1, iFlynn2 );
	rmindx2 = ElleFindBndIndex( iTriple2, iFlynn1 );
	
	if ( rmindx1 == NO_NB || rmindx2 == NO_NB )
    {
        fDataFile.open ( "LogfileTopoChecks_MergeFlynn.txt", fstream::out | fstream::app);
		fDataFile << "ERROR (MergeSmallFlynn): Topology problems..." << endl;
        //OnNodeError( vTriples.front(), msg, NONB_ERR );
        fDataFile.close();
    }

	ElleClearNeighbour( iTriple1, rmindx1 );
	ElleClearNeighbour( iTriple2, rmindx2 );
	
	rmindx2 = ElleFindBndIndex( iTriple2, iFlynn2 );

	ElleSetRegionEntry( iTriple2, rmindx2, iFlynn1 );
	
	//set all the other nodes of former iFlynn2 to iFlynn1

	for ( int i = 0; i < iNum2; i++ )
	{
		if ( ElleNodeIsActive( iNodes2[ i ] ) && iNodes2[ i ] != iTriple1 && iNodes2[ i ] != iTriple2 )
		{
			rmindx2 = ElleFindBndIndex( iNodes2[ i ], iFlynn2 );
			//ElleClearNeighbour( iNodes2[ i ], rmindx2 );
			ElleSetRegionEntry( iNodes2[ i ], rmindx2, iFlynn1 );
		}
	}

	ElleRemoveShrinkingFlynn( iFlynn2 );

	//Check for wrapping Flynn.. Shouldn't be the case because only small Fynns get removed...
	if ( ElleIdMatch( &iFlynn1, &iFlynn2 ) )
    {
        fDataFile.open ( "LogfileTopoChecks_MergeFlynn.txt", fstream::out | fstream::app);
		fDataFile << "ERROR (MergeSmallFlynn): Wrapping Flynn..." << endl;
        //OnError(msg,RGNWRP_ERR);
        fDataFile.close();
    }
	
	if ( iNodes1 ) free( iNodes1 );
	if ( iNodes2 ) free( iNodes2 );
	
	fDataFile.close();
	
	return 1;
}

/*
 * From the checkangle process:
 */
void CheckAngle(double min_ang)
{
    int moved=1, removed=1,i,j,k,max, count=0;
    int nbnodes[3], nb[3], same, min_type;
    double currang,flynn_area;
    
    double ang,dist;
    Coords xy[3], movedist;
    int *ids, num;

    max = ElleMaxNodes();
    while (moved)  {
    for (k=0,moved=0;k<max;k++) {
        if (ElleNodeIsActive(k)) {
            ElleNodePosition(k,&xy[0]);
            ElleNeighbourNodes(k,nbnodes);
            if (ElleNodeIsDouble(k)) {
                j=0; i=1;
                while (j<3) {
                    if (nbnodes[j]!=NO_NB){
                        nb[i]=nbnodes[j];
                        ElleNodePlotXY(nbnodes[j],&xy[i++],&xy[0]);
                    }
                    j++;
                    
                }
                angle0(xy[0].x,xy[0].y,xy[1].x,xy[1].y,xy[2].x,xy[2].y,
                                                             &currang);
                //if(currang==0) OnError("angle error",0);
                        
                ang = fabs(currang);
                  
                if (ang<min_ang && (EllePhaseBoundary(k,nb[1])==1)){
                  if( EllePhaseBoundary(k,nb[2])==1 ){
                    printf("Problem: both boundaries are two phase boundaries\n");
                  }
                  else 
                  {
                    count=0;
                    do {
                        IncreaseAngle(&xy[0],&xy[1],&xy[2],&movedist);
                        xy[0].x += movedist.x;
                        xy[0].y += movedist.y;
                        angle0( xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                                            xy[2].x,xy[2].y,&currang);
                        //if (currang==0)   OnError("angle error",0);
                        ang = fabs(currang);
                        dist=(xy[0].x-xy[1].x)*(xy[0].x-xy[1].x)+
                                 (xy[0].y-xy[1].y)*(xy[0].y-xy[1].y);
                        dist=sqrt(dist);
                        count++;
                     } while (ang<min_ang  &&
                                 (dist>ElleSwitchdistance()*0.1));
                     if (count>1) {
                         // moved at least one step before dist too small
                         ElleSetPosition(k,&xy[0]);
                         ElleCheckDoubleJ(k);
                         moved = 1;
                         printf("movedd1 %d\t\n",k);
                     }
                  }
                    
                 }
                 else if (ang<min_ang /*&& (EllePhaseBoundary(k,nb[2])==1)*/)
                 {
                    count=0;
                    do {
                        IncreaseAngle(&xy[0],&xy[2],&xy[1],&movedist);
                        xy[0].x += movedist.x;
                        xy[0].y += movedist.y;
                        angle0( xy[0].x,xy[0].y,xy[2].x,xy[2].y,
                                            xy[1].x,xy[1].y,&currang);
                        //if (currang==0)   OnError("angle error",0);
                        ang = fabs(currang);
                        dist=(xy[0].x-xy[2].x)*(xy[0].x-xy[2].x)+
                                 (xy[0].y-xy[2].y)*(xy[0].y-xy[2].y);
                        dist=sqrt(dist);
                        count++;
                     } while (ang<min_ang  &&
                                 (dist>ElleSwitchdistance()*0.1));
                     if (count>1) {
                         // moved at least one step before dist too small
                        ElleSetPosition(k,&xy[0]);
                        ElleCheckDoubleJ(k);
                        moved = 1;
                        printf("movedd2 %d\t\n",k);
                     }
                  }
               }
               else if (ElleNodeIsTriple(k)) {
                for (j=0;j<3 ;j++) {
                    i = (j+1)%3;
                    ElleNodePlotXY(nbnodes[j],&xy[1],&xy[0]);
                    ElleNodePlotXY(nbnodes[i],&xy[2],&xy[0]);
                      angle0(xy[0].x,xy[0].y,xy[1].x,xy[1].y,xy[2].x,xy[2].y,
                                                             &currang);
                      //if (currang==0)  OnError("angle error",0);
                      ang = fabs(currang);

                      if (ang<min_ang && (EllePhaseBoundary(k,nbnodes[j])==1)){
                        if((EllePhaseBoundary(k,nbnodes[j])==1)&&
                           (EllePhaseBoundary(k,nbnodes[i])==1)){
                           printf("Problem: both boundaries are two phase boundaries\n");
                        }
                      else {

                   // if (ang<min_ang /*|| ang>(M_PI-min_ang)*/) 
                        
                        count=0;
                        do {
                            IncreaseAngle(&xy[0],&xy[1],&xy[2],&movedist);
                            xy[0].x += movedist.x;
                            xy[0].y += movedist.y;
                            angle0(xy[0].x,xy[0].y,xy[1].x,xy[1].y,
                                            xy[2].x,xy[2].y,&currang);
                            //if (currang==0)   OnError("angle error",0);
                            ang = fabs((double)currang);
                            //printf("ang %lf\t\n",ang);
                            dist=(xy[0].x-xy[1].x)*(xy[0].x-xy[1].x)+
                                 (xy[0].y-xy[1].y)*(xy[0].y-xy[1].y);
                            dist=sqrt(dist);
                            count++;
                        } while (ang<min_ang  &&
                                    (dist>ElleSwitchdistance()*0.1)); 
                        if (count>1) {
                         // moved at least one step before dist too small
                            ElleSetPosition(k,&xy[0]);
                            ElleCheckTripleJ(k);
                            moved = 1;
                            printf("movedt1 %d\t\n",k);
                        }
                        j=3;
                      }
                    }
                  else if (ang<min_ang /*&& (EllePhaseBoundary(k,nbnodes[i])==1)*/)
                    {
                       count=0;
                       do {
                            IncreaseAngle(&xy[0],&xy[2],&xy[1],&movedist);
                            xy[0].x += movedist.x;
                            xy[0].y += movedist.y;
                            angle0( xy[0].x,xy[0].y,xy[2].x,xy[2].y,
                                            xy[1].x,xy[1].y,&currang);
                            //if (currang==0)   OnError("angle error",0);
                            ang = fabs(currang);
                            //printf("ang %lf\t\n",ang);
                            dist=(xy[0].x-xy[2].x)*(xy[0].x-xy[2].x)+
                                 (xy[0].y-xy[2].y)*(xy[0].y-xy[2].y);
                            dist=sqrt(dist);
                            count++;
                        } while (ang<min_ang &&
                                    (dist>ElleSwitchdistance()*0.1));
                        if (count>1) {
                         // moved at least one step before dist too small
                            ElleSetPosition(k,&xy[0]);
                            ElleCheckTripleJ(k);
                            moved = 1;
                            printf("movedt2 %d\t\n",k);
                        }
                        j=3;
                    }
                  }
                }
            }
        }
    }
}
/*
 * Used by CheckAngle:
 */
int IncreaseAngle(Coords *xy,Coords *xy1,Coords *xy2,Coords *diff)
{
    Coords xynew;

    /*
     * 
     * move 0.1 of the distance towards neighbour 1 along boundary
     */
    
    diff->x = xy1->x - xy->x;
    diff->y = xy1->y - xy->y;
    diff->x *= 0.1*ElleSwitchdistance();
    diff->y *= 0.1*ElleSwitchdistance();
}
    
/*
 * Function checking of there are two NON NEIGHBOUR bnodes of one flynn, that 
 * are so close together, that the flynn has to be splitted in two flynns.
 * Split is necessary, if distance of bnodes < ElleSwitchDist. Only the bnodes
 * with the smallest distance are used as start of the new flynn.
 * --> THE BNODES USED FOR SPLITTING MUST BE DOUBLE NODES TO AVOID QUADRUPLE 
 * JUNCTIONS
 */
void CheckIfFlynnsNeedSplit()
{
    int iSplitnode1 = 0,iSplitnode2 = 0;
    int iNewFlynnID = -1; // initialise to -1 to be sure there are no errors
    // Write logfile of how many splits happen with following columns:
    // 1 split_type phase_mergedgrain1 phase_mergedgrain2 id_mergedgrain1 id_mergedgrain2
    // split_type=1: rotation recrystallisation (never in this code)
    // split_type=2: grain dissection (always in this code)
    bool bLogSplits=true;  
    
    // For topolgy checks after a split:
    vector<int> vNodesOld;
    vector<int> vNodesNew;
    int iTmpNode = 0;
    
    // For Logfile
	fstream fDataFile;
    
	fDataFile.open ( "LogfileTopoChecks_SplitFlynns.txt", fstream::out | fstream::app);
	
	fDataFile << "# # # Topology-Check Logfile - CheckIfFlynnsNeedSplit() # # #\n" << endl;
    
    for (int flynn=0; flynn<ElleMaxFlynns(); flynn++)
    {
        if (ElleFlynnIsActive(flynn))
        {            
            if (NoNbNodesTooClose(flynn,&iSplitnode1,&iSplitnode2))
            {             
                /* 
                 * NoNbNodesTooClose returned 1: 
                 * The two split nodes will be the closest no-neighbour double 
                 * nodes in the flynn and their distance is below the switch 
                 * dist: Hence There is a need to split the flynn between them:
                 */
                /*
                 * First check if the line of splitting would lie in the flynn
                 * or not: If it is outside: Do not perform the split: Errors
                 * will occur updating the flynn attributes: Wait until the 
                 * flynn the line is in is called by the loop, here the split
                 * will be possible
                 */
                if (CheckTopoSplitPossible(iSplitnode1,iSplitnode2,flynn))
                {
                    /* Find phases of grains that merge if split tracking is
                     * switched on */
                    int iPhase1=0,iPhase2=0; 
                    int iMergeFlynn1=0,iMergeFlynn2=0;
                    if (bLogSplits)
                    {
                        int iNbFlynns1[3]; 
                        int iNbFlynns2[3]; 
                        double dPhaseTmp;
                        ElleRegions(iSplitnode1,iNbFlynns1);
                        ElleRegions(iSplitnode2,iNbFlynns2);
                        
                        // find phase 1, remember that splitnodes are still 
                        // double nodes:
                        for (int i=0;i<3;i++)
                        {
                            if (iNbFlynns1[i]!=NO_NB)
                                if (iNbFlynns1[i]!=flynn)
                                {
                                    iMergeFlynn1 = iNbFlynns1[i];
                                    ElleGetFlynnRealAttribute(iMergeFlynn1,&dPhaseTmp,VISCOSITY);
                                    iPhase1 = (int)dPhaseTmp;
                                }
                        }
                        // find phase 2, search for remaining flynn
                        for (int i=0;i<3;i++)
                        {
                            if (iNbFlynns2[i]!=NO_NB)
                                if (iNbFlynns2[i]!=flynn)
                                    if (iNbFlynns2[i]!=iMergeFlynn1)
                                {
                                    iMergeFlynn2 = iNbFlynns2[i];
                                    ElleGetFlynnRealAttribute(iMergeFlynn2,&dPhaseTmp,VISCOSITY);
                                    iPhase2 = (int)dPhaseTmp;
                                }
                        }
                    }
                    
                    /* Actually split grains*/                    
                    if (ElleNewGrain(iSplitnode1,iSplitnode2,flynn,&iNewFlynnID))
                        OnError("Error in ElleNewGrain",1);
                        
                    if (ElleFlynnAttributeActive(AGE))
                        UpdateFlynnAges(iNewFlynnID,flynn);
                        
                    fDataFile << "Flynn " << flynn << " was split along " <<
                        "nodes " << iSplitnode1 << " and "
                        << iSplitnode2 << " creating flynn " 
                        << iNewFlynnID << endl;
                    
                    /* Track split and phases of merged grains if this option
                     * is switched on */
                    if (bLogSplits)
                    {
                        fstream fSplitLog;
                        fSplitLog.open ( "Track_SplitEvents.txt", fstream::out | fstream::app);

                        fSplitLog << "1 2 " << iPhase1 << " " << iPhase2 << " "
                                  << iMergeFlynn1 << " " << iMergeFlynn2 << endl;
                        fSplitLog.close();                        
                    }
                    
                    // Topology checks after split:
                    ElleFlynnNodes(iNewFlynnID,vNodesNew);
                    ElleFlynnNodes(flynn,vNodesOld);
                        
                    // Topology checks after split in old flynn:
                    for (int i=0;i<vNodesOld.size();i++)
                    {
                        iTmpNode = vNodesOld[i];
                        if (ElleNodeIsActive(iTmpNode))
                        {
                            if (ElleNodeIsDouble(iTmpNode))
                                ElleCheckDoubleJ(iTmpNode);
                            else 
                            {   if (ElleNodeIsTriple(iTmpNode))
                                    ElleCheckTripleJ(iTmpNode);
                                else
                                {
                                    fDataFile << "Error in topo-check after " <<
                                        "split: Node " << iTmpNode << 
                                        " is neither DJ nor TJ." << endl;
                                }
                            }
                        }        
                    }    
                    // Topology checks after split in new flynn:
                    for (int i=0;i<vNodesNew.size();i++)
                    {
                        iTmpNode = vNodesNew[i];
                        if (ElleNodeIsActive(iTmpNode))
                        {
                            if (ElleNodeIsDouble(iTmpNode))
                                ElleCheckDoubleJ(iTmpNode);
                            else 
                            {   if (ElleNodeIsTriple(iTmpNode))
                                    ElleCheckTripleJ(iTmpNode);
                                else
                                {
                                    fDataFile << "Error in topo-check after " <<
                                        "split: Node " << iTmpNode << 
                                        " is neither DJ nor TJ." << endl;
                                }
                            }
                        }        
                    } // End of last topology check after split
                    vNodesNew.clear();
                    vNodesOld.clear();
                } // end of CheckTopoSplitPossible                 
            } // end of if NoNbNodesTooClose()
        }
    } // end of for loop going through all flynns  
     
    fDataFile.close();
}

/*
 * Function called by CheckIfFlynnsNeedSplit():
 * Searches the two nodes that are needed for the split for topological reasons 
 * and outputs 1 if a split is necessary and 0 if not
 * 
 * --> Checks if there are non-neighbouring nodes in one flynn that are too 
 * close to each others and outputs the two nodes that are closest to each 
 * others AND closer than switchDist
 */
int NoNbNodesTooClose(int iFlynnID, int *iNode1, int *iNode2)
{
    int iSplitNeeded = 0;
    int iNode1Temp = 0, iNode2Temp = 0;
    
    /* Initialise node distances to switchdist, any distance smaller than this 
     * will be stored in this variables overwriting them:*/
    double dSwitchDist = ElleSwitchdistance(); 
     
    double dNodeDist = dSwitchDist, dNodeDistTemp = dSwitchDist;    
    double dNodeSep = 0.0; // to hold the current node separation
    
    vector<int> vFlynnBnodes;
    vector<int> vBnodesNoNb;
    ElleFlynnNodes(iFlynnID,vFlynnBnodes);
            
    for (int i=0; i<vFlynnBnodes.size(); i++)
    {
        iNode1Temp = vFlynnBnodes.at(i);
        /* Make sure only to use double nodes to avoid quadruple junctions:*/
        if (ElleNodeIsDouble(iNode1Temp))
        {   
            FindNoNeighbourBnodes(iNode1Temp,iFlynnID,vFlynnBnodes,vBnodesNoNb);
            
            /* Go through all no-neighbour bnodes of iNode1Temp and search for 
             * the closest one*/
            dNodeDistTemp = dSwitchDist; // reset to switchdist
            while (vBnodesNoNb.size()>0)
            {
                dNodeSep = ElleNodeSeparation(iNode1Temp,vBnodesNoNb.back());
                if (dNodeSep < dNodeDistTemp)
                {
                    /* If this distance is smaller than the smallest distance
                     * observed before: Store the new smallest distance in 
                     * dNodeDist and the corresponding bnode close to iNode1Temp
                     * in iNode2Temp
                     */
                    dNodeDistTemp = dNodeSep;
                    iNode2Temp = vBnodesNoNb.back();
                    iSplitNeeded = 1; // is returned together with splitnodes later
                }  
                vBnodesNoNb.pop_back();                                      
            }
        }
        vBnodesNoNb.clear();      
        
        /* Check if this distance is the smallest one in the flynn:*/
        if (dNodeDistTemp < dNodeDist)
        {
            dNodeDist = dNodeDistTemp;
            *iNode1 = iNode1Temp;
            *iNode2 = iNode2Temp;
        }
    }
    vFlynnBnodes.clear();
    
    return (iSplitNeeded);        
}

/*
 * Called by NoNbNodesTooClose:
 * Stores all bnodes that are NOT a neighbour of node iBnode, but are bnodes of 
 * the same flynn (from vector vFlynnBnodes) in the vector vBnodesNoNb
 * --> At the moment,also 4th order neighbours are not stored
 */
int FindNoNeighbourBnodes(int iBnode, int iFlynn, vector<int> vFlynnBnodes, vector<int> &vBnodesNoNb)
{   
    int iNodePos;
    int iFirst,iLast;
    int iOrder = 4; // limit the output bnodes to non-4th order neighbours
    vector<int>::iterator vItPos;
    
    /*
     * Check if the flynn has sufficient bnodes:
     */
    if (vFlynnBnodes.size()< (iOrder*2+1) )
        return 0;
    
    /*
     * Find the bnode of interest in the vector containing all bnodes of the 
     * flynn:
     */
    vItPos = find (vFlynnBnodes.begin (),vFlynnBnodes.end (), iBnode);
    iNodePos = distance(vFlynnBnodes.begin(),vItPos);
    
    /*
     * Find the positions of the nodes that need to be deleted from the vector
     * to get only neighbours of iOrder-th order
     */
    iFirst = iNodePos-iOrder;
    if (iFirst<0) iFirst += vFlynnBnodes.size();
    iLast = iNodePos+iOrder;
    if (iLast>=vFlynnBnodes.size()) iLast -= vFlynnBnodes.size();
    
    /*
     * Erase all neighbours of iOrder-th order:
     * --> iLast +1 to also delete the node of interest from the list
     */
    if (iFirst < iLast)
        vFlynnBnodes.erase(vFlynnBnodes.begin()+iFirst,vFlynnBnodes.begin()+iLast+1);
    else
    {
        vFlynnBnodes.erase(vFlynnBnodes.begin()+iFirst,vFlynnBnodes.end());
        vFlynnBnodes.erase(vFlynnBnodes.begin(),vFlynnBnodes.begin()+iLast+1);
    }
    
    /*
     * Delete triple junctions of bnode list: They shouldn't be used for splits
     * to avoid quadruple junctions --> Store the bnodes in output vector
     */
    while(vFlynnBnodes.size()>0)
    {        
        if (ElleNodeIsDouble(vFlynnBnodes.back())) 
            vBnodesNoNb.push_back(vFlynnBnodes.back());
            
        // erase the bnode that has just been checked from vFlynnBnodes:
        vFlynnBnodes.pop_back();        
    }
    
    return 0;   
}

/*
 * Check if a potential split line will lie inside the flynn or
 * outside: If it is outside, the split is not performed due to errors updating
 * the flynn attributes. The same split will be performed once the other flynn
 * (in which the split line will lie in) is called.
 * Returns 0 if split cannot be performed, otherwise returns 1
 */
int CheckTopoSplitPossible(int iFirst, int iLast, int iFlynn)
{
    int iSplitPossible = 0;
    Coords cFirst, cLast, cSplitLineMiddle;
    
    ElleNodePosition(iFirst,&cFirst);
    ElleNodePosition(iLast,&cLast);
    
    ElleCoordsPlotXY(&cFirst,&cLast); // changes 1st point 
    
    // Find Midpoint of splitline to later check if it is in the flynn or not:
    cSplitLineMiddle.x = ((cLast.x-cFirst.x)/2)+cFirst.x;
    cSplitLineMiddle.y = ((cLast.y-cFirst.y)/2)+cFirst.y;  
    
    if(EllePtInRegion(iFlynn,&cSplitLineMiddle)) iSplitPossible = 1;
    
    return iSplitPossible;
}

/*
 * Does the same like CrossingsCheck, but if a triple node is moved it checks 
 * if this movement might need a neighbour switch with a neighbouring triple 
 * node (if switch distance between the moved triple node and the closest 
 * neighbouring triple node (if there is one) is reached): If this is the case 
 * it checks if the neighbour switch is possible or if it would cause trouble 
 * because a single sided grain is created: In the latter case the moved triple 
 * node is set back to its original position
 * --> MAYBE THINK ABOUT A BETTER WAY TO SOLVE THIS THAN RESETTING THE POSITION
 */
int FS_CrossingsCheck(int node, Coords *pos_incr)
{
    fstream LogCrossCheck;
    int err=0;
    Coords previousxy;
    
    ElleNodePosition(node,&previousxy);
    
    /* Check if in a double sided flynn a flynn edge is crossed by movement
     * If yes: Movement is avoided by setting pos_incr to 0,0
     */
    Coords cNewPos;
    cNewPos.x = previousxy.x+pos_incr->x;
    cNewPos.y = previousxy.y+pos_incr->y;
    
    if(!FS_CheckCrossingDJ(node,cNewPos))
    {
        LogCrossCheck.open ( "LogfileTopoChecks_CrossingsCheck.txt", fstream::out | fstream::app);
        LogCrossCheck << "INFO (FS_CheckCrossingDJ) Movement of bnode " << node 
                      << " forbidden, because:\n\tMovement would cause " 
                      << "node crossing a flynn edge in a 2 sided flynn " 
                      << "--> No possibility to solve that with topology checks" 
                      << endl;
        LogCrossCheck.close();
        return (err);
    }
    
    ElleUpdatePosition(node,pos_incr);
    
    /*
     * Before Topology checks: Check if node is a triple node and if a triple 
     * switch with a neighbouring triple node (if there is one) will be 
     * possible --> If not the code for topology checks might crash 
     */
    if (ElleNodeIsTriple(node))
    {
        // Find closest neighbouring triple node
        int nbnodes[3];
        double dDist = 1e6; 
        int nbtriple = -1; // pre-allocate variable to store closest triple node
        ElleNeighbourNodes(node,nbnodes);       
        
        for (int i=0;i<3;i++)
        {
            if (nbnodes[i]!=NO_NB && 
                ElleNodeIsTriple(nbnodes[i]) &&
                ElleNodeSeparation(node,nbnodes[i])<dDist)
            {
                nbtriple = nbnodes[i];   
                dDist = ElleNodeSeparation(node,nbnodes[i]);        
            }
        }
        
        /* If a neighbouring node has been found (when nbtriple>-1) and their
         * distance is below the switch distance
         */        
        if (nbtriple>-1 && dDist<ElleSwitchdistance())
        {
            if (!FS_TripleSwitchPossible(node,nbtriple)) // if a switch is not possible
            {
                // Set node position back to initial position and end function
                ElleSetPosition(node,&previousxy);
                //cout << "NOW!" << endl;
                return 0;
            }                        
        }
    }
    
    // A neighbour switch between this and a neighbouring triple node will be
    // possible OR the node is not a triple node:
    err = ElleNodeTopologyCheck(node);
    return(err);
}

/*
 * Check if a triple node switch is allowed:
 * It will not be allowed if it results in single nodes, which will be the case
 * if the triple nodes have completely the same flynn neighbours
 * --> Returns 1 is the switch is possible, 0 if not
 */
int FS_TripleSwitchPossible(int iTripleJ1,int iTripleJ2)
{
    int iIsPossible = 1;
    int iNbFlynns1[3],iNbFlynns2[3];
    vector<int> vNbFlynns1, vNbFlynns2;
    
    // Initial checks:
    if(!ElleNodeIsTriple(iTripleJ1))
    {
        printf("ERROR (FS_TripleSwitchPossible): Node %u is not a triple node\n",iTripleJ1);
        return (0);
    }
    if(!ElleNodeIsTriple(iTripleJ2))
    {
        printf("ERROR (FS_TripleSwitchPossible): Node %u is not a triple node\n",iTripleJ2);
        return (0);
    }    
    
    ElleRegions(iTripleJ1,iNbFlynns1);
    ElleRegions(iTripleJ2,iNbFlynns2);
    
    // Save in vectors and sort the vectors
    for (int i=0;i<3;i++)
    {
        vNbFlynns1.push_back(iNbFlynns1[i]);        
        vNbFlynns2.push_back(iNbFlynns2[i]);         
    }
    sort(vNbFlynns1.begin(),vNbFlynns1.end());
    sort(vNbFlynns2.begin(),vNbFlynns2.end());
    
    // CHECK:
    if (vNbFlynns1[0]==vNbFlynns2[0] &&
        vNbFlynns1[1]==vNbFlynns2[1] &&
        vNbFlynns1[2]==vNbFlynns2[2])
    {
        iIsPossible = 0;
    }   
    
    return (iIsPossible);
}

/*
 * FS_CheckCrossingDJ:
 * In a flynn which has only 2 flynns as neighbours, we need to make sure that 
 * one DJ is not "crossing" or "overtaking" another one:
 * 
 *            |            Y
 * (1)--(6)__(5)           |
 *  |         |            |------X
 * (2)--(3)--(4)
 *            |
 * 
 * Here e.g. (3) should never move further in y-direction than the y-position of
 * (6). It should never be "higher" than (6) which is in 2D topologically 
 * impossible
 * 
 * In order to avoid such a movement the following code is checking the 
 * following:
 * If there is a 2-sided flynn: 
 * Determine all edges of the polygon and the movement vector in the form 
 * y=mx+b and see if they intersect with each others: If the intersection point 
 * now lies on the line segment between the old and new bnode position, the 
 * bnode was "overtaking" the flynn boundary: Performing the movement will 
 * result in the situation we would like to avoid and movement is forbidden.
 * 
 * INPUT: The node of interest (should still be in old position) and the 
 * coordinates of the new position)
 * OUTPUT: 0 if movement causes crossing and hence impossible, 1 if movement is
 * possible
 */
int FS_CheckCrossingDJ(int iNode,Coords cNewPosition)
{
    int i2SidedFlynn = 0;
    vector<int> vBnodes;
    
    // For intersection determination:
    int iEdgeNode1,iEdgeNode2; // two nodes forming an edge of the polygon
    Coords cOldPosition; // old position of iNode
    Coords cEN1,cEN2,cIntersection; // their positions EN1 and EN2
    double dA,dB; // for the line formed by old and new node position
    double dC,dD; // for the line formed by polygon edge
    double dE,dF; // for the line formed by polygon edge
    
    if (!ElleNodeIsActive(!iNode))
    {
        //printf("Error (FS_CheckCrossingDJ): %u is inactive\n",iNode);
        return (0); // node inactive - movement will not be allowed
    }
    
    if (!ElleNodeIsDouble(iNode))
    {
        //printf("Error (FS_CheckCrossingDJ): %u is not a double node\n",iNode);
        return (1); // movement will be allowed
    }
    
    i2SidedFlynn = FS_CheckFor2SidedFlynn(iNode);
    
    /* If there is no 2-sided flynn, the movement is allowed:*/
    if(i2SidedFlynn==-1)  return (1);  
    
    /* If there is a 2-sided flynn go on as described above*/
    ElleNodePosition(iNode,&cOldPosition); 
    ElleCoordsPlotXY(&cNewPosition,&cOldPosition);
     
    // Check if positions coincide --> then "movement" will be allowed
    if (cNewPosition.x==cOldPosition.x && cNewPosition.y == cOldPosition.y)
        return (1);
    
    // Find the line in the form y=dA*x+*dB formed by the old and new node position:  
    dA = fabs(cNewPosition.y-cOldPosition.y)/fabs(cNewPosition.x-cOldPosition.x);
    // For dB use the mean of the solution of the two remaining equations:
    dB = (cOldPosition.y - (dA*cOldPosition.x))+(cNewPosition.y - (dA*cNewPosition.x));
    dB /= 2;
    
    ElleFlynnNodes(i2SidedFlynn,vBnodes);
    vBnodes.push_back(vBnodes[0]); // set 1st bnode at last position as well to also compare last and first node
    
    for (int i=0;i<vBnodes.size()-1;i++)
    {     
        iEdgeNode1 = vBnodes[i];
        iEdgeNode2 = vBnodes[i+1];
        ElleNodePosition(iEdgeNode1,&cEN1);
        ElleNodePosition(iEdgeNode2,&cEN2);
        ElleCoordsPlotXY(&cEN1,&cOldPosition);
        ElleCoordsPlotXY(&cEN2,&cOldPosition);
        
        // Find the edge in the form y=dC*x+*dD formed by two bnodes
        dC = fabs(cEN2.y-cEN1.y)/fabs(cEN2.x-cEN1.x);
        // For dD use the mean of the solution of the two remaining equations:
        dD = (cEN1.y - (dC*cEN1.x))+(cEN2.y - (dC*cEN2.x));
        dD /= 2;
        
        // Find possible intersection with line formed by path of movement
        if (dC!=dA) // otherwise there is no intersection, lines are parallel
        {
            cIntersection.x = (dB-dD)/(dC-dA);
            cIntersection.y = (dC*cIntersection.x) + dD;
            ElleCoordsPlotXY(&cIntersection,&cOldPosition);
        
            // See if intersection point is (1) on the movement path and (2) on
            // the edge between two bnodes
            // --> if yes: The movement needs to be forbidden, because polygon edge is crossed
            if (FS_PtInRect(cIntersection,cEN1,cEN2) &&
                FS_PtInRect(cIntersection,cOldPosition,cNewPosition))
            {
                /* Avoid that one of the nodes is not the node that is moved 
                 * Of course there will always be an intersection if this node
                 * is involved
                 */
                if (iEdgeNode1!=iNode && iEdgeNode2!=iNode)
                {
                    //printf("Moving DJ %u impossible in Flynn %u:\n",iNode,i2SidedFlynn);
                    //printf("Polygon edge between %u and %u at ",iEdgeNode1,iEdgeNode2);
                    //printf("%f,%f\n",cIntersection.x,cIntersection.y);
                    return (0);
                }
            }
            else
            {
                //printf("\n%f != %f \n",(cIntersection.x-cOldPosition.x)/(cNewPosition.x-cOldPosition.x),(cIntersection.y-cOldPosition.y)/(cNewPosition.y-cOldPosition.y));
            }           
        }
        
    }
    vBnodes.clear();
      
    return (1);
}

/*
 * FS_CheckFor2SidedFlynn:
 * Find node's neighbour flynns and check if one of them is 2-sided flynn (has 
 * only 2 flynn neighbours
 * Output the flynn ID if there is a 2-sided flynn, output -1 if there is none
 */
int FS_CheckFor2SidedFlynn(int iBnode)
{
    int iNbFlynnsOfNode[3];
    list<int> lNbFlynnsOfFlynn;
    int i2SidedFlynn = -1;
    
    ElleRegions(iBnode,iNbFlynnsOfNode);
    
    for (int i=0;i<3;i++)
    {
        if (iNbFlynnsOfNode[i]!=NO_NB)
        {
            ElleFlynnNbRegions(iNbFlynnsOfNode[i],lNbFlynnsOfFlynn); 
            // If the following is true, we found a 2sided flynn:
            if (lNbFlynnsOfFlynn.size()<=2) 
                i2SidedFlynn = iNbFlynnsOfNode[i];
        }
        lNbFlynnsOfFlynn.clear();
    }
    
    return (i2SidedFlynn);
}

/*
 * See if a point is in an rectangle that is created by the two edge points 
 * defined in cEdge1 and cEdge2 coordinates. (They will be sorted for smaller 
 * / larger xy values internally.
 * Output 1 if the point is inside, output 0 if not
 */
int FS_PtInRect(Coords cPoint,Coords cEdge1,Coords cEdge2)
{
    int in=0;

    Coords cRectEdges[4];

    // cRectEdges[0] is baseleft corner of rectangle
    // cRectEdges[1] is topright corner of rectangle

    // sort x values
    if (cEdge1.x <= cEdge2.x)
    {
        cRectEdges[BASELEFT].x = cEdge1.x;
        cRectEdges[TOPRIGHT].x = cEdge2.x;
    }
    else
    {
        cRectEdges[BASELEFT].x = cEdge2.x;
        cRectEdges[TOPRIGHT].x = cEdge1.x;
    }

    // sort y values
    if (cEdge1.y <= cEdge2.y)
    {
        cRectEdges[BASELEFT].y = cEdge1.y;
        cRectEdges[TOPRIGHT].y = cEdge2.y;
    }
    else
    {
        cRectEdges[BASELEFT].y = cEdge2.y;
        cRectEdges[TOPRIGHT].y = cEdge1.y;
    }  
    
    in = EllePtInRect(cRectEdges,4,&cPoint);
    return(in);    
}

/* 
 * Do exactly what elle_gg does in terms of topology checks
 */
void ElleGGTopoChecks()
{
    //TotalTime=0;
    //int i, j, k, n;
    //int interval=0,st_interval=0,err=0,max;
    //vector<int> seq;
    //char fname[32];
    //FILE *fp;

    //ElleAddDoubles();
    ////if (ElleDisplay()) EllePlotRegions(ElleCount());
	//ElleCheckFiles();

    ////for (i=0;i<EllemaxStages();i++) {
        //max = ElleMaxNodes();
        //seq.clear();
        //for (j=0;j<max;j++) if (ElleNodeIsActive(j)) seq.push_back(j);
        //random_shuffle(seq.begin(),seq.end());
        //max = seq.size();
        //for (n=0;n<max;n++) {
            //j=seq[n];
            //if (ElleNodeIsActive(j)) {
                //if (ElleNodeIsDouble(j)) {
                    ////MoveDoubleJ(j);
                    //ElleCheckDoubleJ(j);
                //}
                //else if (ElleNodeIsTriple(j)) {
                    ////MoveTripleJ(j);
                    //ElleCheckTripleJ(j);
                //}
            //}
        //}
        ////ElleUpdate();
    ////}
    ///*CheckAngles();*/
    vector <int> iAllBnodes;
    int iBnode = 0;
    
    for (int i=0;i<ElleMaxNodes();i++)
    {
        if (ElleNodeIsActive(i)) iAllBnodes.push_back(i);
    }
    
    random_shuffle(iAllBnodes.begin(),iAllBnodes.end());
    
    for (int j=0;j<iAllBnodes.size();j++)
    {
        iBnode = iAllBnodes.at(j);
        
        if (ElleNodeIsActive(iBnode))
        {
            if (ElleNodeIsDouble(iBnode))
            {
                ElleCheckDoubleJ(iBnode);
            }
            else if (ElleNodeIsTriple(iBnode))
            {
                ElleCheckTripleJ(iBnode);
            }
        }
    }
}

/* BY FLORIAN:
 * 
 * Update the flynn ages by setting them to (-1), ATTENTION: After this, we 
 * still need to finally increase all ages of flynns by 1, but this needs to be 
 * done when ALL recrystallization is over.
 *  
 * New flynn will always be the smaller one:
 * Do not assign new age to old flynn, if it is more than 10x larger than new 
 * one 
 */
void UpdateFlynnAges(int iNew,int iOld)
{
    int iTmp;
    
    if (ElleFlynnIsActive(iNew) && ElleFlynnIsActive(iOld))
    {
        double dAreaNew=ElleRegionArea(iNew);
        double dAreaOld=ElleRegionArea(iOld);
        
        // smaller flynn should be new one
        if (dAreaOld<dAreaNew)
        {
            iTmp = iNew;
            iNew = iOld;
            iOld = iTmp;            
        }
        
        if ((dAreaOld/dAreaNew)<10.0)
        {
            ElleSetFlynnRealAttribute(iNew,(-1.0),AGE);
            ElleSetFlynnRealAttribute(iOld,(-1.0),AGE);
        } 
        else
        {
            ElleSetFlynnRealAttribute(iNew,(-1.0),AGE);
        }
    }    
}

int FS_FlynnIsDangling(int flynn)
{
    /*
     * Florian:
     * I define a "dangling" flynn as a flynn with the following 
     * characteristics:
     * 
     * (1) Has only 2 neighbour flynns
     * (2) Has only 2 triple nodes
     * (3) These triple nodes are neighbours
     * 
     * "Dangling" describes how these flynns usually look like: They are small
     * and attached to another appear to be "dangling" on another flynn
     *
     * If the input flynn is such a flynn, function returns 1, otherwise 0
     */
    int iReturnValue = 0;
    vector<int> vFlynnNodes;
    vector<int> vTriples;
    list<int> lFlynnNbs;
    int iNbNodes[3];
    
    if (ElleFlynnIsActive(flynn))
    {
        /* Check if flynn has 2 or less neighbours */
        lFlynnNbs.clear();
        ElleFlynnNbRegions(flynn,lFlynnNbs);
        if (lFlynnNbs.size()<2) {iReturnValue=1; return iReturnValue;}   // FLYNN HAS ONLY 1 NEIGHBOUR!! DELETE IT!          
        if (lFlynnNbs.size()>2) {iReturnValue=0; return iReturnValue;}
        
        //--> Flynn has 2 neighbours, go on:
                    
        /* Find all triple nodes in flynn*/
        vFlynnNodes.clear();
        vTriples.clear();
        ElleFlynnNodes(flynn,vFlynnNodes);
        for (int node=0;node<vFlynnNodes.size();node++)
            if (ElleNodeIsTriple(vFlynnNodes.at(node)))     
                vTriples.push_back(vFlynnNodes.at(node));
        
        /* If there is only one triple node: Delete flynn 
         * If there are more than 2 triple nodes: Keep flynn*/
        if (vTriples.size()<2) {iReturnValue=1; return iReturnValue;} // delete flynn
        if (vTriples.size()>2) {iReturnValue=0; return iReturnValue;} // keep flynn
        
        //--> Flynn has 2 triple nodes, go on:
        
        /* Delete the flynn, if the 2 triple nodes are neighbours
         * vTriples.size() should be ==2*/
        for (int i=0;i<3;i++) iNbNodes[i] = NO_NB;
        
        /* Take 1st triple node and compare to 2nd one:*/
        ElleNeighbourNodes(vTriples.at(0),iNbNodes);
        /* Check that none of the nb nodes is another triple node:*/
        for (int nb=0;nb<3;nb++)
        {
            if (iNbNodes[nb]!=NO_NB)
            {
                // If one of the neighbours is the 2nd triple node
                if (iNbNodes[nb]==vTriples.at(1)) 
                {
                    iReturnValue=1; 
                    return iReturnValue;
                }
            }
            else
            {
                // This was aparently no triple node, something is wrong
                // with this flynn...lets be brave and delete it:
                iReturnValue=1; 
                return iReturnValue;                                                
            }
        }
    }
    iReturnValue=0;
    return iReturnValue;
}

void UpdateUnodesAndAttribs(bool bWriteUnodeOri2File)
{
    /*
     * FS: Updates dislocation densities in unodes: If a unode changes flynn 
     * (i.e. is swept by a moving or -recrystallising- boundary, which can also
     * happen due to topochecks) its dislocation density is set to zero
     * ATTENTION: This is essentially the same function than the old 
     * "Update_dislocden"
     * 
     * The input bool decides if unode orientation before and after sweeping
     * by boundary is stored in a logfile, set to true for "yes"
     */    
    int iFlynnPhase       = VISCOSITY;  // Storage of phases
    int iFlynnNumber      = F_ATTRIB_C; // Storage of FFT unode update variables
    int iUnodeFlynnNumber = U_ATTRIB_C;
    
	//fstream fDataFile;
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
	
	//fDataFile.open ( "Logfile_FS_UpdateDislocden.txt", fstream::out | fstream::app);
	//fDataFile << "# # # # #   " << Settings_run.Count << "   # # # # #" << endl;
        
    iMaxUnodes = ElleMaxUnodes();
    iMaxFlynns = ElleMaxFlynns();
	//dRoi = sqrt(1.0/(double)iMaxUnodes/3.142)*5; // used for Euler angle reassignment
    dRoi = FS_GetROI4topochecks(8);
    
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
                        //fDataFile << "INFO (UpdateDislocden): Flynn "<<iFlynnId<<" is inactive now, switching unode "
                            //<<i<<" to flynn "<<j<<endl;
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
                //fDataFile << "INFO (UpdateDislocden): Flynn " << iFlynnIdOld << " (partially) renumbered to new flynn " << k << endl;
                
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
                
                // This will not be sufficient, if the old grain ID is still
                // present (i.e. not inactive) due to seperation of one grain
                // into two grains. Therefore there is a need for an additional
                // check for unodes in the part of the seperated grains with the
                // old grain ID:
                vUnodeList.clear(); 
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
    
    if (!ElleUnodeAttributeActive(EULER_3))
        if (!ElleUnodeAttributeActive(U_DISLOCDEN))
            return;
    
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
				//fDataFile << "WARNING: Unode " << i << " changed Phase... Set New DislocDen to 0" << endl; //Shifted Dislocden back." << endl;
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
                        //fDataFile << "WARNING (FS_update_dislocden): Setting ";
                        //fDataFile << "new orientation of swept unode "<<i;
                        //fDataFile << " to old value" << endl;
                        printf("WARNING (UpdateUnodesAndAttribs): Setting new orientation of swept unode %u to old value\n",i);
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
                        //fDataFile << "WARNING (FS_update_dislocden): Setting ";
                        //fDataFile << "new orientation of swept unode "<<i;
                        //fDataFile << " to flynn mean value" << endl;
                        printf("WARNING (UpdateUnodesAndAttribs): Setting new orientation of swept unode %u to flynn mean value\n",i);
                    }
                }
                /*
                 * FS: New way to check which orientations get eaten up by 
                 * moving grain boundaries and which ones are "growing":
                 * Store old and new unode orientation of the swept unode
                 * in a separate textfile called: UnodeOriChangeGBM.txt
                 */         
                if (bWriteUnodeOri2File)
                {
                    fstream fUnodeOri;
                    fUnodeOri.open ( "UnodeOriChangeGBM.txt", fstream::out | fstream::app);
                    //fUnodeOri << "# # # # #   " << Settings_run.Count << "   # # # # #" << endl;
                    fUnodeOri << "# # # # #   unode orientation changes   # # # # #" << endl;
                    fUnodeOri << "id x y old_phase new_phase e1_before e2_before e3_before e1_after e2_after e3_after" << endl;
                    fUnodeOri << i << " " << cUnodepos.x << " " << cUnodepos.y
                              << " " << (int)dPhase1 << " " << (int)dPhase2 
                              << " " << dEulerOld[0] << " " << dEulerOld[1] << " "
                              << dEulerOld[2] << " " << dValEuler[0] << " " 
                              << dValEuler[1] << " " << dValEuler[2] << endl;  
                    fUnodeOri.close();
                }
                          
                if ( (dValEuler[0] >= -180) &&  (dValEuler[0]<=180) && (dValEuler[1] >= -180) &&  (dValEuler[1]<=180) && (dValEuler[2] >= -180) &&  (dValEuler[2]<=180))
                {	
                    //printf(" unodes count %i\n", iCount);
                    //fDataFile << "number of unodes in flynn: " << iCount << endl;
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
    
    /* OLD STUFF FROM GBM CODE, NEVER USED IT, THEREFORE IT IS IN COMMENTS NOW*/
    ////// 2nd part: simulate internal restructuration of grains  
	////// Not active with FFT simulations
    ////// To compare gbm with bnodes simulations: automatic readjust at average value 
    ////// Only scalar dislocation density, not effect on euler orientation of unodes 	
	////double dAvdensity; //a_factor=0.0 redistribution factor; 1.0 full while 0.0 non redistribution
	////int a_factor = 0;
    
    ////UserData userdata;
    ////ElleUserData(userdata);
    ////a_factor= (int)userdata[1];
    
	///////JR since nothing changes if a_factor is 0, don't even do it then...
	////if ( a_factor != 0 )
	////{
		////for (int j=0;j<iMaxFlynns;j++) 
		////{
			////if (ElleFlynnIsActive(j)) 
			////{
				////vUnodeList.clear();
				////dAvdensity=0.0;
				////ElleGetFlynnUnodeList(j,vUnodeList); // get the list of unodes for a flynn
				
				////for (int i=0;i<vUnodeList.size();i++) 
                ////{
					////ElleGetUnodeAttribute(vUnodeList[i],U_DISLOCDEN,&dDensity);					
					////dAvdensity += dDensity; 			
                ////}
					
				////dAvdensity /= vUnodeList.size();
				
				////for (int i=0;i<vUnodeList.size();i++) 
                ////{		
					////ElleGetUnodeAttribute(vUnodeList[i],U_DISLOCDEN,&dDensity);
					////dDensityNew = dDensity-(dDensity-dAvdensity)*a_factor; 
					////ElleSetUnodeAttribute(vUnodeList[i],U_DISLOCDEN, dDensityNew);			
				////}
		
			////}
		////}
	////}
	
	////fDataFile.close();
	return;   
}

double FS_GetROI4topochecks(int iFact)
{
    /* 
     * This is the same than FS_GetROI, just naming is different to avoid 
     * redundant functions...probably there should be a nicer way than just
     * renaming...but anyway...it works*/
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

bool CheckforIslandFlynns()
{
    /* returns true if an island flynn is found, false if not */
    list<int> lNeighbours;
    bool bFound=false;
    for (int i=0;i<ElleMaxFlynns();i++)
    {
        if (ElleFlynnIsActive(i))
        {
            lNeighbours.clear();
            ElleFlynnNbRegions(i,lNeighbours);            
            if (lNeighbours.size()<=1)
            {
                printf("ERROR: Flynn %u has only 1 neighbour!\n",i);
                bFound = true;
            }
        }
    }
    return bFound;
}


