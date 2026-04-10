/*
 * Modification history:
 *Modified 11/12/2008 to add boundary mobility as function of misorientation angle (AGA)
 * JR 6/11/2013: Added mobility and activation energy readings in the MoveNode calculations and GetBoundaryMobility function.
 * FS & JR 11-12/2014 Added checks for NaN erros and logfiles 
 */
#include "FS_header_movenode_pp_fft.h"
#include "FS_topocheck.h"
extern runtime_opts Settings_run;

/*
 * This function returns a vector in movedir. The node should move in the 
 * direction of movedir for the distance of length of movedir. This should be 
 * used with ElleUpdatePosition. The function returns 1 if the node has to be 
 * moved, 0 if not. If it should not be moved, the node either was inactive of 
 * the passed energies did not cause the node movement
 */
int GetMoveDir( int node, Coords pvec, Coords * movedir,double trialdist, int *iNodeTooFast2, double *dTimeStepHighest2,double dModRedFact  )
{
    if ( pvec.x != 0 && pvec.y != 0 )
    {
        if ( ElleNodeIsTriple( node ) ) {
            return MoveTNode( node, pvec, movedir, iNodeTooFast2, dTimeStepHighest2, dModRedFact );
           }
        else {
            return MoveDNode( node, pvec, movedir, iNodeTooFast2, dTimeStepHighest2, dModRedFact );
		}
    }
    else
        return 0;
}

/*!/brief Move a double node
 
This function returns the direction and distance to move the node node in movedir. It returns 0 if movedir is 0 (the node
should not be moved this time) or 1 if the node should be moved. It also takes into account the timestep defined in the
elle-file (usually one year) and the speedup (usually one). The combination of timestep, speedup and switchdistance has to be
balanced. If the switchdistance is very small and the timestep is very large, the calculated distance for the node movement
can be bigger then the switchdistance. In that case, a warning message is logged, movedir is set to 0 and the function returns
0 (meaning the node should not be moved).*/
int MoveDNode( int node, Coords pvec, Coords * movedir, int *iNodeTooFast2, double *dTimeStepHighest2,double dModRedFact  )
{
    double l1, l2, cosseg, cosalpha1, cosalpha2, mob1, mob2, flength,type,len;
    int nbnode[3], node1, node2, err, n;
	int move = 0;
	double dt = 0, dtMAX=0, mob[2];
    Coords F, V0, nodexy, old1xy, old2xy, vec1, vec2, xymid[2];
    char logbuf[4096];
	Coords sup1, sup2, sup3, vec_body,V0B,FB, V0S;
	double density[2],len_body, bodyenergy, diff_density[4],len_sup1,len_sup2,len_sup3;
	double aa,aaa, area, density_energy;
    int rgn[3], nnbnode[3],node3,i,j;
	ERegion rgn2[3];
	int id,l;
	fstream fDataFile;
	
	fDataFile.open ( "Logfile_MoveNode.txt", fstream::out | fstream::app);
	//fDataFile << "# # # # #   " << Settings_run.Count << "   # # # # #" << endl;
	
	// if id != 1 working with flynn, id ==1 working with unodes.. define as a user value 
    // FS: Previously id was user defined (definition in the code), now the code 
    // itself will detect what to use:		
    if(!ElleUnodeAttributeActive(EULER_3)) 
    {
        id = 0; // use unodes if U_EULER_3 is an unode attribute
    }
    else
    {
        id = 1; // use flynns if EULER_3 of C_AXIS are flynn attributes
    }
	
    if ( ElleNodeIsActive( node ) )
    {
        //Get the neighbouring nodes
        if ( err = ElleNeighbourNodes( node, nbnode ) )
            OnError( "MoveDNode", err );		
		l=0;
		for (n=0; n< 3; n++) {

			if (nbnode[n] != NO_NB){

			if (ElleNodeIsActive(nbnode[n])) {		
				nnbnode[l] = nbnode[n];
				l++; 
				}
			}
		}
		
		node1 = nnbnode[0];
        node2 = nnbnode[1];	

// if ( ElleNodeIsActive( node1 ) && ElleNodeIsActive( node2 )) {		
        //Get Boundary mobility
        //mob1 = GetBoundaryMobility( node,node1 );
        //mob2 = GetBoundaryMobility( node,node2 );
	    //Get Boundary mobility
	    
///JR Note: 
	    // around double nodes the phases can't ne different from one segment to the next therefore they have the same mobility.
        mob1 = mob2 = GetBoundaryMobility( node, node1,dModRedFact );///2; //in Elle, boundaries belong to two flynns, therefore, the division by two
	
        //Get the length of the vectors node-node1 and node-node2
        //And of course get the vetors itself (turned by 90deg, we need the vector normals)
        ElleNodePosition( node, & nodexy );
        ElleNodePlotXY( node1, & old1xy, &nodexy );
        ElleNodePlotXY( node2, & old2xy, &nodexy );
       //calculate vectors
        vec1.y = -1*(old1xy.x-nodexy.x);
        vec1.x = old1xy.y-nodexy.y;
		xymid[0].x=nodexy.x+(old1xy.x-nodexy.x)*0.5; // FS: xymid is used in Boundary Mobility function of misorientation
		xymid[0].y=nodexy.y+(old1xy.y-nodexy.y)*0.5;	
				
        //here we get the length of the vector
        l1 = GetVectorLength( vec1 );
        //calculate the second vector
        vec2.y = -1*(old2xy.x - nodexy.x);
        vec2.x = old2xy.y-nodexy.y;
		xymid[1].x=nodexy.x+(old2xy.x-nodexy.x)*0.5;
		xymid[1].y=nodexy.y+(old2xy.y-nodexy.y)*0.5;
        //get length of vector
        l2 = GetVectorLength( vec2 );

	
		// Boundary Mobility function of misorientation
        if (ElleFlynnAttributeActive(EULER_3)|| 
            ElleFlynnAttributeActive(CAXIS)  ||
            ElleUnodeAttributeActive(EULER_3)   ) // FS added this line
        {
		
			int rgn1, rgn2;
    		double misorientation;  
    		n=0;
		    l=0;

    		while ( n < 2 )
    		{

			ElleNeighbourRegion( node, nnbnode[n], &rgn[0] );
    		ElleNeighbourRegion( nnbnode[n], node, &rgn[1] );
				
			if ( id != 1) { // the flynn version .. 															
				misorientation=GetMisorientation(rgn[0],rgn[1],id);
				mob[l]=GetBoundaryMobility(node,nnbnode[n],dModRedFact)*Get2BoundaryMobility(misorientation);
				l++;
				}	
				
			else { // this is the unodes version .. 		
				// First version, take information of the nearest unode in both regions 
				rgn1=SearchUnode(rgn[0],&xymid[n]);
				rgn2=SearchUnode(rgn[1],&xymid[n]);
				
				misorientation=GetMisorientation(rgn1,rgn2,id);
				mob[l]=GetBoundaryMobility(node,nnbnode[n],dModRedFact)*Get2BoundaryMobility(misorientation);
				l++;
				}
			// printf("Node Mis Mob %i %e %e\n",node, misorientation,mob[n]);	
			n++;
			}
			mob1=mob[0];
			mob2=mob[1];
    	}
        if (mob1 == 0) mob1=1e-20; 
        if (mob2 == 0) mob2=1e-20;


        //If the energies are equal, pvec will be the 0 and everything will crash. So we do not allow this.
        //Physically this should be correct too since if pvec is 0, no movement should take place.
        //pvec equals the force
        F.x = pvec.x;
        F.y = pvec.y;
        
        ///FS: This makes F unit vector in direction of maximum free energy reduction with length=1
		len=GetVectorLength(F);
		F.x=F.x/len;
		F.y=F.y/len;

        //not nice, but hey...
        //the normals of the segments have to point in the same direction as F
        //so the cosine of the vectors have to be between 0 and 1
        cosalpha1 = DEGCos( vec1, F );
        cosalpha2 = DEGCos( vec2, F );
		// printf("cosalpha  %e %e\n",cosalpha1,cosalpha2);
		
        //which means if one of them is negative, we make it positive. Is that correct in all
        //cases?
        if ( cosalpha1 < 0 )
            cosalpha1 *= -1;
        if ( cosalpha2 < 0 )
            cosalpha2 *= -1;
		
        //in case of isotropic crystals this is sufficient (because the mobility will be the same on both
        //segments)
        //V0.x = ( 2 * F.x * mob1 ) / ( ( l1 * pow( cosalpha1, 2 ) ) + ( l2 * pow( cosalpha2, 2 ) ) );
        //V0.y = ( 2 * F.y * mob1 ) / ( ( l1 * pow( cosalpha1, 2 ) ) + ( l2 * pow( cosalpha2, 2 ) ) );
        //in case of anisotropic crystals this is needed because it takes the mobilities of both segments
        //into account
        //if the angle between F and the segments is 90 the node can freely move which results in too big
        //steps. Therefore, we forbid it when the cosine is smaller 0.01745 (which is 89DEG)
        /* if ( cosalpha1 < 0.01745 && cosalpha2 < 0.01745 ) { V0.x = 0; V0.y = 0; } else { */

		// LE boundary lengths in Elle units,  mobilities in m2s-1
	l1 *= ElleUnitLength();
	l2 *= ElleUnitLength();
    
		// limited to 89DEG , to test .. 
	if ( cosalpha1 < 0.01745 ) 	cosalpha1=0.01745;
	if ( cosalpha2 < 0.01745 ) 	cosalpha2=0.01745;	
    
    aa=(((l1 * pow( cosalpha1, 2 ))/ mob1) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 ) );			
		
    ///FS: print a view things to check what goes wrong in case of an error:
    if (isnan(aa))
    {
        // Get relevant node xy again and print data:        
        Coords FSnodexy,FSnode1xy,FSnode2xy;
        ElleNodePosition( node, & FSnodexy );
        ElleNodePosition( node1, & FSnode1xy );
        ElleNodePosition( node2, & FSnode2xy );
       
        
        
        if(FSnodexy.x==FSnode2xy.x && FSnodexy.y==FSnode2xy.y ) 
        {
            /* Two nodes are at exactly the same position, which will cause
             * trouble: Trying to solve this using ElleGGTopocheks:*/  
            printf("MoveDnode: PROBLEM WITH COINCIDING NODES ");
            CheckNodesCoincide2();           
            ElleGGTopoChecks();
            
            // Check if problem still is not solved:
            if(FSnodexy.x==FSnode2xy.x && FSnodexy.y==FSnode2xy.y ) 
            {
                printf("\nError in MoveDNode:\n");
                printf("The relevant node coordinates appear equal to the ");
                printf("ones of a neighbour node!\nProblem was not solved ");
                printf("using topology checks\n");
                printf("\nData:\n");
                printf("node: %d, neighbour1: %d, neighbour2: %d\n",node,node1,node2);
                printf("node x,y = %e , %e\n",FSnodexy.x,FSnodexy.y);
                printf("node1 x,y = %e , %e\n",FSnode1xy.x,FSnode1xy.y);
                printf("node2 x,y = %e , %e\n",FSnode2xy.x,FSnode2xy.y);
                printf("F.x = %e, F.y = %e\n",F.x,F.y);
                printf("len = %e\n",len);
                printf("normal to segment1 x,y = %e , %e\n",vec1.x,vec1.y);
                printf("normal to segment2 x,y = %e , %e\n",vec2.x,vec2.y);
                printf("cosalpha1 = %e, cosalpha2 = %e\n",cosalpha1,cosalpha2);
                printf("l1 = %e, l2 = %e\n",l1,l2);
                printf("mob1 = %e, mob2 = %e\n",mob1,mob2);
                printf("aa = %e\n",aa);   
                
                //Write log-elle file at this point
                //time_t timeID = time(0);
                //char filename [50];
                //sprintf(filename,"%ld_NAN_DNode_stage%d.elle",timeID,Settings_run.Count);   
                //ElleWriteData(filename); 
                
                fDataFile << "Stage: " << Settings_run.Count << 
                " ERROR (MoveDNode - nan error): Bnode position of bnode " 
                << node << " coincide with the neighbour bnode " 
                << node2 << endl;  
            }
            else
            {
                printf("MoveTnode: SOLVED\n");
            }
        }
    }
    
    ///FS end of this testing part
    
	if (aa <1e-5 || isnan(aa)) { /// FS added isnan(aa)
		V0.x=0.0;
		V0.y=0.0;
	}
	else
	{
        V0.x = ( 2 * F.x * len ) / aa;
        V0.y = ( 2 * F.y * len ) / aa;
	}

        //V0.x /= ElleUnitLength()*ElleUnitLength(); // FS: WHY UnitLength squared??
        //V0.y /= ElleUnitLength()*ElleUnitLength(); // FS: WHY UnitLength squared??
        V0.x /= ElleUnitLength(); // FS: Removed the square
        V0.y /= ElleUnitLength(); // FS: Removed the square

// Scale simulation to real units and check big steps

        flength = GetVectorLength( V0 );
		dt = ElleTimestep() * ElleSpeedup();
        //This should never happen. If it does, the timesteps are too big (or the switchdistance is too small) and needs to be adjusted.
        //In that case, other modules probably will fail too.
        if ( flength * ElleTimestep() * ElleSpeedup() >  ElleSwitchdistance())
        {
            //sprintf( logbuf, "Distance too big to move double, angles: %lf,%lf,%lf,Distance: %lf, 2*Switchdistance: %lf\n",
            //acos( cosalpha1 ) * RTOD, acos( cosalpha2 ) * RTOD, acos( DEGCos( vec1, vec2 ) ) * RTOD, flength * ElleTimestep(),
            //2 * ElleSwitchdistance() );
            //Log( 2, logbuf );
            //sprintf( logbuf, "Reduced movement of node %d! ", node );
            //Log( 0, logbuf );
            //sprintf( logbuf, "SwitchDistance or TimeStep needs adjustment to prevent this!\n" );
            //Log( 0, logbuf );
			if (ElleSpeedup() > 1.0)
			{
				ElleSetSpeedup(1.0);
				fDataFile << "Stage: " << Settings_run.Count << " (D) Resetting speedup to " << ElleSpeedup() << endl;
                //printf("%d\tResetting speedup to %e!\n", Settings_run.Count, ElleSpeedup() );
				dt = ElleTimestep() * ElleSpeedup();
			}
        	if ( flength * ElleTimestep() >  ElleSwitchdistance() )
			{
                if (FS_NodeOnExcludeBoundary(node))
                {
                    /* Accuracy on this boundary is not that important 
                     * (e.g. air-air): Choose highest number possible without
                     * warning the user*/
                    dt = ElleSwitchdistance()/flength/ElleSpeedup();
                    //printf("WARNING (ModeDNode): Excluded phase boundary node is too fast -> Setting to maximum velocity possible\n");
                }
                else
                {
                    //dt = ElleSwitchdistance(); // BEFORE: If timestep too large no movement at all
                    dtMAX = ElleSwitchdistance()/flength/ElleSpeedup(); ///JR Not sure it maximal movement would be still to big...
                    dt = dtMAX*0.9; // NOW: dt is 90% of maximum timestep possible
                    /*!
                    printf("WARNING (ModeDNode): Timestep too large for movement: Check MoveNode logfile\n");
                    fDataFile << "Stage: " << Settings_run.Count << " (DNode " << node << ") SwitchDistance or TimeStep needs adjustment to prevent this!" 
                    << endl << "Local Resetting timestep to " 
                    << scientific << dtMAX*0.9 << " (90% of max. timestep)! from global time step " 
                    << ElleTimestep() << endl << "--> Maximum possible dt would "
                    "have been: " << scientific << dtMAX << endl;
                    */
                    (*iNodeTooFast2)++;
                    (*dTimeStepHighest2)+=dtMAX;
                    //printf("MoveDNode: Too fast, total number: %u, total sum dtMax = %e\n",*iNodeTooFast2,*dTimeStepHighest2);
                }
            }
		}
		movedir->x = V0.x * dt;
		movedir->y = V0.y * dt;
        FS_SetBnodeVelocityAttrib(node,V0);
		move = 1;
    }
    fDataFile.close();
    return (move);
}


/*!/brief This will move a triple node.
 
See MoveDNode for more explanantion*/
int MoveTNode( int node, Coords pvec, Coords * movedir, int *iNodeTooFast2, double *dTimeStepHighest2,double dModRedFact )
{
    double l1, l2, l3,len, cosalpha1, cosalpha2, cosalpha3, mob1, mob2, mob3, flength;
    int nbnode[3], node1, node2, node3, err;
	int move = 0;
	double dt = 0, dtMAX=0, aa, aaa, mob[2];
    Coords F, V0, nodexy, old1xy,old2xy,old3xy, vec1, vec2, vec3, xymid[2];
    char logbuf[4096];
	Coords sup1, sup2, sup3, vec_body,V0B,FB;
	double density[2],len_body, bodyenergy,diff_density[4],len_sup1,len_sup2,len_sup3;
    int rgn[3];
	int id, n;
	fstream fDataFile;
	
	fDataFile.open ( "Logfile_MoveNode.txt", fstream::out | fstream::app);
	//fDataFile << "# # # # #   " << Settings_run.Count << "   # # # # #" << endl;
	
    // if id != 1 working with flynn, id ==1 working with unodes.. define as a user value 
    // FS: Previously id was user defined (definition in the code), now the code 
    // itself will detect what to use:		
    if(!ElleUnodeAttributeActive(EULER_3)) 
    {
        id = 0; // use unodes if U_EULER_3 is an unode attribute
    }
    else
    {
        id = 1; // use flynns if EULER_3 of C_AXIS are flynn attributes
    }
	
    if ( ElleNodeIsActive( node ) )
    {
        //Get the neighbouring nodes
        if ( err = ElleNeighbourNodes( node, nbnode ) )
            OnError( "MoveNode", err );
        node1 = nbnode[0];
        node2 = nbnode[1];
        node3 = nbnode[2];
// if ( ElleNodeIsActive( node1 ) && ElleNodeIsActive( node2 ) && ElleNodeIsActive( node3 )) {
	
        //Get Boundary mobility
        //mob1 = GetBoundaryMobility( node,node1 );
        //mob2 = GetBoundaryMobility( node,node2 );
        //mob3 = GetBoundaryMobility( node,node3 );
		
		//Get Boundary mobility, only one phase
		
/// JR ADDITION
		// around triple nodes the phases can actually be different and therefore have a different mobility...
		mob1 = GetBoundaryMobility( node,node1, dModRedFact );
        mob2 = GetBoundaryMobility( node,node2, dModRedFact );
        mob3 = GetBoundaryMobility( node,node3, dModRedFact );
		
        //mob1 = mob2 = mob3 =GetBoundaryMobility( node,node1 );
		
        //Get the length of the vectors node-node1 and node-node2
        //And of course get the vetors itself (turned by 90deg)
        ElleNodePosition( node, & nodexy );
        ElleNodePlotXY( node1, & old1xy, &nodexy );
        ElleNodePlotXY( node2, & old2xy, &nodexy );
	    ElleNodePlotXY( node3, & old3xy, &nodexy );
       //calculate vectors
        vec1.y = -1*(old1xy.x-nodexy.x);
        vec1.x = old1xy.y-nodexy.y;
		xymid[0].x=nodexy.x+(old1xy.x-nodexy.x)*0.5;
		xymid[0].y=nodexy.y+(old1xy.y-nodexy.y)*0.5;
		
        //here we get the length of the vector
        l1 = GetVectorLength( vec1 );
        //calculate the second vector
        vec2.y = -1*(old2xy.x - nodexy.x);
        vec2.x = old2xy.y-nodexy.y;
		xymid[1].x=nodexy.x+(old2xy.x-nodexy.x)*0.5;
		xymid[1].y=nodexy.y+(old2xy.y-nodexy.y)*0.5;
		
        //get length of vector
        l2 = GetVectorLength( vec2 );
		//calculate the third vector
        vec3.y = -1*(old3xy.x-nodexy.x);
        vec3.x = old3xy.y-nodexy.y;
		xymid[2].x=nodexy.x+(old3xy.x-nodexy.x)*0.5;
		xymid[2].y=nodexy.y+(old3xy.y-nodexy.y)*0.5;
		
        //get length of vector		
        l3 = GetVectorLength( vec3 );

		// Boundary Mobility function of misorientation
		if (ElleFlynnAttributeActive(EULER_3)|| 
            ElleFlynnAttributeActive(CAXIS)  ||
            ElleUnodeAttributeActive(EULER_3)   ) // FS added this line
        {	
			int rgn1, rgn2;
    		double misorientation;
    		
    		n = 0;
		
    		while( n < 3 )
    		{
				//cout << "(" << n << ") " << node << " " << nbnode[1] << " " << nbnode[2] << " " << nbnode[3] << endl;
				ElleNeighbourRegion( node, nbnode[n], &rgn[0] );
				ElleNeighbourRegion( nbnode[n], node, &rgn[1] );
				
				if ( id != 1) { // this is the flynn version .. 															
					misorientation=GetMisorientation(rgn[0],rgn[1],id);
					mob[n]=GetBoundaryMobility(node,nbnode[n],dModRedFact)*Get2BoundaryMobility(misorientation);
				}	
				else { // this is the unodes version .. 		
					// First version, take information of the nearest unode in both regions 
					rgn1=SearchUnode(rgn[0],&xymid[n]);
					rgn2=SearchUnode(rgn[1],&xymid[n]);
					misorientation=GetMisorientation(rgn1,rgn2,id);
					mob[n] = GetBoundaryMobility(node,nbnode[n],dModRedFact) * Get2BoundaryMobility(misorientation);
				}
				// printf("Node Mis Mob %i %e %e\n",node, misorientation,mob[n]);	
				
				// somehow the normal way resulted in an error because the loop continued to the 4th neighbour...		
				if ( n == 2 )
					n = 5;
				else
					n++;
			}
			mob1=mob[0];
			mob2=mob[1];
			mob3=mob[2];	
		}
		
		if (mob1 == 0) mob1=1e-20; 
		if (mob2 == 0) mob2=1e-20;
		if (mob3 == 0) mob3=1e-20;



       //If the energies are equal, pvec will be the 0 and everything will crash. So we do not allow this.
        //Physically this should be correct too since if pvec is 0, no movement should take place.
        //pvec equals the force
        F.x = pvec.x;
        F.y = pvec.y;

		len=GetVectorLength(F);
		F.x=F.x/len;
		F.y=F.y/len;
		
        cosalpha1 = DEGCos( vec1, F );
        cosalpha2 = DEGCos( vec2, F );
        cosalpha3 = DEGCos( vec3, F );
        if ( cosalpha1 < 0 )
            cosalpha1 *= -1;
        if ( cosalpha2 < 0 )
            cosalpha2 *= -1;
        if ( cosalpha3 < 0 )
            cosalpha3 *= -1;
        //in case of isotropic crystals this is sufficient (because the mobility will be the same on all
        //segments)
        //V0.x = ( 2 * F.x * mob1 )/ ( ( l1 * pow( cosalpha1, 2 ) ) + ( l2 * pow( cosalpha2, 2 ) ) + ( l3 * pow( cosalpha3, 2 ) ) );
        //V0.y = ( 2 * F.y * mob1 )/ ( ( l1 * pow( cosalpha1, 2 ) ) + ( l2 * pow( cosalpha2, 2 ) ) + ( l3 * pow( cosalpha3, 2 ) ) );
        //in case of anisotropic crystals this is needed because it takes the mobilities of both segments
        //into account

		// LE boundary lengths in Elle units,  mobilities in m2s-1
l1 *= ElleUnitLength();
l2 *= ElleUnitLength();
l3 *= ElleUnitLength();
				// limited to 89DEG , to test .. I think can be deleted..
	if ( cosalpha1 < 0.01745 ) 	cosalpha1=0.01745;
	if ( cosalpha2 < 0.01745 ) 	cosalpha2=0.01745;	
	if ( cosalpha3 < 0.01745 ) 	cosalpha3=0.01745;

aa=( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 )
                               + ( l3 * pow( cosalpha3, 2 ) / mob3 ) );			
		
    ///FS: print a view things to check what goes wrong in case of an error:
    if (isnan(aa))
    {
        // Get relevant node xy again and print data:        
        Coords FSnodexy,FSnode1xy,FSnode2xy,FSnode3xy;
        ElleNodePosition( node, & FSnodexy );
        ElleNodePosition( node1, & FSnode1xy );
        ElleNodePosition( node2, & FSnode2xy );
        ElleNodePosition( node3, & FSnode3xy );
       
        if(FSnodexy.x==FSnode3xy.x && FSnodexy.y==FSnode3xy.y) 
        {
            /* Two nodes are at exactly the same position, which will cause
             * trouble: Trying to solve this using ElleGGTopocheks:*/     
            printf("MoveTnode: PROBLEM WITH COINCIDING NODES ");
            CheckNodesCoincide2();
            ElleGGTopoChecks();
            
            // Check if problem still is not solved:
            if(FSnodexy.x==FSnode3xy.x && FSnodexy.y==FSnode3xy.y) 
            {
                printf("\nError in MoveTNode:\n");
                printf("The relevant node coordinates appear equal to the ");
                printf("ones of a neighbour node!\nProblem was not solved ");
                printf("using topology checks!\n");
                printf("\nData:\n");
                printf("node: %d, neighbour1: %d, neighbour2: %d, neighbour 3: %d\n",node,node1,node2,node3);
                printf("node x,y = %e , %e\n",FSnodexy.x,FSnodexy.y);
                printf("node1 x,y = %e , %e\n",FSnode1xy.x,FSnode1xy.y);
                printf("node2 x,y = %e , %e\n",FSnode2xy.x,FSnode2xy.y);
                printf("node3 x,y = %e , %e\n",FSnode3xy.x,FSnode3xy.y);
                printf("F.x = %e, F.y = %e\n",F.x,F.y);
                printf("len = %e\n",len);
                printf("normal to segment1 x,y = %e , %e\n",vec1.x,vec1.y);
                printf("normal to segment2 x,y = %e , %e\n",vec2.x,vec2.y);
                printf("normal to segment3 x,y = %e , %e\n",vec3.x,vec3.y);
                printf("cosalpha1 = %e, cosalpha2 = %e, cosalpha3 = %e\n",cosalpha1,cosalpha2,cosalpha3);
                printf("l1 = %e, l2 = %e, l3 = %e\n",l1,l2,l3);
                printf("mob1 = %e, mob2 = %e, mob3 = %e\n",mob1,mob2,mob3);
                printf("aa = %e\n\n",aa);    
                
                //Write log-elle file at this point
                //time_t timeID = time(0);
                //char filename [50];
                //sprintf(filename,"%ld_NAN_TNode_stage%d.elle",timeID,Settings_run.Count);  
                //ElleWriteData(filename);
                
                fDataFile << "Stage: " << Settings_run.Count << 
                " ERROR (MoveTNode - nan error): Bnode position of bnode " 
                << node << " coincide with the neighbour bnode " << node3 
                << endl; 
            }
            else
            {
                printf("MoveTnode: SOLVED\n");
            }
        }
    }
    
    ///FS: end of this testing part

	if (aa <= 1e-5 || isnan(aa)) { /// FS added isnan(aa)
 		V0.x=0.0;
		V0.y=0.0;
	}
	else
	{
        V0.x = ( 2 * F.x*len)/aa;
        V0.y = ( 2 * F.y*len )/aa ;

	}
        //V0.x /= ElleUnitLength()*ElleUnitLength(); // FS: WHY UnitLength squared??
        //V0.y /= ElleUnitLength()*ElleUnitLength(); // FS: WHY UnitLength squared??
        V0.x /= ElleUnitLength(); // FS: Removed the square
        V0.y /= ElleUnitLength(); // FS: Removed the square

        //Distance (D) of movement =velocity *ElleTimestep()
        flength = GetVectorLength( V0 );
		dt = ElleTimestep() * ElleSpeedup();
        if ( flength * ElleTimestep() * ElleSpeedup() >  ElleSwitchdistance() )
        {

			if (ElleSpeedup() > 1.0)
			{
				ElleSetSpeedup(1.0);
				fDataFile << "Stage: " << Settings_run.Count << " (T) Resetting speedup to " << ElleSpeedup() << endl;
				dt = ElleTimestep() * ElleSpeedup();
			}
        	if ( flength * ElleTimestep() >  ElleSwitchdistance() )
			{
                if (FS_NodeOnExcludeBoundary(node))
                {
                    /* Accuracy on this boundary is not that important 
                     * (e.g. air-air): Choose highest number possible without
                     * warning the user*/
                    dt = ElleSwitchdistance()/flength/ElleSpeedup();
                    //printf("WARNING (ModeTNode): Excluded phase boundary node is too fast -> Setting to maximum velocity possible\n");
                }
                else
                {
                    //dt = ElleSwitchdistance(); // BEFORE: If timestep too large no movement at all 
                    dtMAX = ElleSwitchdistance()/flength/ElleSpeedup(); ///JR Not sure it maximal movement would be still too big ?
                    dt = dtMAX*0.9; // NOW: dt is 90% of maximum timestep possible
                    /*!
                    printf("WARNING (ModeTNode): Timestep too large for movement: Check MoveNode logfile\n");
                    fDataFile << "Stage: " << Settings_run.Count << 
                    " (TNode " << node << ") SwitchDistance or TimeStep needs adjustment to prevent this!" 
                    << endl << "Local Resetting timestep to " << scientific 
                    << dtMAX*0.9 << " (90% of max. timestep)! from global time step " << scientific << ElleTimestep() 
                    << endl << "--> Maximum possible dt would have been: " 
                    << scientific << dtMAX << endl;
                    */
                    (*iNodeTooFast2)++;
                    (*dTimeStepHighest2)+=dtMAX;
                    //printf("MoveTNode: Too fast, total number: %u, total sum dtMax = %e\n",*iNodeTooFast2,*dTimeStepHighest2);
                }
				
			} 
		}
        movedir->x = V0.x * dt;
        movedir->y = V0.y * dt;
        
        FS_SetBnodeVelocityAttrib(node,V0);
        move = 1;

    }
    fDataFile.close();
    return (move);
}

/*
 * Return the misorientation between two flynns or unodes:
 * id == 1 means use unode version and rgn1 and rgn2 will be unode IDs
 * id != 1 means use flynn version and rgn1 and rgn2 will be flynn IDs
 */
double GetMisorientation(int rgn1, int rgn2, int id)
{
    int i;
	double cosalpha = 0;
    Coords_3D xyz[3];
    double val[3];
	double misorientation=0,tmp,dotprod;
	
    for (i=0;i<3;i++) xyz[i].x = xyz[i].y = xyz[i].z = 0;

	if (id != 1) { // flynn version
		
		if (ElleFlynnAttributeActive(EULER_3))
		{                                                                            
    		ElleGetFlynnEuler3(rgn1,&val[0],&val[1],&val[2]);
			ElleGetFlynnEulerCAxis(val[0],val[1],val[2],&xyz[0]);
		
			ElleGetFlynnEuler3(rgn2,&val[0],&val[1],&val[2]);
			ElleGetFlynnEulerCAxis(val[0],val[1],val[2],&xyz[1]);
		}
		else if (ElleFlynnAttributeActive(CAXIS))
		{
			ElleGetFlynnCAxis(rgn1,&xyz[0]);
			ElleGetFlynnCAxis(rgn2,&xyz[1]);
		}
	}
	else {  // unode version, id ==1
		//unode1 		
		ElleGetUnodeAttribute(rgn1, &val[0],E3_ALPHA);
		ElleGetUnodeAttribute(rgn1, &val[1],E3_BETA);
		ElleGetUnodeAttribute(rgn1, &val[2],E3_GAMMA);		
		ElleGetFlynnEulerCAxis(val[0],val[1],val[2],&xyz[0]);	
		//unode2
		ElleGetUnodeAttribute(rgn2, &val[0],E3_ALPHA);
		ElleGetUnodeAttribute(rgn2, &val[1],E3_BETA);
		ElleGetUnodeAttribute(rgn2, &val[2],E3_GAMMA);		
		ElleGetFlynnEulerCAxis(val[0],val[1],val[2],&xyz[1]);
	}		
                                                                           
	dotprod=(xyz[0].x*xyz[1].x)+(xyz[0].y*xyz[1].y)+(xyz[0].z*xyz[1].z);
	                                                                            
	if(fabs(dotprod-1.0) < 0.0001)
	misorientation=0.0;
	else
	misorientation = fabs(acos(dotprod)*RTOD);
	                                                                            
	if(misorientation>90.0)
	misorientation=180-misorientation;	

	
	return(misorientation);
}

/*
 * This is the elle function with some adaptions to this code
 */
int ElleGetFlynnEulerCAxis(double alpha, double beta, double gamma, Coords_3D *dircos)
{
    // double alpha,beta,gamma;                                                                 
    // ElleGetFlynnEuler3(flynn_no,&alpha,&beta,&gamma);
                                                                                
    alpha=alpha*DTOR;
    beta=beta*DTOR;
                                                                                
    dircos->x=sin(beta)*cos(alpha);
    dircos->y=-sin(beta)*sin(alpha);
    dircos->z=cos(beta);
                                                                                
    //converts Euler angles into direction cosines of c-axis
}

/*
 * GetBoundaryMobility:
 * Return the 'real' boundary mobility as a Arrhenius type function of base 
 * mobility and temperature
 * 
 * IMPORTANT INFORMATION:
 * 
 * Intrinsic boundary mobilities (m0) of the sedments are derived from 
 * phase_db.txt input file
 * General formula:  (cf. Humphreys & Hatherly 2004, pp. 123)
 * mobility=mo*exp((-H/(B*T))
 * mobility=mo*exp((-Q/(R*T)) --> This is used here
 * 
 * mo: 
 * Base mobility, taken from mineraldb or phase_db.txt (polyphase gbm input) 
 * B and R:
 * Gas constant R = 8.314472 in Jmol-1K-1
 * Boltzmann-constant B = 1.3806505e-23 J*K^-1 according to this: 
 * http://www2.mpie-duesseldorf.mpg.de/msu-web/msuweb_neu/deutsch/themenkatalog/3d-cellular-automaton/3d-cellular-automaton.htm
 * Q and H:
 * H is activation energy per molecule and Q (in J) is per mole (in J/mol). 
 * For an aluminium polycrystall H = 1.6 eV, 1 eV = 1.6022e-19 J so H for 
 * aluminium should be 2.56353e-19 Joule, so Q = 145379 J/mol. For Olivine it is 
 * around 550 kJ/mol (Karato 1989: Grain Growth Kinetics in Olivine Aggregates, 
 * Tectonophysics, 168, 255-273) nobody knows how much it is for e.g. Quartz or 
 * something, at least I have not found anything anywhere. 
 * 
 * FS: H for ice is determined to 0.53 eV by Nasello et al. 2005, that means:
 * Q = 0.53*1.602e-19J*6.022e23 1/mol = 51.13e3 J/mol
 * FS: Intrinsic or base mobility m0 for ice has been determined to:
 * m0 = 0.023 m4/(s·J) or (m2/Kg)·s Nasello et al. 2005 in 1st stage of groove movement-low velocity
 */
double GetBoundaryMobility( int node, int nb,double dModRedFact )
{
    double dBaseMobility = 0.0, Q = 0.0, T = 0.0, dMobility = 0.0, R = 8.314472;

	/* 
     * H is activation energy per molecule, eV or J
     * Q is activation energy per mole in J mol-1 (or eV mol-1)
     * 
     * When the activation energy is given in molecular units, instead of
     * molar units, the Boltzmann constant (B) is used instead of the
     * gas constant(R). Avogadro = 6.022e23;
     * 
     * J Metamorphic Geol., 1997, 15, 311-322
     * Q = 11 kcal mol-1 = 53240 J mol-1  for quartz
     * Nasello et al. 2005:
     * H = 0.53 eV which makes Q = 51.13e3 J/mol for ice
     */

    T = ElleTemperature()+273.15; // Get T in Kelvin

	// Addition by Jens to read values from the config file instead...
	Q = CheckPair(node, nb, 2); // for ice
    dBaseMobility = CheckPair(node, nb, 0);
	
    // FS: Comments by Albert I guess due to the ?¿ :-)
	// Caution !! Data of Mo from grain boundary Motion of pure ice bicrystals!! with first and second thermal groove!!
    // The intrinsic mobility change 2 order of magnitude (0.023 to 2.3 m4 J-1 s-1 !). Using first, perhaps if driving force 
	// is high change to second ?¿ 
    // At 263ºK, mobil = 1.6e-12 to 1.6e-10 m2 s-1     	
	//mobil=mobility * exp( -(H ) / ( B * T ) );
	// or
    
    /*! Lower the "base mobility" (intrinsic mobility) in the middle third */
    Coords cNodePos;
    ElleNodePosition(node,&cNodePos);
    
    CellData unitcell;
    ElleCellBBox(&unitcell);
    double dBoxHeight = unitcell.cellBBox[TOPLEFT].y - unitcell.cellBBox[BASELEFT].y;
    
    double dUpperLim = 2.0/3.0, dLowerLim = 1.0/3.0;
    double dLoweringFact = 0.0;
    
    dUpperLim *= dBoxHeight;
    dLowerLim *= dBoxHeight;
    
    if (cNodePos.y<dUpperLim)
    {
        if (cNodePos.y>dLowerLim)
        {
            dBaseMobility *=dModRedFact;
        }
	}
    //printf("Not reducing mobility M0 = %f\n",dBaseMobility);
	dMobility = dBaseMobility * exp( -(Q ) / ( R * T ) );
	
    return dMobility;
}

/*
 * Gets boundary mobility as a function of misorientation 
 * (see Holm et al., 2003 etc.)
 */
double Get2BoundaryMobility(double misorientation)
{ 
    double thetaHA=15.0,tmp;   	
    int d=4, AA=5;

    tmp=1.0;
	
	if (misorientation <= thetaHA) 
    {
		tmp=misorientation/thetaHA;  	
		tmp=AA*(pow(tmp,d));
    	tmp=1-exp(-tmp);	
    }
    return(tmp);
}	

/* !/brief Returns the length of the vector */
double GetVectorLength( Coords vec )
{
    return sqrt( ( vec.x * vec.x ) + ( vec.y * vec.y ) );
}

/* !/brief Calculatest the cosine of the two vectors. */
double DEGCos( Coords vec1, Coords vec2 )
{
    return ( vec1.x * vec2.x + vec1.y * vec2.y ) / ( GetVectorLength( vec1 ) * GetVectorLength( vec2 ) );
}

/*
 * Searching nearest unode to any position in space
 */
int SearchUnode (int rgn, Coords *xy)
{
    int max_unodes,k,jj=0, unode_id; 
    double roi, dist, min_dist; 
    Coords xy_unode;
        
    vector<int> unodelist;    	
        
    // Search nearest unode to a space position.  
    // Area of search is limited by ROI 	
    // npts=sqrt(max_unodes);
    // Add a check of unodelist?¿ , if smaller than a critical number than?¿ 		
        
         
        //roi = sqrt(1.0/(double)max_unodes/3.142)*5;	// aprox. limit at 3nd neighbours
        roi = FS_GetROI(5);

        ElleNodeUnitXY(xy);
        ElleGetFlynnUnodeList(rgn,unodelist);
        min_dist=roi;
            
            for(k=0;k<unodelist.size();k++) {
                      
                ElleGetUnodePosition(unodelist[k],&xy_unode); // xy es del node !!
                ElleCoordsPlotXY (&xy_unode, xy);
                dist = pointSeparation(xy,&xy_unode);
                      
                    if (dist<min_dist) {
                        unode_id=unodelist[k];
                        min_dist = dist;					
                        jj ++;
                        }
                  }
                // unodelist.clear();

          // SCAN ALL THE UNODES  
        if (jj == 0) {
            max_unodes = ElleMaxUnodes();
            min_dist=roi;
            
            for(k=0;k<max_unodes;k++) {
                ElleGetUnodePosition(k,&xy_unode);
                ElleCoordsPlotXY (&xy_unode, xy);
                dist = pointSeparation(xy,&xy_unode);
                      
                if (dist<min_dist) {
                    unode_id=k;
                    min_dist = dist;					
                    jj ++;
                    }
                } 
            }
    return(unode_id);
}

/* 
 * FS: This is a function to colour the bnodes for their velocity later
 */
void FS_SetBnodeVelocityAttrib(int node,Coords cVelocity)
{
    if (!ElleNodeAttributeActive(N_ATTRIB_A))
    {
        ElleInitNodeAttribute(N_ATTRIB_A);
        // Actually we should set a default value=0 here        
    }
    double dAbsoluteVelocity = GetVectorLength( cVelocity );
    dAbsoluteVelocity /= 1e-6; // normalise to 1e-6, value will then be in µm/s
    //printf("velocity vector length = %e\n",dAbsoluteVelocity);
    ElleSetNodeAttribute(node,dAbsoluteVelocity,N_ATTRIB_A);
}

/* FS: This one tells you on which boundary type the node sits on in a 2 phase
 * model. 
 * if return = 0: phase1-phase1 (ice-ice)
 * if return = 1: phase1-phase2 (ice-air)
 * if return = 2: phase2-phase2 (air-air)
 */
int FS_DEBUG_FUNCTION_BoundaryType(int iNode)
{
    int iRgns[3]; 
    double dPhase[3];
    bool bNodeisDouble = false;
    
    ElleRegions(iNode,iRgns);
    
    for (int i=0;i<3;i++)
    {
        if (iRgns[i]!=NO_NB)
        {
            ElleGetFlynnRealAttribute(iRgns[i],&dPhase[i],VISCOSITY); 
        }
        else
        {
            bNodeisDouble = true;
            dPhase[i] = 0; // phase can never be 0
        }                
    }
    
    // Decide for boundary type by summing up the values in dPhase:
    double dSumValue = dPhase[0]+dPhase[1]+dPhase[2];
    
    if (bNodeisDouble)
    {
        // double junction
        if (dSumValue/2 == 1) return 0;
        if (dSumValue/2 == 1.5) return 1;
        if (dSumValue/2 == 2) return 2;
    }
    else
    {
        // triple junction
        if (dSumValue/3 == 1) return 0;
        if (dSumValue/3 < 2 && dSumValue/3 > 1) return 1;
        if (dSumValue/3 == 2) return 2;
        
    }
}
