/*Modified 11/12/2008 to add boundary mobility as function of misorientation angle (AGA)  
*/

#include "movenode.h"

double GetMisorientation(int rgn1, int rgn2, int id);
double Get2BoundaryMobility(double misorientation);

/*!This function returns a vector in movedir. The node should move in the direction of movedir for the distance of
length of movedir. This should be used with ElleUpdatePosition. The function returns 1 if the node has to be moved, 0 if not.
If it should not be moved, the node either was inactive of the passed energies did not cause the node movement*/
int GetMoveDir( int node, Coords pvec, Coords * movedir,double trialdist )
{
	
    if ( pvec.x != 0 && pvec.y != 0 )
    {
        if ( ElleNodeIsTriple( node ) )
            return MoveTNode( node, pvec, movedir );
        else
            return MoveDNode( node, pvec, movedir );
    }
    else
        return 0;
}
		

/*!This function calculates a new position of the node node. It should be used together with ElleSetPosition.The function
returns 1 if the node has to be moved, 0 if not. If it should not be moved, the node either was inactive of the passed
energies did not cause the node movement*/
int GetNewPos( int node, double e1, double e2, double e3, double e4, Coords * newpos,double trialdist )
{
    Coords npos, pvec;
    int t;
    //Get the vector of the direction with highest decrease of energy
   pvec.x = -( e1 - e2 ) / ( 2 * trialdist );
    pvec.y = -( e3 - e4 ) / ( 2 * trialdist );
    if ( pvec.x != 0 && pvec.y != 0 )
    {
        ElleNodePosition( node, & npos );
        if ( ElleNodeIsTriple( node ) )
        {
            t = MoveTNode( node, pvec, newpos );
            newpos->x += npos.x;
            newpos->y += npos.y;
        }
        else
        {
            t = MoveDNode( node, pvec, newpos );
            newpos->x += npos.x;
            newpos->y += npos.y;
        }
        return t;
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
int MoveDNode( int node, Coords pvec, Coords * movedir )
{
    double l1, l2, cosseg, cosalpha1, cosalpha2, mob1, mob2, flength,type,len;
    int nbnode[3], node1, node2, err, n;
	int move = 0, dummy=0;
	double dt = 0, mob[2];
    Coords F, V0, nodexy, old1xy, old2xy, vec1, vec2, xymid[2];
    char logbuf[4096];
	Coords sup1, sup2, sup3, vec_body,V0B,FB, V0S;
	double density[2],len_body, bodyenergy, diff_density[4],len_sup1,len_sup2,len_sup3;
	double aa,aaa, area, density_energy;
    int rgn[3], nnbnode[3],node3,i,j;
	ERegion rgn2[3];
	int id,l;
	
	id=0; // if id != 1 working with flynn, id ==1 working with unodes.. define as a user value 
	
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
        mob1 = mob2 =GetBoundaryMobility( node,node1 );///2; //in Elle, boundaries belong to two flynns, therefore, the division by two
	
        //Get the length of the vectors node-node1 and node-node2
        //And of course get the vetors itself (turned by 90deg, we need the vector normals)
        ElleNodePosition( node, & nodexy );
        ElleNodePlotXY( node1, & old1xy, &nodexy );
        ElleNodePlotXY( node2, & old2xy, &nodexy );
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

	
		// Boundary Mobility function of misorientation
		if (ElleFlynnAttributeActive(EULER_3)|| ElleFlynnAttributeActive(CAXIS)) {
		
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
				mob[l]=GetBoundaryMobility(node,nnbnode[n])*Get2BoundaryMobility(misorientation);
				l++;
				}	
				
			else { // this is the unodes version .. 		
				// First version, take information of the nearest unode in both regions 
				rgn1=SearchUnode(rgn[0],&xymid[n]);
				rgn2=SearchUnode(rgn[1],&xymid[n]);
				
				misorientation=GetMisorientation(rgn1,rgn2,id);
				mob[l]=GetBoundaryMobility(node,nnbnode[n])*Get2BoundaryMobility(misorientation);
				l++;
				}
			// printf("Node Mis Mob %i %e %e\n",node, misorientation,mob[n]);	
			n++;
			}
			mob1=mob[0];
			mob2=mob[1];
    	}	

        //If the energies are equal, pvec will be the 0 and everything will crash. So we do not allow this.
        //Physically this should be correct too since if pvec is 0, no movement should take place.
        //pvec equals the force
        F.x = pvec.x;
        F.y = pvec.y;

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

	if (aa <1e-5) {
		V0.x=0.0;
		V0.y=0.0;
	}
	else
	{
        V0.x = ( 2 * F.x*len)/aa;
         V0.y = ( 2 * F.y*len )/aa ;
	}

        V0.x /= ElleUnitLength()*ElleUnitLength();
        V0.y /= ElleUnitLength()*ElleUnitLength();

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
            sprintf( logbuf, "SwitchDistance or TimeStep needs adjustment to prevent this!\n" );
            Log( 0, logbuf );
			if (ElleSpeedup() > 1.0) 
			{
				ElleSetSpeedup(1.0);
            	sprintf( logbuf, "Resetting speedup to %e! ", ElleSpeedup() );
            	Log( 0, logbuf );
			 dt = ElleTimestep() * ElleSpeedup();
			}
        	if ( flength * ElleTimestep() >  ElleSwitchdistance() )
			{
			//	ElleSetTimestep(ElleSwitchdistance()/flength/ElleSpeedup());
            	sprintf( logbuf, "Resetting timestep to %e! ", ElleTimestep() );
            	Log( 0, logbuf );
			//	dt = ElleTimestep() * ElleSpeedup();
				 dt = ElleSwitchdistance();
			}
		}
		
		movedir->x = V0.x * dt;
		movedir->y = V0.y * dt;
	
		if (dummy == 1) printf("DN movedir %i\t %e  %e\n",node, movedir->x,movedir->y);
		move = 1;


    }
    return (move);
}


/*!/brief This will move a triple node.
 
See MoveDNode for more explanantion*/
int MoveTNode( int node, Coords pvec, Coords * movedir )
{
    double l1, l2, l3,len, cosalpha1, cosalpha2, cosalpha3, mob1, mob2, mob3, flength;
    int nbnode[3], node1, node2, node3, err;
	int move = 0, dummy=0;
	double dt = 0, aa, aaa, mob[2];
    Coords F, V0, nodexy, old1xy,old2xy,old3xy, vec1, vec2, vec3, xymid[2];
    char logbuf[4096];
	Coords sup1, sup2, sup3, vec_body,V0B,FB;
	double density[2],len_body, bodyenergy,diff_density[4],len_sup1,len_sup2,len_sup3;
    int rgn[3];
	int id, n;
	
	id=0; // if id != 1 working with flynn, id ==1 working with unodes.. define as a user value 
	
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
        mob1 = mob2 = mob3 =GetBoundaryMobility( node,node1 );
		
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
	if (ElleFlynnAttributeActive(EULER_3)|| ElleFlynnAttributeActive(CAXIS)) {
		
			int rgn1, rgn2;
    		double misorientation;  
    		n=0;
		
    		while ( n < 3 )
    		{
			// if ( rgn[n] != -1 ) {
			ElleNeighbourRegion( node, nbnode[n], &rgn[0] );
    		ElleNeighbourRegion( nbnode[n], node, &rgn[1] );
				
			if ( id != 1) { // this is the flynn version .. 															
				misorientation=GetMisorientation(rgn[0],rgn[1],id);
				mob[n]=GetBoundaryMobility(node,nbnode[n])*Get2BoundaryMobility(misorientation);
				}	
				
			else { // this is the unodes version .. 		
				// First version, take information of the nearest unode in both regions 
				rgn1=SearchUnode(rgn[0],&xymid[n]);
				rgn2=SearchUnode(rgn[1],&xymid[n]);
				
				misorientation=GetMisorientation(rgn1,rgn2,id);
				mob[n]=GetBoundaryMobility(node,nbnode[n])*Get2BoundaryMobility(misorientation);
				}
			// printf("Node Mis Mob %i %e %e\n",node, misorientation,mob[n]);
			// }				
			n++;
			}
	mob1=mob[0];
	mob2=mob[1];
	mob3=mob[2];		
    }	

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

	if (aa <1e-5 ) {
		V0.x=0.0;
		V0.y=0.0;
	}
	else
	{
        V0.x = ( 2 * F.x*len)/aa;
        V0.y = ( 2 * F.y*len )/aa ;

	}

        V0.x /= ElleUnitLength()*ElleUnitLength();
        V0.y /= ElleUnitLength()*ElleUnitLength();

        //Distance (D) of movement =velocity *ElleTimestep()
        flength = GetVectorLength( V0 );
		dt = ElleTimestep() * ElleSpeedup();
        if ( flength * ElleTimestep() * ElleSpeedup() >  ElleSwitchdistance() )
        {
            //sprintf( logbuf, "Distance too big to move triple! angles: %lf,%lf,%lf Distance: %lf, 2*Switchdistance: %lf\n",
                     //acos( cosalpha1 ) * RTOD, acos( cosalpha2 ) * RTOD, acos( cosalpha3 ) * RTOD, flength * ElleTimestep(),
                     //2 * ElleSwitchdistance() );
            //Log( 2, logbuf );
            //sprintf( logbuf, "reduced movement of node %d! ", node );
            //Log( 0, logbuf );
            sprintf( logbuf, "SwitchDistance or TimeStep needs adjustment to prevent this!\n" );
            Log( 0, logbuf );
			if (ElleSpeedup() > 1.0) 
			{
				ElleSetSpeedup(1.0);
            	sprintf( logbuf, "Resetting speedup to %e! ", ElleSpeedup() );
            	Log( 0, logbuf );
				dt = ElleTimestep() * ElleSpeedup();
			}
        	if ( flength * ElleTimestep() >  ElleSwitchdistance() )
			{
				//ElleSetTimestep(ElleSwitchdistance()/flength/ElleSpeedup());
            	sprintf( logbuf, "Resetting timestep to %e! ", ElleTimestep() );
            	Log( 0, logbuf );
				dt = ElleSwitchdistance();
			}
		}
        movedir->x = V0.x * dt;
        movedir->y = V0.y * dt;
		
	   if (dummy == 1) printf("TN movedir %i\t %e\t%e\n",node, movedir->x,movedir->y);
		   
        move = 1;

    }
    return (move);
}

/* !/brief Calculatest the cosine of the two vectors. */
double DEGCos( Coords vec1, Coords vec2 )
{
    return ( vec1.x * vec2.x + vec1.y * vec2.y ) / ( GetVectorLength( vec1 ) * GetVectorLength( vec2 ) );
}

/* !/brief Returns the length of the vector */
double GetVectorLength( Coords vec )
{
    return sqrt( ( vec.x * vec.x ) + ( vec.y * vec.y ) );
}

/*!/brief Get the 'real' boundary mobility
 
General formula: mobility=mo*exp((-H/(B*T)). mo is a base mobility, taken from mineraldb, B(oltzmann-constant) is in J*K^-1
according to this: http://www2.mpie-duesseldorf.mpg.de/msu-web/msuweb_neu/deutsch/themenkatalog/3d-cellular-automaton/3d-cellular-automaton.htm
H (activation energy) for an aluminium polycrystall is 1.6 eV 1 eV = 1.6022e-19 Joule so H for aluminium should be 2.56353e-19
Joule, so it is 145379 J/mol for Olivine it is around 550 kJ/mol (Karato 1989: Grain Growth Kinetics in Olivine Aggregates,
Tectonophysics, 168, 255-273) nobody knows how much it is for e.g. Quartz or something, at least I have not found anything
anywhere. Gas constant 8.314472 in Jmol-1K-1. */
double GetBoundaryMobility( int node, int nb )
{
    double mobility = 0, mobil, arg2, arg1, arg, T, R = 8.314472, Q, H = 145, B = 1.3806505e-23,type;
	double eV_to_J = 1.6022e-19;
    int mineral, rgn[3], n;
    double mob2=0;

	/*
	// stuff for multi phases, only one phase ICE  
    //we need to know which flynn it is and get the boundary mobility for the segments.
    ElleNeighbourRegion( node, nb, &rgn[0] );
    ElleNeighbourRegion( nb, node, &rgn[1] );
            
    ElleGetFlynnMineral( rgn[0], & mineral );
    mobility = GetMineralAttribute( mineral, GB_MOBILITY);
    ElleGetFlynnMineral( rgn[1], & mineral );
    mob2 = GetMineralAttribute( mineral, GB_MOBILITY);
    if (mob2<mobility) mobility=mob2;
	*/
	
    //we need to store activation energies for the MINERAL
	// and decide on the units
	// e.g 0.53eV for ice (-> 8.49e-18J)
	// H *= eV_to_J;
	// H is activation energy per molecule, eV or J
	// Q is activation energy per mole in J mol-1 (or eV mol-1)
	// When the activation energy is given in molecular units, instead of
	// molar units, the Boltzmann constant (B) is used instead of the
	// gas constant(R). Avogadro = 6.022e23;
	//
	// J Metamorphic Geol., 1997, 15, 311-322
	// Q = 11 kcal mol-1 = 53240 J mol-1  for quartz
/*
    mobility += GetMineralAttribute( mineral, GB_MOBILITY);
    mobility /= 2; // LE
*/
    //I suspect that the temperature is in celsius, not in Kelvin?????? If so, we would have to add 273.15 ....
    T = ElleTemperature()+273.15;
	// T= 263; // test ElleTemperature to -10.. 
	
	Q = GetMineralAttribute_ice(GB_ACTIVATION_ENERGY_ice); // for ice
    mobility = GetMineralAttribute_ice(GB_MOBILITY_ice);
    
	// Caution !! Data of Mo from grain boundary Motion of pure ice bicrystals!! with first and second thermal groove!!
    // The intrinsic mobility change 2 order of magnitude (0.023 to 2.3 m4 J-1 s-1 !). Using first, perhaps if driving force 
	// is high change to second ?¿ 
    // At 263ºK, mobil = 1.6e-12 to 1.6e-10 m2 s-1     	
	//mobil=mobility * exp( -(H ) / ( B * T ) );
	// or
	
	mobil=mobility * exp( -(Q ) / ( R * T ) );
	
    return mobil;
}


int GetBodyNodeEnergy2(int n, double *total_energy)
{
    int        i,nb[3];
    int themin, same;
    ERegion rgn[3];
    double  sum_energy;
    double   area[3],energy[3];
    double density[3], energyofdislocations;

    ElleNodeSameMineral(n, &same, &themin); // only allow GBM between same mineral type
    /*!
     * dislocationdensityscaling is the scaling for the DISLOCDEN
     * values in the elle file the values are usually ~1.0 and need
     * to be scaled to m-2
     *
     * the units for energyofdislocations are Jm-1
     *
     * the units for length are m
     */
    double dislocationdensityscaling = 10e13;

    energyofdislocations=(double)GetMineralAttribute_ice(DD_ENERGY_ice);
                                                                               
    ElleRegions(n,rgn);
    sum_energy =0;
     for (i=0;i<2;i++) {
        if (rgn[i]!=NO_NB) {
            // ElleGetFlynnRealAttribute(rgn[i],&density[i],DISLOCDEN);
            // area[i] = fabs(ElleRegionArea (rgn[i]))*
               //               (ElleUnitLength()*ElleUnitLength());
			area[i] = ElleSwitchdistance()*ElleSwitchdistance();
			
			area[i] = area[i]*ElleUnitLength()*ElleUnitLength();
			
            energy[i] = energyofdislocations*area[i];
                            
            //energy[i] = energyofdislocations*area[i]*density[i]*dislocationdensityscaling;
			
            sum_energy += energy[i];
            //printf("area=%le\tdensity=%le\tenergy=%le\n",area[i],density[i],energy[i]);
                                                                                
        }
    }
     // sum_energy=energyofdislocations*dislocationdensityscaling*ElleUnitLength()*ElleUnitLength();                                                                        
    *total_energy=sum_energy;
                                                                                
    return(0);
}

double GetMineralAttribute_ice (int attribute)
{
	double val;
	switch(attribute) {
		
	case GB_MOBILITY_ice : 	
		val=0.023; // m4/s·J or m2/Kg·s Nasello et al. 2005 Intrinsic Mobility Mo; 1st stage of groove movement-low velocity
		break;
	
	case GB_MOBILITY_2_ice : 	
		val=2.3; // m4/s·J or m2/Kg·s Nasello et al. 2005 Intrinsic Mobility Mo; 2nd stage of groove movement- fast
		break;
		
	case DD_ENERGY_ice :
		val=3.5e-10; // Core energy dislocations Gb**2/2, 3500 (MPa) *4.52e-10 (m)**2 /2; Jm-1
		break;
	
	case SURFACE_ENERGY_ice :
		val=0.065; // Jm-2
		break;

	case GB_ACTIVATION_ENERGY_ice :
		val=51.1e3; //  J mol-1
		break;	
	
	}
	return(val);
}

double Get2BoundaryMobility(double misorientation)
{
// Boundary mobility as function of misorientation (see Holm et al., 2003,etc)  
double thetaHA=15.0,tmp;   	
int d=4, AA=5;

tmp=1.0;
	
	if (misorientation <= thetaHA) {
		tmp=misorientation/thetaHA;  	
		tmp=AA*(pow(tmp,d));
    	tmp=1-exp(-tmp);	
		}
return(tmp);
		
	}		

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

int SearchUnode (int rgn, Coords *xy)
{
int max_unodes,k,jj=0, unode_id; 
double roi, dist, min_dist; 
Coords xy_unode;
 	
std::vector<int> unodelist;    	
	
// Search nearest unode to a space position.  
// Area of search is limited by ROI 	
// npts=sqrt(max_unodes);
// Add a check of unodelist?¿ , if smaller than a critical number than?¿ 		
	
	 
    roi = sqrt(1.0/(double)max_unodes/3.142)*5;	// aprox. limit at 3nd neighbours

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
