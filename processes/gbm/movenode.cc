#include <fstream>
#include "movenode.h"

int write_timestep_log(char *msg);

/*!This function returns a vector in movedir. The node should move in the direction of movedir for the distance of
length of movedir. This should be used with ElleUpdatePosition. The function returns 1 if the node has to be moved, 0 if not.
If it should not be moved, the node either was inactive of the passed energies did not cause the node movement*/
int GetMoveDir( int node, double e1, double e2, double e3, double e4, Coords * movedir,double trialdist )
{
    Coords pvec;
    //Get the vector of the direction with highest decrease of energy
    pvec.x = -( e1 - e2 ) / ( 2 * trialdist );
    pvec.y = -( e3 - e4 ) / ( 2 * trialdist );
    if ( pvec.x != 0 && pvec.y != 0 )
    {
            return MoveNode( node, pvec, movedir );
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
        if ( ElleNodeIsActive( node ) )
        {
            ElleNodePosition( node, & npos );
            t = MoveNode( node, pvec, newpos );
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
#if XY
int MoveDNode( int node, Coords pvec, Coords * movedir )
{
    double l1, l2, cosseg, cosalpha1, cosalpha2, mob1, mob2, flength,type,len;
    int nbnode[3], node1, node2, err, n;
    int move = 0;
    double dt = 0;
    Coords F, V0, nodexy, old1xy, old2xy, vec1, vec2;
    char logbuf[4096];
    if ( ElleNodeIsActive( node ) )
    {
        //Get the neighbouring nodes
        if ( err = ElleNeighbourNodes( node, nbnode ) )
            OnError( "MoveDNode", err );
        n = 0;
        while ( n < 3 && nbnode[n] == NO_NB )
            n++;
        node1 = nbnode[n];
        n++;
        while ( n < 3 && nbnode[n] == NO_NB )
            n++;
        node2 = nbnode[n];


        //Get Boundary mobility
        mob1 = GetBoundaryMobility( node,node1 );///2; //in Elle, boundaries belong to two flynns, therefore, the division by two
        mob2 = GetBoundaryMobility( node,node2 );///2;

        //Get the length of the vectors node-node1 and node-node2
        //And of course get the vetors itself (turned by 90deg, we need the vector normals)
        ElleNodePosition( node, & nodexy );
        ElleNodePlotXY( node1, & old1xy, &nodexy );
        ElleNodePlotXY( node2, & old2xy, &nodexy );
       //calculate vectors
        vec1.y = -1*(old1xy.x-nodexy.x);
        vec1.x = old1xy.y-nodexy.y;
        //here we get the length of the vector
        l1 = GetVectorLength( vec1 );
        //calculate the second vector
        vec2.y = -1*(old2xy.x - nodexy.x);
        vec2.x = old2xy.y-nodexy.y;
        //get length of vector
        l2 = GetVectorLength( vec2 );
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
        V0.x = ( 2 * F.x*len) / ( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 ) );
        V0.y = ( 2 * F.y*len ) / ( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 ) );
        /* } */
        V0.x /= ElleUnitLength();
        V0.y /= ElleUnitLength();
        flength = GetVectorLength( V0 );
        dt = ElleTimestep() * ElleSpeedup();
        //This should never happen. If it does, the timesteps are too big (or the switchdistance is too small) and needs to be adjusted.
        //In that case, other modules probably will fail too.
        if ( flength * ElleTimestep() * ElleSpeedup() >  ElleSwitchdistance() )
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
                sprintf( logbuf, "Resetting speedup to %e! ", ElleSpeedup() );
                Log( 0, logbuf );
             dt = ElleTimestep() * ElleSpeedup();
            }
            if ( flength * ElleTimestep() >  ElleSwitchdistance() )
            {
            //    ElleSetTimestep(ElleSwitchdistance()/flength/ElleSpeedup());
            //    sprintf( logbuf, "Resetting timestep to %e! ", ElleTimestep() );
            //    Log( 0, logbuf );
            //    dt = ElleTimestep() * ElleSpeedup();
                dt = ElleSwitchdistance();
            }
        }
        movedir->x = V0.x * dt;
        movedir->y = V0.y * dt;
        move = 1;
    }
    return (move);
}
#endif


/*!/brief This will move a triple node.
This function returns the direction and distance to move the node
in movedir. It returns
 0 if movedir is 0 (the node should not be moved this time) or
 1 if the node should be moved.
It also takes into account the timestep defined in the elle-file
(usually one year) and the speedup (usually one).
The combination of timestep, speedup and switchdistance has to be
balanced. If the switchdistance is very small and the timestep
is very large, the calculated distance for the node movement
can be bigger then the switchdistance. In that case, a warning message is logged, movedir is set to 0 and the function returns
0 (meaning the node should not be moved).*/
 
int MoveNode( int node, Coords pvec, Coords * movedir )
{
    double l[3], len=0.0;
    double cosalpha[3], mob[3], flength;
    int i, n, nbnode[3], node1, node2, node3, err;
    int move = 0;
    double dt = 0;
    Coords F, V0, nodexy, old1xy,old2xy,old3xy, vec1, vec2, vec3;
    char logbuf[4096];
    if ( ElleNodeIsActive( node ) )
    {
        node1 = node2 = node3 = NO_NB;
        for ( i=0; i<3; i++ ) nbnode[i] = NO_NB;
        //Get the neighbouring nodes
        if ( err = ElleNeighbourNodes( node, nbnode ) )
            OnError( "MoveNode", err );
        n = 0;
        while ( n < 3 && nbnode[n] == NO_NB )
            n++;
        node1 = nbnode[n];
        n++;
        while ( n < 3 && nbnode[n] == NO_NB )
            n++;
        node2 = nbnode[n];
        n++;
        while ( n < 3 && nbnode[n] == NO_NB )
            n++;
        if (n<3) node3 = nbnode[n];

        //Get Boundary mobility
        mob[0] = GetBoundaryMobility( node,node1 );
        mob[1] = GetBoundaryMobility( node,node2 );
        if ( node3!=NO_NB )
            mob[2] = GetBoundaryMobility( node,node3 );

        //Get the length of the vectors node-node1 and node-node2
        //And of course get the vetors itself (turned by 90deg)
        ElleNodePosition( node, & nodexy );
        ElleNodePlotXY( node1, & old1xy, &nodexy );
        ElleNodePlotXY( node2, & old2xy, &nodexy );
        if ( node3!=NO_NB )
        ElleNodePlotXY( node3, & old3xy, &nodexy );
       //calculate vectors
        vec1.y = -1*(old1xy.x-nodexy.x);
        vec1.x = old1xy.y-nodexy.y;
        //here we get the length of the vector
        l[0] = l[1] = l[2] = 0.0;
        l[0] = GetVectorLength( vec1 );
        //calculate the second vector
        vec2.y = -1*(old2xy.x - nodexy.x);
        vec2.x = old2xy.y-nodexy.y;
        //get length of vector
        l[1] = GetVectorLength( vec2 );
        if ( node3!=NO_NB ) {
            vec3.y = -1*(old3xy.x-nodexy.x);
            vec3.x = old3xy.y-nodexy.y;
            l[2] = GetVectorLength( vec3 );
        }
       //If the energies are equal, pvec will be the 0 and everything will crash. So we do not allow this.
        //Physically this should be correct too since if pvec is 0, no movement should take place.
        //pvec equals the force
        F.x = pvec.x;
        F.y = pvec.y;
        len=GetVectorLength(F);
        F.x=F.x/len;
        F.y=F.y/len;
        cosalpha[0] = DEGCos( vec1, F );
        if ( cosalpha[0] < 0 )
            cosalpha[0] *= -1;
        cosalpha[1] = DEGCos( vec2, F );
        if ( cosalpha[1] < 0 )
            cosalpha[1] *= -1;
        if ( node3!=NO_NB ) {
            cosalpha[2] = DEGCos( vec3, F );
            if ( cosalpha[2] < 0 )
                cosalpha[2] *= -1;
        }
        //in case of isotropic crystals this is sufficient (because the mobility will be the same on all
        //segments)
        //V0.x = ( 2 * F.x * mob1 )/ ( ( l1 * pow( cosalpha1, 2 ) ) + ( l2 * pow( cosalpha2, 2 ) ) + ( l3 * pow( cosalpha3, 2 ) ) );
        //V0.y = ( 2 * F.y * mob1 )/ ( ( l1 * pow( cosalpha1, 2 ) ) + ( l2 * pow( cosalpha2, 2 ) ) + ( l3 * pow( cosalpha3, 2 ) ) );
        //in case of anisotropic crystals this is needed because it takes the mobilities of both segments
        //into account

        // LE boundary lengths in Elle units,  mobilities in m2s-1
        double denom = 0.0;
        for (i=0; i<2; i++) 
            denom += ((l[i]*ElleUnitLength()*pow( cosalpha[i], 2)) / mob[i]) ;
        if (node3 != NO_NB) 
            denom += ((l[2]*ElleUnitLength()*pow( cosalpha[2], 2)) / mob[2]) ;
        if (denom) {
            V0.x = ( 2 * F.x*len ) / denom;
            V0.y = ( 2 * F.y*len ) / denom;

            V0.x /= ElleUnitLength();
            V0.y /= ElleUnitLength();
            //Distance (D) of movement =velocity *ElleTimestep()
            flength = GetVectorLength( V0 );
            dt = ElleTimestep() * ElleSpeedup();
            move = 1;
            if ( flength * ElleTimestep() * ElleSpeedup() >  ElleSwitchdistance() )
            {
                if (ElleSpeedup() > 1.0) 
                {
                    ElleSetSpeedup(1.0);
                    sprintf( logbuf, "Resetting speedup to %e! ", ElleSpeedup() );
                    Log( 0, logbuf );
                    dt = ElleTimestep();
                }
                if ( flength * ElleTimestep() >  ElleSwitchdistance() )
                {
                    double maxtstep = ElleSwitchdistance()/flength;
                    sprintf( logbuf, "Reducing timestep for node %d from %e to %e! ", node, ElleTimestep(),maxtstep );
                    Log( 0, logbuf );
                    write_timestep_log(logbuf);
                    dt = maxtstep;
                    ElleSetTimestep( maxtstep );
                    move = TIME_REDUCE;
                }
            }
            movedir->x = V0.x * dt;
            movedir->y = V0.y * dt;
        }
        else { // no movement
            V0.x = V0.y = 0.0;
            move = 0;
        }
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
    int mineral, rgn[3], n, same=0;
    double mob2=0;

    //we need to know which flynn it is and get the boundary mobility for the segments.
    ElleNeighbourRegion( node, nb, &rgn[0] );
    ElleNeighbourRegion( nb, node, &rgn[1] );
            
    ElleGetFlynnMineral( rgn[0], & mineral );
    mobility = GetMineralAttribute( mineral, GB_MOBILITY);
    ElleGetFlynnMineral( rgn[1], & mineral );
    mob2 = GetMineralAttribute( mineral, GB_MOBILITY);
    if (mob2<mobility) mobility=mob2;

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
    //I suspect that the temperature is in celsius, not in Kelvin?????? If so, we would have to add 273.15 ....
    T = ElleTemperature()+273.15;
    Q = 53.2e3; // for quartz
     ElleNodeSameMineral(node, &same, &mineral); // only allow GBM between same mineral type
    //  IF !SAME - GET OUT???
    Q = GetMineralAttribute(mineral,GB_ACTIVATION_ENERGY);

    //mobil=mobility * exp( -(H ) / ( B * T ) );
    // or
    mobil=mobility * exp( -(Q ) / ( R * T ) );
    //return 1e-6;
    return mobil;
}

int write_timestep_log(char *msg)
{
    std::string fname = "timestep.log";
    std::ofstream outf(fname.c_str(),std::ios::out|std::ios::app);
    if (outf) {
        outf << msg << std::endl;
    }
    else {
        OnError("elle_gbmunodes (write_timestep_log)",WRITE_ERR);
    }

    outf.close();
    return 0;
}
