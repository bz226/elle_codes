#include "meltmovenode.h"

/*!This function returns a vector in movedir. The node should move in the direction of movedir for the distance of
length of movedir. This should be used with ElleUpdatePosition. The function returns 1 if the node has to be moved, 0 if not.
If it should not be moved, the node either was inactive of the passed energies did not cause the node movement*/
int GetMoveDir( int flynn,int node, double e1, double e2, double e3, double e4, Coords * movedir,double distance )
{
	Coords pvec;
	//Get the vector of the direction with highest decrease of energy
	GetPVector( node, & pvec, e1, e2, e3, e4,distance );
	if ( pvec.x != 0 && pvec.y != 0 )
	{
		if ( ElleNodeIsTriple( node ) )
			return MoveTNode( flynn,node, pvec, movedir );
		else
			return MoveDNode( flynn,node, pvec, movedir );
	}
	else
		return 0;
}

/*!This function returns a vector in movedir. The node should move in the direction of movedir for the distance of
length of movedir. This should be used with ElleUpdatePosition. The function returns 1 if the node has to be moved, 0 if not.
If it should not be moved, the node either was inactive of the passed energies did not cause the node movement*/
int GetMoveDir( int flynn,int node,Coords xyz[3],double E[3], Coords * movedir )
{
    Coords pvec;
    //Get the vector of the direction with highest decrease of energy
    if(GetPVector( node, & pvec, xyz,E)==-1)
        return -50;
    else
    {
    if ( pvec.x != 0 && pvec.y != 0 )
    {
        if ( ElleNodeIsTriple( node ) )
            return MoveTNode( flynn,node, pvec, movedir );
        else
            return MoveDNode( flynn,node, pvec, movedir );
    }
    else
        return 0;
    }
}

/*!This function calculates a new position of the node node. It should be used together with ElleSetPosition.The function
returns 1 if the node has to be moved, 0 if not. If it should not be moved, the node either was inactive of the passed
energies did not cause the node movement*/
int GetNewPos( int flynn,int node, double e1, double e2, double e3, double e4, Coords * newpos,double distance )
{
	Coords npos, pvec;
	int t;
	//Get the vector of the direction with highest decrease of energy
	GetPVector( node, & pvec, e1, e2, e3, e4,distance );
	if ( pvec.x != 0 && pvec.y != 0 )
	{
		ElleNodePosition( node, & npos );
		if ( ElleNodeIsTriple( node ) )
		{
			t = MoveTNode(flynn, node, pvec, newpos );
			newpos->x += npos.x;
			newpos->y += npos.y;
		}
		else
		{
			t = MoveDNode( flynn,node, pvec, newpos );
			newpos->x += npos.x;
			newpos->y += npos.y;
		}
		return t;
	}
	else
		return 0;
}


/*!\brief This function calculates a new position of the node node from three trial positions.
This function calculates a new position of the node node from three trial positions. It should be used together with ElleSetPosition.The function
returns 1 if the node has to be moved, 0 if not and -50 if the tests in GetPVector failed. If that happens,
this function should be called again with slightly different trial positions.
If the node should not be moved, the node either was inactive or the passed
energies did not cause the node movement*/
int GetNewPos( int flynn,int node,Coords xyz[3],double E[3], Coords * newpos )
{
	Coords npos, pvec;
	int t;
	//Get the vector of the direction with highest decrease of energy
	if(GetPVector( node, & pvec, xyz,E)==-1)
		return -50;
	else
	{
		if ( pvec.x != 0 && pvec.y != 0 )
		{
			ElleNodePosition( node, & npos );
			if ( ElleNodeIsTriple( node ) )
			{
				t = MoveTNode(flynn, node, pvec, newpos );
				newpos->x += npos.x;
				newpos->y += npos.y;
			}
			else
			{
				t = MoveDNode( flynn,node, pvec, newpos );
				newpos->x += npos.x;
				newpos->y += npos.y;
			}
			return t;
		}
		else
			return 0;
	}
}


/*!\brief Move a double node
 
This function returns the direction and distance to move the node node in movedir. It returns 0 if movedir is 0 (the node
should not be moved this time) or 1 if the node should be moved. It also takes into account the timestep defined in the
elle-file (usually one year) and the speedup (usually one). The combination of timestep, speedup and switchdistance has to be
balanced. If the switchdistance is very small and the timestep is very large, the calculated distance for the node movement
can be bigger then the switchdistance. In that case, a warning message is logged, movedir is set to 0 and the function returns
0 (meaning the node should not be moved).*/
int MoveDNode(int flynn, int node, Coords pvec, Coords * movedir )
{
	double l1, l2, cosseg, cosalpha1, cosalpha2, mob1, mob2, flength,test1,test2,eul;
	int nbnode[3], node1, node2, err, n;
	Coords F, V0, newxy, nodexy, old1xy, old2xy, vec1, vec2;
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
		mob1 = GetBoundaryMobility( flynn );
		mob2 = GetBoundaryMobility( flynn );

		//Get the length of the vectors node-node1 and node-node2
		//And of course get the vetors itself (turned by 90deg, we need the vector normals)
		ElleNodePosition( node, & nodexy );
		ElleNodePosition( node1, & old1xy );
		ElleNodePosition( node2, & old2xy );
		//for debugging I calculate the cosine between the two segments
		vec1.x = old1xy.x - nodexy.x;
		vec1.y = old1xy.y - nodexy.y;
		vec2.x = old2xy.x - nodexy.x;
		vec2.y = old2xy.y - nodexy.y;
		cosseg = DEGCos( vec1, vec2 );
		//here the real stuff!
		vec1.y = old1xy.x - nodexy.x;
		vec1.x = nodexy.y - old1xy.y;
		//here we get the length of the vector
		l1 = GetVectorLength( vec1 );
		//and here we test for wrapping
		if ( l1 > 0.5 )
			l1 = ChangeLength( old1xy, nodexy );
		//calculate the second vector
		vec2.y = old2xy.x - nodexy.x;
		vec2.x = nodexy.y - old2xy.y;
		//get length of vector
		l2 = GetVectorLength( vec2 );
		//Test for wrapping
		if ( l2 > 0.5 )
			l2 = ChangeLength( old2xy, nodexy );
		//If the energies are equal, pvec will be the 0 and everything will crash. So we do not allow this.
		//Physically this should be correct too since if pvec is 0, no movement should take place.
		//pvec equals the force
		F.x = pvec.x;
		F.y = pvec.y;
		//not nice, but hey...
		//the normals of the segments have to point in the same direction as F
		//so the cosine of the vectors have to be between 0 and 1
		cosalpha1 = DEGCos( vec1, F );
		cosalpha2 = DEGCos( vec2, F );
		//which means if one of them is negative, we make it positive. Is that correct in all
		//cases?
		cosalpha1=fabs(cosalpha1);
		cosalpha2=fabs(cosalpha2);
		//in case of isotropic crystals this is sufficient (because the mobility will be the same on both
		//segments)
		//V0.x = ( 2 * F.x * mob1 ) / ( ( l1 * pow( cosalpha1, 2 ) ) + ( l2 * pow( cosalpha2, 2 ) ) );
		//V0.y = ( 2 * F.y * mob1 ) / ( ( l1 * pow( cosalpha1, 2 ) ) + ( l2 * pow( cosalpha2, 2 ) ) );
		//in case of anisotropic crystals this is needed because it takes the mobilities of both segments
		//into account
		//if the angle between F and the segments is 90 the node can freely move which results in too big
		//steps. Therefore, we forbid it when the cosine is smaller 0.01745 (which is 89DEG)
		/* if ( cosalpha1 < 0.01745 && cosalpha2 < 0.01745 ) { V0.x = 0; V0.y = 0; } else { */
		test1=( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 ) );
		if(test1==0)
		{
			sprintf(logbuf,"I am going to crash from movednodes now");
			//Log(0,logbuf);
		}
		V0.x = ( 2 * F.x ) / ( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 ) );
		V0.y = ( 2 * F.y ) / ( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 ) );
		/* } */
		V0.x*= ( ElleTimestep() )*1e-3*ElleSpeedup();
		V0.y*= ( ElleTimestep() )*1e-3*ElleSpeedup();
		flength = GetVectorLength( V0 );
		//This should never happen. If it does, the timesteps are too big (or the switchdistance is too small) and needs to be adjusted.
		//In that case, other modules probably will fail too.
		if ( flength> ElleSwitchdistance())
		{
			sprintf( logbuf, "Distance too big to move double, angles: %lf,%lf,%lf,Distance: %lf, Switchdistance: %lf\n",
			         acos( cosalpha1 ) * RTOD, acos( cosalpha2 ) * RTOD, acos( DEGCos( vec1, vec2 ) ) * RTOD, flength ,
			         ElleSwitchdistance() );
			//Log( 1, logbuf );
			n=0;
			while(flength> ElleSwitchdistance() )
			{
				V0.x/=1.1;
				V0.y/=1.1;
				flength = GetVectorLength( V0 );
				n++;
			}
			sprintf( logbuf, "I had to decrease the vector length for node %d %d times! ", node,n );
			//Log( 1, logbuf );
			sprintf( logbuf, "SwitchDistance or TimeStep needs adjustment to prevent this!\n" );
			//Log( 1, logbuf );
		}
		movedir->x = V0.x;
		movedir->y = V0.y;
		return 1;
	}
	return 0;
}


/*!\brief This will move a triple node.
 
*See MoveDNode for more explanantion*/
int MoveTNode( int flynn,int node, Coords pvec, Coords * movedir )
{
	double l1, l2, l3, cosalpha1, cosalpha2, cosalpha3, mob1, mob2, mob3, flength,test1;
	int nbnode[3], node1, node2, node3, err,n;
	Coords F, V0, newxy, nodexy, oldxy, vec1, vec2, vec3;
	char logbuf[4096];
	if ( ElleNodeIsActive( node ) )
	{
		//Get the neighbouring nodes
		if ( err = ElleNeighbourNodes( node, nbnode ) )
			OnError( "MoveNode", err );
		node1 = nbnode[0];
		node2 = nbnode[1];
		node3 = nbnode[2];
		//Get Boundary mobility
		mob1 = GetBoundaryMobility(flynn );
		mob2 = GetBoundaryMobility( flynn );
		mob3 = GetBoundaryMobility( flynn );
		//Get the length of the vectors node-node1 and node-node2
		//And of course get the vetors itself (turned by 90deg)
		ElleNodePosition( node, & nodexy );
		ElleNodePosition( node1, & oldxy );
		vec1.y = oldxy.x - nodexy.x;
		vec1.x = nodexy.y - oldxy.y;
		l1 = GetVectorLength( vec1 );
		if ( l1 > 0.5 )
			l1 = ChangeLength( oldxy, nodexy );
		ElleNodePosition( node2, & oldxy );
		vec2.y = oldxy.x - nodexy.x;
		vec2.x = nodexy.y - oldxy.y;
		l2 = GetVectorLength( vec2 );
		if ( l2 > 0.5 )
			l2 = ChangeLength( oldxy, nodexy );
		ElleNodePosition( node3, & oldxy );
		vec3.y = oldxy.x - nodexy.x;
		vec3.x = nodexy.y - oldxy.y;
		l3 = GetVectorLength( vec3 );
		if ( l3 > 0.5 )
			l3 = ChangeLength( oldxy, nodexy );
		//If the energies are equal, pvec will be the e0 and everything will crash. If pvec is 0, no movement shuold take place
		//anyway, therefore...
		F.x = pvec.x;
		F.y = pvec.y;
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
		test1=( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 )
		        + ( l3 * pow( cosalpha3, 2 ) / mob3 ) );
		if(test1==0)
		{
			sprintf(logbuf,"I am going to crash from movetnode now!");
			//Log(0,logbuf);
		}
		V0.x = ( 2 * F.x ) / ( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 )
		                       + ( l3 * pow( cosalpha3, 2 ) / mob3 ) );
		V0.y = ( 2 * F.y ) / ( ( ( l1 * pow( cosalpha1, 2 ) ) / mob1 ) + ( ( l2 * pow( cosalpha2, 2 ) ) / mob2 )
		                       + ( l3 * pow( cosalpha3, 2 ) / mob3 ) );
		//Distance (D) of movement =velocity *ElleTimestep()
		V0.x *= ( ElleTimestep() )*1e-3*ElleSpeedup();
		V0.y *= ( ElleTimestep() )*1e-3*ElleSpeedup();
		flength = GetVectorLength( V0 );
		if ( flength> ElleSwitchdistance() )
		{
			sprintf( logbuf, "Distance too big to move triple! angles: %lf,%lf,%lf Distance: %lf, Switchdistance: %lf\n",
			         acos( cosalpha1 ) * RTOD, acos( cosalpha2 ) * RTOD, acos( cosalpha3 ) * RTOD, flength,
			         ElleSwitchdistance() );
			//Log( 1, logbuf );
			n=0;
			while(flength > ElleSwitchdistance())
			{
				V0.x/=1.1;
				V0.y/=1.1;
				flength = GetVectorLength( V0 );
				n++;
			}
			sprintf( logbuf, "I had to decrease the vector length for node %d %d times! ", node,n );
			//Log( 1, logbuf );
			sprintf( logbuf, "SwitchDistance or TimeStep needs adjustment to prevent this!\n" );
			//Log( 1, logbuf );
		}
		movedir->x = V0.x;
		movedir->y = V0.y;
		return 1;
	}
	return 0;
}

/*!\brief Calculatest the cosine of the two vectors. */
double DEGCos( Coords vec1, Coords vec2 )
{
	return ( vec1.x * vec2.x + vec1.y * vec2.y ) / ( GetVectorLength( vec1 ) * GetVectorLength( vec2 ) );
}

/*!\brief Returns the length of the vector */
double GetVectorLength( Coords vec )
{
	return sqrt( ( vec.x * vec.x ) + ( vec.y * vec.y ) );
}


/*!\brief Calculates the vector pointing down the energy gradient
 
This calculation is explained in various talks from Elle-Meetings. The position of the nodes associated with the energies is VERY IMPORTANT!!!!!
e1=x+distance, e2=x-distance, e3=y+distance, e4=y-distance where x and y are the coordinates of the node to move. 
MAKE SURE THIS IS IN THE CORRECT ORDER!!!!*/
int GetPVector( int node, Coords * pvec, double E1, double E2, double E3, double E4,double distance )
{
	Coords p1, p2, gvector, prev;
	gvector.x = ( E1 - E2 ) / ( 2 * distance );
	gvector.y = ( E3 - E4 ) / ( 2 * distance );
	//But gvector points into the direction of max. increase while we want to have the direction
	//of max. decrease
	pvec->x = -gvector.x;
	pvec->y = -gvector.y;
	//for drawgradient function
	/*double list[10000] [3]; int n = 0; Coords grad; for ( grad.y = p1.y - 0.05; grad.y < p1.y + 0.05; grad.y += 0.001 )
	Act  for ( grad.x = p1.x - 0.05; grad.x < p1.x + 0.05; grad.x += 0.001 ) { list[n] [0] = grad.x; list[n] [1] = grad.y;
	if ( ElleNodeIsActive( node ) ) list[n] [2] = GetNodeEnergy( node, & grad );; n++; } if ( ElleNodeIsActive( node ) )
	DrawGrad( list, p1, * pvec );*/
	return 1;
}

/*!\brief Calculates the vector pointing down the energy gradient using only three positions.
 * xyz holds the three node positions used while E holds the corresponding energies. point xyz[0] belongs to E[0] 
 * and so on.*/
int GetPVector(int node, Coords * pvec,Coords xyz[3],double E[3])
{
	double alpha,beta,test1,test2;
	test1=(xyz[2].y-xyz[0].y-xyz[2].x+xyz[0].x);
	test2=(xyz[1].x-xyz[0].x);
	//if any of these is 0, we would get a division by zero. If that happens, we returned failed so we need new coordinates
	//for the trial positions.
	if(test1==0 || test2==0)
		return -1;
	else
	{
		beta=(E[2]-E[0]-(((E[1]-E[0])/(xyz[1].x-xyz[0].x))*(xyz[2].x-xyz[0].x)))*(xyz[1].x-xyz[0].x);
		beta/=((xyz[2].y-xyz[0].y)*(xyz[1].x-xyz[0].x))-((xyz[1].y-xyz[0].y)*(xyz[2].x-xyz[0].x));
		alpha=(E[1]-E[0]-beta*(xyz[1].y-xyz[0].y))/(xyz[1].x-xyz[0].x);
		pvec->x=-alpha;
		pvec->y=-beta;
		return 1;
	}
}
/*!\brief Changes the length of a segment in case there is wrapping
 
If there is wrapping between the two points, the vector length is adjusted here. We could stick that into
the GetVectorLength function right away?!*/
double ChangeLength( Coords p1, Coords p2 )
{
	double l;
	Coords v;
	if ( p1.x > p2.x )
	{
		if ( p1.x - p2.x >= 3 * ElleSwitchdistance() )
			p2.x += 1;
	}
	else
	{
		if ( p2.x - p1.x >= 3 * ElleSwitchdistance() )
			p1.x += 1;
	}
	if ( p1.y > p2.y )
	{
		if ( p1.y - p2.y >= 3 * ElleSwitchdistance() )
			p2.y += 1;
	}
	else
	{
		if ( p2.y - p1.y >= 3 * ElleSwitchdistance() )
			p1.y += 1;
	}
	v.x = p1.x - p2.x;
	v.y = p1.y - p2.y;
	l = GetVectorLength( v );
	return l;
}

/*!\brief Get the 'real' boundary mobility
 
General formula: mobility=mo*exp((-H/(B*T)). mo is a base mobility, taken from mineraldb, B(oltzmann-constant) is in J*K^-1
according to this: http://www2.mpie-duesseldorf.mpg.de/msu-web/msuweb_neu/deutsch/themenkatalog/3d-cellular-automaton/3d-cellular-automaton.htm
H (activation energy) for an aluminium polycrystall is 1.6 eV 1 eV = 1.6022e-19 Joule so H for aluminium should be 2.56353e-19
Joule, so it is 145379 J/mol for Olivine it is around 550 kJ/mol (Karato 1989: Grain Growth Kinetics in Olivine Aggregates,
Tectonophysics, 168, 255-273) nobody knows how much it is for e.g. Quartz or something, at least I have not found anything
anywhere. Gas constant 8.314472 in Jmol-1K-1. */
double GetBoundaryMobility( int flynn )
{
	double mobility = 0, mobil, arg2, arg1, arg, T, R = 8.314472, H = 145, B = 1.3806505e-23,type;
	int mineral;
    //In the old code: mobility is always user defined and the same for both the crystall and the melt!
    //we set that here to 1e-10 for melt or for -1 flynns (see task) although it should really be what the user set it to.
    //This does not work. Sometimes the flynn is -1 and than of course everything crashes!
    //There is a common mistake in here somewhere!
	if(flynn!=NO_NB && ElleFlynnHasAttribute(flynn,F_ATTRIB_A))
        ElleGetFlynnRealAttribute(flynn,&type,F_ATTRIB_A);
    else
        type=1; //AAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHH
	if(type==1.0)
		mobility=1e-10;
	else
	{
        if(ElleFlynnHasAttribute(flynn,MINERAL))
        {
		ElleGetFlynnMineral( flynn, & mineral );
		mobility = GetMineralAttribute( mineral, GB_MOBILITY );
        }
        else
        mobility=1e-10;
	}
	//I suspect that the temperature is in celsius, not in Kelvin?????? If so, we would have to add 273.15 ....
	T = ElleTemperature();
	//I can not use the Boltzman constant because the number to caclulate arg2 becomes too small for c++ to process
	mobil=mobility * exp( -H / ( R * T ) );
	return mobil;
}
