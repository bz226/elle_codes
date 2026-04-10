#include "jkbmelt.elle.h"


/*!Initialisation function*/
int InitMelt()
{
	int err = 0;
	int max;
	char * infile, logbuf[4096];
	extern UserData CurrData;
	long tmp;
	tmp = currenttime();
#ifdef DEBUG
	tmp = 1028731772;
#endif
	cout << "seed= " << tmp << endl;
	srand( tmp );
	ElleReinit();
	ElleSetRunFunction( Melt );
	infile = ElleFile();
	if ( strlen( infile ) > 0 )
		if ( err = ElleReadData( infile ) )
			OnError( infile, err );
	ElleUserData( CurrData );
	if ( CurrData[EquilMeltFrac] < 0 )
		// calculate global melt fraction for input file
		CurrData[EquilMeltFrac] = CalcAreaLiq( MeltId );
	sprintf( logbuf, "areaequil: %lf\n", CurrData[EquilMeltFrac] );
	//Log( 1, logbuf );
	oldfudge=CurrData[Magic];
}
/*!Main function running through for a given number of stages*/
int Melt()
{
	stats statistic;
	int i, j, k, same,flynn,rgn[3],te,am;
	int interval = 0, st_interval = 0, err = 0, max;
	char fname[FILENAME_MAX], logbuf[4096];
	FILE * fp;
	Coords incr,xy;
	std::vector < int > ran;
	double old_speedup, new_speedup,cdmagic;
	double arealiq = 0;
	extern UserData CurrData;
	double upper = 0.005,test0,test1; // upper bound melt variation
	double lower = 0.001; // lower bound melt variation
	int *farr,num,v,test,newindex,loop,t;
	double length;
	double val;
	double val1,val2,val0;
	bool alterfudge,done=false;
	Coords xy1, xy2;
	/***************************************************
	 * Init melt stats
	 */
	statistic.initmeltstats();
	/***************************************************/
	if ( ElleCount() == 0 )
		ElleAddDoubles();
	if ( ElleDisplay() )
		EllePlotRegions( ElleCount() );
	incroffset = ElleSwitchdistance() * 0.01;
	ElleCheckFiles();
	sprintf(logbuf,"This version is a special version, it lets only nodes move that belong to at least one  solid flynn");
	//Log(0,logbuf);
	max = ElleMaxNodes();
	for ( k = 0; k < max; k++ )
		if ( ElleNodeIsActive( k ) )
			if ( ElleNodeIsDouble( k ) )
				ElleCheckDoubleJ( k );
			else if ( ElleNodeIsTriple( k ) )
				ElleCheckTripleJ( k );
	sprintf( logbuf, "MinNodeSep:%lf SwitchLength:%lf\n", ElleminNodeSep(), ElleSwitchLength() );
	//Log( 2, logbuf );
	//setting the fudge factor to a negative value will keep it fixed at that value
	//This gives problems when restarting the simulation, the fudge factor is positive then and will
	//be altered
	if(CurrData[Magic]<0)
	{
		alterfudge=false;
		//make fudge positive
		cdmagic=CurrData[Magic]*-1;
		sprintf( logbuf, "Fudge factor is negative, I will not touch it!\n");
		//Log( 0, logbuf );
	}
	else
	{
		cdmagic=CurrData[Magic];
		alterfudge=true;
	}
    am=0;
	//do all of this for as many stages as needed
	for ( i = 0; i < EllemaxStages(); i++ )
	{
		//calculate the area of all melt pockets
		arealiq = CalcAreaLiq( MeltId );

		//adjust the desired melt area by adding the user defined increment
		CurrData[EquilMeltFrac] += CurrData[IncMeltfraction];
		/****************************************************
		       * but don't let it drop below 0.7
		       */
		//if(CurrData[EquilMeltFrac]<0.7)
		// CurrData[EquilMeltFrac]=0.7;
		/****************************************************/
		//we only adjust the fudge factor if it is positive. If it is negative, it will not be altered
		if(alterfudge)
		{
			//if we have decreasing melt or increasing melt we have to keep the fudge factor stable
			//so calculate the disequilibirum
			test0=(arealiq - CurrData[EquilMeltFrac] ) / CurrData[EquilMeltFrac];
			test1=fabs(test0);
			//if the difference between the actual melt area and the user defined one (represented by the
			//disequilibrium) is bigger than upper
			if ( test1 > upper ) //.005
			{
				//make it fit later in Get SurfaceEnergy by increasing the surface energy
				//but only if it is not too small (or big). Depending on the starting value
				//of the fudge factor, cdmagic can reach very high/low values. We do not allow that
				//here
				if(cdmagic<10*CurrData[Magic])
					cdmagic = cdmagic * 1.0075;
				//and log a warning
				sprintf( logbuf, "upper: stages:%d arealiq:%lf %%diseq:%lf fudge:%lf equi:%lf\n", i, arealiq,100 * ( fabs( arealiq - CurrData[EquilMeltFrac] ) / CurrData[EquilMeltFrac] ), cdmagic, CurrData[EquilMeltFrac] ); //
				//Log( 1, logbuf );
			}
			//if the difference between the actual melt area and the user defined one is smaller than lower, make it fit
			else if ( test1 < lower ){
				//make it fit by decreasing the surface energy
				//cdmagic = cdmagic / 1.0025;

				//Changed so that it decreases as fast as it increases
				if(cdmagic>CurrData[Magic]/10)
					cdmagic = cdmagic / 1.0075;
				sprintf( logbuf, "lower: stages:%d arealiq:%lf %%diseq:%lf fudge:%lf equi:%lf\n", i, arealiq,100 * ( fabs( arealiq - CurrData[EquilMeltFrac] ) / CurrData[EquilMeltFrac] ), cdmagic, CurrData[EquilMeltFrac] ); //
				//Log( 1, logbuf );
			}
		}
		sprintf( logbuf, "stages:%d arealiq:%lf %%diseq:%lf fudge:%lf equi:%lf\n", i, arealiq,100 * ( fabs( arealiq - CurrData[EquilMeltFrac] ) / CurrData[EquilMeltFrac] ), cdmagic, CurrData[EquilMeltFrac] ); //
		//Log( 0, logbuf );
		//Go through the list of nodes for this stage
		max = ElleMaxNodes();
		//to prevent always using the same sequence of nodes, we randomly shuffle them
		ran.clear();
		for ( j = 0; j < max; j++ )
			if ( ElleNodeIsActive( j ) )
				ran.push_back( j );
		std::random_shuffle( ran.begin(), ran.end() );
		//here we actually go through the list
		for ( j = 0; j < ran.size(); j++ )
		{
			if ( ElleNodeIsActive( ran.at(j)) )
			{
				//Get the flynn the node belongs to
				//is it really always in rgn[0]?
				ElleRegions( ran.at(j), rgn );

				/***************************************
				             /* if(rgn[0]!=NO_NB)
					ElleGetFlynnRealAttribute( rgn[0], &val0, F_ATTRIB_A );
				else
					val0=1;
				if(rgn[1]!=NO_NB)
					ElleGetFlynnRealAttribute( rgn[1], &val1, F_ATTRIB_A );
				else
					val1=1;
				if(ElleNodeIsTriple(ran.at(j)) && rgn[2]!=NO_NB)
					ElleGetFlynnRealAttribute( rgn[2], &val2, F_ATTRIB_A );
				else
					val2=1;*/
				/*******************************************************************/

				/*******************************************************************
				*this lets only a specific grain move, all other grains are fixed. If other grains share a
				            *common boundary, that boundary can move. This is only for the age relationship of a 
				            *growing crystal.
				******************************************************************/                
				/*if(rgn[0]==8)
					val0=0;
				else
					val0=1;
				if(rgn[1]==8)
					val1=0;
				else
					val1=1;
				if(ElleNodeIsTriple(ran.at(j)) && rgn[2]==8)
					val2=0;
				else
					val2=1;*/
				/*******************************************************************/
				/*******************************************************************
				 * Take this out for regular melt stuff!
				*******************************************************************/
				/*if(val0==0 || val1==0 || val2==0)
				{*/
				/*******************************************************************/

				/******************************************************************/
				/*This moves a single, predefined node in a predefined direction (here half the switchdistance upward),
				 * all other nodes move according to their energy values.
				 *******************************************************************/
                 /*te=ran.at(j);
                 
				if(te==1326 && am<3)
				{
                    am+=1;
					//get node position
					ElleNodePosition( te, & xy );
					ElleSetPrevPosition(te,&xy);
					//move node upward
					//sprintf(logbuf,"%lf %lf\n",xy.x,xy.y);
					//Log(0,logbuf);
					xy.y=xy.y+ElleSwitchdistance();
					//sprintf(logbuf,"%lf %lf\n",xy.x,xy.y);
					//Log(0,logbuf);
					ElleSetPosition( te, & xy );
					done=true;
				}
				else
				{*/
					//see if the node needs to be moved
					for(int q=0;q<3;q++)
						if(rgn[q]!=NO_NB)
						{
							Melt_MoveNode4Pos( rgn[q],ran.at(j), & incr, & same, arealiq, CurrData[Mobility], cdmagic, CurrData[EquilMeltFrac],CurrData[XlXlEnergy], CurrData[LiqXlEnergy], CurrData[LiqLiqEnergy] );
							//check if there are topology problems after moving the node
							ElleCrossingsCheck( ran.at(j), & incr );
							break;
						}
					/******************************************************************
					*This belongs to the else from if(ran.at(j)==XXX).
					 */                    
				//}
				/******************************************************************/


				/******************************************************************
				 * This belongs to the if(val0==0 || etc.) for irregular melt stuff, see above
				*******************************************************************/
				//}
				/*******************************************************************/
			}
		}
		for ( j = 0; j < max; j++ )
		{
			if ( ElleNodeIsActive( j ) )
			{
				if ( ElleNodeIsDouble( j ) )
					ElleCheckDoubleJ( j );
				else if ( ElleNodeIsTriple( j ) )
					ElleCheckTripleJ( j );
			}
		}
		//update the screen
		ElleUpdate();
		/***************************************************
		* Update melt stats
		*/
		statistic.SaveMeltStats(F_ATTRIB_A,i);
		/***************************************************/
	}
}

/*!Get the energies of the 4 trial positions and return the new position */
int Melt_MoveNode4Pos( int flynn,int n, Coords * movedir, int * same, double arealiq, double mobfac, double fudge, double areaequil,
                       double energyxlxl, double energyliqxl, double energyliqliq )
{
	register int i, l;
	Coords xy, newxy, incr, prev, new_incr,oldxy;
	ERegion rgn[3];
	double new_area[3], old_area[3];
	double old_density[3], new_density[3],ff=fudge,oldarea;
	char logbuf[4096];
	double e1,e2,e3,e4,switchd = ElleSwitchdistance() /100,dblarealiq,anisotropy=0;
	//calculate the area of the liquid before any node movement
	oldarealiq = CalcAreaLiq( MeltId );
	//get node position
	ElleNodePosition( n, & xy );
	//and its previous position
	ElleNodePrevPosition( n, & prev );
	oldxy=newxy = xy;
	//change x-coordinates of the node for the first trial position
	newxy.x = oldxy.x + switchd;
	newxy.y = oldxy.y;
	ElleSetPosition( n, & newxy );
	//calculate the new total melt area
	dblarealiq = CalcAreaLiq( MeltId );

	//get the energy of the node at the new position
	GetSurfaceNodeEnergy( n, MeltId, & newxy, & e1, dblarealiq, fudge, areaequil, energyxlxl,energyliqxl, energyliqliq );
	//do that again for the other three positions
	newxy.x = oldxy.x - switchd;

	ElleSetPosition( n, & newxy );
	dblarealiq = CalcAreaLiq( MeltId );

	GetSurfaceNodeEnergy( n, MeltId, & newxy, & e2, dblarealiq, fudge, areaequil, energyxlxl,energyliqxl, energyliqliq );
	newxy.x = oldxy.x;
	newxy.y = oldxy.y + switchd;

	ElleSetPosition( n, & newxy );
	dblarealiq = CalcAreaLiq( MeltId );

	GetSurfaceNodeEnergy( n, MeltId, & newxy, & e3, dblarealiq, fudge, areaequil, energyxlxl,energyliqxl, energyliqliq );
	newxy.y = oldxy.y - switchd;
	ElleSetPosition( n, & newxy );
	dblarealiq = CalcAreaLiq( MeltId );

	GetSurfaceNodeEnergy( n, MeltId, & newxy, & e4, dblarealiq, fudge, areaequil, energyxlxl,energyliqxl, energyliqliq );
	newxy.x=oldxy.x;
	newxy.y=oldxy.y;
	ElleSetPosition( n, & oldxy );
	ElleSetPrevPosition( n, & prev );
	//calculate the new node position from the 4 trial energies
	GetNewPos(flynn,n,e1,e2,e3,e4,&newxy,switchd);
	//set the node position to the new position
	ElleSetPosition( n, & newxy );
	//calculate the direction and distance the node has moved (needed for the crossings check)
	movedir->x=newxy.x-oldxy.x;
	movedir->y=newxy.y-oldxy.y;
	//if the flynn has a dislocation density, adjust it here
	if ( ElleFlynnAttributeActive( DISLOCDEN ) )
		for ( l = 0; l < 3; l++ )
			if ( rgn[l] != NO_NB )
			{
				new_area[l] = fabs( ( double )ElleRegionArea( rgn[l] ) );
				if ( new_area[l] > old_area[l] )
				{
					new_density[l] = ( old_area[l] / new_area[l] ) * old_density[l];
					//printf("area ratio= %e\n",old_area[l]/new_area[l]);
					ElleSetFlynnRealAttribute( rgn[l], new_density[l], DISLOCDEN );
					//^****************************EEEEEEEEEKKKKKKK!!!!!!!!!
				}
			}
	ElleSetPrevPosition( n, & oldxy );
}

/*!Get the energies of the 3 trial positions and return the new position */
int Melt_MoveNode3Pos( int flynn,int n, Coords * movedir, int * same, double arealiq, double mobfac, double fudge, double areaequil,
                       double energyxlxl, double energyliqxl, double energyliqliq )
{
	register int i, l;
	Coords xy, newxy, incr, prev, new_incr,oldxy,xyz[3];
	ERegion rgn[3];
	double new_area[3], old_area[3];
	double old_density[3], new_density[3];
	char logbuf[4096];
	double E[3],switchd = ElleSwitchdistance() /100,dblarealiq;
	bool go=true;
	double anisotropy=0;
	//get node position
	ElleNodePosition( n, & xy );
	//and its previous position
	ElleNodePrevPosition( n, & prev );
	oldxy= xy;
	//get vector from original point to a new point that is switchd away
	newxy.x = switchd;
	newxy.y = switchd;
	//do the following until all test's in GetPVector are passed
	while(go)
	{
		go=false;
		//rotate vector by a 31 degree
		newxy.x=(0.857*newxy.x)+(0.515*newxy.y);
		newxy.y=(-0.515*newxy.x)+(0.857*newxy.y);
		//do the following three times for the three trial positions
		for(l=0;l<3;l++)
		{
			//rotate vector by a 120 degrees
			newxy.x=(0.866*newxy.y)+(-0.5*newxy.x);
			newxy.y=(-0.5*newxy.y)-(0.866*newxy.x);
			//caclculate a trial position
			xyz[l].x=oldxy.x+newxy.x;
			xyz[l].y=oldxy.y+newxy.y;
			//set new trial position
			ElleSetPosition( n, & xyz[l] );
			//calculate the new total melt area
			dblarealiq = CalcAreaLiq( MeltId );
			//get the energy of the node at the new position
			anisotropy+=GetSurfaceNodeEnergy( n, MeltId,  &xyz[l],  &E[l], dblarealiq, fudge, areaequil, energyxlxl,energyliqxl, energyliqliq );
		}
		ElleSetPosition( n, & oldxy );
		ElleSetPrevPosition( n, & prev );
		//calculate the new node position from the 3 trial energies
		if(GetMoveDir(flynn,n,xyz,E,movedir)==-50)
		{
			go=true;
			sprintf(logbuf,"Had to get new trial position for GetNewPos");
			//Log(0,logbuf);
		}
	}
	//calculate the new position
	newxy.x=oldxy.x+movedir->x;
	newxy.y=oldxy.y+movedir->y;

	//set the node position to the new position
	ElleSetPosition( n, & newxy );

	//if the flynn has a dislocation density, adjust it here
	if ( ElleFlynnAttributeActive( DISLOCDEN ) )
		for ( l = 0; l < 3; l++ )
			if ( rgn[l] != NO_NB )
			{
				new_area[l] = fabs( ( double )ElleRegionArea( rgn[l] ) );
				if ( new_area[l] > old_area[l] )
				{
					new_density[l] = ( old_area[l] / new_area[l] ) * old_density[l];
					//printf("area ratio= %e\n",old_area[l]/new_area[l]);
					ElleSetFlynnRealAttribute( rgn[l], new_density[l], DISLOCDEN );
					//^****************************EEEEEEEEEKKKKKKK!!!!!!!!!
				}
			}
	ElleSetPrevPosition( n, & oldxy );
}

/*!Currently not used*/
int GetBodyNodeEnergy( int n, double * total_energy )
{
	int i, nb[3];
	ERegion rgn[3];
	double sum_energy;
	double area[3], energy[3];
	double density[3];
	UserData udata;
	ElleRegions( n, rgn );

	sum_energy = 0;
	//do this for all surrounding flynns
	for ( i = 0; i < 3; i++ )
	{
		//if there are only two surrounding flynns, one value of rgn is NO_NB (NO_NeighBour)
		if ( rgn[i] != NO_NB )
		{
			ElleGetFlynnRealAttribute( rgn[i], & density[i], DISLOCDEN );
			area[i] = fabs( ElleRegionArea( rgn[i] ) ) * ( lengthscale * lengthscale );
			energy[i] = energyofdislocations * area[i] * density[i] * dislocationdensityscaling;
			sum_energy += energy[i];
			//printf("area=%le\tdensity=%le\tenergy=%le\n",area[i],density[i],energy[i]);
		}
	}
	* total_energy = sum_energy;
	return ( 0 );
}

/*!Currently not really used, the cls_factor (aka mobfac) is always 1*/
int GetCSLFactor( Coords_3D xyz[3], float * mobfac )
{
	double misorientation;
	double tmp;
	double dotprod;

	dotprod = ( xyz[0].x * xyz[1].x ) + ( xyz[0].y * xyz[1].y ) + ( xyz[0].z * xyz[1].z );

	if ( fabs( dotprod - 1.0 ) < 0.0001 )
		misorientation = 0.0;
	else
		misorientation = fabs( acos( dotprod ) * RTOD );

	if ( misorientation > 90.0 )
		misorientation = 180 - misorientation;

	tmp = 1.0 / ( ( misorientation / 90.0 ) + 0.1 );

	tmp = ( 10.0 - tmp ) / 10.0;

	//*mobfac=(float)tmp;  // gbm version (supresses sub-gbm)
	//*mobfac=1.0-tmp; // sub gbm version (supresses gbm)


	// CSL function produces 0% at 0 degrees, 95% at 10 degrees
	// and 99% at 20 degrees c axis misorientation

	* mobfac = 1.0; // normal mode, no CSL
}

#if XY
/*!Calculate direction cosines from euler angles.*/
int ElleGetFlynnEulerCAxis( int flynn_no, Coords_3D * dircos )
{
	double alpha, beta, gamma;

	ElleGetFlynnEuler3( flynn_no, & alpha, & beta, & gamma );

	alpha = alpha * DTOR;
	beta = beta * DTOR;

	dircos->x = sin( beta ) * cos( alpha );
	dircos->y = -sin( beta ) * sin( alpha );
	dircos->z = cos( beta );

	//converts Euler angles into direction cosines of c-axis
}
#endif

/*!This is a bit weired since it calculates the surface energy for a segment, not actually for the node (which is correct).*/
double GetSurfaceNodeEnergy( int n, int meltmineral, Coords * xy, double * energy, double arealiq, double fudge, double areaequil,
                             double energyxlxl, double energyliqxl, double energyliqliq )
{
	int i, nb[3];
	Coords xynb;
	double cosalpha, alpha, E;
	double len;
	int G0, G1;
	ERegion rgn[3], rgn2, rgn3;
	Coords_3D xyz[3];
	float csl_factor;
	double val2, val3;
	double energyofsurface,eg1=1,eg2=1,anisotropy=0;
	bool rg2=false,rg3=false;

	for ( i = 0; i < 3; i++ )
		xyz[i].x = xyz[i].y = xyz[i].z = 0;

	ElleRegions( n, rgn );
	ElleNeighbourNodes( n, nb );
	E = 0.0;

	for ( i = 0; i < 3; i++ )
	{
		if ( nb[i] != NO_NB )
		{
			ElleRelPosition( xy, nb[i], & xynb, & len );
			ElleNeighbourRegion( nb[i], n, & rgn2 ); //get one neighbouring flynn
			ElleNeighbourRegion( n, nb[i], & rgn3 ); //get other neighbouring flynn

			//get code of one neighbouring flynn (melt or solid)
			if(rgn2!=NO_NB)
				ElleGetFlynnRealAttribute( rgn2, & val2, meltmineral );
			if(rgn3!=NO_NB)
				//get code of other neighbouring flynn (melt or solid)
				ElleGetFlynnRealAttribute( rgn3, & val3, meltmineral );

			energyofsurface = energyliqxl; // interface energy of xl-melt

			//if they are equal, both flynns are liquid or both are solid
			if ( val2 == val3 )
			{
				if ( val2 == 0 )
					//both are solid (xl-xl boundary)
					energyofsurface = energyxlxl;
				else
					//both are liquid (melt-melt boundary)
					energyofsurface = energyliqliq;
			}

			//The cls_factor is always 1
			// with the rg2 and rg3 values, the flynns do not have to have a default anymore.
			// Either they have an orientation or they don't, if they don't fine too.*/
			if (rgn2!=NO_NB && rgn3!=NO_NB && ElleFlynnHasAttribute( rgn2, CAXIS ) && ElleFlynnHasAttribute( rgn3, CAXIS ) )
			{
				//I am not sure, but I think this does not work, the values that are
				//returned are very strange!!!!
				ElleGetFlynnCAxis( rgn3, & xyz[0] );
				ElleGetFlynnCAxis( rgn2, & xyz[1] );
				rg3=rg2=true;
				GetCSLFactor( xyz, & csl_factor );
			}
			else if (rgn2!=NO_NB && ElleFlynnHasAttribute( rgn2, CAXIS ) )
			{
				rg2=true;
				ElleGetFlynnCAxis( rgn2, & xyz[0] );
				csl_factor = 1.0;
			}
			else if (rgn3!=NO_NB &&ElleFlynnHasAttribute( rgn3, CAXIS ) )
			{
				rg3=true;
				ElleGetFlynnCAxis( rgn3, & xyz[0] );
				csl_factor = 1.0;
			}
			else if ( rgn2!=NO_NB && rgn3!=NO_NB && ElleFlynnHasAttribute( rgn2, EULER_3 ) && ElleFlynnHasAttribute( rgn3, EULER_3 ) )
			{
				rg3=rg2=true;
				ElleGetFlynnEulerCAxis( rgn3, & xyz[0] );
				ElleGetFlynnEulerCAxis( rgn2, & xyz[1] );
				GetCSLFactor( xyz, & csl_factor );
			}
			else if ( rgn2!=NO_NB && ElleFlynnHasAttribute( rgn2, EULER_3 ) )
			{
				rg2=true;
				ElleGetFlynnEulerCAxis( rgn2, & xyz[0] );
				csl_factor = 1.0;
			}
			else if ( rgn3!=NO_NB && ElleFlynnHasAttribute( rgn3, EULER_3 ) )
			{
				rg3=true;
				ElleGetFlynnEulerCAxis( rgn3, & xyz[0] );
				csl_factor = 1.0;
			}
			else
				csl_factor = 1.0;
			csl_factor = 1.0; //  <**************************EEEEEEEEEEEEEEKKKKKK

			if ( len == 0.0 )
				len = 1.0;
			//only do this if at least one of them actually has values and also do it if both of them
			//have values
			if((rg2 || rg3) || (rg2 && rg3))
			{
				cosalpha = ( ( double )xynb.x * xyz[0].x + xynb.y * xyz[0].y ) / len;

				// make sure cosalpha is in range -1,1 for acos
				if ( cosalpha > 1.0 )
					cosalpha = 1.0;
				if ( cosalpha < -1.0 )
					cosalpha = -1.0;
				if ( cosalpha >= 0.0 )
					alpha = acos( cosalpha );
				else
					alpha = acos( -cosalpha );
				//Get the angle energy
				eg1= GetAngleEnergy( alpha );
			}
			//if they both have values, do also this because there is something in xyz[1]
			if(rg3 && rg2)
			{
				cosalpha = ( ( double )xynb.x * xyz[1].x + xynb.y * xyz[1].y ) / len;

				// make sure cosalpha is in range -1,1 for acos
				if ( cosalpha > 1.0 )
					cosalpha = 1.0;
				if ( cosalpha < -1.0 )
					cosalpha = -1.0;
				if ( cosalpha >= 0.0 )
					alpha = acos( cosalpha );
				else
					alpha = acos( -cosalpha );
				//Get the angle energy
				eg2= GetAngleEnergy( alpha );
			}
			//set the values back to false for the next run
			rg2=rg3=false;
			//if there is no angle energy for one or more, it is still 1 so it does not count
			//in the calculations below
			E += len * eg1 * csl_factor * energyofsurface * lengthscale;
			E += len * eg2 * csl_factor * energyofsurface * lengthscale;
			anisotropy+=eg1+eg2;
			//Set the angle energies back to 1 for the next run
			eg1=eg2=1;
		}
	}
	//Area balancing fudge to ensure global melt fraction is roughly stable
	//If arealiq is smaller than areaequil, this calculation adds energy to the surface energy.
	//If arealiq is bigger than areaequil, this calculations adds just a little bit of energy to the surface energy.
	//So depending on the melt fraction, the surface energy returned here gets an extra boost or not.
	//This also means that the direction the node is eventually going to move towards will be in a direction so that the melt
	//fraction stays stable or is adjusted to the equilibrium melt fraction).
	//
	//
	//
	//We have to add the pressure-energy term here. If it is negative, this means that the melt pocket needs to get melt
	//so each node movement that increases the area of the melt pocket should be favored (or vice versa).
	//We could still use the fudge factor, but I think this is not very usefull anymore.


	/*!Used like this, there is a tricky balancing problem between the surface energies, the fudge-factor and the
	 * angle energies. I have not solved it yet, but it works if fudge =surface energy=1.
	 */
	E += fabs((arealiq - areaequil ) / areaequil ) * fudge;
	* energy = ( float )E;


	//test to see if can make things more edgy when there is an anisotropy
	if(anisotropy==0)
		return 1;
	else
		return anisotropy;
}

/*!Get's the energy for the angle angle from a lookup table in the elle-file*/
double GetAngleEnergy( double angle )
{
	int angdeg;
	double energy;
	//make sure the angle is an integer, not a double
	angdeg = ( int )( angle * RTOD + 0.5 );
	//Looks up the energy if orientation information for a flynn is stored
	energy = ElleEnergyLUTValue( angdeg );
	return ( energy );
}

/*!\brief Calculate area of a single melt pocket.*/
double CalcAreaLiq( int meltmineral )
{
	int i;
	int max = ElleMaxFlynns();
	double val;
	double area = 0;

	if ( ElleFlynnAttributeActive( meltmineral ) )
	{
		for ( i = 0; i < max; i++ )
		{
			if ( ElleFlynnIsActive( i ) )
			{
				ElleGetFlynnRealAttribute( i, & val, meltmineral );
				if ( val > 0 )
					area += ElleFlynnArea( i );
			}
		}
	}
	return ( area );
}

void ElleNodeSameAttrib( int node, int * same, double * sameval, int attrib_id )
{
	int i, j;
	double theval = 0;
	double dval[3];
	ERegion rgn[3];
	* same = 1;
	* sameval = 0;
	if ( ElleFlynnAttributeActive( attrib_id ) )
	{
		ElleRegions( node, rgn );

		for ( i = 0; i < 3; i++ )
		{
			dval[i] = 0;
			if ( rgn[i] != NO_NB )
			{
				ElleGetFlynnRealAttribute( rgn[i], & dval[i], attrib_id );
				theval = dval[i];
			}
		}

		for ( i = 0; i < 3; i++ )
			if ( rgn[i] != NO_NB && dval[i] != theval )
				* same = 0;

		if ( * same == 1 )
			* sameval = theval;
	}
}



/*!Returns the distance between two nodes in the current lengthscale.*/
double GetDistanceP1P2(int nextnode,int prevnode)
{
	Coords xy1,xy2,xy3;
	double len;
	//get the distance between the two
	ElleNodePosition( nextnode, & xy1 );
	ElleNodePosition( prevnode, & xy2 );
	xy3.x=xy1.x-xy2.x;
	xy3.y=xy2.y-xy2.y;
	//GetVectorLength is defined in meltmovenode.cc
	len=GetVectorLength(xy3);
	//if there is wrapping between the two points, we have to adjust the length here
	if(len>=0.5)
		//ChangeLength is defined in meltmovenode.cc
		len=ChangeLength(xy1,xy2);
	//return the distance to the next melt pocket corrected for the lengthscale of this model
	return len*lengthscale;
}
