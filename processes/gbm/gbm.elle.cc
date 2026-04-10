#include "gbm.elle.h"
#include "lut.h"

using std::vector;

int GetBodyNodeEnergy(int n, double *total_energy);
double  GetAngleEnergy(double angle);
double GetOrientationEnergy(int rgn1, int rgn2, Coords *nbxy, double len);
int GetCSLFactor(Coords_3D xyz[3], double *factor);

//The nodes are moved along the energy gradient
//Written by Dr. J.K. Becker

/*!/brief Calculates the (Surface) Energy of a node
 
This really only calculates the surface energy of the node, nothing else. General equation is E=en(l1+l2+l3)*lengthscale
with E=energy, en=surface energy and l1/l2/l3 the length of the segments next to the node adjusted to the lengthscale */
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

    //Each node can belong to more than one flynn, each flynn can have a different SURFACE_ENERGY
    //We add them all up
    ElleRegions( node, rgn );
    n = 0;
    while ( n < 3 )
    {
        if ( rgn[n] != -1 )
        {
            ElleGetFlynnMineral( rgn[n], & mineral );
            energyofsurface = GetMineralAttribute( mineral, SURFACE_ENERGY );///2;
            //en += GetMineralAttribute( mineral, SURFACE_ENERGY );///2;

            if (ElleFlynnAttributeActive(EULER_3)||
                            ElleFlynnAttributeActive(CAXIS))
            {
                int nbrgn;
                Coords relxy;
                double len;
                ElleNeighbourRegion(nbnode[n], node, &nbrgn);
                // len is the same as l1, l2, l3 depending on n
                ElleRelPosition(xy,nbnode[n],&relxy,&len);
                energyofsurface *= GetOrientationEnergy(rgn[n],nbrgn,&relxy,len);
            }
            en += energyofsurface;
        }
        n++;
    }
    E = en * ( l1 + l2 ) * ElleUnitLength();
    if ( ElleNodeIsTriple( node ) )
        E +=  en*l3 * ElleUnitLength();
    if(ElleFlynnAttributeActive(DISLOCDEN)) {
        GetBodyNodeEnergy(node,&bodyenergy);
        E += bodyenergy;
    }
    return E;
}

int GetBodyNodeEnergy(int n, double *total_energy)
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

    energyofdislocations=(double)GetMineralAttribute(themin,DD_ENERGY);
                                                                               
    ElleRegions(n,rgn);
    sum_energy =0;
    for (i=0;i<3;i++) {
        if (rgn[i]!=NO_NB) {
            ElleGetFlynnRealAttribute(rgn[i],&density[i],DISLOCDEN);
            area[i] = fabs(ElleRegionArea (rgn[i]))*
                              (ElleUnitLength()*ElleUnitLength());
            energy[i] = energyofdislocations*area[i]*density[i]*
                              dislocationdensityscaling;

            sum_energy += energy[i];
            //printf("area=%le\tdensity=%le\tenergy=%le\n",area[i],density[i],energy[i]);
                                                                                
        }
    }
                                                                                
    *total_energy=sum_energy;
                                                                                
    return(0);
}

double GetOrientationEnergy(int rgn1, int rgn2, Coords *nbxy, double len)
{
    int i;
    int mineral;
    double csl_factor=1.0;
    double cosalpha = 0, alpha=0.0;
    double energy = 0, en=0;
    Coords_3D xyz[3];

    for (i=0;i<3;i++) xyz[i].x = xyz[i].y = xyz[i].z = 0;
    if (ElleFlynnAttributeActive(EULER_3))
    {
        ElleGetFlynnEulerCAxis(rgn1,&xyz[0]);
        ElleGetFlynnEulerCAxis(rgn2,&xyz[1]);
    }
    else if (ElleFlynnAttributeActive(CAXIS))
    {
        ElleGetFlynnCAxis(rgn1,&xyz[0]);
        ElleGetFlynnCAxis(rgn2,&xyz[1]);
    }
    GetCSLFactor(xyz,&csl_factor);
    if (len==0.0) len = 1.0;
    cosalpha = (nbxy->x * xyz[0].x + nbxy->y * xyz[0].y)/len;
        /*
         * make sure cosalpha is in range -1,1 for acos
         */
    if (cosalpha > 1.0) cosalpha = 1.0;
    if (cosalpha < -1.0) cosalpha = -1.0;
    if (cosalpha >= 0.0) alpha = acos(cosalpha);
    else                 alpha = acos(-cosalpha);
    if (ElleEnergyLUTSize()>0)  {
        en= GetAngleEnergy(alpha);
    }
    else {
        ElleGetFlynnMineral( rgn1, & mineral );
        en=GetMineralAttribute( mineral, SURFACE_ENERGY );
    }
    //energy = GetAngleEnergy(alpha)*csl_factor;
    energy += en;
    cosalpha = (nbxy->x * xyz[1].x + nbxy->y * xyz[1].y)/len;
        /*
         * make sure cosalpha is in range -1,1 for acos
         */
    if (cosalpha > 1.0) cosalpha = 1.0;
    if (cosalpha < -1.0) cosalpha = -1.0;
    if (cosalpha >= 0.0) alpha = acos(cosalpha);
    else                 alpha = acos(-cosalpha);
    if (ElleEnergyLUTSize()>0)  {
        en= GetAngleEnergy(alpha);
    }
    else {
        ElleGetFlynnMineral( rgn2, & mineral );
        en=GetMineralAttribute( mineral, SURFACE_ENERGY );
    }
    //energy += GetAngleEnergy(alpha)*csl_factor;
    energy += en;
    return(energy);
}

double GetAngleEnergy(double angle)
{
    int angdeg;
    double energy;
                                                                                
    angdeg = (int)(angle*RTOD + 0.5);
    energy = ElleEnergyLUTValue(angdeg);
    return(energy);
}
                                                                                
#if XY
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos);
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos)
{
    double alpha,beta,gamma;
                                                                                
    ElleGetFlynnEuler3(flynn_no,&alpha,&beta,&gamma);
                                                                                
    alpha=alpha*DTOR;
    beta=beta*DTOR;
                                                                                
    dircos->x=sin(beta)*cos(alpha);
    dircos->y=-sin(beta)*sin(alpha);
    dircos->z=cos(beta);
                                                                                
    //converts Euler angles into direction cosines of c-axis
}
#endif
 
int GetCSLFactor(Coords_3D xyz[3], double *factor)
{
    double misorientation;
    double tmp;
    double dotprod;
                                                                                
    dotprod=(xyz[0].x*xyz[1].x)+(xyz[0].y*xyz[1].y)+(xyz[0].z*xyz[1].z);
                                                                                
    if(fabs(dotprod-1.0) < 0.0001)
    misorientation=0.0;
    else
    misorientation = fabs(acos(dotprod)*RTOD);
                                                                                
    if(misorientation>90.0)
    misorientation=180-misorientation;
                                                                                
    tmp=1.0/((misorientation/90.0)+0.1);
                                                                                
    //tmp=pow(tmp,4.0);
                                                                                
    //tmp=(10000.0-tmp)/10000.0;
    tmp=pow(tmp,1.0);
                                                                                
    tmp=(10.0-tmp)/10.0;
    //printf("misorientation %le\tfactor %le\n",misorientation,tmp);
                                                                                
    *factor=tmp;  // gbm version (supresses sub-gbm)
    //*factor=1.0-tmp; // sub gbm version (supresses gbm)
                                                                                
                                                                                
    // CSL function produces 0% at 0 degrees, 95% at 10 degrees
    // and 99% at 20 degrees c axis misorientation
}

/*!/brief Initializes elle_gbm
 
Standard initialization of the grain growth module. */
int InitGrowth()
{
    int err = 0;
    char * infile;

    ElleReinit();
    ElleSetRunFunction( GBMGrowth );

    infile = ElleFile();
    if ( strlen( infile ) > 0 )
    {
        if ( err = ElleReadData( infile ) )
            OnError( infile, err );
        ElleAddDoubles();
    }
}

/* !/brief this is called to when it is time to move a node (and also to check if a node has to be moved) */
int GGMoveNode( int node, Coords * xy )
{
    double e0, e1, e2, e3, e4, switchd = ElleSwitchdistance()/100;
    Coords oldxy, newxy, prev;

    e0 = e1 = e2 = e3 = e4 = 0.0;
    ElleNodePosition( node, & oldxy );
    e0 = GetNodeEnergy( node, & oldxy );
    newxy.x = oldxy.x + switchd;
    newxy.y = oldxy.y;
    e1 = GetNodeEnergy( node, & newxy );
    e1 -= e0;
    newxy.x = oldxy.x - switchd;
    e2 = GetNodeEnergy( node, & newxy );
    e2 -= e0;
    newxy.x = oldxy.x;
    newxy.y = oldxy.y + switchd;
    e3 = GetNodeEnergy( node, & newxy );
    e3 -= e0;
    newxy.y = oldxy.y - switchd;
    e4 = GetNodeEnergy( node, & newxy );
    e4 -= e0;
    return GetMoveDir( node, e1, e2, e3, e4, xy ,switchd);
}

int write_data(int stage)
{
    FILE *f2,*f1;
    double area,arean,meangrowth[20],gmax[20],gmin[20],ma,m2;
    int side[20],sides,n,*ids,num,flynn,numberflynns=0,c;
    //~ for(n=0;n<20;n++)
    //~ {
    //~ side[n]=0;
    //~ meangrowth[n]=0;
    //~ gmax[n]=-10;
    //~ gmin[n]=10;
    //~ }

    //~ //write out number of grains and numbers of sides if stage is multiple of 1000
    //~ //F_ATTRIB_A has area at last test
    //~ if(fmod(stage,500)==0)
    //~ {
    //~ f2=fopen("grains-sides.csv","a");
    //~ f1=fopen("growth-sides.csv","a");
    //~ for(c=0;c<ElleMaxFlynns();c++)
    //~ {
    //~ sides=0;
    //~ if(ElleFlynnIsActive(c))
    //~ {
    //~ numberflynns++;
    //~ ElleFlynnNodes(c, &ids, &num);
    //~ for(n=0;n<num;n++)
    //~ {
    //~ if(ElleNodeIsTriple( ids[n] ))
    //~ sides++;
    //~ }
    //~ if(sides<20)
    //~ {
    //~ side[sides]++;
    //~ ElleGetFlynnRealAttribute(c, &area, F_ATTRIB_A);
    //~ arean=ElleFlynnArea(c);
    //~ ma=arean-area;
    //~ meangrowth[sides]+=ma;
    //~ if(ma>gmax[sides])
    //~ gmax[sides]=ma;
    //~ if(ma<gmin[sides])
    //~ gmin[sides]=ma;
    //~ ElleSetFlynnRealAttribute(c, arean, F_ATTRIB_A);
    //~ }
    //~ }
    //~ }
    //~ fprintf(f2,"%d,%d",stage,numberflynns);
    //~ fprintf(f1,"%d,%d",stage,numberflynns);
    //~ for(n=3;n<12;n++)
    //~ {
    //~ if(side[n]==0)
    //~ fprintf(f2,",0");
    //~ else
    //~ fprintf(f2,",%d",side[n]);
    //~ if(side[n]==0)
    //~ fprintf(f1,",0,0,0");
    //~ else
    //~ fprintf(f1,",%e,%e,%e",meangrowth[n]/side[n],gmax[n],gmin[n]);
    //~ }
    //~ for(n=3;n<12;n++)
    //~ if(side[n]==0)
    //~ fprintf(f2,",0");
    //~ else
    //~ fprintf(f2,",%e",side[n]/numberflynns);
    //~ fprintf(f2,"\n");
    //~ fclose(f2);
    //~ fprintf(f1,"\n");
    //~ fclose(f1);
    //~ }
    //~ if(fmod(stage,1000)==0)
    //~ {
    //~ f1=fopen("kreis-stat.csv","a");
    //~ fprintf(f1,"%d,%e,%d\n",stage,ElleFlynnArea(1),num);
    //~ fclose(f1);
    //~ }

    return 1;
}

/* !/brief Runs through all the nodes (in random order) and moves them also writes out some statistics, turn
this on or off using STATS*/
int GBMGrowth()
{
    bool surface_only = true;
    int i, j, max, move=0, err=0;
    int k, kmax, iter=0, itermax=5;
    int same=0, themin;
    double oldTimestep;
    Coords newxy;
    vector < int > ran;
    if ( ElleCount() == 0 )
        ElleAddDoubles();
    if ( ElleDisplay() )
        EllePlotRegions( ElleCount() );
    ElleCheckFiles();
    
    if (ElleFlynnAttributeActive(DISLOCDEN) ||
                ElleFlynnAttributeActive(EULER_3) ||
                        ElleFlynnAttributeActive(CAXIS))
        surface_only = false;
    //Go through the list of nodes for the stages
    for ( i = 0; i < EllemaxStages(); i++ )
    {
        int t_reduce_nodes = 0;
        max = ElleMaxNodes();
        //to prevent moving a single node always at the same time, we shuffel them randomly at each step
        ran.clear();
        for ( j = 0; j < max; j++ )
            if ( ElleNodeIsActive( j ) )
                ran.push_back( j );
        std::random_shuffle( ran.begin(), ran.end() );
        for ( j = 0; j < ran.size(); j++ )
        {
            double tot_time = 0.0;
            if ( ElleNodeIsActive( ran.at( j ) ) )
            {

             // only allow GBM between same mineral type
                ElleNodeSameMineral(ran.at(j), &same, &themin);
                if ( same==1 ) {
                    iter=0;
                    oldTimestep = ElleTimestep();
                    move = GGMoveNode( ran.at( j ), & newxy );
                 // reduce timestep for this node until move is != TIME_REDUCE
                    while ( move==TIME_REDUCE && iter<itermax &&
                                                 tot_time<oldTimestep ) {
                        tot_time += ElleTimestep();
                        ElleSetTimestep( oldTimestep-tot_time );
                        iter++;
                        ElleUpdatePosition( ran.at( j ), & newxy );
                        if (surface_only)
                        {
                            if (ElleNodeIsDouble( ran.at( j ) ))
                                ElleCheckDoubleJ( ran.at( j ) );
                            else if (ElleNodeIsTriple( ran.at( j ) ))
                                ElleCheckTripleJ( ran.at( j ) );
                        }
                        else
                        {
                            ElleCrossingsCheck( ran.at( j ), & newxy );
                            if (ElleNodeIsActive( ran.at( j )))
                                if (ElleNodeIsDouble( ran.at( j )))
                                    ElleCheckDoubleJ( ran.at( j ));
                        }
                        if (ElleNodeIsActive(ran.at( j )))
                            move = GGMoveNode( ran.at( j ), & newxy );
                        else 
                            move = 0;
                    };
                    if ( move==TIME_REDUCE || (iter>0) ) {
                        if (iter==itermax) 
                            Log(0,"Problem reducing timestep");
                        ElleSetTimestep( oldTimestep );
                        t_reduce_nodes++;
                        if (t_reduce_nodes > ran.size()/10)
                            OnError("Timestep too big for microstructure",0);
                    }
                    else if (move==1) {
                        ElleUpdatePosition( ran.at( j ), & newxy );
                        if (surface_only)
                        {
                            if (ElleNodeIsDouble( ran.at( j ) ))
                                ElleCheckDoubleJ( ran.at( j ) );
                            else if (ElleNodeIsTriple( ran.at( j ) ))
                                ElleCheckTripleJ( ran.at( j ) );
                        }
                        else
                        {
                            ElleCrossingsCheck( ran.at( j ), & newxy );
                            if (ElleNodeIsActive( ran.at( j )))
                                if (ElleNodeIsDouble( ran.at( j )))
                                    ElleCheckDoubleJ( ran.at( j ));
                        }
                    }
                }
            }
        }
        max = ElleMaxNodes();
        for (j=0;j<max;j++) {
            if (ElleNodeIsActive(j))
                if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
                else if (ElleNodeIsTriple(j)) ElleCheckTripleJ(j);
        }
        //ElleFlynnTopologyCheck();
        ElleUpdate();
    }
}
