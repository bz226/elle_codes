#include "gbm.elle.h"
#include "lut.h"
#include "gpcclip.h"
#include "polygon.h"

using std::vector;

double  GetAngleEnergy(double angle);
double GetOrientationEnergy(int rgn1, int rgn2, Coords *nbxy, double len, int id);
int GetCSLFactor(Coords_3D xyz[3], double *factor);

int update_dislocden();
double area_triangle (Coords n1, Coords n2, Coords n3);
double density_flynn(int node, Coords *xy); 
int diagonal=0;  // 0 use 4 dummy nodes <default>, 1 use diagonal positions 
double afactor=0; // 0 non redistribution of DD inside flynn <default>, 1 full-automatic redistribution  
int TNcheckopt=0; // ==1 non check TN

// Update information
/*	10/12/2008 Modified to add a variable to select flynns or unodes information to calculate the misorientation and anisotropy of boundary energy

/* 	GBM driven by surface energy and strain stored energy. The volumetric energy is calculated using data from flynn or unodes layer
* 	The process verify the condition of grain boundary with thickness = 0; 
* 	if GB_thickness != 0 then need to modify ebergy surface of both interfaces 
*	The nodes are moved along the direction of maximum reduction of energy similar than described in Becker et al. (2008).    
*
* 	Volumetric energy is calculated using clipping function, gradient of dislocation density is interpolated.
* 	DataUser parameters:
* 	1) The local gradient of energy is calculated using 4-dummy nodes or 8-dummy nodes
*   2) A parameter to simulate redistribution of properties inside flynns (DDs) 
*   3) if user[3]=1 avoid check TN..
*/

/*!/brief Calculates the (Surface) Energy of a node
 
This really only calculates the surface energy of the node, nothing else. General equation is E=en(l1+l2+l3)*lengthscale
with E=energy, en=surface energy and l1/l2/l3 the length of the segments next to the node adjusted to the lengthscale */

double GetNodeEnergy( int node, Coords * xy )
{
    int err, n, node2, node1, node3, nbnode[3], mineral, rgn[3],nnbnode[2];
    Coords n1, n2, n3, v1, v2, v3, vv[3];
    double l1, l2, l3, E, en = 0;
    double bodyenergy=0, energyofsurface=0, boundary_width=0.0;
	int i,j,id;
	
	id=0; // if id != 1 working with flynn, id ==1 working with unodes 
	
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
	vv[0].x= n1.x - xy->x;
	vv[0].y= n1.y - xy->y;
	
    v2.x = n2.x - xy->x;
    v2.y = n2.y - xy->y;
	vv[1].x= n2.x - xy->x;
	vv[1].y= n2.y - xy->y;
	
    l1 = GetVectorLength( v1 );
    l2 = GetVectorLength( v2 );

    //if the node is a triple, we have to get the third node and the length of the segment
    if ( ElleNodeIsTriple( node ) )
    {
        node3 = nbnode[2];
        ElleNodePlotXY( node3, & n3, xy );
        v3.x = n3.x - xy->x;
        v3.y = n3.y - xy->y;
		vv[2].x= n3.x - xy->x;
	    vv[2].y= n3.y - xy->y;
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
            // energyofsurface = GetMineralAttribute( mineral, SURFACE_ENERGY );///2;
			energyofsurface=GetMineralAttribute_ice(SURFACE_ENERGY_ice);			
            //en += GetMineralAttribute( mineral, SURFACE_ENERGY );///2;

            if (ElleFlynnAttributeActive(EULER_3)||
							ElleFlynnAttributeActive(CAXIS))
			{
				int nbrgn, rgn1, rgn2;
    			Coords relxy, xymid;
				double len, vmid;
				ElleNeighbourRegion(nbnode[n], node, &nbrgn);
				
				/*
				if ( id != 1) { // this is the flynn version .. 
					
                // len is the same as l1, l2, l3 depending on n
                ElleRelPosition(xy,nbnode[n],&relxy,&len);
				energyofsurface *= GetOrientationEnergy(rgn[n],nbrgn,&relxy,len,id); 
					// printf("rgn[n],nbrgn,energy  %i %i %e\n",rgn[n],nbrgn,energyofsurface);
				 // energyofsurface *= 1;	
					
				}
				else { // this is the unodes version .. 
					
				// midpoint segment 
				xymid.x= xy->x+vv[n].x*0.5;
				xymid.y= xy->y+vv[n].y*0.5;
				
				// First version, take information of the nearest unode in both regions 
				rgn1=SearchUnode(rgn[n],&xymid);
				rgn2=SearchUnode(nbrgn,&xymid);
				
                // len is the same as l1, l2, l3 depending on n
                ElleRelPosition(xy,nbnode[n],&relxy,&len);
				// energyofsurface *= 1; 
				energyofsurface *= GetOrientationEnergy(rgn1,rgn2,&relxy,len,id); 
				 // printf("UNODES rgn[n],nbrgn,energy  %i %i %e\n",rgn1,rgn2,energyofsurface);
				}
				*/
				energyofsurface *= 1;
			}
			en += energyofsurface;
        }
        n++;
    }
	
	    E = en * ( l1 + l2 ) * ElleUnitLength();

	    if ( ElleNodeIsTriple( node ) )
        E = (E + en*l3 * ElleUnitLength());
		
		{ 	
    	if ( ElleNodeIsTriple( node ) )
        	E /= 3.0;
		else E /= 2.0;
		}
	
    return E;
}

double GetNodeStoredEnergy( int node, Coords * xy )
{

	double area, density=0, density_energy;
	double dE;
	
	// First PART: Calculates the swept area 
	// Second PART: Calculates average dislocation density 
	area=area_swept_gpclip(node, xy);
 	// density=density_unodes(node, xy); 
	// density= density_flynn (node, xy);	
	
	if (ElleUnodeAttributeActive(U_DISLOCDEN)) 
				density=density_unodes(node, xy); 		
	else { 
		if (ElleFlynnAttributeActive(DISLOCDEN))
			density= density_flynn (node, xy);
  		else density=0;		
		}
		
    density_energy= GetMineralAttribute_ice (DD_ENERGY_ice);   	

	dE = density*area*density_energy; 
		
	return dE;
	
	}

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
	area_swept *= ElleUnitLength()*ElleUnitLength();
	
	return (area_swept);

}

double area_triangle (Coords n1, Coords n2, Coords n3)
{
	return (((n2.x-n1.x)*(n3.y-n1.y)-(n3.x-n1.x)*(n2.y-n1.y))/2.0);
}	

double density_unodes (int node, Coords * xy ) 
{

	int max_unodes,i,j,k, rgn[3], jj=0; 
	double roi, e[2], dist_total, val, dist, disloc; 
	Coords xy_unode;
 	
    std::vector<int> unodelist;    	
	
// Calculate the local density of a x,y position.  
// Brute routine with large list of the unodes of the flynns that belongs the bnode of reference 
// Area of search is limited by ROI 	
// npts=sqrt(max_unodes);
// Add a check of unodelist?¿ , if smaller than a critical number than?¿ 		
	
    max_unodes = ElleMaxUnodes();	 
    roi = sqrt(1.0/(double)max_unodes/3.142)*3;	// aprox. limit at 2nd neighbours
	e[0] = e[1] = dist_total = 0.0;

	ElleNodeUnitXY(xy);
	
  	for (i=0;i<ElleMaxFlynns();i++) 
   	{
      if (ElleFlynnIsActive(i)) 
	  {   
        if (EllePtInRegion(i,xy)) 
		{
			ElleGetFlynnUnodeList(i,unodelist); 
			
			for(k=0;k<unodelist.size();k++) {
				  
                ElleGetUnodePosition(unodelist[k],&xy_unode); // xy es del node !!
				ElleCoordsPlotXY (&xy_unode, xy);
                dist = pointSeparation(xy,&xy_unode);
				  
                if (dist<roi) {
	            	ElleGetUnodeAttribute(unodelist[k],&val,U_DISLOCDEN);
                  	e[0] +=  val*(roi-dist);	
                  	dist_total += (roi-dist);
					jj ++;
					}
              	}
			// unodelist.clear();
						       break;
			}
          }
	  }
		  
          disloc=e[0]/dist_total; 

	if (jj == 0) {
		jj=0;
		e[0]=dist_total=0.0;
    // roi = sqrt(1.0/(double)max_unodes/3.142)*4;	// aprox. limit at 2nd neighbours
	
		for(k=0;k<max_unodes;k++) {
				  
                ElleGetUnodePosition(k,&xy_unode); // xy es del node !!
				ElleCoordsPlotXY (&xy_unode, xy);
                dist = pointSeparation(xy,&xy_unode);
				  
                if (dist<roi) {
	            	ElleGetUnodeAttribute(k,&val,U_DISLOCDEN);
                  	e[0] +=  val*(roi-dist);	
                  	dist_total += (roi-dist);
					jj ++;
					}
              	}
				          disloc=e[0]/dist_total; 
			}
		
/*		
          unodelist.clear();
		  ElleRegions(node,rgn); //find the neighbours flynns of the bnode
		  
          for(j=0;j<3;j++) {
            if (rgn[j] != NO_NB) {
				
              ElleGetFlynnUnodeList(rgn[j],unodelist);//get the list of unodes
								
              for(k=0;k<unodelist.size();k++) {
				  
                ElleGetUnodePosition(unodelist[k],&xy_unode); // xy es del node !!
				ElleCoordsPlotXY (&xy_unode, xy);
                dist = pointSeparation(xy,&xy_unode);
				  
                if (dist<roi) {
	            	ElleGetUnodeAttribute(unodelist[k],&val,U_DISLOCDEN);
                  	e[0] +=  val*(roi-dist);	
                  	dist_total += (roi-dist);
					jj ++;
					}
              	}
            
			unodelist.clear();
			}
          }
		  
          disloc=e[0]/dist_total; 
          unodelist.clear();
		  */

		   // add a routine if nan error 
		   //  printf("disloc %i %e\n", jj, disloc);
	unodelist.clear();
	return (disloc); 

}	

double density_flynn(int node, Coords *xy) 
{

int i;
double density;
// disloc stored at flynn level 
 
		 ElleNodeUnitXY(xy);
	
  for (i=0;i<ElleMaxFlynns();i++) 
   {
      if (ElleFlynnIsActive(i)) 
	  {   
         if (EllePtInRegion(i,xy)) 
		 {
			ElleGetFlynnRealAttribute(i, &density,DISLOCDEN); 
			 break;
		 }	 
	 }
 }

 return(density);
 
}

double GetOrientationEnergy(int rgn1, int rgn2, Coords *nbxy, double len, int id)
{
    int i;
	double csl_factor=1.0;
	double cosalpha = 0, alpha=0.0;
	double energy = 0;
    Coords_3D xyz[3];
    double val[3];
	
    for (i=0;i<3;i++) xyz[i].x = xyz[i].y = xyz[i].z = 0;

// Take euler angle and convert to vector 
// For columnar models, sufficient; General three-dimensional or high symmetry material use full-misorientation  	

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
	energy = GetAngleEnergy(alpha)*csl_factor;

    cosalpha = (nbxy->x * xyz[1].x + nbxy->y * xyz[1].y)/len;
        /*
         * make sure cosalpha is in range -1,1 for acos
         */
    if (cosalpha > 1.0) cosalpha = 1.0;
    if (cosalpha < -1.0) cosalpha = -1.0;
    if (cosalpha >= 0.0) alpha = acos(cosalpha);
    else                 alpha = acos(-cosalpha);
    energy += GetAngleEnergy(alpha)*csl_factor;
	return(energy);
}


/*
// original version 
double GetOrientationEnergy(int rgn1, int rgn2, Coords *nbxy, double len)
{
    int i;
	double csl_factor=1.0;
	double cosalpha = 0, alpha=0.0;
	double energy = 0;
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
        
        //* make sure cosalpha is in range -1,1 for acos
         
	if (cosalpha > 1.0) cosalpha = 1.0;
	if (cosalpha < -1.0) cosalpha = -1.0;
	if (cosalpha >= 0.0) alpha = acos(cosalpha);
	else                 alpha = acos(-cosalpha);
	energy = GetAngleEnergy(alpha)*csl_factor;
    cosalpha = (nbxy->x * xyz[1].x + nbxy->y * xyz[1].y)/len;
        
         //* make sure cosalpha is in range -1,1 for acos
         
    if (cosalpha > 1.0) cosalpha = 1.0;
    if (cosalpha < -1.0) cosalpha = -1.0;
    if (cosalpha >= 0.0) alpha = acos(cosalpha);
    else                 alpha = acos(-cosalpha);
    energy += GetAngleEnergy(alpha)*csl_factor;
	return(energy);
}
*/	

double GetAngleEnergy(double angle)
{
    int angdeg;
    double energy;
                                                                                
    angdeg = (int)(angle*RTOD + 0.5);
    energy = ElleEnergyLUTValue(angdeg);
    return(energy);
}
                                                                                
#if XY
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
#endif
 
// Used as an aproach to the Read-Shockley relation
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

int GGMoveNode_all( int node, Coords * xy )
{
    double e0, e1, e2, e3, e4, switchd = ElleSwitchdistance()/5;
    Coords oldxy, newxy, prev;
	double es1, es2, es3, es4; 
	Coords pvec, pvec_strain, pvec_surf;
	Coords pvec_strain2, pvec_surf2;
	double switchdd;
	// int diagonal=0; 
	
	// UserData udata;
    // ElleUserData(udata);
    // diagonal=(int)udata[0];
	
    ElleNodePosition( node, & oldxy );
	
    newxy.x = oldxy.x + switchd;
    newxy.y = oldxy.y;
    e1 = GetNodeEnergy( node, & newxy );
    es1= GetNodeStoredEnergy( node, & newxy );
	
    newxy.x = oldxy.x - switchd;
    e2 = GetNodeEnergy( node, & newxy );
	es2= GetNodeStoredEnergy( node, & newxy );
	
    newxy.x = oldxy.x;
    newxy.y = oldxy.y + switchd;
    e3 = GetNodeEnergy( node, & newxy );
	es3= GetNodeStoredEnergy( node, & newxy );
	
    newxy.y = oldxy.y - switchd;
    e4 = GetNodeEnergy( node, & newxy );
	es4 = GetNodeStoredEnergy( node, & newxy );

	pvec_surf2.x=e1-e2;
	pvec_surf2.y=e3-e4;
	pvec_strain2.x=es1-es2;
	pvec_strain2.y=es3-es4;

// 04/12/2008
// Default version with 4-pivot points, but added the possibility to use diagonal positions to increase stability of triple nodes
// 4-pivot points + small time steps seems that TNs are normally stable but some drift if strain stored energy >>> boundary energy 

	if (diagonal==1) {
	
    double e5, e6, e7, e8;
    double es5, es6, es7, es8; 
	e5=e6=e7=e8=0;
	es5=es6=es7=es8=0;
	switchdd=switchd/sqrt(2.0);

    	newxy.x = oldxy.x + switchdd;
		newxy.y = oldxy.y + switchdd;
		e5 = GetNodeEnergy( node, & newxy );
		es5 = GetNodeStoredEnergy( node, & newxy );
		
    	newxy.x = oldxy.x - switchdd;
		newxy.y = oldxy.y - switchdd;	
		e6 = GetNodeEnergy( node, & newxy );
		es6= GetNodeStoredEnergy( node, & newxy );
		
		newxy.x = oldxy.x + switchdd;
		newxy.y = oldxy.y - switchdd;
		e7 = GetNodeEnergy( node, & newxy );
		es7= GetNodeStoredEnergy( node, & newxy );
		
		newxy.x = oldxy.x - switchdd;
		newxy.y = oldxy.y + switchdd;
		e8 = GetNodeEnergy( node, & newxy );
		es8= GetNodeStoredEnergy( node, & newxy );
	
	
    pvec_surf2.x +=((e5-e6+e7-e8)/sqrt(2.0));
    pvec_surf2.y +=((e5-e6-e7+e8)/sqrt(2.0));
	pvec_strain2.x +=((es5-es6+es7-es8)/sqrt(2.0));
	pvec_strain2.y +=((es5-es6-es7+es8)/sqrt(2.0));

    pvec_surf2.x /= 2.0;
    pvec_surf2.y /= 2.0;
	pvec_strain2.x /= 2.0;
	pvec_strain2.y /= 2.0;
	}

    //Get the vector of the direction with highest decrease of surface energy
   	pvec_surf.x = -1*pvec_surf2.x / ( 2 * switchd );
    pvec_surf.y = -1*pvec_surf2.y / ( 2 * switchd );

    // Get the vector of direction with "highest" decrease of strain stored energy
    // call GGMoveNode_test( int node, es1, es2, es3, es4, trialdist);
	
	pvec_strain.x= 2*pvec_strain2.x/( 2*switchd ); // added factor 2
	pvec_strain.y= 2*pvec_strain2.y/( 2*switchd );

	// Get resolved "maximum" direction 

	pvec.x=pvec_surf.x+pvec_strain.x;
	pvec.y=pvec_surf.y+pvec_strain.y;

	return GetMoveDir( node, pvec, xy ,switchd);
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
    int i, j, max, err;
	int same=0, themin;
    Coords newxy;
    vector < int > ran;
	int num;
	int *ids;
	double boundary_width;
	// int diagonal=0; 
	// double afactor=0;
 	UserData udata;
 	
    if ( ElleCount() == 0 )
        ElleAddDoubles();
    if ( ElleDisplay() )
        EllePlotRegions( ElleCount() );
    ElleCheckFiles();

	
	 ElleUserData(udata);
	 diagonal=(int)udata[0];
	 afactor= udata[1];
	 TNcheckopt=(int)udata[2];
	
	if (diagonal != 0) printf("8-node version, afactor= %lf\n", afactor);
	else
		printf("4-node version, afactor= %lf\n", afactor);
/*	
    if (ElleFlynnAttributeActive(DISLOCDEN) ||
				ElleFlynnAttributeActive(EULER_3) ||
						ElleFlynnAttributeActive(CAXIS))
		surface_only = false;
*/
	 //surface_only = false;
    //Go through the list of nodes for the stages
	
// Init flynns and unodes attributes 

	if (!ElleUnodeAttributeActive(ATTRIB_C))
    ElleInitUnodeAttribute(U_ATTRIB_C);
	if (!ElleFlynnAttributeActive(F_ATTRIB_C))
	ElleInitFlynnAttribute(F_ATTRIB_C); 	
			
	int jj;
    for (i=0;i<ElleMaxUnodes();i++)
    {
	    jj=ElleUnodeFlynn(i);
	    ElleSetUnodeAttribute(i,U_ATTRIB_C, double(jj));
    }

    for (i=0;i<ElleMaxFlynns();i++) 
	{
    	if (ElleFlynnIsActive(i)) 
		{
			ElleSetFlynnRealAttribute(i,double(i),F_ATTRIB_C);
			}
	}

    for ( i = 0; i < EllemaxStages(); i++ )
    {
		printf("Stage : %d\n", i);
		boundary_width=ElleBndWidth();
		// printf(" Elle Boundary Width %e\n",boundary_width);
        max = ElleMaxNodes();
        //to prevent moving a single node always at the same time, we shuffel them randomly at each step
        ran.clear();

        for ( j = 0; j < max; j++ )
            if ( ElleNodeIsActive( j ) )
                ran.push_back( j ); // copy nodes id to <vector> run
			
        std::random_shuffle( ran.begin(), ran.end() );

/*			
        if(ElleFlynnIsActive(1) && fmod(i,20)==0)
        {
            FILE *f=fopen("test.csv","a");
            fprintf(f,"%d,%e,%d\n",i,ElleFlynnArea(1),ran.size());
            fclose(f);
        }
*/		
        for ( j = 0; j < ran.size(); j++ )
        {
            if ( ElleNodeIsActive( ran.at( j ) ) )
            {
			 // only allow GBM between same mineral type
    			ElleNodeSameMineral(ran.at(j), &same, &themin);
				   // printf("%i\n",ran.at(j)); 
				
                if ( same==1 && GGMoveNode_all( ran.at( j ), & newxy ) )
                {
					if (surface_only)
					{
						ElleUpdatePosition( ran.at( j ), & newxy );
	                    if (ElleNodeIsDouble( ran.at( j ) ))
                            ElleCheckDoubleJ( ran.at( j ) );
						 else  	 
						 if (ElleNodeIsTriple( ran.at( j ) )) { 
                          	if (TNcheckopt!=1) ElleCheckTripleJ( ran.at( j ) );
					 		}
					}
					else
					{

						ElleCrossingsCheck( ran.at( j ), & newxy );
				        if (ElleNodeIsActive( ran.at( j )))
			     	       if (ElleNodeIsDouble( ran.at( j )))
								ElleCheckDoubleJ( ran.at( j ));
						   	  else if (ElleNodeIsTriple( ran.at( j ) )){
                              if (TNcheckopt!=1) ElleCheckTripleJ( ran.at( j ) );
							  }
					}
                }
              }
            }
			
        max = ElleMaxNodes();
        for (j=0;j<max;j++) {
            if (ElleNodeIsActive(j))
                if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
                else if (ElleNodeIsTriple(j)){
					 if (TNcheckopt!=1)	ElleCheckTripleJ(j);
					 }						 
        }
		
        //ElleFlynnTopologyCheck();

      err=update_dislocden();	
		        ElleUpdate();
	
    }
}


int update_dislocden()
{
	// Simplified version of GBM unodes update 
	int j, i, ii;
    int max_flynns;
	int flynnid, flynnold,flynnold2, count;
	int valold; // afactor;
    double new_area,old_area,new_density,density;
 	double val2, max_unodes, old_flynn, roi, val;
	double val_euler[3], dist_total, min_dist; 
	double density_min,dist,a_factor=0;
	vector<int> unodelist;
	Coords ref, xy;
	UserData udata; 
		
    density_min=0.0; // implicit in the GBM scheme	
	max_unodes = ElleMaxUnodes();
	roi = sqrt(1.0/max_unodes/3.142)*5;
	
	// 1st step UPDATE -- cylcle through unodes and correct flynn unodelists 
    for (i=0;i<max_unodes;i++)   
    {
		
	ElleGetUnodeAttribute(i,U_ATTRIB_C,&old_flynn);				
	flynnold= int (old_flynn);
	flynnid=ElleUnodeFlynn(i);

	// take correct flynnid
	
	if (!ElleFlynnIsActive(flynnid)) {
		ElleGetUnodePosition(i,&xy);
		ii=ElleUnodeFlynn(i);	
	   	for (j=0;j<ElleMaxFlynns();j++) 
   		{
      		if (ElleFlynnIsActive(j)) 
	  		{  
         		if (EllePtInRegion(j,&xy)) 
		 		{
					flynnid=j;
					ElleAddUnodeToFlynn(flynnid, i);
					printf(" new flynn, inactive flynn %i  %i\n", flynnid, ii);	  					
					break;
				}
			}
		}
	  }	
	}

	// 2nd step check a flynn parameter that store identification of flynn 
	// only flynn active ...

	max_flynns = ElleMaxFlynns(); 
    for (j=0;j<max_flynns;j++) 
	{
    	if (ElleFlynnIsActive(j))  
		{
		ElleGetFlynnRealAttribute(j, &val,F_ATTRIB_C);
		valold=int(val);	
 		if (valold != j) {
			ElleGetFlynnUnodeList(j,unodelist); 
			count=unodelist.size();

			for(i=0;i<count;i++){
				ElleGetUnodeAttribute(i,U_ATTRIB_C,&old_flynn);		
				flynnold= int (old_flynn);
				if (flynnold == valold) ElleSetUnodeAttribute(i,U_ATTRIB_C, double(j));				
				}
				unodelist.clear();
				printf("renumbered flynn ?¿");
			}
			ElleSetFlynnRealAttribute(j,double(j),F_ATTRIB_C);
		}
	}
		
	// 3rd step UPDATE DDs and EULER... this version use nearest unode
	for (i=0;i<max_unodes;i++)   
   	{	
		ElleGetUnodeAttribute(i,U_ATTRIB_C,&old_flynn);				
		flynnold= int (old_flynn);
		flynnid=ElleUnodeFlynn(i);
		
	if ( flynnold != flynnid ) {		
		printf("old new flynn %i %i\n",flynnold,flynnid);
		
		// Update euler angles. Nearest unode, use quaternion for interpolate the Euler orientation 	
   	 	ElleGetFlynnUnodeList(flynnid,unodelist);
		count=unodelist.size();
		ElleGetUnodePosition(i,&ref);
		
		dist_total=0.0;
		min_dist=1;
		ii=0;
		
		for (j=0; j<count; j++) {	

		ElleGetUnodeAttribute(unodelist[j],U_ATTRIB_C,&old_flynn);	
		flynnold2= int (old_flynn);

		if ( flynnold2 == flynnid ) {

		ElleGetUnodePosition(unodelist[j],&xy);	
		ElleCoordsPlotXY (&ref, &xy);			  
        dist = pointSeparation(&ref,&xy);
				
			if (dist<roi) {
			ii++;
				if ( dist < min_dist) {
					ElleGetUnodeAttribute(unodelist[j], &val_euler[0],E3_ALPHA);
					ElleGetUnodeAttribute(unodelist[j], &val_euler[1],E3_BETA);
					ElleGetUnodeAttribute(unodelist[j], &val_euler[2],E3_GAMMA);	
					min_dist = dist;					
				}                  	
			}
		}
	}
	printf(" unodes count %i\n", ii); 	
	// set new information 
 	ElleSetUnodeAttribute(i,U_DISLOCDEN, density_min);
 	ElleSetUnodeAttribute(i,E3_ALPHA, val_euler[0]);
 	ElleSetUnodeAttribute(i,E3_BETA, val_euler[1]);
 	ElleSetUnodeAttribute(i,E3_GAMMA, val_euler[2]); 			
 	ElleSetUnodeAttribute(i,U_ATTRIB_C, double(flynnid));
	
		unodelist.clear();	
    }
	}


	// 2nd part: simulate internal restructuration of grains  
	// Not active with FFT simulations
    // To compare gbm with bnodes simulations: automatic readjust at average value 
    // Only scalar dislocation density, not effect on euler orientation of unodes 	
	double avdensity; //a_factor=0.0 redistribution factor; 1.0 full while 0.0 non redistribution

	   	for (j=0;j<ElleMaxFlynns();j++) 
   		{
      		if (ElleFlynnIsActive(j)) 
	  		{  
				unodelist.clear();
				avdensity=0.0;
    			ElleGetFlynnUnodeList(j,unodelist); // get the list of unodes for a flynn
				count=unodelist.size();
				
		for(i=0;i<count;i++){
	
			ElleGetUnodeAttribute(unodelist[i],U_DISLOCDEN,&density);					
			avdensity += density; 			
			}
		avdensity /= count;
		
		for(i=0;i<count;i++){		
		  	ElleGetUnodeAttribute(unodelist[i],U_DISLOCDEN,&density);
		  	new_density= density-(density-avdensity)*a_factor; 
 	      	ElleSetUnodeAttribute(unodelist[i],U_DISLOCDEN, new_density);			
		 	}
		
	 	  }
 		}
		 
	return 0;
}
