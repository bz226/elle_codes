
#include "FS_fft2elle.h"

main(int argc, char **argv)
{
    int err=0;
    UserData userdata;

    /*
     * initialise
     */
    ElleInit();
    ElleSetOptNames("ImportDDs","ExcludePhaseID","UnodeDimEunodes","unused","unused","unused","unused","unused","unused");
    
    ///FS added default userdata
    ElleUserData(userdata);
    userdata[0] =0; // Import dislocdens or not 0=don't import, 1=import Default: 1
    userdata[1] =0; // Phase with with flynn VISCOSITY == userdata[1] will be EXCLUDED FROM DISLOCDEN update (DD's are set to 0). For NO exclusion type input:0
    userdata[2] =0; // Generate data for passive markers, 0=don't generate it, type the dimension (128,256) to generate it Default: 0
    ElleSetUserData(userdata);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitMoveBnodes);

    if (err=ParseOptions(argc,argv))
        OnError("",err);
    ElleSetSaveFrequency(1);

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * set the base for naming statistics and elle files
     */
    char cFileroot[] = "fft2elle";
    ElleSetSaveFileRoot(cFileroot);

    /*
     * run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 


/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitMoveBnodes()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(MoveBnodes);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
        /*
         * check for any necessary attributes which may
         * not have been in the elle file
         */
    }
}

/*!
 */
int MoveBnodes()
{
    int err=0,i,j;
	int import_DD=0;
    vector < int > U_attrib_init;
    vector < int > N_attrib_init;
    vector<int> attribs(3,NO_VAL);
    attribs[0] = ATTRIB_A;
    attribs[1] = ATTRIB_B;

        /*!
         * Use ATTRIB_A to hold strain in x direction
         * Use ATTRIB_B to hold strain in y direction
         */
    if (!ElleUnodeAttributeActive(ATTRIB_A)) {
        ElleInitUnodeAttribute(ATTRIB_A);
        U_attrib_init.push_back(ATTRIB_A);
    }
    if (!ElleUnodeAttributeActive(ATTRIB_B)) {
        ElleInitUnodeAttribute(ATTRIB_B);
        U_attrib_init.push_back(ATTRIB_B);
    }
    if (!ElleNodeAttributeActive(ATTRIB_A)) {
        ElleInitNodeAttribute(ATTRIB_A);
        N_attrib_init.push_back(ATTRIB_A);
    }
    if (!ElleNodeAttributeActive(ATTRIB_B)) {
        ElleInitNodeAttribute(ATTRIB_B);
        N_attrib_init.push_back(ATTRIB_B);
    }
	UserData userdata;
    ElleUserData(userdata);
    import_DD=(int)userdata[0]; // import DDs
    
    /*
     * Try and already reset the cell here
     */
    //ResetCell(dd[1][0],dd[1][1],dd[1][2],dd[2][0]);
    
    /*
     * Save the unode positions increments from the fft output
     *     in attribs
     * (if unodexyz.out was the strain increment, we could call
           SetUnodeAttributesFromFile("unodexyz.out",attribs);)
     */
    err = SetUnodeStrainFromFile("unodexyz.out",attribs);
    if (err) OnError("unodexyz.out",err);
		
     err = LoadDataTemp("temp-FFT.out"); /// FS changed this, because temp-FFT.out needs to be loaded instead of temp.out
     if (err) OnError("temp-FFT.out",err);	/// FS changed this, because temp-FFT.out needs to be loaded instead of temp.out
     printf("%lf\t%lf\t%lf\n",dd[1][0],dd[1][1],dd[1][2]);
	 printf("shear strain increment %lf\n", dd[2][0]);
	
    /*
     * Calculate the strain at the bnodes, using the 
     * the unode strain values from each flynn neighbour
     * of the bnode
     */
    SetBnodeStrain(attribs);
    
    //ResetCell(dd[1][0],dd[1][1],dd[1][2],dd[2][0]);
    
    /*! 
     * FS: For creation of passive marker grid with plot_ugrid:
     */
    int iEunodeDim = (int)userdata[2];
    if (iEunodeDim != 0) FS_SetEunodeStrain(attribs);
    
    ResetCell(dd[1][0],dd[1][1],dd[1][2],dd[2][0]);
    
    /*
     * Set the new bnode and unode positions
     */
    PositionFromStrain(attribs);
    /*
     * set the euler angles to the values from the FFT output
     */
    attribs[0] = E3_ALPHA;
    attribs[1] = E3_BETA;
    attribs[2] = E3_GAMMA;
    err = SetUnodeAttributesFromFile("unodeang.out",attribs);
    if (err) OnError("unodeang.out",err);
		
	/*
	* set the values of dislocation density from the FFT output
	*/
	if (import_DD == 1)
        if (!ElleUnodeAttributeActive(U_DISLOCDEN))	
	 		ElleInitUnodeAttribute(U_DISLOCDEN);

    FS_SetUnodeDD("tex.out");
    
	/*
	* check if unodes changes of flynn 
	*/
	FS_CheckUnodes();
	
    /*
     * remove any temporary attributes which do not need to
     * be written to the output elle file
     */
    if (err = ElleRemoveUnodeAttributes(U_attrib_init))
        OnError("MoveBnodes ",ATTRIBID_ERR);
    if (err = RemoveNodeAttributes(N_attrib_init))
        OnError("MoveBnodes ",ATTRIBID_ERR);
		
    /*
     * write the updated Elle file
     */
    if (err=ElleWriteData("fft2elle.elle"))
        OnError("",err);
}

int SetUnodeStrainFromFile(const char *fname,vector<int> &attribs)
{
        /*
         * Read the new unode position from the fft output file
         * Set the attributes to be the new positions
         * Assume square grid of sqrt(max_unodes) per row
         */
    int err=0;
    int id, count, i, max_unodes;
    double eps = 1e-6;
    Coords xy, fft_xy;
    for (i=0, count=0;i<attribs.size();i++) if (attribs[i]!=NO_VAL) count++;
    if (count<1) return (ATTRIBID_ERR);
    double val[3];
    ifstream datafile(fname);
    
    if (!datafile) return(OPEN_ERR);
    if (!ElleUnodeAttributeActive(attribs[0]))
        ElleInitUnodeAttribute(attribs[0]);
    if (!ElleUnodeAttributeActive(attribs[1]))
        ElleInitUnodeAttribute(attribs[1]);
        
    while (datafile) 
    {
        datafile >> id >> val[0] >> val[1] >> val[2];
        fft_xy.x = val[0];
        ElleGetUnodePosition(id,&xy);
        fft_xy.y = xy.y; //elle2fft only shifts x coord for simple sh
        if (fabs(dd[1][1]) > eps) 
            fft_xy.y = val[1]; //pure sh
        ElleCoordsPlotXY(&xy,&fft_xy);
        //elle2fft moved unodes to regular grid in unitcell
        ElleSetUnodeAttribute(id,val[0]-xy.x,attribs[0]);
        ElleSetUnodeAttribute(id,val[1]-xy.y,attribs[1]);
    }
    datafile.close();

#if XY

   // reposition of unodes to old cellBox
    // if (ElleSSOffset() != 0.0) 
		{
	max_unodes = ElleMaxUnodes(); // maximum node number used
    for (i=0;i<max_unodes;i++)   // cycle through unodes
    {
        ElleGetUnodePosition(i,&xy);
        ElleNodeUnitXY(&xy);
        ElleSetUnodePosition(i,&xy);
    }
}	
#endif

    return(err);
}


int LoadDataTemp(const char *fname)
{
        /*
         * Read the new cellbox size and displacements
         */
    int err=0;
    int id,i=0,j=0;
    double val[3];

    ifstream datafile(fname);
    if (!datafile) return(OPEN_ERR);
      for (j=0;j<3;j++) 
        {
        datafile >> val[0] >> val[1] >> val[2];
        for (i=0;i<3;i++) dd[j][i]=(double)val[i];
        //printf(" i j v1 v2 v3 dd1 dd2 dd3 %d %d %lf %lf %lf %lf %lf %lf \n",i,j,val[0],val[1],val[2],dd[j][0],dd[j][1],dd[j][2]);
	    
    	}
    
    datafile.close();

    return(err);
}

int SetUnodeAttributesFromFile(const char *fname,vector<int> &attribs)
{
    int id, count, i;
    Coords xy;
    for (i=0, count=0;i<attribs.size();i++) if (attribs[i]!=NO_VAL) count++;
    if (count<1) return (ATTRIBID_ERR);
    double *val = new double[count];
    ifstream datafile(fname);
    if (!datafile) return(OPEN_ERR);
    for (i=0;i<count;i++) {
        ElleInitUnodeAttribute(attribs[i]);
    }
    while (datafile) {
        datafile >> id ;
        for (i=0;i<count && datafile;i++) {
            datafile >> val[i];
            ElleSetUnodeAttribute(id,val[i],attribs[i]);
        }
    }
    datafile.close();
    delete [] val;
    return(0);
}

#if XY
int RemoveUnodeAttributes(vector<int> &attribs);
int RemoveUnodeAttributes(vector<int> &attriblist)
{
    int id, k, max_unodes=0, err=0;
    double x, y, tmp;
    max_unodes = ElleMaxUnodes(); // maximum unode number used
    for(id=0;id<max_unodes && !err;id++) {
        for(k=0;k<attriblist.size() && !err;k++) {
            ElleRemoveUnodeAttribute(id,attriblist[k]);
        }
    }
    for(k=0;k<attriblist.size() && !err;k++) {
        ElleRemoveDefaultUnodeAttribute(attriblist[k]);
    }
    return(err);
}
#endif

void SetBnodeStrain(vector<int> & attriblist)
{
    int i, j, k, count, numunodes, numattribs;
    int rgns[3], trace,numbnodes;
    int max_bnodes, max_unodes, max_nbs, bd_max_nbs;
    double val[2], roi=0.01;
    double e[3], dist, dist_total, total_count=0;
    Coords xy, ref, ref2;
CellData unitcell;
ElleCellBBox(&unitcell);
                                                                                
    for (i=0, numattribs=0;i<attriblist.size();i++)
        if (attriblist[i]!=NO_VAL) numattribs++;
    std::vector<int> bdnodelist, unodelist;
                                                                                
    max_bnodes = ElleMaxNodes(); // maximum bnode number used
    max_unodes = ElleMaxUnodes(); // maximum unode number used
    
    /*
     * FS: The product of boxwidth + boxheight will not remain constant in a 
     * pure shear simulation. Hence, a more accurate determination of ROI is 
     * used here using not sqrt(1.0 / (...), but sqrt(width*height / (...)
     * --> FOR THIS APPROACH THE BOX SHOULD NOT BE A PARALLELOGRAM ETC
     */  
    //double dBoxHeight = 0.0;
    //double dBoxWidth = 0.0;
    //double dBoxArea = 0.0;
    //dBoxHeight = unitcell.cellBBox[TOPLEFT].y - unitcell.cellBBox[BASELEFT].y;
    //dBoxWidth = unitcell.cellBBox[BASERIGHT].x - unitcell.cellBBox[BASELEFT].x;
    //dBoxArea = dBoxHeight*dBoxWidth; 
    //roi = sqrt(dBoxArea/(double)max_unodes/3.142)*3; // VIP, ROI controls number of unodes used to interpolate the local displacement field
    
    roi = FS_GetROI(3);
		
    for (i=0;i<max_bnodes;i++)   // cycle through bnodes
    {

		trace=0;
		count=0;
		
        if(ElleNodeIsActive(i)){
					numbnodes++;
          e[0] = e[1] = dist_total = 0.0;
		  ElleNodePosition(i,&ref);
          ElleRegions(i,rgns); //find the neighbour flynns
          for(j=0;j<3;j++) 
          {
            if (rgns[j] != NO_NB) 
            {
              unodelist.clear();
              ElleGetFlynnUnodeList(rgns[j],unodelist);//get the list of unodes
              for(k=0;k<unodelist.size();k++) 
              {
                ElleGetUnodePosition(unodelist[k],&xy);
                //Need to adjust val[] if using wrapped unode
                //ElleCoordsPlotXY(&xy,&ref);
	            ElleGetUnodeAttribute(unodelist[k],&val[0],attriblist[0]);
                ElleGetUnodeAttribute(unodelist[k],&val[1],attriblist[1]);
		
                //Need to adjust val[] if using wrapped unode
                //ElleCoordsPlotXY(&xy,&ref); // FS STILL: WHY IN COMMENTS?
                dist = pointSeparation(&ref,&xy);

                if (dist<roi) 
                {
                  e[0] +=  val[0]*(roi-dist);
                  e[1] +=  val[1]*(roi-dist);
                  dist_total += (roi-dist);
					trace=1;
					count++;
                }
              }
            }
          }
if (trace==0) {
cout << i<<' '<<"not found in flynn nbs"<<ref.x<<' '<<ref.y<<endl;
}
		  
		  // alternative if error in bnodes layer, scan all unodes list 
		  if (trace == 0) { 
              for(k=0;k<max_unodes;k++) {
                ElleGetUnodePosition(k,&xy);
	            ElleGetUnodeAttribute(k,&val[0],attriblist[0]);
                ElleGetUnodeAttribute(k,&val[1],attriblist[1]);
		
                //Need to adjust val[] if using wrapped unode
                //ElleCoordsPlotXY(&xy,&ref); // FS STILL: WHY IN COMMENTS?
                dist = pointSeparation(&ref,&xy);

                if (dist<roi) {
                  e[0] +=  val[0]*(roi-dist);
                  e[1] +=  val[1]*(roi-dist);
                  dist_total += (roi-dist);
					trace=1;
					count++;
                }
              }
		  printf("error bnode layer,no unodes dist < ROI, alternative scan all unodes\n");
		  }			  
		  
          ElleSetNodeAttribute(i,e[0]/dist_total,attriblist[0]);
          ElleSetNodeAttribute(i,e[1]/dist_total,attriblist[1]);
          unodelist.clear();
        }
		total_count += count;
    }
	
	total_count /= numbnodes;
	
// printf("average picked unodes %lf\n", total_count);
	
}

void PositionFromStrain(vector<int> & attriblist)
{
    /*
     * Move the elle unodes (no longer need the original positions)
     * Move the bnodes by the values in attriblist
     */
    int i, j, k, count, numnodes, numattribs;
    int max_nodes, rgn[3], nbs=0;
    double val;
    Coords xy, incr;

    max_nodes = ElleMaxUnodes(); // maximum node number used
    for (i=0;i<max_nodes;i++)   // cycle through unodes
    {
        ElleGetUnodePosition(i,&xy);
        ElleGetUnodeAttribute(i,&incr.x,attriblist[0]);
        ElleGetUnodeAttribute(i,&incr.y,attriblist[1]);
        xy.x += incr.x;
        xy.y += incr.y;
        ElleNodeUnitXY(&xy);
        ElleSetUnodePosition(i,&xy);
    }
    max_nodes = ElleMaxNodes(); // maximum node number used
    for (i=0;i<max_nodes;i++)   // cycle through bnodes
    {
        if(ElleNodeIsActive(i)){
            incr.x = incr.y = 0;
            ElleNodePosition(i,&xy);
            incr.x = ElleNodeAttribute(i,attriblist[0]);
            incr.y = ElleNodeAttribute(i,attriblist[1]);
            xy.x += incr.x;
            xy.y += incr.y;
			//ElleCopyToPosition(i,&xy);
            ElleSetPosition(i,&xy); // modify to ElleUpdatePosition?     
				
        }
    }
	
/*  this adds doubles and allows triple switches
  * Just check dj density?
*/
        for (j=0;j<max_nodes;j++) {
            if (ElleNodeIsActive(j))
                if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
                else if (ElleNodeIsTriple(j)) ElleCheckTripleJ(j);
					
        }
     ElleAddDoubles();
}

void ResetCell(double xstrain, double ystrain, double zstrain, double offset)
{
	//assume shortening parallel to y
    CellData unitcell;
    Coords  corners[4]; 
	double cum_offset;
    double dSSOffset = ElleSSOffset(); // FS added for offset update after ElleUpdateCellBBox

	cum_offset = ElleCumSSOffset();
	    ElleCellBBox(&unitcell);
	
// this assumes linear gradient for simple shear 
// 0->offset as y varies 0->1
// xstrain, ystrain are used if pure shear deformation, 0 for simple shear
    corners[BASELEFT].x = unitcell.cellBBox[BASELEFT].x;
    corners[BASELEFT].y = unitcell.cellBBox[BASELEFT].y;
    corners[BASERIGHT].x = unitcell.cellBBox[BASERIGHT].x+xstrain;
    corners[BASERIGHT].y = unitcell.cellBBox[BASERIGHT].y;
    corners[TOPRIGHT].x = unitcell.cellBBox[TOPRIGHT].x+offset+xstrain;
    corners[TOPRIGHT].y = unitcell.cellBBox[TOPRIGHT].y+ystrain;
    corners[TOPLEFT].x = unitcell.cellBBox[TOPLEFT].x+offset;
    corners[TOPLEFT].y = unitcell.cellBBox[TOPLEFT].y+ystrain; // FS corrected this: was "TOPRIGHT" on the right and side of "=" sign before
	
// FS uncommented this #if XY --> #endif part
#if XY
	temp=dd[2][0]; // incremental shear strain
		 	    ElleSetCellBBox(&corners[BASELEFT], &corners[BASERIGHT],
                    &corners[TOPRIGHT], &corners[TOPLEFT]);

	 cum_offset +=temp; 

     ElleSetCumSSOffset(cum_offset);
	  offset=modf(ElleCumSSOffset(),&tmp);
	  ElleSetSSOffset(offset);
#endif
		ElleUpdateCellBBox(&corners[BASELEFT], &corners[BASERIGHT],
                    &corners[TOPRIGHT], &corners[TOPLEFT]);	 
}


void check_error_UNUSED()
{

	// modified version than used in elle2fft process
	// 23th Nov. 2008
	
    int i,j,k,id,count, dummy_int, max_unodes;	
    double gamma_unode, gamma, gamma_int=4.0;
	double gamma_pos, gamma_neg, gamma_unode2, dist, dist_min; 
	double val[5], old_flynn, dummy;
	vector <int> unodelist; 
        vector <int> reassignedlist;
	Coords xy,xy_unode; 
	
// recheck unodes list 
	
	max_unodes = ElleMaxUnodes(); 


	for (j=0;j<ElleMaxFlynns();j++) {
		
    	if (ElleFlynnIsActive(j)) {
            /*
            // just look at the neighbour regions? May be an error
            // if narrow neighbour or unodes too sparse.
            std::list<int> nbflynns;
            ElleFlynnNbRegions(j,nbflynns);
            */
			
 	    	unodelist.clear();
        	ElleGetFlynnUnodeList(j,unodelist);
    		count = unodelist.size();
			 
            for (i=0;i<count;i++)   // cycle through unodes
            {
                ElleGetUnodePosition(unodelist[i],&xy);

                if (!EllePtInRegion(j,&xy)) 
                {
                    for (k=0;k<ElleMaxFlynns();k++)
                    {
                        if (ElleFlynnIsActive(k))
                        {  
                            if (EllePtInRegion(k,&xy))
                            {
                                ElleRemoveUnodeFromFlynn(j,unodelist[i]);
                                ElleAddUnodeToFlynn(k, unodelist[i]);
                                //printf("unode %d from flynn %d to flynn %d\n",
                                //unodelist[i],j,k);  
                                reassignedlist.push_back(unodelist[i]);
                                //SetUnodeAttributeFromNbFlynn(unodelist[i],k,EULER_3);
                                break;
                            }		 
                        }	   
                    }
                }	   
            }
	 }
  }
  for (j=0;j<reassignedlist.size();j++) {
    k = ElleUnodeFlynn(reassignedlist[j]);
    SetUnodeAttributeFromNbFlynn(reassignedlist[j],k,EULER_3,reassignedlist);
    // What about DISLOCDEN ?
  }
}

int SetUnodeAttributesFromFile2_UNUSED(const char *fname)
{
	int err=0,opt_1,opt_2, dum1, dum2;
    int id=0, i, j,max_unodes,jj ;
    Coords xy;
	double val[12],dens,iwork;
    
    /*
     * FS: Stuff to exclude some unodes from dislocden update (e.g. flynns with 
     * bubbles should always have DD = 0). The unode will be excluded if the 
     * flynn's ID on this unode > 0 and == userdata[1]
     */
    UserData userdata;
    ElleUserData(userdata);
    int iFlynnID = 0;
    double dFlynnViscValue = 0;
    int iExcludeValue = (int)userdata[1];

	
/*	
printf("\t*** Export data from FFT to ELLE ***\n");	
printf("\t*** using as default tex.out file ***\n");
	
printf("Import data to store in U_ATTRIB_A and U_ATTRIB_B\n");
printf("NON import dislocation density to U_DISLOCDEN !!\n");	
printf("[4] normalized strain rate\n");
printf("[5] normalized stress\n");
printf("[6] activity basal mode\n");
printf("[7] activity prismatic mode\n");
printf("[8] Geometrical density of dislocation ()\n");
printf("[9] Stattistical density of dislocation ()\n");	
printf("[10] identification of Fourier Point \n");	
printf("[11] FFT grain nunmber\n");
printf("?? (exp. 4 5 +<return>)\n");	
	
cin >> opt_1 >> opt_2; 
*/
	
	id=0;
    ifstream datafile(fname);
    if (!datafile) return(OPEN_ERR);
    while (datafile) {
	
datafile>>val[0]>>val[1]>>val[2]>>val[3]>>val[4]>>val[5]>>val[6]>>val[7]>>val[8]>>val[9]>>val[10]>>val[11];	
		// iwork=val[4]*val[5];
			// do somthing with DDs..
            
                // Get old dislocden:
                
                ElleGetUnodeAttribute(id,&dens,U_DISLOCDEN);
                
                /*
                 * FS: Check if flynn VISCOSITY is == userdata[1], if yes 
                 * this area (like a bubble) is excluded from dislocden update 
                 * and gets dislocden = 0
                 */
                if (ElleFlynnAttributeActive(VISCOSITY))
                {
                    iFlynnID=ElleUnodeFlynn(id);
                    ElleGetFlynnRealAttribute(iFlynnID,&dFlynnViscValue,VISCOSITY);
                }     
                
                if (iExcludeValue != 0 && iExcludeValue == (int)dFlynnViscValue)
                {
                    // Exclude Unode from updating the dislocden. 
                    // Insead set added dislocden to 0:
                    //ElleSetUnodeAttribute(id,(dens+0.0),U_DISLOCDEN);
                    // Or maybe better: Set e.g. bubble dislocden to zero:
                    ElleSetUnodeAttribute(id,0.0,U_DISLOCDEN);
                }
                else
                {
                    // Update dislocden like usual                    
                    ElleSetUnodeAttribute(id,(dens+val[8]),U_DISLOCDEN);
                }	
		id++;
    }
    datafile.close();

	max_unodes = ElleMaxUnodes();
    for (i=0;i<max_unodes;i++)   // cycle through unodes	
    {
	    jj=ElleUnodeFlynn(i);
	    ElleSetUnodeAttribute(i,U_ATTRIB_C, double(jj)); 
    }

    return(err);
}

void FS_SetUnodeDD(const char *fname)
{
    /*
     * FS:
     * This is a new version of the code. It checks if unode changed flynn due 
     * to sweeping boundary and assignes correct flynns to unodes WITHOUT 
     * updating U_ATTRIB_C, because this is done by the FS_topocheck code
     * that always should run after FFT.
     */
    int iUnodeID=0;
	double val[12],dDens=0.0;
    
    UserData userdata;
    ElleUserData(userdata);
    int iFlynnID = 0;
    double dFlynnVisc = 0.0;
    int iUpdateDDs = (int)userdata[0];
    int iExcludeValue = (int)userdata[1];

	
/*	
printf("\t*** Export data from FFT to ELLE ***\n");	
printf("\t*** using as default tex.out file ***\n");
	
printf("Import data to store in U_ATTRIB_A and U_ATTRIB_B\n");
printf("NON import dislocation density to U_DISLOCDEN !!\n");	
printf("[4] normalized strain rate\n");
printf("[5] normalized stress\n");
printf("[6] activity basal mode\n");
printf("[7] activity prismatic mode\n");
printf("[8] Geometrical density of dislocation ()\n");
printf("[9] Stattistical density of dislocation ()\n");	
printf("[10] identification of Fourier Point \n");	
printf("[11] FFT grain nunmber\n");
printf("?? (exp. 4 5 +<return>)\n");	
	
cin >> opt_1 >> opt_2; 
*/
	

    ifstream datafile(fname);
    if (!datafile) 
    {
        printf("ERROR (FS_SetUnodeAttributesFromFile): tex.out missing\n");
    }
    else
    {
        if (iUpdateDDs==1)
        {	
            iUnodeID=0;
            while (datafile) 
            {
                datafile >>val[0]>>val[1]>>val[2]>>val[3]>>val[4]>>val[5]
                         >>val[6]>>val[7]>>val[8]>>val[9]>>val[10]>>val[11];		

                // Get old dislocden:
                ElleGetUnodeAttribute(iUnodeID,&dDens,U_DISLOCDEN);

                /*
                * FS: Check if flynn VISCOSITY is == userdata[1], if yes 
                * this area (like a bubble) is excluded from dislocden update 
                * and gets dislocden = 0
                * 
                * THIS IS OLD
                */
                /*!
                if (ElleFlynnAttributeActive(VISCOSITY))
                {
                    iFlynnID=ElleUnodeFlynn(iUnodeID);
                    ElleGetFlynnRealAttribute(iFlynnID,&dFlynnVisc,VISCOSITY);
                }     

                if (iExcludeValue != 0 && iExcludeValue == (int)dFlynnVisc)
                {
                    // Exclude Unode from updating the dislocden. 
                    // Insead set added dislocden to 0:
                    //ElleSetUnodeAttribute(iUnodeID,(dDens+0.0),U_DISLOCDEN);
                    // Or maybe better: Set e.g. bubble dislocden to zero:
                    ElleSetUnodeAttribute(iUnodeID,0.0,U_DISLOCDEN);
                }
                else
                {
                    // Update dislocden like usual                    
                    ElleSetUnodeAttribute(iUnodeID,(dDens+val[8]),U_DISLOCDEN);
                }	
                */
                ElleSetUnodeAttribute(iUnodeID,(dDens+val[8]),U_DISLOCDEN);
                iUnodeID++;
            }
        }
    }
    datafile.close();

    return;
}

void FS_CheckUnodes()
{
    /* This checks if a unode changed its host flynn during the process and
     * updates its DD and euler_3 if needed */    
    int iFlynnId=0;
    Coords cUnodeXY;
    int iMaxUnodes=ElleMaxUnodes();
    int iMaxFlynns=ElleMaxFlynns();
    vector<int> vReAssign;
    vector<int> vUnodeList;
    
    /* STEP1:
     * Check if attributes storing flynn ID are active, if not assign them 
     * according to the actual situation, updates and corrections will follow */
    if (!ElleUnodeAttributeActive(U_ATTRIB_C))
    {
        ElleInitUnodeAttribute(U_ATTRIB_C);
        for (int unode=0;unode<iMaxUnodes;unode++)
        {
            iFlynnId = ElleUnodeFlynn(unode);
            ElleSetUnodeAttribute(unode,U_ATTRIB_C,(double)iFlynnId);
            // This might not be correct yet, but will be updated during this
            // function
        }
    }
    if (!ElleFlynnAttributeActive(F_ATTRIB_C))
    {
        ElleInitFlynnAttribute(F_ATTRIB_C);    
        for (int flynn=0;flynn<iMaxFlynns;flynn++)
            if (ElleFlynnIsActive(flynn))
                ElleSetFlynnRealAttribute(flynn,(double)flynn,F_ATTRIB_C);
    }
    
    /* STEP 2:
     * Go through all unodes and check if they are in the correct flynn */
    for (int unode=0;unode<iMaxUnodes;unode++)
    {
        bool bFound=false; // will be true once correct host flynn is found
        iFlynnId = ElleUnodeFlynn(unode);
        ElleGetUnodePosition(unode,&cUnodeXY);
        if (ElleFlynnIsActive(iFlynnId))
        {
            if (EllePtInRegion(iFlynnId,&cUnodeXY)) 
            {
                bFound=true;
                ElleSetUnodeAttribute(unode,U_ATTRIB_C,(double)iFlynnId); 
            }
        }
        
        if (!bFound)
        {
            /* Need to search for the correct host flynn*/
            for (int flynn=0;flynn<iMaxFlynns;flynn++)
            {
                if (ElleFlynnIsActive(flynn))
                {
                    if (EllePtInRegion(flynn,&cUnodeXY)) 
                    {
                        ElleAddUnodeToFlynn(flynn,unode); 
                        vReAssign.push_back(unode);
                        // To later identify the unodes that need reassignment:
                        ElleSetUnodeAttribute(unode,U_ATTRIB_C,-1.0); 
                        bFound=true;
                        break;
                    }
                }
            }
        }
    }
    //ElleWriteData("test.elle");
    
    /* STEP 3:
     * Re assign unode attributes euler_3 and dislocden: Use euler3 of closest 
     * unode in that flynn and set dislocden to zero */
    if (ElleUnodeAttributeActive(EULER_3))
        if (ElleUnodeAttributeActive(U_DISLOCDEN))
            for (int i=0;i<vReAssign.size();i++)
            {
                FS_ReAssignAttribsSweptUnodes(vReAssign[i]); 
            }
                    
    vReAssign.clear();
}

void FS_ReAssignAttribsSweptUnodes(int iUnodeID)
{
    /* Re assign unode attributes euler_3 and dislocden: Use euler3 of closest 
     * unode in that flynn and set dislocden to zero */
    int iFlynnId=0;    
    double dRoi = FS_GetROI(8);
    double dTest=0.0;
    double dNewEuler[3];
    double dEulerOld[3];
    double dTmpEuler[3];
    for (int ii=0;ii<3;ii++) 
    {
        dNewEuler[ii]=0.0;
        dEulerOld[ii]=0.0;
        dTmpEuler[ii]=0.0;
    }
    double dDensityMin = 0.0; // Set new dislocden to this value, i.e. to zero
    vector<int> vUnodeList;
    Coords cUnodeXY;
    Coords cRefXY;
    
    /* Get info about the unode of interest */
    ElleGetUnodePosition(iUnodeID,&cRefXY);    
    iFlynnId = ElleUnodeFlynn(iUnodeID); // that will be the correct flynn
    vUnodeList.clear();
    ElleGetFlynnUnodeList(iFlynnId,vUnodeList);  
    ElleGetUnodeAttribute(iUnodeID,&dEulerOld[0],E3_ALPHA);
    ElleGetUnodeAttribute(iUnodeID,&dEulerOld[1],E3_BETA);
    ElleGetUnodeAttribute(iUnodeID,&dEulerOld[2],E3_GAMMA);   
    
    /* Go to each unode in this list and check if it is NOT a unode that 
     * still needs to be reassigned:
     * If yes: Search for the closest unode to the unode of interest */
    double dMinDist = 1000000.0;
    double dDist    = 0.0;
    int iCount      = 0;
    int iNbUnode    = 0;
    for (int j=0;j<vUnodeList.size();j++)
    {
        ElleGetUnodeAttribute(vUnodeList[j],&dTest,U_ATTRIB_C);
        if (dTest>=0.0) //U_ATTRIB_C will be -1 if unode was swept
        {
            ElleGetUnodePosition(vUnodeList[j],&cUnodeXY);
            ElleCoordsPlotXY(&cRefXY,&cUnodeXY);			  
            dDist = pointSeparation(&cRefXY,&cUnodeXY);
            
            if (dDist<=dRoi && dDist<dMinDist)
            {
                iCount++;      
                dMinDist=dDist;
                iNbUnode = vUnodeList[j];
            }                                 
        }
    }
    
    if (iCount>0)
    {                            
        /* Found the closest nb unode, store its orientation */
        ElleGetUnodeAttribute(iNbUnode,&dNewEuler[0],E3_ALPHA);
        ElleGetUnodeAttribute(iNbUnode,&dNewEuler[1],E3_BETA);
        ElleGetUnodeAttribute(iNbUnode,&dNewEuler[2],E3_GAMMA);       
    }
    else
    {
        /* No unodes found in roi, use mean value of the whole flynn
         * Only if there are no more unodes in flynn (meaning that 
         * vUnodeList.size()==0) keep old orientation */
        if (vUnodeList.size()==0) // unlikely, but may be possible
        {
            for (int ii=0;ii<3;ii++) dNewEuler[ii] = dEulerOld[ii];
            printf("WARNING (FS_ReAssignAttribsSweptUnodes):\nSetting new ");
            printf("orientation of swept unode %u to old value\n",iUnodeID);
        }
        else
        {
            double dDistTotal=0.0;
            dDist=0.0;
            for (int j=0;j<vUnodeList.size();j++)
            {
                ElleGetUnodeAttribute(vUnodeList[j],&dTest,U_ATTRIB_C);
                if (dTest>=0.0)
                {     
                    ElleGetUnodePosition(vUnodeList[j],&cUnodeXY);	
                    ElleCoordsPlotXY(&cRefXY,&cUnodeXY);			  
                    dDist = pointSeparation(&cRefXY,&cUnodeXY);
                    
                    ElleGetUnodeAttribute(vUnodeList[j],&dNewEuler[0],E3_ALPHA);
                    ElleGetUnodeAttribute(vUnodeList[j],&dNewEuler[1],E3_BETA);
                    ElleGetUnodeAttribute(vUnodeList[j],&dNewEuler[2],E3_GAMMA);
                    dTmpEuler[0] += (dNewEuler[0]*dDist);  
                    dTmpEuler[1] += (dNewEuler[1]*dDist);
                    dTmpEuler[2] += (dNewEuler[2]*dDist); 
                    dDistTotal += dDist;                       
                }             
            }
            // to be on the save side: 
            if (dDistTotal<=0.0) // only one unode in this flynn, which is the swept one, reset to old value
            {
                for (int ii=0;ii<3;ii++) dNewEuler[ii] = dEulerOld[ii];
                printf("WARNING (FS_ReAssignAttribsSweptUnodes):\nSetting ");
                printf("new orientation of swept unode ");
                printf("%u to old value\n",iUnodeID);
                dDistTotal = 1.0;
            }
            else
            {
                printf("WARNING (FS_ReAssignAttribsSweptUnodes):\nSetting ");
                printf("new orientation of swept unode ");
                printf("%u to flynn %u mean value\n",iUnodeID,iFlynnId);
            }
            dNewEuler[0] = dTmpEuler[0]/dDistTotal;
            dNewEuler[1] = dTmpEuler[1]/dDistTotal;
            dNewEuler[2] = dTmpEuler[2]/dDistTotal;
            //ElleSetUnodeAttribute(iUnodeID,U_ATTRIB_B,-1.0);
        }
    }
    /* Set freshly determined euler angles and remaining attributes*/
    ElleSetUnodeAttribute(iUnodeID,E3_ALPHA, dNewEuler[0]);
    ElleSetUnodeAttribute(iUnodeID,E3_BETA, dNewEuler[1]);
    ElleSetUnodeAttribute(iUnodeID,E3_GAMMA, dNewEuler[2]); 
    
    ElleSetUnodeAttribute(iUnodeID,U_DISLOCDEN,dDensityMin);
    ElleSetUnodeAttribute(iUnodeID,U_ATTRIB_C, double(iFlynnId));
    vUnodeList.clear();  
}

/*!
   this fn assumes attr is EULER_3
 */
int SetUnodeAttributeFromNbFlynn(int unum,int flynnid,int attr,vector<int> &rlist)
{
    int count, j, k, id;
	double val[5], dummy;
	double dist, dist_min; 
	vector <int> unodelist; 
	Coords xy,xy_unode; 

 	if (ElleFlynnIsActive(flynnid)) {
    	ElleGetFlynnUnodeList(flynnid,unodelist); // get the list of unodes for a flynn
    	count = unodelist.size();
     
		// change the tracer parameter .. F_ATTRIB_B & U_VISCOSITY ?¿ constant not interval, double convert to int  
		// the tracer layer in gbm is U_ATTRIB_C but in elle2fft & fft2elle is U_ATTRIB_A !! caution unify attrib in the code!!
		// modified trace in next piece of program 
		
			ElleGetUnodePosition(unum,&xy);
        	dist_min=1.0;
			id=-1;	
			
		 	for (k=0; k<count; k++) {
              if (unodelist[k]!=unum &&
                                (find(rlist.begin(),rlist.end(),unodelist[k])==rlist.end())) {
				ElleGetUnodePosition(unodelist[k],&xy_unode);
				ElleCoordsPlotXY (&xy, &xy_unode);				  
        		dist = pointSeparation(&xy,&xy_unode);
				
	        	if (dist<dist_min) { // option interpolate... quaternions 
					id = k;
					dist_min=dist;
				}
			  }
			}
		
		  if (id!=-1) {
                ElleGetUnodeAttribute(unum,&val[0],&val[1],&val[2],EULER_3);
			  	//printf ("check_error setting unode id %d (%lf %lf %lf) ",
                         //unum,val[0],val[1],val[2]); 
                //  get the unode attributes
                ElleGetUnodeAttribute(unodelist[id],&val[0],&val[1],&val[2],EULER_3);
			    // set new unode values
				ElleSetUnodeAttribute(unum,val[0],val[1],val[2],EULER_3);
			  	//printf ("to values for %d (%lf %lf %lf) \n", 
                       //unodelist[id],val[0],val[1],val[2]); 
		}

                 else {
                        printf ("check_error NOT setting unode attr %d for flynn %d\n",
                         unum,flynnid);
                 }
 unodelist.clear(); 
  }
  
 }
int RemoveNodeAttributes(vector<int> & attriblist)
{
    int err=0;
    int i,j;
	
	for (j=0; j<attriblist.size();j++) {
        // do we care if indx==NO_INDX ie attribute not found?
		int indx=ElleRemoveNodeAttribute(attriblist[j]);
	}
    return(err);
}

/*
 * This function is used to generate the data for the passive marker grids and
 * store it in U_FINITE_STRAIN CURR_S_X and Y 
 */
void FS_SetEunodeStrain(vector<int> & attriblist)
{
    int max_unodes;
    
    double offset = dd[2][0];
    double xstrain = dd[1][0]; // not used at the moment
    double ystrain = dd[1][1];
    
    int iCoordOnOtherSideY, iCoordOnOtherSideX;
    
    double straintmp[3], dist, dist_total;
                                                                                
    double val[2], roi;
    Coords ptxy, refxy;
                                                                   
    max_unodes = ElleMaxUnodes(); // maximum unode number used    
    
    printf("Setting Eunode Strain - Dimension %i\n",(int)sqrt(max_unodes));
    
    /*
     * FS: The product of boxwidth + boxheight will not remain constant in a 
     * pure shear simulation. Hence, a more accurate determination of ROI is 
     * used here using not sqrt(1.0 / (...), but sqrt(width*height / (...)
     * --> FOR THIS APPROACH THE BOX SHOULD NOT BE A PARALLELOGRAM ETC
     */  
     
    //roi = sqrt(dBoxArea = 1/(double)max_unodes/3.142)*3; // VIP, ROI controls number of unodes used to interpolate the local displacement field

    roi = FS_GetROI(3);    
	
    /*
     * Update positions from previous deformation step as stored in 
     * U_FINITE_STRAIN --> i.e.: Set current position to previous position or 
     * initialize U_FINITE_STRAIN
     */
    FS_ReadEunodeStrain();

    /*
     * Cycle throug all extra unodes: Read old and calculate and set new finite
     * position
     */       
    
    double dEunodesxy[2];

	for (int k=0; k<max_unodes; k++)
    {
        straintmp[0] = straintmp[1] = dist_total = 0;
        
        ElleGetUnodeAttribute(k,PREV_S_X,&dEunodesxy[0]); 
        ElleGetUnodeAttribute(k,PREV_S_Y,&dEunodesxy[1]); 
        
        ptxy.x = dEunodesxy[0];
        ptxy.y = dEunodesxy[1];

		// Cycle through all unodes to find the ones in region of interest (roi):
		for (int i=0;i<max_unodes;i++)
        {
            ElleGetUnodePosition(i,&refxy);  
            iCoordOnOtherSideX = 0;
            iCoordOnOtherSideY = 0;
			FS_CoordsPlotXY(&refxy,&ptxy,&iCoordOnOtherSideX,&iCoordOnOtherSideY);
			dist = pointSeparation(&refxy,&ptxy);      
			if (dist<roi) 
            {
                ElleGetUnodeAttribute(i,&val[0],attriblist[0]);
                ElleGetUnodeAttribute(i,&val[1],attriblist[1]);
                
                /* FS: 
                 * The next step is important: If a unode 
                 * its neighbour are on opposite sides of the box in 
                 * y-direction, a correction of x-strain is needed in simple or
                 * correction of y-strain in pure shear is needed in order to
                 * avoid the errors on top and bottom bow the ugrid image.
                 * "offset" or "ystrain,xstrain" have to be either subtracted or added 
                 * depending on where the neighbour unode was (e.g. on other 
                 * side in x or y direction): This is controlled by 
                 * "iCoordOnOtherSideY and X" being either +1 or -1
                 */                
                if (iCoordOnOtherSideY != 0) 
                {
                    val[0] = val[0]+(offset*iCoordOnOtherSideY); // offset is always 0 in pure shear 
                    val[1] = val[1]+(ystrain*iCoordOnOtherSideY); // ystrain is always 0 in simple shear
                }               
                if (iCoordOnOtherSideX != 0) 
                {
                    val[0] = val[0]+(xstrain*iCoordOnOtherSideX); // xstrain is always 0 in simple shear 
                }
                straintmp[0] +=  val[0]*(roi-dist);
                straintmp[1] +=  val[1]*(roi-dist);
                dist_total += (roi-dist);
			}
        }	
        
        straintmp[0] /= dist_total;
        straintmp[1] /= dist_total;
        // Set new current unode position for finite strain:
        ElleSetUnodeAttribute(k,dEunodesxy[0]+straintmp[0],CURR_S_X);
        ElleSetUnodeAttribute(k,dEunodesxy[1]+straintmp[1],CURR_S_Y);	               
    }    
}

double FS_GetROI(int iFact)
{
    /*
     * FS: The product of boxwidth + boxheight will not remain constant in a 
     * pure shear simulation and unode distances change. Hence, a more accurate 
     * determination of ROI is used here using not sqrt(1.0 / (...), but 
     * sqrt(width*height / (...)
     * --> FOR THIS APPROACH THE BOX SHOULD NOT BE A PARALLELOGRAM ETC
     *  --> Added: Well actually it doesn't matter if the box has simple shear 
     *             component (if it is a parallelogram) --> the height and width
     *             as calculated here will stay the same anyway 
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

/*
 * Read U_FINITE_STRAIN from unodes and shift current x and y to previous x and
 * y. Set current to 0 for the moment, it will be updated by FS_SetEunodeStrain.
 * SPECIAL CASE: If U_FINITE_STRAIN is not active it will be initiated and 
 * default values will be:
 * starting position: Position of the unode in a regular grid of size dim x dim
 * previous position = starting position
 * current position = 0,0
 */
void FS_ReadEunodeStrain()
{
    
    /* If strain attribute is not active at the moment initialize it and 
     * fill with starting values*/
    if (!ElleUnodeAttributeActive(U_FINITE_STRAIN))
    {
        ElleInitUnodeAttribute(U_FINITE_STRAIN);
        
        /* It is necessary to determine box size of the model to correctly set 
         * the initial passive marker positions */
        CellData unitcell;
        ElleCellBBox(&unitcell);
        double dBoxHeight = 0.0;
        double dBoxWidth = 0.0;
        dBoxHeight = unitcell.cellBBox[TOPLEFT].y - unitcell.cellBBox[BASELEFT].y;
        dBoxWidth = unitcell.cellBBox[BASERIGHT].x - unitcell.cellBBox[BASELEFT].x;
            
        UserData userdata;
        ElleUserData(userdata);
        
        int iDim = (int)userdata[2];
        int iUnodeID = 0;
        double dStartPos[2];
        
        for (int j=0;j<iDim; j++) 
        {
			for (int i=0;i<iDim; i++) 
            {
                dStartPos[0]=dStartPos[1]=0.0;
                
				dStartPos[0]= i*(dBoxWidth/(double)iDim);
				dStartPos[1]= j*(dBoxHeight/(double)iDim);
                
                ElleSetUnodeAttribute(iUnodeID,START_S_X,dStartPos[0]);
                ElleSetUnodeAttribute(iUnodeID,START_S_Y,dStartPos[1]);
                ElleSetUnodeAttribute(iUnodeID,PREV_S_X,dStartPos[0]);
                ElleSetUnodeAttribute(iUnodeID,PREV_S_Y,dStartPos[1]);
                ElleSetUnodeAttribute(iUnodeID,CURR_S_X,0.0);
                ElleSetUnodeAttribute(iUnodeID,CURR_S_Y,0.0);
                
				iUnodeID ++;
			}
		}
    }
    else // U_FINITE_STRAIN is active: Shift current to previous position
    {
        double dPosition[2];
        
        for (int i=0;i<ElleMaxUnodes();i++)
        {
            dPosition[0]=dPosition[1]=0.0;
            
            ElleGetUnodeAttribute(i,CURR_S_X,&dPosition[0]);
            ElleGetUnodeAttribute(i,CURR_S_Y,&dPosition[1]);
            
            ElleSetUnodeAttribute(i,PREV_S_X,dPosition[0]);
            ElleSetUnodeAttribute(i,PREV_S_Y,dPosition[1]);
            ElleSetUnodeAttribute(i,CURR_S_X,0.0);
            ElleSetUnodeAttribute(i,CURR_S_Y,0.0);            
        }
    }
}

/*
 * Read timestep that has been set in ppc.in
 */
double ReadPPCInTimestep(char *fnameppcin)
{
    double dTimestepPPCIN = 0.0;
    
    string line;
    stringstream linestr;
    bool bReadData = false;
    
    /*
     * Read FFT timestep from ppc.in
     */
    ifstream ppcfile(fnameppcin);
    
    if (!ppcfile)
    {
        printf("ERROR: Missing file ppc.in to read FFT timestep\n");
        return 0;
    }
    else
    {
        while(getline(ppcfile,line))
        {
            if (bReadData)
            {
                // Read timestep from FFT input file called ppc.in
                linestr << line;
                linestr >> dTimestepPPCIN;
                linestr.clear();
                //printf("ppc in timestep: %e\n",dTimestepPPCIN);
                bReadData = false;                    
            }
            if (line.find("* other") != string::npos)
                bReadData = true;
        }
        ppcfile.close();
    }        
    
    return (dTimestepPPCIN);
}

/*
 * Difference to standard Elle function: It outputs 1 or -1 if the coordinates 
 * are on opposite sides of the unit cell in either x-or y -direction.
 * In y-direction:
 * In simple shear this would require correction by subtracting 
 * or adding "offset" to the incremental strain in x-direction and adjustments 
 * in ystrain for pure shear
 * 
 * In x-direction:
 * Requires adjustments of xstrain in pure shear
 */
void FS_CoordsPlotXY(Coords *xy, Coords *prevxy,int *iOnOtherSideX,int *iOnOtherSideY)
{
    int cnt;
    double unitsize_x,unitsize_y;
    double deformx, deformy;
    CellData unitcell;

    ElleCellBBox(&unitcell);
    deformx = unitcell.cellBBox[TOPLEFT].x-unitcell.cellBBox[BASELEFT].x;
    deformy = unitcell.cellBBox[BASELEFT].y-unitcell.cellBBox[BASERIGHT].y;
    /* 
     * assumes that the unit cell remains a parallelogram
     * (simple shear ?)
     * assuming yoffset is zero
     * unitcell.xoffset is the simple shear (x) + any cell deformation
     */
    unitsize_x = unitcell.xlength;
    unitsize_y = unitcell.ylength;
    
    // Y-component
    
    if ((xy->y - prevxy->y) >= unitsize_y*MAX_SIZE) 
    {
        xy->y -= unitsize_y;
        xy->x -= unitcell.xoffset;
        while ((xy->y - prevxy->y) >= unitsize_y*MAX_SIZE) 
        {
            xy->y -= unitsize_y;
            xy->x -= unitcell.xoffset;
        }
        *iOnOtherSideY = -1;
    }
    else if ((xy->y - prevxy->y) < -unitsize_y*MAX_SIZE) 
    {
        xy->y += unitsize_y;
        xy->x += unitcell.xoffset;
        while ((xy->y - prevxy->y) < -unitsize_y*MAX_SIZE) 
        {
            xy->y += unitsize_y;
            xy->x += unitcell.xoffset;
        }
        *iOnOtherSideY=1;
    }
    
    // X-component
    
    if ((xy->x - prevxy->x) >= unitsize_x*MAX_SIZE) 
    {
        xy->x -= unitsize_x;
        xy->y -= unitcell.yoffset;
        while ((xy->x - prevxy->x) >= unitsize_x*MAX_SIZE) 
        {
            xy->x -= unitsize_x;
            xy->y -= unitcell.yoffset;
        }
        *iOnOtherSideX = -1;
    }
    else if ((xy->x - prevxy->x) < -unitsize_x*MAX_SIZE) 
    {
        xy->x += unitsize_x;
        xy->y += unitcell.yoffset;
        while ((xy->x - prevxy->x) < -unitsize_x*MAX_SIZE) 
        {
            xy->x += unitsize_x;
            xy->y += unitcell.yoffset;
        }
        *iOnOtherSideX = 1;
    }

}
