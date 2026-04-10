#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <math.h>
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "parseopts.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "setup.h"
#include "triattrib.h"
#include "unodes.h"
#include "polygon.h"
#include "crossings.h"
#include "check.h"

using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;

int InitMoveBnodes(), MoveBnodes();
void SetBnodeStrain(vector<int> & attriblist);
void PositionFromStrain(vector<int> &attriblist);
int SetUnodeAttributesFromFile(const char *fname,vector<int> &attribs);
int SetUnodeStrainFromFile(const char *fname,vector<int> &attribs);
void ResetCell(double xstrain, double ystrain, double zstrain, double ssoffset);
int RemoveNodeAttributes(vector<int> & attriblist);
int LoadDataTemp(const char *fname);
double dd[3][3];
//int SetUnodeAttributeFromNbFlynn(int unum,int flynnid,int attr);
int SetUnodeAttributeFromNbFlynn(int unum,int flynnid,int attr,vector<int> &rlist);
void check_error();

int SetUnodeAttributesFromFile2(const char *fname);

main(int argc, char **argv)
{
    int err=0;
    UserData udata;

    /*
     * initialise
     */
    ElleInit();

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
    ElleSetSaveFileRoot("fft2elle");

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
     * Save the unode positions increments from the fft output
     *     in attribs
     * (if unodexyz.out was the strain increment, we could call
           SetUnodeAttributesFromFile("unodexyz.out",attribs);)
     */
    err = SetUnodeStrainFromFile("unodexyz.out",attribs);
    if (err) OnError("unodexyz.out",err);
		
     err = LoadDataTemp("temp.out");
     if (err) OnError("temp.out",err);	
     printf("%lf\t%lf\t%lf\n",dd[1][0],dd[1][1],dd[1][2]);
	 printf("shear strain increment %lf\n", dd[2][0]);

	
    /*
     * Calculate the strain at the bnodes, using the 
     * the unode strain values from each flynn neighbour
     * of the bnode
     */
     SetBnodeStrain(attribs);
	 
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
	if (import_DD == 1) {
        if (!ElleUnodeAttributeActive(U_DISLOCDEN))	
	 		ElleInitUnodeAttribute(U_DISLOCDEN);
		if (!ElleUnodeAttributeActive(U_ATTRIB_C)) {
            ElleInitUnodeAttribute(U_ATTRIB_C);
			U_attrib_init.push_back(U_ATTRIB_C);
		}
		
		err = SetUnodeAttributesFromFile2("tex.out"); //file is tex.out
    	if (err) OnError("tex.out",err);
	}
	/*
	* check if unodes changes of flynn 
	*/
	check_error();
	
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
    CellData unitcell;
    for (i=0, count=0;i<attribs.size();i++) if (attribs[i]!=NO_VAL) count++;
    if (count<1) return (ATTRIBID_ERR);
    double val[3];
    ifstream datafile(fname);
    if (!datafile) return(OPEN_ERR);
    if (!ElleUnodeAttributeActive(attribs[0]))
        ElleInitUnodeAttribute(attribs[0]);
    if (!ElleUnodeAttributeActive(attribs[1]))
        ElleInitUnodeAttribute(attribs[1]);
    ElleCellBBox(&unitcell);
    while (datafile) {
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
LE
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
    roi = sqrt(1.0/(double)max_unodes/3.142)*3; // VIP, ROI controls number of unodes used to interpolate the local displacement field
		
    for (i=0;i<max_bnodes;i++)   // cycle through bnodes
    {

		trace=0;
		count=0;
		
        if(ElleNodeIsActive(i)){
					numbnodes++;
          e[0] = e[1] = dist_total = 0.0;
		  ElleNodePosition(i,&ref);
          ElleRegions(i,rgns); //find the neighbour flynns
          for(j=0;j<3;j++) {
            if (rgns[j] != NO_NB) {
              unodelist.clear();
              ElleGetFlynnUnodeList(rgns[j],unodelist);//get the list of unodes
              for(k=0;k<unodelist.size();k++) {
                ElleGetUnodePosition(unodelist[k],&xy);
                //Need to adjust val[] if using wrapped unode
                //ElleCoordsPlotXY(&xy,&ref);
	            ElleGetUnodeAttribute(unodelist[k],&val[0],attriblist[0]);
                ElleGetUnodeAttribute(unodelist[k],&val[1],attriblist[1]);
		
                //Need to adjust val[] if using wrapped unode
                //ElleCoordsPlotXY(&xy,&ref);
                dist = pointSeparation(&ref,&xy);

                if (dist<roi) {
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
                //ElleCoordsPlotXY(&xy,&ref);
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
    corners[TOPLEFT].y = unitcell.cellBBox[TOPRIGHT].y+ystrain;
	
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


void check_error()
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

        if (!EllePtInRegion(j,&xy)) {
		  	for (k=0;k<ElleMaxFlynns();k++){
      	   		if (ElleFlynnIsActive(k)){  
         	 		if (EllePtInRegion(k,&xy)){
                        ElleRemoveUnodeFromFlynn(j,unodelist[i]);
   			ElleAddUnodeToFlynn(k, unodelist[i]);
  				printf("unode %d from flynn %d to flynn %d\n",
                               unodelist[i],j,k);  
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

int SetUnodeAttributesFromFile2(const char *fname)
{
	int err=0,opt_1,opt_2, dum1, dum2;
    int id=0, i, j,max_unodes,jj ;
    Coords xy;
	double val[12],dens,iwork;

	
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
				ElleGetUnodeAttribute(id,&dens,U_DISLOCDEN);	
				ElleSetUnodeAttribute(id,(dens+val[8]),U_DISLOCDEN);	
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
			  	printf ("check_error setting unode id %d (%lf %lf %lf) ",
                         unum,val[0],val[1],val[2]); 
                //  get the unode attributes
                ElleGetUnodeAttribute(unodelist[id],&val[0],&val[1],&val[2],EULER_3);
			    // set new unode values
				ElleSetUnodeAttribute(unum,val[0],val[1],val[2],EULER_3);
			  	printf ("to values for %d (%lf %lf %lf) \n", 
                       unodelist[id],val[0],val[1],val[2]); 
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
