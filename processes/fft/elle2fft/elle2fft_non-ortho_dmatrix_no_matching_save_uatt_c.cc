#include <string>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <iomanip>
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
#include "log.h"
#include "setup.h"
#include "triattrib.h"
#include "unodes.h"
#include "polygon.h"
#include "elle2fft_beta.h"
#include "mat.h"
#define PI 3.141592654

using std::list;
using std::vector;
using std::ios;
using std::cout;
using std::list;
using std::vector;
using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::string;
using std::setw;
using std::setprecision;
using std::setfill;

#define DEFAULT_NAME "make.out"
#define DEFAULT_NAME2 "temp.out"

int InitElle2PPC(), Elle2PPC();
int FindSQParams( int *numperrow, double *dx, double *dy, Coords *origin);
int FindUnode(Coords *ptxy, Coords *origin,
              double dx, double dy,
              int numperrow);
// int FindUnode(Coords *ptxy, int start);
int FindUnode_alb(Coords *ptxy, int start);
void check_error();
// double **dmatrix_alb(long nrl,long nrh,long ncl,long nch); using dmatrix in mat.h
double		**pts=0;		/* memory array */
void nrerror(char error_text[]);


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
    ElleSetInitFunction(InitElle2PPC);

    if (err=ParseOptions(argc,argv))
        OnError("",err);
    ElleSetSaveFrequency(1);
	
	/*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("inifft");

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

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
int InitElle2PPC()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(Elle2PPC);

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
		    if (!ElleUnodeAttributeActive(ATTRIB_A))
            ElleInitUnodeAttribute(U_ATTRIB_A);
		    if (!ElleUnodeAttributeActive(DISLOCDEN))
            ElleInitUnodeAttribute(U_DISLOCDEN);
		    if (!ElleUnodeAttributeActive(ATTRIB_C))
            ElleInitUnodeAttribute(U_ATTRIB_C);			
    }
}

/*!
 * Write the input file for PPC (fft)
 * Format is:
 * int (no. of grains)
 * ** one line per grain with the following format:
 * dbl dbl dbl dbl int int (average properties of each grain and grain id)
 * ** one line per fourier point with the following format:
 * dbl dbl dbl int int int int int(euler angles, row, col, depth of
 * each pt, grain id, phase id)
 */
int Elle2PPC()
{
    int err=0, max;
    bool sq_grid = true;
    int i, j, n, jgrain;
    int r=0, c=0, rows, cols, cstep=1, spacing=1;
    int elle_cols;
    int grain_count;
    double val[5], jphase;

    double dx,dy,elle_dx,elle_dy;
    Coords xy,fft_xy,origin;
    CellData unitcell;
	int max_unodes,max_flynns, SQ_pattern,rad_euler;
	// extern VoronoiData *V_data;	// Voronoi version
	UserData userdata;
	
	/*
 	* indices for User data values for this process
 	*/
	
	ElleUserData(userdata);
 	rows=(int)userdata[N_rows];
	SQ_pattern=(int)userdata[SQ_patterns]; // a value of 0 force irregular pattern of unodes, to avoid error with FindSQ to models with low resolution
    rad_euler=(int)userdata[Rad_euler]; // a value of 0 euler angles in input elle file are in radians
	
	cols=rows;
	// check_error(); // Fix errors of PIC, GBM, modify to layer u_attrib_c 

	printf("cols %d\n", cols);	
    fft_xy.x = 0.0;
    fft_xy.y = 0.0;
     ElleCellBBox(&unitcell);
    dx = unitcell.xlength/cols;
    dy = unitcell.ylength/rows;

	// allocating array memory 
	pts=dmatrix(0,rows*cols-1,0,7);
	int pts_i[rows*cols];

    if ( FindSQParams( &elle_cols, &elle_dx, &elle_dy, &origin ) != 0 )
        sq_grid=false;
	if ( SQ_pattern != 1) sq_grid=false;
    if (!sq_grid) {
        Log(0,"Does not look like square grid. Using general algorithm - be patient");
       // err=ElleVoronoiUnodes(); // version using voronois tessalation 
    }
	printf("Now, start to calculate fft points\n");
	
	if (ElleSSOffset() != 0.0) Log(0,"Export to a non orthogonal grid ");  

    for (n=0,r=0;r<rows;r++) {   // cycle through fft points
        fft_xy.x = 0.0;
		
		if (ElleSSOffset() != 0.0) fft_xy.x = fft_xy.x + ElleSSOffset()*fft_xy.y;
		 
        for (c=0;c<cols;c++) {
			// printf("r and c: %d\t%d\n",r,c);
			
            if (!sq_grid) i=FindUnode_alb(&fft_xy, 0);
            else i=FindUnode(&fft_xy, &origin, elle_dx, elle_dy, elle_cols);	
	
              if (i!=NO_VAL) {
                //  get the unode attribute
                ElleGetUnodeAttribute(i,&val[0],&val[1],&val[2],EULER_3);
				  
                pts[n][0] = val[0];
	            pts[n][1] = val[1];
				pts[n][2] = val[2];
				pts[n][3] = (double)ElleUnodeFlynn(i);
				ElleGetUnodeAttribute(i,&val[4],U_DISLOCDEN);
				pts[n][7] = val[4];
				  
				// get the unode flynn_id
       			jgrain = ElleUnodeFlynn(i);
				ElleGetFlynnRealAttribute(jgrain,&jphase,DISLOCDEN); // Flynn dislocation density is used as phase indicative  
				pts_i[n]=(int)jphase;
				  
				// rad2degrees 
				if (rad_euler == 0) {
					pts[n][0]=pts[n][0]*180/PI;
					pts[n][1]=pts[n][1]*180/PI;
                	pts[n][2]=pts[n][2]*180/PI;
				}					
					
            }
              else OnError("No matching unode",0);
				  
            fft_xy.x += dx;
            n++;			
        }
        fft_xy.y += dy;
		printf("r : %d\n",r);
    }
	
    // Open the file
    string outfilename(ElleOutFile());
    if (outfilename.length()==0) {
		outfilename = DEFAULT_NAME;
	}
	grain_count = ElleNumberOfGrains();
    vector<int> flynn_ids(grain_count);

    ofstream outf(outfilename.c_str());
    outf << setfill(' ') << setw(10) << setprecision(5);

    // Write the first line
    if (outf) outf << grain_count << endl;

    // Write the first block
    max = ElleMaxFlynns();
    for (i=0,j=0; i<max && outf; i++) {
        if (ElleFlynnIsActive(i)) {
    		ElleGetFlynnEuler3(i,&val[0],&val[1],&val[2]);
			outf<<val[0]<<'\t'<<val[1]<<'\t'<<val[2]<<'\t'<<val[0]<<'\t'
				<<'0'<<'\t'<<i<<endl;
            flynn_ids[j] = i;
            j++;
        }
	}
	
    // Write the second block
    for (r=1;r<=rows;r++) {   // cycle through fft points
        i=(r-1)*cols;
        for (c=1;c<=cols;c++,i++) {
            outf << pts[i][0] <<'\t'
                     << pts[i][1] <<'\t'
                     << pts[i][2] <<'\t'
            		 << c << '\t'
                     << r <<'\t'
                     << '1' << '\t';
			j=0;
			pts[i][4]= (c-1)*unitcell.xlength/rows;
			pts[i][5]= (r-1)*unitcell.ylength/cols;			
            while (flynn_ids[j]!=(int)pts[i][3]) j++;
            outf << j+1 <<'\t' << pts_i[i] <<endl;
			pts[i][6]=j+1;
        }
    }
    outf.close();
		
//Reassign unodes values using fft points calculated 
    max_flynns = ElleMaxFlynns(); // maximum flynn number used
    max_unodes = ElleMaxUnodes(); // maximum unode number used

       for (j=0;j<max_flynns;j++) // cycle through flynns
	   {
       if (ElleFlynnIsActive(j))  // process if flynn is active 
		{
             for (i=0;i<max_unodes;i++){
				xy.x = pts[i][4]+ElleSSOffset()*pts[i][5]; // to modify.. .. ..phi::  shear offset 
				xy.y = pts[i][5];
				ElleSetUnodePosition(i,&xy);
				ElleSetUnodeAttribute(i,pts[i][0],pts[i][1],pts[i][2],EULER_3);
				ElleSetUnodeAttribute(i,pts[i][6],U_ATTRIB_A);
				ElleSetUnodeAttribute(i,pts[i][7],U_DISLOCDEN);
			 }
		 }
        }

// Assign U_ATRIB_C 
// Added to be compatible with new version of gbm using unodes
// full version with check of uncoherent unodeslist with flynn
	
	int ii,jj;
		
    for (i=0;i<max_unodes;i++)   // cycle through unodes
    {
	    jj=ElleUnodeFlynn(i);
		
 		 if(ElleFlynnIsActive(jj)) ElleSetUnodeAttribute(i,U_ATTRIB_C, double(jj));
		 else {
			ii=ElleUnodeFlynn(i);
       		ElleGetUnodePosition(i,&xy);
			ElleNodeUnitXY(&xy); // plot in unitcell?¿ 
			printf (" %e %e\n", xy.x, xy.y);
			
	   		for (j=0;j<ElleMaxFlynns();j++) 
   			{
      			if (ElleFlynnIsActive(j)) 
	  			{  
         			if (EllePtInRegion(j,&xy)) 
		 			{
						jj=j;
						ElleAddUnodeToFlynn(jj, i);
						ElleSetUnodeAttribute(i,U_ATTRIB_C, double(jj));	
						break;
					}		 
	     		}	   
      		}
	    printf(" jj ii %i  %i\n", jj, ii);	  
  	   }
 	}
  
        ElleUpdate();
	
// desallocating array memory 	
	// free_dmatrix(pts,0,rows,0,cols);
	// pts=0;
	
 // rewrite temp.out file 
	ofstream outf2("temp.out");
	outf2 <<  unitcell.xlength <<'\t'<< unitcell.ylength <<'\t'<< '1' << endl;
	outf2 << "0 0 0"<< endl;
	outf2 << ElleSSOffset() << endl;
    outf2.close();	 
	
}

/*!
 * Assumes SQ_GRID spatial distribution of unodes
 */
int FindSQParams( int *numperrow, double *dx, double *dy, Coords *origin)
{
    int err=0;
    int i, j;
    int max_unodes = ElleMaxUnodes();
    Coords refxy, xy;
    double eps;
                                                                                
    ElleGetUnodePosition(0,&refxy);
    ElleGetUnodePosition(1,&xy);
    *dx = (xy.x-refxy.x);
    eps = *dx*0.1;
    *numperrow=1;
    while (*numperrow<max_unodes && (fabs(xy.y-refxy.y)<eps)) {
        (*numperrow)++;
        ElleGetUnodePosition(*numperrow,&xy);
    }
    if (*numperrow<max_unodes) {
        *dy = xy.y-refxy.y;
        origin->x = refxy.x - *dx/2;
        origin->y = refxy.y - *dy/2;
    }
    i=0;
    while (i<max_unodes && !err) {
        i++;
        for (j=1;j<*numperrow && !err;j++,i++) {
            ElleGetUnodePosition(i,&xy);
            if (fabs(xy.y-refxy.y)>eps || fabs(xy.x-refxy.x-*dx)>eps)
                err = 1;
            refxy = xy;
        }
        refxy.x = origin->x + *dx/2;
        refxy.y += *dy;
    }
    return (err);
}
                                                                                
int FindUnode(Coords *ptxy, Coords *origin,
              double dx, double dy,
              int numperrow)
{
    int i, j, id=NO_VAL;
    int max_unodes = ElleMaxUnodes();
                                                                                
    i = (int) ((ptxy->y-origin->y)/dy -0.5);
    j =  (int) ((ptxy->x-origin->x)/dx -0.5);
    if ((i*numperrow + j)<max_unodes) id = i*numperrow + j;
    return(id);
}
                                                                                
/*!
 * No assumptions about the spatial distribution of unodes
 */
/*
int FindUnode(Coords *ptxy, int start)
{
    int i, j, id=NO_VAL, unode_vpts;
    int max_unodes = ElleMaxUnodes();
    Coords rect[4];
    Coords *bndpts, refxy, xy;
    double roi = ElleUnodeROI();
	extern VoronoiData *V_data;
	
    rect[0].x = ptxy->x - roi;
    rect[0].y = ptxy->y - roi;
    rect[1].x = ptxy->x + roi;
    rect[1].y = ptxy->y - roi;
    rect[2].x = ptxy->x + roi;
    rect[2].y = ptxy->y + roi;
    rect[3].x = ptxy->x - roi;
    rect[3].y = ptxy->y + roi;
   // cycle through unodes
    for (i=start;i<max_unodes && id==NO_VAL;i++)
    {
        ElleGetUnodePosition(i,&refxy);
        ElleCoordsPlotXY(&refxy,ptxy);
        if (EllePtInRect(rect,4,&refxy)) {
            list<int> pt_list;
            list<int>::iterator it;
            ElleUnodeVoronoiPts(i,pt_list);
            unode_vpts=pt_list.size();
            bndpts = new Coords[unode_vpts];
            for (j=0,it=pt_list.begin();it!=pt_list.end();it++,j++) {
                // xy = V_data->vpoints[*it];
				V_data->vpoints[*it].getPosition(&xy);
                ElleCoordsPlotXY(&xy,&refxy);
                bndpts[j] = xy;
            }
            if (EllePtInRegion(bndpts,unode_vpts,ptxy)) {
                id = i;
            }
            delete [] bndpts;
        }
		//printf("unode number i: %d\n",i);
    }
    return(id);
}
*/

int FindUnode_alb(Coords *ptxy, int start)
{
    int i, k, id=NO_VAL;
    int max_unodes = ElleMaxUnodes();
    Coords rect[4];
    Coords *bndpts, refxy, xy, xy_unode;
    double dist, dist_min=10;
	// extern VoronoiData *V_data;
	vector <int> unodelist;

    // ElleNodeUnitXY(&ptxy); Point relative to unit cell
	
   for (i=0;i<ElleMaxFlynns();i++) 
   {
      if (ElleFlynnIsActive(i)) 
	  {  
		 ElleNodeUnitXY(ptxy);
         if (EllePtInRegion(i,ptxy)) 
		 {
			ElleGetFlynnUnodeList(i,unodelist);
			for(k=0;k<unodelist.size();k++) 
			{
                ElleGetUnodePosition(unodelist[k],&xy_unode);	 
				ElleCoordsPlotXY (&xy_unode, ptxy);			  
                dist = pointSeparation(ptxy,&xy_unode);

                if (dist<dist_min) 
				{
					id = unodelist[k];
					dist_min=dist;
				}
		     }			 
	     }	   
      }	  
  }

	if (id ==NO_VAL) {
	dist_min=10;	
	// alternative, scan all the unodes
	for(k=0;k<max_unodes;k++)
	{
		ElleNodeUnitXY(ptxy);
		ElleGetUnodePosition(k,&xy_unode);	 
		ElleCoordsPlotXY (&xy_unode, ptxy);			  
        dist = pointSeparation(ptxy,&xy_unode);

                if (dist<dist_min){
					id = k;
					dist_min=dist;
					}
		}
printf("No matching unode %d\n",id);
	}
    return(id);
}

/*
double **dmatrix_alb(long nrl,long nrh,long ncl,long nch)
{
	long i;
	double **m;

	m=(double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
	if (!m) nrerror("allocation failure 1 in dmatrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(double*) malloc((unsigned) (nch-ncl+1)*sizeof(double));
		if (!m[i]) nrerror("allocation failure 2 in dmatrix()");
		m[i] -= ncl;
	}
	return m;
}

void nrerror(char error_text[])
{
	printf("Numerical Recipes run-time error...\n");
	printf("%s\n",error_text);
	printf("...now exiting to system...\n");
	exit(1);
}
*/
