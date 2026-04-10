 /*****************************************************
 * Copyright: (c) 2009 L. A. Evans, A. Griera
 * File:      $RCSfile: elle2fft.cc,v $
 * Revision:  $Revision: 1.8 $
 * Date:      $Date: 2014/05/02 07:21:54 $
 * Author:    $Author: levans $
 *
 * Elle Project Software
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA
 ******************************************************/
 /*
  * ATTENTION:
  * Edited by Florian Steinbach NOT to regard any flynns in make.out file
  * --> turned out to be useful for code stability
  */
  bool bFS_ExcludeFlynns = true; // set to true to disregard flynns
#include <string>
#include <iostream>
#include <fstream>
#include <climits>
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
#include "convert.h"
#include "mat.h"
#include "FS_elle2fft.h"

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

int InitElle2fft(), Elle2fft();
int FindSQParams( int *numperrow, double *dx, double *dy, Coords *origin);
int SetUnodeAttributeFromNbFlynn(int unum,int flynnid,int attr,
                                vector<int> &rlist);

main(int argc, char **argv)
{
    int err=0;
    UserData userdata;

    /*
     * initialise
     */
    ElleInit();

    ElleUserData(userdata);
    userdata[N_rows]=256; // default number of rows
    userdata[SQ_patterns]=0; // default grid pattern is square (overwritten in code)
    userdata[Rad_euler]=1; // default for input in degrees
    ElleSetUserData(userdata);

    ElleSetOptNames("Rows","GridPattern",
                    "Rad_Euler","unused",
                    "unused","unused","unused",
                    "unused","unused");

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitElle2fft);

    if (err=ParseOptions(argc,argv))
        OnError("",err);
    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("elle2fft");

    ElleSetSaveFrequency(1);

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
int InitElle2fft()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(Elle2fft);

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
        if (!ElleFlynnAttributeActive(VISCOSITY))
        ElleInitFlynnAttribute(VISCOSITY);
        if (!ElleUnodeAttributeActive(U_DISLOCDEN))
        ElleInitUnodeAttribute(U_DISLOCDEN);
    }
}

/*!
 * Write the input file for fft (PPC)
 * Format is:
 * int (no. of grains)
 * ** one line per grain with the following format:
 * dbl dbl dbl dbl int int (average properties of each grain and grain id)
 * ** one line per fourier point with the following format:
 * dbl dbl dbl int int int int int(euler angles, row, col, depth of
 * each pt, grain id, phase id)
 */
int Elle2fft()
{
    int err=0, max;
    bool sq_grid = true;
    int i, j, n;
    int r=0, c=0, rows=256, cols=256, cstep=1, spacing=1;
    int elle_cols;
    int grain_count, jgrain;
    int *pts_i=0;
    double val[4], jphase;
    double      **pts=0;        /* memory array */
    
    double dx,dy,elle_dx,elle_dy;
    Coords xy,fft_xy,origin;
    CellData unitcell;
    int max_unodes,max_flynns, SQ_pattern,rad_euler;
    UserData userdata;
    vector < int > F_attrib_init;
    vector < int > U_attrib_init;

        /*
         * check for any temporary attributes which may
         * not have been in the elle file
         * If they are initialised then put them on a list
         *  to be deleted before writing the output file
         */
    if (!ElleUnodeAttributeActive(U_ATTRIB_A)) {
        ElleInitUnodeAttribute(U_ATTRIB_A);
        U_attrib_init.push_back(U_ATTRIB_A);
    }
    if (!ElleUnodeAttributeActive(U_ATTRIB_C)) {
        ElleInitUnodeAttribute(U_ATTRIB_C);
        U_attrib_init.push_back(U_ATTRIB_C);
    }
    ElleUserData(userdata);
    rows=cols=(int)userdata[N_rows];
     // a value of 0 forces irregular pattern of unodes to avoid error
     // with FindSQ to models with low resolution
    SQ_pattern=(int)userdata[SQ_patterns];
     // a value of 0 euler angles in input elle file are in radians
    rad_euler=(int)userdata[Rad_euler];

    fft_xy.x = 0.0; //unitcell.cellBBox[BASE_LEFT].x  ??
    fft_xy.y = 0.0; //unitcell.cellBBox[BASE_LEFT].y  ??
    ElleCellBBox(&unitcell);
    dx = unitcell.xlength/cols;
    dy = unitcell.ylength/rows;

    // allocating array memory
    pts=dmatrix(0,rows*cols-1,0,7);
    if (pts==0) OnError("elle2fft pts",MALLOC_ERR);
    if (( pts_i=(int *)malloc(rows*cols*sizeof(int)))==0)
         OnError("elle2fft pts_i",MALLOC_ERR);

    double row_incr = 0.0;
    for (n=0,r=0;r<rows;r++) {   // cycle through fft points
        fft_xy.y = r * dy;
        row_incr = ElleSSOffset() * fft_xy.y/unitcell.ylength;
        for (c=0;c<cols;c++,n++) {
            fft_xy.x = c*dx + row_incr; //fft pt is non-ortho
            ElleNodeUnitXY(&fft_xy);
            i=FindUnode(&fft_xy, 0);
            if (i!=NO_VAL) {
                //  get the unode attribute
                ElleGetUnodeAttribute(i,&val[0],&val[1],&val[2],EULER_3);
                pts[n][0] = val[0];
	            pts[n][1] = val[1];
				pts[n][2] = val[2];
				pts[n][3] = (double)ElleUnodeFlynn(i);
               //  it is not reset in the 2nd write block below
				pts[n][4] = fft_xy.x; // the unitcell pos of the fft_pt
				pts[n][5] = fft_xy.y;
                ElleGetUnodeAttribute(i,&val[3],U_DISLOCDEN);
                pts[n][7] = val[3];
                // setting this shows that initially unodes and fft do
                // not match - unodes start at dx/2,dy/2   fft at 0,0
                //pts[n][6] = i;

                // get the unode flynn_id
                jgrain = ElleUnodeFlynn(i);
                // Flynn VISCOSITY density is used as phase indicative
                ElleGetFlynnRealAttribute(jgrain,&jphase,VISCOSITY);
                if (jphase > INT_MAX || jphase < INT_MIN)
                  pts_i[n]=1;
                else
                  pts_i[n]= static_cast<int> (jphase);

                // rad2degrees
                if (rad_euler == 0) {
                    pts[n][0]=pts[n][0]*RTOD;
                    pts[n][1]=pts[n][1]*RTOD;
                    pts[n][2]=pts[n][2]*RTOD;
                }
            }
            else  {
                OnError("No matching unode",0);
            }
        }
    }
    // Open the file
    string outfilename(ElleOutFile());
    if (outfilename.length()==0) {
		outfilename = DEFAULT_NAME;
	}
	grain_count = ElleNumberOfGrains();
    vector<int> flynn_ids(grain_count);

    ofstream outf(outfilename.c_str());
    	
    // Write the first line
    if (outf) 
    {
        if(!bFS_ExcludeFlynns) // BY FS
        {
            outf << grain_count << endl; 
            outf << setfill(' ') << setw(10) << setprecision(5); 
        }
        else
        {
            outf << "0" << endl;
        }
    }
    else OnError((char *)outfilename.c_str(),OPEN_ERR);

    // Write the first block: COMMENTS BY FS: Added the option not to do this, to increase code stability 
    max = ElleMaxFlynns();
    for (i=0,j=0; i<max && outf; i++) 
    {
        if (ElleFlynnIsActive(i)) 
        {
            if(!bFS_ExcludeFlynns) // BY FS
            {
                ElleGetFlynnEuler3(i,&val[0],&val[1],&val[2]);
                
                outf<<val[0]<<'\t'<<val[1]<<'\t'<<val[2]<<'\t'<<val[0]<<'\t'
                        <<'0'<<'\t'<<i<<endl;
            }

            flynn_ids[j] = i;
            j++;
        }
    }
    if (!outf) OnError((char *)outfilename.c_str(),WRITE_ERR);
#if XY
    // Write the second block
    for (r=1;r<=rows && outf;r++) // cycle through fft points
    {   
        i=(r-1)*cols;
        for (c=1;c<=cols && outf;c++,i++) 
        {
            outf << pts[i][0] <<'\t'
                     << pts[i][1] <<'\t'
                     << pts[i][2] <<'\t'
                     << c << '\t'
                     << r <<'\t'
                     << '1' << '\t';
			j=0;
            //pts[i][4]= (c-1)*unitcell.xlength/rows;
            //pts[i][5]= (r-1)*unitcell.ylength/cols;
			pts[i][3] = (double)ElleUnodeFlynn(i);
            while (flynn_ids[j]!=(int)pts[i][3]) j++;
            if (bFS_ExcludeFlynns) // by FS
                outf << 0 <<'\t' << pts_i[i] <<endl; // FS (!) Editet this line, where 0 is was before: j+1
            else
                outf << j+1 <<'\t' << pts_i[i] <<endl; // FS (!) This what was here before the if (bFS_...) loop was created
                
            //pts[i][6]=j+1;
        }
    }
    if (!outf) OnError((char *)outfilename.c_str(),WRITE_ERR);
    outf.close();
#endif

//Reassign unodes values using fft points calculated
    max_flynns = ElleMaxFlynns(); // maximum flynn number used
    max_unodes = ElleMaxUnodes(); // maximum unode number used

	int k, newj;
    vector <int> reassignedlist;
    // just look at the neighbour regions? May be an error
    // if narrow neighbour or unodes too sparse.
    std::list<int> nbflynns;
    std::list<int> :: iterator it;
    bool found = false;
    for (i=0;i<max_unodes;i++){
// LE                xy.x = pts[i][4]+ElleSSOffset()*pts[i][5]; // to modify.. .. ..phi::  shear offset
//LE should the unodes be moved to the position of the fft pt?
        xy.x = pts[i][4];
        xy.y = pts[i][5];
        ElleSetUnodePosition(i,&xy);
        ElleSetUnodeAttribute(i,pts[i][0],pts[i][1],pts[i][2],EULER_3);
        ElleSetUnodeAttribute(i,pts[i][6],U_ATTRIB_A);
        ElleSetUnodeAttribute(i,pts[i][7],U_DISLOCDEN);
	j = ElleUnodeFlynn(i);
	newj = pts[i][3];
        if (j!=newj) {
            ElleRemoveUnodeFromFlynn(j,i);
            ElleAddUnodeToFlynn(newj, i);
        }
        if (!EllePtInRegion(newj,&xy)) {
            nbflynns.clear();
            ElleFlynnNbRegions(newj,nbflynns);
            found = false;
            for (it=nbflynns.begin(); it!=nbflynns.end() && !found; it++){
                k = *it;
                if (ElleFlynnIsActive(k)){
                    if (EllePtInRegion(k,&xy)){
                        ElleRemoveUnodeFromFlynn(newj,i);
                        ElleAddUnodeToFlynn(k, i);
                        reassignedlist.push_back(i);
                        found=true;
                    }
                }
            }
            // if not found in nb flynns then search all flynns
            for (k=0;k<ElleMaxFlynns() && !found;k++){
                if (ElleFlynnIsActive(k)){
                    if (EllePtInRegion(k,&xy)){
                        ElleRemoveUnodeFromFlynn(newj,i);
                        ElleAddUnodeToFlynn(k, i);
                        //printf("unode %d from flynn %d to flynn %d\n",
                               //i,j,k);
                        reassignedlist.push_back(i);
                        found=true;
                        break;
                    }
                }
            }
        }
    }
	for (j=0;j<reassignedlist.size();j++) {
		k = ElleUnodeFlynn(reassignedlist[j]);
		SetUnodeAttributeFromNbFlynn(reassignedlist[j],k,EULER_3,reassignedlist);
        ElleGetUnodeAttribute(reassignedlist[j],&val[0],&val[1],&val[2],EULER_3);
        pts[reassignedlist[j]][0] = val[0];
	    pts[reassignedlist[j]][1] = val[1];
		pts[reassignedlist[j]][2] = val[2];
    // What about DISLOCDEN ??
	}
    // Write the second block
    for (r=1;r<=rows && outf;r++) // cycle through fft points
    {
        i=(r-1)*cols;
        for (c=1;c<=cols && outf;c++,i++) 
        {
            outf << pts[i][0] <<'\t'
                     << pts[i][1] <<'\t'
                     << pts[i][2] <<'\t'
                     << c << '\t'
                     << r <<'\t'
                     << '1' << '\t';
			j=0;
            //pts[i][4]= (c-1)*unitcell.xlength/rows;
            //pts[i][5]= (r-1)*unitcell.ylength/cols;
            // get the unode flynn_id
            jgrain = ElleUnodeFlynn(i);
            
            // Flynn VISCOSITY density is used as phase indicative
            ElleGetFlynnRealAttribute(jgrain,&jphase,VISCOSITY);
            
            if (jphase > INT_MAX || jphase < INT_MIN) pts_i[i]=1;
            else pts_i[i]= static_cast<int> (jphase);
            
            pts[i][3] = (double)jgrain;
            
            while (flynn_ids[j]!=(int)pts[i][3]) j++;
            
            if (bFS_ExcludeFlynns) // by FS
                outf << 0 <<'\t' << pts_i[i] <<endl; // FS (!) Editet this line, where 0 is was before: j+1
            else
                outf << j+1 <<'\t' << pts_i[i] <<endl; // FS (!) This what was here before the if (bFS_...) loop was created
            //pts[i][6]=j+1;
        }
    }
    if (!outf) OnError((char *)outfilename.c_str(),WRITE_ERR);
    outf.close();
#if XY
/*
  IS THIS NECESSARY ??? 
  In this version U_ATTRIB_C is not written to file so
  loop commented out
 */
// Assign U_ATRIB_C
// Added to be compatible with new version of gbm using unodes
// full version with check of uncoherent unodeslist with flynn

    int ii,jj;

    for (i=0;i<max_unodes;i++)   // cycle through unodes
    {
        jj=ElleUnodeFlynn(i);

         if(ElleFlynnIsActive(jj)) ElleSetUnodeAttribute(i,U_ATTRIB_C,
double(jj));
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
       }
    }
#endif
    if (err = ElleRemoveUnodeAttributes(U_attrib_init))
        OnError("Elle2fft",ATTRIBID_ERR);
    /*
     * write the updated Elle file
     */
    if (err=ElleWriteData("elle2fft.elle"))
        OnError("",err);
// deallocating array memory
    free_dmatrix(pts,0,rows*cols-1,0,7);
    pts=0;

 // rewrite temp.out file
    ofstream outf2(DEFAULT_NAME2);
    outf2 <<  unitcell.xlength <<'\t'<< unitcell.ylength <<'\t'<< '1' << endl;
    outf2 << "0 0 0"<< endl;
//  unodes are "unwrapped"
    outf2 << ElleSSOffset() << endl;
//    outf2 << unitcell.cellBBox[TOPLEFT].x-unitcell.cellBBox[BASELEFT].x << endl;
    outf2.close();

	return(err);
}

/*!
 * Checks for a SQ_GRID spatial distribution of unodes
 * Calculates and sets the grid parameters
 * Returns 0 if grid is regular, rectangular else returns 1
 */
int FindSQParams( int *numperrow,
				  double *dx, double *dy,
				  Coords *origin )
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
                                                                                

/*!
   this fn assumes attr is EULER_3
 */
int SetUnodeAttributeFromNbFlynn(int unum,int flynnid,int attr,
                                vector<int> &rlist)
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
                (find(rlist.begin(),rlist.end(),unodelist[k])==rlist.end()))
{
                ElleGetUnodePosition(unodelist[k],&xy_unode);
                ElleCoordsPlotXY (&xy, &xy_unode);
                dist = pointSeparation(&xy,&xy_unode);

                if (dist<dist_min) { // option interpolate...quaternions
                    id = k;
                    dist_min=dist;
                }
              }
            }

          if (id!=-1) {
                ElleGetUnodeAttribute(unum,&val[0],&val[1],&val[2],EULER_3);
                //printf ("check_error setting unode id %d (%lf %lf %lf) ",
                 //        unum,val[0],val[1],val[2]);
                //  get the unode attributes
                ElleGetUnodeAttribute(unodelist[id],&val[0],&val[1],&val[2],EULER_3);
                // set new unode values
                ElleSetUnodeAttribute(unum,val[0],val[1],val[2],EULER_3);
                //printf ("to values for %d (%lf %lf %lf) \n",
                 //      unodelist[id],val[0],val[1],val[2]);
            }
        else if (unodelist.size()==1) {
    		ElleGetFlynnEuler3(flynnid,&val[0],&val[1],&val[2]);
                ElleSetUnodeAttribute(unum,val[0],val[1],val[2],EULER_3);
                printf ("small flynn %d, one unode %d\n",unum,flynnid);
            }
        else {
                printf ("check_error NOT setting unode attr %d for flynn %d\n", unum,flynnid);
            }

 unodelist.clear();
  }
 }
#if XY
extern VoronoiData *V_data;
int FindUnode(Coords *ptxy, Coords *origin,
              double dx, double dy,
              int numperrow);
int FindUnode(Coords *ptxy, int start);
/*!
 * Assumes SQ_GRID spatial distribution of unodes
 * defined by the passed parameters
 */
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
int FindUnode(Coords *ptxy, int start)
{
    int i, j, id=NO_VAL, unode_vpts;
    int max_unodes = ElleMaxUnodes();
    Coords rect[4];
    Coords *bndpts, refxy, xy;
    double roi = ElleUnodeROI();
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
                V_data->vpoints[*it].getPosition(&xy);
                ElleCoordsPlotXY(&xy,&refxy);
                bndpts[j] = xy;
            }
            if (EllePtInRegion(bndpts,unode_vpts,ptxy)) {
                id = i;
            }
            delete [] bndpts;
        }
    }
    return(id);
}
#endif
