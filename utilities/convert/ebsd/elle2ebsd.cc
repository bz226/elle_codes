 /*****************************************************
 * Copyright: (c) 2007 L. A. Evans, S. Piazolo
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
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
/*!
	\file		elle2ebsd.cc
	\brief		elle to ebsd file conversion utility
	\par		Description:
This is a process reads an elle file which has unodes with Euler angle
attributes. The output is an ebsd file ".txt" which can be imported
into channel5.  The density of ebsd points is determined by user data input
                                                                                
udata[0]: vertical length that the files should have in Channel [164]
udata[1]: horizontal length that the files should have in Channel [164]
udata[2]: vertical step [2]
udata[3]: horizontal step [2]
udata[4]: Flag for Euler angle conversion. Elle is now using Bunge (ZXZ)
as default. [0]
                                                                                
Example:
elle2ebsd -i Grain1.elle -u 200 200 2 2
this results in Grain1.elle.txt file with X Y values:
0 0
2 0
4 0
 .
 .
398 0
0 2 2 2
 .
 .
398 398
If the unode grid does not match the requested hkl grid, the Euler
values are determined by a point-in-region test for the hkl point (in
Elle units). The regions tested are the unode voronoi cells.

Need to run unix2dos on the output (.txt file) for hkl software to read
it successfully
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <list>
#include <algorithm>
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "runopts.h"
#include "parseopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "triattrib.h"
#include "polygon.h"
#include "unodes.h"
#include "convert.h"
#include "setup.h"
#include "stats.h"
#include "log.h"
#include "mat.h"
#include "elle2ebsd.h"

using namespace std;

int InitThisProcess(), SetUnodes();
void SetUnodeAttributeFromFlynn(int flynnid,int attr_id);
void CalcBunge2Roe (double phi1, double PHI, double phi2, double *psi,
double *theta, double *Phi);
void CalcRoe2Bunge  (double psi, double theta, double Phi, double *phi1, double *PHI, double *phi2);
int FindSQParams( int *numperrow, double *dx, double *dy, Coords *origin);
void eulerZXZtoCaxis_d(double e1, double e2, double e3,
                       double *cax1, double *cax2);
void CheckBunge_d(double *e1, double *e2, double *e3);

/*#define pi 3.1415927*/
#define pi 180.0

int main(int argc, char **argv)
{
    int err=0;
    UserData udata;
    extern int InitThisProcess(void);
 
    /*
     * initialise
     */
    ElleInit();
    
    ElleSetOptNames("HKLRows","HKLColumns",
					"HorizStep","VertStep",
					/*"ZYZEuler",*/
                    "unused",
					"unused","unused","unused","unused");
    ElleUserData(udata);
    udata[RowsE2C] = 164;
    udata[ColsE2C] = 164;
    udata[CStepE2C] = 2;
    udata[RStepE2C] = 2;
    udata[ZYZEuler] = 0;
    ElleSetUserData(udata);

    if (err=ParseOptions(argc,argv))
        OnError("",err);

    /*
     * set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);


    /*
     * set the interval for writing to the stats file
    ES_SetstatsInterval(100);
     */
	ElleSetDisplay(0);

    /*
     * set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);

    /*
     * set the base for naming statistics and elle files
     */
    ElleSetSaveFileRoot("elle2ebsd");

    /*
     * run your initialisation function and start the application
     */
    StartApp();
    
     return(0);
} 

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 * The Symmetry entry is probably 43 (Cubic) or one of the following:
 * Triclinic    1    CalciumIronAluminumSilicate
 * Monoclinic    2    Augite
 * Monoclinic    20    Muscovite
 * Orthorhombic    22    Aragonite
 * Tetragonal    42    BaTiO3
 * Trigonal    32    Calcite
 * Hexagonal    62    Beryllium
 * Cubic        43    Bornite
 * Cubic        43    GaSb,Ni,Al
 * In this file, the default symmetry is cubic, if anything else is needed
 */


int InitThisProcess()
{
    char *infile;
    int err=0;
    
    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(SetUnodes);

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
        if (!ElleUnodesActive()) OnError("No unodes in file",0);
        if (!ElleUnodeAttributeActive(EULER_3))
                ElleInitUnodeAttribute(EULER_3);
    }
}

int SetUnodes()
{
    bool sq_grid = true;
    int i, j, k, n;
    int max_stages, ZYZ_euler=0,max_flynns, max_unodes,distance;
    int r=0, c=0, rows=0, cols=0;
    int elle_cols;
    double e1,e2,e3;
    double hkl1, hkl2, hkl3;
    double dx,dy,elle_dx,elle_dy,cstep=1.0, spacing=1.0;
    Coords xy,hkl_xy,origin;

    UserData udata;
    CellData unitcell;
    char *infile,infilea[100]="\0";
    
    infile = ElleFile();

    strcat(infilea,infile);
    strcat(infilea,".txt");
    
    ElleCellBBox(&unitcell);

    ElleUserData(udata);
    rows = (int)udata[RowsE2C];
    cols = (int)udata[ColsE2C];
    cstep = udata[CStepE2C];
    spacing = udata[RStepE2C];
    /* this should be zero as Elle and ebsd both use ZXZ
    ZYZ_euler = (int)udata[ZYZEuler]; */
    
    max_unodes = ElleMaxUnodes(); // maximum unode number used
    double hkl_pts[rows*cols][7]; // unode location, euler & caxis vals in hkl fmt

    hkl_xy.x = 0.0;
    hkl_xy.y = 0.0;
    dx = unitcell.xlength/cols;
    dy = unitcell.ylength/rows;
    if ( FindSQParams( &elle_cols, &elle_dx, &elle_dy, &origin ) != 0 )
        sq_grid=false;
    if (!sq_grid) {
//change this to find closest unode to hkl pt using 
// point-in-hkl-pt rect
        Log(0,"Does not look like square grid. Using general algorithm - be patient");
        /*ElleVoronoiUnodes();*/
    }
    for (n=0,r=0;r<rows;r++) {   // cycle through hkl points
        hkl_xy.x = 0.0;
        for (c=0;c<cols;c++) { 
            hkl_pts[n][0] = c*cstep;
            hkl_pts[n][1] = (rows-1 - r)*spacing;
            if (!sq_grid) i=FindUnode(&hkl_xy, 0); 
            else i=n;
            // need to do more work to match when ebsd dimensions
            // are not the same as elle dimensions ie xy positions
            // do not match or interpolate.  Convert hkl dx, dy to
            // Elle dx, dy?  i=FindUnode(&hkl_xy, &origin, dx, dy, cols);
            if (i!=NO_VAL) {
                //  get the unode attribute
                ElleGetUnodeAttribute(i,&e1,&e2,&e3,EULER_3);
                if (ZYZ_euler==1)
                {
	/*****************************************/
                      CalcBunge2Roe(e1, e2, e3,
								  &hkl_pts[n][2],
								  &hkl_pts[n][3],
								  &hkl_pts[n][4]);
                      eulerZXZtoCaxis_d(e1,e2,e3,
                                  &hkl_pts[n][5],&hkl_pts[n][6]);
	/*****************************************/
                }
                else
                {
	/*****************************************/
		CheckBunge_d(&e1, &e2, &e3);
                hkl_pts[n][2] = e1;
                hkl_pts[n][3] = e2;
                hkl_pts[n][4] = e3;
                eulerZXZtoCaxis_d(e1,e2,e3,
                                  &hkl_pts[n][5],&hkl_pts[n][6]);
                }
            }
            else {
                OnError("No matching unode",0);
            }
            hkl_xy.x += dx;
            n++;
        }
        hkl_xy.y += dy;
    }
	ofstream outf(infilea);
	outf << setw(10) << setprecision(7);
       // output Euler angles in degrees
	if (outf) outf << "X Y Euler1 Euler2 Euler3 Azimuth Dip" << endl;
    for (r=rows-1;r>=0;r--) {   // cycle through hkl points
        i=r*cols;
        for (c=0;c<cols;c++,i++) { 
		    outf << hkl_pts[i][0] << ' '
                     << hkl_pts[i][1] <<' '
				     << hkl_pts[i][2] << ' '
                     << hkl_pts[i][3] <<' '
                     << hkl_pts[i][4] <<' '
                     << hkl_pts[i][5] <<' '
					 << hkl_pts[i][6] <<endl;
	    }
	}
#if XY
       // output orientation matrix - row major order
        double a[3][3];
	if (outf) outf << "X Y g11 g12 g13 g21 g22 g23 g31 g32 g33" << endl;
    for (r=rows-1;r>=0;r--) {   // cycle through hkl points
        i=r*cols;
        for (c=0;c<cols;c++,i++) { 
           orientmatZXZ(a,hkl_pts[i][2]*DTOR,
                       hkl_pts[i][3]*DTOR,hkl_pts[i][4]*DTOR);
	    outf << hkl_pts[i][0] << ' '
                     << hkl_pts[i][1] <<' '
                     << a[0][0] << ' ' << a[0][1] <<' ' << a[0][2] <<' '
                     << a[1][0] << ' ' << a[1][1] <<' ' << a[1][2] <<' '
                     << a[2][0] << ' ' << a[2][1] <<' ' << a[2][2] <<' '
					  <<endl;
	    }
	}
#endif
	outf.close();
	/*****************************************/
} 


// calculate Elle to ZYZ Euler format
void CalcBunge2Roe (double phi1, double PHI, double phi2, double *psi, double *theta, double *Phi)
{
  double psi_calc, theta_calc,Phi_calc;

  psi_calc=phi1-(pi/2);
  theta_calc = PHI;
  Phi_calc = phi2+(pi/2);
  *psi=psi_calc;
  *theta=theta_calc;
  *Phi=Phi_calc;

}

// calculate ZYZ to HKL Euler format
void CalcRoe2Bunge  (double psi, double theta, double Phi, double *phi1, double *PHI, double *phi2)
{
 double phi1_calc, PHI_calc, phi2_calc;
  phi1_calc=psi+(pi/2);
  PHI_calc = theta;
  phi2_calc = Phi-(pi/2);
  *phi1=phi1_calc;
  *PHI=PHI_calc;
  *phi2=phi2_calc;

}

extern VoronoiData *V_data;
/*!
 * Assumes SQ_GRID spatial distribution of unodes
 */
int FindSQParams( int *numperrow, double *dx, double *dy, Coords *origin )
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

void eulerZXZtoCaxis_d(double e1, double e2, double e3,
                       double *cax1, double *cax2)
{
    double pole[3];
    pole[0] = pole[1] = pole[2] = 0.0;
    eulerZXZ2cax(e1,e2,e3,pole);
    CartesianToPolar(pole[0],pole[1],pole[2],cax1,cax2);
    *cax1 *= RTOD;
    *cax2 *= RTOD;
    MathToGeo(cax1,cax2);
    
}

void CheckBunge_d(double *e1, double *e2, double *e3)
{
    double tmp[3],bori[3];
	tmp[0]=bori[0]= *e1 * DTOR;
	tmp[1]=bori[1]= *e2 * DTOR;
	tmp[2]=bori[2]= *e3 * DTOR;
	eulerRange(tmp,bori);
	*e1 = bori[0] * RTOD;
	*e2 = bori[1] * RTOD;
	*e3 = bori[2] * RTOD;
}

   
#if XY

void firo(double *a, double *phi, double *rho)
{
    double z,zz;

    z=sqrt((a[0]*a[0])+(a[1]*a[1]));
    zz=sqrt((a[0]*a[0])+(a[1]*a[1])+(a[2]*a[2]));
    if(zz >= 0.0001)
    {
        if(z >= 0.00001)
        {
            *phi=fabs(acos(a[0]/z));
        }
        else
            *phi=0.0;

        if(a[1] < 0.0)
            *phi=-(*phi);

        *rho=acos(a[2]/zz);
    }
    else
    {
        *phi=0.0;
        *rho=0.0;
    }

    if(*rho*180.0/M_PI >= 90.0)
    {
        *rho=M_PI-*rho;

        if(*rho*180.0/M_PI >= 90.0)
        {
            *rho=M_PI-*rho;
            *phi=*phi+M_PI;
        }
    }
    else
    {
        // if phi is zero leave it so that azimuth is 90
        if (*phi!=0) *phi=*phi+M_PI;
    }

}
These are now in the base library
int FindUnode(Coords *ptxy, Coords *origin,
              double dx, double dy,
              int numperrow);
int FindUnode(Coords *ptxy, int start);
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
    bool min_set=false;
    int i, j, id=NO_VAL;
    int max_unodes = ElleMaxUnodes();
    Coords refxy, xy;
    double min=0.0, dist_sq=0;
    double roi = ElleUnodeROI();
    for (i=start;i<max_unodes && id==NO_VAL;i++)   // cycle through unodes
    {
        ElleGetUnodePosition(i,&refxy);
        /*ElleCoordsPlotXY(&refxy,ptxy); //should we assume wrapping??*/
        dist_sq = (ptxy->x-refxy.x)*(ptxy->x-refxy.x) +
                  (ptxy->y-refxy.y)*(ptxy->y-refxy.y);
        if (!min_set) {
            id = i;
            min = dist_sq;
            min_set = true;
        }
        else if (dist_sq<min) {
            id = i;
            min = dist_sq;
        }
    }
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
    for (i=start;i<max_unodes && id==NO_VAL;i++)   // cycle through unodes
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
if (i==73) cout << *it << ' ';
            }
if (i==73) cout << endl;
            if (EllePtInRegion(bndpts,unode_vpts,ptxy)) {
                id = i;
            }
            delete [] bndpts;
        }
    }
    return(id);
}
#endif
