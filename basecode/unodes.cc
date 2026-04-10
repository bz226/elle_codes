 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "unodesP.h"
#include "unodes.h"
#include "nodes.h"
#include "interface.h"
#include "file.h"
#include "attribarray.h"
#include "error.h"
#include "general.h"
#include "runopts.h"
#include "polygon.h"
#include "tripoly.h"
#include "polyutils.h"
#include "../utilities/gpc/gpcclip.h"
#include "check.h"
#include "convert.h"
#include "log.h"
/*#include "timefn.h"*/

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::vector;
using std::list;
using std::pair;

int initUnodes();
int initUnodes(int num);
int findFlynn(Coords *xy);
void ElleFindLocalFlynns(set_int &flist, Coords *bnd, int numpts,
                         Coords *centre);
int VoronoiRegionUnodes(Coords *bnd,int numpts,struct triangulateio *out,
                struct triangulateio *vorout);
int findUnodeNeighbours(struct triangulateio *out,
                        struct triangulateio *vorout,
                        Coords *bnd, int num);
int findUnodeArea( VoronoiData *vdata );
int findFlynnUnodeNeighbours(struct triangulateio *out);
int findFlynnUnodeArea(int flynnid,struct triangulateio *out,
                             struct triangulateio *vorout,
                             Coords *bnd, int num_bnd_nodes, 
                             int *xflags, int *yflags,
                             vector<int> &unodelist);
void updateUnodeVoronoi(int pt1, int pt2, int newpt, int rgn);
extern void WriteVoronoiAsPoly(struct triangulateio *vorout);

/* make this a class */
vector<Unode> *Unodes = 0;
double ROI = 0.01;
AttributeArray DfltUnodeAttrib(0);
int CONCactive = 0;
VoronoiData *V_data = 0;

int UnodesActive()
{
    return(Unodes!=0);
}

int ElleVoronoiActive()
{
    return(V_data!=0 && V_data->vpoints.size()>0);
}

/*****************************************************************
 * Function that gets the id of a Unode in the vector *Unodes 
 * and returns a pointer of type Unode * to that Unode. 
 * Used by Mike.
 * Needed because cannot call to Unodes out of unodes.cc
 *
 * included Feb. 2002 
 ****************************************************************/

Unode *ElleGetParticleUnode(int id)
{
  if (Unodes)  // if there is a vector of Unodes 
    {
      return &(*Unodes)[id]; // return pointer to that Unode
    }
}

int ElleMaxUnodes()
{
    int num=0;
    if (Unodes!=0) num = Unodes->size();
    return(num);
}

int ElleInitUnodes()
{
    return(ElleInitUnodes(104));
}

void ElleInitUnodeAttribute(int id)
{
// should this also check if Unodes are active ??
    if (!DfltUnodeAttrib.hasAttribute(id))
        DfltUnodeAttrib.initAttribute(id);
}

void ElleRemoveDefaultUnodeAttribute(const int id)
{
    DfltUnodeAttrib.removeAttribute(id);
}

void ElleSetDefaultUnodeAttribute(double val, const int id)
{
    if (DfltUnodeAttrib.hasAttribute(id))
        DfltUnodeAttrib.setAttribute(id,val);
}

void ElleGetDefaultUnodeAttribute(double *val, const int id)
{
    if (DfltUnodeAttrib.hasAttribute(id))
        DfltUnodeAttrib.getAttribute(id,val);
}

void ElleSetDefaultUnodeAttribute(double val1, double val2, double val3,
                                   const int id)
{
    int err=0;

    switch (id) {
    case EULER_3:
        if (DfltUnodeAttrib.hasAttribute(E3_ALPHA))
            DfltUnodeAttrib.setAttribute(E3_ALPHA,val1);
        else err=ATTRIBID_ERR;
        if (!err && DfltUnodeAttrib.hasAttribute(E3_BETA))
            DfltUnodeAttrib.setAttribute(E3_BETA,val2);
        else err=ATTRIBID_ERR;
        if (!err && DfltUnodeAttrib.hasAttribute(E3_GAMMA))
            DfltUnodeAttrib.setAttribute(E3_GAMMA,val3);
        else err=ATTRIBID_ERR;
    case CAXIS:
        if (DfltUnodeAttrib.hasAttribute(CAXIS))
            DfltUnodeAttrib.setAttribute(CAXIS,val1);
        else err=ATTRIBID_ERR;
        if (!err && DfltUnodeAttrib.hasAttribute(CAXIS_Y))
            DfltUnodeAttrib.setAttribute(CAXIS_Y,val2);
        else err=ATTRIBID_ERR;
        if (!err && DfltUnodeAttrib.hasAttribute(CAXIS_Z))
            DfltUnodeAttrib.setAttribute(CAXIS_Z,val3);
        else err=ATTRIBID_ERR;
    default: err=ATTRIBID_ERR;
    }
    if (err) OnError("ElleSetDefaultUnodeAttribute",err);
}

void ElleGetDefaultUnodeAttribute(double *val1, double *val2, double *val3,
                                   const int id)
{
    int err=0;

    switch (id) {
    case EULER_3:
        if (DfltUnodeAttrib.hasAttribute(E3_ALPHA))
            DfltUnodeAttrib.getAttribute(E3_ALPHA,val1);
        else err=ATTRIBID_ERR;
        if (!err && DfltUnodeAttrib.hasAttribute(E3_BETA))
            DfltUnodeAttrib.getAttribute(E3_BETA,val2);
        else err=ATTRIBID_ERR;
        if (!err && DfltUnodeAttrib.hasAttribute(E3_GAMMA))
            DfltUnodeAttrib.getAttribute(E3_GAMMA,val3);
        else err=ATTRIBID_ERR;
    case CAXIS:
        if (DfltUnodeAttrib.hasAttribute(CAXIS))
            DfltUnodeAttrib.getAttribute(CAXIS,val1);
        else err=ATTRIBID_ERR;
        if (!err && DfltUnodeAttrib.hasAttribute(CAXIS_Y))
            DfltUnodeAttrib.getAttribute(CAXIS_Y,val2);
        else err=ATTRIBID_ERR;
        if (!err && DfltUnodeAttrib.hasAttribute(CAXIS_Z))
            DfltUnodeAttrib.getAttribute(CAXIS_Z,val3);
        else err=ATTRIBID_ERR;
    default: err=ATTRIBID_ERR;
    }
    if (err) OnError("ElleGetDefaultUnodeAttribute",err);
}

int ElleRemoveUnodeAttributes(vector<int> &attriblist)
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

void ElleRemoveUnodeAttribute(int u_id, const int id)
{
    if (Unodes) {
        if (u_id < Unodes->size()) (*Unodes)[u_id].removeAttribute(id);
        else OnError("Invalid unode index",0);
    }
}

void UnodesClean()
{
    if (Unodes) {
        delete Unodes;
        Unodes = 0;
        DfltUnodeAttrib.clean();
    }
}

int ElleInitUnodes(int numPerRow)
{
	return( ElleInitUnodes(numPerRow, HEX_GRID));
}

int ElleInitUnodes(int numPerRow, int pattern)
{
    int row, col, i, maxr, maxc, last=0;
    CellData unitcell;
    Coords xy, start;

    Unodes = new vector<Unode>;
    V_data = new VoronoiData;
    if (Unodes==0 || V_data==0) return(MALLOC_ERR);
	bool haveflynns = (ElleNumberOfGrains()>0);
    ElleCellBBox(&unitcell);
    double xoffset = unitcell.cellBBox[TOPLEFT].x -
                        unitcell.cellBBox[BASELEFT].x;
    double yoffset = unitcell.cellBBox[TOPRIGHT].y -
                        unitcell.cellBBox[TOPLEFT].y;
//rows is multiple of 15
//cols is multiple of 13
//to make triangles close to equilateral
    double xstep = unitcell.xlength/numPerRow;
    double ystep = xstep*cos(PI/6.0);
	if (pattern==SQ_GRID) ystep =  unitcell.ylength/numPerRow;
    double xlen = unitcell.xlength-xstep*0.1;
    double ylen = unitcell.ylength-ystep*0.1;
    double tot_x = 0.0;
    double tot_y = 0.0;
    start.x = xy.x = 0.0;
    if (pattern==SQ_GRID) start.y = xy.y = 0.0;
	else start.y = xy.y = ystep/2;
    i=row=0;
    while (xy.y < ylen) {
        while (xy.x < xlen) {
            Unode tmp(i,&xy);
            Unodes->push_back(tmp);
            if (haveflynns) last = ElleAddUnodeToFlynn(i,&xy,last);
            i++;
            xy.x += xstep;
        }
        row++;
        if (row%2 && pattern==HEX_GRID) xy.x = start.x + xstep/2;
        else xy.x = start.x;
        xy.y += ystep;
    }
    ROI = xstep;
    ROI = xstep * 1.84;
		
#if XY
    double step = len/(numPerRow+1);
    double step2 = step+step;
    maxr = (int)(unitcell.ylength/step-0.5);
    maxc = (int)(unitcell.xlength/step-0.5)/2;
    xy = start = unitcell.cellBBox[BASELEFT];
    xy.y = start.y+step;
    /*
     * should also include x and y offsets for skewed cells
     */
    for (row=0,i=0; row<maxr; row++,xy.y+=step) {
        if (row%2) xy.x = start.x+step2;
        else xy.x = start.x+step;
        for (col=0; col<maxc; col++,xy.x+=step2,i++) {
            Unode tmp(i,&xy);
            Unodes->push_back(tmp);
        }
    }
#endif
    
    return(0);
}

int ElleInitRanUnodes(int upercell, int subset, int pattern)
{
    int row, col, i, maxr, maxc, last=0;
    CellData unitcell;
    Coords xy, start;
	time_t timetmp;
	
    Unodes = new vector<Unode>;
    V_data = new VoronoiData;
    if (Unodes==0 || V_data==0) return(MALLOC_ERR);
	bool haveflynns = (ElleNumberOfGrains()>0);

    int extra;
	int j,k,m;
	double x,y,test;
	timetmp=time(0);
	ElleInitRandom((unsigned long int)timetmp);
	
	if(pattern==RAN_GRID)
		extra=1;
	else
		extra=0;
	

	for(k=0,i=0;i<subset+extra;i++) // no of 
	{
		for(j=0;j<subset+extra;j++)
		{
			if(pattern==SEMI_RAN_GRID)
			{
				for(m=0;m<upercell;m++)
				{
					do
					{
						x=(double)(ElleRandomD()/subset);
						y=(double)(ElleRandomD()/subset);
						test=(double)(ElleRandomD()/subset);
					}while (test > .1);

					xy.x=x+(i/(double) subset);
					xy.y=y+(j/(double) subset);
					Unode tmp(k,&xy);
            		Unodes->push_back(tmp);
            		if (haveflynns) last = ElleAddUnodeToFlynn(k,&xy,last);
					k++;
				}
			}
			else
			{
				xy.x=ElleRandomD();
				xy.y=ElleRandomD();
				Unode tmp(k,&xy);
            	Unodes->push_back(tmp);
            	if (haveflynns) last = ElleAddUnodeToFlynn(k,&xy,last);
				k++;

			}

		}
	}
    
    return(0);
}

int ElleRotateUnodes(double rad,Coords *origin)
{
    Coords xy, newxy;

    if (Unodes) {
        for (vector<Unode>::iterator it=Unodes->begin();
                                      it!=Unodes->end();it++) {
            it->getPosition(&xy);
            rotate_coords(xy.x,xy.y,origin->x,origin->y,
                          &newxy.x,&newxy.y,rad);
            it->setPosition(&newxy);
        }
    }
    return(0);
}

int UnodeAttributeRange(int attr,double *min,double *max)
{
    unsigned char set=0;
    double val=0.0, b=0.0, c=0.0;
	Coords_3D xyz;

    if (Unodes) {
        switch(attr) {
        case EULER_3:
            DfltUnodeAttrib.getAttribute(E3_ALPHA,&val);
            //val=fmod((val+90.0),30.0)/15;
            //val=fmod(val,180.0)/90;
            //if(val > 1.0) val=2.0-val;
            break;
        case CAXIS:
            DfltUnodeAttrib.getAttribute(CAXIS,&xyz);
			CartesianToPolar(xyz.x,xyz.y,xyz.z,&val,&b);
            val *= RTOD;
            break;
        default:
            DfltUnodeAttrib.getAttribute(attr,&val);
            break;
        }
        *min = *max = val;
        for (vector<Unode>::iterator it=Unodes->begin();
                                      it!=Unodes->end();it++) {
            if (it->hasAttribute(attr)) {
                switch(attr) {
                case EULER_3:
                    it->getAttribute(E3_ALPHA,&val);
                    /*val=fmod((val+90.0),30.0)/15;*/
                    /*val=fmod(val,180.0)/90;*/
                    /*if(val > 1.0) val=2.0-val;*/
                    break;
        		case CAXIS:
            		it->getAttribute(CAXIS,&xyz);
					CartesianToPolar(xyz.x,xyz.y,xyz.z,&val,&b);
            		val *= RTOD;
            		break;
                default:
                    it->getAttribute(attr,&val);
                    break;
                }
				if (val<*min) *min=val;
				if (val>*max) *max=val;
            }
        }
    }
    return(0);
}

const int MAX_LINE_LEN=256;

int ElleReadUnodeData(FILE *fp, char str[], int *attr, int maxa)
{
    /*
     * read data for all unconnected nodes
     * labels for any attributes
     * id, position and attribute data on a separate
     * line for each node with the format: %d %f %f [[%f]%f]..
     * (unode id , x, y, attr val,....)
     */
    int err=0, i, j, num, found=0, last=0;
    double val=0;
    Coords xy, xynb;

	bool haveflynns = (ElleNumberOfGrains()>0);
    if (Unodes==0) Unodes = new vector<Unode>;
    if (V_data==0) V_data = new VoronoiData;
    if (Unodes==0 || V_data==0) return(MALLOC_ERR);
    for (j=0;j<maxa;j++)
        ElleInitUnodeAttribute(attr[j]);
    if (UnodeAttributeActive(CONC_A)) ElleSetUnodeCONCactive(1);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            i = atoi(str);
            if ((num = fscanf(fp,"%lf %lf", &xy.x, &xy.y))!=2)
                                                 return(READ_ERR);
            Unode tmp(i,&xy);
            j=0;
            while (j<maxa) {
                if ((num = fscanf(fp,"%lf", &val))!=1)
                    return(READ_ERR);
                tmp.setAttribute(attr[j],val);
                if (attr[j]==CONC_A) found=1;
                j++;
            }
            if (maxa==0)  dump_comments( fp );
#if XY
if (!found) {
val = (double)ElleAddUnodeToFlynn(i,&xy,(int)val);
tmp.setAttribute(CONC_A,val);
}
else
#endif
            Unodes->push_back(tmp);
            if (haveflynns) last = ElleAddUnodeToFlynn(i,&xy,last);

            if (i==2) {
                ElleGetUnodePosition(1,&xynb);
                ElleCoordsPlotXY(&xynb,&xy);
                ROI = pointSeparation(&xy,&xynb);
                ROI = pointSeparation(&xy,&xynb) * 1.84;
#ifdef DEBUG
                cerr << "calculated unode ROI " << ROI << endl;
#endif
            }
            num = fscanf(fp,"\n");
        }
    }
    return(err);
}

int ElleReadUnodeRealAttribData(FILE *fp, char str[], int attr_id)
{
    /*
     * read data for all unconnected nodes
     * line for each node with the format: %d %f [[%f]%f]..
     * (unode id, attr val,....)
     */
    int err=0, i, j, num, found=0, last=0;
    int max;
    double val=0, val_3[3]={0,0,0};
    Coords xy;

    max = ElleMaxUnodes();
    switch(attr_id) {
    case  U_CONC_A:
      int id=CONC_A;
      ElleInitUnodeAttribute(id);
      ElleSetUnodeCONCactive(1);
      while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            if ((num = fscanf(fp,"%lf\n", &val)) != 1)
                return(READ_ERR);
            DfltUnodeAttrib.setAttribute(id,val);
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            i = atoi(str);
            if ((num = fscanf(fp,"%lf", &val))!=1)
                    return(READ_ERR);
            if (i<max) (*Unodes)[i].setAttribute(id,val);
            num = fscanf(fp,"\n");
        }
      }
      break;
    }
    return(err);
}

int ElleReadUnodeRealAttribData(FILE *fp,char str[],int *keys,int count)
{
    /*
     * Read unode attribute data in an elle file.
     * Labels then one or more data on a separate
     * line for each unode with the format: %d %f %f ....
     * (unode id number then values in the order read into keys)
     * CAXIS is always in files as orientation, plunge and is converted
	 * to xyz. So sending count of one and checking for special case
     */
    int i;
    int nn, num, err=0;
    int max;
    double val, val2;
	Coords_3D xyz;

    max = ElleMaxUnodes();
    for (i=0;i<count;i++)
        ElleInitUnodeAttribute(keys[i]);
    if (UnodeAttributeActive(CONC_A)) ElleSetUnodeCONCactive(1);
    while (!feof(fp) && !err) {
        if ((num = fscanf(fp,"%s", str))!=1 && !feof(fp))
            return(READ_ERR);
        else if (str[0] == '#') dump_comments( fp );
        else if (!strcmp(str,SET_ALL)) {
            for (i=0;i<count;i++) {
                if ((num = fscanf(fp," %lf", &val)) != 1)
                    return(READ_ERR);
				if (keys[0]==CAXIS) {
					if ((num = fscanf(fp," %lf", &val2)) != 1)
							                    return(READ_ERR);
			        val*= DTOR;
			        val2 *= DTOR;
			        PolarToCartesian(&xyz.x,&xyz.y,&xyz.z,val,val2);
					DfltUnodeAttrib.setAttribute(keys[i],&xyz);
					dump_comments(fp);
				}
				else DfltUnodeAttrib.setAttribute(keys[i],val);
            }
        }
        else if (str[0]<'0' || str[0]>'9') return(0);
        else {
            if ((nn = atoi(str))==MAX_INDX) return(MAXINDX_ERR);
            for (i=0;i<count;i++) {
                if ((num = fscanf(fp," %lf", &val)) != 1)
                    return(READ_ERR);
                if (keys[0]==CAXIS) {
                    if ((num = fscanf(fp," %lf", &val2)) != 1) 
                         return(READ_ERR);
					val*= DTOR;
					val2 *= DTOR;
					PolarToCartesian(&xyz.x,&xyz.y,&xyz.z,val,val2);
                    if (nn<max) (*Unodes)[nn].setAttribute(keys[i],&xyz);
					dump_comments(fp);
				}
				else if (nn<max) (*Unodes)[nn].setAttribute(keys[i],val);
            }
        }
        num = fscanf(fp,"\n");
    }
    return(0);
}

int ElleWriteUnodeData(const char *fname)
{
    /*
     * writes data for all unconnected nodes
     * label
     * id, position on a separate line for each node
     *  with the format: %d %f %f
     * (unode id , x, y)
     * writes attribute data for all unconnected nodes
     * attribute label(s)
     * id and attribute data on a separate line for each node
     *  with the format: %d %f [[%f]%f]..
     * (unode id , attr val,....)
     */
    char label[20];
    int err=0, i, j;
    int oldp;
    int file_id=NO_VAL, cnt=0;
    double val, val_3[3];
    double dfltval, dfltval_3[3];
	vector<Unode>::iterator it;

    if (Unodes) {
		ofstream outf(fname,ios::out|ios::app);
        if (!outf) return(OPEN_ERR);
        if (err=ElleWriteUnodeLocation(outf)) return(err);
        /* this should work on dflt attrib list which is a
           member of unodearray class */
        int maxa = DfltUnodeAttrib.numAttributes();
        int *attr_ids = new int[maxa];
        DfltUnodeAttrib.getList(attr_ids,maxa);
        oldp = outf.precision(8);
        outf.setf(ios::scientific,ios::floatfield);
        for (i=0;i<maxa && !err;i++) {
            switch(attr_ids[i]) {
            case  CONC_A:
              file_id=U_CONC_A;
              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
              outf << label << endl;
              err=ElleWriteUnodeAttributeData(outf,&attr_ids[i],1);
              break;
            case  START_S_X:
            case  START_S_Y:
            case  PREV_S_X:
            case  PREV_S_Y:
            case  CURR_S_X:
            case  CURR_S_Y:
              file_id=U_FINITE_STRAIN;
              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
              if (!err) outf << label;
              cnt=0;
              j=i;
              while(j<maxa && cnt<NUM_FINITE_STRAIN_VALS &&
                    id_match(FiniteStrainKeys,attr_ids[j],label)){
                  outf << " " << label;
                  j++;
                  cnt++;
              }
              outf << endl;
              err = ElleWriteUnodeAttributeData(outf,&attr_ids[i],cnt);
              i=j-1; // for loop will incr i
              break;
            case  E_XX:
            case  E_YY:
            case  E_XY:
            case  E_YX:
            case  E_ZZ:
            case  INCR_S:
            case  BULK_S:
            case  F_INCR_S:
            case  F_BULK_S:
              file_id=U_STRAIN;
              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
              if (!err) outf << label;
              cnt=0;
              j=i;
              while(j<maxa && cnt<NUM_FLYNN_STRAIN_VALS &&
                    id_match(FlynnStrainKeys,attr_ids[j],label)){
                  outf << " " << label;
                  j++;
                  cnt++;
              }
              outf << endl;
              err = ElleWriteUnodeAttributeData(outf,&attr_ids[i],cnt);
              i=j-1; // for loop will incr i
              break;
            case EULER_3:
            case E3_ALPHA:
              file_id=U_EULER_3;
              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
              if (!err) outf << label;
              cnt=3;
              outf << endl;
              err = ElleWriteUnodeAttributeData(outf,&attr_ids[i],cnt);
              i += cnt-1; // for loop will incr i
              break;
            case CAXIS:
            case CAXIS_X:
              file_id=U_CAXIS;
              if (!id_match(FileKeys,file_id,label)) err = KEY_ERR;
              if (!err) outf << label;
              cnt=1;
              outf << endl;
              err = ElleWriteUnodeAttributeData(outf,&attr_ids[i],cnt);
              i += 2; // allow for 3 attrib in dflt list,for loop will incr i
              break;
            case DISLOCDEN:
              file_id=U_DISLOCDEN;
			  // FALL THROUGH
            default:
              if (!id_match(FileKeys,attr_ids[i],label)) {
                  cerr << "Ignoring unknown unode attribute " <<
                         attr_ids[i] << endl;
              }
              else {
                  outf << label << endl;
                  err=ElleWriteUnodeAttributeData(outf,&attr_ids[i],1);
              }
              break;
            }
        }
        if (!outf)  err=WRITE_ERR;
        if (!err) {
                outf.precision(oldp);
                outf.setf(ios::fixed,ios::floatfield);
        }
        delete [] attr_ids;
    }
    return(err);
}

int ElleWriteUnodeLocation(ofstream &outf)
{
    char label[20];
    int err=0;
    int oldp;
	vector<Unode>::iterator it;

    if (Unodes) {
        if (!id_match(FileKeys,UNODES,label)) err=KEY_ERR;
        else outf << label << endl;
        if (!err) {
            oldp = outf.precision(8);
            for (it=Unodes->begin();it!=Unodes->end() && outf;it++)
                outf << (*it) << endl;
            if (!outf)  err=WRITE_ERR;
            outf.precision(oldp);
        }
    }
    return(err);
}

int ElleWriteUnodeAttributeData(ofstream &outf,int *keys,int count)
{
    /*
     * writes attribute data for all unconnected nodes
     * attribute label(s)
     * id and attribute data on a separate line for each node
     *  with the format: %d %f [[%f]%f]..
     * (unode id , attr val,....)
     */
    int err=0, i, j=0;
    int file_id=NO_VAL, cnt=0, write=0;
	Coords_3D val_3,dfltval_3;
    double eps=1e-15;
	vector<Unode>::iterator it;

    if (Unodes) {
        /* this should work on dflt attrib list which is a
           member of unodearray class */
        double dflts[MAX_VALS];
        double vals[MAX_VALS];
		int writecnt=count;
		if (keys[0]==CAXIS) writecnt=2;
        for(j=0;j<count;j++)
			if (keys[0]==CAXIS) {
	            DfltUnodeAttrib.getAttribute(keys[j],&dfltval_3);
				CartesianToPolar(dfltval_3.x,dfltval_3.y,dfltval_3.z,
								&dflts[0],&dflts[1]);
				dflts[0] *= RTOD;
				dflts[1] *= RTOD;
			}
			else
            	DfltUnodeAttrib.getAttribute(keys[j],&dflts[j]);
        outf << SET_ALL;
        for(j=0;j<writecnt;j++) outf << ' ' << dflts[j];
        outf << endl;
        for (it=Unodes->begin(); it!=Unodes->end() && outf;it++) {
            if (keys[0]==CAXIS) for(j=0;j<writecnt;j++) vals[j]=dflts[j];
            for(j=0;j<count;j++) {
                vals[j]=dflts[j];
                if (it->hasAttribute(keys[j]))
				if (keys[0]==CAXIS) {
	            	it->getAttribute(keys[j],&val_3);
					CartesianToPolar(val_3.x,val_3.y,val_3.z,
								&vals[0],&vals[1]);
					vals[0] *= RTOD;
					vals[1] *= RTOD;
				}
				else
                    it->getAttribute(keys[j],&vals[j]);
            }
                     // shouldn't have attribute if equal to dflt but 
                     // setAttribute doesn't check
            for(j=0,write=0;j<writecnt;j++)
                if (vals[j]!=dflts[j] && (keys[j]!=CONC_A ||
                                          vals[j]>dflts[j]+eps ||
                                          vals[j]<dflts[j]-eps))
                    write=1;
            if (write) {
                outf << it->id();
                for(j=0;j<writecnt;j++)
                    outf << ' ' << vals[j];
                outf << endl;
            }
        }
        if (!outf)  err=WRITE_ERR;
    }
    return(err);
}

#if XY

int ElleWriteUnodeData(const char *fname)
{
    /*
     * writes data for all unconnected nodes
     * label then labels for any attributes
     * id, position and attribute data on a separate
     * line for each node with the format: %d %f %f [[%f]%f]..
     * (unode id , x, y, attr val,....)
     */
    char label[20];
    int err=0, i;
    Coords xy;

    if (Unodes) {
		ofstream outf(fname,ios::out|ios::app);
        if (!outf) return(OPEN_ERR);
        if (!id_match(FileKeys,UNODES,label)) return(KEY_ERR);
        outf << label << ' ';
        /* this should work on dflt attrib list which is a
           member of unodearray class */
        int maxa = (*Unodes)[0].numAttributes();
        int *attr_ids = new int[maxa];
        (*Unodes)[0].getList(attr_ids,maxa);
        for (i=0;i<maxa && !err;i++) {
            if (!id_match(FileKeys,attr_ids[0],label)) err = KEY_ERR;
            outf << ' ' << label;
        }
        outf << endl;
        delete [] attr_ids;
        if (!err) {
            int oldp = outf.precision(8);
            for (vector<Unode>::iterator it=Unodes->begin();
                                      it!=Unodes->end() && outf;it++) 
                outf << (*it) << endl;
            if (!outf)  err=WRITE_ERR;
            outf.precision(oldp);
        }
    }
    return(err);
}
#endif

int LoadZIPUnodeRealAttribData( gzFile in, char str[], int attr_id )
{
  /* * read data for all unconnected nodes * line for each node with the format: %d %f [[%f]%f]..
  * (unode id, attr val,....) */
  int err = 0, i, j, num, found = 0, last = 0;
  int max;
  double val = 0, val_3[3] =
  {
    0, 0, 0
  };
  Coords xy;
  max = ElleMaxUnodes();
  switch ( attr_id )
  {
    case U_CONC_A:
      int id = CONC_A;
      ElleInitUnodeAttribute( id );
      ElleSetUnodeCONCactive( 1 );
      while ( !gzeof( in ) && !err )
      {
        str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
        if ( str[0] == '#' ) gzReadLineSTD( in );
        else if ( !strcmp( str, SET_ALL ) )
        {
          val = atof( ( gzReadSingleString( in ) ).c_str() );
          DfltUnodeAttrib.setAttribute( id, val );
        }
        else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
        else
        {
          i = atoi( str );
          val = atof( ( gzReadSingleString( in ) ).c_str() );
          if ( i < max ) ( * Unodes ) [i].setAttribute( id, val );
        }
      }
    break;
  }
  return ( err );
}

int LoadZIPUnodeRealAttribData( gzFile in, char str[], int * keys, int count )
{
  /* * Read unode attribute data in an elle file. * Labels then one or more data on a separate
  * line for each unode with the format: %d %f %f .... * (unode id number then values in the order read into keys) */
  int i;
  int nn, num, err = 0;
  int max;
  double val;

  max = ElleMaxUnodes();
  for ( i = 0; i < count; i++ )
    ElleInitUnodeAttribute( keys[i] );
  if ( UnodeAttributeActive( CONC_A ) ) ElleSetUnodeCONCactive( 1 );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' )
      gzReadLineSTD( in );
    else if ( !strcmp( str, SET_ALL ) )
    {
      for ( i = 0; i < count; i++ )
      {
        val = atof( ( gzReadSingleString( in ) ).c_str() );
        DfltUnodeAttrib.setAttribute( keys[i], val );
      }
    }
    else if ( str[0] < '0' || str[0] > '9' )
      return ( 0 );
    else
    {
      if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR );
      for ( i = 0; i < count; i++ )
      {
        val = atof( ( gzReadSingleString( in ) ).c_str() );
        if ( nn < max ) ( * Unodes ) [nn].setAttribute( keys[i], val );
      }
    }
  }
  return ( 0 );
}

int LoadZIPUnodeData( gzFile in, char str[], int * attr, int maxa )
{
  /* * read data for all unconnected nodes * labels for any attributes * id, position and attribute data on a separate
  * line for each node with the format: %d %f %f [[%f]%f].. * (unode id , x, y, attr val,....) */
  int err = 0, i, j, num, found = 0, last = 0;
  double val = 0;
  Coords xy, xynb;

  bool haveflynns = ( ElleNumberOfGrains() > 0 );
  Unodes = new vector < Unode >;
  if ( Unodes == 0 ) return ( MALLOC_ERR );
  for ( j = 0; j < maxa; j++ )
    ElleInitUnodeAttribute( attr[j] );
  if ( UnodeAttributeActive( CONC_A ) ) ElleSetUnodeCONCactive( 1 );
  while ( !gzeof( in ) && !err )
  {
    str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() );
    if ( str[0] == '#' ) gzReadLineSTD( in );
    else if ( str[0] < '0' || str[0] > '9' ) return ( 0 );
    else
    {
      i = atoi( str );
      xy.x = atof( ( gzReadSingleString( in ) ).c_str() );
      xy.y = atof( ( gzReadSingleString( in ) ).c_str() );
      Unode tmp( i, & xy ); j = 0;
      while ( j < maxa )
      {
        val = atof( ( gzReadSingleString( in ) ).c_str() );
        tmp.setAttribute( attr[j], val );
        if ( attr[j] == CONC_A ) found = 1;
        j++;
      }
#if XY
      if ( !found )
      {
        val = ( double )ElleAddUnodeToFlynn( i, & xy, ( int )val );
        tmp.setAttribute( CONC_A, val );
      }
      else
#endif
      if ( haveflynns ) last = ElleAddUnodeToFlynn( i, & xy, last );
      Unodes->push_back( tmp );

      if ( i == 2 )
      {
        ElleGetUnodePosition( 1, & xynb );
        ElleCoordsPlotXY( & xynb, & xy );
        ROI = pointSeparation( & xy, & xynb );
        ROI = pointSeparation( & xy, & xynb ) * 1.84;
#ifdef DEBUG
        cerr << "calculated unode ROI " << ROI << endl;
#endif
      }
    }
  }
  return ( err );
}

/*int LoadZIPUnodeRealAttribData( gzFile in, char str[], int * keys, int count ) {

int i; int nn, num, err = 0; int max; double val;

max = ElleMaxUnodes(); for ( i = 0; i < count; i++ ) ElleInitUnodeAttribute( keys[i] );
if ( UnodeAttributeActive( CONC_A ) ) ElleSetUnodeCONCactive( 1 ); while ( !gzeof( in ) && !err ) {
str = ( ( char * ) ( gzReadSingleString( in ) ).c_str() ); else if ( str[0] == '#' ) gzReadLineSTD( in );
else if ( !strcmp( str, SET_ALL ) ) { for ( i = 0; i < count; i++ ) { val = atof( ( gzReadSingleString( in ) ).c_str() );
DfltUnodeAttrib.setAttribute( keys[i], val ); } } else if ( str[0] < '0' || str[0] > '9' ) return ( 0 ); else {
if ( ( nn = atoi( str ) ) == MAX_INDX ) return ( MAXINDX_ERR ); for ( i = 0; i < count; i++ ) {
str = gzReadSingleString( in );

if ( nn < max ) ( * Unodes ) [nn].setAttribute( keys[i], val ); } } } return ( 0 ); }*/



int SaveZIPUnodeAttributeData( gzFile out, int * keys, int count )
{
  /* * writes attribute data for all unconnected nodes * attribute label(s)
  * id and attribute data on a separate line for each node *  with the format: %d %f [[%f]%f].. * (unode id , attr val,....) */

  int err = 0, i, j = 0;
  int file_id = NO_VAL, cnt = 0, write = 0;
  double val, val_3[3];
  double dfltval, dfltval_3[3];
  double eps = 1e-15;
  vector < Unode >::iterator it;

  if ( Unodes )
  {
    /* this should work on dflt attrib list which is a member of unodearray class */
    double dflts[MAX_VALS];
    double vals[MAX_VALS];
    for ( j = 0; j < count; j++ )
      DfltUnodeAttrib.getAttribute( keys[j], & dflts[j] );
    gzprintf( out, "%s", SET_ALL );
    for ( j = 0; j < count; j++ )
      gzprintf( out, " %lf\n", dflts[j] );
    for ( it = Unodes->begin(); it != Unodes->end() /* && outf */; it++ )
    {
      for ( j = 0; j < count; j++ )
      {
        vals[j] = dflts[j];
        if ( it->hasAttribute( keys[j] ) )
          it->getAttribute( keys[j], & vals[j] );
      }
      // shouldn't have attribute if equal to dflt but
      // setAttribute doesn't check
      for ( j = 0, write = 0; j < count; j++ )
        if ( vals[j] != dflts[j] && ( keys[j] != CONC_A || vals[j] > dflts[j] + eps || vals[j] < dflts[j] - eps ) )
          write = 1;
      if ( write )
      {
        gzprintf( out, "%d", it->id() );
        for ( j = 0; j < count; j++ )
          gzprintf( out, " %lf", vals[j] );
        gzprintf( out, "\n" );
      }
    }
  }
  return ( err );
}

int SaveZIPUnodeData( gzFile out )
{
  /* * writes data for all unconnected nodes * label * id, position on a separate line for each node
  *  with the format: %d %f %f * (unode id , x, y) * writes attribute data for all unconnected nodes * attribute label(s)
  * id and attribute data on a separate line for each node *  with the format: %d %f [[%f]%f].. * (unode id , attr val,....) */
  char label[20];
  int err = 0, i, j;
  int oldp;
  int file_id = NO_VAL, cnt = 0;
  double val, val_3[3];
  double dfltval, dfltval_3[3];
  vector < Unode >::iterator it;

  if ( Unodes )
  {
    if ( err = SaveZIPUnodeLocation( out ) ) return ( err );
    /* this should work on dflt attrib list which is a member of unodearray class */
    int maxa = DfltUnodeAttrib.numAttributes();
    int * attr_ids = new int[maxa];
    DfltUnodeAttrib.getList( attr_ids, maxa );
    for ( i = 0; i < maxa && !err; i++ )
    {
      switch ( attr_ids[i] )
      {
        case CONC_A:
          file_id = U_CONC_A;
          if ( !id_match( FileKeys, file_id, label ) ) err = KEY_ERR;
          gzprintf( out, "%s\n", label );
          err = SaveZIPUnodeAttributeData( out, & attr_ids[i], 1 );
        break;
        case START_S_X:
        case START_S_Y:
        case PREV_S_X:
        case PREV_S_Y:
        case CURR_S_X:
        case CURR_S_Y:
          file_id = U_STRAIN;
          if ( !id_match( FileKeys, file_id, label ) ) err = KEY_ERR;
          if ( !err ) gzprintf( out, "%s", label );
          cnt = 0;
          j = i;
          while ( j < maxa && cnt < NUM_FINITE_STRAIN_VALS && id_match( FiniteStrainKeys, attr_ids[j], label ) )
          {
            gzprintf( out, " %s", label );
            j++;
            cnt++;
          }
          gzprintf( out, "\n" );
          err = SaveZIPUnodeAttributeData( out, & attr_ids[i], cnt );
          i = j - 1; // for loop will incr i
        break;
        case EULER_3:
        case E3_ALPHA:
          file_id = U_EULER_3;
          if ( !id_match( FileKeys, file_id, label ) ) err = KEY_ERR;
          if ( !err ) gzprintf( out, "%s", label );
          cnt = 3;
          gzprintf( out, "\n" );
          err = SaveZIPUnodeAttributeData( out, & attr_ids[i], cnt );
          i += cnt - 1; // for loop will incr i
        break;
        default:
          if ( !id_match( FileKeys, attr_ids[i], label ) )
          {
            cerr << "Ignoring unknown unode attribute " << attr_ids[i] << endl;
          }
          else
          {
            gzprintf( out, "%s\n", label );
            err = SaveZIPUnodeAttributeData( out, & attr_ids[i], 1 );
          }
        break;
      }
    }
    /* if (!outf)  err=WRITE_ERR; if (!err) { outf.precision(oldp); outf.setf(ios::fixed,ios::floatfield); } */
    delete[] attr_ids;
  }
  return ( err );
}

int SaveZIPUnodeLocation( gzFile out )
{
  int count, n, err = 0;
  char label[20];
  Coords xy;
  count = ElleMaxUnodes();
  if ( !id_match( FileKeys, UNODES, label ) ) err = KEY_ERR;
  else
    gzprintf( out, "%s\n", label );
  if ( ElleUnodesActive() )
    for ( n = 0; n < count; n++ )
    {
      ElleGetUnodePosition( n, & xy );
      gzprintf( out, "%d %lf %lf\n", n, xy.x, xy.y );
    }
  return ( err );
}

void UnodeAttributeList(int **attr_ids,int *maxa)
{
    *attr_ids = 0; *maxa = 0;
    if (Unodes) {
        *maxa = DfltUnodeAttrib.numAttributes();
        if (*maxa>0)  {
            if ((*attr_ids = (int *)malloc(*maxa * sizeof(int)))==0)
                OnError("UnodeAttributeList",MALLOC_ERR);
            DfltUnodeAttrib.getList(*attr_ids,*maxa);
        }
    }
}

unsigned char UnodeAttributeActive(int id)
{
    unsigned char found=0;
    int i;
    if (Unodes) 
        if (DfltUnodeAttrib.hasAttribute(id)) found = 1;
    return(found);
}

void TriangulateUnodes(int flynnid,struct triangulateio *out)
{
     TriangulateUnodes(flynnid,out,CONC_A);
}

void TriangulateUnodes(int flynnid,struct triangulateio *out,
                       int attr)
{
    char fname[81];
    int *ids, num, i;
    int num_bnd_nodes,xflags[2],yflags[2];
    double attr_val;
    struct flagvals triflags;
    Coords current, prev, min, max;
    CellData unitcell;
	vector<int> :: iterator it;

    ElleFlynnNodes(flynnid,&ids,&num_bnd_nodes);
	vector<int> unodelist;
    ElleGetFlynnUnodeList(flynnid,unodelist);
    num = num_bnd_nodes + unodelist.size();
    double *xvals = new double[num];
    double *yvals = new double[num];
    double *attrib = new double[num];
    ElleNodePosition(ids[0],&prev);
    for (i=0; i<num_bnd_nodes; i++) {
        ElleNodePlotXY(ids[i],&current,&prev);
        xvals[i] = current.x;
        yvals[i] = current.y;
        attrib[i] = ids[i] * -1;
        /*attrib[i+num] = ElleNodeAttribute(ids[i],CONC_A);*/
        prev = current;
    }

    /*
     * reposition unodes into current flynn position
     */

    ElleCheckUnit(xvals,yvals,num_bnd_nodes,xflags,yflags,&min,&max);
    ElleCellBBox(&unitcell);
    i=num_bnd_nodes;
    for (it = unodelist.begin(); it != unodelist.end(); it++,i++) {
        ElleGetUnodePosition((*it),&current);
        if (xflags[0])
            while (current.x > max.x) {
                current.x -= unitcell.xlength;
                current.y -= unitcell.yoffset;
            }
        if (xflags[1])
            while (current.x < min.x) {
                current.x += unitcell.xlength;
                current.y += unitcell.yoffset;
            }
        if (yflags[0])
            while (current.y > max.y) {
                current.y -= unitcell.ylength;
                current.x -= unitcell.xoffset;
            }
        if (yflags[1])
            while (current.y < min.y) {
                current.y += unitcell.ylength;
                current.x += unitcell.xoffset;
            }
        xvals[i] = current.x;
        yvals[i] = current.y;
        attrib[i] = *it;
        /*attrib[i+num] = attr_val;*/
        /*prev = current;*/
    }
    triflags.area = 0;
    triflags.quality = 0;
    triflags.midptnodes = 0;
    triflags.bndpts = 0;
    triflags.edges = 0;
    triflags.voronoi = 0;
    if (tripolypts(out, &triflags, num_bnd_nodes,
                   xvals, yvals, attrib, num, 1))
        OnError("error triangulating",0);
    /*
     * if unodes close to boundary, triangle may reset the bnd marker
     */
    for (i=num_bnd_nodes;i<num;i++) out->pointmarkerlist[i]=0;
#ifdef DEBUG
    sprintf(fname,"tri%d.poly",flynnid);
    WritePoly(out,fname);
#endif
    delete[] xvals;
    delete[] yvals;
    delete[] attrib;
    if (ids) free(ids);
}

void ElleGetUnodePosition(int id, Coords *xy)
{
    if (Unodes) {
        if (id < Unodes->size()) (*Unodes)[id].getPosition(xy);
        else OnError("Invalid unode index",0);
    }
}

void ElleSetUnodePosition(int id, Coords *xy)
{
    if (Unodes) {
        if (id < Unodes->size()) (*Unodes)[id].setPosition(xy);
        else OnError("Invalid unode index",0);
    }
}

void ElleGetUnodePolyInfo(int id,const int attrib,
                          Coords *xy,double *val)
{
    int err=0;

    (*Unodes)[id].getPosition(xy);
    if ((*Unodes)[id].hasAttribute(attrib))
        err=(*Unodes)[id].getAttribute(attrib,val);
    else err=DfltUnodeAttrib.getAttribute(attrib,val);
    if (err)
        OnError("ElleGetUnodePolyInfo",err);
}

void ElleGetUnodeArea(int id, double *area)
{
    if (Unodes) {
        if (id < Unodes->size()) *area=(*Unodes)[id].area();
        else OnError("Invalid unode index",0);
    }
}

void ElleSetUnodeArea(int id, double area)
{
    if (Unodes) {
        if (id < Unodes->size()) (*Unodes)[id].setArea(area);
        else OnError("Invalid unode index",0);
    }
}

// use this
int ElleGetUnodeAttribute(int id, double *val,const int attr_id )
{
    int err=0;
    double val2;
	Coords_3D caxisval;

    switch (attr_id) {
    case CAXIS:
        /*
         * returns orientation if CAXIS and only
         * one value requested
         */
        if ((*Unodes)[id].hasAttribute(attr_id)) {
            err=(*Unodes)[id].getAttribute(CAXIS,&caxisval);
        }
        else {
            err=DfltUnodeAttrib.getAttribute(CAXIS,&caxisval);
    	}
		CartesianToPolar(caxisval.x,caxisval.y,caxisval.z,
						val,&val2);
		*val *= RTOD;
		val2 *= RTOD;
        break;
    default: 
        if ((*Unodes)[id].hasAttribute(attr_id))
            err=(*Unodes)[id].getAttribute(attr_id,val);
        else if (err=DfltUnodeAttrib.getAttribute(attr_id,val)) {
            err = 0;
            if (ElleFlynnAttributeActive(attr_id)) {
                int flynn = ElleUnodeFlynn(id);
                ElleGetFlynnRealAttribute(flynn,val,attr_id);
            }
            else err = ATTRIBID_ERR;
        }
        break;
    }
    if (err)
        OnError("ElleGetUnodeAttribute",err);
    return(err);
}

int ElleGetUnodeAttribute(int id, double *val1, double *val2, double *val3,
                          const int attr_id )
{
    int err=0;
    int flynn = NO_NB;
	Coords_3D caxisval;

    switch (attr_id) {
    case EULER_3:
        if ((*Unodes)[id].hasAttribute(attr_id)) {
            err=(*Unodes)[id].getAttribute(E3_ALPHA,val1);
            if (!err) err=(*Unodes)[id].getAttribute(E3_BETA,val2);
            if (!err) err=(*Unodes)[id].getAttribute(E3_GAMMA,val3);
        }
        else if ( err=DfltUnodeAttrib.getAttribute(E3_ALPHA,val1)) {
            err = 0;
            flynn = ElleUnodeFlynn(id);
            ElleGetFlynnEuler3(flynn,val1,val2,val3);
        }
        else {
            if (!err) err=DfltUnodeAttrib.getAttribute(E3_BETA,val2);
            if (!err) err=DfltUnodeAttrib.getAttribute(E3_GAMMA,val3);
       }
        break;
    case CAXIS:
        if ((*Unodes)[id].hasAttribute(attr_id)) 
            err=(*Unodes)[id].getAttribute(CAXIS,&caxisval);
		else
			err=DfltUnodeAttrib.getAttribute(CAXIS,&caxisval);
		*val1=caxisval.x;
		*val2=caxisval.y;
		*val3=caxisval.z;
        break;
    default:   err=ATTRIBID_ERR;
        break;
    }
    if (err)
        OnError("ElleGetUnodeAttribute",err);
    return(err);
}

int ElleGetUnodeAttribute(int id, double *val1, double *val2,
                          const int attr_id )
{
    int err=0;
	Coords_3D caxisval;

    switch (attr_id) {
    case CAXIS:
        if ((*Unodes)[id].hasAttribute(attr_id)) {
            err=(*Unodes)[id].getAttribute(CAXIS,&caxisval);
        }
        else {
            err=DfltUnodeAttrib.getAttribute(CAXIS,&caxisval);
    	}
		CartesianToPolar(caxisval.x,caxisval.y,caxisval.z,
						val1,val2);
		*val1 *= RTOD;
		*val2 *= RTOD;
        break;
    default:   err=ATTRIBID_ERR;
        break;
    }
    if (err)
        OnError("ElleGetUnodeAttribute",err);
    return(err);
}

int ElleGetUnodeAttribute(int id,const int attr_id, double *val )
{
    return( ElleGetUnodeAttribute(id,val,attr_id));
}

// use this
int ElleSetUnodeAttribute(int id, double val,const int attr_id )
{
    int err=0;

    if (err=(*Unodes)[id].setAttribute(attr_id,val))
        OnError("ElleSetUnodeAttribute",err);
    return(err);
}

int ElleSetUnodeAttribute(int id, double val1, double val2, double val3,
                          const int attr_id )
{
    int err=0;
	Coords_3D caxisval;

    switch (attr_id) {
    case EULER_3:
        err=(*Unodes)[id].setAttribute(E3_ALPHA,val1);
        if (!err) err=(*Unodes)[id].setAttribute(E3_BETA,val2);
        if (!err) err=(*Unodes)[id].setAttribute(E3_GAMMA,val3);
        break;
    case CAXIS:
		caxisval.x=val1;
		caxisval.y=val2;
		caxisval.z=val3;
        err=(*Unodes)[id].setAttribute(CAXIS,&caxisval);
        break;
    default:   err=ATTRIBID_ERR;
        break;
    }
    if (err)
        OnError("ElleSetUnodeAttribute",err);
    return(err);
}

int ElleSetUnodeAttribute(int id, double val1, double val2,
                          const int attr_id )
{
    int err=0;
	Coords_3D caxisval;

    switch (attr_id) {
    case CAXIS:
		val1*= DTOR;
		val2 *= DTOR;
		PolarToCartesian(&caxisval.x,&caxisval.y,&caxisval.z,val1,val2);
        err=(*Unodes)[id].setAttribute(CAXIS,&caxisval);
        break;
    default:   err=ATTRIBID_ERR;
        break;
    }
    if (err)
        OnError("ElleSetUnodeAttribute",err);
    return(err);
}

int ElleSetUnodeAttribute(int id,const int attr_id, double val )
{
    int err=0;

    if (err=(*Unodes)[id].setAttribute(attr_id,val))
        OnError("ElleSetUnodeAttribute",err);
    return(err);
}

int ElleFindUnodeRow(const double yval)
{
    int row=0;
    double ycurr=0;
    Coords xy;
	vector<Unode>::iterator it;
    if (Unodes) {
        it = Unodes->begin();
        it->getPosition(&xy);
        ycurr = xy.y;
        while (xy.y<yval && it!=Unodes->end()) {
            it->getPosition(&xy);
            it++;
            if (xy.y!=ycurr) { row++; ycurr=xy.y; }
        }
    }
    return(row);
}

int ElleFindUnodeColumn(const double xval)
{
    int col=0;
    double xcurr=0;
    Coords xy;
	vector<Unode>::iterator it;
    if (Unodes) {
        it = Unodes->begin();
        it->getPosition(&xy);
        xcurr = xy.x;
        while (xy.x<xval && it!=Unodes->end()) {
            it->getPosition(&xy);
            it++;
            if (xy.x!=xcurr) { col+=2; xcurr=xy.x; }
        }
    }
    return(col);
}

double ElleUnodeROI()
{
    return(ROI);
}

int ElleUnodeCONCactive()
{
    return(CONCactive);
}

void ElleSetUnodeCONCactive(int val)
{
    CONCactive = val;
}

void ElleLocalLists( int n, Coords *incr, set_int **unodeset,
                     set_int **flynnset, double roi )
{
    int i, j, max, err=0;
    int rgn[3], nbrgn[3], nb[3];
    Coords old_node,pivot_node,current;
    double rect_big[3][2][2];
    double tri_pos[3][2];
    set_int :: iterator it;

    ElleRegions(n,rgn);
    ElleNeighbourNodes(n,nb);

    ElleNodePosition(n,&old_node);
    tri_pos[0][0]= old_node.x;
    tri_pos[0][1]= old_node.y;

    tri_pos[1][0] = old_node.x + incr->x;
    tri_pos[1][1] = old_node.y + incr->y;

    flynnset[0] = new set_int;
    /* for each neighbour node*/
    for (i=0;i<3;i++) {
        //for each gb segment adjacent to the moving node
        if (nb[i]!=NO_NB) {
            flynnset[0]->insert(rgn[i]);
            ElleRegions(nb[i],nbrgn);
            for(j=0;j<3;j++)
                if(nbrgn[j] != NO_NB)  flynnset[0]->insert(nbrgn[j]);
            ElleNodePlotXY(nb[i],&pivot_node,&old_node);
            tri_pos[2][0]= pivot_node.x;
            tri_pos[2][1]= pivot_node.y;

           // make bounding rect that includes all close unodes
            grow_rect( tri_pos, rect_big[i], roi*1.2);
        }
    }
    for (i=0,it=flynnset[0]->begin();it!=flynnset[0]->end();
                                       i++,it++) {
        unodeset[i] = new set_int;
		vector<int> unodelist;
        ElleGetFlynnUnodeList(*it,unodelist);

        for (vector<int> :: iterator vit = unodelist.begin();
                            vit != unodelist.end() && !err; vit++) {
            ElleGetUnodePosition((*vit),&current);
            ElleCoordsPlotXY(&current,&old_node);

            for (j=0;j<3;j++) {
                if(nb[j] != NO_NB) {
                    if( current.x>rect_big[j][0][0] &&
                        current.y>rect_big[j][0][1] &&
                        current.x<rect_big[j][1][0] &&
                        current.y<rect_big[j][1][1] ){// inside bounding rect
                        unodeset[i]->insert(*vit);
                    }
                }
            }
        }
    }
    CombineLocalLists(unodeset,flynnset);
}
    
void CombineLocalLists( set_int **unodelists,
                        set_int **flynnlists)
{
    int i=0, j;
    set_int :: iterator it;

    if (!unodelists[COMBINED_LIST])
        unodelists[COMBINED_LIST] = new set_int;
    for (j=0;j<COMBINED_LIST;j++) {
        if (unodelists[j]) {
          for (it=unodelists[j]->begin();it!=unodelists[j]->end();it++)
              unodelists[COMBINED_LIST]->insert(*it);
        }
    }
}

void ElleReassignUnodes( set_int *unodelist, set_int *flist )
{
    int found=0;
    Coords xy;
    set_int :: iterator f_it;
    set_int :: iterator it, it2;

    // strip out unodes from flynn lists
    for (it = unodelist->begin(); it != unodelist->end(); it++) {
        for (f_it = flist->begin(), found=0; f_it != flist->end() &&
                                                     !found; f_it++) {
            if (ElleRemoveUnodeFromFlynn(*f_it, *it)==0) {
                ElleSetUnodeFlynn(*it,NO_VAL);
                found=1;
            }
        }
    }
/*
int s1=unodelist->size();
int s2 = flist->size();
if (s1>0){
for (f_it = flist->begin(); f_it != flist->end(); f_it++) 
cout << *f_it << " ";
cout << endl;
for (it = unodelist->begin(); it != unodelist->end(); it++) 
cout << *it << " ";
cout << endl;
}
*/
    double eps = 1.5e-6;
    // reassign unodes to flynn lists
    for (it = unodelist->begin(); it != unodelist->end(); it++) {
        ElleGetUnodePosition(*it,&xy);
        found=0;
        int adjust=0;
        while ( adjust<3 && !found ) {
        for (f_it = flist->begin(); f_it != flist->end() &&
                                                     !found; f_it++) {
            
            if (EllePtInRegion((*f_it),&xy)) {
                ElleAddUnodeToFlynn((*f_it),(*Unodes)[*it].id());
                ElleSetUnodeFlynn(*it,*f_it);
                /*ElleSetUnodeAttribute((*Unodes)[*it].id(),CONC_A,*/
                                          /*(double)(*f_it));*/
                found=1;
            }
        }
        if (!found) {
            for (f_it = flist->begin(); f_it != flist->end()&&!found; f_it++) {
                list<int> nbflynns;
                ElleFlynnNbRegions(*f_it,nbflynns);
                for (list<int>::iterator tmp = nbflynns.begin();
                               tmp != nbflynns.end()&&!found; tmp++) {
                    if (EllePtInRegion((*tmp),&xy)) {
                        ElleAddUnodeToFlynn((*tmp),(*Unodes)[*it].id());
                        ElleSetUnodeFlynn(*it,*tmp);
                        /*ElleSetUnodeAttribute((*Unodes)[*it].id(),CONC_A,*/
                                          /*(double)(*tmp));*/
                        found=1;
                    }
                }
            }
        } // tried all flynn nbs 
        if (!found) {
cout << xy.x << " " << xy.y << endl;
for (set_int::iterator tmpf_it = flist->begin(); tmpf_it !=
flist->end(); tmpf_it++) {
cout << *tmpf_it << " ";
std::string fname = "flynn";
std::stringstream ss;
ss << (*tmpf_it);
fname+=ss.str();
fname +=".poly";
 ElleWriteFlynnPolyFile(*tmpf_it,(char *)(fname.c_str()));
}
cout << endl;
          if (adjust>1) {
            OnError("ElleAddUnodeToFlynn: unode and flynn not matched",0);
          }
          else if (adjust==1) {
            xy.x -= (eps+eps); // try eps in other direction
            adjust++;
            Log(1,"ElleAddUnodeToFlynn: unode not in a flynn, adjusting unode position");
          }
          else {
            xy.x += eps;
            adjust++;
            Log(1,"ElleAddUnodeToFlynn: unode not in a flynn, adjusting unode position");
          }
        } //!found so adjusting
    } //for each unode
    } //!found and adjusting
}

void ElleFindLocalFlynns(set_int &flist, Coords *bnd, int numpts,
                         Coords *centre)
{
    int i, j, max, rgns[3];
    Coords xy;

    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNodePlotXY(i,&xy,centre);
            if (EllePtInRect(bnd,numpts,&xy)) {
                ElleRegions(i,rgns);
                for (j=0;j<3;j++)
                    if (rgns[j]!=NO_NB) flist.insert(rgns[j]);
            }
        }
    }
}

void ElleUnodeAddVoronoiPt( int id, int index)
{
    if (id>=0 && id <ElleMaxUnodes())
    (*Unodes)[id].addVoronoiPt(index);
}

void ElleUnodeInsertVoronoiPt( int id, int post_index, int index)
{
    if (id>=0 && id <ElleMaxUnodes())
    (*Unodes)[id].insertVoronoiPt(post_index,index);
}

void ElleUnodeDeleteVoronoiPt( int id, int index)
{
    if (id>=0 && id <ElleMaxUnodes())
    (*Unodes)[id].deleteVoronoiPt(index);
}

int ElleUnodeNumberOfVoronoiPts( int id)
{
    return((*Unodes)[id].numberOfVoronoiPts());
}

void ElleUnodeVoronoiPts( int id, list<int> &l)
{
    if (id>=0 && id <ElleMaxUnodes())
    (*Unodes)[id].voronoiPts(l);
}

void ElleUnodeAddNearNb( int id, int nb)
{
    if (id>=0 && id <ElleMaxUnodes() &&
        nb>=0 && nb <ElleMaxUnodes())
        (*Unodes)[id].addNearNb(nb);
}

int ElleUnodeNumberOfNearNbs( int id)
{
    return((*Unodes)[id].numberOfNearNbs());
}

void ElleSetUnodeFlynn( int id, int flynnid)
{
    (*Unodes)[id].setFlynn(flynnid);
}

int ElleUnodeFlynn( int id)
{
    return((*Unodes)[id].flynn());
}

void ElleGetUnodeNearNbs( int id, vector<int> &nbs)
{
    (*Unodes)[id].getNearNbsList(nbs);
}

void ElleUnodeWriteNearNbs( std::ostream & os)
{
    int i;
    int max = ElleMaxUnodes();
    for (i=0;i<max;i++) {
        os << i << "  ";
        (*Unodes)[i].writeNearNbList(os);
    }
}

/*!
  This function is for the voronoi of all unodes in the unitcell.
  It is separate from the fn to voronoi unodes within a flynn as
  a rectangular unitcell is assumed.
 */                                                                                
int ElleVoronoiUnodes()
{
    int err=0;
    int i, k, id;
    Coords tmp;
    Coords vorbnd[4], bnd[4];
    struct triangulateio out, vorout;
    CellData udata;

    ElleCellBBox(&udata);
    for (i=0;i<4;i++) bnd[i] = udata.cellBBox[i];
/*
    if (roi<(udata.xoffset*0.2)) roi = udata.xoffset*0.2;
*/
    //  set the limit of the cell as the region -
    //  VoronoiRegionUnodes will add margins
    double roi = 0.0;
    bnd[0].x -= roi;
    bnd[0].y -= roi;
    bnd[1].x += roi;
    bnd[1].y -= roi;
    bnd[2].x += roi;
    bnd[2].y += roi;
    bnd[3].x -= roi;
    bnd[3].y += roi;
    int numpts = 4;

    err = VoronoiRegionUnodes(bnd,numpts,&out,&vorout);

/* vorbnd should match the region used in  VoronoiRegionUnodes */
    roi = ElleUnodeROI()*2; // assumes constant
    if (roi<(udata.xoffset*0.2)) roi = udata.xoffset*0.2;

    vorbnd[0].x = bnd[0].x - roi;
    vorbnd[0].y = bnd[0].y - roi;
    vorbnd[1].x = bnd[1].x + roi;
    vorbnd[1].y = bnd[1].y - roi;
    vorbnd[2].x = bnd[2].x + roi;
    vorbnd[2].y = bnd[2].y + roi;
    vorbnd[3].x = bnd[3].x - roi;
    vorbnd[3].y = bnd[3].y + roi;
    V_data->vsegs.clear();
    V_data->vpoints.clear();
/*
  Get rid of norms and points with large x or y values as they can cause
  program to be stuck in PlotXY fnsi (repositiioning to unitcell).
  Can't rely on attribute of voronoi pts if point has been added, attribute
  is interpolated from tri pts
  vorout.pointattributelist is a pointer to out.pointattributelist
  Use pt in polyon bnd
    for (k=0;k<vorout.numberofedges;k++) {
        i=(int)vorout.pointattributelist[vorout.edgelist[k*2]];
        if (i>=4*ElleMaxUnodes()||i<=-4*ElleMaxUnodes()) {
            vorout.edgelist[k*2]=-vorout.numberofpoints;
            printf("%d %d -> -1\n",i,vorout.edgelist[k*2]);
        }
        i=(int)vorout.pointattributelist[vorout.edgelist[k*2+1]];
        if (i>=4*ElleMaxUnodes()||i<=-4*ElleMaxUnodes()) {
            vorout.edgelist[k*2+1]=-vorout.numberofpoints;
            printf("%d %d -> -1\n",i,vorout.edgelist[k*2]+1);
        }
    }
*/
    for (i=0;i<vorout.numberofpoints;i++) {
        tmp.x = vorout.pointlist[i*2];
        tmp.y = vorout.pointlist[i*2+1];
        if (!EllePtInRect(vorbnd,4,&tmp)) {
            id=-1;
        }
        else {
            id = ElleAddVoronoiPt(tmp);
        }
        if (id!=i) {
              for (k=0;k<vorout.numberofedges;k++) {
                if (vorout.edgelist[k*2]==i) vorout.edgelist[k*2]=id;
                if (vorout.edgelist[k*2+1]==i) vorout.edgelist[k*2+1]=id;
              }
        }
    }
/*
for (k=0;k<vorout.numberofedges;k++) {
if (vorout.edgelist[k*2]==295||vorout.edgelist[k*2+1]==295||
vorout.edgelist[k*2]==291||vorout.edgelist[k*2+1]==291)
printf("After loop -edge = %d, end1 = %d, end2 = %d\n",k,vorout.edgelist[k*2],
vorout.edgelist[k*2+1]);
}
*/
    for (i=0;i<vorout.numberofedges;i++) {
        if (vorout.edgelist[i*2]!=-1 &&
			vorout.edgelist[i*2+1]!=-1) {
          V_data->vsegs.push_back(vorout.edgelist[i*2]);
          V_data->vsegs.push_back(vorout.edgelist[i*2+1]);
        }
    }
	/*! for each unode, find nearest nbs */
    findUnodeNeighbours(&out,&vorout,bnd,numpts);
	/*!
     *  find voronoi regions, match to a unode,
     *  save anticlockwise ordered voronoi point list
     */
    findUnodeArea(V_data);
    return(err);
}

int ElleAddVoronoiPt(const Coords pt)
{
    int i=NO_VAL;
    Coords relpos, tmp;
    if (ElleFindVoronoiPt(pt,&i)==false) {
        relpos.x = pt.x; relpos.y = pt.y;
        ElleNodeUnitXY(&relpos);
        VoronoiPt tmppt(&relpos);
        V_data->vpoints.push_back(tmppt);
        i = V_data->vpoints.size() - 1;
    }
    return(i);
}

bool ElleFindVoronoiPt(const Coords pt, int *index)
{
    int i=0;
    double eps=1.5e-6; // same eps as crossings.cc
    Coords relpos, tmp;
    *index = NO_VAL;
    bool matched = false;
    vector<VoronoiPt>::iterator itvc=V_data->vpoints.begin();
    while (itvc!=V_data->vpoints.end() && !matched) {
        relpos.x = pt.x; relpos.y = pt.y;
        itvc->getPosition(&tmp);
        ElleNodeUnitXY(&relpos);
        ElleCoordsPlotXY(&tmp,&relpos);
        if (fabs(tmp.x-relpos.x)<eps && fabs(tmp.y-relpos.y)<eps) {
            matched = true;
        }
        else {
            itvc++;
            i++;
        }
    }
    if (matched) *index = i;
    return(matched);
}

int WriteTriangulation(struct triangulateio *out,const char *name);
#ifdef DEBUG
#endif
int VoronoiRegionUnodes(Coords *bnd,int numpts,struct triangulateio *out,
				struct triangulateio *vorout)
{
	int err=0;
    char fname[81];
    int num, i, j;
    int num_tripts=0, num_brdrpts=0;
    int xflags[2],yflags[2],xflags_brdr[2],yflags_brdr[2];
    double roi;
    double start=0.0;
    struct flagvals triflags;
    Coords current, prev, min, max, min_brdr, max_brdr, brdr[4];
    Coords wrap[9];
    CellData unitcell;
    vector<Coords> checkUnits;
	vector<int> :: iterator it;

	vector<int> brdrlist;
    num = ElleMaxUnodes();
    double *bndxvals = new double[numpts];
    double *bndyvals = new double[numpts];
    double *brdrxvals = new double[numpts];
    double *brdryvals = new double[numpts];
    for (i=0; i<numpts; i++) {
        bndxvals[i] = bnd[i].x;
        bndyvals[i] = bnd[i].y;
    }
    ElleCellBBox(&unitcell);
    roi = ElleUnodeROI()*2; // assumes constant
    if (roi<(unitcell.xoffset*0.2)) roi = unitcell.xoffset*0.2;
//this is for a rectangle - need to generalise for polygon
	brdr[0].x = brdrxvals[0] = bnd[0].x - roi;
	brdr[0].y = brdryvals[0] = bnd[0].y - roi;
	brdr[1].x = brdrxvals[1] = bnd[1].x + roi;
	brdr[1].y = brdryvals[1] = bnd[1].y - roi;
	brdr[2].x = brdrxvals[2] = bnd[2].x + roi;
	brdr[2].y = brdryvals[2] = bnd[2].y + roi;
	brdr[3].x = brdrxvals[3] = bnd[3].x - roi;
	brdr[3].y = brdryvals[3] = bnd[3].y + roi;
    ElleCheckUnit(brdrxvals,brdryvals,numpts,xflags_brdr,yflags_brdr,
                   &min_brdr,&max_brdr);
    Coords pos,pos2;
    current.x = unitcell.cellBBox[0].x;
    current.y = unitcell.cellBBox[0].y;
    checkUnits.push_back(current);
    pos.x = current.x - unitcell.xlength;
    pos.y = current.y;
    checkUnits.push_back(pos);
    pos.x = current.x + unitcell.xlength;
    pos.y = current.y;
    checkUnits.push_back(pos);
    pos.x = current.x - unitcell.xoffset;
    pos.y = current.y - unitcell.ylength;
    pos2.y = current.y; pos2.x=pos.x+unitcell.xlength;
    while( ( pos.x<=max_brdr.x && pos.x>=min_brdr.x ) ||
            ( pos2.x<=max_brdr.x && pos2.x>=min_brdr.x )) {
        checkUnits.push_back(pos);
        pos2.x -= unitcell.xlength;
        pos.x -= unitcell.xlength;
    }
    pos.x = current.x - unitcell.xoffset + unitcell.xlength;
    pos.y = current.y - unitcell.ylength;
    pos2.y = current.y; pos2.x=pos.x;
    while( ( pos.x<=max_brdr.x && pos.x>=min_brdr.x ) ||
            ( pos2.x<=max_brdr.x && pos2.x>=min_brdr.x )) {
        checkUnits.push_back(pos);
        pos2.x += unitcell.xlength;
        pos.x += unitcell.xlength;
    }
    pos.x = current.x + unitcell.xoffset;
    pos.y = current.y + unitcell.ylength;
    pos2.y = pos.y; pos2.x=pos.x+unitcell.xlength;
    while( ( pos.x<=max_brdr.x && pos.x>=min_brdr.x ) ||
            ( pos2.x<=max_brdr.x && pos2.x>=min_brdr.x )) {
        checkUnits.push_back(pos);
        pos2.x -= unitcell.xlength;
        pos.x -= unitcell.xlength;
    }
    pos.x = current.x + unitcell.xoffset + unitcell.xlength;
    pos.y = current.y + unitcell.ylength;
    pos2.y = pos.y; pos2.x=pos.x+unitcell.xlength;
    while( ( pos.x<=max_brdr.x && pos.x>=min_brdr.x ) ||
            ( pos2.x<=max_brdr.x && pos2.x>=min_brdr.x )) {
        checkUnits.push_back(pos);
        pos2.x += unitcell.xlength;
        pos.x += unitcell.xlength;
    }
    for (i=0; i <num; i++) {
        ElleGetUnodePosition((i),&current);
        prev = current;
        for (j=0;j<checkUnits.size(); j++) {
            current.x = prev.x+checkUnits[j].x;
            current.y = prev.y+checkUnits[j].y;
    	    if (current.x>=min_brdr.x && current.x<=max_brdr.x &&
        	      current.y>=min_brdr.y && current.y<=max_brdr.y &&
                  EllePtInRegion(brdr,numpts,&current)) {
                brdrlist.push_back(i);
                brdrlist.push_back(j);
                num_brdrpts++;
            }
        }
    }
    double *xvals = new double[num_brdrpts];
    double *yvals = new double[num_brdrpts];
    double *attrib = new double[num_brdrpts];
    for (it = brdrlist.begin(), i=0; it != brdrlist.end(); it++,i++) {
        ElleGetUnodePosition((*it),&current);
        attrib[i] = *it+1;
        it++;
        xvals[i] = current.x+checkUnits[*it].x;
        yvals[i] = current.y+checkUnits[*it].y;
/*
 * this marks the pt if it is outside the unit cell
 * If region wraps or region is unit cell, unodes need to
 * identify nbs, so don't mark negative.
 * NearNb list is a set so won't contain multiple entries
 */
        if (*it > 0) attrib[i] *= -1;
    }
    triflags.area = 0;
    triflags.quality = 0;
    triflags.midptnodes = 0;
    triflags.bndpts = 0;
    triflags.voronoi = 1;
    if (trivorpolypts(out, vorout,&triflags, 0,
                   xvals, yvals, attrib, num_brdrpts, 1))
        OnError("error triangulating",0);

#ifdef DEBUG
    WriteTriangulation(out,"tri.poly");
    WriteVoronoiAsPoly(vorout);
    exit(0);
#endif
    num_tripts = out->numberofpoints;
    num_tripts = vorout->numberofpoints;
    delete[] xvals;
    delete[] yvals;
    delete[] attrib;
	return(err);
}

int findUnodeNeighbours(struct triangulateio *out, 
                        struct triangulateio *vorout,
                        Coords *bnd, int num)
{
    int i,j,k,id[3],tri_id[3];
    Coords xy;
    CellData unitcell;

    /*
     * this should look at the voronoi edges
     * if a vor edge has the unode id as one vertex then
     * the other is a nb  (unless the edge is a ray)
     *
     * the current implementation just looks at the triangulation 
     * vertices and can miss nbs?
     */
    for (i=0;i<out->numberoftriangles;i++) {
        k=out->trianglelist[i*out->numberofcorners];
        for (j=0;j<3;j++) {
           tri_id[j] = out->trianglelist[i*out->numberofcorners+j];
           id[j] = (int)
             out->pointattributelist[tri_id[j]*
                                     out->numberofpointattributes];
           int incr = ((id[j]>=0) ? -1 : 1);
           id[j] += incr;
        }
        for (j=0;j<3;j++) {
           /* check that nb is also in region - fix */
           if (id[j]>=0) {
           ElleUnodeAddNearNb(abs(id[j]),abs(id[(j+1)%3]));
           ElleUnodeAddNearNb(abs(id[j]),abs(id[(j+2)%3]));
           }
           else {
           if (id[(j+1)%3]>=0)
           ElleUnodeAddNearNb(abs(id[j]),id[(j+1)%3]);
           if (id[(j+2)%3]>=0)
           ElleUnodeAddNearNb(abs(id[j]),id[(j+2)%3]);
           }
        }
    }
    /*ElleUnodeWriteNearNbs(cout);*/
    ElleCellBBox(&unitcell);
    double xlimit = unitcell.xlength/2.0;
    double ylimit = unitcell.ylength/2.0;
    double xdiff=0, ydiff=0, eps=1.0e-6; 
    for (j=0;j<out->numberofedges;j++) {
        id[0] = (int)
                 out->pointattributelist[out->edgelist[j*2]*
                                         out->numberofpointattributes]-1;
        id[1] = (int)
                 out->pointattributelist[out->edgelist[j*2+1]*
                                         out->numberofpointattributes]-1;
        int end1 =  vorout->edgelist[j*2];
        int end2 =  vorout->edgelist[j*2+1];
        if (id[0]>=0) {
           if (end1!=-1)
                ElleUnodeAddVoronoiPt(abs(id[0]),end1);
            if (end2!=-1)
                ElleUnodeAddVoronoiPt(abs(id[0]),end2);
        }
        if (id[1]>=0) {
            if (end1!=-1)
                ElleUnodeAddVoronoiPt(abs(id[1]),end1);
            if (end2!=-1)
                ElleUnodeAddVoronoiPt(abs(id[1]),end2);
        }
    }
#ifdef DEBUG
/*for (j=0;j<ElleMaxUnodes();j++) {*/
for (j=0;j<1;j++) {
        list<int> pt_list;
    list<int>::iterator it;
        ElleUnodeVoronoiPts(j,pt_list);
        for (it=pt_list.begin();it!=pt_list.end();it++) {
            std::cout << *it << ' ';
        }
std::cout <<std::endl;
    }
#endif
}

int findUnodeArea( VoronoiData *vdata )
{
    int i,j,err=0;
    int maxu;
    int unode_vpts;
    double area=0;
    Coords xy, vxy;
    list<int>::iterator it;
    list< pair<double,int> >ordered_list;
    list< pair<double,int> >::iterator ito;

    maxu = ElleMaxUnodes();
    for (i=0;i<maxu;i++) {
        ordered_list.clear();
        list<int> pt_list;
        pt_list.clear();
        ElleUnodeVoronoiPts(i,pt_list);
        ElleGetUnodePosition(i,&xy);
        unode_vpts=pt_list.size();
        for (it=pt_list.begin();it!=pt_list.end();it++) {
            vxy.x = vxy.y = 0.0;
            vdata->vpoints[*it].getPosition(&vxy);
            ElleCoordsPlotXY(&vxy,&xy);
            pair<double,int> tmp(polar_angle(vxy.x-xy.x,vxy.y-xy.y),
                                 *it);
            ito=ordered_list.begin();
            while (ito!=ordered_list.end() &&
                               tmp.first > (*ito).first)ito++;
                ordered_list.insert(ito,tmp);
            (*Unodes)[i].deleteVoronoiPt(*it);
        }
        vector<Coords> vpolypts(unode_vpts);
        for (ito=ordered_list.begin(),j=0;ito!=ordered_list.end();
                                                   ito++,j++) {
            (*Unodes)[i].addVoronoiPt(ito->second);
#ifdef DEBUG
if (i==73) {

pt_list.clear();
ElleUnodeVoronoiPts(i,pt_list);
for (it=pt_list.begin();it!=pt_list.end();it++) {
cout << *it << ' ';
}
cout << endl;
}
#endif
            vxy.x = vxy.y = 0.0;
            vdata->vpoints[ito->second].getPosition(&vxy);
            ElleCoordsPlotXY(&vxy,&xy);
            vpolypts[j] = vxy;
        }
        area = polyArea(vpolypts);
        ElleSetUnodeArea(i,area);
    }
}

/*!
Gets the triangulation and the voronoi of the unodes and
a rectangular boundary which encloses the flynn.
Uses the flynn boundary as the clipping polygon.
Finds the unode neighbours with the polygon and the
unode areas after the voronoi is clipped by the polygon
If a list of unodes is passed, only the voronoi regions of
these unodes are included in the V_data.
*/

int VoronoiFlynnUnodes(int flynnid)
{
    vector<int> all_unodes;
    ElleGetFlynnUnodeList(flynnid,all_unodes);
    return(VoronoiFlynnUnodes(flynnid,all_unodes));
}

int VoronoiFlynnUnodes(int flynnid, vector<int> &sublist)
{
    int err=0;
    int *ids, num, i, num_limit=4;
    int num_bnd_nodes,xflags[2],yflags[2];
    struct flagvals triflags;
    Coords current, prev, min, max, rel_pos;
    CellData unitcell;
    struct triangulateio out, vorout;
    vector<int> :: iterator it;
                                                                                
    ElleFlynnNodes(flynnid,&ids,&num_bnd_nodes);
    vector<int> unodelist;
    ElleGetFlynnUnodeList(flynnid,unodelist);
    num = num_limit + unodelist.size();
    double *xvals = new double[num];
    double *yvals = new double[num];
    double *attrib = new double[num];
    double *bndxvals = new double[num_bnd_nodes];
    double *bndyvals = new double[num_bnd_nodes];
    Coords *bnd = new Coords[num_bnd_nodes];
    ElleNodePosition(ids[0],&prev);
    for (i=0; i<num_bnd_nodes; i++) {
        ElleNodePlotXY(ids[i],&current,&prev);
        bndxvals[i] = current.x;
        bndyvals[i] = current.y;
        bnd[i] = current;
        prev = current;
    }
    ElleCheckUnit(bndxvals,bndyvals,num_bnd_nodes,xflags,yflags,&min,&max);
    ElleCellBBox(&unitcell);
    xvals[0] = min.x-unitcell.xlength/2.0;
    yvals[0] = min.y-unitcell.ylength/2.0;
    attrib[0] = -1;
    xvals[1] = max.x+unitcell.xlength/2.0;
    yvals[1] = min.y-unitcell.ylength/2.0;
    attrib[1] = -1;
    xvals[2] = max.x+unitcell.xlength/2.0;
    yvals[2] = max.y+unitcell.ylength/2.0;
    attrib[2] = -1;
    xvals[3] = min.x-unitcell.xlength/2.0;
    yvals[3] = max.y+unitcell.ylength/2.0;
    attrib[3] = -1;
    i = 4;
    /*
     * reposition unodes into current flynn position
     */
    for (it = unodelist.begin(); it != unodelist.end(); it++,i++) {
        ElleGetUnodePosition((*it),&current);
        if (!ElleCoordRelativePosition(&current,
                                        bnd,num_bnd_nodes,
                                        xflags,yflags,
                                        &rel_pos) ) {
            int k;
            printf("Problem %d %lf %lf\n",(*it),current.x,current.y);
            printf("flags %d %d %d %d\n",xflags[0],xflags[1],yflags[0],yflags[1]);
            for (k=0;k<num_bnd_nodes;k++)
                printf("%d %lf %lf\n",k,bndxvals[k],bndyvals[k]);
            OnError("VoronoiFlynnUnodes-position error",0);
        }
        xvals[i] = rel_pos.x;
        yvals[i] = rel_pos.y;
        attrib[i] = *it;
    }

    initio(&out);
    initio(&vorout);
    triflags.area = 0;
    triflags.quality = 0;
    triflags.midptnodes = 0;
    triflags.bndpts = 0;
    triflags.voronoi = 1;
    if (trivorpolypts(&out, &vorout,&triflags, num_limit,
                   xvals, yvals, attrib, num, 1))
        OnError("error triangulating",0);
#ifdef DEBUG
    WriteVoronoiAsPoly(&vorout);
    WriteTriangulation(&out,"tri.poly");
    exit(0);
#endif

    err = findFlynnUnodeNeighbours(&out);
    err = findFlynnUnodeArea(flynnid,&out,&vorout,bnd,num_bnd_nodes,
                             xflags, yflags,sublist);

    cleanio(&out);
    cleanio(&vorout);
    delete[] xvals;
    delete[] yvals;
    delete[] bndxvals;
    delete[] bndyvals;
    delete[] attrib;
    delete[] bnd;
    if (ids) free(ids);
    return(err);
}

/*!
  Uses the unode triangulation to determine the nbs
 */
int findFlynnUnodeNeighbours(struct triangulateio *out)
{
    int err=0;
    int i, j, k, tri_id[3], id[3];
                             
    for (i=0;i<out->numberoftriangles;i++) {
        k=out->trianglelist[i*out->numberofcorners];
        for (j=0;j<3;j++) {
           tri_id[j] = out->trianglelist[i*out->numberofcorners+j];
           id[j] = (int)
             out->pointattributelist[tri_id[j]*
                                     out->numberofpointattributes];
        }
        for (j=0;j<3;j++) {
            if (id[j]>=0) {
              if (id[(j+1)%3]>=0)
                ElleUnodeAddNearNb(id[j],id[(j+1)%3]);
              if (id[(j+2)%3]>=0)
                ElleUnodeAddNearNb(id[j],id[(j+2)%3]);
            }
        }
    }
    return(err);
}
/*!
  Uses the triangulation and voronoi.
  Clips the voronoi of the unode against the flynn boundary (bnd)
  Adds the clip region points to the flynn's VoronoiData
  Each unode stores the indices into the flynn VoronoiData
  The area of the clip region is the unode area
 */ 
int findFlynnUnodeArea(int flynnid,struct triangulateio *out,
                             struct triangulateio *vorout,
                             Coords *bnd, int num_bnd_nodes, 
                             int *xflags, int *yflags,
                             vector<int> &unodelist)
{
    int err=0;
    int i, j, k, n, index;
    int nbs[3], nb_flynn;
    int num_attrib =  out->numberofpointattributes;
    int u_id;
    Coords tmp_c, current, rel_pos;
    vector<Coords> polybnd(bnd,bnd+num_bnd_nodes);
    vector< vector<Coords> > res;
    vector<Coords> :: iterator itc;
    vector<set_int> tmpnodes(out->numberofpoints);
    set_int :: iterator its;

    /*
     * Find the voronoi edges for all the unodes.
     * Uses the Triangle relationship that a Delaunay edge i
     * is the dual of the Voronoi edge i.
     */
    for (i=0;i<out->numberofedges;i++) {
        k = out->edgelist[i*2];
        u_id = (int)(out->pointattributelist[k*num_attrib]);
        if (u_id >= 0 && (find(unodelist.begin(),unodelist.end(),u_id)
                                != unodelist.end())) {
          tmpnodes[k].insert(vorout->edgelist[i*2]);
          tmpnodes[k].insert(vorout->edgelist[i*2+1]);
        }
        k = out->edgelist[i*2+1];
        u_id = (int)(out->pointattributelist[k*num_attrib]);
        if (u_id >= 0 && (find(unodelist.begin(),unodelist.end(),u_id)
                                != unodelist.end())) {
          tmpnodes[k].insert(vorout->edgelist[i*2]);
          tmpnodes[k].insert(vorout->edgelist[i*2+1]);
        }
    }
    /*!
      Clip the voronoi against the flynn boundary and
      store the points and area
     */
    vector<int> ids;
    ElleFlynnNodes(flynnid,ids);
          // may need to insert elements so use list
    list< std::pair<int,Coords> > flynnnodes;
    std::pair<int,Coords> tmppair, prev;
    for (j=0;j<num_bnd_nodes;j++)  {
        tmppair.first = ids[j];
        tmppair.second = polybnd[j];
        flynnnodes.push_back(tmppair);
    }
    for (i=0; i<tmpnodes.size(); i++) {
        if (tmpnodes[i].size()>0) {
          u_id = (int)(out->pointattributelist[i*num_attrib]);
          /*
           * vpoly is the voronoi region for the unode, u_id
           */
          vector<Coords> vpoly;
          for (its=tmpnodes[i].begin(); its!=tmpnodes[i].end(); its++) {
            k=*its;
            tmp_c.x = vorout->pointlist[k*2];
            tmp_c.y = vorout->pointlist[k*2+1];
            vpoly.push_back(tmp_c);
          }
          /*
           * vpoly needs to be ordered (anticlockwise)
           * around the unode. Reposition the unode,
           * if necessary
           */
          ElleGetUnodePosition(u_id, &current);
          ElleCoordRelativePosition(&current,
                                        bnd,num_bnd_nodes,
                                        xflags,yflags,
                                        &rel_pos);
          sortCoordsOnAngle(vpoly,&rel_pos);
          /*
           * res holds the resulting clip regions
           * There should only be one but sometimes other tiny
           * regions are returned. This could possibly be avoided
           * by altering the GPC epsilon but I haven't tested it
           */
          gpcclip(vpoly,polybnd,res,GPC_INT);
          index=0;
          if (res.size()>1) {
            double max_area = 0.0, area = 0.0;
            for (j=0; j<res.size(); j++) {
              area = fabs(polyArea(res[j]));
              if (area > max_area) {max_area=area; index=j;}
            }
          }
          /*
           * If a voronoi point is on a flynn segment,
           * make the voronoi pt a Bnode (dj) so that it is included
           * in the bnd pts for the neighbouring flynn (assumes the
           * flynns are processed in numerical order).
           * This is necessary for the voronoi2flynn code
           */
          //double eps=1.0e-6; // same eps as crossings.cc
          double eps=1.5e-6; // same eps as crossings.cc
          Coords relpos, newpt;
          bool matched = false;
          // returned poly is clockwise
          // use reverse iterator?
          list<std::pair<int,Coords> >::iterator it=flynnnodes.begin();
          for (j=res[index].size()-1; j>=0; j--) {
            /*
             * Any matching voronoi point is marked as a Bnode
             * May have problems with coincident unodes or clip pts
             * If so, check separation before adding pts?
             */
          	  matched = false;
          	  nb_flynn = NO_NB;
              relpos.x = res[index][j].x; relpos.y = res[index][j].y;
              for (it=flynnnodes.begin(); it!=flynnnodes.end() && !matched;
                                                                   it++) {
                  if (fabs((it->second).x-relpos.x)<eps &&
                      fabs((it->second).y-relpos.y)<eps) {
                      matched = true;
                  }
              }
              prev = flynnnodes.back();
              it=flynnnodes.begin();
              while (it!=flynnnodes.end() && !matched) {
                  if ( PntOnLine( (int)((prev.second).x /eps),
                                           (int)((prev.second).y /eps),
                                           (int)((it->second).x /eps),
                                           (int)((it->second).y /eps),
                                            (int)(relpos.x /eps),
                                            (int)(relpos.y /eps)) ==2) {
                      int dj;
                      ElleNodeUnitXY(&relpos);
                      ElleInsertDoubleJAtPoint(prev.first,it->first,
                                               &dj,&relpos);
                      tmppair.first = dj;
                      tmppair.second = relpos;
                      ElleCoordsPlotXY(&(tmppair.second),&(prev.second));
                      flynnnodes.insert(it,tmppair);
                      /*
                       * if nb flynn of dj has already been processed
                       * (i.e. id<flynnid) then this voronoi pt has to
                       * be added to the list of pts of one of the
                       * nb flynn's unodes
                       */
                      ElleRegions(dj,nbs);
                      for (k=0;k<3;k++)
                          if (nbs[k]!=NO_NB && nbs[k]!=flynnid)
                              nb_flynn = nbs[k];
                      matched = true;
                  }
                  else {
                      prev = *it;
                      it++;
                  }
              }
            /*
             * this call adds the point to the flynn VoronoiData
             * or finds a matching point
             * The return value is an index into the point list
             */
              n = ElleAddVoronoiPt(res[index][j]);
              if (n!=NO_VAL) {
                  ElleUnodeAddVoronoiPt( u_id,n);
                  if (matched) {
                      V_data->vpoints[n].markAsBnode();
                      if (nb_flynn!=NO_VAL) {
                          int prevnode, nextnode;
                          if ( (ElleFindVoronoiPt(prev.second,&prevnode)==true) &&
                               (ElleFindVoronoiPt(it->second,&nextnode)==true) )
                            updateUnodeVoronoi(nextnode, prevnode, n, nb_flynn);
                      }
                  }
              }
          }
          ElleSetUnodeArea(u_id,fabs(polyArea(res[index])));
          for (j=0; j<res.size(); j++) res[j].clear();
          res.clear();
        }
    }
    
    return(err);
}

                                                                                
void updateUnodeVoronoi(int pt1, int pt2, int newpt, int rgn)
{
    int i, prev;
    bool inserted;
    list<int> pt_list;
    list<int> :: iterator itp;
    vector<int> l;
    ElleGetFlynnUnodeList(rgn,l);
    for (vector<int>::iterator it=l.begin();
                                      it!=l.end();it++) {
        pt_list.clear();
        ElleUnodeVoronoiPts(*it,pt_list);
        inserted = false;
        itp = pt_list.begin();
        prev = pt_list.back();
        while (itp!=pt_list.end() && !inserted) {
          if (prev==pt1 && *itp==pt2) {
            ElleUnodeInsertVoronoiPt(*it,pt2,newpt);
            inserted = true;
          }
          else if (prev==pt2 && *itp==pt1) {
            ElleUnodeInsertVoronoiPt(*it,pt1,newpt);
            inserted = true;
          }
          else {
            prev = *itp;
            itp++;
          }
        }
    }
}

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
#if XY
/*!
 * No assumptions about the spatial distribution of unodes
 * Assume wrapping is allowed
 * Locate the closest unode to the position ptxy, starting with
 * the unode at index=start
 */
int FindUnode(Coords *ptxy, int start)
{
    int id=NO_VAL;
    id = FindUnode(ptxy, start, true);
    return(id);
}
#endif
int FindUnode(Coords *ptxy, int start)
{
    bool min_set=false;
    int i, j, id=NO_VAL;
    int max_unodes = ElleMaxUnodes();
    Coords refxy, xy;
    double min=0.0, dist_sq=0;
    double roi = ElleUnodeROI();
    for (i=start;i<max_unodes;i++)   // cycle through unodes
    {
        ElleGetUnodePosition(i,&refxy);
        // only check if unode in rect 3*roi around ptxy??
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
 * If there are flynns, only check the unodes for the flynn the 
 * pt lies within
 * Locate the closest unode to the position ptxy, starting with
 * the unode at index=start
 * Allow wrapping to be turned off
 */
int FindUnode(Coords *ptxy, int start, bool wrap)
{
    bool min_set=false;
    int i, j, k, id=NO_VAL;
    int max_unodes = ElleMaxUnodes();
    Coords refxy, xy;
    double dist_min=0.0, dist_sq=0;
    vector <int> unodelist;

    // ElleNodeUnitXY(&ptxy); Point relative to unit cell
   
    for (i=0;i<ElleMaxFlynns();i++)
    {
      if (ElleFlynnIsActive(i))
      { 
          if (EllePtInRegion(i,ptxy))
          {
             ElleGetFlynnUnodeList(i,unodelist);
             for(k=0;k<unodelist.size();k++)
             {
                 ElleGetUnodePosition(unodelist[k],&refxy);
                 if (wrap) ElleCoordsPlotXY (&refxy, ptxy);
                 dist_sq = (ptxy->x-refxy.x)*(ptxy->x-refxy.x) +
                           (ptxy->y-refxy.y)*(ptxy->y-refxy.y);
 
                 if (!min_set) {
                     id = unodelist[k];
                     dist_min = dist_sq;
                     min_set = true;
                 }
                 else if (dist_sq<dist_min)
                 {
                     id = unodelist[k];
                     dist_min=dist_sq;
                 }
              }
           }   
        }  
    }

    if (id ==NO_VAL) {
      min_set=false;
      // alternative, scan all the unodes
      for (i=start;i<max_unodes;i++)   // cycle through unodes
      {
         ElleGetUnodePosition(i,&refxy);
         // only check if unode in rect 3*roi around ptxy??
         if (wrap)
             ElleCoordsPlotXY(&refxy,ptxy); //should we assume wrapping??
         dist_sq = (ptxy->x-refxy.x)*(ptxy->x-refxy.x) +
                   (ptxy->y-refxy.y)*(ptxy->y-refxy.y);
         if (!min_set) {
             id = i;
             dist_min = dist_sq;
             min_set = true;
         }
         else if (dist_sq<dist_min) {
             id = i;
             dist_min = dist_sq;
         }
       }
     }
     return(id);
}
#if XY
/*
 * there is another version in interface which adds the 
 * mass by flynn unode list
 * therefore, it also checks that all unodes are assigned
 * but area calc is invalid if any unodes not assigned ?
 */
double ElleTotalUnodeMass(int attr_id)
{
    int j, max;
    double conc, total=0, area;

    if (ElleUnodesActive() && UnodeAttributeActive(attr_id)) {
        max = ElleMaxUnodes();
        area = 1.0*ElleUnitLength()*ElleUnitLength()/max;
        for (j=0;j<max;j++) {
            ElleGetUnodeAttribute(j,&conc,attr_id);
            total += conc*area;
        }
    }
    return(total);
}
void ElleFindLocalUnodes(set_int &flist, list<int> *ulist,
                         Coords *bnd, int numpts, Coords *centre)
{
    int i, max;
    Coords xy;

    if (Unodes) {
        for (set_int::iterator f_it=flist.begin();
                                      f_it!=flist.end();f_it++) {
				vector<int> l;
            ElleGetFlynnUnodeList(*f_it,l);
            for (vector<int>::iterator it=l.begin();
                                      it!=l.end();it++) {
                (*Unodes)[(*it)].getPosition(&xy);
                ElleCoordsPlotXY(&xy,centre);
                if (EllePtInRect(bnd,numpts,&xy)) {
                    ulist->push_back((*Unodes)[(*it)].id());
                    ElleRemoveUnodeFromFlynn(*f_it,(*Unodes)[(*it)].id());
                    ElleSetUnodeFlynn((*Unodes)[(*it)],NO_VAL);
                }
            }
        }
    }
}
/*
        xy.x = out->pointlist[k*2];
        xy.y = out->pointlist[k*2+1];
        if (EllePtInRegion(bnd,num,&xy)) {
*/
#endif
