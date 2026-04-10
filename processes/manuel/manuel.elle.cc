#include <stdio.h>
#include <math.h>
#include <string.h>
#include <vector>
#include "nodes.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "string_utils.h"
#include "runopts.h"
#include "file.h"
#include "init.h"
#include "unodes.h"
#include "interface.h"
#include "update.h"
#include "stats.h"
#include "manuel.h"

using std::vector;

typedef vector<double> vecdbl;

int InitSS();
int Simple_Shear();
void SaveNodeMassValues(vector<vecdbl *> &M);
void CalculateConcentration(vector<vecdbl *> &M);

int InitSS()
{
    int err=0;
    int max;
    char *infile;

    ElleReinit();
    ElleSetRunFunction(Simple_Shear);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
	return(err);
}


int Simple_Shear()
{
    int i, j, k, n;
    int st_interval=0,interval=0,err=0,max,iter=1;
    int ndj, ntj, ntot;
    double y1, y2, v1, v2, offset, cum_offset, timestep;
    const double vmax = 0.1;
    double tmp, rate;
    Coords incr,xy,curr;
    UserData udata;
    CellData        unitcell;

double mass,startmass;
    ElleUserData(udata);
    rate = udata[XVelocity]; /*  shear strain rate */
    timestep = ElleTimestep();
    if (rate==0.0) v1 = -0.01; /* default value if not set by user */
    else {
        ElleSetStages(1);
        v1 = -rate * timestep;
        while (fabs(v1)>vmax) { v1 /= 10.0; iter *=10; }
    }
    v1 *= 0.5; /* vel on min y boundary */
    v2 = -v1; /* vel on max y boundary */

    incr.y = 0;
    ElleCellBBox(&unitcell);
    y1 = unitcell.cellBBox[BASELEFT].y;
    y2 = unitcell.cellBBox[TOPLEFT].y;

    if (ElleCount()&&ES_statsInterval()>0) ES_WriteStatistics(0);
    if (ElleDisplay()) ElleUpdateDisplay();

    for (i=0;i<EllemaxStages();i++) {
      vector<vecdbl *> mass_array;
      if (ElleNodeCONCactive()) {
        /*
         * store node mass attributes
         */
          SaveNodeMassValues(mass_array);
      }
      ElleCellBBox(&unitcell);
      cum_offset = ElleCumSSOffset();
      /*
       * move nodes
       */
      for (n=0;n<iter;n++) {
        max = ElleMaxNodes();
        for (k=0;k<max;k++) {
            if (ElleNodeIsActive(k)) {
                ElleNodePosition(k,&xy);
                curr.x = xy.x + ((xy.y - y1)*v2 + (y2 - xy.y)*v1);
                curr.y = xy.y;
                ElleCopyToPosition(k,&curr);
            }
        }
        if (ElleUnodesActive()) {
            max = ElleMaxUnodes();
            for (k=0;k<max;k++) {
                ElleGetUnodePosition(k,&xy);
                incr.x = ((xy.y - y1)*v2 + (y2 - xy.y)*v1);
                incr.y = 0;
                if (ElleUnodeAttributeActive(U_STRAIN)) {
                    ElleGetUnodeAttribute(k,&curr.x,CURR_S_X);
                    ElleGetUnodeAttribute(k,&curr.y,CURR_S_Y);
                    ElleSetUnodeAttribute(k,curr.x,PREV_S_X);
                    ElleSetUnodeAttribute(k,curr.y,PREV_S_Y);
                    curr.x += incr.x;
                    curr.y += incr.y;
                    ElleSetUnodeAttribute(k,curr.x,CURR_S_X);
                    ElleSetUnodeAttribute(k,curr.y,CURR_S_Y);
                }
                xy.x += incr.x;
                xy.y += incr.y;
                ElleSetUnodePosition(k,&xy);
            }
        }
        for (j=0;j<4;j++)
            unitcell.cellBBox[j].x += ((unitcell.cellBBox[j].y-y1)*v2 +
                                       (y2-unitcell.cellBBox[j].y)*v1);
      }
      ElleUpdateCellBBox(&unitcell.cellBBox[BASELEFT],
                      &unitcell.cellBBox[BASERIGHT],
                      &unitcell.cellBBox[TOPRIGHT],
                      &unitcell.cellBBox[TOPLEFT]);
#if XY
      ElleSetCellBBox(&unitcell.cellBBox[BASELEFT],
                      &unitcell.cellBBox[BASERIGHT],
                      &unitcell.cellBBox[TOPRIGHT],
                      &unitcell.cellBBox[TOPLEFT]);
      ElleCellBBox(&unitcell);
      ElleSetCumSSOffset(cum_offset+unitcell.xoffset);
      offset = modf(ElleCumSSOffset(),&tmp);
      ElleSetSSOffset(offset);
#endif
      /*
       * update concentration attributes
       */
      if (ElleNodeCONCactive()) {
          CalculateConcentration(mass_array);
          for (k=0;k<mass_array.size();k++)
              delete mass_array[k];
      }
      for (k=0;k<max;k++) {
          if (ElleNodeIsActive(k)) {
              if (ElleNodeIsDouble(k)) 
                  ElleCheckDoubleJ(k);
              else if (ElleNodeIsTriple(k)) 
                  ElleCheckTripleJ(k);
          }
      }
	  ElleUpdate();
    }
	return(err);
}

void SaveNodeMassValues(vector<vecdbl *> &M)
{
    int cnt=0, k, ik, ij, max, maxa, *active=0, ntot;
    int ndj, ntj;
	vector<int> conc_attr;

    ElleNumberOfNodes(&ndj,&ntj);
    ntot = ndj+ntj;
    ElleNodeAttributeList(&active,&maxa);
    /*
     * count and store the node concentration attributes
     */
    for (ik=0,cnt=0;ik<maxa;ik++) {
        if (ElleNodeCONCAttribute(active[ik])) {
            cnt++;
            conc_attr.push_back(active[ik]);
        }
    }
    if (active) free(active);

    max = ElleMaxNodes();
    for (k=0,ij=0;k<max,ij<ntot;k++) {
        if (ElleNodeIsActive(k)) {
            M.push_back( new vecdbl(cnt,0.0));
            for (ik=0;ik<cnt;ik++) {
                (*M[ij])[ik] = ElleNodeConcToMass(k,conc_attr[ik]);
            }
            ij++;
        }
    }
}

void CalculateConcentration(vector<vecdbl *> &M)
{
    int cnt=0, k, ik, ij, max, maxa, *active=0, ntot;
    double conc=0.0;
	vector<int> conc_attr;

    ntot = M.size();
    if (ntot>0) cnt = (*M[0]).size();
    ElleNodeAttributeList(&active,&maxa);
    /*
     * count and store the node concentration attributes
     */
    for (ik=0,cnt=0;ik<maxa;ik++) {
        if (ElleNodeCONCAttribute(active[ik])) {
            cnt++;
            conc_attr.push_back(active[ik]);
        }
    }
    if (active) free(active);

    max = ElleMaxNodes();
    for (k=0,ij=0;k<max,ij<ntot;k++) {
        if (ElleNodeIsActive(k)) {
            for (ik=0;ik<cnt;ik++) {
                conc=ElleNodeMassToConc(k,((*M[ij])[ik]),conc_attr[ik]);
                ElleSetNodeAttribute(k,conc,conc_attr[ik]);
            }
            ij++;
        }
    }
}
