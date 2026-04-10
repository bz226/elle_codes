#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <list>
#include "nodes.h"
#include "display.h"
#include "error.h"
#include "runopts.h"
#include "file.h"
#include "init.h"
#include "check.h"
#include "crossings.h"
#include "stats.h"
#include "interface.h"
#include "polygon.h"
#include "unodes.h"

using std::cout;
using std::endl;

typedef std::set<int,std::less<int> > set_int;

int RepositionNode(int node);
int ProcessFunction(), InitThisProcess();

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;

    /*
     * clear the data structures
     */
    ElleReinit();

    ElleSetRunFunction(ProcessFunction);

    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError("",err);
        /*
         * initialize any necessary flynn attributes which
         * are not in the input file
         */
    }

}

int ProcessFunction()
{
    int i, j, k;
    int interval=0,st_interval=0,err=0,max,maxu;
    CellData data;
    Coords corners[4],xy;
	double eps=1e-5; /* basil uses float */
    double offset;
    double tmp;

    interval = EllesaveInterval();

    st_interval = ES_statsInterval();
    if (st_interval>0) ES_WriteStatistics(0);

    if (ElleDisplay()) EllePlotRegions(ElleCount());
    ElleCellBBox(&data);
    /*
     * setting CellBBox will put offsets back to 0
    offset = ElleCumSSOffset()+data.xoffset;
     */
	/* if no pure shear component */
	if (fabs(data.ylength-1.0)<eps) {
        corners[BASELEFT].x = 0; corners[BASELEFT].y = 0;
        corners[BASERIGHT].x = 1.0; corners[BASERIGHT].y = 0;
        corners[TOPRIGHT].x = 1.0; corners[TOPRIGHT].y = 1.0;
        corners[TOPLEFT].x = 0; corners[TOPLEFT].y = 1.0;
	}
	else {
        corners[BASELEFT].x = 0; corners[BASELEFT].y = 0;
        corners[BASERIGHT].x = data.xlength; corners[BASERIGHT].y = 0;
        corners[TOPRIGHT].x = data.xlength; corners[TOPRIGHT].y = data.ylength;
        corners[TOPLEFT].x = 0; corners[TOPLEFT].y = data.ylength;
	}
/*
	else {
    corners[BASELEFT].x = data.cellBBox[BASELEFT].x;
	corners[BASELEFT].y = data.cellBBox[BASELEFT].y;
    corners[BASERIGHT].x = corners[BASELEFT].x+data.xlength;
	corners[BASERIGHT].y = data.cellBBox[BASERIGHT].y;
    corners[TOPRIGHT].x = corners[BASERIGHT].x;
	corners[TOPRIGHT].y = corners[BASERIGHT].y+data.ylength;
    corners[TOPLEFT].x = corners[BASELEFT].x;
	corners[TOPLEFT].y = corners[TOPRIGHT].y;
	}
*/
    ElleSetCellBBox(&corners[BASELEFT], &corners[BASERIGHT],
                    &corners[TOPRIGHT], &corners[TOPLEFT]);
/*
    ElleSetCumSSOffset(data.cum_xoffset);
	if (fabs(data.ylength-1.0)<eps) {
        offset = modf(ElleCumSSOffset(),&tmp);
        ElleSetSSOffset(offset);
	}
	else {
        offset = ElleCumSSOffset();
        if (data.xlength<offset) offset = offset-data.xlength;
        ElleSetSSOffset(offset);
	}
*/

    max = ElleMaxNodes();
    for (j=0;j<max;j++) {
        if (ElleNodeIsActive(j)) {
            err = RepositionNode(j);
        }
    }
    max = ElleMaxNodes();
    for (j=0, err=0;j<max;j++) {
        if (ElleNodeIsActive(j)) {
            if (err=ElleNodeTopologyCheck(j))
                OnError("TopologyCheck",err);
        }
    }
    max = ElleMaxNodes();
    for (j=0;j<max;j++) {
        if (ElleNodeIsActive(j)) {
            ElleNodePosition(j,&xy);
            if (xy.x <corners[0].x || xy.x > corners[1].x ||
                xy.y < corners[0].y || xy.y > corners[2].y )
             cout << j << " " << xy.x << " " << xy.y << endl;
	    /*
    	 * unit cell may be "stretching"
         * add doubles and remove redundant nodes from TopologyCheck
    	 * checkangle may not be in the elle loop
    	 */
            if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
         }
    }
    if (ElleUnodesActive()) {
        maxu = ElleMaxUnodes();
        for (j=0;j<maxu;j++) {
            ElleGetUnodePosition(j,&xy);
            ElleNodeUnitXY(&xy);
            ElleSetUnodePosition(j,&xy);
        }
    }
	/*
	 * unit cell may be "stretching"
	 * checkangle may not be in the elle loop
	 */
	ElleAddDoubles(); 


    if (ElleDisplay()) 
        ElleUpdateDisplay();
    if (err=ElleWriteData("repos.elle"))
        OnError("",err);
}

int RepositionNode(int node)
{
    Coords xy;

    ElleNodePosition(node,&xy);
    ElleNodeUnitXY(&xy);
    ElleCopyToPosition(node,&xy);
    /*ElleSetPosition(node,&xy);*/
}

                                                                                
                                                                                
#if XY
int TopologyCheck(int node)
{
    int i, nbs[3], fnew, curr;
    int replaced=1, small_flynn=NO_NB;
    set_int nbflynnset;
    set_int::iterator its,its_end;
    std::list<int>::iterator it;
    Intersection isect;
                                                                                
    double min_area = ElleminNodeSep() * ElleminNodeSep() * SIN60 * 0.5;
                                                                                
    ElleRegions(node,nbs);
    for (i=0;i<3;i++) {
        if (nbs[i]!=NO_NB) {
            nbflynnset.insert(nbs[i]);
            std::list<int> nbflynns;
            ElleFlynnNbRegions(nbs[i], nbflynns);
            for (it=nbflynns.begin();it!=nbflynns.end();it++)
                nbflynnset.insert(*it);
         }
     }
     its=nbflynnset.begin();
     while (its!=nbflynnset.end()) {
         curr = *its;
         if (ElleFlynnIsActive(curr) &&
                   !ElleRegionIsSimple(curr,&isect)) {
             fprintf(stderr," %d not simple, node %d\n",
                                                 curr,isect.node1a);
             if (ElleReplaceIntersection(&isect,&fnew)==0) {
                 replaced=1;
                 small_flynn=NO_NB;
                 if (fnew!=NO_NB) {
                     nbflynnset.insert(fnew);
                     its = nbflynnset.find(curr);
                     if (ElleRegionArea(fnew)<min_area) small_flynn=fnew;
                     else if (ElleRegionArea(isect.chk_rgn)<min_area)
                         small_flynn=isect.chk_rgn;
                 }
                 if (small_flynn!=NO_NB) {
                     std::list<int> nbflynns;
                     ElleFlynnNbRegions(small_flynn,nbflynns);
                     // nbs are in order of common bnd length
                     ElleMergeFlynnsNoCheck(nbflynns.front(),small_flynn);
                     nbflynnset.erase(nbflynnset.find(small_flynn));
                     its = nbflynnset.find(curr);
                 }
             }
             else
                 OnError("TopologyCheck problem",0);
             its=nbflynnset.begin();
        }
        else if (its!=nbflynnset.end()) its++;
    }
}
void TopologyCheck()
{
    int i,max;
    Intersection isect;
                                                                                
    for (i=0;i<ElleMaxFlynns();i++) {
        while (ElleFlynnIsActive(i) && !ElleRegionIsSimple(i,&isect)) {
if (ElleNodeIsActive(isect.node1a)) writeNodeLocalPolyFile(isect.node1a);
                fprintf(stderr," %d not simple, node %d\n",i,isect.node1a);
                if (ElleReplaceIntersection(&isect)==0)
                    fprintf(stderr,"rearranged\n");
                else 
                    OnError("TopologyCheck problem",0);
            }
        }
    }
}
#endif
