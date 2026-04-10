 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <cstdlib>
#include <cmath>
#include "attribute.h"
#include "flynnarray.h"
#include "interface.h"
#include "file.h"
#include "nodes.h"
#include "unodes.h"
#include "segments.h"
#include "polygon.h"
#include "error.h"
#include "convert.h"
#include "runopts.h"
#include "misorient.h"
#include "mineraldb.h"
#include "log.h"
#include "mat.h"
#include "erand.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

using std::vector;
using std::list;

FlynnArray flynns;
Erand erandom;

int ElleSetFlynnActive(int index)
{
    flynns.setFlynnActive(index);
    return 0;
}

int ElleMaxFlynns()
{
    return(flynns.max());
}

void ElleFlynnSetColor(int index,bool c,int r,int g, int b)
{
    flynns.setColor(index,c,r,g,b);
}

void ElleFlynnGetColor(int index,bool *c,int *r,int *g, int *b)
{
    flynns.getColor(index,c,r,g,b);
}

int ElleNumberOfGrains()
{
    return(flynns.numberOfGrains());
}

int ElleNumberOfSubgrains()
{
    return(flynns.numberOfSubgrains());
}

int ElleMaxLevels()
{
    return(flynns.maxLevels());
}

int ElleFlynnLevel(const int index)
{
    return(flynns.getFlynnLevel(index));
}

int ElleFindSpareFlynn()
{
    return(flynns.spareFlynn());
}

unsigned char ElleFlynnIsActive(int index)
{
    return(flynns.isFlynnActive(index));
}

int ElleGetFlynnFirstNode(int index)
{
    return(flynns.getFlynnFirstnode(index));
}

void ElleSetFlynnFirstNode(int index, int first)
{
    flynns.setFlynnFirstnode(index,first);
}

void ElleCleanFlynnArray()
{
    flynns.clean();
}

int ElleUnodesActive()
{
    return(UnodesActive());
}

void ElleCleanUnodeArray()
{
    UnodesClean();
}

int ElleAddFlynnChild(int index, int child)
{
    flynns.addFlynnChild(index,child);
    return 0;
}

unsigned char isParent(int flynn1)
{
    return((flynns.isParent(flynn1)) ? 1 : 0);
}

int getParent(int flynn1)
{
    return(flynns.getParent(flynn1));
}

unsigned char hasParent(int flynn1)
{
    return((flynns.hasParent(flynn1)) ? 1 : 0);
}

unsigned char isChildOf(int flynn1,int flynn2)
{
    return((flynns.isChildOf(flynn1,flynn2)) ? 1 : 0);
}

unsigned char isFlynnIntegerAttribute(int attribid)
{
    Attribute x(attribid,0);

    return((unsigned char) x.isIntAttribute());
}

unsigned char isFlynnRealAttribute(int attribid)
{
    Attribute x(attribid,0);

    return((unsigned char) !(x.isIntAttribute()));
}

unsigned char isFlynnAttribute(int attribid)
{
    Attribute x(attribid,0);

    return((unsigned char) x.isFlynnAttribute());
}

/*
int getFirstNode(const int flynnid)
{
    int first;
    if ((first = flynns.getFirstNode(flynnid))==NO_NB)
        OnError("getFirstNode",0);
}
*/

void ElleCopyFlynnAttributes(int srcindex, int destindex)
{
    flynns.copyAttributes(srcindex,destindex);
}

void EllePromoteFlynn(int flynnindex)
{
    flynns.promoteFlynn(flynnindex);
}

int ElleSplitFlynnandPromote(const int type, int index, int *c1, int *c2,
                             float dx, float dy)
{
    unsigned char ischild=0;
    int subflynns[2];
    int retval=0;

    ischild = hasParent(index);
    if ((retval=ElleSplitFlynn(type,index,&subflynns[0],&subflynns[1],
                               dx,dy))==0) {

        /*if (ElleFlynnHasAttribute(index,SPLIT))//pre hierarchy*/
        if (ElleFlynnAttributeActive(SPLIT)) 
            ElleSetFlynnSplit(index,1);
        EllePromoteFlynn(subflynns[0]);
        EllePromoteFlynn(subflynns[1]);
        ElleRemoveFlynn(index);

        *c1 = subflynns[0];
        *c2 = subflynns[1];
    }
    return(retval);
}

void ElleSplitFlynnOnArea(const int type, int index,
                          float dx, float dy)
{
    int c1, c2;
    int tmp;
    double val, oldminarea, minarea;
    double area, maxarea;

    if (ElleFlynnIsActive(index)) {
        if (ElleFlynnAttributeActive(MINERAL)) {
            ElleGetFlynnIntAttribute(index,&tmp,MINERAL);
            val = GetMineralAttribute(tmp,MAX_FLYNN_AREA);
            oldminarea = ElleMinFlynnArea();
            minarea = GetMineralAttribute(tmp,MIN_FLYNN_AREA);
            ElleSetMinFlynnArea(minarea);
        }
        else {
            val = ElleMaxFlynnArea();
        }
        maxarea = (double)val;
        area = fabs(ElleRegionArea(index));
        if  (area>maxarea) {
            c1 = c2 = NO_NB;
            if (ElleSplitFlynnandPromote(type,index,&c1,&c2,dx,dy)==0) {
                ElleSplitFlynnOnArea(type,c1,maxarea,dx,dy);
                ElleSplitFlynnOnArea(type,c2,maxarea,dx,dy);
            }
        }
        if (ElleFlynnAttributeActive(MINERAL))
            ElleSetMinFlynnArea(oldminarea);
    }
}

// passing the area should be redundant once the mineraldb is
// properly implemented
void ElleSplitFlynnOnArea(const int type, int index, double maxarea,
                          float dx, float dy)
{
    int c1, c2;
    double area;

    if (ElleFlynnIsActive(index)) {
        area = fabs(ElleRegionArea(index));
        if  (area>maxarea) {
            c1 = c2 = NO_NB;
            if (ElleSplitFlynnandPromote(type,index,&c1,&c2,dx,dy)==0) {
                ElleSplitFlynnOnArea(type,c1,maxarea,dx,dy);
                ElleSplitFlynnOnArea(type,c2,maxarea,dx,dy);
            }
        }
    }
}

#if XY
void ElleSplitFlynnChildrenOnArea(int index, double maxarea)
{
    int c1, c2;
    int err=0;
    double area;
	vector<int> l(0);
	vector<int>::iterator it;

    if (ElleFlynnIsActive(index)) {
        if (!isParent(index)) ElleSplitFlynn(index,&c1,&c2);
        flynns.getChildList(index,l);
        for (it = l.begin();it!=l.end() && !err;it++) {
            ElleSplitFlynnOnArea(*it,maxarea);
        }
    }
}
#endif

int ElleNucleateFlynnFromNode(int node, int *newid)
{
    int err=0;
    int i,nbnodes[3],nn, nb[3], pick;
    Coords xyn, xynn, xy0, xy1, midpt;
    ERegion rgn;
    double ang, sep, xpts[4], ypts[4];

    ElleNeighbourNodes(node,nbnodes);
    /*
     * insert  double nodes on all boundaries
     */
    for (i=0;i<3;i++) {
        if (nbnodes[i]!=NO_NB) {
            ElleInsertDoubleJ(node,nbnodes[i],&nn,0.5);
            nbnodes[i] = nn;
        }
    }
    pick=0; if (nbnodes[pick]==NO_NB) pick++;
    ElleNeighbourRegion(nbnodes[pick],node,&rgn);
    if ((nb[0] = ElleFindBndNb(node, rgn))==NO_NB)
        OnError("ElleNucleateFlynnfromNode - Bnd",0);
    if (ElleNodeIsTriple(node)) {
        /*
         * form new grain
         */
        flynns.newFlynn(nbnodes[0],nb[0],rgn,newid);
        /*
         * move TJ to position of 3rd neighbour and delete that nb
         */
        if (nbnodes[1]==nb[0]) nb[0] = nbnodes[2];
        else nb[0] = nbnodes[1];
        ElleNodePosition(nb[0],&xyn);
        ElleSetPosition(node,&xyn);
        ElleDeleteDoubleJNoCheck(nb[0]);
    }
    else if (ElleNodeIsDouble(node)) {
        nb[1] = nbnodes[pick];
        /*
         * form new grain
         */
        flynns.newFlynn(nb[1],nb[0],rgn,newid);

        ElleInsertDoubleJ(nb[0],nb[1],&nn,0.5);
        ElleNodePosition(nb[0],&xy0);
        ElleNodePlotXY(node,&xyn,&xy0);
        ElleNodePlotXY(nb[1],&xy1,&xyn);
        ang = PI/2;
        midpt.x = xy0.x - (xy0.x - xy1.x)*0.5;
        midpt.y = xy0.y - (xy0.y - xy1.y)*0.5;
        sep = ElleNodeSeparation(nb[0],nn);
        ElleNodePlotXY(nn,&midpt,&xy0);
        xyn.x = midpt.x + (midpt.x-xy0.x)*cos(ang) + (midpt.y-xy0.y)*sin(ang);
        xyn.y = midpt.y + (midpt.y-xy0.y)*cos(ang) - (midpt.x-xy0.x)*sin(ang);
        xynn.x = midpt.x + (midpt.x-xy1.x)*cos(ang) + (midpt.y-xy1.y)*sin(ang);
        xynn.y = midpt.y + (midpt.y-xy1.y)*cos(ang) - (midpt.x-xy1.x)*sin(ang);
        xpts[0] = xy1.x; ypts[0] = xy1.y;
        xpts[1] = xyn.x; ypts[1] = xyn.y;
        xpts[2] = xy0.x; ypts[2] = xy0.y;
        xpts[3] = xynn.x; ypts[3] = xynn.y;
        if (polyArea(xpts,ypts,4) > 0.0) {
            ElleSetPosition(node,&xyn);
            ElleSetPosition(nn,&xynn);
        }
        else {
            ElleSetPosition(node,&xynn);
            ElleSetPosition(nn,&xyn);
        }
    }
    return(err);
}
#if XY
int ElleNucleateFlynnFromNode(int node, int *newid)
{
    int err=0;
    int i,nbnodes[3],nn, nb[3], pick;
    int *node_ids,num_nodes;
    Coords xyn, xynn, xy0, xy1, incr_xy, midpt;
    ERegion rgn;
    double ang, xpts[4], ypts[4];

    ElleNeighbourNodes(node,nbnodes);
    /*
     * ensure all neighbours are double nodes
     */
    for (i=0;i<3;i++) {
        if (nbnodes[i]!=NO_NB && ElleNodeIsTriple(nbnodes[i])) {
            ElleInsertDoubleJ(node,nbnodes[i],&nn,0.5);
            nbnodes[i] = nn;
        }
    }
    pick=0; if (nbnodes[pick]==NO_NB) pick++;
    ElleNeighbourRegion(nbnodes[pick],node,&rgn);
    if ((nb[0] = ElleFindBndNb(node, rgn))==NO_NB)
        OnError("ElleNucleateFlynnfromNode - Bnd",0);
    if (ElleNodeIsTriple(node)) {
        /*
         * form new grain
         */
        flynns.newFlynn(nbnodes[0],nb[0],rgn,newid);
        /*
         * move TJ to position of 3rd neighbour and delete that nb
         */
        if (nbnodes[1]==nb[0]) nb[0] = nbnodes[2];
        else nb[0] = nbnodes[1];
        ElleNodePosition(nb[0],&xyn);
        ElleSetPosition(node,&xyn);
        ElleDeleteDoubleJNoCheck(nb[0]);
    }
    else if (ElleNodeIsDouble(node)) {
        nb[1] = nbnodes[pick];
        /*ElleNeighbourRegion(nb[0],node,&rgn);*/
        /*
         * form new grain
         */
        flynns.newFlynn(nb[1],nb[0],rgn,newid);

        ElleInsertDoubleJ(nb[0],nb[1],&nn,0.5);
        ElleNodePosition(nb[0],&xy0);
        ElleNodePlotXY(node,&xyn,&xy0);
        ElleNodePlotXY(nb[1],&xy1,&xyn);
        ang = PI*0.5*0.333;
        midpt.x = xy0.x - (xy0.x - xy1.x)*0.5;
        midpt.y = xy0.y - (xy0.y - xy1.y)*0.5;
        xyn.x = (midpt.x-xy0.x) * cos(ang) +
                          (midpt.y-xy0.y) * sin(ang);
        xyn.y = -(midpt.x-xy0.x) * sin(ang) +
                          (midpt.y-xy0.y) * cos(ang);
        xyn.x += xy0.x; xyn.y += xy0.y;
        xynn.x = (midpt.x-xy0.x) * cos(-ang) +
                          (midpt.y-xy0.y) * sin(-ang);
        xynn.y = -(midpt.x-xy0.x) * sin(-ang) +
                          (midpt.y-xy0.y) * cos(-ang);
        xynn.x += xy0.x; xynn.y += xy0.y;
        xpts[0] = xy1.x; ypts[0] = xy1.y;
        xpts[1] = xyn.x; ypts[1] = xyn.y;
        xpts[2] = xy0.x; ypts[2] = xy0.y;
        xpts[3] = xynn.x; ypts[3] = xynn.y;
        if (polyArea(xpts,ypts,4) > 0.0) {
            ElleSetPosition(node,&xyn);
            ElleSetPosition(nn,&xynn);
        }
        else {
            ElleSetPosition(node,&xynn);
            ElleSetPosition(nn,&xyn);
        }
        /**newid = rgn;*/
    }
    return(err);
}
#endif

int ElleWriteParentFlynn(int index, FILE *fp)
{
    int err=0,num;
	vector<int> l(0);
	vector<int>::iterator it;

    if (index < ElleMaxFlynns()) {
        flynns.getChildList(index,l);
        if ((num=fprintf(fp,"%d",index))<0) err = WRITE_ERR;
        if (!err) {
            /*if ((num=fprintf(fp," %d",ElleGetFlynnFirstNode(index)))<0)*/
                /*err = WRITE_ERR;*/
            /*else*/
                if ((num=fprintf(fp," %ld",l.size()))<0) err = WRITE_ERR;
                for (it = l.begin();it!=l.end() && !err;it++) 
                    if ((num=fprintf(fp," %d",(*it)))<0)
                        err=WRITE_ERR;
        }
    }
    return(err);
}

void ElleUpdateFirstNodes(int old, int nn, int flynnindex)
{
    flynns.updateFirstNodes(old, nn, flynnindex);
}

void ElleRemoveFlynn(int flynnindex)
{
    flynns.removeFlynn(flynnindex);
}

void ElleRemoveShrinkingFlynn(int flynnindex)
{
    flynns.removeShrinkingFlynn(flynnindex);
}

void ElleFlynnNodes(int index, int **ids, int *num)
{
    flynns.flynnNodes(index,ids,num);
}

void ElleFlynnNodes(int index, vector<int> &ids)
{
    flynns.flynnNodes(index,ids);
}

int ElleFlynnNodePositions(int index, vector<Coords> &rel_pos,
                           Coords *ref)
{
    return(flynns.flynnNodePositions(index,rel_pos,ref));
}

double ElleFlynnArea(int index)
{
    return(flynns.flynnArea(index));
}

int ElleFlynnNodeCount(int index)
{
    return(flynns.flynnNodeCount(index));
}

int ElleSplitFlynn(const int type, int index,
                   int *child1, int *child2,
                   float dx, float dy)
{
    return(flynns.splitFlynn(type, index, child1, child2, dx, dy));
}

int ElleSplitWrappingFlynn(int index, int node1, int node2,
                            int *child1, int *child2)
{
    return(flynns.splitWrappingFlynn(index, node1, node2,
                                      child1, child2));
}

int ElleDirectSplitWrappingFlynn(int index, int node1, int node2,
                            int *child1, int *child2, Coords *dir)
{
    return(flynns.directSplitWrappingFlynn(index, node1, node2,
                                      child1, child2, dir));
}

int ElleCheckSmallFlynn(const int index)
{
    double minarea;

    minarea = ElleminNodeSep()*ElleminNodeSep()*SIN60*0.5;
    return(flynns.checkSmallFlynn(index,minarea));
}

bool ElleFlynnAttributesAllowMerge(const int flynn1, const int flynn2)
{
    return(flynns.attributesAllowMerge(flynn1,flynn2));
}

int ElleMergeFlynns(int keepflynn, int removeflynn)
{
    return(flynns.mergeFlynns(keepflynn,removeflynn));
}

int ElleMergeFlynnsNoCheck(int keepflynn, int removeflynn)
{
    return(flynns.mergeFlynnsNoCheck(keepflynn,removeflynn));
}

void ElleFlynnNbRegions(const int index, list<int> &nbflynns)
{
    flynns.neighbourRegions(index,nbflynns);
}

/*
int ElleFlynnAddVoronoiPt(const int flynnid, Coords pt )
{
    flynns.addVoronoiPt(flynnid,pt);
}
*/

double ElleFindFlynnMinArea(int id)
{
    int i;
    double minarea;

    if (ElleFlynnAttributeActive(MINERAL)) {
        ElleGetFlynnIntAttribute(id,&i,MINERAL);
        minarea = GetMineralAttribute(i,MIN_FLYNN_AREA);
    }
    else
        minarea = ElleMinFlynnArea();
    return(minarea);
}

void ElleInitFlynnAttribute(int id)
{
    if (!flynns.hasAttribute(id)) {
        flynns.initAttribute(id);
        /*ElleCheckFlynnDefaults(id);*/
    }
}

int EllenumFlynnAttributes(int flynnindex)
{
    return(flynns.numFlynnAttributes(flynnindex));
}

void ElleFlynnDfltAttributeList(int **active, int *maxa)
{
    *active=0;
    if ((*maxa = flynns.numDfltAttributes()) > 0) {
        if ((*active = (int *)malloc(*maxa * sizeof(int)))==0)
            OnError("ElleFlynnDfltAttributeList",MALLOC_ERR);
        flynns.getDfltAttributeList(*active,*maxa);
    }
}


void ElleSetDefaultFlynnCAxis(Coords_3D *val)
{
    Coords_3D dflt;

    if (flynns.hasAttribute(CAXIS)) {
        flynns.getDfltAttribute(CAXIS,&dflt);
        flynns.setDfltAttribute(CAXIS,val);
    }
    else
        OnError("ElleSetDefaultFlynnCAxis",ATTRIBID_ERR);
/*
    max = ElleMaxFlynns();
    for (j=0;j<max;j++) {
        if (flynns.isFlynnActive(j) &&
                    flynns.hasFlynnAttribute(j,CAXIS)) {
            flynns.getFlynnAttribute(j,CAXIS,&curr);
            if (curr.x==dflt.x && curr.y==dflt.y && curr.z==dflt.z)
                flynns.setFlynnAttribute(j,CAXIS,val);
        }
    }
*/
}

void ElleSetDefaultFlynnEuler3(double alpha, double beta, double gamma)
{
    double dflta, dfltb, dfltc;

    if (flynns.hasAttribute(E3_ALPHA)) {
        flynns.getDfltAttribute(E3_ALPHA,&dflta);
        flynns.setDfltAttribute(E3_ALPHA,alpha);
    }
    else
        OnError("ElleSetDefaultFlynnEuler3",ATTRIBID_ERR);
    if (flynns.hasAttribute(E3_BETA)) {
        flynns.getDfltAttribute(E3_BETA,&dfltb);
        flynns.setDfltAttribute(E3_BETA,beta);
    }
    else
        OnError("ElleSetDefaultFlynnEuler3",ATTRIBID_ERR);
    if (flynns.hasAttribute(E3_GAMMA)) {
        flynns.getDfltAttribute(E3_GAMMA,&dfltc);
        flynns.setDfltAttribute(E3_GAMMA,gamma);
    }
    else
        OnError("ElleSetDefaultFlynnEuler3",ATTRIBID_ERR);
/*
    max = ElleMaxFlynns();
    for (j=0;j<max;j++) {
        if (flynns.isFlynnActive(j) &&
                    ElleFlynnHasAttribute(j,EULER_3)) {
            flynns.getFlynnAttribute(j,E3_ALPHA,&curra);
            flynns.getFlynnAttribute(j,E3_BETA,&currb);
            flynns.getFlynnAttribute(j,E3_GAMMA,&currc);
            if (curra==dflta && currb==dfltb && currc==dfltc){
                flynns.setFlynnAttribute(j,E3_ALPHA,alpha);
                flynns.setFlynnAttribute(j,E3_BETA,beta);
                flynns.setFlynnAttribute(j,E3_GAMMA,gamma);
            }
        }
    }
*/
}

unsigned char ElleFlynnAttributeActive(int id)
{
    return(flynns.hasAttribute(id));
}

void ElleSetDefaultFlynnAge(double val, int type)
{
    double dflt;

    if (flynns.hasAttribute(type)) {
        flynns.getDfltAttribute(type,&dflt);
        flynns.setDfltAttribute(type,val);
    }
    else
        OnError("ElleSetDefaultFlynnAge",ATTRIBID_ERR);
}

void ElleSetDefaultFlynnIntAttribute(int val,int id)
{
    double dflt;

    if (flynns.hasAttribute(id)) {
        flynns.getDfltAttribute(id,&dflt);
        flynns.setDfltAttribute(id,(double)val);
    }
    else
        OnError("ElleSetDefaultFlynnIntAttrib",ATTRIBID_ERR);
}

void ElleSetDefaultFlynnRealAttribute(double val,int id)
{
    double dflt;

    if (flynns.hasAttribute(id)) {
        flynns.getDfltAttribute(id,&dflt);
        flynns.setDfltAttribute(id,val);
    }
    else
        OnError("ElleSetDefaultFlynnRealAttrib",ATTRIBID_ERR);
}

unsigned char ElleFlynnHasAttribute(int index,int id)
{
    /*
     * this stuff needs more thought for attributes
     * with multiple entries, eg CAXIS and EULER_3
     */
    bool found = 0;
    if (id==EULER_3) 
        found =  (flynns.hasFlynnAttribute(index,E3_ALPHA) &&
                   flynns.hasFlynnAttribute(index,E3_BETA) &&
                    flynns.hasFlynnAttribute(index,E3_GAMMA));
    else found = flynns.hasFlynnAttribute(index,id);
    return(found ? 1 : 0);
}

void ElleDefaultFlynnEuler3(double *alpha, double *beta, double *gamma)
{
    flynns.getDfltAttribute(E3_ALPHA,alpha);
    flynns.getDfltAttribute(E3_BETA,beta);
    flynns.getDfltAttribute(E3_GAMMA,gamma);
}

double ElleDefaultFlynnAge(int type)
{
    double val;

    flynns.getDfltAttribute(type,&val);
    return(val);
}

void ElleDefaultFlynnCAxis(Coords_3D *val)
{
    flynns.getDfltAttribute(CAXIS,val);
}

void ElleRemoveDefaultFlynnAttribute(int idval)
{
    flynns.removeDfltAttribute(idval);
}

void ElleRemoveFlynnAttribute(int index, int idval)
{
    flynns.removeFlynnAttribute(index, idval);
}

void ElleSetFlynnCAxis(int index, Coords_3D *val)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,CAXIS,val))
        OnError("ElleSetFlynnCAxis",err);
}

void ElleGetFlynnCAxis(int index, Coords_3D *valptr)
{
    int err=0;
    if (err = flynns.getFlynnAttribute(index,CAXIS,valptr)) {
        if (err==ATTRIBID_ERR)
            ElleDefaultFlynnCAxis(valptr);
        else OnError("ElleGetFlynnCAxis",err);
    }
}

void ElleSetFlynnEuler3(int index, double alpha, double beta, double gamma)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,E3_ALPHA,alpha))
        OnError("ElleSetFlynnEuler3 alpha",err);
    if (err = flynns.setFlynnAttribute(index,E3_BETA,beta))
        OnError("ElleSetFlynnEuler3 beta",err);
    if (err = flynns.setFlynnAttribute(index,E3_GAMMA,gamma))
        OnError("ElleSetFlynnEuler3 gamma",err);
}

void ElleGetFlynnEuler3(int index, double *alpha, double *beta, double *gamma)
{
    int err=0;
    if (err = flynns.getFlynnAttribute(index,E3_ALPHA,alpha)) {
        if (err==ATTRIBID_ERR)
            ElleDefaultFlynnEuler3(alpha,beta,gamma);
        else OnError("ElleGetFlynnEuler3 alpha",err);
    }
    else {
        err = flynns.getFlynnAttribute(index,E3_BETA,beta);
        err = flynns.getFlynnAttribute(index,E3_GAMMA,gamma);
        if (err) OnError("ElleGetFlynnEuler3 gamma",err);
    }
}

//---------------------------------------------------------------------------
//Function ElleGetFlynnEulerCAxis
//
//  converts Euler angles into direction cosines of c-axis
//
/*!
    \brief      converts Euler angles into direction cosines of c-axis
    \param      flynn_no  the id of the flynn
    			dircos		the adress of the object to hold the cosine
							values (3 doubles)
    \exception  [name]  [Brief description]
    \return     \a err 0 if successful else 1
    \par        Description:
        Sets dircos to direction cosines of c-axis lower hemisphere
		assuming Bunge (ZXZ) convention for Euler angles and 
		RD is +x, TD is +y, ND is +z (out of page, in front)
        
    \remarks
        [Multi-line remarks]
    \warning
        [Multi-line warnings]
    \par        Example:
\verbatim
[Multi-line code example]
\endverbatim
*/
int ElleGetFlynnEulerCAxis(int flynn_no, Coords_3D *dircos)
{
    double alpha,beta,gamma;
    double xyz[3];
    
    ElleGetFlynnEuler3(flynn_no,&alpha,&beta,&gamma);

    //converts Euler angles (deg) into direction cosines of c-axis
    
    eulerZXZ2cax(alpha, beta, gamma, &xyz[0]);
    dircos->x=xyz[0];
    dircos->y=xyz[1];
    dircos->z=xyz[2];
}

void ElleSetFlynnIntAttribute(int index, int val, int id)
{
    int err=0;
    if (isFlynnIntegerAttribute(id))
        err = flynns.setFlynnAttribute(index,id,val);
    else err=ATTRIBID_ERR;
    if (err) OnError("ElleSetFlynnIntAttribute",err);
}

void ElleGetFlynnIntAttribute(int index, int *val, int id)
{
    int err=0;

    if (isFlynnIntegerAttribute(id)) {
        err = flynns.getFlynnAttribute(index,id,val);
        if (err==ATTRIBID_ERR)
            err = flynns.getDfltAttribute(id,val);
    }
    else err=ATTRIBID_ERR;
    if (err) OnError("ElleGetFlynnIntAttribute",err);
}

int ElleDefaultFlynnIntAttribute(int id)
{
    int err=0;
    int val=0;

    if (isFlynnIntegerAttribute(id)) 
        err = flynns.getDfltAttribute(id,&val);
    else err=ATTRIBID_ERR;
    if (err) OnError("ElleDefaultFlynnIntAttribute",err);
    return(val);
}

double ElleDefaultFlynnRealAttribute(int id)
{
    int err=0;
    double fval=0;

    if (isFlynnRealAttribute(id)) 
        err = flynns.getDfltAttribute(id,&fval);
    else err=ATTRIBID_ERR;
    if (err) OnError("ElleDefaultFlynnRealAttribute",err);
    return(fval);
}

void ElleSetFlynnRealAttribute(int index, double val, int id)
{
    int err=0;

        if (id==F_BULK_S) id=BULK_S;
        if (id==F_INCR_S) id=INCR_S;
    if (err = flynns.setFlynnAttribute(index,id,val))
        OnError("ElleSetFlynnRealAttribute",err);
}

void ElleGetFlynnRealAttribute(int index, double *valptr, int id)
{
    int err=0;
        if (id==F_BULK_S) id=BULK_S;
        if (id==F_INCR_S) id=INCR_S;
    if (isFlynnAttribute(id)) {
        err = flynns.getFlynnAttribute(index,id,valptr);
        if (err==ATTRIBID_ERR)
            err = flynns.getDfltAttribute(id,valptr);
    }
    else err=ATTRIBID_ERR;
    if (err) OnError("ElleGetFlynnRealAttribute",err);
}

/*
 * Can just use the routine above for all attributes
 * or use ones like those below with a check that the
 * type is in the set of allowable types for the attribute
 * eg for Strain, the type should be in the set of strain keys
 */

void ElleSetFlynnStrain(int index, double val, int type)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,type,val))
        OnError("ElleSetFlynnStrain",err);
}

void ElleGetFlynnStrain(int index, double *valptr, int type)
{
    int err=0;
    if (err = flynns.getFlynnAttribute(index,type,valptr))
        OnError("ElleGetFlynnStrain",err);
}

int ElleFindFlynnAttributeRange(int attr,double *amin,double *amax)
{
    int err=0;
    int max,i,ival=0,set=0;
    double val=0,tmpmin=0,tmpmax=0,b,c;
    Coords_3D coordval;
    double theta,phi;

    *amin = *amax = val;
    if (ElleFlynnAttributeActive(attr)) {
        max = ElleMaxFlynns();
        for (i=0;i<max;i++) {
            if (ElleFlynnIsActive(i) /*&& ElleFlynnHasAttribute(i,attr)*/) {
                val = 0;
                switch(attr) {
                case EXPAND:
                case MINERAL:
                case GRAIN  :ElleGetFlynnIntAttribute(i,&ival,attr);
                             val = (double)ival;
                             break;
                case VISCOSITY:
                case S_EXPONENT:
                case ENERGY:
                case AGE:
                case CYCLE:
                case DISLOCDEN:
                case F_ATTRIB_A:
                case F_ATTRIB_B:
                case F_ATTRIB_C:
                case F_ATTRIB_D:
                case F_ATTRIB_E:
                case F_ATTRIB_F:
                case F_ATTRIB_I:
                case F_ATTRIB_J:
                case F_ATTRIB_K:
                case TAU_XX:
                case TAU_YY:
                case TAU_ZZ:
                case TAU_XY:
                case TAU_1:
                case PRESSURE:
                             ElleGetFlynnRealAttribute(i,&val,attr);
                             break;
                case CAXIS: ElleGetFlynnCAxis(i,&coordval);
                            CartesianToPolar(coordval.x,coordval.y,
                                             coordval.z,&theta,&phi);
                            if (theta<0.0) theta = PI-theta;
             /****************** should this be
                            if (theta<0.0) theta += 2.0*PI;
              ******************/
                            theta *= RTOD;
                            val = theta;
                            break;
                case INCR_S:
                case F_INCR_S:
                case BULK_S:
                case F_BULK_S:
                case E_XX:
                case E_XY:
                case E_YX:
                case E_YY:
                case E_ZZ: ElleGetFlynnStrain(i,&val,attr);
                           break;
                case E3_ALPHA:
                case EULER_3: ElleGetFlynnEuler3(i,&val,&b,&c);
                           //val=fmod((val+90.0),30.0)/15;
                           //if(val > 1.0) val=2.0-val;
                           break;
                }
                if (!set) {
                    tmpmax = val;
                    tmpmin = val;
                    set = 1;
                }
                else {
                    if (val > tmpmax) tmpmax = val;
                    if (val < tmpmin) tmpmin = val;
                }
            }
        }
        *amax = tmpmax;
        *amin = tmpmin;
    }
    else err = ATTRIBID_ERR;
    return(err);
}

int ElleFindBndAttributeRange(int attr,double *amin,double *amax)
{
    int err=0;
    /* check for EULER_3 active?? */
    FindBndAttributeRange(attr,amin,amax);
    return(err);
}

int ElleWriteFlynnPolyFile(int flynnindex,char *fname)
{
	if (ElleFlynnIsActive(flynnindex))
        flynns.writePolyFile(flynnindex,fname);
}

int ElleWriteFlynnUnodePolyFile(int flynnindex,const int attrib,char *fname)
{
    flynns.writeUnodePolyFile(flynnindex,attrib,fname);
}

/*
 * reduce search time by giving it a starting flynn
 * and searching the neighbours of that flynn before
 * resorting to a full search
 */
int ElleAddUnodeToFlynn(int id, Coords *xy, int start)
{
    int i, rgn=NO_NB, not_checked=1, repos;
	Coords orig_xy;
    CellData unitcell;

    ElleCellBBox(&unitcell);

	list<int> nbflynns;
	list<int>::iterator it;

    int max = ElleMaxFlynns();
    /*
     * check the search flynn is active and valid
     */
    if (start>=max) start=0;
    if (!ElleFlynnIsActive(start) || isParent(start)) {
        i = (start+1)%max;
        while ((!ElleFlynnIsActive(i) || isParent(i)) && i!=start)
            i = (i+1)%max;
        if (i!=start) start=i;
        else OnError("ElleAddUnodeToFlynn-no valid flynns",0);
    }
	orig_xy = *xy;
	repos = 0;
    /*
     * check the first flynn
     */
	while (rgn==NO_NB && repos<9) {
    if (ElleFlynnIsActive(start) && !isParent(start))
        if (EllePtInRegion(start,&orig_xy)) {
            flynns.addUnode(start,id);
            ElleSetUnodeFlynn(id,start);
            rgn=start;
        }
    if (rgn==NO_NB) {
        /*
         * check the first flynn's neighbours
         */
        ElleFlynnNbRegions(start,nbflynns);
        for (it=nbflynns.begin();it!=nbflynns.end() && rgn==NO_NB;
                                                           it++) {
            if (EllePtInRegion((*it),&orig_xy)) {
                flynns.addUnode((*it),id);
                ElleSetUnodeFlynn(id,(*it));
                rgn=(*it);
            }
        }
    }
    if (rgn==NO_NB) {
        /*
         * check the remaining flynns
         */
        i = (start+1)%max;
        while (rgn==NO_NB && i!=start) {
            not_checked =
                (find(nbflynns.begin(),nbflynns.end(),i)==nbflynns.end());
            if (not_checked && ElleFlynnIsActive(i) && !isParent(i))
                if (EllePtInRegion(i,&orig_xy)) {
                    flynns.addUnode(i,id);
                    ElleSetUnodeFlynn(id,i);
                    rgn=i;
                }
            i=(i+1)%max;
        }
    }
	repos++;
	if (rgn==NO_NB) {
		switch (repos) {
		case 1: orig_xy.x = xy->x + unitcell.xlength;
				orig_xy.y = xy->y + unitcell.ylength;
				break;
		case 2: orig_xy.x = xy->x + unitcell.xlength;
				orig_xy.y = xy->y;
				break;
		case 3: orig_xy.y = xy->y + unitcell.ylength;
				orig_xy.x = xy->x;
				break;
		case 4: orig_xy.x = xy->x - unitcell.xlength;
				orig_xy.y = xy->y;
				break;
		case 5: orig_xy.y = xy->y - unitcell.ylength;
				orig_xy.x = xy->x;
				break;
		case 6: orig_xy.x = xy->x - unitcell.xlength;
				orig_xy.y = xy->y - unitcell.ylength;
				break;
		case 7: orig_xy.x = xy->x - unitcell.xlength;
				orig_xy.y = xy->y + unitcell.ylength;
				break;
		case 8: orig_xy.x = xy->x + unitcell.xlength;
				orig_xy.y = xy->y - unitcell.ylength;
				break;
		default: repos=9; break;
		}
		}
    }

    if (rgn==NO_NB)
        OnError("ElleAddUnodeToFlynn: unode and flynn not matched",0);
    return(rgn);
}

void ElleAddUnodeToFlynn(int flynnid, int id)
{
    flynns.addUnode(flynnid,id);
    ElleSetUnodeFlynn(id,flynnid);
}

int ElleRemoveUnodeFromFlynn(int flynnid, int id)
{
    ElleSetUnodeFlynn(id,NO_VAL);
    return(flynns.removeUnode(flynnid, id));
}

void ElleGetFlynnUnodeList(int flynnid, vector<int> &l)
{
    flynns.getUnodeList(flynnid, l);
}

unsigned char ElleUnodeAttributeActive(int id)
{
    return(UnodeAttributeActive(id));
}

void ElleUnodeAttributeList(int **attr_ids, int *maxa)
{
    UnodeAttributeList(attr_ids,maxa);
}

int ElleFindUnodeAttributeRange(int attr,double *amin,double *amax)
{
    int err=0;

    err = UnodeAttributeRange(attr,amin,amax);
    return(err);
}

void ElleSegmentAttributeList(int **attr_ids, int *maxa)
{
    SegmentAttributeList(attr_ids,maxa);
}

int ElleFindSegmentAttributeRange(int attr,double *amin,double *amax)
{
    int err=0;

    err = SegmentAttributeRange(attr,amin,amax);
    return(err);
}

int ElleVoronoiFlynnUnodes()
{
    int max, k, err=0;
                                                                                
    max = ElleMaxFlynns();
    for (k=0;k<max;k++) {
        if (ElleFlynnIsActive(k)) {
            err = VoronoiFlynnUnodes(k);
                                                                                
        }
    }
    return(err);
}

int ElleVoronoiFlynnUnodes(std::list<int> &flynnlist)
{
    int max, k, err=0;

    flynnlist.sort();
    for (std::list<int> :: iterator it = flynnlist.begin();
                    it != flynnlist.end() ; it++) {
        if (ElleFlynnIsActive(*it)) {
            err = VoronoiFlynnUnodes(*it);
                                                                                
        }
    }
    return(err);
}

int ElleVoronoiFlynnUnodes(std::list<int> &flynnlist,
                           std::vector<int> &unodelist)
{
    int max, k, err=0;

    flynnlist.sort();
    for (std::list<int> :: iterator it = flynnlist.begin();
                    it != flynnlist.end() ; it++) {
        if (ElleFlynnIsActive(*it)) {
            err = VoronoiFlynnUnodes(*it,unodelist);
                                                                                
        }
    }
    return(err);
}

// assumes conc is in Elle units
double ElleTotalUnodeMass(int attr_id)
{
    int i, max, maxu, no_unodes=0;
    double lattice_mass=0.0;
    double conc;
    double node_mass;

    maxu = ElleMaxUnodes();
    max = ElleMaxFlynns();

    if (maxu>0 && ElleUnodeAttributeActive(attr_id)) {
        double node_area=1.0/maxu;
        for(i=0;i<max;i++)
        {
            if(ElleFlynnIsActive(i))
            {
				vector<int> unodelist;
                ElleGetFlynnUnodeList(i,unodelist);

                for (vector<int> :: iterator it = unodelist.begin();
                    it != unodelist.end() ; it++)
                {
                    ElleGetUnodeAttribute((*it),&conc,attr_id);
                    node_mass=conc*node_area;
                    lattice_mass+=node_mass;

                    no_unodes++;
                }
            }
        }
        if(no_unodes<max) {
            sprintf( logbuf, "only %d of all %d unodes accounted for\n",
                                                      no_unodes, max );
            Log( 2,logbuf );
        }
    }
    return(lattice_mass);
}

int SaveZIPParentFlynn( int index, gzFile out )
{
  int err = 0, num;
  vector < int > l( 0 );
  vector < int >::iterator it;
                                                                                
  if ( index < ElleMaxFlynns() )
  {
    flynns.getChildList( index, l );
    if ( ( num = gzprintf( out, "%d", index ) ) < 0 ) err = WRITE_ERR;
    if ( !err )
    {
      /* if ((num=gzprintf(out," %d",ElleGetFlynnFirstNode(index)))<0)
 * */
      /* err = WRITE_ERR; */
      /* else */
      if ( ( num = gzprintf( out, " %d", l.size() ) ) < 0 )
         err = WRITE_ERR;
      for ( it = l.begin(); it != l.end() && !err; it++ )
        if ( ( num = gzprintf( out, " %d", ( * it ) ) ) < 0 )
          err = WRITE_ERR;
    }
  }
  return ( err );
}

void ElleInitRandom(unsigned long int seed)
{
    erandom.seed(seed);
}

unsigned long int ElleRandom()
{
    return(erandom.ran());
}

//---------------------------------------------------------------------------
//Function ElleRandomD()
//
/*!
	\brief		returns a random number [0,1)
	\return		\a erandom.randouble()
	\par		Description:
This function returns a double precision floating point number uniformly
distributed in the range [0,1). The range includes 0.0 but excludes 1.0.
	\remarks
erandom is an Erand object which is initialised during ElleInit and is a
class which uses one of the gsl (GNU scientific Library) random number generators.
\verbatim
[Multi-line code example]
\endverbatim
		[Multi-line detailed description]
*/
double ElleRandomD()
{
    return(erandom.randouble());
}
///////////////////////////////////////////////////////
// functions below this line are being phased out
// they should not be used and only remain for backward
//     compatibility
//////////////////////////////////////////////////////

void ElleSetFlynnEuler3(int index, float alpha, float beta, float gamma)
{
    int err=0;

    if (err = flynns.setFlynnAttribute(index,E3_ALPHA,(double)alpha))
        OnError("ElleSetFlynnEuler3 alpha",err);
    if (err = flynns.setFlynnAttribute(index,E3_BETA,(double)beta))
        OnError("ElleSetFlynnEuler3 beta",err);
    if (err = flynns.setFlynnAttribute(index,E3_GAMMA,(double)gamma))
        OnError("ElleSetFlynnEuler3 gamma",err);
}

void ElleGetFlynnEuler3(int index, float *alpha, float *beta, float *gamma)
{
    int err=0;
    Coords_3D tmp;
    if (err = flynns.getFlynnAttribute(index,E3_ALPHA,&tmp.x)) {
        if (err==ATTRIBID_ERR)
            ElleDefaultFlynnEuler3(&tmp.x,&tmp.y,&tmp.z);
        else OnError("ElleGetFlynnEuler3 alpha",err);
    }
    else {
        err = flynns.getFlynnAttribute(index,E3_BETA,&tmp.y);
        err = flynns.getFlynnAttribute(index,E3_GAMMA,&tmp.z);
        if (err) OnError("ElleGetFlynnEuler3 gamma",err);
    }
    *alpha = (float)tmp.x;
    *beta = (float)tmp.y;
    *gamma = (float)tmp.z;
}

void ElleSetFlynnRealAttribute(int index, float val, int id)
{
    int err=0;
    
        if (id==F_BULK_S) id=BULK_S;
        if (id==F_INCR_S) id=INCR_S;
    if (err = flynns.setFlynnAttribute(index,id,(double)val))
        OnError("ElleSetFlynnRealAttribute",err);
}

void ElleGetFlynnRealAttribute(int index, float *valptr, int id)
{
    int err=0;
    double tmp=0;

    if (isFlynnRealAttribute(id)) {
        err = flynns.getFlynnAttribute(index,id,&tmp);
        if (err==ATTRIBID_ERR)
            err = flynns.getDfltAttribute(id,&tmp);
    }
    else err=ATTRIBID_ERR;
    if (err) OnError("ElleGetFlynnRealAttribute",err);
    else *valptr = (float)tmp;
}

void ElleSetFlynnEnergy(int index, float val)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,ENERGY,(double)val))
        OnError("ElleSetFlynnEnergy",err);
}

void ElleGetFlynnEnergy(int index, float *valptr)
{
    int err=0;
    double tmp;
    if (err = flynns.getFlynnAttribute(index,ENERGY,&tmp)) {
        if (err==ATTRIBID_ERR)
            tmp=ElleDefaultFlynnRealAttribute(ENERGY);
        else
            OnError("ElleGetFlynnEnergy",err);
    }
    *valptr = (float)tmp;
}

void ElleSetFlynnViscosity(int index, float val)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,VISCOSITY,(double)val))
        OnError("ElleSetFlynnViscosity",err);
}

void ElleGetFlynnViscosity(int index, float *valptr)
{
    int err=0;
    double tmp;
    if (err = flynns.getFlynnAttribute(index,VISCOSITY,&tmp)) {
        if (err==ATTRIBID_ERR)
            tmp=ElleDefaultFlynnRealAttribute(VISCOSITY);
        else
            OnError("ElleGetFlynnViscosity",err);
    }
    *valptr = (float)tmp;
}

void ElleSetFlynnDislocDen(int index, float val)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,DISLOCDEN,(double)val))
        OnError("ElleSetFlynnDislocDen",err);
}

void ElleGetFlynnDislocDen(int index, float *valptr)
{
    int err=0;
    double tmp;
    if (err = flynns.getFlynnAttribute(index,DISLOCDEN,&tmp)) {
        if (err==ATTRIBID_ERR)
            tmp=ElleDefaultFlynnRealAttribute(DISLOCDEN);
        else
            OnError("ElleGetFlynnDislocDen",err);
    }
    *valptr = (float)tmp;
}

void ElleSetFlynnAge(int index, float val, int type)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,type,(double)val))
        OnError("ElleSetFlynnAge",err);
}

void ElleGetFlynnAge(int index, float *valptr, int type)
{
    int err=0;
    double tmp;
    if (err = flynns.getFlynnAttribute(index,type,&tmp)) {
        if (err==ATTRIBID_ERR)
            tmp=ElleDefaultFlynnRealAttribute(type);
        else
            OnError("ElleGetFlynnAge",err);
    }
    *valptr = (float)tmp;
}

void ElleSetFlynnColour(int index, int val)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,COLOUR,val))
        OnError("ElleSetFlynnColour",err);
}

void ElleGetFlynnColour(int index, int *valptr)
{
    int err=0;
    if (err = flynns.getFlynnAttribute(index,COLOUR,valptr)) {
        if (err==ATTRIBID_ERR)
            *valptr=ElleDefaultFlynnIntAttribute(COLOUR);
        else
            OnError("ElleGetFlynnColour",err);
    }
}

void ElleSetFlynnMineral(int index, int val)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,MINERAL,val))
        OnError("ElleSetFlynnMineral",err);
}

void ElleSetFlynnSplit(int index, int val)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,SPLIT,val))
        OnError("ElleSetFlynnSplit",err);
}

void ElleSetFlynnGrain(int index, int val)
{
    int err=0;
    if (err = flynns.setFlynnAttribute(index,GRAIN,val))
        OnError("ElleSetFlynnGrain",err);
}

void ElleGetFlynnMineral(int index, int *valptr)
{
    int err=0;
    if (err = flynns.getFlynnAttribute(index,MINERAL,valptr)) {
        if (err==ATTRIBID_ERR)
            *valptr=ElleDefaultFlynnIntAttribute(MINERAL);
        else
            OnError("ElleGetFlynnMineral",err);
    }
}

void ElleGetFlynnSplit(int index, int *valptr)
{
    int err=0;
    if (err = flynns.getFlynnAttribute(index,SPLIT,valptr)) {
        if (err==ATTRIBID_ERR)
            *valptr=ElleDefaultFlynnIntAttribute(SPLIT);
        else
            OnError("ElleGetFlynnSplit",err);
    }
}

void ElleGetFlynnGrain(int index, int *valptr)
{
    int err=0;
    if (err = flynns.getFlynnAttribute(index,GRAIN,valptr)) {
        if (err==ATTRIBID_ERR)
            *valptr=ElleDefaultFlynnIntAttribute(GRAIN);
        else
            OnError("ElleGetFlynnGrain",err);
    }
}
/*
 * set the new default value and reset the value for
 * any flynns which currently have the default value (??)
 * (should only be called at the beginning of a run)
 */
void ElleSetDefaultFlynnExpand(int val)
{
    int dflt;

    if (flynns.hasAttribute(EXPAND)) {
        flynns.getDfltAttribute(EXPAND,&dflt);
        flynns.setDfltAttribute(EXPAND,val);
    }
    else
        OnError("ElleSetDefaultFlynnExpand",ATTRIBID_ERR);
/*
    max = ElleMaxFlynns();
    for (j=0;j<max;j++) {
        if (flynns.isFlynnActive(j) &&
                    flynns.hasFlynnAttribute(j,EXPAND)) {
            flynns.getFlynnAttribute(j,EXPAND,&curr);
            if (curr == dflt)
                flynns.setFlynnAttribute(j,EXPAND,val);
        }
    }
*/
}

void ElleSetDefaultFlynnGrain(int val)
{
    int dflt;

    if (flynns.hasAttribute(GRAIN)) {
        flynns.getDfltAttribute(GRAIN,&dflt);
        flynns.setDfltAttribute(GRAIN,val);
    }
    else
        OnError("ElleSetDefaultFlynnGrain",ATTRIBID_ERR);
}

void ElleSetDefaultFlynnSplit(int val)
{
    int dflt;

    if (flynns.hasAttribute(SPLIT)) {
        flynns.getDfltAttribute(SPLIT,&dflt);
        flynns.setDfltAttribute(SPLIT,val);
    }
    else
        OnError("ElleSetDefaultFlynnSplit",ATTRIBID_ERR);
}

void ElleSetDefaultFlynnMineral(int val)
{
    int dflt;

    if (flynns.hasAttribute(MINERAL)) {
        flynns.getDfltAttribute(MINERAL,&dflt);
        flynns.setDfltAttribute(MINERAL,val);
    }
    else
        OnError("ElleSetDefaultFlynnMineral",ATTRIBID_ERR);
}

void ElleSetDefaultFlynnColour(int val)
{
    int dflt;

    if (flynns.hasAttribute(COLOUR)) {
        flynns.getDfltAttribute(COLOUR,&dflt);
        flynns.setDfltAttribute(COLOUR,val);
    }
    else
        OnError("ElleSetDefaultFlynnColour",ATTRIBID_ERR);
}

float ElleDefaultFlynnEnergy()
{
    double val;

    flynns.getDfltAttribute(ENERGY,&val);
    return((float)val);
}

float ElleDefaultFlynnViscosity()
{
    double val;

    flynns.getDfltAttribute(VISCOSITY,&val);
    return((float)val);
}

float ElleDefaultFlynnDislocDen()
{
    double val;

    flynns.getDfltAttribute(DISLOCDEN,&val);
    return((float)val);
}

int ElleDefaultFlynnColour()
{
    int val;

    flynns.getDfltAttribute(COLOUR,&val);
    return(val);
}

int ElleDefaultFlynnSplit()
{
    int val;

    flynns.getDfltAttribute(SPLIT,&val);
    return(val);
}

int ElleDefaultFlynnGrain()
{
    int val;

    flynns.getDfltAttribute(GRAIN,&val);
    return(val);
}

int ElleDefaultFlynnMineral()
{
    int val;

    flynns.getDfltAttribute(MINERAL,&val);
    return(val);
}

int ElleDefaultFlynnExpand()
{
    int val;

    flynns.getDfltAttribute(EXPAND,&val);
    return(val);
}

void ElleSetDefaultFlynnDislocDen(float val)
{
    double dflt;

    if (flynns.hasAttribute(DISLOCDEN)) {
        flynns.getDfltAttribute(DISLOCDEN,&dflt);
        flynns.setDfltAttribute(DISLOCDEN,(double)val);
    }
    else
        OnError("ElleSetDefaultFlynnDislocDen",ATTRIBID_ERR);
/*
    max = ElleMaxFlynns();
    for (j=0;j<max;j++) {
        if (flynns.isFlynnActive(j) &&
                    flynns.hasFlynnAttribute(j,DISLOCDEN)) {
            flynns.getFlynnAttribute(j,DISLOCDEN,&curr);
            if (curr == dflt)
                flynns.setFlynnAttribute(j,DISLOCDEN,val);
        }
    }
*/
}

void ElleSetDefaultFlynnViscosity(float val)
{
    double dflt;

    if (flynns.hasAttribute(VISCOSITY)) {
        flynns.getDfltAttribute(VISCOSITY,&dflt);
        flynns.setDfltAttribute(VISCOSITY,(double)val);
    }
    else
        OnError("ElleSetDefaultFlynnViscosity",ATTRIBID_ERR);
}

void ElleSetDefaultFlynnEnergy(float val)
{
    int j, max;
    double dflt;

    if (flynns.hasAttribute(ENERGY)) {
        flynns.getDfltAttribute(ENERGY,&dflt);
        flynns.setDfltAttribute(ENERGY,(double)val);
    }
    else
        OnError("ElleSetDefaultFlynnEnergy",ATTRIBID_ERR);
}

// not used with hierarchy
void ElleCheckFlynnDefaults(int id)
{
    int max,i;
    int idflt;
    double fdflt;
    double alpha, beta, gamma;
    Coords_3D xyzdflt;

    /* this is only working for one level of flynns - fix ****/
    /* too complex  - fix ****/
    switch(id) {
    case EXPAND  :
    case COLOUR  :
    case SPLIT   :
    case GRAIN   :
    case MINERAL :
                  flynns.getDfltAttribute(id,&idflt);
                  break;
    case VISCOSITY:
    case S_EXPONENT:
    case VELOCITY:
    case ENERGY  :
    case AGE:
    case CYCLE:
    case DISLOCDEN:
    case F_ATTRIB_A:
    case F_ATTRIB_B:
    case F_ATTRIB_C:
    case F_ATTRIB_D:
    case F_ATTRIB_E:
    case F_ATTRIB_F:
    case F_ATTRIB_I:
    case F_ATTRIB_J:
    case F_ATTRIB_K:
    case TAU_XX:
    case TAU_YY:
    case TAU_ZZ:
    case TAU_XY:
    case TAU_1:
    case PRESSURE:
                  flynns.getDfltAttribute(id,&fdflt);
                  break;
    case CAXIS   :
                  flynns.getDfltAttribute(id,&xyzdflt);
                  break;
    case EULER_3 :
                  flynns.getDfltAttribute(E3_ALPHA,&alpha);
                  flynns.getDfltAttribute(E3_BETA,&beta);
                  flynns.getDfltAttribute(E3_GAMMA,&gamma);
                  break;
    case INCR_S:
    case F_INCR_S:
    case BULK_S:
    case F_BULK_S:
    case E_XX    :
    case E_XY    :
    case E_YX    :
    case E_YY    :
    case E_ZZ    :
                  flynns.getDfltAttribute(id,&fdflt);
                  break;
    default      : OnError("ElleCheckFlynnDefaults",0);
                  break;
    }
    //
    // initialise the attribute for the top level flynns
    //
    max = ElleMaxFlynns();
    for (i=0; i<max; i++) {
        if (ElleFlynnIsActive(i) && !hasParent(i)
                                 && !ElleFlynnHasAttribute(i,id))
            switch(id) {
            case EXPAND  :
            case COLOUR  :
            case SPLIT   :
            case GRAIN   :
            case MINERAL :
                      flynns.setFlynnAttribute(i,id,idflt);
                      break;
            case VISCOSITY:
            case S_EXPONENT:
            case ENERGY  :
            case AGE:
            case CYCLE:
            case DISLOCDEN:
            case F_ATTRIB_A:
            case F_ATTRIB_B:
            case F_ATTRIB_C:
            case F_ATTRIB_D:
            case F_ATTRIB_E:
            case F_ATTRIB_F:
            case F_ATTRIB_I:
            case F_ATTRIB_J:
            case F_ATTRIB_K:
            case TAU_XX:
            case TAU_YY:
            case TAU_ZZ:
            case TAU_XY:
            case TAU_1:
            case PRESSURE:
                      flynns.setFlynnAttribute(i,id,fdflt);
                      break;
            case EULER_3 :
                      flynns.setFlynnAttribute(i,E3_ALPHA,alpha);
                      flynns.setFlynnAttribute(i,E3_BETA,beta);
                      flynns.setFlynnAttribute(i,E3_GAMMA,gamma);
                      break;
            case INCR_S:
            case F_INCR_S:
            case BULK_S:
            case F_BULK_S:
            case E_XX    :
            case E_XY    :
            case E_YX    :
            case E_YY    :
            case E_ZZ    :
                      flynns.setFlynnAttribute(i,id,fdflt);
                      break;
            case CAXIS   :
                      flynns.setFlynnAttribute(i,id,&xyzdflt);
                      break;
            }
    }
}
