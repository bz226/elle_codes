 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
/*!
    \file       voronoi2flynns.cc
    \brief      convert unode voronoi regions to flynns
    \par        Description:
        Functions required to convert unode voronoi
        regions into flynns with the same attributes
        as the unodes
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <utility>
#include "attrib.h"
#include "nodes.h"
#include "unodes.h"
#include "update.h"
#include "error.h"
#include "parseopts.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "tripoly.h"
#include "setup.h"
#include "init.h"
#include "check.h"
#include "general.h"
#include "polygon.h"
#include "polyutils.h"
#include "gpcclip.h"
#include "splitting.h"
#include "error.h"
                                                                                
using std::string;
using std::ofstream;
using std::list;
using std::vector;
using std::pair;
using std::cout;
using std::endl;

void updateLists(int node1, int node2, int nn, int curr_rgn);
void copyUnodeAttributesToFlynn(int unodeid, int flynnid);
void initDfltFlynnAttributesFromUnodes();
int InsertAndUpdateBnode(int bnode1, int bnode2, int dj,
                        int *nb_bnode);

int Voronoi2Flynns(bool voronoi_by_flynn)
{
    int i, k, j;
    int err=0,max,first, last,prev, dj, nb_bnode, curr_node;
    bool matched=false;
    double area=0;
    Coords xy, vxy;
    string ofilename(ElleFile()), areaname(ElleFile());
    extern VoronoiData *V_data;
    
    /*!
      voronoi_by_flynn==false: voronoi ignores flynn boundaries
      voronoi_by_flynn!=true: voronoi is for each flynn
      if the voronoi regions are calculated successfully,
      the points and segments stored as flynns with id of
      the enclosed unode.
      The flynns will be assigned the attributes of the corresponding
      unode
      WARNINGS:
      any original flynns and bnodes are deleted
      the bnode attributes are not applied or interpolated
     */

    if (voronoi_by_flynn) err = ElleVoronoiFlynnUnodes();
    else err = ElleVoronoiUnodes();
    ElleCleanFlynnArray();
    ElleCleanNodeArray();

    if (ElleVoronoiActive() && !err) {
        int count = 0, seg=0, last;
        list<int> pt_list;
        list<int> :: iterator it;
        Coords pos;
        for (i=0;i<ElleMaxUnodes();i++) {
            pt_list.clear();
            ElleUnodeVoronoiPts(i,pt_list);
            /*cout << i << ' ' ; */
            for (it = pt_list.begin();it != pt_list.end(); it++) {
                if (*it>=ElleMaxNodes() || !ElleNodeIsActive(*it)) {
                  ElleSetNodeActive(*it);
                  V_data->vpoints[*it].getPosition(&pos);
                  ElleSetPosition(*it,&pos);
				}
            /*cout << *it << ' ' ; */
			}
        /*cout << endl;*/
		}

        initDfltFlynnAttributesFromUnodes();

        for (i=0;i<ElleMaxUnodes();i++) {
            pt_list.clear();
            ElleUnodeVoronoiPts(i,pt_list);
            if ((count=pt_list.size())>2) { // how many segs is minimum?
                it = pt_list.begin();
                last = pt_list.back();
                first = prev = *it;
                if (!ElleFlynnIsActive(i)) {
                  ElleSetFlynnActive(i);/* fixes i>MaxGrains */
                }
                if (prev>=ElleMaxNodes() || !ElleNodeIsActive(prev)) {
                  ElleSetNodeActive(prev);
                  V_data->vpoints[prev].getPosition(&pos);
                  ElleSetPosition(prev,&pos);
                }
                seg=0;
                while (seg<count) {
                    if (seg==(count-1))
                      curr_node = ElleGetFlynnFirstNode(i);
                    else curr_node = *(++it);
                    // check curr_node can be attached to prev
                    if (curr_node>=ElleMaxNodes() || !ElleNodeIsActive(curr_node)) {
                      ElleSetNodeActive(curr_node);
                      V_data->vpoints[curr_node].getPosition(&pos);
                      ElleSetPosition(curr_node,&pos);
                    }
                    else if (ElleNodeIsTriple(curr_node)) {
                      if ((k=ElleFindNbIndex(prev,curr_node))==NO_NB) {
                          dj = ElleFindSpareNode();
                          err = InsertAndUpdateBnode(curr_node, prev, dj,
                                                     &nb_bnode);
#if XY
#endif
                          updateLists(curr_node,nb_bnode,dj,i);
                          updateLists(nb_bnode,prev,dj,i);
                          if (prev == last) last = dj;
                          curr_node = dj;
                      }
                    }
                    if (ElleGetFlynnFirstNode(i)==NO_VAL)
                      ElleSetFlynnFirstNode(i,prev);
                    // check prev so curr_node can attach to it
                    if (ElleNodeIsTriple(prev)) {
                      if ((k=ElleFindNbIndex(curr_node,prev))==NO_NB) {
                          dj = ElleFindSpareNode();
                          err = InsertAndUpdateBnode(prev, curr_node, dj,
                                                     &nb_bnode);
                          ElleUpdateFirstNodes(prev,dj,i);
                          updateLists(prev,nb_bnode,dj,i);
                          updateLists(curr_node,nb_bnode,dj,i);
                          if (prev==last) last = dj;
                          if (prev==first) first = dj;
                          prev = dj; 
                      }
                    }
                    if (err=ElleSetNeighbour(prev,
                            ElleFindNbIndex(curr_node,prev),
                            curr_node,&i))
                        return(err);
                    prev = curr_node;
                    seg++;
                }
                if (err=ElleSetNeighbour(last,
                        ElleFindNbIndex(first,last),
                        first,&i))
                    return(err);
                copyUnodeAttributesToFlynn(i,i);
            }
        }
        //ElleWriteData("vor2flynn.elle");
    }
    return(err);
}

int InsertAndUpdateBnode(int bnode1, int bnode2, int dj,
                        int *nb_bnode)
{
    int err = 0;
    int j, k;
    int nb[3];
    double eps=1.5e-6; // same eps as crossings.cc and unodes.cc
    Coords xy, tmp;
    vector< Coords > ptcoords(4), nodecoords(3);

    ElleNodePosition(bnode1,&xy);
    ElleNeighbourNodes(bnode1,nb);
    for (j=0; j<3 ; j++) {
        ElleNodePlotXY(nb[j],&ptcoords[j],&xy);
        nodecoords[j] = ptcoords[j];
    }
    ElleNodePlotXY(bnode2,&ptcoords[3],&xy);
    sortCoordsOnAngle(ptcoords,&xy);
    ElleNodePosition(bnode2,&tmp);
    ElleCoordsPlotXY(&tmp,&xy);

    int index1, index2;
    j=0;
    while (j<4 && (fabs(ptcoords[j].x-tmp.x)>eps ||
                   fabs(ptcoords[j].y-tmp.y)>eps )) j++;
    if (j==0) j=4; j--;
    tmp = ptcoords[j];
    index1=0;
    while (index1<3 && (fabs(nodecoords[index1].x-tmp.x)>eps ||
                        fabs(nodecoords[index1].y-tmp.y)>eps )) index1++;
    if (index1==3) cout << "HELP\n";
    j = (j+2)%4;
    tmp = ptcoords[j];
    index2=0;
    while (index2<3 && (fabs(nodecoords[index2].x-tmp.x)>eps ||
                        fabs(nodecoords[index2].y-tmp.y)>eps )) index2++;
    if (index2==3) cout << "HELP\n";
    xy.x = nodecoords[index2].x - (nodecoords[index2].x-xy.x)*0.9;
    xy.y = nodecoords[index2].y - (nodecoords[index2].y-xy.y)*0.9;
    ElleSetPosition(dj,&xy);
    int rgns[3], nbrgns[3];
    ElleRegions(bnode1,rgns);
    ElleSetNeighbour(bnode1,index2,dj,0);
    ElleSetNeighbour(dj,NO_VAL,nb[index2],&rgns[index2]);
    if ((k=ElleFindNbIndex(bnode1,nb[index2]))!=NO_NB) {
        ElleRegions(nb[index2],nbrgns);
        ElleSetNeighbour(nb[index2],k,dj,0);
        ElleSetNeighbour(dj,NO_VAL,bnode1,&nbrgns[k]);
        /*if (prev==first) first = dj;*/
    }
    else ElleSetNeighbour(dj,NO_NB,bnode1,0);
    if ((k=ElleFindNbIndex(bnode1,bnode2))!=NO_NB) {
        ElleRegions(bnode2,nbrgns);
        ElleSetNeighbour(bnode2,k,dj,0);
        ElleSetRegionEntry(dj, ElleFindNbIndex(bnode1,dj), nbrgns[k]);
    }
    *nb_bnode = nb[index2];
    return(err);
}

void updateLists(int node1, int node2, int nn, int curr_rgn)
{
    int i, nb1, nb2, nb[3], prev;
    bool inserted;
    list<int> pt_list;
    list<int> :: iterator it;
    list<int> :: reverse_iterator itr;
    ElleNeighbourNodes(nn,nb);
    i=0;
    while (i<3 && nb[i]==NO_NB) i++;
    if (i<3) nb1 = nb[i];
    i++;
    while (i<3 && nb[i]==NO_NB) i++;
    if (i<3) nb2 = nb[i];
    // ERROR CHECK needed
    for (i=curr_rgn+1;i<ElleMaxUnodes();i++) {
        pt_list.clear();
        ElleUnodeVoronoiPts(i,pt_list);
        inserted = false;
        it = pt_list.begin();
        prev = pt_list.back();
        while (it!=pt_list.end() && !inserted) {
          if (prev==node1 && *it==node2) {
            ElleUnodeInsertVoronoiPt(i,node2,nn);
            inserted = true;
          }
          else if (prev==node2 && *it==node1) {
            ElleUnodeInsertVoronoiPt(i,node1,nn);
            ElleUnodeDeleteVoronoiPt(i,node1);
            inserted = true;
          }
          else {
            prev = *it;
            it++;
          }
        }
    }
}

        // Set the flynn attributes equal to the default unode
        //   attributes
void initDfltFlynnAttributesFromUnodes()
{
    int i, *attr_id, maxa=0;
    int f_attr = NO_VAL;
    double val=0, dflt=0;
    list<int> pt_list;
    list<int> :: iterator it;

    ElleUnodeAttributeList(&attr_id, &maxa);
    for (i=0; i<maxa; i++) {
        if (attr_id[i]>400) f_attr = attr_id[i]-400;
        else f_attr = attr_id[i];
        if (isFlynnAttribute(f_attr) ) {
            if (!ElleFlynnAttributeActive(f_attr))
                ElleInitFlynnAttribute(f_attr);
            ElleGetDefaultUnodeAttribute(&dflt,attr_id[i]);
            ElleSetDefaultFlynnRealAttribute(dflt,f_attr);
        }
    }
    if (attr_id!=0) free(attr_id);
}

void copyUnodeAttributesToFlynn(int unodeid, int flynnid)
{
    int i, j, *attr_id, maxa=0;
    int u_attr = NO_VAL;
    double val=0, dflt=0;
    extern vector<Unode> *Unodes;

    ElleFlynnDfltAttributeList(&attr_id, &maxa);
    for (j=0; j<maxa; j++) {
        if (attr_id[j]==VISCOSITY) u_attr = U_VISCOSITY;
        else u_attr = attr_id[j];
        if ((*Unodes)[unodeid].hasAttribute(u_attr)) {
            ElleGetUnodeAttribute(unodeid, u_attr, &val);
            ElleSetFlynnRealAttribute(flynnid, val, attr_id[j]);
        }
    }
    if (attr_id!=0) free(attr_id);
}
