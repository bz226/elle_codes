 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <iterator>
#include <cstdlib>
#include "attrib.h"
#include "flynnarray.h"
#include "flynns.h"
#include "attribute.h"
#include "nodes.h"
#include "bflynns.h"
#include "interface.h"
#include "file.h"
#include "error.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
std::ostream & operator<< (std::ostream &os, const Flynn &t)
{
    if (t.hasParent())
        os << "parent=" << t.getParent() << " ";
    if (t.isParent())
        copy(t.child_list.begin(),t.child_list.end(),
						std::ostream_iterator<int>(os," "));
    return os;
}

void Flynn::addChild( int child )
{
    if (child_list.size()>0) {
        it = child_list.begin();
        while ((*it) < child && it!=child_list.end()) it++;
        if ((*it) == child) //flag error 
            OnError("addChild: child already in list",0);
        child_list.insert(it,child);
    }
    else 
        child_list.push_back(child);
}

void Flynn::removeChild( int child )
{
    /*it = find(child_list.begin(),child_list.end(),child);*/
    it = child_list.begin();
    while (it!=child_list.end() && *it!=child) it++;
    if (it != child_list.end()) child_list.erase(it);
    else //flag error 
        OnError("removeChild: child not in list",0);
}

int Flynn::removeUnode( int id )
{
    int err=1;
    it = unode_list.begin();
    while (it!=unode_list.end() && *it!=id) it++;
    if (it != unode_list.end()) {
        unode_list.erase(it);
        err=0;
    }
    return(err);
}

/*
int addVoronoiPt(const Coords pt)
{
    int i=NO_VAL;
    double eps=1.5e-6; // same eps as unodes.cc
    vector<Coords>::iterator itvc=_vpoints.begin();
    while (itvc!=_vpoints.end() &&
               fabs(itvc->x-pt.x)>eps && fabs(itvc->y-pt.y)>eps) {
        itvc++; i++;
    }
    if (itvc==_vpoints.end()) { _vpoints.push_back(pt); i++; }
    return(i);
}

int Flynn::getFirstNode()
{
    int first=NO_NB;

    if (!isParent()) first=first_node;
    else {
    	std::list<int> LClist;
        makeLClist( LClist );
        first = findFirstNode( LClist );
    }
    return first;
}
*/
/*
 ****************this doesn't work for 2-sided grains!!!!!*********
 */

void Flynn::flynnNodes(int flynnid, int **id, int *num)
{
    int i=0,*iptr;
    int start,curr,prev,next,second;
	std::vector<int> nodes;

    start = curr = prev = ElleGetFlynnFirstNode(flynnid);
    nodes.push_back(start);
    i=1;
    if (ElleCheckForTwoSidedGrain(start,&next) && 
            ElleTwoSidedGrainId(start,next)==flynnid ) {
        nodes.push_back(next);
        i++;
    }
    else {
        while ((next = ElleFindBndNode(curr,prev,flynnid))!=start) {
            nodes.push_back(next);
            prev = curr;
            curr = next;
            i++;
        }
    }
    *num = i;
    if ((*id = (int *)malloc(*num * sizeof(int)))==0)
        OnError("flynnNodes",MALLOC_ERR);
    for (i=0,iptr=*id;i<*num;i++) iptr[i] = nodes[i];
}

void Flynn::flynnNodes(int flynnid, std::vector<int> &ids)
{
    int start,curr,prev,next;
    int len, cnt=0;

    len = ids.size();
    start = curr = prev = ElleGetFlynnFirstNode(flynnid);
    if (cnt < len) ids[cnt] = start;
    else ids.push_back(start);
    cnt++;
    if (ElleCheckForTwoSidedGrain(start,&next)) {
        if (cnt < len) ids[cnt] = next;
        else ids.push_back(next);
        cnt++;
    }
    else {
        while ((next = ElleFindBndNode(curr,prev,flynnid))!=start) {
            if (cnt < len) ids[cnt] = next;
            else ids.push_back(next);
            cnt++;
            prev = curr;
            curr = next;
        }
    }
    while (len > cnt) { ids.pop_back(); len--; }
}

int Flynn::flynnNodeCount(int flynnid)
{
    int i,start,curr,prev,next;

    start = curr = prev = ElleGetFlynnFirstNode(flynnid);
    if (start==NO_NB) i=0;
    else {
        if (ElleCheckForTwoSidedGrain(start,&next)) {
            i=2;
        }
        else {
            i=1;
            while ((next = ElleFindBndNode(curr,prev,flynnid))!=start) {
                prev = curr;
                curr = next;
                i++;
            }
        }
    }
    return(i);
}

/*
 * find a node on the boundary of a parent flynn
 * ie separating a flynn in the child list from one that is
 *      not a child of this parent flynn
 * only called if the flynn has children
 */
int Flynn::findFirstNode(std::list<int> &LClist)
{
    int first=NO_NB, rgns[3], i;
	std::vector<int>::iterator v_it;
	std::list<int>::iterator l_it;
    it = LClist.begin();
    while (it!=LClist.end() && first==NO_NB) {
		std::vector<int> nodes;
        flynnNodes(*it,nodes);
        for (v_it=nodes.begin();v_it!=nodes.end() && first==NO_NB;
                  v_it++) {
            ElleRegions(*v_it,rgns);
            i=0;
            while (i<3 && first==NO_NB) {
                if (rgns[i]!=NO_NB &&
                    (l_it=find(LClist.begin(),LClist.end(),rgns[i]))
                                ==LClist.end()) first=*v_it;
                i++;
            }
        }
        it++;
    }
    return first;
}

/*
 * make a list of all the lowest level children of this flynn
 *      (ie children which are not parents)
int Flynn::makeLClist(std::list<int> &LClist)
{
    std::list<int>::iterator l_it;
    it = child_list.begin();
    while (it!=child_list.end()) {
        if (flynns.a[*it].isParent()) flynns.a[*it].makeLClist(LClist);
        else LClist.push_back(*it);
        it++;
    }
    return 0;
}
 */

void Flynn::getAttributeList(int *attr, int maxa)
{
    attributes.getList(attr, maxa);
}

void Flynn::getIntAttributeList(std::vector<int> &attr)
{
    attributes.getIntList(attr);
}

void Flynn::clean()
{
    state = INACTIVE;
    parent = NO_NB;
    first_node = NO_NB;
    removeChildList();
    removeUnodeList();
    /*removeVoronoiPts();*/
    attributes.clean();
}
