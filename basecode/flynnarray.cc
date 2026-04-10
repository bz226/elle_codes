 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <fstream>
#include <iterator>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include "attrib.h"
#include "pointdata.h"
#include "consts.h"
#include "flynnarray.h"
#include "splitting.h"
#include "interface.h"
#include "tripoly.h"
#include "runopts.h"
#include "polygon.h"
#include "file.h"
#include "error.h"
#include "unodes.h"
#include "mineraldb.h"
#include "log.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

using std::ofstream;
using std::cout;
using std::endl;
using std::vector;
using std::list;
using std::pair;

void FlynnArray::setFlynnActive(int index)
{
    int i;

    if (index >= (int)_a.size()) {
        Flynn tmp;
        for (i=_a.size()-1;i<index;i++) _a.push_back(tmp);
    }
    _a[index].setState(ACTIVE);
}

//
// if the flynn has a parent -
// add its children to the parent's list & reset childrens' parent
// it should only be deleted if it has no attributes??
//     so assume there aren't any (this may change)
//
void FlynnArray::removeFlynn(int index)
{
    int i, pindex=NO_NB;
    vector<int> l(0);
    vector<int>::iterator it;

    if (index < (int)_a.size()) {
        if (_a[index].hasParent()) {
            pindex = _a[index].getParent();
            _a[pindex].removeChild(index);
        }
        if (_a[index].isParent()) {
            _a[index].getChildList(l);
            for (it = l.begin();it!=l.end();it++){
                if (pindex!=NO_NB) _a[pindex].addChild(*it);
                _a[(*it)].setParent(pindex);
            }
        }
#ifdef DEBUG
int rgns[3];
int max=ElleMaxNodes();
for(i=0;i<max;i++) {
if (ElleNodeIsActive(i)) {
ElleRegions(i,rgns);
for(int j=0;j<3;j++)
if (rgns[j]==index)
printf("rgn found %d %d\n",i,j);
}
}
#endif
        _a[index].clean();
    }
    else OnError("removing invalid flynn",0);
}

//
// if the flynn has a parent -
// if the disappearing flynn was an only child, delete the parent
// do this with attributes??? (flynns are removed if they have
//     no attributes)
//
void FlynnArray::removeShrinkingFlynn(int index)
{
    int pindex=NO_NB;

    if (index < (int)_a.size()) {
        pindex = index;
        do {
          index = pindex;
          if (_a[index].hasParent()) {
            pindex = _a[index].getParent();
            _a[pindex].removeChild(index);
          }
          else pindex = NO_NB;
          _a[index].clean();
        } while(pindex!=NO_NB && !_a[pindex].isParent());
    }
    else OnError("removing invalid flynn",0);
}

void FlynnArray::promoteFlynn(int flynnindex)
{
    int pindex = NO_NB, gpindex = NO_NB;

    if (flynnindex < (int)_a.size()) {
        pindex = _a[flynnindex].getParent();
        if (pindex != NO_NB) {
            copyAttributes(pindex,flynnindex);
            gpindex = _a[pindex].getParent();
            _a[flynnindex].setParent(_a[pindex].getParent());
            _a[pindex].removeChild(flynnindex);
            if (gpindex!=NO_NB) _a[gpindex].addChild(flynnindex);
            if (!_a[pindex].isParent()) removeFlynn(pindex);
        }
    }
}

void FlynnArray::clean()
{
    int max=_a.size();
    for (int i=0; i<max; i++)
        if (_a[i].getState()) _a[i].clean();
    _dfltAttributes.clean();
}

void FlynnArray::setColor(int index,bool c, int r,int g, int b)
{
      if (index < (int)_a.size())
              _a[index].setColor(c,r,g,b);
}

void FlynnArray::getColor(int index,bool *c,int *r,int *g, int *b)
{
      if (index < (int)_a.size())
              _a[index].getColor(c,r,g,b);
}

bool FlynnArray::isFlynnActive(int index)
{
    if (index < (int)_a.size())
        return(_a[index].getState());
    return(0);
}

int FlynnArray::spareFlynn()
{
    int index=0;
    while (index<(int)_a.size() && _a[index].getState()) index++;
    setFlynnActive(index);;
    return(index);
}

int FlynnArray::maxLevels()
{
    int count=0, max=0, p;
    for (it_a=_a.begin(),p=0;it_a!=_a.end();it_a++,p++) {
        if ((*it_a).getState() && !(*it_a).isParent()) {
            count = getFlynnLevel(p);
            if (count>max) max=count;
        }
    }
    return(max);
}

int FlynnArray::getFlynnLevel(const int index)
{
    int p, count=0;
    p = index;
    if (_a[p].getState()) {
        count=1;
        while ((p=_a[p].getParent())!=NO_NB) count++;
    }
    return(count);
}

void FlynnArray::setFlynnFirstnode(int index, int first)
{
    if (index >= (int)_a.size()) OnError("setFlynnFirstnode",
                                       RANGE_ERR);
    _a[index].setFirstNode(first);
}

double FlynnArray::flynnArea(const int index)
{
    double area=0.0;

    if (!_a[index].isParent())
        area=ElleRegionArea(index);
    else {
        list<int> LClist;
        list<int>::iterator it;
        makeLClist(index,LClist);
        for (it=LClist.begin();it!=LClist.end();it++)
            area += ElleRegionArea(*it);
    }
    return area;
}

int FlynnArray::getFlynnFirstnode(int index)
{
    int first;
    /*if (!_a[index].isParent())*/
        /*first=_a[index].getFirstNode();*/
    /*else first=_a[index].findFirstNode();*/
    if (!_a[index].isParent()) first=_a[index].getFirstNode();
    else {
        list<int> LClist;
        makeLClist(index,LClist);
        first = _a[index].findFirstNode(LClist);
    }
    return first;
}

/*
 * make a list of all the lowest level children of this flynn
 *      (ie children which are not parents)
 */
int FlynnArray::makeLClist(int index,list<int> &LClist)
{
    vector<int> l(0);
    vector<int>::iterator l_it;
    _a[index].getChildList(l);
    l_it = l.begin();
    while (l_it!=l.end()) {
        if (_a[*l_it].isParent()) makeLClist(*l_it,LClist);
        else LClist.push_back(*l_it);
        l_it++;
    }
    return 0;
}

void FlynnArray::addFlynnChild(int index,int child)
{
    if (index >= (int)_a.size()) OnError("addFlynnChild",LIMIT_ERR);
    if (_a[index].getState()==INACTIVE) 
        OnError("addFlynnChild",GRNNUM_ERR);
    else {
        _a[child].setParent(index);
        _a[index].addChild(child);
    }
}

void FlynnArray::printFlynn(int index)
{
    if (index < (int)_a.size()) cout << _a[index];
}

//
// WARNING
// flynns which have not split can be counted as both grains and
//   subgrains in the following fns
//
int FlynnArray::numberOfGrains()
{
    int count=0;

    it_a = _a.begin();
    while (it_a!=_a.end()) {
        // only top level flynns (no parent)
        if ((*it_a).getState()==ACTIVE && !((*it_a).hasParent()))
            count++;
        it_a++;
    }
    return(count);
}

int FlynnArray::numberOfSubgrains()
{
    int count=0;

    it_a = _a.begin();
    while (it_a!=_a.end()) {
        // only lowest level flynns which have a parent
        if ((*it_a).getState()==ACTIVE && !((*it_a).isParent())
                                       && (*it_a).hasParent())
            count++;
        it_a++;
    }
    return(count);
}

//
// true if flynn1 is a child of flynn2
//
bool FlynnArray::isChildOf(int flynn1, int flynn2)
{
    int curr,next;

    if (flynn1 >= 0 && flynn1 < (int)_a.size()) {
        curr = flynn1;
        while ((next=(_a[curr].getParent()))!=NO_NB) {
            if (next==flynn2) return(1);
            else curr = next;
        }
    }
    return(0);
}

void FlynnArray::updateFirstNodes(int old, int nn, const int flynnindex)
{
    int p;

    if (_a[flynnindex].getFirstNode()==old)
        _a[flynnindex].setFirstNode(nn);
    p = flynnindex;
    while (_a[p].hasParent()) {
        p = _a[p].getParent();
        if (_a[p].getFirstNode()==old) _a[p].setFirstNode(nn);
    }
}

void FlynnArray::newFlynn(int start, int end, int oldindex,
                          int *newindex)
{
    int i, j, n, prev;
    *newindex = spareFlynn();

    _a[*newindex].setFirstNode(start);
    /*
     * reset the boundary node region entries
     */
    prev = end;
    n = start;
    do {
        if ((j=ElleNewFindBndNb(n,prev,oldindex))==NO_NB)
            OnError("New flynn",NONB_ERR);
        if ((i = ElleFindNbIndex(j,n))==NO_NB)
            OnError("New flynn nb index",NONB_ERR);
        ElleSetRegionEntry(n,i,*newindex);
        prev = n;
        n = j;
    } while (n!= end);
    /*
     * close the flynn boundary
     */
    if (ElleNodeIsDouble(start) &&
                              ElleNodeIsDouble(end)) {
        ElleJoinTwoDoubleJ(start,end,oldindex,*newindex);
    }
    else {
        if ((i = ElleFindNbIndex(start,end))==NO_NB)
            OnError("New flynn nb index",NONB_ERR);
        ElleSetRegionEntry(end,i,*newindex);
    }

    _a[oldindex].setFirstNode(start);
    /*copyAttributes(oldindex,*newindex);*/
}

void FlynnArray::neighbourRegions(const int flynnid, list<int> &nbflynns)
{
    int i, k, prev;
    ERegion rgn;
    list<int>::iterator its, its1, its2;
    list<int>tmp, tmpcnt;
    vector<int>::reverse_iterator vi;
    vector<int>ids;
    list< pair<int,double> > segs;
    list< pair<int,double> >::iterator segs_it;
    list< pair<double,int> > len_order;
    list< pair<double,int> >::iterator it;

    nbflynns.erase(nbflynns.begin(),nbflynns.end()); // clear the list

    /*
     * for all the boundary nodes,
     * make a temporary list of the adjacent flynns
     * (except for the flynn itself)
     */
    flynnNodes(flynnid, ids);
    ids.push_back(ids[0]); //close polygon
    vi=ids.rbegin();
    prev = *vi; vi++;
    while (vi!=ids.rend()) {
        ElleNeighbourRegion(prev,*vi,&rgn);
        tmp.push_back(rgn);
        pair<int,double> seg(rgn,
                                ElleNodeSeparation(*vi,prev));
        segs.push_back(seg);
        prev = *vi; vi++;
    }
    /*
     * insert the nb flynns in the nbflynn list
     * ordered on frequency (high to low)
     * Flynn with longest common boundary should
     * be first in list.
     */
    tmp.sort();
    segs.sort();
    segs_it = segs.begin();
    i = segs_it->first;
    pair<double,int> len(segs_it->second,i);
    segs_it++;
    while ( segs_it !=segs.end()) {
        if (segs_it->first==i) len.first+=segs_it->second;
        else {
            len_order.push_back(len);
            len.first=segs_it->second;
            i = len.second = segs_it->first;
        }
        segs_it++;
    }
    len_order.push_back(len);
    len_order.sort();
    for (it=len_order.begin();it!=len_order.end();it++)
        nbflynns.push_front(it->second);
#if XY
    its = tmp.begin();
    while ( its !=tmp.end()) {
        i = *its;
        k=0;
        k=count(tmp.begin(),tmp.end(),i);
        its1 = tmpcnt.begin();
        its2 = nbflynns.begin();
        while (its1!=tmpcnt.end() && *its1>k ) {its1++;its2++;}
        nbflynns.insert(its2,i);
        tmpcnt.insert(its1,k);
        while (its!=tmp.end() && (*its==i)) its++;
    }
#endif
}

int FlynnArray::checkSmallFlynn(const int index, double min_area)
{
    int deleted=0, k;
    double flynn_area;

    if (_a[index].getState()) {
        flynn_area = ElleRegionArea(index);
        if (flynn_area<min_area) {
            list<int> rgn;
            neighbourRegions(index,rgn);

            list<int>::iterator its=rgn.begin();
            for (k=0,deleted=0;its!=rgn.end() && !deleted;k++,its++) {
                if (attributesAllowMerge(*its,index)) {
                    if (mergeFlynnsNoCheck(*its,index)==0) {
                        deleted=1;
                        sprintf(logbuf, "merged %d into %d\n", index, * its );
                        Log(2,logbuf);
                    }
                }
            }
            if ( !deleted ) {
              sprintf(logbuf, "Could not merge %d - two phases\n", index );
              Log(1,logbuf);
            }
        }
    }
    return(deleted);
}

void FlynnArray::removeFlynnAttribute(const int index, const int idval)
{
    _a[index].removeFlynnAttribute(idval);
}

//---------------------------------------------------------------------------
//Function removeDfltAttribute
/*!
    \brief      removes an attribute from all flynns and from the
                default list
    \param      idval  a valid flynn attribute id
*/
void FlynnArray::removeDfltAttribute(const int idval)
{
    if (_dfltAttributes.hasAttribute(idval)) {
        int i, max;
        max = _a.size();
        for (i=0;i<max;i++) 
            if (isFlynnActive(i) && hasFlynnAttribute(i,idval))
                removeFlynnAttribute(i,idval);
        _dfltAttributes.removeAttribute(idval);
    }
}

//
// only looks at the specified flynn - does not search up the tree
//
bool FlynnArray::hasFlynnAttribute(const int index, const int id)
{
    if (index < (int)_a.size())
        return(_a[index].hasFlynnAttribute(id));
    return(0);
}

int FlynnArray::numFlynnAttributes(const int flynnindex)
{
    return(_a[flynnindex].numAttributes());
}

//
// currently, this is only called when a flynn is promoted
// (otherwise, it will have the attributes of the parent)
// so just take the attributes that it doesn't have in its own
// array (in case parent has a value for other children)
//
void FlynnArray::copyAttributes(const int src, const int dest)
{
    int *attr, maxa, i;
    int ival;
    int err=0;
    double fval;
    Coords_3D xyz;

    maxa = _a[src].numAttributes();
    attr = new int[maxa];
    _a[src].getAttributeList(attr,maxa);
    for (i=0; i< maxa; i++) {
      if (!_a[dest].hasFlynnAttribute(attr[i]))
        switch(attr[i]) {
        case EXPAND:
        case SPLIT :
        case GRAIN :
        case MINERAL:
        case COLOUR:
                     getFlynnAttribute(src,attr[i],&ival);
                     _a[dest].setFlynnAttribute(attr[i],ival);
                     break;
        case CAXIS  :getFlynnAttribute(src,attr[i],&xyz);
                     _a[dest].setFlynnAttribute(attr[i],&xyz);
                     while (i<maxa && (attr[i]==CAXIS_Z ||
                                       attr[i]==CAXIS_Y ||
                                       attr[i]==CAXIS_X ||
                                       attr[i]==CAXIS     )) i++;
                     break;
        case VISCOSITY:
        case S_EXPONENT:
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
        case ENERGY :
        case E3_ALPHA:
        case E3_BETA :
        case E3_GAMMA :
        case F_INCR_S:
        case INCR_S:
        case BULK_S:
        case F_BULK_S:
        case E_XX   :
        case E_XY   :
        case E_YX   :
        case E_YY   :
        case E_ZZ   :
        case TAU_XX   :
        case TAU_YY   :
        case TAU_ZZ   :
        case TAU_XY   :
        case PRESSURE :
                     getFlynnAttribute(src,attr[i],&fval);
                     _a[dest].setFlynnAttribute(attr[i],fval);
                     break;
        default:     err = ATTRIBID_ERR;
                     break;
       }
    }
    delete[] attr;
    if (err) OnError("copyAttributes",err);
}

int FlynnArray::setFlynnAttribute(const int index, const int id,
                                  int val)
{
    int err=0;
    if (index < (int)_a.size())
        err = _a[index].setFlynnAttribute(id,val);
    else
        err = RANGE_ERR;
    return(err);
}

int FlynnArray::setFlynnAttribute(const int index, const int id,
                                  double val)
{
    int err=0;
    if (index < (int)_a.size())
        err = _a[index].setFlynnAttribute(id,val);
    else
        err = RANGE_ERR;
    return(err);
}

int FlynnArray::setFlynnAttribute(const int index, const int id,
                                  Coords_3D *val)
{
    int err=0;
    if (index < (int)_a.size())
        err = _a[index].setFlynnAttribute(id,val);
    else
        err = RANGE_ERR;
    return(err);
}

//
// returns 0 if the flynn (or a parent) has the attribute
// else returns an error
//
int FlynnArray::getFlynnAttribute(const int index, const int id,
                                  int *valptr)
{
    bool found = 0;
    int flynnindex = index, err = 0;

    if (index >= (int)_a.size()) err = RANGE_ERR;
    else {
        found = _a[flynnindex].hasFlynnAttribute(id);
        while (!found && _a[flynnindex].hasParent()) {
            flynnindex = _a[flynnindex].getParent();
            found = _a[flynnindex].hasFlynnAttribute(id);
        }
        if (found) 
           err = _a[flynnindex].getFlynnAttribute(id,valptr);
        else err = ATTRIBID_ERR;
    }
    return(err);
}

int FlynnArray::getFlynnAttribute(const int index, const int id,
                                  double *valptr)
{
    bool found = 0;
    int flynnindex = index, err = 0;

    if (index<0 || index >= (int)_a.size()) err = RANGE_ERR;
    else {
        found = _a[flynnindex].hasFlynnAttribute(id);
        while (!found && _a[flynnindex].hasParent()) {
            flynnindex = _a[flynnindex].getParent();
            found = _a[flynnindex].hasFlynnAttribute(id);
        }
        if (found) 
           err = _a[flynnindex].getFlynnAttribute(id,valptr);
        else err = ATTRIBID_ERR;
    }
    return(err);
}

int FlynnArray::getFlynnAttribute(const int index, const int id,
                                  Coords_3D *valptr)
{
    bool found = 0;
    int flynnindex = index, err = 0;

    if (index >= (int)_a.size()) return(RANGE_ERR);
    else {
        found = _a[flynnindex].hasFlynnAttribute(id);
        while (!found && _a[flynnindex].hasParent()) {
            flynnindex = _a[flynnindex].getParent();
            found = _a[flynnindex].hasFlynnAttribute(id);
        }
        if (found) 
            err = _a[flynnindex].getFlynnAttribute(id,valptr);
        else err = ATTRIBID_ERR;
    }
    return(err);
}

int FlynnArray::mergeFlynns(int flynn1, int flynn2)
{
    int *ids, i, num_elle_nodes, cnt=0, start=NO_NB, finish=NO_NB;
    int node, err=0;

    // check that these are lowest level flynns
    // check that flynn1 != flynn2
    if (flynn1==flynn2 || _a[flynn1].isParent() || _a[flynn2].isParent() ||
        _a[flynn1].getParent()!=_a[flynn2].getParent())
        err=1;
    else {
        _a[flynn1].flynnNodes(flynn1, &ids, &num_elle_nodes);
        for (i=0;i<num_elle_nodes;i++) {
            node = ids[i];
            if (ElleNodeIsTriple(node) &&
                          ElleFindBndIndex(node,flynn2)!=NO_NB) {
                if (start==NO_NB) start=node;
                else if (finish==NO_NB) finish=node;
                cnt++;
            }
        }
        free(ids);
        // cnt should be 2 if valid for split
        // cnt of 4 may indicate split wrapping grain which should
        //   not be merged
        if (cnt!=2) {
            err=1;
            if (cnt==4) cout << "merge will cause wrapping flynn" << endl;
            else if (cnt) cout << "invalid TJ count " << cnt << endl;
        }
        else {
        // check attributes
        // area average float ones - don't merge if int ones differ
            if (attributesAllowMerge(flynn1,flynn2)) {
                RemoveBoundary(&start,&finish,flynn1,flynn2);
                removeShrinkingFlynn(flynn2);
    /*********** temporary */
                /*_a[flynn2].clean();*/
            }
            else err=1;
        }
    }
    return(err);
}

int FlynnArray::mergeFlynnsNoCheck(int flynn1, int flynn2)
{
    int *ids, i, num_elle_nodes, cnt=0, start=NO_NB, finish=NO_NB;
    int node, err=0;

    // check that these are lowest level flynns
    // check that flynn1 != flynn2
    if (flynn1==flynn2 || _a[flynn1].isParent() || _a[flynn2].isParent())
        err=1;
    else {
        _a[flynn1].flynnNodes(flynn1, &ids, &num_elle_nodes);
        for (i=0;i<num_elle_nodes;i++) {
            node = ids[i];
            if (ElleNodeIsTriple(node) &&
                          ElleFindBndIndex(node,flynn2)!=NO_NB) {
                if (start==NO_NB) start=node;
                else if (finish==NO_NB) finish=node;
                cnt++;
            }
        }
        free(ids);
        // cnt should be 2 if valid for split
        // cnt of 4 may indicate split wrapping grain which should
        //   not be merged
        if (cnt!=2) {
            err=1;
            if (cnt==4) cout << "merge will cause wrapping flynn" << endl;
            else if (cnt) cout << "invalid TJ count " << cnt << endl;
        }
        else {
            RemoveBoundary(&start,&finish,flynn1,flynn2);
            removeShrinkingFlynn(flynn2);
    /*********** temporary */
            /*_a[flynn2].clean();*/
        }
    }
    return(err);
}

//---------------------------------------------------------------------------
//Function attributesAllowMerge
/*!
    \brief      checks that the attributes of the 2 flynns allow merging
    \param      flynn1, flynn2 - active flynns
    \warning    only checks integer attributes
    \return     \a true if integer attributes are equal
                \a else false
*/
bool FlynnArray::attributesAllowMerge(const int flynn1,const int flynn2)
{
    int ival1, ival2;
    bool allowed=true;
    int err=0;
    int i, *active=0, maxa=0;
    
    active=0;
    if ((maxa = numDfltAttributes()) > 0) {
        if ((active = new int[maxa])==0)
            OnError("ElleFlynnDfltAttributeList",MALLOC_ERR);
        getDfltAttributeList(active,maxa);
    }
    for (i=0; allowed && i<maxa; i++) {
        Attribute x(active[i],0);
        if (x.isIntAttribute()) {
            err = getFlynnAttribute(flynn1,active[i],&ival1);
            if (err==ATTRIBID_ERR)
                err = getDfltAttribute(active[i],&ival1);
            if (!err) {
                err = getFlynnAttribute(flynn2,active[i],&ival2);
                if (err==ATTRIBID_ERR)
                    err = getDfltAttribute(active[i],&ival2);
            }
            if (err) OnError("attributesAllowMerge",err);
            allowed = (ival1==ival2);
        }
    }
    if (active) delete[] active;
    return(allowed);
}

int FlynnArray::splitFlynn(const int type,int flynnindex,
                           int *child1,int *child2,
                           float dx, float dy)
{
    int *ids, num, i, old, num_elle_nodes, first_try;
    int start, end, node, nb1, newid, next, last;
    int f1, f2;
    int walks=0, err=0;
    double area;
    Coords xy, current, prev;
    ERegion rgn1, rgn2, oldrgn;
    struct triangulateio out;
    struct flagvals flags;
    vector<PointData> boundary;

    _a[flynnindex].flynnNodes(flynnindex, &ids, &num_elle_nodes);
    // Do not try to split flynn with <3 triple Bnodes
    for (i=0,num=0;i<num_elle_nodes;i++)
        if (ElleNodeIsTriple(ids[i])) num++;
    if (num<=3) return(1);
    first_try = start = i = ElleRandom()%num_elle_nodes;

    //
    // start split at a DJ (add one if necessary)
    //
    if (!ElleNodeIsDouble(ids[i])) {
        i=(i+1)%num_elle_nodes;
        while (i!=first_try && !ElleNodeIsDouble(ids[i]))
            i=(i+1)%num_elle_nodes;
        if (i!=first_try) {
            first_try = start = i;
        }
        else {
            ElleInsertDoubleJ(ids[first_try],
                       ids[(first_try+1)%num_elle_nodes],&newid,0.5);
            free(ids);
            old= num_elle_nodes;
            _a[flynnindex].flynnNodes(flynnindex, &ids, &num_elle_nodes);
            if (num_elle_nodes != old+1)
                OnError("splitFlynn - nodecount",0);
            first_try = start = (first_try+1)%num_elle_nodes;
        }
    }

    /*
     * triangulate the flynn
     */
    int *elle_id = new int[num_elle_nodes*2];
    for (i=0;i<num_elle_nodes*2;i++) elle_id[i] = NO_NB;
    double *xvals = new double[num_elle_nodes*2];
    double *yvals = new double[num_elle_nodes*2];
    ElleNodePosition(ids[num_elle_nodes-1],&current);
    for (i=1,num=0; i<=num_elle_nodes; i++) {
        node=ids[i%num_elle_nodes];
        prev = current;
        ElleNodePlotXY(node,&current,&prev);
        if (ElleNodeIsTriple(ids[(i-1)%num_elle_nodes]) &&
                                   ElleNodeIsTriple(node)) {
            xy.x = current.x - (current.x-prev.x)*0.5;
            xy.y = current.y - (current.y-prev.y)*0.5;
            xvals[num] = xy.x;
            yvals[num] = xy.y;
            num++;
        }
        xvals[num] = current.x;
        yvals[num] = current.y;
        elle_id[num] = node;
        num++;
    }
    /*flags.area = ElleminNodeSep() * ElleminNodeSep();*/
    flags.area = ElleminNodeSep() * ElleminNodeSep() * 0.5;
    flags.quality = 10;
    flags.midptnodes = 0;
    flags.bndpts = 0;
    flags.edges = 0;
    flags.voronoi = 0;
    if (tripolypts(&out, &flags, num, xvals, yvals, 0, num, 0))
        OnError("splitFlynn - error triangulating",0);

    /*
     * Walk across the triangulation
     */
    switch(type) {
    case RANDOM:
        while (RandomSplit(&out, boundary, ids, elle_id, num, ids[start])) {
            walks++;
            if (walks>9) {
                i = (start+1)%num_elle_nodes;
                while (i!=first_try && !ElleNodeIsDouble(ids[i]))
                         i=(i+1)%num_elle_nodes;
                if (i!=first_try) {
                    start = i;
                    walks=0;
                }
                else {
                    WriteTriangulation(&out,"error.poly");
                    return(10);
                /*OnError("Failed 10 random walks",0);*/
                }
            }
        }
        break;
    case DIRECT:
        area = ElleRegionArea(flynnindex);
        while (DirectSplit(&out, boundary, area, elle_id, num, ids[start],
                               dx,dy)) {
            walks++;
            if (walks>9) {
                i = (start+1)%num_elle_nodes;
                while (i!=first_try && !ElleNodeIsDouble(ids[i]))
                         i=(i+1)%num_elle_nodes;
                if (i!=first_try) {
                    start = i;
                    walks=0;
                }
                else {
                    WriteTriangulation(&out,"error.poly");
                    return(10);
                /*OnError("Failed 10 random walks",0);*/
                }
            }
        }
        break;
    default: break;
    }

    /*
     * create 2 new children
     */
    f1 = spareFlynn(); // first -> end
    f2 = spareFlynn(); // end -> first
    _a[flynnindex].addChild(f1);
    _a[flynnindex].addChild(f2);
    _a[f1].setParent(flynnindex);
    _a[f2].setParent(flynnindex);
    rgn1 = f1;
    rgn2 = f2;
    oldrgn = flynnindex;

    CreateBoundary(boundary,f1,f2,elle_id,num,&end);

#if XY
    ResetRegionEntries(end,&end,ids[start],flynnindex,rgn1);
#endif
    next = old = end;
    last = ids[start];
    _a[f1].setFirstNode(last);
    do {
        if ((nb1=ElleFindBndNode(next,old,flynnindex))==NO_NB)
            OnError("splitFlynn",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnError("splitFlynn nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,rgn1);
        old = next;
        next = nb1;
    } while (next!=last);
    if ((old=ElleFindBndNode(next,old,rgn1))==NO_NB)
        OnError("splitFlynn",NONB_ERR);
#if XY
    ResetRegionEntries(last,&old,end,flynnindex,rgn1);
#endif
    last = end;
    _a[f2].setFirstNode(last);
    do {
        if ((nb1=ElleFindBndNode(next,old,flynnindex))==NO_NB)
            OnError("splitFlynn",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnError("splitFlynn nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,rgn2);
        old = next;
        next = nb1;
    } while (next!=last);

    *child1 = f1;
    *child2 = f2;
    delete [] xvals;
    delete [] yvals;
    delete [] elle_id;
    if (ids) free(ids);
    return(err);
}

int FlynnArray::splitWrappingFlynn(int flynnindex,
                                   int node1, int node2,
                                   int *child1,int *child2)
{
    unsigned char done=0;
    int *ids, num, i, old, num_elle_nodes;
    int start, end, nb1, newid, next, last, node;
    int splitstart, splitend, num1to2, num2to1, startindex, endindex;
    int f1, f2;
    int walks=0;
    double dx,dy;
    Coords xy, current, prev;
    ERegion rgn1, rgn2, oldrgn;
    struct triangulateio out;
    struct flagvals flags;
    vector<PointData> boundary;

    _a[flynnindex].flynnNodes(flynnindex, &ids, &num_elle_nodes);

    //
    // start split on shortest segment joining node1 and node2
    // start split at a DJ (add one if necessary)
    //
    i=0;
    while (i<num_elle_nodes && ids[i]!=node1)
        i=(i+1)%num_elle_nodes;
    start = i;
    i=(i+1)%num_elle_nodes;
    num1to2=0;
    while (i<num_elle_nodes && ids[i]!=node2) {
        i=(i+1)%num_elle_nodes;
        num1to2++;
    }
    end = i;
    num2to1 = num_elle_nodes - 2 - num1to2;
    if (num2to1 > num1to2) {
        splitstart = node1;
        startindex = start;
        splitend = node2;
        endindex = end;
    }
    else {
        splitstart = node2;
        startindex = end;
        splitend = node1;
        endindex = start;
    }
        
    next = i = (startindex+1)%num_elle_nodes; start = NO_NB;
    while (i!=endindex && !ElleNodeIsDouble(ids[i]))
        i=(i+1)%num_elle_nodes;
    if (i!=endindex) start = i;
    else {
        ElleInsertDoubleJ(ids[startindex],ids[next],&newid,0.5);
        free(ids);
        old= num_elle_nodes;
        _a[flynnindex].flynnNodes(flynnindex, &ids, &num_elle_nodes);
        if (num_elle_nodes != old+1)
            OnError("splitFlynn - nodecount",0);
        start = next;
    }

    /*
     * triangulate the flynn
     */
    int *elle_id = new int[num_elle_nodes*2];
    for (i=0;i<num_elle_nodes*2;i++) elle_id[i] = NO_NB;
    double *xvals = new double[num_elle_nodes*2];
    double *yvals = new double[num_elle_nodes*2];
    ElleNodePosition(ids[num_elle_nodes-1],&current);
    for (i=1,num=0; i<num_elle_nodes; i++) {
        node=ids[i%num_elle_nodes];
        prev = current;
        ElleNodePlotXY(node,&current,&prev);
        if (ElleNodeIsTriple(ids[(i-1)%num_elle_nodes]) &&
                                   ElleNodeIsTriple(node)) {
            xy.x = current.x - (current.x-prev.x)*0.5;
            xy.y = current.y - (current.y-prev.y)*0.5;
            xvals[num] = xy.x;
            yvals[num] = xy.y;
            num++;
        }
        xvals[num] = current.x;
        yvals[num] = current.y;
        elle_id[num] = node;
        num++;
    }
    /*flags.area = ElleminNodeSep() * ElleminNodeSep() * 2;*/
    flags.area = ElleminNodeSep() * ElleminNodeSep() * 0.5;
    flags.quality = 10;
    flags.midptnodes = 0;
    flags.bndpts = 0;
    flags.edges = 0;
    flags.voronoi = 0;
    if (tripolypts(&out, &flags, num, xvals, yvals, 0, num, 0))
        OnError("splitFlynn - error triangulating",0);

    /*
     * Walk across the triangulation
     */
    for (i=0; i<num; i++) {
        if (elle_id[i]==node1) {current.x = xvals[i]; current.y=yvals[i]; }
        if (elle_id[i]==node2) {prev.x = xvals[i]; prev.y=yvals[i]; }
    }
    dx = prev.x-current.x;
    dy = prev.y-current.y;
    while (!done && walks<10) {
        if (RandomSplit(&out, boundary, ids, elle_id, num,ids[start])==0) {
            /* does the new boundary end within the right segment */
            i=0;
            while (elle_id[i]!=splitend) i = (i+1)%num;
            i = (i+1)%num;
            int j=0;
            while (j<num && elle_id[i]!=splitstart) {
                endindex = boundary[boundary.size()-1].idvalue();
                if (i == endindex) done=1;
                i = (i+1)%num;
                j++;
            }
        }
        walks++;
    }
    if (walks==10) {
        WriteTriangulation(&out,"error.poly");
        return(1);
    }

    /*
     * create 2 new children
     */
    f1 = spareFlynn(); // first -> end
    f2 = spareFlynn(); // end -> first
    _a[flynnindex].addChild(f1);
    _a[flynnindex].addChild(f2);
    _a[f1].setParent(flynnindex);
    _a[f2].setParent(flynnindex);
    rgn1 = f1;
    rgn2 = f2;
    oldrgn = flynnindex;

    CreateBoundary(boundary,f1,f2,elle_id,num,&end);

    next = old = end;
    last = ids[start];
    _a[f1].setFirstNode(last);
    do {
        if ((nb1=ElleFindBndNode(next,old,flynnindex))==NO_NB)
            OnError("splitFlynn",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnError("splitFlynn nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,rgn1);
        old = next;
        next = nb1;
    } while (next!=last);
    /*old = next;*/
    if ((old=ElleFindBndNode(next,old,rgn1))==NO_NB)
        OnError("splitFlynn",NONB_ERR);
    last = end;
    _a[f2].setFirstNode(last);
    do {
        if ((nb1=ElleFindBndNode(next,old,flynnindex))==NO_NB)
            OnError("splitFlynn",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnError("splitFlynn nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,rgn2);
        old = next;
        next = nb1;
    } while (next!=last);

    *child1 = f1;
    *child2 = f2;
    delete [] xvals;
    delete [] yvals;
    delete [] elle_id;
    if (ids) free(ids);
    return(0);
}

int FlynnArray::directSplitWrappingFlynn(int flynnindex,
                                   int node1, int node2,
                                   int *child1,int *child2,
                                   Coords *dir)
{
    unsigned char done=0;
    int *ids, num, i, j, old, num_elle_nodes;
    int start, end, nb1, newid, next, last;
    int splitstart, splitend, num1to2, num2to1, startindex, endindex;
    int f1, f2;
    int walks=0, nodes_added=0;
    int *elle_id;
    double *xvals=0, *yvals=0;
    Coords xy, current, prev;
    ERegion rgn1, rgn2, oldrgn;
    struct triangulateio out;
    struct flagvals flags;
    vector<PointData> boundary;

    _a[flynnindex].flynnNodes(flynnindex, &ids, &num_elle_nodes);

    //
    // start split at a DJ (add one if necessary)
    //
    i=0;
    while (i<num_elle_nodes && ids[i]!=node1)
        i=(i+1)%num_elle_nodes;
    start = startindex = i;
    while (i<num_elle_nodes && ids[i]!=node2)
        i=(i+1)%num_elle_nodes;
    end = endindex = i;
        
/*
    next = i = (startindex+1)%num_elle_nodes; start = NO_NB;
    while (i!=endindex && !ElleNodeIsDouble(ids[i]))
        i=(i+1)%num_elle_nodes;
    if (i!=endindex) start = i;
    else {
*/
    if (!ElleNodeIsDouble(ids[startindex])) {
        next = (startindex+1)%num_elle_nodes;
        ElleInsertDoubleJ(ids[startindex],ids[next],&newid,0.5);
        nodes_added++;
    }
    if (!ElleNodeIsDouble(ids[endindex])) {
        next = (endindex+1)%num_elle_nodes;
        ElleInsertDoubleJ(ids[endindex],ids[next],&newid,0.5);
        nodes_added++;
    }
    if (nodes_added>0) {
        free(ids);
        old= num_elle_nodes;
        _a[flynnindex].flynnNodes(flynnindex, &ids, &num_elle_nodes);
        if (num_elle_nodes != old+1)
            OnError("splitFlynn - nodecount",0);
        i=0;
        while (i<num_elle_nodes && ids[i]!=node1)
            i=(i+1)%num_elle_nodes;
        start = startindex = i;
        while (i<num_elle_nodes && ids[i]!=node2)
            i=(i+1)%num_elle_nodes;
        end = endindex = i;
    }
    splitstart = ids[start];
    splitend = ids[end];
    double sep;
    sep=EllemaxNodeSep();
    if ((sep=ElleNodeSeparation(splitstart,splitend))>EllemaxNodeSep()) {
    /*
     * triangulate the flynn
     */
    elle_id = new int[num_elle_nodes*2];
    for (i=0;i<num_elle_nodes*2;i++) elle_id[i] = NO_NB;
    xvals = new double[num_elle_nodes*2];
    yvals = new double[num_elle_nodes*2];
    ElleNodePosition(ids[num_elle_nodes-1],&current);
    for (i=0,num=0; i<num_elle_nodes; i++) {
        prev = current;
        ElleNodePlotXY(ids[i],&current,&prev);
        if (i==0) j=num_elle_nodes-1;
        else j=i-1;
        if (ElleNodeIsTriple(ids[j]) &&
                                   ElleNodeIsTriple(ids[i])) {
            xy.x = current.x - (current.x-prev.x)*0.5;
            xy.y = current.y - (current.y-prev.y)*0.5;
            xvals[num] = xy.x;
            yvals[num] = xy.y;
            num++;
        }
        xvals[num] = current.x;
        yvals[num] = current.y;
        elle_id[num] = ids[i];
        num++;
    }
    flags.area = ElleminNodeSep() * ElleminNodeSep() * 0.5;
    flags.quality = 10;
    flags.midptnodes = 0;
    flags.bndpts = 0;
    flags.edges = 0;
    flags.voronoi = 0;
    if (tripolypts(&out, &flags, num, xvals, yvals, 0, num, 0))
        OnError("splitFlynn - error triangulating",0);

    /*
     * Walk across the triangulation
     */
    for (i=0; i<num; i++) {
        if (elle_id[i]==node1) {current.x = xvals[i]; current.y=yvals[i]; }
        if (elle_id[i]==node2) {prev.x = xvals[i]; prev.y=yvals[i]; }
    }
    while (!done && walks<10) {
        // should be directed from node1 towards node2 
        if (DirectSplit(&out, boundary, ElleRegionArea(flynnindex),
                        elle_id, num, ids[start],
                        (float)dir->x,(float)dir->y)==0) {
            done=1;
            /* does the new boundary end within the right segment
            i=0;
            while (elle_id[i]!=splitend) i = (i+1)%num;
            i = (i+1)%num;
            while (elle_id[i]!=splitstart) {
                endindex = boundary[boundary.size()-1].idvalue();
                if (i == endindex) done=1;
                i = (i+1)%num;
            } */
        }
        walks++;
    }
    if (walks==10) {
        WriteTriangulation(&out,"error.poly");
        return(1);
    }
    }
    else {
    /*
     * Put the  nodes in the boundary list
     */
    ElleNodePosition(splitstart,&prev);
    PointData bndnode(prev.x,prev.y,splitstart,1);
    boundary.push_back(bndnode);
    ElleNodePlotXY(splitend,&xy,&prev);
    bndnode.setvalues(xy.x,xy.y,splitend,1);
    boundary.push_back(bndnode);
    }

    /*
     * create 2 new children
     */
    f1 = spareFlynn(); // first -> end
    f2 = spareFlynn(); // end -> first
    _a[flynnindex].addChild(f1);
    _a[flynnindex].addChild(f2);
    _a[f1].setParent(flynnindex);
    _a[f2].setParent(flynnindex);
    rgn1 = f1;
    rgn2 = f2;
    oldrgn = flynnindex;

    CreateBoundary(boundary,f1,f2,elle_id,num,&end);

    next = old = end;
    last = ids[start];
    _a[f1].setFirstNode(last);
    do {
        if ((nb1=ElleFindBndNode(next,old,flynnindex))==NO_NB)
            OnError("splitFlynn",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnError("splitFlynn nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,rgn1);
        old = next;
        next = nb1;
    } while (next!=last);
    /*old = next;*/
    if ((old=ElleFindBndNode(next,old,rgn1))==NO_NB)
        OnError("splitFlynn",NONB_ERR);
    last = end;
    _a[f2].setFirstNode(last);
    do {
        if ((nb1=ElleFindBndNode(next,old,flynnindex))==NO_NB)
            OnError("splitFlynn",NONB_ERR);
        if ((i = ElleFindNbIndex(nb1,next))==NO_NB)
            OnError("splitFlynn nb index",NONB_ERR);
        ElleSetRegionEntry(next,i,rgn2);
        old = next;
        next = nb1;
    } while (next!=last);

    *child1 = f1;
    *child2 = f2;
    delete [] xvals;
    delete [] yvals;
    delete [] elle_id;
    if (ids) free(ids);
    return(0);
}

int FlynnArray::flynnNodePositions(int flynnindex,
                                   vector<Coords> &rel_pos,
                                   Coords *ref)
{
    int num, *ids, i, j, start=0;
    double min, dist;
    Coords xy,prev;

    _a[flynnindex].flynnNodes(flynnindex, &ids, &num);
    if (!ref) ElleNodePosition(ids[0],&prev);
    else prev = *ref;
    /*
     * find the flynn node that is closest to the test point
     * otherwise flynn may be located on other side of unit cell
     */
    ElleNodePosition(ids[0],&prev);
    min = (prev.x-ref->x)*(prev.x-ref->x) + (prev.y-ref->y)*(prev.y-ref->y);
    for (j=1;j<num;j++) {
        ElleNodePosition(ids[j],&prev);
        dist = (prev.x-ref->x)*(prev.x-ref->x) +
                      (prev.y-ref->y)*(prev.y-ref->y);
        if (dist<min) {
            min = dist;
            start = j;
        }
    }
    for (j=0,i=start;j<num;j++,i=(i+1)%num) {
        ElleNodePlotXY(ids[i],&xy,&prev);
        rel_pos.push_back(xy);
        prev=xy;
    }
    if (ids) free(ids);
    return(0);
}

int FlynnArray::writePolyFile(int flynnindex, const char *fname)
{
    int num, *ids, i;
    Coords xy,prev;
    ofstream outfile(fname);

    if (!outfile) OnError("writePolyFile",OPEN_ERR);
    _a[flynnindex].flynnNodes(flynnindex, &ids, &num);
    outfile << num << " 2" << " 0" << " 1" << endl;
    ElleNodePosition(ids[0],&prev);
    for (i=0;i<num;i++) {
        ElleNodePlotXY(ids[i],&xy,&prev);
        outfile << i << ' ' << xy.x << ' ' << xy.y << ' ' << ids[i] << endl;
        prev=xy;
    }
    outfile << num << ' ' << '1' << endl;
    for (i=0;i<num;i++)
        outfile << i << ' ' << i << ' ' << (i+1)%num << ' ' << "1" << endl;
    outfile << "0" << endl;
    outfile << "0" << endl;
    if (ids) free(ids);
    return(0);
}

int FlynnArray::writeUnodePolyFile(int flynnindex, const int attrib,
                                   const char *fname)
{
    int num, *ids, i;
    double val=0.0;
    Coords xy,prev;
    vector<int> unode_ids(0);
    ofstream outfile(fname);

    if (!outfile) OnError("writePolyFile",OPEN_ERR);
    _a[flynnindex].flynnNodes(flynnindex, &ids, &num);
    _a[flynnindex].getUnodeList(unode_ids);
    outfile << num+unode_ids.size() << " 2" << " 0" << " 1" << endl;
    /*
     * write boundary node positions
     */
    ElleNodePosition(ids[0],&prev);
    for (i=0;i<num;i++) {
        ElleNodePlotXY(ids[i],&xy,&prev);
        outfile << i << ' ' << xy.x << ' ' << xy.y << ' ' << "1" << endl;
        prev=xy;
    }
    /*
     * write unconnected node positions
     */
    for (vector<int>::iterator it=unode_ids.begin();
                                       it!=unode_ids.end(); it++,i++) {
        if (attrib!=NO_VAL && UnodeAttributeActive(attrib)) {
            ElleGetUnodePolyInfo(*it,attrib,&xy,&val);
            outfile << i << ' ' << xy.x << ' ' << xy.y << ' ' 
                << val << ' ' << "0" << endl;
        }
        else {
            ElleGetUnodePosition(*it,&xy);
            outfile << i << ' ' << xy.x << ' ' << xy.y << ' ' 
                << *it << ' ' << "0" << endl;
        }
    }
    /*
     * write segment info (boundaries)
     */
    outfile << num << ' ' << '1' << endl;
    for (i=0;i<num;i++)
        outfile << i << ' ' << i << ' ' << (i+1)%num << ' ' << "1" << endl;
    outfile << "0" << endl;
    outfile << "0" << endl;
    if (ids) free(ids);
    return(0);
}

std::ostream & operator<< (std::ostream &os, FlynnArray &t)
{
    int num_nodes, *nodes, i, j;
    int max;

    max = t._a.size();
    for (i=0; i<max; i++)
        if (t._a[i].getState()==ACTIVE) {
            os << i << ' ';
            t._a[i].flynnNodes(i,&nodes,&num_nodes);
            for (j=0; j<num_nodes; j++)
                os << nodes[j] << ' ';
            os << endl;
            if (nodes) free(nodes);
        }
    return os;
}
