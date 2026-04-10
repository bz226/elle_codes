 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_unodesP_h
#define _E_unodesP_h
#include <iostream>
#include <list>
#include <vector>
#include <set>
#include <algorithm>
#include "attrib.h"
#include "attribarray.h"

typedef std::set<int,std::less<int> > set_int;

// unconnected node class

// Coords should be class
class Unode {
    int _id;
    int _flynn;
    Coords _pos;
    double _area;
    AttributeArray _attributes;
    std::vector<int> _vedges;
    std::list<int> _vpoints;
    set_int _near_nbs;
    
public:
    Unode():_id(0),_flynn(NO_VAL),_area(0),_attributes(0),_vpoints(),_near_nbs() { }
    Unode(int idval,Coords *xy):_id(idval), _pos(*xy),
                                _area(0),_attributes(0),_vpoints(),_near_nbs() { }
    Unode(int idval,int flynn,Coords *xy):_id(idval),_flynn(flynn),_pos(*xy),
                                   _area(0),_attributes(0),_vpoints(),_near_nbs() { }
    int id() const { return(_id); }
    int flynn() const { return(_flynn); }
    void setFlynn(const int f_id) { _flynn=f_id; }
    double area() const { return(_area); }
    void setArea(const double a) { _area=a; }
    void setPosition(const Coords *xy) { _pos.x=xy->x; _pos.y=xy->y; }
    void getPosition(Coords *xy) { xy->x=_pos.x; xy->y=_pos.y; }
    bool hasAttribute(const int id) {
             return(_attributes.hasAttribute(id)); }
    void removeAttribute(const int id) { 
             _attributes.removeAttribute(id); }
    int numAttributes() { return(_attributes.numActive()); }
	std::vector<double> *getAttributeValues() {
             return(_attributes.getAttributeValues());
             }
    void getList(int *ids, int max) { _attributes.getList(ids,max); }
    int setAttribute(const int id, int val) {
             _attributes.setAttribute(id,val); return(0); }
    int setAttribute(const int id, double val) {
             _attributes.setAttribute(id,val); return(0); }
    int setAttribute(const int id, Coords_3D *val) {
             _attributes.setAttribute(id,val); return(0); }
    int getAttribute(const int id, int *val) {
             return(_attributes.getAttribute(id,val)); }
    int getAttribute(const int id, double *val) {
             return(_attributes.getAttribute(id,val)); }
    int getAttribute(const int id, Coords_3D *val) {
             return(_attributes.getAttribute(id,val)); }
    void addVoronoiPt(const int index) {
             std::list<int>::iterator it;
             if ((it=find(_vpoints.begin(),_vpoints.end(),index))==
                                                       _vpoints.end())
                   _vpoints.push_back(index); }
    // catch postNb not in list?
    void insertVoronoiPt(const int postNb,const int index) {
             std::list<int>::iterator it;
             if ((it=find(_vpoints.begin(),_vpoints.end(),postNb))!=
                                                       _vpoints.end())
                   _vpoints.insert(it,index); }
    void voronoiPts(std::list<int> &l)
             { for (std::list<int>::iterator it=_vpoints.begin();
                 it!= _vpoints.end();it++)
                  l.push_back(*it); }
    void deleteVoronoiPt(const int index)
             { std::list<int>::iterator it;
               if ((it=find(_vpoints.begin(),_vpoints.end(),index))!=
                                                       _vpoints.end())
                   _vpoints.erase(it); }
    void addNearNb(const int nb)
             { _near_nbs.insert(nb); }
    void deleteNearNb(const int nb)
             { _near_nbs.erase(nb); }
    int numberOfVoronoiPts()
             { return(_vpoints.size()); }
    int numberOfNearNbs()
             { return(_near_nbs.size()); }
    void getNearNbsList(std::vector< int > &nbs)
             {  set_int::iterator it;
                for( it=_near_nbs.begin();it!=_near_nbs.end();it++)
					nbs.push_back(*it);
             }
    void writeNearNbList(std::ostream &os) {
                set_int::iterator it;
                for( it=_near_nbs.begin();it!=_near_nbs.end();it++)
					os << *it << ' ';
                os << std::endl;
             }
    const Unode & operator=(const Unode &t) {
                           _id = t._id;
                           _pos.x = t._pos.x;
                           _pos.y = t._pos.y;
                           _area = t._area;
                           _attributes = t._attributes;
                           _vpoints = t._vpoints;
                           _near_nbs = t._near_nbs;
                           return t; }
    friend std::ostream & operator<< (std::ostream &os, /*const*/ Unode &t) {
                os << t.id() << " ";
                os << t._pos.x << " ";
                os << t._pos.y << " ";
/*
                os.setf(ios::scientific,ios::floatfield);
				std::vector<double> *attr_vals =
                    t._attributes.getAttributeValues();
                copy(attr_vals->begin(),attr_vals->end(),
				std::ostream_iterator<double>(os," "));
                os.setf(ios::fixed,ios::floatfield);
                delete attr_vals;
*/
                return os;
            }
};

#endif
