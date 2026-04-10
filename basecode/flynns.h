 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_flynns_h
#define _E_flynns_h
#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include "attrib.h"
#include "nodes.h"
#include "attribarray.h"

// flynn class

class Flynn {
    bool state,color;
    int first_node,red,green,blue;
    int parent;
	std::list<int> child_list;
	std::list<int> unode_list;
	std::list<int>::iterator it;
    AttributeArray attributes;
/*
    std::vector<int> _vedges;
    std::vector<Coords> _vpoints;
	std::vector<Coords>::iterator itvc;
*/
public:
    Flynn():state(INACTIVE),first_node(NO_NB),parent(NO_NB),color(false) { }
    Flynn(bool stateval,int first,int p) :
         state(stateval),first_node(first),parent(p) { }
    void setState(bool stateval) { state = stateval; }
    bool getState() const { return state; }
    void setFirstNode(int first) { first_node = first; }
    /*int getFirstNode();*/
    int getFirstNode() { return first_node; }
    /*int getFirstNode() { int first;*/
                         /*if (!isParent()) first=first_node;*/
                         /*else first=findFirstNode();*/
                         /*return(first); }*/
    int findFirstNode(std::list<int> &LClist);
    void getColor(bool *c,int *r,int *g,int *b)
        { if(color) {*c=color;*r=red;*g=green;*b=blue;}
         else{*c=color;*r=-1;*g=-1;*b=-1;} }
    void setColor(bool c,int r,int g, int b)
        { color=c;red=r;green=g,blue=b; }
    void setParent(int parentval) { parent = parentval; }
    int getParent() const { return parent; }
    bool hasParent() const { return(parent != NO_NB); }
    bool isParent() const { return(child_list.size() > 0); }
    void addChild(int child);
    void getChildList(std::vector<int> &l) { it=child_list.begin();
                                        while(it!=child_list.end()) {
                                            l.push_back(*it);
                                            it++;
                                        }
                                      }
    void removeChild(int child);
    void removeChildList() { 
             child_list.erase(child_list.begin(),child_list.end()); }
    bool hasFlynnAttribute(const int id) {
             return(attributes.hasAttribute(id)); }
    void removeFlynnAttribute(const int id) { 
             attributes.removeAttribute(id); }
    int numAttributes() { return(attributes.numActive()); }
    void getAttributeList(int *attr, const int maxa);
    void getIntAttributeList(std::vector<int> &attr);
    int setFlynnAttribute(const int id, int val) {
             attributes.setAttribute(id,val); return(0); }
    int setFlynnAttribute(const int id, double val) {
             attributes.setAttribute(id,val); return(0); }
    int setFlynnAttribute(const int id, Coords_3D *val) {
             attributes.setAttribute(id,val); return(0); }
    int getFlynnAttribute(const int id, int *val) {
             return(attributes.getAttribute(id,val)); }
    int getFlynnAttribute(const int id, double *val) {
             return(attributes.getAttribute(id,val)); }
    int getFlynnAttribute(const int id, Coords_3D *val) {
             return(attributes.getAttribute(id,val)); }
    void flynnNodes(int flynnid, int **id, int *num);
    void flynnNodes(int flynnid, std::vector<int> &ids);
    int flynnNodeCount(int flynnid);
    void clean();
    void addUnode(int id) { unode_list.push_back(id); }
    void getUnodeList(std::vector<int> &l) { it=unode_list.begin();
                                        while(it!=unode_list.end()) {
                                            l.push_back(*it);
                                            it++;
                                        }
                                       }
    int removeUnode(int id);
    void removeUnodeList() { 
             unode_list.erase(unode_list.begin(),unode_list.end()); }
/*
    int addVoronoiPt(const Coords pt);
    void voronoiPts(std::list<Coords> &l)
             { for (itvc=_vpoints.begin(); itvc!= _vpoints.end();itvc++)
                  l.push_back(*itvc); }
    void removeVoronoiPts() { _vpoints.clear(); }
*/
    const Flynn & operator=(const Flynn &t) {
                           state = t.state;
                           first_node = t.first_node;
                           parent = t.parent;
                           child_list = t.child_list;
                           unode_list = t.unode_list;
                           it = t.it;
                           attributes = t.attributes;
                           return t; }
    friend std::ostream & operator<< (std::ostream &, const Flynn &);
};

#endif
