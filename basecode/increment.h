 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_increment_h
#define _E_increment_h
#include <iostream>
#include <vector>
#include <algorithm>
#include "attrib.h"
#include "unodes.h"

// increment class

class Increment {
private:
    double _area;
    double _gb_area;
    double _gb_mass_i;
    double _gb_mass_f;
    double _swept_mass_i;
    double _swept_mass_f;
    Coords _mid_pt;

    void clear() { _area = _gb_area = _gb_mass_i = _gb_mass_f = 
                   _swept_mass_i = _swept_mass_f = 0.0;
                   _mid_pt.x = _mid_pt.y = 0.0;
                 }
    void set_up
public:
    Increment():_mid_pt.x=_mid_pt.y=0.0 { }
    void setState(bool stateval) { state = stateval; }
    bool getState() const { return state; }
    void setFirstNode(int first) { first_node = first; }
    /*int getFirstNode();*/
    int getFirstNode() { return first_node; }
    /*int getFirstNode() { int first;*/
                         /*if (!isParent()) first=first_node;*/
                         /*else first=findFirstNode();*/
                         /*return(first); }*/
    int findFirstNode(list<int> &LClist);
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
