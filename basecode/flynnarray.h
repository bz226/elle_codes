 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_flynnarray_h
#define _E_flynnarray_h
#include <iostream>
#include <vector>
#include <list>
#include "attrib.h"
#include "attribarray.h"
#include "flynns.h"

const int INIT_UNIT_NUM = 512;

class FlynnArray {
	std::vector<Flynn> _a;
	std::vector<Flynn> :: iterator it_a;
    AttributeArray _dfltAttributes;
public:
    FlynnArray(): _a(INIT_UNIT_NUM),_dfltAttributes(0) { }
    FlynnArray(int size): _a(size) { }
    void setFlynnActive(int index);
    void setFlynnInactive(int index);
    bool isFlynnActive(int index);
    int spareFlynn();
    void getColor(int index,bool *c,int *r,int *g,int *b);
    void setColor(int index,bool c,int r,int g, int b);
    int maxLevels();
    int getFlynnLevel(const int index);
    void setFlynnFirstnode(int index, int first);
    int getFlynnFirstnode(int index);
    double flynnArea(const int index);
    int makeLClist(int index, std::list<int> &LClist);
    void addFlynnChild(int index,int child);
    void printFlynn(int index);
    void flynnNodes(int index, int **ids, int *num) {
        _a[index].flynnNodes(index,ids,num); }
    void flynnNodes(int index, std::vector <int> &ids) {
        _a[index].flynnNodes(index,ids); }
    int flynnNodePositions(int flynnindex, std::vector<Coords> &rel_pos,
                           Coords *ref);
    int flynnNodeCount(int index) {
        return(_a[index].flynnNodeCount(index)); }
    /*int addVoronoiPt(int index, Coords pt) {*/
        /*return(_a[index].addVoronoiPt(pt)); }*/
    bool isChildOf(int flynn1, int flynn2);
    bool isParent(int flynn1) { return(_a[flynn1].isParent()); }
    bool hasParent(int flynn1) { return(_a[flynn1].hasParent()); }
    bool getParent(int flynn1) { return(_a[flynn1].getParent()); }
    void removeFlynn(int index);
    void removeShrinkingFlynn(int index);
    void promoteFlynn(int index);
    void clean();
    void newFlynn(int start, int end, int oldindex, int *newindex);
    void updateFirstNodes(int old, int nn, int flynnindex);
    void neighbourRegions(const int flynnid, std::list<int> &nbflynns);
    bool hasFlynnAttribute(const int index, const int id);
    bool hasAttribute(const int id) {
         return(_dfltAttributes.hasAttribute(id)); }
    int numDfltAttributes() { return(_dfltAttributes.numAttributes()); }
    int numFlynnAttributes(const int flynnindex);
    void initAttribute(const int id) {
         _dfltAttributes.initAttribute(id); }
    void removeFlynnAttribute(const int index, const int id);
    void removeDfltAttribute(const int id);
    void getDfltAttributeList(int *active, int maxa) {
        _dfltAttributes.getList(active,maxa); }
    void copyAttributes(const int src, const int dest);
    int setFlynnAttribute(const int index, const int id, int val);
    int setFlynnAttribute(const int index, const int id, double val);
    int setFlynnAttribute(const int index, const int id, Coords_3D *val);
    int getFlynnAttribute(const int index, const int id, int *valptr);
    int getFlynnAttribute(const int index, const int id, double *valptr);
    int getFlynnAttribute(const int index, const int id, Coords_3D *valptr);
    int setDfltAttribute(const int id, int val) {
        _dfltAttributes.setAttribute(id, val); }
    int setDfltAttribute(const int id, double val) {
        _dfltAttributes.setAttribute(id, val); }
    int setDfltAttribute(const int id, Coords_3D *val) {
        _dfltAttributes.setAttribute(id, val); }
    int getDfltAttribute(const int id, int *valptr) {
        return(_dfltAttributes.getAttribute(id, valptr)); }
    int getDfltAttribute(const int id, double *valptr) {
        return(_dfltAttributes.getAttribute(id, valptr)); }
    int getDfltAttribute(const int id, Coords_3D *valptr) {
        return(_dfltAttributes.getAttribute(id, valptr)); }
    int max() { return(_a.size()); }
    int numberOfGrains();
    int numberOfSubgrains();
    bool attributesAllowMerge(const int flynn1, const int flynn2);
    int splitFlynn(const int type, int flynnindex, int *child1, int *child2,
                   float dx, float dy);
    int splitWrappingFlynn(int flynnindex,
                                   int node1, int node2,
                                   int *child1,int *child2);
    int directSplitWrappingFlynn(int flynnindex,
                                   int node1, int node2,
                                   int *child1,int *child2,
                                   Coords *dir);
    int mergeFlynns(int flynn1, int flynn2);
    int mergeFlynnsNoCheck(int flynn1, int flynn2);
    int checkSmallFlynn(const int index, double min_area);
    int writePolyFile(int flynnindex, const char *fname);
    void getChildList(int flynnindex, std::vector<int> &l)
              { _a[flynnindex].getChildList(l); }
    void addUnode(int flynnindex, int unode_id )
              { _a[flynnindex].addUnode(unode_id); }
    int removeUnode(int flynnindex, int unode_id )
              { return(_a[flynnindex].removeUnode(unode_id)); }
    int writeUnodePolyFile(int,const int ,
                           const char *);
    int getUnodeList(int flynnindex,std::vector<int> &l) 
              { _a[flynnindex].getUnodeList(l); }
    friend std::ostream & operator<< (std::ostream &, FlynnArray &);
};

#endif
