 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#ifndef _E_nodesP_h
#define _E_nodesP_h

#define  INITIAL_ATTRIBS  12

typedef struct {
    int nbnodes[3];
    int grains[3];
    double bndtype[3];
    int type;
    int state;
    double x;
    double y;
    double prev_x;
    double prev_y;
    double *attributes;
} NodeAttrib;

typedef struct {
    NodeAttrib *elems;
    int *randomorder;
    int maxnodes;
    int maxattributes; /* currently INITIAL_ATTRIBS */
    int activeattributes[INITIAL_ATTRIBS];
    int CONCactive;
    int topo_change;
} NodeArray;

extern NodeAttrib *ElleNode(int node);
#endif
