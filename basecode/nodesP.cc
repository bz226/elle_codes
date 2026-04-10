 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile: nodesP.cc,v $
 * Revision:  $Revision: 1.16 $
 * Date:      $Date: 2012/08/16 06:42:34 $
 * Author:    $Author: levans $
 *
 ******************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include "nodesP.h"
#include "nodes.h"
#include "unodes.h"
#include "file.h"
#include "runopts.h"
#include "polygon.h"
#include "error.h"

/*****************************************************

static const char rcsid[] =
       "$Id: nodesP.cc,v 1.16 2012/08/16 06:42:34 levans Exp $";

******************************************************/
#define XY 0
#if XY
#endif

#define INITIAL_NODES 500

/*
AttributeArray DefaultNodeAttribs(0);
*/
double DefaultNodeAttribs[INITIAL_ATTRIBS]= {
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0,
                          0.0
                         };


NodeArray Nodes;
void ElleRecalculateNodeCONC(int node, int update_node);

void ElleInitNodeArray(void)
{
    int err=0, i;
    /*
     * allocate an initial node array
     */
    Nodes.elems = 0;
    Nodes.randomorder = 0;
    Nodes.maxnodes = 0;
    Nodes.CONCactive = 0;
    Nodes.topo_change = 0;
    Nodes.maxattributes = 0;
    for (i=0;i<INITIAL_ATTRIBS;i++)
        Nodes.activeattributes[i] = NO_INDX;
    if (err = ElleReallocateNodeArray(INITIAL_NODES))
        OnError( "Reallocating node array",err );
}

NodeAttrib *ElleNode(int node)
{
    if (node >= Nodes.maxnodes) OnError("ElleNode", NODENUM_ERR);
    return( &(Nodes.elems[node]));
}

int ElleMaxNodes()
{
    return(Nodes.maxnodes);
}

int ElleMaxNodeAttributes()
{
    return(Nodes.maxattributes);
}

int ElleNodeCONCactive()
{
    return(Nodes.CONCactive);
}

void ElleSetNodeCONCactive(int val)
{
    Nodes.CONCactive = val;
}

int ElleNodeChange()
{
    return(Nodes.topo_change);
}

void ElleSetNodeChange(int val)
{
    Nodes.topo_change = val;
}

int *ElleNodeOrderArray()
{
    return(Nodes.randomorder);
}

int ElleNodeIsDouble(int node)
{
    NodeAttrib *p;

    if (node >= Nodes.maxnodes) OnError("ElleNodeIsDouble",NODENUM_ERR);
    p = ElleNode(node);
    return ((p->type==DOUBLE_J) ? 1 : 0);
}

int ElleNodeIsTriple(int node)
{
    NodeAttrib *p;

    if (node >= Nodes.maxnodes) OnError("ElleNodeIsTriple",NODENUM_ERR);
    p = ElleNode(node);
    return ((p->type==TRIPLE_J) ? 1 : 0);
}


void ElleSetNodeActive(int node)
{
    int max,err;

    max = ElleMaxNodes();
    if (node>=max) {
        if (node==MAX_INDX) OnError("ElleSetNodeActive",MAXINDX_ERR);
        if (node > INDX_CHECK) max = MAX_INDX;
        else max = (int)(node*1.5);
        if (err = ElleReallocateNodeArray(max))
            OnError( "Reallocating node array",err );
    }
    Nodes.elems[node].state = ACTIVE;
    ElleSetNodeChange(1);
}

void ElleCleanNodeArray()
{
    int max,maxa,i;

    maxa = Nodes.maxattributes;
    for (i=0;i<maxa;i++) 
        Nodes.activeattributes[i] = NO_INDX;
    max = ElleMaxNodes();
    ElleClearNodeArray(0,max-1);
    Nodes.maxattributes = 0;
}

void ElleRemoveNodeArray()
{
    int max,err,i;
    NodeAttrib *p;

    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        p = ElleNode(i);
        if (p->attributes!=0) {free(p->attributes); p->attributes=0;}
    }
    if (Nodes.elems) {free(Nodes.elems); Nodes.elems=0;}
    if (Nodes.randomorder) {free(Nodes.randomorder); Nodes.randomorder=0;}
    for (i=0;i<Nodes.maxattributes;i++)
        Nodes.activeattributes[i] = NO_INDX;
    Nodes.maxattributes=0;
    Nodes.maxnodes=0;
}

void ElleClearNodeEntry(int indx)
{

    if (indx >= Nodes.maxnodes)
        OnError("ElleClearNodeEntry",NODENUM_ERR);
    ElleClearNodeArray(indx,indx);
}

int ElleReallocateNodeArray(int size)
{
    int oldsize;
    int i;
    NodeAttrib *p;

    oldsize = Nodes.maxnodes;
    if ((Nodes.elems = (NodeAttrib *)realloc(Nodes.elems,
                                           size*sizeof(NodeAttrib))) ==0
        /*|| (Nodes.randomorder = (int *)realloc(Nodes.randomorder,
                                           size*sizeof(int))) ==0*/)
         return( MALLOC_ERR );
    Nodes.maxnodes = size;

    for (i=oldsize, p = &(Nodes.elems[oldsize]);i<size;i++,p++) 
        if ((p->attributes = (double *)malloc(INITIAL_ATTRIBS*
                                        sizeof(double)))==0)
             return( MALLOC_ERR );
    ElleClearNodeArray(oldsize,size-1);
    return( 0 );
}

void ElleClearNodeArray(int start, int end)
{
    int i, j;
    int maxa;
    NodeAttrib *p;

    if (end >= Nodes.maxnodes || start < 0) 
        OnError("ElleClearNodeArray",RANGE_ERR);
    maxa = Nodes.maxattributes;
    for (i=start, p = &(Nodes.elems[start]); i<=end; i++, p++) {
        p->x = 0.0;
        p->y = 0.0;
        for (j=0;j<3;j++)p->nbnodes[j] = NO_NB;
        for (j=0;j<3;j++)p->grains[j] = NO_NB;
        for (j=0;j<3;j++)p->bndtype[j] = NO_NB;
        p->type = INACTIVE;
        p->state = INACTIVE;
        p->x = p->y = p->prev_x = p->prev_y = 0.0;
        for (j=0;j<maxa;j++) {
            p->attributes[j] = DefaultNodeAttribs[j];
        }
    }
    ElleSetNodeChange(1);
}

int ElleFindSpareNode()
{
    int index = 0,j;
    int max, maxa;
    NodeAttrib *p;

    max = ElleMaxNodes();
    p = Nodes.elems;
    while (index < max && p[index].state != INACTIVE) index++;
    ElleSetNodeActive(index);
    maxa = Nodes.maxattributes;
    p = Nodes.elems; /* in case it was reallocated */
    for (j=0;j<maxa;j++)
        p[index].attributes[j] = DefaultNodeAttribs[j];
    return(index);
}

void ElleSetRegionEntry(int node, int indx, ERegion id)
{
    NodeAttrib *p;

    if (indx>2 || indx<0) OnError("ElleSetRegionEntry",LIMIT_ERR);
    p = ElleNode(node);
    p->grains[indx] = id;
}

int ElleSetNeighbour(int node, int indx, int nb, ERegion *id)
{
    int i=0, count;
    NodeAttrib *p;

    p = ElleNode(node);
    if (indx==NO_NB) {
        while (i<3 && p->nbnodes[i]!=NO_NB) i++;
        indx = i;
    }
    if (indx>2 || indx<0) return(LIMIT_ERR);
    p->nbnodes[indx]=nb;
    for (i=0,count=0;i<3;i++) if (p->nbnodes[i]!=NO_NB) count++;
    if (count==3) {
        p->type &= ~DOUBLE_J;
        p->type |= TRIPLE_J;
    }
    else if (count==2) {
        p->type &= ~TRIPLE_J;
        p->type |= DOUBLE_J;
    }
    if (id!=0) {
        p->grains[indx] = *id;
    }
    return(0);
}

int ElleClearNeighbour(int node, int indx)
{
    int i=0, count;
    ERegion rgn;
    NodeAttrib *p;

    if (indx>2 || indx<0) return(LIMIT_ERR);
    if (ElleNodeIsSingle(node)) ElleDeleteSingleJ(node);
    else {
        rgn = NO_NB;
        ElleSetNeighbour(node,indx,NO_NB,&rgn);
        p = ElleNode(node);
        for (i=0,count=0;i<3;i++) if (p->nbnodes[i]!=NO_NB) count++;
        if (count==2) {
            p->type &= ~TRIPLE_J;
            p->type |= DOUBLE_J;
        }
        else if (count==1) {
            p->type &= ~DOUBLE_J;
        }
    ElleSetNodeChange(1);
    }
    return(0);
}

int ElleNeighbourNodes(int node, int *nbnodes)
{
   /*
    * Parameters: node - the node number
    *             nbnodes - an integer array with size >= 3
    * Action: Sets the first 3 entries of nbnodes to the node numbers of
   *         the neighbours of the node (one of the entries may be NO_NB
   *         if node is only connected to two neighbours)
    * Return value: NODENUM_ERR if the requested node is inactive
    *               else 0
    */
    int i;
    NodeAttrib *p;

    if (!ElleNodeIsActive(node))
        OnError("NeighbourNodes",NODENUM_ERR);
    /*
     * find the node numbers of the neighbours
     */
    p = ElleNode(node);
    for (i=0;i<3;i++) nbnodes[i]=p->nbnodes[i];
    return(0);
}

int ElleRegions(int node, ERegion *rgns)
{
   /*
    * Parameters: node - the node number
    *             rgns - a Region array with size >= 3
    * Action: Sets the first 3 entries of rgns to the regions of
    *         the neighbours of the node (one of the entries
    *         may be NO_NB, NO_NB, NO_NB if node double
    */
    int i;
    NodeAttrib *p;

    if (!ElleNodeIsActive(node))
        OnError("ElleRegions",NODENUM_ERR);
    p = ElleNode(node);
    for (i=0;i<3;i++) {
        rgns[i]=p->grains[i];
    }
    return(0);

}

void ElleNeighbourRegion(int node, int nb, ERegion *region)
{
    int i;
    NodeAttrib *p;

    if (!ElleNodeIsActive(node))
        OnError("NeighbourRegion",NODENUM_ERR);
    if ((i = ElleFindNbIndex(nb,node))==NO_NB)
        OnError("NeighbourRegion",NONB_ERR);
    p = ElleNode(node);
    *region = p->grains[i];
}

void ElleSetPosition(int node, Coords *position)
{
   /*
    * Parameters: node - the node number
    *             position - the address of a variable of type Coords
    * Action: Sets the position of node to the values in position
    *         after converting them to unit coordinates, if necessary
    */
    Coords xy;
    NodeAttrib *p;

    p = ElleNode(node);
    p->prev_x = p->x;
    p->prev_y = p->y;
    xy.x = position->x;
    xy.y = position->y;
    ElleNodeUnitXY(&xy);
    p->x = xy.x;
    p->y = xy.y;
}

void ElleSetPrevPosition(int node, Coords *position)
{
   /*
    * Parameters: node - the node number
    *             position - the address of a variable of type Coords
    * Action: Sets the previous position of node to the values in
    *         position after converting them to unit coordinates,
    *         if necessary (usually used when initialising a new node)
    */
    Coords xy;
    NodeAttrib *p;

    p = ElleNode(node);
    xy.x = position->x;
    xy.y = position->y;
    ElleNodeUnitXY(&xy);
    p->prev_x = xy.x;
    p->prev_y = xy.y;
}

void ElleCopyToPosition(int node, Coords *position)
{
   /*
    * Parameters: node - the node number
    *             position - the address of a variable of type Coords
    * Action: Sets the position of node to the values in position
    */
    NodeAttrib *p;

    p = ElleNode(node);
    p->prev_x = p->x;
    p->prev_y = p->y;
    p->x = position->x;
    p->y = position->y;
}

void ElleCopyToPrevPosition(int node, Coords *position)
{
   /*
    * Parameters: node - the node number
    *             position - the address of a variable of type Coords
    * Action: Sets the previous position of node to the values in
    *         position (usually used when initialising a new node)
    */
    NodeAttrib *p;

    p = ElleNode(node);
    p->prev_x = position->x;
    p->prev_y = position->y;
}

void ElleUpdatePosition(int node, Coords *increment)
{
   /*
    * Parameters: node - the node number
    *             increment - the address of a variable of type Coords
    * Action: Increments the position of node by the values in increment
    *         after converting them to unit coordinates, if necessary
    *         If any node attributes are concentration values, need to
    *         recalculate as boundary areas will have changed and/or 
    *         unodes have been swept
    */
    Coords xy;
    NodeAttrib *p;

    int update_node=1, i;
    set_int *unodelists[MAX_LOCAL_FLYNNS+1];
    set_int *flynnlists[MAX_LOCAL_FLYNNS+1];
    for (i=0;i<MAX_LOCAL_FLYNNS+1;i++) {
        unodelists[i]=0;
        flynnlists[i]=0;
    }

    p = ElleNode(node);
    p->prev_x = p->x;
    p->prev_y = p->y;
    xy.x = p->x + increment->x;
    xy.y = p->y + increment->y;
    if (isnan(xy.x) || isnan(xy.y))
        OnError("nan position",0);
    ElleNodeUnitXY(&xy);
    if (UnodesActive()) {
        ElleLocalLists(node,increment,unodelists,flynnlists,
                                               ElleUnodeROI());
        if (ElleUnodeCONCactive() && ElleNodeCONCactive()) {
/*
           int rgn[3];
           bool do_conc=0;
           ElleRegions(node,rgn);
           if ((rgn[0]==2||rgn[0]==4||rgn[0]==NO_NB) &&
              (rgn[1]==2||rgn[1]==4||rgn[1]==NO_NB) &&
              (rgn[2]==2||rgn[2]==4||rgn[2]==NO_NB)) do_conc=1;
           if (ElleNodeIsDouble(node) && do_conc){
*/
            UpdateGBMUnodeValues(node,*increment,
                                 unodelists, flynnlists[0],
                                 ElleUnodeROI());
            update_node=0;
            /*}*/
        }
    }
    p->x = xy.x;
    p->y = xy.y;


    if (UnodesActive()) {
        ElleReassignUnodes(unodelists[COMBINED_LIST],flynnlists[0]);
        for (i=0;i<MAX_LOCAL_FLYNNS+1;i++) {
            if (unodelists[i]!=0)
                unodelists[i]->erase(unodelists[i]->begin(),
                                     unodelists[i]->end());
            if (flynnlists[i]!=0)
                flynnlists[i]->erase(flynnlists[i]->begin(),
                                     flynnlists[i]->end());
        }
    }

    if (ElleNodeCONCactive() && update_node)
        ElleRecalculateNodeCONC(node,update_node);
}

/*
 * estimate the concentration of conc attribute from the values 
 * of the unodes listed (unodes in a flynn or unodes inside a
 * bounding region)
 */
double ElleNodeMarginConcEstimate(int node, int attr_id,
				std::vector<int> *unodelist, double roi)
{
    double val=0.0;
    Coords xy;

    ElleNodePosition(node,&xy);
    val = EstimateConcFromUnodeValues(&xy,attr_id,unodelist,roi);
    return(val);
}

void ElleNodePosition(int node, Coords *position)
{
   /*
    * Parameters: node - the node number
    *             position - the address of a variable of type Coords
    * Action: Sets the x and y fields of position to those of the
    *         requested node
    * Return value: NODENUM_ERR if the requested node is inactive
    *               else 0
    */
    NodeAttrib *p;

    if (node >= ElleMaxNodes())
        OnError("ElleNodePosition",NODENUM_ERR);
    if (!ElleNodeIsActive(node))
        OnError("ElleNodePosition",NODENUM_ERR);
    p = ElleNode(node);
    position->x = p->x;
    position->y = p->y;
}

void ElleNodePrevPosition(int node, Coords *previous)
{
   /*
    * Parameters: node - the node number
    *             previous - the address of a variable of type Coords
    * Action: Sets the x and y fields of previous to those of the
    *         requested node's position before it was last moved
    * Return value: NODENUM_ERR if the requested node is inactive
    *               else 0
    */
    NodeAttrib *p;

    if (node >= ElleMaxNodes())
        OnError("ElleNodePrevPosition",NODENUM_ERR);
    if (!ElleNodeIsActive(node))
        OnError("ElleNodePrevPosition",NODENUM_ERR);
    p = ElleNode(node);
    previous->x = p->prev_x;
    previous->y = p->prev_y;
}

int ElleNodeIsActive(int node)
{
    NodeAttrib *p;

    if (node<0 || node >= ElleMaxNodes()) return(0);
    p = ElleNode(node);
    return ((p->state==ACTIVE) ? 1 : 0);
}

int ElleNodeIsSingle(int node)
{
    int cnt,i,nbnodes[3];

    ElleNeighbourNodes(node,nbnodes);
    for (i=0,cnt=0;i<3;i++)
        if (nbnodes[i]!=NO_NB) cnt++;
    return ((cnt==1) ? 1 : 0);
}

int ElleConcAIndex()
{
    return(ElleFindNodeAttribIndex(CONC_A));
}

int ElleVelocityXIndex()
{
    return(ElleFindNodeAttribIndex(VEL_X));
}

int ElleVelocityYIndex()
{
    return(ElleFindNodeAttribIndex(VEL_Y));
}

int ElleTauXXIndex()
{
    return(ElleFindNodeAttribIndex(TAU_XX));
}

int ElleTauYYIndex()
{
    return(ElleFindNodeAttribIndex(TAU_YY));
}

int ElleTauZZIndex()
{
    return(ElleFindNodeAttribIndex(TAU_ZZ));
}

int ElleTauXYIndex()
{
    return(ElleFindNodeAttribIndex(TAU_XY));
}

int ElleTau1Index()
{
    return(ElleFindNodeAttribIndex(TAU_1));
}

int EllePressureIndex()
{
    return(ElleFindNodeAttribIndex(PRESSURE));
}

int ElleStrainIncrIndex()
{
    return(ElleFindNodeAttribIndex(INCR_S));
}

int ElleBulkStrainIndex()
{
    return(ElleFindNodeAttribIndex(BULK_S));
}

void ElleInitNodeAttribute(int id)
{
    int index, max, i;
    double val=0;

    if (ElleNodeAttributeActive(id)) return;
    if ((index=ElleSpareNodeAttribIndex(id))==NO_INDX)
        OnError("ElleInitNodeAttribute",MAXATTRIB_ERR);
    switch(id) {
    case VEL_X: val = ElleNodeDefaultVelocityX();
                break;
    case VEL_Y: val = ElleNodeDefaultVelocityY();
                break;
    case TAU_XX: val = DefaultNodeAttribs[I_TAU_XX];
                break;
    case TAU_YY: val =  DefaultNodeAttribs[I_TAU_YY];
                break;
    case TAU_ZZ: val =  DefaultNodeAttribs[I_TAU_ZZ];
                break;
    case TAU_XY: val =  DefaultNodeAttribs[I_TAU_XY];
                break;
    case TAU_1 : val =  DefaultNodeAttribs[I_TAU_1 ];
                break;
    case PRESSURE: val =  DefaultNodeAttribs[I_PRESSURE];
                break;
    case CONC_A: val = ElleNodeDefaultConcA();
                 ElleSetNodeCONCactive(1);
                break;
    case INCR_S: val =  DefaultNodeAttribs[I_SR_INCR];
                break;
    case BULK_S: val =  DefaultNodeAttribs[I_STRN_BULK];
                break;
    case ATTRIB_A:
    case ATTRIB_B:
    case ATTRIB_C:
    case N_MCONC_A_1:
    case N_MCONC_A_2:
    case N_MCONC_A_3:
    case N_MATTRIB_A_1:
    case N_MATTRIB_A_2:
    case N_MATTRIB_A_3:
    case N_MATTRIB_B_1:
    case N_MATTRIB_B_2:
    case N_MATTRIB_B_3:
    case N_MATTRIB_C_1:
    case N_MATTRIB_C_2:
    case N_MATTRIB_C_3: val = 0;
                break;
    default:     OnError("ElleInitNodeAttribute",ATTRIBID_ERR);
                break;
    }
    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) 
            ElleSetNodeAttribute(i,val,id);
    }
}

void ElleInitNodeMarginAttribute(int id)
{
    switch(id) {
      case CONC_A: ElleInitNodeAttribute(N_MCONC_A_1);
                   ElleInitNodeAttribute(N_MCONC_A_2);
                   ElleInitNodeAttribute(N_MCONC_A_3);
                break;
      case ATTRIB_A: ElleInitNodeAttribute(N_MATTRIB_A_1);
                   ElleInitNodeAttribute(N_MATTRIB_A_2);
                   ElleInitNodeAttribute(N_MATTRIB_A_3);
                break;
      case ATTRIB_B: ElleInitNodeAttribute(N_MATTRIB_B_1);
                   ElleInitNodeAttribute(N_MATTRIB_B_2);
                   ElleInitNodeAttribute(N_MATTRIB_B_3);
                break;
      case ATTRIB_C: ElleInitNodeAttribute(N_MATTRIB_C_1);
                   ElleInitNodeAttribute(N_MATTRIB_C_2);
                   ElleInitNodeAttribute(N_MATTRIB_C_3);
                break;
      default:     OnError("ElleInitMarginNodeAttribute",ATTRIBID_ERR);
                break;
    }
}

int ElleRemoveNodeAttribute(int index)
{
    int i=0, match=NO_INDX;

    for (i=0;i<Nodes.maxattributes && match==NO_INDX;i++) {
        if (Nodes.activeattributes[i]==index) {
            match = i;
            Nodes.activeattributes[i]=NO_INDX;
            Nodes.maxattributes--;
        }
    }
    return(match);
}

int ElleFindNodeAttribIndex(int index)
{
    int i=0, match=NO_INDX;

    for (i=0;i<Nodes.maxattributes && match==NO_INDX;i++)
        if (Nodes.activeattributes[i]==index) match = i;
    return(match);
}

int ElleSpareNodeAttribIndex(int index)
{
    int i, match=NO_INDX;

    for (i=0;i<INITIAL_ATTRIBS && match==NO_INDX;i++) {
        if (Nodes.activeattributes[i]==NO_INDX)  {
            match = i;
            Nodes.activeattributes[i] = index;
            Nodes.maxattributes++;
        }
    }
    return (match);
}

void ElleSetNodeAttribute(int node, double val, int index)
{
    int attribindex;
    NodeAttrib *p;

    if ((attribindex=ElleFindNodeAttribIndex(index))==NO_INDX)
        OnError("ElleSetNodeAttribute",RANGE_ERR);
    p = ElleNode(node);
    p->attributes[attribindex] = val;
}

void ElleSetNodeMarginAttribute(int node, double val,
                                            int index, int rgn)
{
    int i, idx=NO_NB;
    int attribindex;
    NodeAttrib *p;

    if ((i=ElleFindBndIndex(node,rgn))==NO_NB)
        OnNodeError(node,"ElleSetNodeMarginAttribute",NONB_ERR);
    else {
      switch (index) {
      case CONC_A:
        if (i==0) idx=N_MCONC_A_1;
        else if (i==1) idx=N_MCONC_A_2;
        else if (i==2) idx=N_MCONC_A_3;
        break;
      case ATTRIB_A:
        if (i==0) idx=N_MATTRIB_A_1;
        else if (i==1) idx=N_MATTRIB_A_2;
        else if (i==2) idx=N_MATTRIB_A_3;
        break;
      case ATTRIB_B:
        if (i==0) idx=N_MATTRIB_B_1;
        else if (i==1) idx=N_MATTRIB_B_2;
        else if (i==2) idx=N_MATTRIB_B_3;
        break;
      case ATTRIB_C:
        if (i==0) idx=N_MATTRIB_C_1;
        else if (i==1) idx=N_MATTRIB_A_2;
        else if (i==2) idx=N_MATTRIB_C_3;
        break;
      }
    }
    if ((attribindex=ElleFindNodeAttribIndex(idx))==NO_INDX)
        OnError("ElleSetNodeAttribute",RANGE_ERR);
    p = ElleNode(node);
    p->attributes[attribindex] = val;
}

double ElleNodeAttribute(int node, int id)
{
    int attribindex;
    NodeAttrib *p;

    if ((attribindex=ElleFindNodeAttribIndex(id))==NO_INDX)
        OnError("ElleNodeAttribute",RANGE_ERR);
    p = ElleNode(node);
    return(p->attributes[attribindex]);
}

double ElleNodeMarginAttribute(int node, int index, int rgn)
{
    int i, idx=NO_NB;
    int attribindex;
    NodeAttrib *p;

    if ((i=ElleFindBndIndex(node,rgn))==NO_NB)
        OnNodeError(node,"ElleSetNodeMarginAttribute",NONB_ERR);
    else {
      switch (index) {
      case CONC_A:
        if (i==0) idx=N_MCONC_A_1;
        else if (i==1) idx=N_MCONC_A_2;
        else if (i==2) idx=N_MCONC_A_3;
        break;
      case ATTRIB_A:
        if (i==0) idx=N_MATTRIB_A_1;
        else if (i==1) idx=N_MATTRIB_A_2;
        else if (i==2) idx=N_MATTRIB_A_3;
        break;
      case ATTRIB_B:
        if (i==0) idx=N_MATTRIB_B_1;
        else if (i==1) idx=N_MATTRIB_B_2;
        else if (i==2) idx=N_MATTRIB_B_3;
        break;
      case ATTRIB_C:
        if (i==0) idx=N_MATTRIB_C_1;
        else if (i==1) idx=N_MATTRIB_C_2;
        else if (i==2) idx=N_MATTRIB_C_3;
        break;
      }
    }
    if ((attribindex=ElleFindNodeAttribIndex(idx))==NO_INDX)
        OnError("ElleSetNodeAttribute",RANGE_ERR);
    p = ElleNode(node);
    return(p->attributes[attribindex]);
}

int ElleIndexNb(int node, int index)
{
    int max;
    NodeAttrib *p;

    max = ElleMaxNodes();
    p = ElleNode(node);
    if (node >= max) OnError("ElleIndexNb",NODENUM_ERR);
    return(p->nbnodes[index]);
}

double ElleNodeDefaultConcA()
{
    return(DefaultNodeAttribs[I_CONC_A]);
}

double ElleNodeDefaultVelocityX()
{
    return(DefaultNodeAttribs[I_VEL_X]);
}

double ElleNodeDefaultVelocityY()
{
    return(DefaultNodeAttribs[I_VEL_Y]);
}

void ElleSetNodeDefaultConcA(double val)
{
    DefaultNodeAttribs[I_CONC_A] = val;
}

void ElleSetNodeDefaultVelocityX(double val)
{
    DefaultNodeAttribs[I_VEL_X] = val;
}

void ElleSetNodeDefaultVelocityY(double val)
{
    DefaultNodeAttribs[I_VEL_Y] = val;
}

void ElleIntrpAttributes(int n)
{
    int i, nb[3],nb1,nb2;
    int maxa=0, *active=0;
    double val,val1,val2;
    double d1,d2;
    Coords xy;

    if (!ElleNodeIsDouble(n))
        OnError("ElleIntrpAttributes - not double",0);
    ElleNeighbourNodes(n,nb);
    ElleNodePosition(n,&xy);
    if (nb[0]==NO_NB) {
        nb1 = nb[1];
        nb2 = nb[2];
    }
    if (nb[1]==NO_NB) {
        nb1 = nb[0];
        nb2 = nb[2];
    }
    else {
        nb1 = nb[0];
        nb2 = nb[1];
    }
    /*
     * interpolate
     */
    d1 = ElleNodeSeparation(n,nb1);
    d2 = ElleNodeSeparation(n,nb2);
    ElleNodeAttributeList(&active,&maxa);
    for (i=0;i<maxa;i++) {
        if (active[i]!=VEL_X && active[i]!=VEL_Y) {
            val1 = ElleNodeAttribute(nb1,active[i]);
            val2 = ElleNodeAttribute(nb2,active[i]);
            if ((d1 + d2) <= 0) val = (val1+val2)/2;
            else val = (double)((d1*val2 + d2*val1) / (d1 + d2));
            ElleSetNodeAttribute(n,val,active[i]);
        }
    }
    if (active) free(active);
}

void ElleRedistributeNodeCONC(int node,int nb1,int nb2)
{
    int i, j, nbs[3];
    int maxa=0, *active=0;
    double conc1, conc2, conc1new, conc2new;
    double len1=0, len2=0, m_add=0, totlen1=0;
    double d1=0, d2=0, newlen=0;

    /*
     * node is about to be deleted so its mass is split b/w
     * nbs and the concentrations recalculated
     *    L11\            /L21
     *        \  d1   d2 /
     *         1----n---2     
     *        /          \
     *    L12/            \L22
     * len1 and len2 are the boundary lengths to the other 
     * nbs of nb1 and nb2 (L11+L12, L21+L22)
     * newlen is length of boundary 1-2 after n removed
     * it may straighten and not be equal to d1+d2
     *
     * if nb1==nb2 - removing 2 sided grain
     * if nb2==NO_NB - removing singleJ
     */
    ElleNeighbourNodes(nb1,nbs);
    for (j=0;j<3;j++)
        if (nbs[j]!=NO_NB && nbs[j]!=node)
            len1 += ElleNodeSeparation(nb1,nbs[j]);
    d1 = ElleNodeSeparation(node,nb1);
    if (nb2!=NO_NB) {
        ElleNeighbourNodes(nb2,nbs);
        for (j=0;j<3;j++) 
            if (nbs[j]!=NO_NB && nbs[j]!=node)
                len2 += ElleNodeSeparation(nb2,nbs[j]);
        d2 = ElleNodeSeparation(node,nb2);
        newlen = ElleNodeSeparation(nb1,nb2);
    }
    ElleNodeAttributeList(&active,&maxa);
    for (i=0;i<maxa;i++) {
        if (ElleNodeCONCAttribute(active[i])) {
            conc1 = ElleNodeAttribute(nb1,active[i]);
            m_add = ElleNodeAttribute(node,active[i]) * (d1+d2) * 0.5;
            totlen1 = len1+d1;
            if (nb2!=NO_NB && nb1!=nb2) m_add *= 0.5;
            if (nb1==nb2) totlen1 += d2;
            conc1new = (totlen1*conc1*0.5 + m_add)/((len1+newlen)*0.5);
            ElleSetNodeAttribute(nb1,conc1new,active[i]);
            if (nb2!=NO_NB && nb1!=nb2) {
                conc2 = ElleNodeAttribute(nb2,active[i]);
                conc2new = ((len2+d2)*conc2*0.5 + m_add)/((len2+newlen)*0.5);
                ElleSetNodeAttribute(nb2,conc2new,active[i]);
            }
            ElleSetNodeAttribute(node,0,active[i]);
        }
    }
    if (active) free(active);
}

void ElleRecalculateNodeCONC(int node, int update_node)
{
    unsigned char done=0;
    int i, j, k, nbs[3], nbnodes[3];
    int maxa=0, *active=0;
    double conc, newconc;
    double len=0, newlen=0, tmp;
    double d[3], newd[3];

    /*
     * node has moved
     * node and nb concentrations recalculated
     *    L11\            /L21
     *        \  d1   d2 /
     *         1----n---2     
     *        /          \
     *    L12/            \L22
     * len is the boundary lengths to the other 
     * nbs of nb1 and nb2 (L11+L12, L21+L22)
     * 
     */
    for (j=0;j<3;j++) d[j]=newd[j]=0;
    ElleNodeAttributeList(&active,&maxa);
    ElleNeighbourNodes(node,nbs);
    for (j=0;j<3;j++) {
        if (nbs[j]!=NO_NB) {
            len = newlen = 0.0;
            d[j] = ElleNodePrevSeparation(node,nbs[j]);
            newd[j] = ElleNodeSeparation(node,nbs[j]);
           /*
            * may have 2-sided grains so check before
            * recalculating concentration
            */
            for (i=0,done=0;i<j&&!done;i++) if (nbs[i]==nbs[j]) done=1;
            if (!done) {
              ElleNeighbourNodes(nbs[j],nbnodes);
              for (i=0;i<3;i++) {
                  if (nbnodes[i]!=NO_NB) {
                      if (nbnodes[i]!=node) {
                          tmp = ElleNodeSeparation(nbs[j],nbnodes[i]);
                          len += tmp;
                          newlen += tmp;
                      }
                      else {
                          len += d[j];
                          newlen += newd[j];
                      }
                  }
              }
              for (k=0;k<maxa;k++) {
                  if (ElleNodeCONCAttribute(active[k])) {
                      conc = ElleNodeAttribute(nbs[j],active[k]);
                      newconc = conc*len/newlen;
                      ElleSetNodeAttribute(nbs[j],newconc,active[k]);
                  }
              }
            }
        }
    }
    if (update_node!=0) {
    for (k=0;k<maxa;k++) {
        if (ElleNodeCONCAttribute(active[k])) {
            conc = ElleNodeAttribute(node,active[k]);
            newconc = conc*(d[0]+d[1]+d[2])/(newd[0]+newd[1]+newd[2]);
            ElleSetNodeAttribute(node,newconc,active[k]);
        }
    }
    }
    if (active) free(active);
}
#if XY
void ElleRecalculateNodeCONC(int node)
{
    int i, j, k, nbs[3], nbnodes[3];
    int maxa=0, *active=0;
    double conc, newconc;
    double len=0, newlen=0;
    double d[3], newd[3];

    /*
     * node has moved
     * node and nb concentrations recalculated
     *    L11\            /L21
     *        \  d1   d2 /
     *         1----n---2     
     *        /          \
     *    L12/            \L22
     * len is the boundary lengths to the other 
     * nbs of nb1 and nb2 (L11+L12, L21+L22)
     * 
     */
    for (j=0;j<3;j++) d[j]=newd[j]=0;
    ElleNodeAttributeList(&active,&maxa);
    ElleNeighbourNodes(node,nbs);
    for (j=0;j<3;j++) {
        if (nbs[j]!=NO_NB) {
            len = newlen = 0.0;
            d[j] = ElleNodePrevSeparation(node,nbs[j]);
            newd[j] = ElleNodeSeparation(node,nbs[j]);
            ElleNeighbourNodes(nbs[j],nbnodes);
            for (i=0;i<3;i++) {
                if (nbnodes[i]!=NO_NB) {
                    if (nbnodes[i]!=node) 
                        len += ElleNodeSeparation(nbs[j],nbnodes[i]);
                }
            }
            newlen = len+newd[j];
            len += d[j];
            for (k=0;k<maxa;k++) {
                if (ElleNodeCONCAttribute(active[k])) {
                    conc = ElleNodeAttribute(nbs[j],active[k]);
                    newconc = conc*len/newlen;
                    ElleSetNodeAttribute(nbs[j],newconc,active[k]);
                }
            }
        }
    }
    for (k=0;k<maxa;k++) {
        if (ElleNodeCONCAttribute(active[k])) {
            conc = ElleNodeAttribute(node,active[k]);
            newconc = conc*(d[0]+d[1]+d[2])/(newd[0]+newd[1]+newd[2]);
            ElleSetNodeAttribute(node,newconc,active[k]);
        }
    }
    if (active) free(active);
}
#endif

void ElleRecalculateNodeCONCrb(int node1, int node2)
{
    int i, j, k, nbs[3], nbnodes[3];
    int maxa=0, *active=0;
    double conc, newconc;
    double len1=0, newlen1=0;
    double len2=0, newlen2=0, len12;

    /*
     * boundary between node1 and node2 removed
     * node1 and node2 concentrations recalculated
     *    L11\            /L21
     *        \  d1   d2 /
     *         1----n---2     
     *        /          \
     *    L12/            \L22
     * len is the boundary lengths to the other 
     * nbs of node1 and node2 (L11+L12, L21+L22)
     * 
     */
    ElleNodeAttributeList(&active,&maxa);
    len12 = ElleNodeSeparation(node1,node2);
    ElleNeighbourNodes(node1,nbs);
    for (j=0;j<3;j++) {
        if (nbs[j]!=NO_NB)
            len1 += ElleNodeSeparation(node1,nbs[j]);
    }
    newlen1 = len1-len12;
    ElleNeighbourNodes(node2,nbs);
    for (j=0;j<3;j++) {
        if (nbs[j]!=NO_NB)
            len2 += ElleNodeSeparation(node2,nbs[j]);
    }
    newlen2 = len2-len12;
    for (k=0;k<maxa;k++) {
        if (ElleNodeCONCAttribute(active[k])) {
            conc = ElleNodeAttribute(node1,active[k]);
            newconc = conc*len1/newlen1;
            ElleSetNodeAttribute(node1,newconc,active[k]);
            conc = ElleNodeAttribute(node2,active[k]);
            newconc = conc*len2/newlen2;
            ElleSetNodeAttribute(node2,newconc,active[k]);
        }
    }
    if (active) free(active);
}

int ElleNodeCONCAttribute(int attr_id)
{
    int is_conc=0;

    if (attr_id==CONC_A) is_conc=1;
    return(is_conc);
}

int ElleNodeNumAttributes()
{
    return(Nodes.maxattributes);
}

void ElleNodeAttributeList(int **active, int *max)
{
    int i, *p;

    *active = 0;
    if ((*max = ElleNodeNumAttributes())!=0) {
        if ((*active=(int *)malloc(*max * sizeof(int)))==0)
            OnError("ElleNodeAttributeList",MALLOC_ERR);
        for(i=0, p=*active;i<*max;i++,p++)
            *p = Nodes.activeattributes[i];
    }
}

void ElleSetBoundaryAttribute(int node, int index, double val)
{
    int max;
    NodeAttrib *p;

    max = ElleMaxNodes();
    if (node >= max) OnError("ElleSetBoundaryAttribute",NODENUM_ERR);
    p = ElleNode(node);
    p->bndtype[index] = val;
}

// Nov. 13th, 2015: Florian Steinbach added this function: ElleGetNodeAttribute
void ElleGetNodeAttribute(int node, double *val, int id)
{
    int attribindex;
    NodeAttrib *p;
    
    if ((attribindex=ElleFindNodeAttribIndex(id))==NO_INDX)
        OnError("ElleSetNodeAttribute",RANGE_ERR);
    
    p = ElleNode(node);
    *val = p->attributes[attribindex];
}

double ElleGetBoundaryAttribute(int node, int nb)
{
    int i, max;
    NodeAttrib *p;

    max = ElleMaxNodes();
    if (node >= max || nb >= max)
        OnError("ElleGetBoundaryAttribute",NODENUM_ERR);
    if (nb < node) {
        if ((i = ElleFindNbIndex(nb,node))==NO_NB)
            OnError("ElleGetBoundaryAttribute",0);
        p = ElleNode(node);
    }
    else {
        if ((i = ElleFindNbIndex(node,nb))==NO_NB)
            OnError("ElleGetBoundaryAttribute",0);
        p = ElleNode(nb);
    }
    return(p->bndtype[i]);
}

#define MAX_SIZE  0.5
/*#define MAX_SIZE  0.4*/

void ElleNodePlotXY(int current, Coords *xy, Coords *prevxy)
{
    int max;
    NodeAttrib *p;

    max = ElleMaxNodes();
    if (current >= max) OnError("ElleNodePlotXY",NODENUM_ERR);
    p = ElleNode(current);
    xy->x = p->x;
    xy->y = p->y;
    ElleCoordsPlotXY(xy, prevxy);
}

int ellepointplotxy_(double *x, double *y, double *prevx, double *prevy)
{
    int err=0;
    Coords xy, prevxy;

    xy.x = *x;
    xy.y = *y;
    prevxy.x = *prevx;
    prevxy.y = *prevy;
    ElleCoordsPlotXY(&xy, &prevxy);
    *x = xy.x;
    *y = xy.y;
    return(err);
}

void ElleCoordsPlotXY(Coords *xy, Coords *prevxy)
{
    int cnt;
    double unitsize_x,unitsize_y;
    double deformx, deformy;
    CellData unitcell;

    ElleCellBBox(&unitcell);
    deformx = unitcell.cellBBox[TOPLEFT].x-unitcell.cellBBox[BASELEFT].x;
    deformy = unitcell.cellBBox[BASELEFT].y-unitcell.cellBBox[BASERIGHT].y;
    /* 
     * assumes that the unit cell remains a parallelogram
     * (simple shear ?)
     * assuming yoffset is zero
     * unitcell.xoffset is the simple shear (x) + any cell deformation
     */
    unitsize_x = unitcell.xlength;
    unitsize_y = unitcell.ylength;
    
    // Y-component
    
    if ((xy->y - prevxy->y) >= unitsize_y*MAX_SIZE) 
    {
        xy->y -= unitsize_y;
        xy->x -= unitcell.xoffset;
        while ((xy->y - prevxy->y) >= unitsize_y*MAX_SIZE) 
        {
            xy->y -= unitsize_y;
            xy->x -= unitcell.xoffset;
        }
    }
    else if ((xy->y - prevxy->y) < -unitsize_y*MAX_SIZE) 
    {
        xy->y += unitsize_y;
        xy->x += unitcell.xoffset;
        while ((xy->y - prevxy->y) < -unitsize_y*MAX_SIZE) 
        {
            xy->y += unitsize_y;
            xy->x += unitcell.xoffset;
        }
    }
    
    // X-component
    
    if ((xy->x - prevxy->x) >= unitsize_x*MAX_SIZE) 
    {
        xy->x -= unitsize_x;
        xy->y -= unitcell.yoffset;
        while ((xy->x - prevxy->x) >= unitsize_x*MAX_SIZE) 
        {
            xy->x -= unitsize_x;
            xy->y -= unitcell.yoffset;
        }
    }
    else if ((xy->x - prevxy->x) < -unitsize_x*MAX_SIZE) 
    {
        xy->x += unitsize_x;
        xy->y += unitcell.yoffset;
        while ((xy->x - prevxy->x) < -unitsize_x*MAX_SIZE) 
        {
            xy->x += unitsize_x;
            xy->y += unitcell.yoffset;
        }
    }
}

bool ElleCoordRelativePosition(Coords *current,
                              Coords *bnd,
                              int num_bnd_nodes,
                              int *xflags,int *yflags,
                              Coords *rel_pos)
{
    bool inside=true;
    int i,j, k;
    double eps=1.5e-6;
    double incrx, incry, xoffset;
    Coords tmp;
    CellData unitcell;

    ElleCellBBox(&unitcell);
    incrx = unitcell.xlength;
    incry = unitcell.ylength;
    xoffset = unitcell.xoffset;
    if ((unitcell.cum_xoffset<-eps || unitcell.cum_xoffset>eps) &&
            xoffset>unitcell.xlength-eps && xoffset<unitcell.xlength+eps)
        xoffset=0;

    *rel_pos = *current;
    if (!EllePtInRegion(bnd,num_bnd_nodes,current)) {
        inside = false;
        tmp = *current;
        j=0;
        while (!inside && j<xflags[0]) {
            k=0;
            while (!inside && k<=yflags[0]) {
                tmp.x = current->x - (j+1)*incrx - k*xoffset;
                tmp.y = current->y - k*incry;
                if (EllePtInRegion(bnd,num_bnd_nodes,&tmp)) {
                    inside= true;
                    *rel_pos = tmp;
                }
                k++;
            }
            k=0;
            while (!inside && k<=yflags[1]) {
                tmp.x = current->x - (j+1)*incrx + k*xoffset;
                tmp.y = current->y + k*incry;
                if (EllePtInRegion(bnd,num_bnd_nodes,&tmp)) {
                    inside= true;
                    *rel_pos = tmp;
                }
                k++;
            }
            j++;
        }
        j=0;
        while (!inside && j<xflags[1]) {
            k=0;
            while (!inside && k<=yflags[0]) {
                tmp.x = current->x + (j+1)*incrx - k*xoffset;
                tmp.y = current->y - k*incry;
                if (EllePtInRegion(bnd,num_bnd_nodes,&tmp)) {
                    inside= true;
                    *rel_pos = tmp;
                }
                k++;
            }
            k=0;
            while (!inside && k<=yflags[1]) {
                tmp.x = current->x + (j+1)*incrx + k*xoffset;
                tmp.y = current->y + k*incry;
                if (EllePtInRegion(bnd,num_bnd_nodes,&tmp)) {
                    inside= true;
                    *rel_pos = tmp;
                }
                k++;
            }
            j++;
        }
        j=0;
        while (!inside && j<yflags[0]) {
            tmp.x = current->x - (j+1)*xoffset;
            tmp.y = current->y - (j+1)*incry;
            if (EllePtInRegion(bnd,num_bnd_nodes,&tmp)) {
                inside= true;
                *rel_pos = tmp;
            }
            j++;
        }
        j=0;
        while (!inside && j<yflags[1]) {
            tmp.x = current->x + (j+1)*xoffset;
            tmp.y = current->y + (j+1)*incry;
            if (EllePtInRegion(bnd,num_bnd_nodes,&tmp)) {
                inside= true;
                *rel_pos = tmp;
            }
            j++;
        }
    }
    return(inside);
}

void ElleNodeUnitXY(Coords *xy)
{
    double eps;
    double minx,maxx,miny,maxy,xp;
    double offset;
    CellData unitcell;

    /*
     * assume simple shear in x-direction
     * adjust y & x vals if above or below unit cell
     * adjust if x val outside unit cell
     */
    eps = 1.5e-6;
	/* basil uses float */
    /*eps = 1.0e-5;*/
    ElleCellBBox(&unitcell);
    maxy = unitcell.cellBBox[TOPLEFT].y;
    miny = unitcell.cellBBox[BASELEFT].y;
    maxx = unitcell.cellBBox[BASERIGHT].x;
    minx = unitcell.cellBBox[BASELEFT].x;
    offset = unitcell.cellBBox[TOPLEFT].x-unitcell.cellBBox[BASELEFT].x;
    while (xy->y > maxy+eps) {
        xy->y -= unitcell.ylength;
        xy->x -= unitcell.xoffset;
    }
    while (xy->y < miny-eps) {
        xy->y += unitcell.ylength;
        xy->x += unitcell.xoffset;
    }
    xp = (xy->y - miny)/unitcell.ylength * offset
                              + unitcell.cellBBox[BASELEFT].x;
    if (xy->x < (xp-eps)) {
        while (xy->x < xp) {
            xy->x += unitcell.xlength;
            xy->y += unitcell.yoffset;
        }
    }
    else {
        xp += unitcell.xlength;
        if (xy->x > (xp+eps)) {
            while (xy->x > xp) {
                xy->x -= unitcell.xlength;
                xy->y -= unitcell.yoffset;
            }
        }
    }
}
#if XY
#endif
void OnNodeError(int node, const char *message,int err)
{
    if (ElleNodeIsActive(node)) writeNodeLocalPolyFile(node);
    OnError(message,err);
}

void writeNodeLocalPolyFile(int index)
{
    Coords bbox[4];
    int found=0;
    int i, j, n, s, max;
	std::vector<int> :: iterator it, it2;
    int rgns[3],nbs[3];
    Coords origin, xy;
	std::ofstream outfile("topo_err.poly");

    ElleNodePosition(index,&origin);
    ElleFindLocalBBox(&origin,bbox,2);
    /*
     * store the nodes that are in the bbox
     */
	std::vector<int> nlist;
    max = ElleMaxNodes();
    for (i=0;i<max;i++) {
        if (ElleNodeIsActive(i)) {
            ElleNodePlotXY(i,&xy,&origin);
            if (EllePtInRegion(bbox,4,&xy)) 
                nlist.push_back(i);
        }
    }
    /*
     * write the nodes (elle id as boundary flag)
     * count the segments
     */
    outfile << nlist.size() << " 2" << " 1" << std::endl;
    for (n=s=0,it = nlist.begin();it != nlist.end(); it++) {
      ElleNodePlotXY((*it),&xy,&origin);
      outfile << n << ' ' << xy.x << ' ' << xy.y <<
               ' ' << (*it) << std::endl;
      n++;
      ElleNeighbourNodes((*it),nbs);
      for(j=0; j<3; j++)
          if (nbs[j]!=NO_NB && nbs[j]<(*it) &&
              find(nlist.begin(),nlist.end(),nbs[j])!=nlist.end())
              s++;
    }
    /*
     * write the segments (flynn id as boundary flag)
     */
    outfile << s << " 1" << std::endl;
    s=0;
    for (i=0,it = nlist.begin();it != nlist.end(); it++,i++) {
        ElleNeighbourNodes((*it),nbs);
        for(j=0; j<3; j++) {
            if (nbs[j]!=NO_NB && nbs[j]<(*it)){
                for (n=found=0,it2=nlist.begin();it2!=nlist.end()&&!found;
                                                            it2++,n++) {
                    if ((*it2)==nbs[j])  {
                        ElleRegions((*it),rgns);
                        outfile << s << ' ' << i << ' ' << n << ' ' <<
                                   rgns[j] << std::endl;
                        s++;
                        found=1;
                    }
                }
            }
        }
    }
    outfile << "0 0" << std::endl;
    outfile << "node " << index << std::endl;
    outfile.close();
}
