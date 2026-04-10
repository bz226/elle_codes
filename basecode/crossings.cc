 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <stdio.h>
#include <math.h>
#include "interface.h"
#include "nodes.h"
#include "check.h"
#include "crossings.h"
#include "error.h"
#include "polygon.h"
#include "convert.h"
#include "general.h"
#include "bflynns.h"
#include "runopts.h"
#include "flynnarray.h"
#include "log.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

void ElleReplaceTripleJ( Intersection *intersect, int *dj,int num);
void ElleReplaceDoubleJ(Intersection *intersect,int num);
void ElleCheckTJAngles(int node, Coords *newxy, Angle_data *ang, int *count);
void InitAngleData(Angle_data *ang);

int ElleCrossingsCheck(int node, Coords *pos_incr)
{
    int err=0;
    Coords xy;

    ElleNodePosition(node,&xy);
    xy.x += pos_incr->x;  xy.y+= pos_incr->y;
    /*ElleSetPosition(node,&xy);*/
    ElleUpdatePosition(node,pos_incr);
    err = ElleNodeTopologyCheck(node);
    return(err);
}
#if XY
int ElleCrossingsCheck(int node, Coords *pos_incr)
{
    int nbnodes[3], tmpnbs[3], i, j, angcnt=0, count=0, segcount=0;
    int nb, pivot, newdj[3];
    int rgn_checked[6],c=0,checked=0;
    int rearranged=0, triflynn=0, *list, move=1, start;
    double dist[3];
    double area1, area2;
    Coords newxy, currxy, nbxy, prevxy, checkpt;
    Angle_data ang[3];
    Intersection intersect[MAX_I_CNT], isectsegs[MAX_I_CNT], tmp;
    ERegion rgn, rgn2, nbrgns[3], chk_rgn;

    
#if XY
#endif

    for (i=0;i<MAX_I_CNT;i++) {
        InitIntersection(&intersect[i]);
        InitIntersection(&isectsegs[i]);
    }
    for (i=0;i<3;i++) InitAngleData(&ang[i]);

    ElleNodePosition(node,&currxy);
    newxy.x = currxy.x + pos_incr->x;
    newxy.y = currxy.y + pos_incr->y;
    /*
     * check whether the node movement causes an angle flip
     */
    ElleCheckAngles(node,&newxy,ang,&angcnt);

    if (!angcnt && ElleNodeIsTriple(node)) {
        ElleCheckTJAngles(node,&newxy,ang,&angcnt);
        if (angcnt==1) {
            ElleMoveLink(0,&ang[0],0);
            Log( 1, "moving TJ\n" );
            return(0);
        }
        else if (angcnt>1) {
            Log( 1, "Not moving TJ\n" );
            return(1);
        }
    }

    ElleNeighbourNodes(node,nbnodes);
    /*
     * for each neighbouring region
     * check whether the node moves across a segment
     */
    ElleRegions(node,nbrgns);
    for (i=0;i<3;i++) {
        if (nbrgns[i]!=NO_NB) {
            ElleNeighbourRegion(node,nbnodes[i],&rgn);
            if (j=ElleFindIntersection(node,NO_NB,nbrgns[i],&currxy,
                                     &newxy,&intersect[count]))
                count+=j;
            ElleNeighbourRegion(nbnodes[i],node,&rgn);
            if (j=ElleFindIntersection(node,nbnodes[i],rgn,&currxy,
                                     &newxy,&intersect[count]))
                count+=j;
        }
    }
    if (count==0) {
        for (i=0;i<3;i++) {  /* for each segment */
            if (nbnodes[i]!=NO_NB) {
                ElleNodePlotXY(nbnodes[i],&nbxy,&currxy);
                ElleNeighbourRegion(node,nbnodes[i],&rgn);
                if (j=ElleFindIntersection(node,nbnodes[i],rgn,&newxy,
                                        &nbxy, &intersect[count])) 
                    count+=j;
                ElleNeighbourRegion(nbnodes[i],node,&rgn);
                if (j=ElleFindIntersection(node,nbnodes[i],rgn,&newxy,
                                         &nbxy,&intersect[count]))
                    count+=j;
            }
        }
    }
    else if (ElleNodeIsTriple(node)) {
        checkpt.x = currxy.x + pos_incr->x*0.1;
        checkpt.y = currxy.y + pos_incr->y*0.1;
        chk_rgn=NO_NB;
        for (i=0,checked=0;i<3&&!checked;i++) {  /* for each region */
            if (nbnodes[i]!=NO_NB) {
                ElleNeighbourRegion(node,nbnodes[i],&rgn);
                if (EllePtInRegion(rgn,&checkpt)) {
                    checked=1;
                    chk_rgn=rgn;
                }
                else {
                    ElleNeighbourRegion(nbnodes[i],node,&rgn);
                    if (EllePtInRegion(rgn,&checkpt)) {
                        checked=1;
                        chk_rgn=rgn;
                    }
                }
            }
        }
        if (chk_rgn==NO_NB) OnError("no region to check",0);
        /*
         * do segments in order of rgn bnd
         */
        for (i=0;i<3;i++) {
            ElleNeighbourRegion(node,nbnodes[i],&rgn);
            ElleNeighbourRegion(nbnodes[i],node,&rgn2);
            if (rgn!=chk_rgn) {
                if (rgn2!=chk_rgn) tmpnbs[1]=nbnodes[i];
                else tmpnbs[2] = nbnodes[i];
            }
            else tmpnbs[0] = nbnodes[i];
        }
        for (i=0;i<3;i++) {  /* for each segment */
            if (tmpnbs[i]!=NO_NB) {
                ElleNodePlotXY(tmpnbs[i],&nbxy,&currxy);
                if (j=ElleFindIntersection(node,tmpnbs[i],chk_rgn,&newxy,
                                             &nbxy,&isectsegs[segcount]))
                        segcount+=j;
            }
        }
    }
    j = 0;
    if (count>1) {
        /*
         * find the closest intersection
         */
        for (i=0;i<count;i++)
            dist[i] = (intersect[i].xingpt.x - currxy.x) *
                      (intersect[i].xingpt.x - currxy.x) +
                      (intersect[i].xingpt.y - currxy.y) *
                      (intersect[i].xingpt.y - currxy.y);
        i=1;
        while (i<count) {
            if (dist[i] < dist[j]) j=i;
            i++;
        }
    }
    if (angcnt) {
        /*
         * angle pivot without vector or segment crossing
         */
        if (count) {
            /*if (ElleNodeIsDouble(intersect[j].pivot))*/
                /*ElleDeleteDoubleJ(intersect[j].pivot);*/
            /*else {*/
            /*
             * move link
             */
                ElleMoveLink(&intersect[j],&ang[0],&newxy);
                rearranged = 1;
            /*}*/
        }
        else {
        /*
         * triangular grain closing on itself?
         */
        triflynn=0;
        ElleNeighbourRegion(node,ang[0].pivot,&rgn);
         /*
          * can flynns with more than 5 nodes invert?
          * what is a test for any flynn inverting?
          */
        if (ElleFlynnNodeCount(rgn)<6) {
            ElleNeighbourRegion(ang[0].pivot,node,&rgn2);
            triflynn=1;
        }
        else {
            ElleNeighbourRegion(ang[0].pivot,node,&rgn);
            if (ElleFlynnNodeCount(rgn)<6) {
                ElleNeighbourRegion(node,ang[0].pivot,&rgn2);
                triflynn=1;
            }
        }
        if (triflynn) {
            area1 = ElleRegionArea(rgn);
            ElleNodePrevPosition(node,&prevxy);
            ElleSetPosition(node,&newxy);
            area2 = ElleRegionArea(rgn);
            ElleSetPosition(node,&currxy);
            ElleSetPrevPosition(node,&prevxy);
            if (area2 < area1 || area1 < 0) {
                /*
                 * flynn inverting
                 */
                ElleMergeFlynnsNoCheck(rgn2,rgn);
                count=0;
            }
        }
        else {
         /*
          * angle very small and closing but not quite intersect
          * assume it has closed
          */
            ElleMoveLink(0,&ang[0],0);
        }
        }
#if XY
#endif
    }
    else if (count>0) {
        if (ElleNodeIsTriple(node)) {
            if (ElleNodeIsActive(node)) writeNodeLocalPolyFile(node);
            if (segcount) {
                ElleReplaceTripleJ(isectsegs,newdj,segcount);
#ifdef DEBUG
                if (ElleNodeIsActive(newdj[0])) writeNodeLocalPolyFile(newdj[0]);
#endif
            }
        }
        else  {
            if (ElleNodeIsActive(node)) writeNodeLocalPolyFile(node);
            ElleReplaceDoubleJ(&intersect[j],count);
            if (ElleNodeIsActive(node)) writeNodeLocalPolyFile(node);
        }
    }
    else {
       /*
        * move the node
        */
        ElleUpdatePosition(node,pos_incr);
    }
    if (rearranged) {
        if (ElleNodeIsDouble(node)) ElleCheckDoubleJ(node);
        else if (ElleNodeIsTriple(node)) ElleCheckTripleJ(node);
        if (ElleNodeIsActive(node))    writeNodeLocalPolyFile(node);
    }
    return(0);
}
#endif

void ElleMoveLink(Intersection *intersect, Angle_data *ang,
                  Coords *newxy)
{
    int node,nnew,i;
    int curr,end,nbnodes[3];
    Coords midpt, xy;
    ERegion rgn1,rgn2,rgn,oldrgn,nbrgn,tmp[3];
    /*
     * insert a new DJ in the intersected segment ( at
     * crossing point ) and change the boundaries so 
     * the node is joined to the new node instead of 
     * the pivotted node
     */
    if (intersect) node = intersect->node1a;
    else node = ang->node;
    ElleNeighbourRegion(node,ang->pivot,&rgn1);
    ElleNeighbourRegion(ang->pivot,node,&rgn2);
    ElleNeighbourRegion(ang->pivot,ang->nb,&nbrgn);
    if (intersect) {
        ElleInsertDoubleJ(intersect->node1b,intersect->node2b,&nnew,0.5);
        ElleSetPosition(nnew,&intersect->xingpt);
        ElleSetPrevPosition(nnew,&intersect->xingpt);
    }
    else {
        ElleInsertDoubleJ(ang->pivot,ang->nb,&nnew,0.5);
    }
    ElleNeighbourRegion(node,ang->pivot,&rgn1);
    /*if (EllePtInRegion(rgn1,newxy)) {*/
    /*if (rgn1==nbrgn && !ElleNodeIsDouble(ang->pivot)) {*/
    if (rgn1==nbrgn) {
        curr = ang->pivot;
        end = nnew;
        rgn = rgn2;
        oldrgn = rgn1;
    }
    else {
        curr = nnew;
        end = ang->pivot;
        rgn = rgn1;
        oldrgn = rgn2;
    }
    if ((i = ElleFindNbIndex(ang->pivot,node))==NO_NB)
        OnNodeError(node,"ElleMoveLink",NONB_ERR);
    /*
     * nnew is a new node so this will be the third entry and
     * loop below will find the other nb with rgn2 before node
     */
    ElleSetNeighbour(nnew,NO_NB,node,&rgn2);
    ElleSetNeighbour(node,i,nnew,0);
    if ((i = ElleFindNbIndex(node,ang->pivot))==NO_NB)
        OnNodeError(node,"ElleMoveLink",NONB_ERR);
    ElleClearNeighbour(ang->pivot,i);
    if (ElleNodeIsSingle(ang->pivot)) {
        ElleDeleteSingleJ(ang->pivot);
        if (end==ang->pivot) end = ang->nb;
        else curr = ang->nb;
    }
    if (ElleNodeIsTriple(nnew)) {
        while (curr!=end) {
            ElleNeighbourNodes(curr,nbnodes);
            if ((i=ElleFindBndIndex(curr,oldrgn))!=NO_NB){
                ElleSetRegionEntry(curr,i,rgn);
                ElleUpdateFirstNodes(curr,nnew, oldrgn);
                curr = nbnodes[i];
            }
        }
        if (end==ang->pivot) ElleUpdateFirstNodes(end,nnew,oldrgn);
    }
}

void ElleReplaceTripleJ( Intersection *intersect,int *dj,int num)
{
    /*
     * assumes they are in order for the same chk_rgn
     */
    int node,  nb, i, index, nbindex;
    float frac = 0.5;
    double eps;
    ERegion rgn, nbrgn, newrgn;

    eps = 1.0e-6;
    node = intersect[0].node1a;
    frac = 1.0/num;
    for (i=0;i<num;i++) {
        /*
         * replace nb - node with nb - newnode at xingpt
         */
        nb = intersect[i].node2a;
        if ((index = ElleFindNbIndex(node,nb))==NO_NB)
            OnNodeError(node,"ReplaceTripleJ",NONB_ERR);
        ElleNeighbourRegion(node,nb,&rgn);
        ElleNeighbourRegion(nb,node,&nbrgn);
        if ((nbindex = ElleFindNbIndex(nb,node))==NO_NB)
            OnNodeError(node,"ReplaceTripleJ",NONB_ERR);
        ElleInsertDoubleJ(intersect[i].node1b,intersect[i].node2b,
                                  &dj[i],1.0/(num-i+1));
        ElleSetRegionEntry(dj[i],ElleFindNbIndex(intersect[i].node2b,dj[i])
                                               ,nbrgn);
        ElleSetNeighbour(nb,index,dj[i],0);
        ElleSetNeighbour(dj[i],NO_NB,nb,&rgn);
        if (i==num-1) {
            ElleNewGrain(dj[i],nb,intersect[i].chk_rgn,&newrgn);
            ElleSetFlynnFirstNode(intersect[i].chk_rgn,dj[0]);
            ElleSetFlynnFirstNode(newrgn,dj[i]);
        }
        else {
            if (i<num-1 && intersect[i+1].node1b==intersect[0].node1b &&
                           intersect[i+1].node2b==intersect[0].node2b)
                intersect[i+1].node1b=dj[i];
        }
        ElleClearNeighbour(node,nbindex);
    }
}


void ElleReplaceDoubleJ( Intersection *intersect,int num)
{
    int node, nbnodes[3], nnew, i;
    double eps;
    double diffx, diffy;
    Coords midpt, xy;
    ERegion rgn, nbrgns[3], newrgn;

    eps = 1.0e-6;
    node = intersect->node1a;
    /*ElleNodePosition(node,&xy);*/
    ElleNodePlotXY(node,&xy,&(intersect->xingpt));
    /*
     * if node is very close to crossing point, do nothing
     */
    diffx = (double)xy.x - intersect->xingpt.x;
    diffy = (double)xy.y - intersect->xingpt.y;
    if (fabs(diffx)<eps && fabs(diffy)<eps) {
        sprintf(logbuf,"Trouble %d seg %d %d\n",node,
                             intersect->node1b,intersect->node2b);
        Log(1,(const char *)logbuf);
    }
    /*else {*/
        /*
         * find the regions on either side of the crossed segment
         * intersect->chk_rgn is intersect->node1b to intersect->node2b
         */
        ElleNeighbourRegion(intersect->node2b,intersect->node1b,&rgn);
        midpt.x = xy.x - (xy.x - intersect->xingpt.x)*0.5;
        midpt.y = xy.y - (xy.y - intersect->xingpt.y)*0.5;
        if (EllePtInRegion(intersect->chk_rgn,&midpt))
            rgn = intersect->chk_rgn;
        ElleInsertDoubleJ(intersect->node1b,intersect->node2b,&nnew,0.5);
        ElleSetPosition(nnew,&(intersect->xingpt));
        ElleSetPrevPosition(nnew,&(intersect->xingpt));
        ElleNewGrain(node,nnew,rgn,&newrgn);
        ElleSwitchTripleNodesForced(node,nnew);
    /*}*/
}

int ElleFindIntersection(int node1, int nb, ERegion rgn, Coords *strt_pos,
                          Coords *end_pos, Intersection *intersect)
{
    char msg[80];
    int i, j, k, found = 0, inb, start=NO_NB, num_segs, count=0;
    int *id, num_nodes;
    Coords xy,prev,last;
    double x1, y1, x2, y2, x3, y3, x4, y4, x, y;

    sprintf(msg,"%s %d %d Region=%d ",
                    "ElleFindIntersection", node1, nb, rgn);
    /*
     * get the list of nodes - work on lowest child level
     * set x1 y1(node pos) x2 y2(node prev pos) x3 y3(start)
     */
    prev = *strt_pos;
    ElleFlynnNodes(rgn,&id,&num_nodes);
    /*ElleUnitNodes(rgn,&id,&num_nodes);*/
    /*ElleNodePosition(node1,&prev);*/
    /*ElleNodePlotXY(node1,&xy,&prev);*/
    x1 = end_pos->x;
    y1 = end_pos->y;
    x2 = prev.x;
    y2 = prev.y;
    /*
     * find the node for the first segment to be checked
     * node1---nb---start   or  nb---node1---start
     */
    i=0; start = node1;
    last = prev;
    while (i<num_nodes && id[i]!=start) i++;
    i = (i+1) % num_nodes;
    if (nb!=NO_NB && id[i] == nb) i = (i+1) % num_nodes;
    inb = (i+1) % num_nodes;
    ElleNodePlotXY(id[i],&xy,&last);
    x3 = xy.x;
    y3 = xy.y;
    last=xy;
    num_segs = num_nodes-2;
    if (nb!=NO_NB)
        num_segs = num_nodes-3; /* don't check requested segment
                                 * or it's neighbour segments */
    /*
     * check this region for intersections with x1,y1---x2,y2
     */
    for (j=0;j<num_segs /*&& !found*/;j++) {
        ElleNodePlotXY(id[inb],&xy,&prev);
        x4 = xy.x;
        y4 = xy.y;
        prev=xy;
        k=lines_intersect(x1,y1,x2,y2,x3,y3,x4,y4,&x,&y);
        /*if (k==2) printf("colinear\n");*/
        if (k==1) {
#ifdef DEBUG
           printf("intersect %d %d %d %d region %d\n",
                   node1,nb,id[i],id[inb],rgn);
#endif
           /* set values */ 
           intersect[count].node1a = node1;
           intersect[count].node2a = nb;
           intersect[count].node1b = id[i];
           intersect[count].node2b = id[inb];
           intersect[count].xingpt.x = x;
           intersect[count].xingpt.y = y;
           intersect[count].chk_rgn = rgn;
           found = 1;
           if (count<MAX_I_CNT-1) count++;
           else OnError("FindIntersection",RANGE_ERR);
        }
        i = inb;
        inb = (i+1) % num_nodes;
        x3 = x4; y3 = y4;
    }
    if (id) free(id);
    return(count);
}

void ElleCheckAngles(int node, Coords *newxy, Angle_data *ang, int *count)
{
    int i,j;
    int nbnodes[3], nbnbnodes[3];
    int index=0;
    double currang, prevang, min, curr;
    float eps, angeps = 0.0087; /* 1deg */
    double sep1, sep2;
    Coords tmp2, xy, prev;
    /*
     * compare current angle with that before node moved
     * if sign changes, set pvt and nb and angle
     */

    eps = 1.0e-6;
    *count = 0;
    ElleNodePosition(node,&prev);
    ElleNeighbourNodes(node,nbnodes);
    for (i=0;i<3 /*&& *count==0*/;i++) {
        if (nbnodes[i]!=NO_NB) {
            ElleNeighbourNodes(nbnodes[i],nbnbnodes);
            ElleNodePlotXY(nbnodes[i],&xy,&prev);
            for (j=0;j<3;j++) {
                if (nbnbnodes[j]!=NO_NB && nbnbnodes[j]!=node) {
                    ElleNodePlotXY(nbnbnodes[j],&tmp2,&prev);
                    if (angle( xy.x,xy.y,tmp2.x,tmp2.y,newxy->x,newxy->y,
                                                             &currang ))
                        OnNodeError(node,"angle error",0);
                    if (angle( xy.x,xy.y,tmp2.x,tmp2.y,prev.x,prev.y,
                                                             &prevang ))
                        OnNodeError(node,"angle error",0);
                    if ((fabs(currang-prevang))<PI &&
                                 ((currang < -eps && prevang > eps) ||
                                  (currang > eps && prevang < -eps)) ||
                                  (fabs(currang) < angeps && 
                                   fabs(currang) < fabs(prevang)) ) {
#ifdef DEBUG
                        printf("ang = %lf, prev = %lf\n",currang,prevang);
#endif
                        ang[*count].pivot = nbnodes[i];
                        ang[*count].nb = nbnbnodes[j];
                        ang[*count].node = node;
                        ang[*count].prev_angle = prevang;
                        (*count)++;
                    }
                }
            }
        }
    }
    if (*count > 1) {
        min = fabs(ang[0].prev_angle);
        index = 0;
        i=1;
        while (i < *count) {
            if (ang[i].pivot!=ang[0].pivot) {
                sep1 = ElleNodeSeparation(node,ang[i].pivot);
                sep2 = ElleNodeSeparation(node,ang[0].pivot);
                if (sep1 < sep2) index = i;
            }
            else {
                curr = fabs(ang[i].prev_angle);
                if (curr < min) {
                    min = curr;
                    index = i;
                }
            }
            i++;
        }
        /*
         * copy the data for the smallest angle or closest pivot
         *  to index 0
         */
        if (index!=0) ang[0] = ang[index];
    }
}

void ElleCheckTJAngles(int node, Coords *newxy, Angle_data *ang, int *count)
{
    int i,j,k;
    int nbnodes[3], nbnbnodes[3];
    int found=0;
    double currang, prevang, min, curr;
    double eps;
    double angeps, diff;
    angeps = 0.0087; /* 1deg */
    Coords tmp2, xy, prev;
    ERegion rgn, nbrgns[3];
    /*
     * compare current angle with that before node moved
     * if sign changes, set pvt and nb and angle
     */

    eps = 1.0e-6;
    *count = 0;
    /*ElleNodePrevPosition(node,&prev);*/
    /*ElleNodePlotXY(node,&tmp1,&prev);*/
    ElleNodePosition(node,&prev);
    /*tmp1.x = prev.x + incr->x;*/
    /*tmp1.y = prev.y + incr->y;*/
    ElleNeighbourNodes(node,nbnodes);
    for (i=0;i<3 && *count==0;i++) {
            ElleNodePlotXY(nbnodes[i],&xy,&prev);
            ElleNeighbourRegion(node,nbnodes[i],&rgn);
            found = 0;
            for (j=0;j<3 && !found;j++) {
                ElleNeighbourNodes(nbnodes[j],nbnbnodes);
                ElleRegions(nbnodes[j],nbrgns);
                k=0;
                while (k<3 && (nbnbnodes[k]==NO_NB || nbnbnodes[k]!=node ||
                            nbrgns[k]!=rgn)) k++;
                if (k<3) {
                    found = 1;
                    ElleNodePlotXY(nbnodes[j],&tmp2,&prev);
                    if (angle( newxy->x,newxy->y,xy.x,xy.y,tmp2.x,tmp2.y,
                                                             &currang ))
                        OnNodeError(node,"angle error",0);
                    if (angle( prev.x,prev.y,xy.x,xy.y,tmp2.x,tmp2.y,
                                                             &prevang ))
                        OnNodeError(node,"angle error",0);
                    diff = fabs(currang-prevang);
                    /*if ((fabs(currang-prevang))<PI &&*/
                    if (diff<PI &&
                             ((currang < -eps && prevang > eps) ||
                                  (currang > eps && prevang < -eps) ||
                                   (fabs(currang) < angeps && 
                                   fabs(currang) < fabs(prevang))) ) {
#ifdef DEBUG
                        printf("ang = %lf, prev = %lf\n",currang,prevang);
#endif

                        ang[*count].pivot = node;
                        ang[*count].nb = nbnodes[j]; /* ?? */
                        ang[*count].node = nbnodes[i]; /* ?? */
                        ang[*count].prev_angle = prevang;
                        (*count)++;
                    }
                }
            }
    }
}

int ElleRegionIsSimple(ERegion poly, Intersection *intersect)
{
    int i, j, k, *id, num_nodes, seg, next_seg, next_index, count;
    int found=0;
    Coords xy,prev,*pos=0;
    Coords rect[4], testrect[4];

    rect[0].x = rect[3].x = -(EllemaxNodeSep()*4.0);
    rect[0].y = rect[1].y = -(EllemaxNodeSep()*4.0);
    rect[1].x = rect[2].x = EllemaxNodeSep()*4.0;
    rect[2].y = rect[3].y = EllemaxNodeSep()*4.0;
    ElleFlynnNodes(poly,&id,&num_nodes);
    ElleNodePosition(id[0],&prev);
    if ((pos = new Coords[num_nodes])==0)
        OnError("ElleRegionIsSimple",MALLOC_ERR);
    
    if (num_nodes>3) { //can't intersect unless more than 3 sides
        pos[0]= prev; 
        for (j=1;j<num_nodes;j++) {
            ElleNodePlotXY(id[j],&pos[j],&prev);
            prev = pos[j];
        }
        count = num_nodes-3; //don't check seg or its nb segs
        for (seg=0;seg<num_nodes-2 && !found;seg++) {
            for (i=0;i<4;i++) {
                testrect[i].x = pos[seg].x+rect[i].x;
                testrect[i].y = pos[seg].y+rect[i].y;
            }
            for (next_seg=seg+2,j=0; next_seg<num_nodes && j<count && !found;
                                                   next_seg++,j++) {
                next_index = (next_seg+1)%num_nodes;
                k=0;
                if (EllePtInRect(testrect,4,&pos[next_seg]))
                    k=lines_intersect(pos[seg].x,pos[seg].y,
                                      pos[seg+1].x,pos[seg+1].y,
                                      pos[next_seg].x,pos[next_seg].y,
                                      pos[next_index].x,pos[next_index].y,
                                      &xy.x,&xy.y);
                /*if (k==2) printf("colinear\n");*/
                if (k==1) {
#ifdef DEBUG
           printf("intersect %d %d %d %d region %d\n",
                   node1,nb,id[i],id[inb],rgn);
#endif
                    /* set values */ 
                    intersect->node1a = id[seg];
                    intersect->node2a = id[seg+1];
                    intersect->node1b = id[next_seg];
                    intersect->node2b = id[next_index];
                    intersect->xingpt.x = xy.x;
                    intersect->xingpt.y = xy.y;
                    intersect->chk_rgn = poly;
                    found = 1;
                }
            }
        }
        if (id) free(id);
        if (pos) delete[] pos;
    }
    return(!found);
}

int ElleNodeTopologyCheck(int node)
{
    int i, nbs[3], fnew, curr, err=0;
    int replaced=1, small_flynn=NO_NB;
    int child1, child2, *ids, num, indxa, indxb;
    int diff, splitstart, splitend,rgna,rgnb;
    Coords dir[3], xy;
    set_int nbflynnset;
    set_int::iterator its,its_end;
    std::list<int>::iterator it;
    Intersection isect;

    double min_area = ElleminNodeSep() * ElleminNodeSep() * SIN60 * 0.25;

    ElleRegions(node,nbs);
    for (i=0;i<3;i++) {
        if (nbs[i]!=NO_NB) {
            nbflynnset.insert(nbs[i]);
// only do the immediate nb regions
#if XY
            std::list<int> nbflynns;
            ElleFlynnNbRegions(nbs[i], nbflynns);
            for (it=nbflynns.begin();it!=nbflynns.end();it++)
                nbflynnset.insert(*it);
#endif
         }
     }
     its=nbflynnset.begin();
     while (its!=nbflynnset.end()) {
         curr = *its;
         if (ElleFlynnIsActive(curr) &&
                   !ElleRegionIsSimple(curr,&isect)) {
             sprintf(logbuf," %d not simple, node %d\n",
                                                 curr,isect.node1a);
             Log(2,(const char *)logbuf);
             ElleNeighbourRegion(isect.node2a,isect.node1a,&rgna);
             ElleNeighbourRegion(isect.node2b,isect.node1b,&rgnb);
             /*
              * don't split the flynn if it is pinching off a tiny
              * area (3 nodes) - it will be deleted anyway
              */
             if (rgna==rgnb &&
                (ElleFindNbIndex(isect.node2b,isect.node1a)==NO_NB ||
                  ElleNodeIsTriple(isect.node2b)||
                   ElleNodeIsTriple(isect.node1a))&&
                (ElleFindNbIndex(isect.node2a,isect.node1b)==NO_NB ||
                   ElleNodeIsTriple(isect.node2a)||
                   ElleNodeIsTriple(isect.node1b))) {
#if XY
                 ElleFlynnNodes(rgna,&ids,&num);
                 indxa=0;
                 while (indxa<num && ids[indxa]!=isect.node1a)
                         indxa++;
                 indxb=0;
                 while (indxb<num && ids[indxb]!=isect.node1b)
                         indxb++;
                 // split small part
                 //diff = abs(indxa-indxb);
                 // split large part
                 diff = num-abs(indxa-indxb);
                 splitstart = ids[(indxa-diff/2)%num];
                 splitend = ids[((num-diff)/2+indxa)%num];
                 ElleNodePosition(splitend,&dir[0]);
                 ElleNodePlotXY(splitstart,&xy,&dir[0]);
                 dir[0].x-=xy.x; dir[0].y-=xy.y;
                 rotate_coords(dir[0].x,dir[0].y,0,0,
                               &dir[1].x,&dir[1].y,PI_2);
                 rotate_coords(dir[0].x,dir[0].y,0,0,
                               &dir[2].x,&dir[2].y,-PI_2);
                 for (i=0,err=1;i<3 && err; i++) {
                     err=ElleDirectSplitWrappingFlynn(rgna,splitstart,splitend,
                               &child1, &child2, &dir[i]);
                 }
                 if (err)
                     OnError("Could not split wrapping flynn",0);
                 else {
                     EllePromoteFlynn(child1);
                     EllePromoteFlynn(child2);
                     ElleRemoveFlynn(rgna);
                     nbflynnset.insert(child1);
                     nbflynnset.insert(child2);
                     nbflynnset.erase(nbflynnset.find(rgna));
                }
#endif
             sprintf(logbuf," not splitting wrapping flynn %d \n",rgna);
             Log(2,(const char *)logbuf);
             }
             if (ElleReplaceIntersection(&isect,&fnew)==0) {
                 replaced=1;
                 small_flynn=NO_NB;
                 if (fnew!=NO_NB) {
                     nbflynnset.insert(fnew);
                     its = nbflynnset.find(curr);
                     if (fabs(ElleFlynnArea(fnew))<min_area)
                         small_flynn=fnew;
                 }
                 if (small_flynn!=NO_NB) {
                     std::list<int> nbflynns;
                     ElleFlynnNbRegions(small_flynn,nbflynns);
                     // nbs are in order of common bnd length
                     // try for one that allows merge
                     it=nbflynns.begin();
                     while (it!=nbflynns.end() && 
                         !ElleFlynnAttributesAllowMerge(small_flynn,*it))it++;
                     if (it==nbflynns.end()) it = nbflynns.begin();
                     ElleMergeFlynnsNoCheck(*it,small_flynn);
                     sprintf(logbuf,"merged %d into %d\n",
                                           small_flynn,*it);
                     Log(2,(const char *)logbuf);
                     nbflynnset.erase(nbflynnset.find(small_flynn));
                     its = nbflynnset.find(curr);
                 }
             }
             else
                 OnError("TopologyCheck problem",0);
             its=nbflynnset.begin();
        }
        else if (its!=nbflynnset.end()) its++;
    }
    its=nbflynnset.begin();
    while (its!=nbflynnset.end()) {
        // deleting a double can cause a flynn merge so check flynn is
        // still active or run throught the list and erase inactive
        // flynns
        if (ElleFlynnIsActive(*its) && (ElleRegionArea(*its)<min_area)) {
             std::list<int> nbflynns;
             ElleFlynnNbRegions(*its,nbflynns);
             // nbs are in order of common bnd length
             // try for one that allows merge
             it=nbflynns.begin();
             while (it!=nbflynns.end() && 
                 !ElleFlynnAttributesAllowMerge(*its,*it))it++;
#if XY
// Need to check if nb flynns are smaller?
             std::list<int>::iterator itmp=nbflynns.begin();
             while (itmp!=nbflynns.end() && ElleRegionArea(*itmp)<area) itmp++;
             if (itmp==nbflynns.end()) {
                 sprintf(logbuf,"all neighbour flynns smaller than %d\n",
                                           *its);
                 Log(2,(const char *)logbuf);
            }
#endif
             if (it==nbflynns.end()) it = nbflynns.begin();
             if (ElleMergeFlynnsNoCheck(*it,*its)==0) {
                 sprintf(logbuf,"merged %d into %d\n",
                                           *its,*it);
                 Log(2,(const char *)logbuf);
            }
        }
        its++;
    }
    return(err);
}

int ElleFlynnTopologyCheck()
{

    int i, j, max, nbs[3], fnew, curr, err=0;
    int replaced=1, small_flynn=NO_NB;
    int child1, child2, *ids, num, indxa, indxb;
    int diff, splitstart, splitend,rgna,rgnb;
    Coords dir[3], xy;
    set_int nbflynnset;
    set_int::iterator its,its_end;
    std::list<int>::iterator it;
    Intersection isect;
                                                                            
    double min_area = ElleminNodeSep() * ElleminNodeSep() * SIN60 * 0.5;
                                                                            
    for (i=0;i<ElleMaxFlynns();i++) 
        if (ElleFlynnIsActive(i)) nbflynnset.insert(i);
     its=nbflynnset.begin();
     while (its!=nbflynnset.end()) {
         curr = *its;
         if (ElleFlynnIsActive(curr) &&
                   !ElleRegionIsSimple(curr,&isect)) {
             sprintf(logbuf," %d not simple, node %d\n",
                                                 curr,isect.node1a);
             Log(2,(const char *)logbuf);
             ElleNeighbourRegion(isect.node2a,isect.node1a,&rgna);
             ElleNeighbourRegion(isect.node2b,isect.node1b,&rgnb);
             /*
              * don't split the flynn if it is pinching off a tiny
              * area (3 nodes) - it will be deleted anyway
              */
             if (rgna==rgnb &&
                (ElleFindNbIndex(isect.node2b,isect.node1a)==NO_NB ||
                  ElleNodeIsTriple(isect.node2b)||
                   ElleNodeIsTriple(isect.node1a))&&
                (ElleFindNbIndex(isect.node2a,isect.node1b)==NO_NB ||
                   ElleNodeIsTriple(isect.node2a)||
                   ElleNodeIsTriple(isect.node1b))) {
#if XY
                 ElleFlynnNodes(rgna,&ids,&num);
                 indxa=0;
                 while (indxa<num && ids[indxa]!=isect.node1a)
                         indxa++;
                 indxb=0;
                 while (indxb<num && ids[indxb]!=isect.node1b)
                         indxb++;
                 // split small part
                 //diff = abs(indxa-indxb);
                 // split large part
                 diff = num-abs(indxa-indxb);
                 splitstart = ids[(diff/2+indxa)%num];
                 splitend = ids[((num-diff)/2+indxa)%num];
                 ElleNodePosition(splitend,&dir[0]);
                 ElleNodePlotXY(splitstart,&xy,&dir[0]);
                 dir[0].x-=xy.x; dir[0].y-=xy.y;
                 rotate_coords(dir[0].x,dir[0].y,0,0,
                               &dir[1].x,&dir[1].y,PI_2);
                 rotate_coords(dir[0].x,dir[0].y,0,0,
                               &dir[2].x,&dir[2].y,-PI_2);
                 for (i=0,err=1;i<3 && err; i++) {
                     err=ElleDirectSplitWrappingFlynn(rgna,splitstart,splitend,
                               &child1, &child2, &dir[i]);
                 }
                 if (err)
                     OnError("Could not split wrapping flynn",0);
                 else {
                     EllePromoteFlynn(child1);
                     EllePromoteFlynn(child2);
                     ElleRemoveFlynn(rgna);
                     nbflynnset.insert(child1);
                     nbflynnset.insert(child2);
                     nbflynnset.erase(nbflynnset.find(rgna));
                 }
#endif
             sprintf(logbuf," not splitting wrapping flynn %d \n",rgna);
             Log(2,(const char *)logbuf);
             }
             if (ElleReplaceIntersection(&isect,&fnew)==0) {
                 replaced=1;
                 small_flynn=NO_NB;
                 if (fnew!=NO_NB) {
                     nbflynnset.insert(fnew);
                     its = nbflynnset.find(curr);
                 }
                 if (small_flynn!=NO_NB) {
                     std::list<int> nbflynns;
                     ElleFlynnNbRegions(small_flynn,nbflynns);
                     // nbs are in order of common bnd length
                     // try for one that allows merge
                     it=nbflynns.begin();
                     while (it!=nbflynns.end() && 
                         !ElleFlynnAttributesAllowMerge(small_flynn,*it))it++;
                     if (it==nbflynns.end()) it = nbflynns.begin();
                     ElleMergeFlynnsNoCheck(*it,small_flynn);
                     nbflynnset.erase(nbflynnset.find(small_flynn));
                     its = nbflynnset.find(curr);
                 }
             }
             else
                 OnError("ElleFlynnTopologyCheck problem",0);
             its=nbflynnset.begin();
        }
        else if (its!=nbflynnset.end()) its++;
    }
    its=nbflynnset.begin();
    while (its!=nbflynnset.end()) {
        if (ElleRegionArea(*its)<min_area) {
             std::list<int> nbflynns;
             ElleFlynnNbRegions(*its,nbflynns);
             // nbs are in order of common bnd length
             // try for one that allows merge
             it=nbflynns.begin();
             while (it!=nbflynns.end() && 
                 !ElleFlynnAttributesAllowMerge(*its,*it))it++;
             if (it==nbflynns.end()) it = nbflynns.begin();
             if (ElleMergeFlynnsNoCheck(*it,*its)==0) {
                 sprintf(logbuf,"merged %d into %d\n",
                                       *its,*it);
                 Log(2,(const char *)logbuf);
             }
        }
        its++;
    }
    max = ElleMaxNodes();
    for (j=0;j<max;j++)
    {
        if (ElleNodeIsActive(j))
            if (ElleNodeIsDouble(j)) ElleCheckDoubleJ(j);
                                                                                
    }
    return(0);
}

int ElleReplaceIntersection( Intersection *intersect, int *rgn_new )
{
    int err=0;
    int newa_in_n2a, n2a_in_newa, newb_in_n2b, n2b_in_newb;
    int newb_in_n1b, n1b_in_newb;
    int  rgna, rgnb, newa, newb, tmpa, tmpb;
    double len2b1a=0.0, len2a1b=0.0;
    bool bnd2a1b=false, bnd2b1a=false;

    *rgn_new=NO_NB;
    ElleNeighbourRegion(intersect->node2a,intersect->node1a,&rgna);
    ElleNeighbourRegion(intersect->node2b,intersect->node1b,&rgnb);
    ElleInsertDoubleJAtPoint(intersect->node1a,intersect->node2a,&newa,
                             &(intersect->xingpt));
    ElleInsertDoubleJAtPoint(intersect->node1b,intersect->node2b,&newb,
                             &(intersect->xingpt));
    newa_in_n2a = ElleFindNbIndex(newa,intersect->node2a);
    n2a_in_newa = ElleFindNbIndex(intersect->node2a,newa);
    newb_in_n2b = ElleFindNbIndex(newb,intersect->node2b);
    n2b_in_newb = ElleFindNbIndex(intersect->node2b,newb);
    newb_in_n1b = ElleFindNbIndex(newb,intersect->node1b);
    n1b_in_newb = ElleFindNbIndex(intersect->node1b,newb);
    ElleSetNeighbour(newb,n2b_in_newb,intersect->node2a,0);
    ElleSetNeighbour(intersect->node2a, newa_in_n2a, newb,0);
    ElleSetNeighbour(newa,n2a_in_newa,intersect->node2b,0);
    ElleSetNeighbour(intersect->node2b, newb_in_n2b, newa,0);
    if (ElleFindNbIndex(intersect->node2a,intersect->node1b)!=NO_NB) {
        len2a1b = ElleNodeSeparation(intersect->node2a,intersect->node1b);
            bnd2a1b = true;
    }
    if (ElleFindNbIndex(intersect->node2b,intersect->node1a)!=NO_NB) {
        len2b1a = ElleNodeSeparation(intersect->node2b,intersect->node1a);
            bnd2b1a = true;
    }
    if (bnd2b1a && bnd2a1b) {
        if (len2a1b>len2b1a) bnd2a1b = false;
        else bnd2b1a = false;
    }
    
    if (bnd2a1b) {
        ElleJoinTwoDoubleJ(newa, newb, rgnb, rgna);
        ElleNeighbourRegion(newb,intersect->node2a,&tmpb);
        tmpa=intersect->node2b;
        ResetRegionEntries(newb,&tmpa,intersect->node1b,
                            intersect->chk_rgn,rgnb);
        ElleClearNeighbour(intersect->node1b,
                  ElleFindNbIndex(newb,intersect->node1b));
        ElleClearNeighbour(newb,
                  ElleFindNbIndex(intersect->node1b,newb));
        ElleSetFlynnFirstNode(intersect->chk_rgn, newa);
        if (ElleNodeIsSingle(intersect->node1b)) {
            ElleDeleteSingleJ(intersect->node1b);
            if (ElleNodeIsDouble(intersect->node2a))
                ElleCheckDoubleJ(intersect->node2a);
        }
        if (ElleNodeIsActive(newb) && ElleNodeIsDouble(newb))
            ElleDeleteDoubleJNoCheck(newb);
#if XY
#endif
    }
    else if (bnd2b1a) {
        ElleJoinTwoDoubleJ(newa, newb, rgnb, rgna);
        ElleNeighbourRegion(newa,intersect->node2b,&tmpb);
        tmpa=intersect->node2a;
        ResetRegionEntries(newa,&tmpa,intersect->node1a,
                            intersect->chk_rgn,rgna);
        ElleClearNeighbour(newa,
                  ElleFindNbIndex(intersect->node1a,newa));
        ElleClearNeighbour(intersect->node1a,
                  ElleFindNbIndex(newa,intersect->node1a));
        ElleSetFlynnFirstNode(intersect->chk_rgn, newb);
        if (ElleNodeIsSingle(intersect->node1a)) {
            ElleDeleteSingleJ(intersect->node1a);
            if (ElleNodeIsDouble(intersect->node2b))
                ElleCheckDoubleJ(intersect->node2b);
        }
        if (ElleNodeIsActive(newa) && ElleNodeIsDouble(newa))
            ElleDeleteDoubleJNoCheck(newa);
    }
    else {
        ElleNeighbourRegion(newb,intersect->node1b, &tmpa);
        ElleNeighbourRegion(intersect->node2a,newb, &tmpb);
        ElleJoinTwoDoubleJ(newa, newb, tmpa, tmpb);
        ElleNewGrain(newb,intersect->node1b,intersect->chk_rgn,rgn_new);
        ElleUpdateFirstNodes(newb, newa, intersect->chk_rgn);
    }
    return(err);
}

void InitIntersection(Intersection *intersect)
{
    intersect->node1a=intersect->node2a=intersect->node1b=
                intersect->node2b = NO_NB;
    intersect->xingpt.x = intersect->xingpt.y = 0.0;
    intersect->chk_rgn = NO_NB;
    intersect->pivot = NO_NB;
}

void InitAngleData(Angle_data *ang)
{
    ang->pivot = NO_NB;
    ang->nb = NO_NB;
    ang->node = NO_NB;
    ang->prev_angle = 0;
}

#if XY
void ElleReplaceTripleJ( Intersection *intersect,int *dj,int num)
{
    int node, nbnodes[3], nb, i;
    double eps;
    double diffx, diffy;
    Coords midpt, xy;
    ERegion rgn, nbrgns[3], newrgn;

    eps = 1.0e-6;
    node = intersect->node1a;
    /*ElleNodePosition(node,&xy);*/
    ElleNodePlotXY(node,&xy,&(intersect->xingpt));
    /*
     * if node is very close to crossing point, do nothing
     */
    diffx = xy.x - intersect->xingpt.x;
    diffy = xy.y - intersect->xingpt.y;
    if (fabs(diffx)<eps && fabs(diffy)<eps) {
        sprintf( logbuf, "Trouble %d seg %d %d\n",
                     node, intersect->node1b, intersect->node2b );
        Log( 1, (const char *)logbuf );
    }
    else {
        /*
         * find the regions on either side of the crossed segment
         * intersect->chk_rgn is intersect->node1b to intersect->node2b
         */
        ElleNeighbourRegion(intersect->node2b,intersect->node1b,&rgn);
        midpt.x = xy.x - (xy.x - intersect->xingpt.x)*0.9;
        midpt.y = xy.y - (xy.y - intersect->xingpt.y)*0.9;
        if (EllePtInRegion(intersect->chk_rgn,&midpt))
            rgn = intersect->chk_rgn;
        /*
         * find the neighbour for this region
         */
        if ((nb = ElleFindBndNb(node, rgn))==NO_NB)
            OnNodeError(node,"ElleReplaceTripleJ",0);
        ElleInsertDoubleJ(intersect->node1b,intersect->node2b,&dj[0],0.5);
        /*
         * insert and join DJs
         */
        ElleSetPosition(dj[0],&(intersect->xingpt));
        ElleSetPrevPosition(dj[0],&(intersect->xingpt));
        ElleInsertDoubleJ(node,nb,&dj[1],0.1);
        ElleNewGrain(dj[1],dj[0],rgn,&newrgn);
        ElleSwitchTripleNodesForced(dj[1],dj[0]);
    }
}
#endif
