 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nodes.h"
#include "convert.h"
#include "general.h"
#include "polygon.h"
#include "runopts.h"
#include "flynnarray.h"
#include "interface.h"
#include "splitting.h"
#include "error.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
#define PI_3 (PI/3)

using std::ofstream;
using std::cout;
using std::endl;
using std::vector;
using std::list;

void GetAngle(int node1,int node2,Coords ddnode, double *testangle);

#if XY
void CreateBoundary(vector<PointData> &nodes,int nodenbbnd,
                    int nbnodebnd,
                    int *elle_id,int num,int *end)
{
    int next,curr,endindex,i;
    int nb1, nb2, newid;
    Coords xy;

    endindex = nodes[nodes.size()-1].idvalue();
    if (elle_id[endindex]==NO_NB) {
        i=(endindex+1)%num;
        while (elle_id[i]==NO_NB) i=(i+1)%num;
        nb1 = elle_id[i];
        i=(endindex-1+num)%num;
        while (elle_id[i]==NO_NB) i=(i-1+num)%num;
        nb2 = elle_id[i];
        ElleInsertDoubleJ(nb1,nb2,&newid,0.5);
        elle_id[endindex] = newid;
        xy.x = nodes[nodes.size()-1].xvalue();
        xy.y = nodes[nodes.size()-1].yvalue();
        /*xy.x = xvals[endindex];*/
        /*xy.y = yvals[endindex];*/
        ElleSetPosition(newid,&xy);
    }
    curr = next = elle_id[nodes[0].idvalue()];
    for (i=0; i<nodes.size()-2; i++) {
        next = ElleFindSpareNode();
        xy.x = nodes[i+1].xvalue();
        xy.y = nodes[i+1].yvalue();
        ElleSetPosition(next,&xy);
        ElleSetNeighbour(curr,NO_NB,next,&nodenbbnd);
        ElleSetNeighbour(next,NO_NB,curr,&nbnodebnd);
        curr = next;
    }
    *end = elle_id[endindex];
    ElleSetNeighbour(next,NO_NB,*end,&nodenbbnd);
    ElleSetNeighbour(*end,NO_NB,next,&nbnodebnd);
}
#endif

void CreateBoundary(vector<PointData> &nodes,int nodenbbnd,
                    int nbnodebnd,
                    int *elle_id,int num,int *end)
{
    int next,curr,endindex,i;
    int nb1, nb2, newid, success=0;
    double min,sqx,sqy;
    Coords xy, try_xy, prev_xy;

    endindex = nodes[nodes.size()-1].idvalue();
    if (elle_id[endindex]==NO_NB) {
        i=(endindex+1)%num;
        while (elle_id[i]==NO_NB) i=(i+1)%num;
        nb1 = elle_id[i];
        i=(endindex-1+num)%num;
        while (elle_id[i]==NO_NB) i=(i-1+num)%num;
        nb2 = elle_id[i];
        ElleInsertDoubleJ(nb1,nb2,&newid,0.5);
        elle_id[endindex] = newid;
        xy.x = nodes[nodes.size()-1].xvalue();
        xy.y = nodes[nodes.size()-1].yvalue();
        /*xy.x = xvals[endindex];*/
        /*xy.y = yvals[endindex];*/
        ElleSetPosition(newid,&xy);
    }
    curr = next = elle_id[nodes[0].idvalue()];
    xy.x = nodes[0].xvalue();
    xy.y = nodes[0].yvalue();
    min = ElleminNodeSep() * 1.5;
    prev_xy=xy;
    for (i=0; i<nodes.size()-2; i++) {
        try_xy.x = nodes[i+1].xvalue();
        try_xy.y = nodes[i+1].yvalue();
        sqx =(prev_xy.x-try_xy.x)*(prev_xy.x-try_xy.x);
        sqy =(prev_xy.y-try_xy.y)*(prev_xy.y-try_xy.y);
        if((min*min)<(sqx+sqy)){
            xy=try_xy;
            next = ElleFindSpareNode();
            ElleSetPosition(next,&xy);
            ElleSetNeighbour(curr,NO_NB,next,&nodenbbnd);
            ElleSetNeighbour(next,NO_NB,curr,&nbnodebnd);
            prev_xy=xy;
            curr = next;
        } 
    }
    *end = elle_id[endindex];
    ElleSetNeighbour(next,NO_NB,*end,&nodenbbnd);
    ElleSetNeighbour(*end,NO_NB,next,&nbnodebnd);
    ElleSetNodeChange(1); // set here in case only 2djs joined
}

void RemoveBoundary(int *start, int *finish, int rgn1, int rgn2)
{
    int next, indx, tmp;

    // remove the DJs on the boundary
    if ((next=ElleFindBndNode(*start,*start,rgn1))==NO_NB)
            OnError("RemoveBoundary",NONB_ERR);
    // start should be the start of the boundary to be removed
    // if walking around rgn1
    if (!(ElleNodeOnRgnBnd(next,rgn1) &&
                    ElleNodeOnRgnBnd(next,rgn2))) {
        tmp = *start;
        *start = *finish;
        *finish = tmp;
        if ((next=ElleFindBndNode(*start,*start,rgn1))==NO_NB)
            OnError("RemoveBoundary",NONB_ERR);
    }
    //  if rgn1 was 2tjs (start,finish) +bnd of djs,
	//  ElleDeleteDoubleJ will have removed rgn1 so keep checking
    while (next!=*finish && ElleFlynnIsActive(rgn1)) {
        ElleDeleteDoubleJNoCheck(next);
        if ((next=ElleFindBndNode(*start,*start,rgn1))==NO_NB &&
						ElleFlynnIsActive(rgn1) )
            OnError("RemoveBoundary",NONB_ERR);
    }
    //  if rgn2 was triangular, ElleDeleteDoubleJ will have removed
    //   it when it found a 2-sided grain
    if (ElleNodeIsTriple(*start)) {
/*printf("b4 unlink mass = %.8e\n",ElleTotalNodeMass(147));*/
        // unlink the TJs
        if (ElleNodeCONCactive())
            ElleRecalculateNodeCONCrb(*start,*finish);

        if ((indx = ElleFindNbIndex(*finish,*start))==NO_NB)
            OnError("RemoveBoundary unlinking ",NONB_ERR);
        ElleClearNeighbour(*start,indx);
        if ((indx = ElleFindNbIndex(*start,*finish))==NO_NB)
            OnError("RemoveBoundary unlinking ",NONB_ERR);
        ElleClearNeighbour(*finish,indx);

        ResetRegionEntries(*start,start,*finish,rgn2,rgn1);
    }
/*printf("after rb mass = %.8e\n",ElleTotalNodeMass(147));*/
}

int RandomSplit(struct triangulateio *out, vector<PointData> &boundary,
                int *ids, int *elle_id, int num, int start)
{
    int startindx, newid, end, j, previndx, next;
    int cnt, n, numtries, failed=0;
    Coords xy, current, prev;;
    vector<int> nbnodes;

    boundary.erase(boundary.begin(),boundary.end());
    next = boundary.size();
    startindx=0;
    while (startindx<num && elle_id[startindx]!=start) startindx++;
    if (startindx<num) previndx = j = startindx;
    else OnError("RandomSplit - start node not found",0);
    /*
     * Put the starting node in the boundary list
     */
    ElleNodePosition(start,&xy);
    PointData bndnode(out->pointlist[j*2],out->pointlist[j*2+1],j,1);
    boundary.push_back(bndnode);
    end = NO_NB;
    while (!failed && end==NO_NB) {
        /*
         * for each step - find random neighbour
         *   that hasn't already been included
         */
        findNeighbourPts(j,previndx,out,&cnt,nbnodes,1);
        if (cnt==0) return(1);
        n = ElleRandom()%cnt;
        next = nbnodes[n];
        numtries = 1;
        if (cnt<3) failed=1; /* only nb bnd pts, won't split flynn */
#if XY
        while (!failed
               && ((out->pointmarkerlist[next] &&
                    (next<num && elle_id[next]!=NO_NB &&
                       ElleNodeIsTriple(elle_id[next] ||
                        (boundary.size()==1 &&
                       ElleFindNbIndex(elle_id[next],start)!=NO_NB))))
                   || (out->pointmarkerlist[next] &&
                        (boundary.size()==1 &&
                        index_diff(startindx,next,num)==1))
                   || in_array(next,boundary)
                   || (out->pointmarkerlist[next] && next>=num))
              ) {
         }
#endif
         /*
          * This logic doesn't allow a single step split.
          * Short paths can be valid if the grain is narrow,
          * so maybe we should fix the check above so that it
          * only precludes single step boundaries that join
          * neighbouring elle boundary nodes.
          */
         /*
          * try to avoid
          * a boundary node that is an elle TJ
          * a neighbouring elle boundary node on the first step
          * a node that has already been visited - loop
          */
        while (!failed
               /*&& ((out->pointmarkerlist[next] &&
 * boundary.size()==1)*/
               && ((out->pointmarkerlist[next] &&
                     elle_id[next]!=NO_NB &&
                       ElleFindNbIndex(elle_id[next],start)!=NO_NB)
                   || (next<num && elle_id[next]!=NO_NB &&
                       ElleNodeIsTriple(elle_id[next]))
                   || in_array(next,boundary)
                   )
              ) {
            if (numtries<cnt) {
                n = (n+1)%cnt;
                numtries++;
                next = nbnodes[n];
            }
            else failed=1;
        }
        if (!failed) {
            previndx = j;
            j = nbnodes[n];
            if (out->pointmarkerlist[j]) end = j;

            bndnode.setvalues(out->pointlist[j*2],out->pointlist[j*2+1],j,1);
            boundary.push_back(bndnode);
        }
    }
//if (!failed) {
//copy(boundary.begin(),boundary.end(),ostream_iterator<PointData>(cout));
//cout << endl;
//}
    return(failed);
}

int DirectSplit(struct triangulateio *out, vector<PointData> &boundary,
                double f_area, int *elle_id, int num, int start,
                float dx, float dy)
{
    int startindx, newid, end, j, previndx, next;
    int cnt, n, prev, numtries, failed=0, sign=1;
    double l, ddx, ddy, difx, dify;
    double tmpangle, angle1, angle2;
    int i, nghbr[2], nb[3], sign1, sign2;
    int size;
    unsigned char found1, found2;
    double *xpts=0, *ypts=0, minarea=0, area=0, area2=0;
    Coords xy, current, ddnode;
	vector<int> nbnodes(0);
   
    Coords prevnode;

    boundary.erase(boundary.begin(),boundary.end());
    next = boundary.size();
    startindx=0;
    while (startindx<num && elle_id[startindx]!=start) startindx++;
    if (startindx<num) previndx = j = startindx;
    else OnError("DirectSplit - start node not found",0);
    /*
     * Put the starting node in the boundary list
     */
    ElleNodePosition(start,&xy);
    PointData bndnode(out->pointlist[j*2],out->pointlist[j*2+1],j,1);
    boundary.push_back(bndnode);
    end = NO_NB;
    if (dx==0 && dy ==0) {
    /* creates starting vector for split if no specific dx, dy is passed*/
       
       dx = (float)(ElleRandom()%10000+1);
       dy =(float)(ElleRandom()%10000+1);    
   }
   
   /* convert into unit vector for original vector*/
   l = sqrt((double)((dx*dx)+(dy*dy)));
   ddx=dx/l;
   ddy=dy/l;

   ddnode.x = ddx;
   ddnode.y = ddy;

    /*
     * find the node numbers of the neighbours
     */
   ElleNeighbourNodes(start,nb);
   i=0;
   while (i<3 && nb[i]==NO_NB) i++;
   nghbr[0] = nb[i]; i++;
   while (i<3 && nb[i]==NO_NB) i++;
   nghbr[1] = nb[i];
   /*if (ElleNodeOnRgnBnd(nghbr[0],num) && ElleNodeOnRgnBnd(nghbr[0],num)){*/
      GetAngle(start,nghbr[0],ddnode,&angle1);
      if (fabs(angle1)>PI_2) sign1=0;
      else sign1=1;
      GetAngle(start,nghbr[1],ddnode,&angle2);
      if (fabs(angle2)>PI_2) sign2=0;
      else sign2=1;
/*
   }
   else{
       failed =1;
   }
*/
   if ((fabs(angle1)>PI_3 && sign1==1) ||
       (fabs(angle1)<PI_3 && sign1==0)) found1=1;
   if ((fabs(angle2)>PI_3 && sign2==1) ||
       (fabs(angle2)<PI_3 && sign2==0)) found2=1;

//NB commented out so splitting wrapping flynn won't fail le 5/9/05
/*
   if ((found1==1) && (found2==1)){
      failed = 0;
   }
   else{
      failed = 1;
   }
*/

   while (!failed && end==NO_NB) {
        /*
         * for each step - find random neighbour
         *   that hasn't already been included
         */
          findNeighbourPts(j,previndx,out,&cnt,nbnodes,1,&ddnode,sign);
          n = 0;
          numtries = 0;
     	  if (cnt ==0) failed = 1;
          else next = nbnodes[n];   
          while (!failed
                   /*&& ((out->pointmarkerlist[next] &&
 * boundary.size()==1)*/
               && ((out->pointmarkerlist[next] &&
                     elle_id[next]!=NO_NB &&
                       ElleFindNbIndex(elle_id[next],start)!=NO_NB)
                       || (next<num && elle_id[next]!=NO_NB &&
                           ElleNodeIsTriple(elle_id[next]))
                       || in_array(next,boundary)
                       )
                  ) {
                if (numtries<cnt) {
                    n = (n+1)%cnt;
                    numtries++;
                    next = nbnodes[n];   
                }
                else failed=1;
          }
          if (!failed) {
                previndx = j;
                j = nbnodes[n];
                if (out->pointmarkerlist[j]) end = j;

                bndnode.setvalues(out->pointlist[j*2],out->pointlist[j*2+1],j,1);
                if (boundary.size() == 1) {
                   /*difx=out->pointlist[j*2]-out->pointlist[previndx*2];
                   dify=out->pointlist[j*2+1]-out->pointlist[previndx*2+1];
                   if (dify>0) sign = 1;
                   else sign=0;*/
                   /*
                    * find angle between start node->unit vector
                    * and start node->first step segment
                    */
                   angle (out->pointlist[previndx*2],
                          out->pointlist[previndx*2+1], 
                         out->pointlist[j*2],
                         out->pointlist[j*2+1],
                         out->pointlist[previndx*2]+ddx,
                         out->pointlist[previndx*2+1]+ddy,&tmpangle);
                   if (fabs(tmpangle)>PI_2) sign=0;
                   else sign=1;

               }
                boundary.push_back(bndnode);
          }
    }
    
    minarea=(double)ElleMinFlynnArea();
    if (!failed && minarea>0) {
      // check that the new flynns will be larger than the min area
      size = num + boundary.size();
      if ((xpts = (double *)malloc(size*sizeof(double)))==0)
        OnError("DirectSplit",MALLOC_ERR);
      if ((ypts = (double *)malloc(size*sizeof(double)))==0)
        OnError("DirectSplit",MALLOC_ERR);
      for (i=0; i<boundary.size(); i++) {
        xpts[i] = (double)boundary[i].xvalue();
        ypts[i] = (double)boundary[i].yvalue();
      }
      current.x = boundary[boundary.size()-1].xvalue();
      current.y = boundary[boundary.size()-1].yvalue();
      j = end+1;
      while (j!=startindx) {
        if (elle_id[j]!=NO_NB) {
            ElleNodePlotXY(elle_id[j],&xy,&current);
            xpts[i] = xy.x;
            ypts[i] = xy.y;
            current = xy;
            i++;
        }
        j=(j+1)%num;
      }
      i--;
      area = polyArea(xpts,ypts,i);
      if (xpts) free(xpts);
      if (ypts) free(ypts);
      area2 = f_area - area;
      if (area<minarea || area2<minarea) failed=1;
   }
//if (!failed) {
//copy(boundary.begin(),boundary.end(),ostream_iterator<PointData>(cout));
//cout << endl;
//}
    return(failed);
}

void GetAngle(int node1,int node2,Coords ddnode, double *testangle)
{
    Coords xy1, xy2;
    double tmpangle;

    ElleNodePosition(node2,&xy1);
    ElleNodePosition(node1,&xy2);
    angle (xy1.x,xy1.y,xy2.x,xy2.y,ddnode.x,ddnode.y,&tmpangle);
    *testangle=tmpangle;
}

int index_diff(int val1, int val2, int size)
{
    int diff = abs(val1-val2);
    return((diff==(size-1)) ? 1 : diff);
}

/*
 * find all the neighbour nodes
 * don't include the previous node
 * if !incl_bnd then don't include boundary nodes
 */
void findNeighbourPts(int node, int prev, struct triangulateio *out,
                      int *cnt, vector<int> &nbnodes,
                      unsigned char incl_bnd)
{
    unsigned char found;
    int i, j, nb;

    *cnt = 0;
    for (i=0; i<out->numberofedges; i++) {
        nb = NO_NB;
        if (out->edgelist[i*2]==node && out->edgelist[i*2+1]!=prev &&
            (incl_bnd || out->pointmarkerlist[out->edgelist[i*2+1]]==0))
            nb = out->edgelist[i*2+1];
        else if (out->edgelist[i*2+1]==node && out->edgelist[i*2]!=prev &&
            (incl_bnd || out->pointmarkerlist[out->edgelist[i*2]]==0))
           nb = out->edgelist[i*2];
        if (nb != NO_NB) {
            for (j=0,found=0; j<*cnt && !found; j++)
                if (nbnodes[j]==nb) found=1;
            if (!found) {
                if (*cnt<nbnodes.size())
                    nbnodes[*cnt] = nb;
                else nbnodes.push_back(nb);
                (*cnt)++;
            }
        }
    }
}
/*
 * find all the neighbour nodes
 * don't include the previous node
 * if !incl_bnd then don't include boundary nodes
 * and sort on dx, dy
 */
void findNeighbourPts(int node, int prev, struct triangulateio *out,
                      int *cnt, vector<int> &nbnodes,
                      unsigned char incl_bnd, Coords *ddnode, int sign)
{   vector<int> :: iterator itv;
    unsigned char found;
    int i, j, nb, index;
    float tmpangle;
	list<double> angles(0);
	list<int> indices(0);
	list<double>::iterator itang;
	list<int>::iterator itnb;
    
    double dx2,dy2,dx1,dy1,checkangle;

    *cnt = 0;
    for (i=0; i<out->numberofedges; i++) {
        nb = NO_NB;
        if (out->edgelist[i*2]==node && out->edgelist[i*2+1]!=prev &&
            (incl_bnd || out->pointmarkerlist[out->edgelist[i*2+1]]==0))
            {nb = out->edgelist[i*2+1];}
        else if (out->edgelist[i*2+1]==node && out->edgelist[i*2]!=prev &&
            (incl_bnd || out->pointmarkerlist[out->edgelist[i*2]]==0))
            {nb = out->edgelist[i*2];}
        if (nb != NO_NB) {
            found=0;
             /*
              * check whether this node is already in the list
              */
            for (itnb=indices.begin();itnb!=indices.end() && !found;itnb++)
                if (*itnb==nb ) found=1;
            
/*
            dx2=out->pointlist[node*2]-out->pointlist[nb*2];     
            dy2=out->pointlist[node*2+1]-out->pointlist[nb*2+1]; 
            checkangle =((dx*dx2+dy*dy2)/(sqrt(dx2*dx2+dy2*dy2)));   
            
            if ( sign==0 && checkangle>0) found=1;
            if ( sign==1 && checkangle<0) found=1;
*/
            /*
             * find the angle between the vector and the node->nb segment
             * if sign is 0 accept nbs with angles > 90
             * if sign is 1 accept nbs with angles < 90
             */
                   angle (out->pointlist[node*2],
                          out->pointlist[node*2+1], 
                         out->pointlist[nb*2],
                         out->pointlist[nb*2+1],
                         out->pointlist[node*2]+ddnode->x,
                         out->pointlist[node*2+1]+ddnode->y,&tmpangle);
            if (fabs(tmpangle)>(PI_2*1.3) && sign==1) found=1;
            if (fabs(tmpangle)<(PI_2/1.3) && sign==0) found=1;
            /*
             * acceptable nbs are stored in increasing angle order
             */
            if (!found) {
              itang=angles.begin();itnb=indices.begin(); index=0;
              tmpangle=fabs(tmpangle);
              if (tmpangle>PI_2) tmpangle = PI - tmpangle;
              while (itang!=angles.end() && (*itang < tmpangle) 	
		  /*&& checkangle>90*/)                       
		     {itang++;itnb++;index++; }                  
	      if (itang==angles.end())
   		  {angles.push_back(tmpangle); indices.push_back(nb);} 
              else
		  {angles.insert(itang,tmpangle);indices.insert(itnb,nb);}  
              (*cnt)++; 
/* this line to print the angle values
for (itang=angles.begin();itang!=angles.end();itang++) cout<< *itang << " "; 
*/
            }
        }

    }
    for (itnb=indices.begin(),index=0;itnb!=indices.end();
                                               itnb++,index++)
        if (index<nbnodes.size()) nbnodes[index] = *itnb;
        else  nbnodes.push_back(*itnb);
/* this line to print the node list
   cout<<endl;
for (itv=nbnodes.begin();itv!=nbnodes.end();itv++) cout << *itv << " ";
cout<<endl;
*/
}

int WriteTriangulation(struct triangulateio *out,const char *name)
{
    int i, j;
	ofstream outfile(name);

    if (!outfile) OnError("WriteTriangulation",OPEN_ERR);
    outfile << out->numberofpoints << " 2 " << out->numberofpointattributes << " 1" << endl;
    for (i=0;i<out->numberofpoints;i++) {
        outfile << i << " " << out->pointlist[i*2];
        outfile << " " << out->pointlist[i*2+1];
        for (j=0;j<out->numberofpointattributes;j++)
            outfile << " " <<
               out->pointattributelist[i*out->numberofpointattributes+j];
        outfile << " " << out->pointmarkerlist[i] << endl;
    }
    outfile << out->numberofedges << " " << " 1" << endl;
    for (i=0;i<out->numberofedges;i++) {
        outfile << i << " " << out->edgelist[i*2];
        outfile << " " << out->edgelist[i*2+1];
        outfile << " " << out->edgemarkerlist[i] << endl;
    }
    outfile << "0" << endl;
    outfile << "0" << endl;
    return(0);
}

bool in_array(int idval, vector<PointData> &v)
{
    bool found = 0;
	vector<PointData>::iterator it = v.begin();
    while (it!=v.end() && !found) {
        if ((*it).id_match(idval)) found = 1;
        else it++;
    }
    return(found);
}
