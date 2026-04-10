 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <stdlib.h>
#include <utility>
#include <list>
/* If SINGLE is defined when triangle.o is compiled, it should also be
   defined here
   SINGLE should not be defined for Elle code
    */

/*#define SINGLE*/

#ifdef SINGLE
#define REAL float
#else /* not SINGLE */
#define REAL double
#endif /* not SINGLE */
#include "triangle.h"
#include "nodes.h"
#include "triattrib.h"
#include "splitting.h"
#include "unodes.h"
#include "file.h"
#include "general.h"
#include "error.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/

Mesh MeshData;

void ElleInitTriAttributes()
{
    if ((MeshData.tri = (struct triangulateio *)
          malloc(sizeof(struct triangulateio)))==0)
             OnError("ElleInitTriAttributes",MALLOC_ERR);
    initio(MeshData.tri);
    MeshData.attribute = 0;
    MeshData.num_attributes = 0;
    MeshData.ref_id = NO_NB;
}

void ElleRemoveTriAttributes()
{
    ElleClearTriAttributes();
    if (MeshData.tri) free (MeshData.tri);
}

void ElleClearTriAttributes()
{
    if (MeshData.tri != 0) cleanio(MeshData.tri);
    if (MeshData.attribute != 0) {
        if (MeshData.attribute->elem != 0) free(MeshData.attribute->elem);
        free(MeshData.attribute);
    }
    MeshData.attribute = 0;
    MeshData.num_attributes = 0;
    MeshData.ref_id = NO_NB;
}

void ElleSetTriReferenceId(int ref)
{
    MeshData.ref_id = ref;
}

void ElleAllocTriAttributes(int index)
{
    int i;

    if (MeshData.num_attributes > index)
        OnError("ElleAllocTriAttributes",INDEX_ERR);
    MeshData.num_attributes++;
    if (MeshData.attribute==0) {
        if ((MeshData.attribute = (attr_array *) malloc(
                  sizeof(attr_array) * MeshData.num_attributes))==0)
            OnError("ElleAllocTriAttributes",MALLOC_ERR);
    }
    else {
        if ((MeshData.attribute = (attr_array *) realloc(
                  MeshData.attribute,
                  sizeof(attr_array) * MeshData.num_attributes))==0)
            OnError("ElleAllocTriAttributes",MALLOC_ERR);
    }

    MeshData.attribute[MeshData.num_attributes-1].init_min = 0.0;
    MeshData.attribute[MeshData.num_attributes-1].init_max = 0.0;
    if ((MeshData.attribute[MeshData.num_attributes-1].elem =
                               (double *) malloc( sizeof(double) *
                                     MeshData.tri->numberoftriangles))==0)
        OnError("ElleAllocTriAttributes",MALLOC_ERR);
    for (i=0;i<MeshData.tri->numberoftriangles;i++)
        MeshData.attribute[0].elem[i] = 0.0;
}

/*
 * this just sets the index triangle attribute to the average
 * of the attributes for each (non-boundary) vertex node
 * needs work
 */
void ElleSetTriAttributeFromNodes(int index, int attr_id)
{
    int i,j,numpts,numptattrib,*p,*node;
    int id, num, flynn_id=NO_NB;
    double total, val, val2;

    ElleAllocTriAttributes(index);
    if (ElleNodeMarginAttributeActive(attr_id))
        flynn_id=MeshData.ref_id;
    /*numptattrib = MeshData.tri->numberofpointattributes;*/
    /*if (nindex > numptattrib) OnError("",MAXATTRIB_ERR);*/
    numpts = MeshData.tri->numberofcorners;
    for (i=0,p = MeshData.tri->trianglelist;
                i<MeshData.tri->numberoftriangles;i++,p+=numpts) {
        total=0;
        num=0;
        for (j=0,node=p;j<3;j++,node++) {
            /*total+=MeshData.tri->pointattributelist[(*node)*/
                                                    /**numptattrib+nindex];*/
            id = (int)MeshData.tri->pointattributelist[(*node)];
            if (MeshData.tri->pointmarkerlist[(*node)]==0) {
			    switch(attr_id) {
				case CAXIS: ElleGetUnodeAttribute(id,&val,&val2,attr_id );
							break;
				default: ElleGetUnodeAttribute(id, &val,attr_id);
							break;
				}
                total += val;
                num++;
            }
            else if (flynn_id!=NO_NB) {
                val = ElleNodeMarginAttribute(id,attr_id,flynn_id);
                total += val;
                num++;
            }
        }
        if (num>0) MeshData.attribute[index].elem[i] = total/num;
    }
}

int ElleSetTriAttributeRange(int index)
{
    int i, max;
    double *p, amax, amin;

    if (MeshData.tri==0 || MeshData.attribute==0)
        OnError("ElleSetTriAttributeMax",DATA_ERR );
    p = MeshData.attribute[index].elem;
    max = MeshData.tri->numberoftriangles;
    amax = amin = p[0];
    for (i=0;i<max;i++) {
        if (p[i] > amax) amax = p[i];
        if (p[i] < amin) amin = p[i];
    }
    MeshData.attribute[index].init_min = amin;
    MeshData.attribute[index].init_max = amax;
    return(0);
}

double ElleTriAttributeMax(int index)
{
    if (MeshData.tri==0 || MeshData.attribute==0)
        OnError("ElleTriAttributeMax",DATA_ERR );
    return(MeshData.attribute[index].init_max);
}

double ElleTriAttributeMin(int index)
{
    if (MeshData.tri==0 || MeshData.attribute==0)
        OnError("ElleTriAttributeMin",DATA_ERR );
    return(MeshData.attribute[index].init_min);
}

void ElleSetTriAttributeMax(int index,double val)
{
    if (MeshData.tri==0 || MeshData.attribute==0)
        OnError("ElleSetTriAttributeMax",DATA_ERR );
    MeshData.attribute[index].init_max=val;
}

void ElleSetTriAttributeMin(int index,double val)
{
    if (MeshData.tri==0 || MeshData.attribute==0)
        OnError("ElleSetTriAttributeMin",DATA_ERR );
    MeshData.attribute[index].init_min=val;
}

int ElleMaxTriangles()
{
    return(MeshData.tri->numberoftriangles);
}

double ElleTriAttribute(int id, int index)
{
    if (MeshData.tri==0 || MeshData.attribute==0)
        OnError("ElleTriAttribute",DATA_ERR );
    return(MeshData.attribute[index].elem[id]);
}

void ElleTriPointPosition(int id, double *x, double *y)
{
    REAL *p;

    p = MeshData.tri->pointlist;
    *x = p[id*2];
    *y = p[id*2+1];
}

void ElleTrianglePoints(int index, int *pt_ids)
{
    int numpts, *p;

    numpts = MeshData.tri->numberofcorners;
    p = MeshData.tri->trianglelist;
    p += index*numpts;
    pt_ids[0] =  *p; p++;
    pt_ids[1] =  *p; p++;
    pt_ids[2] =  *p;
}

/*
 * index is an elle id (unode)
 * this gets the elle ids for the triangulation pts neighbours
 * assumes the first point attribute is its elle id
 * neighbours are ordered anticlockwise
 * needs two functions as a node and unode could have the same id
 */
void ElleGetTriPtNeighbours(int id, std::vector<int> &ids,
				std::vector<int> &on_bnd, unsigned char bnd )
{
    int i,j,k,numpts;
    int index;
    Coords xy,nbxy;

    if (MeshData.tri==0)
        OnError("ElleGetTriPtNeighbours - no triangulation data",0);
    numpts = MeshData.tri->numberofpoints;
    for (i=0, index=NO_NB;i<numpts && index==NO_NB;i++)
        if ((int)MeshData.tri->pointattributelist[i]==id &&
                 MeshData.tri->pointmarkerlist[i]==0)
        /*if ((int)MeshData.tri->pointattributelist[i]==id)*/
            index = i;

    ids.erase(ids.begin(),ids.end());
    on_bnd.erase(on_bnd.begin(),on_bnd.end());
    if (index!=NO_NB) {
		std::list< std::pair<int,double> > ordered_list;
		std::list< std::pair<int,double> >::iterator it;
        ElleGetUnodePosition(id,&xy);
		std::vector<int> nbnodes;
		std::vector<int>::iterator v_it;
        findNeighbourPts(index,NO_NB,MeshData.tri,&k,nbnodes,bnd);
        for (i=0,v_it=nbnodes.begin();v_it!=nbnodes.end();v_it++,i++) {
            j=(int)MeshData.tri->pointattributelist[(*v_it)];
            if (MeshData.tri->pointmarkerlist[(*v_it)])
                ElleNodePlotXY(j,&nbxy,&xy);
            else {
                ElleGetUnodePosition(j,&nbxy);
                ElleCoordsPlotXY(&nbxy,&xy);
            }
			std::pair<int,double> p((*v_it),polar_angle(nbxy.x-xy.x,
                                              nbxy.y-xy.y));
            it=ordered_list.begin();
            while (it!=ordered_list.end() && p.second > (*it).second)
                it++;
            ordered_list.insert(it,p);
        }
        for (it=ordered_list.begin(); it!=ordered_list.end(); it++) {
            j=(int)MeshData.tri->pointattributelist[(*it).first];
            ids.push_back(j);
            if (MeshData.tri->pointmarkerlist[(*it).first]==0) k=0;
            else k=1;
            on_bnd.push_back(k);
        }
    }
}

/*
 * index is an elle id (boundary node)
 * this gets the elle ids for the triangulation pts neighbours
 * assumes the first point attribute is its elle id
 * neighbours are ordered anticlockwise starting
 * with a nb_id (if found) then list pts inside the polygon
 * needs two functions as a node and unode could have the same id
 */
void ElleGetNodeTriPtNeighbours(int id, std::vector<int> &ids,
				std::vector<int> &on_bnd, int nb_id,
                                unsigned char bnd )
{
    int i,j,k,numpts;
    int index;
    Coords xy, nbxy;

    if (MeshData.tri==0)
        OnError("ElleGetTriPtNeighbours - no triangulation data",0);
    numpts = MeshData.tri->numberofpoints;
    for (i=0, index=NO_NB;i<numpts && index==NO_NB;i++)
        if ((int)MeshData.tri->pointattributelist[i]==id &&
                 MeshData.tri->pointmarkerlist[i]!=0)
            index = i;

    ids.erase(ids.begin(),ids.end());
    on_bnd.erase(on_bnd.begin(),on_bnd.end());
    if (index!=NO_NB) {
		std::list< std::pair<int,double> > ordered_list;
		std::list< std::pair<int,double> >::iterator it, start;
        ElleNodePosition(id,&xy);
		std::vector<int> nbnodes;
		std::vector<int>::iterator v_it;
        findNeighbourPts(index,NO_NB,MeshData.tri,&k,nbnodes,bnd);
        for (i=0,v_it=nbnodes.begin();v_it!=nbnodes.end();v_it++,i++) {
            j=(int)MeshData.tri->pointattributelist[(*v_it)];
            if (MeshData.tri->pointmarkerlist[(*v_it)]) {
                ElleNodePlotXY(j,&nbxy,&xy);
            }
            else {
                ElleGetUnodePosition(j,&nbxy);
                ElleCoordsPlotXY(&nbxy,&xy);
            }
			std::pair<int,double> p((*v_it),polar_angle(nbxy.x-xy.x,
                                              nbxy.y-xy.y));
            it=ordered_list.begin();
            while (it!=ordered_list.end() && p.second > (*it).second)
                it++;
            ordered_list.insert(it,p);
        }
        it=ordered_list.begin();
        while (it!=ordered_list.end() &&
          MeshData.tri->pointattributelist[(*it).first]!=nb_id) it++;
        start = it;
        for (it=start; it!=ordered_list.end(); it++) {
            j=(int)MeshData.tri->pointattributelist[(*it).first];
            ids.push_back(j);
            if (MeshData.tri->pointmarkerlist[(*it).first]==0) k=0;
            else k=1;
            on_bnd.push_back(k);
        }
        for (it=ordered_list.begin(); it!=start; it++) {
            j=(int)MeshData.tri->pointattributelist[(*it).first];
            ids.push_back(j);
            if (MeshData.tri->pointmarkerlist[(*it).first]==0) k=0;
            else k=1;
            on_bnd.push_back(k);
        }
    }
}

int ElleTriPtOnBoundary(int index)
{
    int i,on_bnd=0,found=0;

    if (MeshData.tri==0)
        OnError("ElleTriPtOnBoundary - no triangulation data",0);
    int numpts = MeshData.tri->numberofpoints;
    for (i=0, found=0;i<numpts && !found;i++)
        if ((int)MeshData.tri->pointattributelist[i]==index) {
            on_bnd = MeshData.tri->pointmarkerlist[i];
            found = 1;
        }
    return(on_bnd);
}
#if XY
void ElleGetTriPtNeighbours(int id, int **nbs, int *num,
                            unsigned char bnd )
{
    int i,j,numpts,numptattrib,*p,*node;
    int id;
    double total, val;

    if (MeshData.tri==0)
        OnError("ElleGetTriPtNeighbours - no triangulation data",0);
    numpts = MeshData.tri->numberofpoints;
    for (i=0, index=NO_NB;i<numpts && index==NO_NB;i++)
        if ((int)MeshData.tri->pointattributelist[i]==id &&
                 MeshData.tri->pointmarkerlist[i]==0)
            index = i;
    if (index!=NO_NB) {
		std::vector<int> nbnodes;
		std::vector<int>::iterator it;
        findNeighbourPts(index,NO_NB,MeshData.tri,&j,nbnodes,bnd);
        *num = nbnodes.size();
        if (*num>0) *nbs = new int[*num];
        for (i=0,it=nbnodes.begin();it!=nbnodes.end();it++,i++)
            (*nbs)[i] = (int)MeshData.tri->pointattributelist[(*it)];
    }
    else {
        *num = 0;
        *nbs = 0;
    }
}
#endif
